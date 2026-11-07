"""Local delay correlations -> displacement populations -> predictive memory.

Fixed visual comparisons are a disclosed inductive bias, not supplied motion
labels. FFT is only an efficient implementation of local image correlation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from SecondPass.OutcomeContrastiveRViT.model import SpatialGRU, CLSGRU


class LocalCorrespondence(nn.Module):
    def __init__(self):
        super().__init__()
        window = torch.hann_window(50, periodic=False)
        self.register_buffer('window', window[:, None]*window[None, :])
        delta = torch.arange(-10, 11)
        dy, dx = torch.meshgrid(delta, delta, indexing='ij')
        self.register_buffer('offset_xy', torch.stack((dx, dy), -1).flatten(0, 1).float())
        self.register_buffer('fft_indices', (dy.remainder(64)*64 + dx.remainder(64)).flatten())

    def forward(self, frames):
        # Fixed local contrast and divisive energy normalization. 4x4 overlapping
        # 50px receptive fields, spaced25px; periodic boundary matches the renderer.
        b, t, _, h, w = frames.shape
        gray = frames.mean(2).reshape(b*t, 1, h, w) - .5
        patches = F.unfold(F.pad(gray, (12, 13, 12, 13), mode='circular'), 50, stride=25)
        patches = patches.transpose(1, 2).reshape(b, t, 16, 50, 50)
        patches = (patches - patches.mean((-1, -2), keepdim=True))*self.window
        spectrum = torch.fft.rfft2(patches, s=(64, 64))
        corr = torch.fft.irfft2(spectrum[:, :-1].conj()*spectrum[:, 1:], s=(64, 64))
        energy = patches.square().sum((-1, -2))
        norm = (energy[:, :-1]*energy[:, 1:]).sqrt().clamp_min(1e-8)
        corr = corr.flatten(-2)[..., self.fft_indices]/norm[..., None]
        return corr.permute(0, 1, 3, 2).reshape(b, t-1, 441, 4, 4)


def warp(image, displacement):
    """Periodic backward sampling: positive dx moves content right. Pixels only."""
    b, _, h, w = image.shape
    flow = F.interpolate(displacement, (h, w), mode='bilinear', align_corners=False)
    y, x = torch.meshgrid(torch.arange(h, device=image.device), torch.arange(w, device=image.device), indexing='ij')
    # Circular padding makes bilinear interpolation continuous across the seam.
    xx = (x[None] - flow[:, 0]).remainder(w) + 1
    yy = (y[None] - flow[:, 1]).remainder(h) + 1
    grid = torch.stack((2*xx/(w+1)-1, 2*yy/(h+1)-1), -1)
    return F.grid_sample(F.pad(image, (1, 1, 1, 1), mode='circular'), grid,
                         mode='bilinear', align_corners=True)


def image_error(prediction, target, source):
    """Contrast-energy normalized squared error; denominator has no learned path."""
    error = (prediction-target).square().mean((1, 2, 3))
    energy = .5*((source-.5).square().mean((1, 2, 3)) + (target-.5).square().mean((1, 2, 3)))
    return error/energy.clamp_min(1e-5)


class CorrespondencePredictiveMemory(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.front = LocalCorrespondence()
        # Same small learned correction at every location and candidate displacement.
        self.refine = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 1, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(.04)))
        self.token = nn.Linear(441, width)
        self.visual_cls = nn.Parameter(torch.randn(1, 1, width)*.02)
        self.position = nn.Parameter(torch.randn(1, 17, width)*.02)
        self.norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.feedback = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))
        self.spatial = SpatialGRU(width)
        self.cls = CLSGRU(width)
        self.transform = nn.Sequential(nn.Linear(441, 128), nn.GELU(), nn.Linear(128, 64))
        self.classifier = nn.Sequential(nn.Linear(3*width+5, 128), nn.GELU(), nn.Linear(128, 2))
        self.width = width

    def forward(self, frames, return_details=False):
        if frames.ndim != 5 or frames.shape[1] != 6 or frames.shape[2] not in (1, 3) or tuple(frames.shape[-2:]) != (100, 100):
            raise ValueError('Expected six raw100x100 grayscale or RGB frames')
        gray = frames.mean(2, keepdim=True)
        correlations = self.front(gray)
        b = len(frames)
        maps = correlations.permute(0, 1, 3, 4, 2).reshape(-1, 1, 21, 21)
        maps = maps + .1*self.refine(maps)
        temperature = self.log_temperature.exp().clamp(.015, .2)
        population = (maps.flatten(1)/temperature).softmax(1).reshape(b, 5, 4, 4, 441).permute(0, 1, 4, 2, 3)
        flow = torch.einsum('btkhw,kd->btdhw', population, self.front.offset_xy)
        # True one-step prediction: interval0 estimates motion to predict frame2
        # from frame1; interval3 predicts frame5 from frame4. Neither sees target.
        pred_before = warp(gray[:, 1], flow[:, 0])
        pred_after = warp(gray[:, 4], flow[:, 3])
        e_before = image_error(pred_before, gray[:, 2], gray[:, 1])
        e_after = image_error(pred_after, gray[:, 5], gray[:, 4])
        # Expectation learned from first clip predicts the second clip's motion.
        cross = warp(gray[:, 3], flow[:, :2].mean(1))
        e_cross = image_error(cross, gray[:, 4], gray[:, 3])
        # Compare full displacement populations, not privileged direction angles.
        pooled = population.mean((-1, -2))
        before, after = pooled[:, :2].mean(1), pooled[:, 3:].mean(1)
        mismatch = (before-after).abs().sum(1)
        cosine = 1-F.cosine_similarity(before, after, dim=1)
        evidence = torch.stack((mismatch, cosine, e_before, e_after, e_cross), 1)
        field = frames.new_zeros(b, self.width, 4, 4)
        cls = frames.new_zeros(b, self.width)
        before_cls = None
        for index in range(5):
            tokens = self.token(population[:, index].flatten(2).transpose(1, 2)*10)
            current = torch.cat((self.visual_cls.expand(b, -1, -1), tokens), 1)+self.position
            memory = torch.cat((cls[:, None], field.flatten(2).transpose(1, 2)), 1)
            q, m = self.norm(current), self.memory_norm(memory)
            visual = current + (self.attention(q,q,q,need_weights=False)[0] + self.feedback(q,m,m,need_weights=False)[0])/math.sqrt(2)
            visual = visual + self.ffn(visual)
            field = self.spatial(visual[:, 1:].transpose(1, 2).reshape(b, self.width, 4, 4), field)
            cls = self.cls(visual[:, 0], cls)
            if index == 1:
                before_cls = cls
        logits = self.classifier(torch.cat((before_cls, cls, (cls-before_cls).abs(), evidence), 1))
        transformations = self.transform(torch.stack((before, after), 1)*10)
        if return_details:
            return logits, cls, dict(transformations=transformations, prediction_loss=.5*(e_before+e_after).mean(),
                prediction_errors=torch.stack((e_before, e_after, e_cross), 1), population=population,
                flow=flow, evidence=evidence, temperature=temperature)
        return logits, cls
