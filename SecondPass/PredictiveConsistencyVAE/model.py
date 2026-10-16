"""Fresh small VAE, causal three-frame predictor, and same-clip VICReg views.

The context encoder never receives the predicted target. Two augmented views
come from the same ordered clip, preserving its internal motion direction.
VICReg statistics must be computed across the full effective batch by caller.
"""
import torch
from torch import nn
from torch.nn import functional as F
from SecondPass.PixelContrastiveVAE.model import support_balanced_reconstruction


def _encoder(channels):
    return nn.Sequential(
        nn.Conv2d(channels, 24, 3, stride=2, padding=1), nn.GELU(),
        nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
        nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GELU(),
        nn.Conv2d(64, 8, 1), nn.GELU())


class PredictiveConsistencyVAE(nn.Module):
    def __init__(self, latent_dim=128, embedding_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = _encoder(9)
        self.mu_head = nn.Linear(8 * 13 * 13, latent_dim)
        self.logvar_head = nn.Linear(8 * 13 * 13, latent_dim)
        # This branch sees three observed frames, never the three future targets.
        self.context_encoder = _encoder(9)
        self.context_mu_head = nn.Linear(8 * 13 * 13, latent_dim)
        self.predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(),
                                       nn.Linear(latent_dim, latent_dim))
        self.decode_input = nn.Linear(latent_dim, 64 * 13 * 13)
        self.decode13 = nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1)
        self.decode25 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.decode50 = nn.ConvTranspose2d(24, 9, 4, stride=2, padding=1)
        self.projector = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(),
                                       nn.Linear(latent_dim, embedding_dim))

    def encode(self, clips):
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, 3, 100, 100):
            raise ValueError('Expected batch x3frames x3channels x100 x100')
        x = self.encoder((clips - .5).reshape(-1, 9, 100, 100)).flatten(1)
        return self.mu_head(x), self.logvar_head(x).clamp(-8., 4.)

    def project(self, mu):
        # Unnormalized outputs are essential: variance is encouraged before any
        # unit normalization, which would otherwise conceal dimensional collapse.
        return self.projector(mu)

    def context(self, observed_three):
        if observed_three.ndim != 5 or tuple(observed_three.shape[1:]) != (3, 3, 100, 100):
            raise ValueError('Causal context requires exactly three observed frames')
        x = self.context_encoder((observed_three - .5).reshape(-1, 9, 100, 100)).flatten(1)
        return self.context_mu_head(x)

    def decode(self, z):
        x = F.gelu(self.decode_input(z)).reshape(-1, 64, 13, 13)
        x = F.gelu(self.decode13(x))[:, :, :25, :25]
        x = F.gelu(self.decode25(x))
        return torch.sigmoid(self.decode50(x)).reshape(-1, 3, 3, 100, 100)

    def predict_next(self, observed_three):
        return self.decode(self.predictor(self.context(observed_three)))[:, 2]

    def rollout(self, observed_three, steps=3, first_prediction=None):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('Require a positive integer rollout length')
        predictions = []
        context = observed_three
        for step in range(steps):
            next_frame = first_prediction if step == 0 and first_prediction is not None else self.predict_next(context)
            predictions.append(next_frame)
            # No future images, teacher forcing, or gradient detachment enters.
            context = torch.cat([context[:, 1:], next_frame[:, None]], dim=1)
        return torch.stack(predictions, dim=1)

    def forward(self, clips):
        mu, logvar = self.encode(clips)
        z = mu + torch.exp(.5 * logvar) * torch.randn_like(mu) if self.training else mu
        context_mu = self.context(clips)
        prediction_mu = self.predictor(context_mu)
        prediction = self.decode(prediction_mu)[:, 2]
        return dict(reconstruction=self.decode(z), mu=mu, logvar=logvar, z=z,
                    prediction=prediction,
                    future_prediction=self.rollout(clips, steps=3, first_prediction=prediction),
                    context_mu=context_mu, prediction_mu=prediction_mu)


def augment_clips(clips, noise=True, contrast=True, max_shift=4):
    """Independent per-clip views with coherent toroidal translation and contrast.

One spatial shift and contrast factor apply to all frames in each clip.
Independent pixel noise is then applied; no frame reversal, rotations or flips.
The two calls defining the views must draw independently from the same RNG.
"""
    views = []
    for clip in clips:
        dy, dx = torch.randint(-max_shift, max_shift + 1, (2,), device=clip.device).tolist()
        shifted = torch.roll(clip, shifts=(dy, dx), dims=(-2, -1))
        factor = torch.empty((), device=clip.device, dtype=clip.dtype).uniform_(.8, 1.2) if contrast else 1.
        view = .5 + factor * (shifted - .5)
        if noise:
            view = view + .01 * torch.randn_like(view)
        views.append(view.clamp(0., 1.))
    return torch.stack(views)


def vicreg_loss(view1, view2):
    """Global-batch VICReg, variance floor1, sample covariance, explicit weights."""
    if view1.ndim != 2 or view1.shape != view2.shape or len(view1) < 2:
        raise ValueError('VICReg requires matching full-batch n xD views, n>=2')
    invariance = (view1 - view2).square().mean()
    variance = .5 * (F.relu(1. - torch.sqrt(view1.var(dim=0, unbiased=True) + 1e-4)).mean() +
                     F.relu(1. - torch.sqrt(view2.var(dim=0, unbiased=True) + 1e-4)).mean())
    def covariance_penalty(view):
        centered = view - view.mean(dim=0)
        covariance = centered.T.matmul(centered) / (len(view) - 1)
        off_diagonal = covariance - torch.diag(torch.diag(covariance))
        return off_diagonal.square().sum() / view.shape[1]
    covariance = .5 * (covariance_penalty(view1) + covariance_penalty(view2))
    return dict(view_loss=invariance + variance + .04 * covariance,
                invariance=invariance, variance=variance, covariance=covariance)


def local_losses(result, clips, beta=1e-4):
    """Per-clip reconstruction/prediction/KL for weighted microbatch backward."""
    if clips.ndim != 5 or tuple(clips.shape[1:]) != (6, 3, 100, 100):
        raise ValueError('Objective requires three context plus three future frames')
    recon = support_balanced_reconstruction(result['reconstruction'], clips[:, :3])
    prediction = support_balanced_reconstruction(result['future_prediction'], clips[:, 3:6])
    kl_value = .5 * (result['mu'].square() + result['logvar'].exp() - 1. - result['logvar']).mean()
    return dict(loss=recon + prediction + beta * kl_value, recon=recon, prediction=prediction,
                kl=kl_value, full_mse=(result['reconstruction'] - clips[:, :3]).square().mean(),
                prediction_mse=(result['future_prediction'] - clips[:, 3:6]).square().mean())


def losses(first_result, second_result, first_clip, second_clip,
           view1_projected, view2_projected, beta=1e-4):
    first = local_losses(first_result, first_clip, beta)
    second = local_losses(second_result, second_clip, beta)
    result = {key: .5 * (first[key] + second[key]) for key in first}
    views = vicreg_loss(view1_projected, view2_projected)
    result['loss'] = result['loss'] + views['view_loss']
    result.update(views)
    return result
