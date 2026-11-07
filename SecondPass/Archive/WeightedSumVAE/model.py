"""Image-only VAE of the exact 1:2:3 ordered-frame sum, then a learned decoder.

The single RGB input is S/6, with S=X[t-2]+2X[t-1]+3X[t]. This reversible
normalization keeps the decoder's sigmoid domain; raw S is never clipped.
No direction, change label, prediction target or contrastive target enters VAE
training. The subsequent classifier uses the two ordered posterior means.
"""
import torch
from torch import nn
from torch.nn import functional as F


def weighted_sum(clips, normalized=True):
    """Chronological B x 3frames x 3RGB x 100 x100 -> single weighted RGB."""
    if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, 3, 100, 100):
        raise ValueError('Expected batch x3frames x3channels x100 x100')
    weights = clips.new_tensor([1., 2., 3.]).reshape(1, 3, 1, 1, 1)
    summed = (clips * weights).sum(1)
    return summed / 6. if normalized else summed


class WeightedSumVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 8, 1), nn.GELU())
        self.mu_head = nn.Linear(8 * 13 * 13, latent_dim)
        self.logvar_head = nn.Linear(8 * 13 * 13, latent_dim)
        self.decode_input = nn.Linear(latent_dim, 64 * 13 * 13)
        self.decode13 = nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1)
        self.decode25 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.decode50 = nn.ConvTranspose2d(24, 3, 4, stride=2, padding=1)

    def encode(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 100, 100):
            raise ValueError('Expected batch x3channels x100 x100 normalized sums')
        features = self.encoder(images - .5).flatten(1)
        return self.mu_head(features), self.logvar_head(features).clamp(-8., 4.)

    def decode(self, z):
        x = F.gelu(self.decode_input(z)).reshape(-1, 64, 13, 13)
        x = F.gelu(self.decode13(x))[:, :, :25, :25]
        x = F.gelu(self.decode25(x))
        return torch.sigmoid(self.decode50(x))

    def forward(self, images):
        mu, logvar = self.encode(images)
        z = mu + torch.exp(.5 * logvar) * torch.randn_like(mu) if self.training else mu
        reconstruction = self.decode(z)
        return dict(reconstruction=reconstruction, raw_reconstruction=6. * reconstruction,
                    mu=mu, logvar=logvar, z=z)


def loss_terms(result, targets, beta=1e-4):
    """Ordinary normalized-image MSE plus mean-coordinate diagonal Gaussian KL.

    Raw-sum MSE is exactly 36 times normalized MSE. Foreground-support error is
    a diagnostic only and never weights the reconstruction objective.
    """
    error = (result['reconstruction'] - targets).square()
    reconstruction = error.mean()
    kl = .5 * (result['mu'].square() + result['logvar'].exp() - 1. - result['logvar']).mean()
    support = (targets - .5).abs() > 1. / 255.
    support_mse = (error * support).sum() / support.sum().clamp_min(1)
    return dict(loss=reconstruction + beta * kl, recon=reconstruction, kl=kl,
                full_mse=reconstruction, raw_mse=36. * reconstruction,
                support_mse=support_mse)


class NonlinearChangeDecoder(nn.Module):
    """Ordered concatenation [mu_before, mu_after], trained after the VAE."""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2 * latent_dim, 256), nn.GELU(),
                                     nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, mu_before, mu_after=None):
        pair = mu_before if mu_after is None else torch.cat([mu_before, mu_after], dim=1)
        return self.network(pair).squeeze(-1)


@torch.no_grad()
def encode_pair(model, before_clips, after_clips):
    """Deterministic frozen-stage2 inputs without decoder or VAE sampling."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Stage2 requires a frozen VAE in evaluation mode')
    return model.encode(weighted_sum(before_clips))[0], model.encode(weighted_sum(after_clips))[0]
