"""Fresh ordered three-frame convolutional VAE with an explicit spatial latent."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from SecondPass.TwoFrameRViT.model import CompressionStage, ResidualBlock


class TripletEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(9, 32, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, 32), nn.GELU(),
                                  ResidualBlock(32), ResidualBlock(32))
        self.stage50 = CompressionStage(32, 64)
        self.stage25 = CompressionStage(64, 128)
        self.stage13 = CompressionStage(128, 256)

    def forward(self, ordered_rgb: Tensor) -> Tensor:
        field = self.stage25(self.stage50(self.stem(ordered_rgb - .5)))
        return self.stage13(F.pad(field, (0, 1, 0, 1), mode='replicate'))


class TripletDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_layers = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.up25 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.PixelShuffle(2),
                                  nn.GroupNorm(8, 128), nn.GELU())
        self.layers25 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.up50 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.PixelShuffle(2),
                                  nn.GroupNorm(8, 64), nn.GELU(),
                                  ResidualBlock(64), ResidualBlock(64))
        self.up100 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.PixelShuffle(2),
                                   nn.GroupNorm(8, 32), nn.GELU(),
                                   ResidualBlock(32), ResidualBlock(32))
        self.output = nn.Conv2d(32, 9, 1)

    def forward(self, latent: Tensor) -> Tensor:
        field = self.up25(self.latent_layers(latent))[:, :, :25, :25]
        field = self.up100(self.up50(self.layers25(field)))
        return self.output(field).sigmoid().reshape(latent.shape[0], 3, 3, 100, 100)


class ThreeFrameConvVAE(nn.Module):
    d_model = 256
    latent_shape = (256, 13, 13)
    logvar_bounds = (-8., 4.)

    def __init__(self, checkpoint_encoder: bool = True, checkpoint_decoder: bool = True):
        super().__init__()
        self.checkpoint_encoder = checkpoint_encoder
        self.checkpoint_decoder = checkpoint_decoder
        self.encoder = TripletEncoder()
        self.mu_head = nn.Conv2d(256, 256, 1)
        self.logvar_head = nn.Conv2d(256, 256, 1)
        self.decoder = TripletDecoder()

    @staticmethod
    def _validate(triplets: Tensor) -> None:
        if triplets.ndim != 5 or tuple(triplets.shape[1:]) != (3, 3, 100, 100):
            raise ValueError('expected ordered [batch,3 frames,3 RGB,100,100] triplets')
        if triplets.dtype != torch.float32 or torch.is_autocast_enabled(triplets.device.type):
            raise ValueError('ThreeFrameConvVAE requires FP32 and disabled autocast')

    def posterior(self, triplets: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(triplets)
        ordered = triplets.reshape(triplets.shape[0], 9, 100, 100)
        if self.checkpoint_encoder and self.training and torch.is_grad_enabled():
            field = checkpoint(self.encoder, ordered, use_reentrant=False)
        else:
            field = self.encoder(ordered)
        mu = self.mu_head(field)
        logvar = self.logvar_head(field).clamp(*self.logvar_bounds)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + (logvar * .5).exp() * torch.randn_like(mu)

    def encode(self, triplets: Tensor, sample: bool = False) -> Tensor:
        mu, logvar = self.posterior(triplets)
        return self.reparameterize(mu, logvar) if sample else mu

    def decode(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError('expected [batch,256,13,13] latent')
        if latent.dtype != torch.float32 or torch.is_autocast_enabled(latent.device.type):
            raise ValueError('decoder requires FP32 and disabled autocast')
        if self.checkpoint_decoder and self.training and torch.is_grad_enabled():
            return checkpoint(self.decoder, latent, use_reentrant=False)
        return self.decoder(latent)

    def forward(self, triplets: Tensor, sample: bool | None = None) -> dict[str, Tensor]:
        mu, logvar = self.posterior(triplets)
        z = self.reparameterize(mu, logvar) if (self.training if sample is None else sample) else mu
        return dict(reconstruction=self.decode(z), mu=mu, logvar=logvar, z=z)


def losses(reconstruction: Tensor, target: Tensor, mu: Tensor, logvar: Tensor,
           beta: float = 1e-4) -> dict[str, Tensor]:
    """Support-balanced reconstruction plus mean diagonal-normal KL.

    Data-derived support uses only target pixel contrast. Every trial/frame has
    equal weight. If a region is empty, use the available region's mean alone.
    This weighted reconstruction objective is not the standard pixelwise ELBO.
    """
    ThreeFrameConvVAE._validate(target)
    if reconstruction.shape != target.shape or mu.shape != logvar.shape:
        raise ValueError('reconstruction/posterior shapes must match')
    if not beta >= 0:
        raise ValueError('beta must be nonnegative')
    error = (reconstruction - target).square().mean(dim=2)
    support = (target - .5).abs().amax(dim=2) > (1. / 255.)
    foreground_count = support.sum(dim=(-2, -1))
    background_count = (~support).sum(dim=(-2, -1))
    foreground = (error * support).sum(dim=(-2, -1)) / foreground_count.clamp_min(1)
    background = (error * ~support).sum(dim=(-2, -1)) / background_count.clamp_min(1)
    has_foreground, has_background = foreground_count > 0, background_count > 0
    available = has_foreground.to(error.dtype) + has_background.to(error.dtype)
    recon = ((foreground + background) / available).mean()
    foreground_mse = foreground.sum() / has_foreground.sum().clamp_min(1)
    background_mse = background.sum() / has_background.sum().clamp_min(1)
    kl = .5 * (mu.square() + logvar.exp() - 1. - logvar).mean()
    return dict(loss=recon + beta * kl, recon=recon, foreground_mse=foreground_mse,
                background_mse=background_mse, full_mse=error.mean(), kl=kl)
