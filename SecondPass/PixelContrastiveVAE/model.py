"""Small shared three-frame VAE with a detached raw-pixel distance target."""
import torch
from torch import nn
from torch.nn import functional as F


class PixelContrastiveVAE(nn.Module):
    def __init__(self, latent_dim=128, embedding_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 8, 1), nn.GELU())
        self.mu_head = nn.Linear(8 * 13 * 13, latent_dim)
        self.logvar_head = nn.Linear(8 * 13 * 13, latent_dim)
        self.decode_input = nn.Linear(latent_dim, 64 * 13 * 13)
        self.decode13 = nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1)
        self.decode25 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.decode50 = nn.ConvTranspose2d(24, 9, 4, stride=2, padding=1)
        self.projection = nn.Linear(latent_dim, embedding_dim)

    def decode(self, z):
        x = F.gelu(self.decode_input(z)).reshape(-1, 64, 13, 13)
        # 13 -> 26, remove the final row/column to recover the encoder's25-grid.
        x = F.gelu(self.decode13(x))[:, :, :25, :25]
        x = F.gelu(self.decode25(x))
        return torch.sigmoid(self.decode50(x)).reshape(-1, 3, 3, 100, 100)

    def forward(self, clips):
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, 3, 100, 100):
            raise ValueError('Expected clips with shape batch x3frames x3channels x100 x100')
        x = self.encoder((clips - .5).reshape(-1, 9, 100, 100)).flatten(1)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(-8., 4.)
        z = mu + torch.exp(.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return dict(reconstruction=self.decode(z), mu=mu, logvar=logvar, z=z,
                    embedding=F.normalize(self.projection(mu), dim=1, eps=1e-8))


def decayed_image(clips):
    """Oldest-to-newest weights1/7,2/7,4/7, centered at the known gray background."""
    weights = clips.new_tensor([1., 2., 4.]).reshape(1, 3, 1, 1, 1) / 7.
    return ((clips - .5) * weights).sum(dim=1)


def decayed_distance(first_clip, second_clip):
    """Detached cosine distance between raw decayed images, with no labels/angles."""
    first = F.normalize(decayed_image(first_clip.detach()).flatten(1), dim=1, eps=1e-8)
    second = F.normalize(decayed_image(second_clip.detach()).flatten(1), dim=1, eps=1e-8)
    return (1. - (first * second).sum(dim=1)).clamp(0., 2.).detach()


def embedding_distance(first_embedding, second_embedding):
    return (1. - (first_embedding * second_embedding).sum(dim=1)).clamp(0., 2.)


def support_balanced_reconstruction(reconstruction, target):
    """Average support/background MSE separately for each image, then average."""
    error = (reconstruction - target).square()
    support = ((target.detach() - .5).abs() > (1. / 255.)).to(error.dtype)
    background = 1. - support
    dims = (2, 3, 4)
    foreground_count = support.sum(dim=dims)
    background_count = background.sum(dim=dims)
    foreground = (error * support).sum(dim=dims) / foreground_count.clamp_min(1.)
    background_error = (error * background).sum(dim=dims) / background_count.clamp_min(1.)
    # Ordinary data always has both classes; this handles empty masks sensibly.
    both = ((foreground_count > 0) & (background_count > 0)).to(error.dtype)
    return ((foreground + background_error) / (1. + both)).mean()


def losses(first_result, second_result, first_clip, second_clip, beta=1e-4, pair_weight=1.):
    recon = .5 * (support_balanced_reconstruction(first_result['reconstruction'], first_clip) +
                  support_balanced_reconstruction(second_result['reconstruction'], second_clip))
    def kl(result):
        return .5 * (result['mu'].square() + result['logvar'].exp() - 1. - result['logvar']).mean()
    kl_value = .5 * (kl(first_result) + kl(second_result))
    target = decayed_distance(first_clip, second_clip)
    distance = embedding_distance(first_result['embedding'], second_result['embedding'])
    pair_loss = .5 * (distance - target).square().mean()
    # Excluding the unused projection branch entirely preserves the plain VAE
    # control's absent projection gradients and absent Adam state.
    total = recon + beta * kl_value
    if pair_weight != 0:
        total = total + pair_weight * pair_loss
    full_mse = .5 * ((first_result['reconstruction'] - first_clip).square().mean() +
                     (second_result['reconstruction'] - second_clip).square().mean())
    return dict(loss=total, recon=recon, kl=kl_value, pair_loss=pair_loss,
                pixel_distance=target, embedding_distance=distance, full_mse=full_mse)
