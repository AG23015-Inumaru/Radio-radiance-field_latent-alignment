"""A grayscale VAE with a spatial Gaussian latent [B, C, H/f, W/f]."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MapVAEConfig:
    image_height: int = 800
    image_width: int = 1200
    in_channels: int = 1
    latent_channels: int = 4
    channels: tuple = (32, 64, 128)
    residual_blocks: int = 1

    def __post_init__(self):
        self.channels = tuple(self.channels)
        values = (self.image_height, self.image_width, self.latent_channels, *self.channels)
        if not self.channels or any(type(x) is not int or x < 1 for x in values):
            raise ValueError("Image dimensions and channel counts must be positive integers")
        if self.in_channels != 1:
            raise ValueError("This experiment preserves the existing grayscale preprocessing")
        if type(self.residual_blocks) is not int or self.residual_blocks < 0:
            raise ValueError("residual_blocks must be a nonnegative integer")
        if self.image_height % self.factor or self.image_width % self.factor:
            raise ValueError(f"Image height and width must be divisible by {self.factor}")

    @property
    def factor(self):
        return 2 ** len(self.channels)

    @property
    def latent_shape(self):
        return (self.latent_channels, self.image_height // self.factor,
                self.image_width // self.factor)


def group_norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            group_norm(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            group_norm(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class GrayMapVAE(nn.Module):
    """No flatten/dense bottleneck or encoder-to-decoder skip connections."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MapVAEConfig()
        cfg = self.config
        layers, previous = [], cfg.in_channels
        for channels in cfg.channels:
            layers.extend([nn.Conv2d(previous, channels, 4, stride=2, padding=1),
                           group_norm(channels), nn.SiLU()])
            layers.extend(ResidualBlock(channels) for _ in range(cfg.residual_blocks))
            previous = channels
        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Conv2d(previous, cfg.latent_channels, 1)
        self.logvar_head = nn.Conv2d(previous, cfg.latent_channels, 1)
        self.from_latent = nn.Conv2d(cfg.latent_channels, previous, 3, padding=1)
        self.bottleneck = nn.Sequential(
            *(ResidualBlock(previous) for _ in range(cfg.residual_blocks)))
        stages = []
        outputs = list(reversed(cfg.channels[:-1])) + [cfg.in_channels]
        for i, channels in enumerate(outputs):
            stage = [nn.Conv2d(previous, channels, 3, padding=1)]
            if i < len(outputs) - 1:
                stage.extend([group_norm(channels), nn.SiLU()])
                stage.extend(ResidualBlock(channels) for _ in range(cfg.residual_blocks))
            stages.append(nn.Sequential(*stage))
            previous = channels
        self.decoder_stages = nn.ModuleList(stages)

    def encode(self, image):
        expected = (self.config.in_channels, self.config.image_height, self.config.image_width)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected:
            raise ValueError(f"Expected input [B, {expected}], got {tuple(image.shape)}")
        features = self.encoder(image)
        return self.mu_head(features), self.logvar_head(features).clamp(-12.0, 12.0)

    @staticmethod
    def reparameterize(mu, logvar, generator=None):
        noise = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        return mu + torch.exp(0.5 * logvar) * noise

    def decode(self, latent):
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.config.latent_shape:
            raise ValueError(f"Expected spatial latent [B, {self.config.latent_shape}]")
        features = self.bottleneck(self.from_latent(latent))
        for stage in self.decoder_stages:
            features = stage(F.interpolate(features, scale_factor=2, mode="nearest"))
        return torch.sigmoid(features)

    def forward(self, image, sample=True, generator=None):
        mu, logvar = self.encode(image)
        latent = self.reparameterize(mu, logvar, generator) if sample else mu
        return self.decode(latent), mu, logvar


def loss_terms(reconstruction, image, mu, logvar, recon_loss="mse"):
    """Both KL reductions are batch means; sum covers ALL C/H/W elements."""
    if recon_loss not in ("mse", "l1"):
        raise ValueError("recon_loss must be mse or l1")
    error = reconstruction - image
    reconstruction_loss = error.square().mean() if recon_loss == "mse" else error.abs().mean()
    element_kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar)
    kl_sum = element_kl.sum(dim=(1, 2, 3)).mean()
    kl_mean = element_kl.mean()
    return reconstruction_loss, kl_sum, kl_mean
