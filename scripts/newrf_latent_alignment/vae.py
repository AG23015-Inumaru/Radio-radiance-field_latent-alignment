"""VAE architecture retained from the user's train_projection_head2.py."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn as nn

from .common import load_tensor_file

@dataclass
class VAEConfig:
    image_height: int = 800
    image_width: int = 1200
    in_channels: int = 1
    latent_dim: int = 32
    channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 256)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class GrayVectorVAE(nn.Module):
    """A convolutional VAE whose per-image latent is a vector [latent_dim]."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        encoder_layers: List[nn.Module] = []
        in_ch = config.in_channels
        for out_ch in config.channels:
            encoder_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    _group_norm(out_ch),
                    nn.SiLU(inplace=True),
                ]
            )
            in_ch = out_ch
        self.encoder_cnn = nn.Sequential(*encoder_layers)

        spatial_sizes = [(config.image_height, config.image_width)]
        h, w = config.image_height, config.image_width
        for _ in config.channels:
            h, w = h // 2, w // 2
            if h < 1 or w < 1:
                raise ValueError("Image size is too small for the number of downsampling blocks.")
            spatial_sizes.append((h, w))

        self.encoded_height, self.encoded_width = spatial_sizes[-1]
        flat_dim = config.channels[-1] * self.encoded_height * self.encoded_width
        self.flat_dim = flat_dim

        self.fc_mu = nn.Linear(flat_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, config.latent_dim)
        self.fc_decode = nn.Linear(config.latent_dim, flat_dim)

        decoder_output_channels = list(reversed(config.channels[:-1])) + [config.in_channels]
        decoder_layers: List[nn.Module] = []

        current_h, current_w = spatial_sizes[-1]
        decoder_in_ch = config.channels[-1]
        target_sizes = list(reversed(spatial_sizes[:-1]))

        for i, (out_ch, (target_h, target_w)) in enumerate(
            zip(decoder_output_channels, target_sizes)
        ):
            output_padding = (target_h - 2 * current_h, target_w - 2 * current_w)
            if output_padding[0] not in (0, 1) or output_padding[1] not in (0, 1):
                raise ValueError(f"Invalid decoder output_padding={output_padding}")

            decoder_layers.append(
                nn.ConvTranspose2d(
                    decoder_in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=output_padding,
                )
            )

            is_last = i == len(decoder_output_channels) - 1
            if not is_last:
                decoder_layers.extend([_group_norm(out_ch), nn.SiLU(inplace=True)])

            current_h, current_w = target_h, target_w
            decoder_in_ch = out_ch

        decoder_layers.append(nn.Sigmoid())
        self.decoder_cnn = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.encoder_cnn(x).flatten(start_dim=1)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features).clamp(min=-12.0, max=12.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.config.channels[-1], self.encoded_height, self.encoded_width)
        return self.decoder_cnn(h)

    def forward(self, x: Tensor, sample: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon = self.decode(z)
        return recon, mu, logvar


def load_frozen_vae(checkpoint, config_path=None, device="cpu"):
    """Load the user's train_gray_vae2 / train_projection_head2 checkpoint."""
    checkpoint = Path(checkpoint).expanduser()
    saved = load_tensor_file(checkpoint)
    if not isinstance(saved, dict):
        raise TypeError("Expected a VAE state_dict or a dictionary checkpoint")
    config = dict(saved.get("config", {}))
    if config_path is not None:
        supplied = json.loads(Path(config_path).expanduser().read_text())
        for key, value in supplied.items():
            def comparable(item):
                if key != "channels":
                    return item
                return tuple(int(x) for x in (item.split(",") if isinstance(item, str) else item))
            if key in config and comparable(config[key]) != comparable(value):
                raise ValueError(f"VAE configuration conflicts with checkpoint: {key}")
        config.update(supplied)
    required = ("image_height", "image_width", "in_channels", "latent_dim", "channels")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing VAE settings {missing}; supply --vae-config JSON")
    channels = config["channels"]
    if isinstance(channels, str):
        channels = [int(value) for value in channels.split(",")]
    cfg = VAEConfig(image_height=int(config["image_height"]), image_width=int(config["image_width"]),
                    in_channels=int(config["in_channels"]), latent_dim=int(config["latent_dim"]),
                    channels=tuple(channels))
    if cfg.in_channels != 1:
        raise ValueError("This checkpoint adapter expects the existing grayscale VAE")
    state = saved.get("model_state", saved.get("model_state_dict", saved))
    vae = GrayVectorVAE(cfg)
    vae.load_state_dict(state, strict=True)
    return vae.to(device).eval().requires_grad_(False)


def load_image(view, config):
    """Keep the grayscale and Lanczos preprocessing of the previous VAE code."""
    with Image.open(view["image"]) as image:
        expected = (view["intrinsics"]["width"], view["intrinsics"]["height"])
        if image.size != expected:
            raise ValueError(f'{view["view_id"]}: image size {image.size} != camera size {expected}')
        if not math.isclose(image.width / image.height, config.image_width / config.image_height,
                            rel_tol=1e-6):
            raise ValueError("Teacher image and VAE must have the same aspect ratio")
        image = image.convert("L").resize((config.image_width, config.image_height), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).unsqueeze(0) / 255.0
