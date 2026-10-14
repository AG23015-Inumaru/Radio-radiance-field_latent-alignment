"""Shared data, checkpoint and reconstruction diagnostics for the map VAE."""
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from ..alignment import split_views
from ..cameras import cache_views, load_views
from ..common import finite_tensor, load_tensor_file
from ..vae import load_image
from .vae_map import GrayMapVAE, MapVAEConfig, loss_terms


class MapViewImages(Dataset):
    def __init__(self, views, indices, config):
        self.views = [views[i] for i in indices]
        self.config = config

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        return load_image(self.views[index], self.config)


def load_split_manifest(path):
    document, digest = load_views(path)
    if not all(v.get("split") in ("train", "val", "test") for v in document["views"]):
        raise ValueError("Every view must have its existing train/val/test split")
    return document, digest, split_views(cache_views(document))


def config_dict(config):
    result = asdict(config)
    result["channels"] = list(config.channels)
    return result


def read_config(path=None, saved=None):
    settings = json.loads(Path(path).expanduser().read_text(encoding="utf-8")) if path else saved
    return MapVAEConfig(**settings) if settings else MapVAEConfig()


def load_map_checkpoint(path, device="cpu"):
    saved = load_tensor_file(path)
    if saved.get("kind") != "gray_map_vae" or saved.get("schema_version") != 1:
        raise ValueError("Expected a map-VAE checkpoint, not the previous vector-VAE checkpoint")
    model = GrayMapVAE(read_config(saved=saved["config"]))
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device).eval().requires_grad_(False), saved


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def tensor_image(tensor):
    pixels = tensor.detach().cpu()[0].clamp(0, 1).numpy()
    return Image.fromarray((pixels * 255).round().astype(np.uint8))


@torch.no_grad()
def validate(model, loader, device, mode="vae", sample_seed=1234):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    totals = {k: 0.0 for k in ("val_mu_mse", "val_sample_mse", "val_mu_l1", "val_kl_sum", "val_kl_mean")}
    count = 0
    for image in loader:
        image = image.to(device)
        mu, logvar = model.encode(image)
        mean_reconstruction = model.decode(mu)
        sample_reconstruction = model.decode(model.reparameterize(mu, logvar, generator)) if mode == "vae" else mean_reconstruction
        mse, kl_sum, kl_mean = loss_terms(mean_reconstruction, image, mu, logvar)
        batch = {"val_mu_mse": mse, "val_sample_mse": (sample_reconstruction - image).square().mean(),
                 "val_mu_l1": (mean_reconstruction - image).abs().mean(),
                 "val_kl_sum": kl_sum if mode == "vae" else mse.new_zeros(()),
                 "val_kl_mean": kl_mean if mode == "vae" else mse.new_zeros(())}
        for key, value in batch.items():
            totals[key] += finite_tensor(value, key).item() * len(image)
        count += len(image)
    if count == 0:
        raise ValueError("Evaluation split is empty")
    result = {key: value / count for key, value in totals.items()}
    for kind in ("mu", "sample"):
        result[f"val_{kind}_psnr_db"] = -10 * math.log10(max(result[f"val_{kind}_mse"], 1e-12))
    return result


@torch.no_grad()
def save_preview(model, dataset, path, device, mode="vae", native_dir=None):
    """Save a labelled overview and optionally two examples at native resolution."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(1234)
    width = min(480, model.config.image_width)
    height = round(width * model.config.image_height / model.config.image_width)
    rows = min(4, len(dataset))
    if not rows:
        raise ValueError("Preview dataset is empty")
    labels = ("Input", "Decode(mu)", "Decode(sampled z)") if mode == "vae" else ("Input", "AE reconstruction")
    sheet = Image.new("L", (len(labels) * width, rows * (height + 24)), color=255)
    draw = ImageDraw.Draw(sheet)
    if native_dir:
        Path(native_dir).mkdir(parents=True, exist_ok=True)
    for index in range(rows):
        image = dataset[index]
        mu, logvar = model.encode(image.unsqueeze(0).to(device))
        tiles = [tensor_image(image), tensor_image(model.decode(mu)[0])]
        if mode == "vae":
            tiles.append(tensor_image(model.decode(model.reparameterize(mu, logvar, generator))[0]))
        for column, (label, tile) in enumerate(zip(labels, tiles)):
            top = index * (height + 24)
            draw.text((column * width + 4, top + 4), f"{label} #{index}", fill=0)
            sheet.paste(tile.resize((width, height), Image.Resampling.LANCZOS), (column * width, top + 24))
            if native_dir and index < 2:
                suffix = ("input", "mu", "sample")[column]
                tile.save(Path(native_dir) / f"{index:02d}_{suffix}.png")
    sheet.save(path)
