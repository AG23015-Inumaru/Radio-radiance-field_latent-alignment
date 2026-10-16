#!/usr/bin/env python3
"""Preserve RF ray-distance bins and learn their aggregation into f32 VAE means.

Use the existing Docker checkout and the accompanying alignment_common_f32.py.
The reference alignment run supplies its RF cache, VAE, manifest and training
settings. RF weights, sampling and source hashes are kept identical. Depth-bin
extraction is resumable; completed caches can be trained without the RF source.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import alignment_common_f32 as base


def load_tensor(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def local_file(directory, relative):
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory)
    except ValueError:
        raise ValueError(f"Cache entry is outside its directory: {path}")
    return path


def pool_distance_bins(features, weights, distances, edges):
    """Return [rays,bins,C+2]: sum(w*h), sum(w), mean_distance/far.

    Bins are [edge[k], edge[k+1]); the final bin includes far. Zero-mass
    bins use their midpoint for the distance channel and retain mass=0.
    This is RF sampling distance, not optical surface depth.
    """
    if features.ndim != 3 or weights.shape != features.shape[:2] or distances.shape != weights.shape:
        raise ValueError("Expected features [R,S,C] and weights/distances [R,S]")
    for value, label in ((features, "RF samples"), (weights, "RF weights"), (distances, "RF distances")):
        base.finite(value, label)
    if (weights < 0).any() or (distances < edges[0] - 1e-6).any() or (distances > edges[-1] + 1e-5).any():
        raise ValueError("Negative RF weights or sample distance outside the declared bins")
    rays, _, channels = features.shape
    bins = len(edges) - 1
    assignment = torch.bucketize(distances.contiguous(), edges[1:-1], right=True)
    pooled = features.new_zeros((rays, bins, channels))
    pooled.scatter_add_(1, assignment[..., None].expand_as(features), weights[..., None] * features)
    mass = weights.new_zeros((rays, bins))
    mass.scatter_add_(1, assignment, weights)
    depth_sum = weights.new_zeros((rays, bins))
    depth_sum.scatter_add_(1, assignment, weights * distances)
    centres = (edges[:-1] + edges[1:]) / 2
    representative = torch.where(mass > 1e-8, depth_sum / mass.clamp_min(1e-8), centres[None])
    return torch.cat((pooled, mass[..., None], (representative / edges[-1])[..., None]), dim=-1)


@torch.no_grad()
def extract_one_view(field, origins, directions, sampling, edges, ray_chunk, point_chunk):
    chunks = []
    positive, count = 0, 0
    low, high = float("inf"), -float("inf")
    device_edges = torch.tensor(edges, dtype=torch.float32, device=field.device)
    for start in range(0, len(origins), ray_chunk):
        origin = torch.as_tensor(origins[start:start + ray_chunk], device=field.device)
        direction = torch.as_tensor(directions[start:start + ray_chunk], device=field.device)
        points, distances = field.samplers.sample_stratified(
            origin, direction, sampling["near"], sampling["far"],
            n_samples=sampling["samples"], perturb=False, inverse_depth=sampling["inverse_depth"])
        raw, hidden = field._query(field.coarse, points, direction, point_chunk, field.model_name == "coarse")
        weights = field._weights(raw, distances, sampling["carrier_ghz"])
        if field.model_name == "fine":
            points, distances, _, _ = field.samplers.sample_hierarchical(
                origin, direction, distances, weights,
                n_new_samples=sampling["fine_samples"], perturb=False)
            raw, hidden = field._query(field.selected, points, direction, point_chunk, True)
            weights = field._weights(raw, distances, sampling["carrier_ghz"])
        chunks.append(pool_distance_bins(hidden, weights, distances, device_edges).cpu())
        density = raw[..., -1]
        positive += int((density > 0).sum())
        count += density.numel()
        low, high = min(low, float(density.min())), max(high, float(density.max()))
    values = torch.cat(chunks)
    mass = values[..., -2]
    total_mass = mass.sum(1)
    stats = {"raw_density_min": low, "raw_density_max": high,
             "raw_density_positive_fraction": positive / count,
             "mass_mean": float(total_mass.mean()), "mass_max": float(total_mass.max()),
             "mass_nonzero_fraction": float((total_mass > 1e-8).float().mean()),
             "bin_mass_mean": mass.mean(0).tolist(),
             "bin_nonzero_fraction": (mass > 1e-8).float().mean(0).tolist()}
    return values, stats


def check_shard(record, view_id, signature, shape):
    if (record.get("kind") != "rf_depth_bin_view" or record.get("schema_version") != 1
            or record.get("view_id") != view_id or record.get("feature_signature") != signature):
        raise ValueError(f"Wrong depth-cache record for {view_id}")
    value = base.finite(record["features"], "RF distance-bin features").float()
    if list(value.shape) != list(shape):
        raise ValueError(f"Depth-cache shape differs for {view_id}: {list(value.shape)}")
    if (value[:, -2] < 0).any() or (value[:, -1] < -1e-6).any() or (value[:, -1] > 1 + 1e-6).any():
        raise ValueError(f"Invalid mass or normalized sample distance in {view_id}")
    return value


def prepare_depth_cache(args, document, metadata, manifest_sha, reference, old_x,
                        reference_sha, cameras, common, field_type):
    original = reference["provenance"]
    sampling = original["sampling"]
    keys = ("near", "far", "samples", "fine_samples", "inverse_depth", "carrier_ghz", "height", "width")
    if any(key not in sampling for key in keys):
        raise ValueError("The reference RF cache lacks its complete sampling settings")
    if not (0 < sampling["near"] < sampling["far"]):
        raise ValueError("Invalid near/far in the reference RF cache")
    if sampling["samples"] < 3 or sampling["fine_samples"] < 1 or sampling["carrier_ghz"] <= 0:
        raise ValueError("Invalid sample counts or carrier frequency in the reference cache")
    if original.get("pooling") != "sum(w*h),sum(w)":
        raise ValueError("Reference RF pooling differs from the previous experiment")
    channels = old_x.shape[1] - 1
    if list(old_x.shape[2:]) != [sampling["height"], sampling["width"]]:
        raise ValueError("RF spatial shape and reference sampling settings differ")
    # NeWRF's sampler may force its first point to distance approximately zero.
    edges = np.linspace(0, sampling["far"], args.depth_bins + 1).tolist()
    shape = [args.depth_bins, channels + 2, sampling["height"], sampling["width"]]
    provenance = {**original, "pooling": "distance_bins:sum(w*h),sum(w),weighted_mean_z/far",
                  "bin_edges": edges, "reference_feature_signature": reference["feature_signature"]}
    signature = common.fingerprint(provenance)
    spec = {"schema_version": 1, "kind": "rf_depth_bins", "feature_shape": shape,
            "manifest_sha256": manifest_sha, "reference_features_sha256": reference_sha,
            "feature_signature": signature, "provenance": provenance, "bin_edges": edges}
    directory = Path(args.cache_dir)
    spec_path = directory / "spec.json"
    index_path = directory / "index.json"
    if directory.exists():
        if not spec_path.is_file() or base.read_json(spec_path) != spec:
            raise ValueError(f"Depth-cache settings differ; choose a new --cache-dir: {directory}")
    else:
        if args.train_only:
            raise FileNotFoundError(f"Depth cache is missing: {directory}")
        directory.mkdir(parents=True, exist_ok=False)
        base.write_json(spec_path, spec)
    if args.train_only and not index_path.is_file():
        raise ValueError("Depth extraction is incomplete; run again without --train-only")
    if index_path.is_file():
        completed = base.read_json(index_path)
        if any(completed.get(key) != value for key, value in spec.items()):
            raise ValueError("Completed depth-cache index differs from its extraction specification")
        base.view_lookup(completed["views"], [v["view_id"] for v in metadata], "Depth cache")
        print(f"Reusing completed RF distance-bin cache: {directory}", flush=True)
        return index_path
    filenames = [f"{i:06d}.pt" for i in range(len(metadata))]
    missing = [name for name in filenames if not (directory / name).is_file()]
    field = None
    if missing:
        if not all((args.newrf_root, args.newrf_config, args.newrf_checkpoint)):
            raise ValueError("Re-extraction requires --newrf-root, --newrf-config and --newrf-checkpoint "
                             "from the original RF extraction. The pooled .pt cannot recover distance bins.")
        print(f"Loading frozen NeWRF ({original['newrf']['model']}); {len(missing)} views need extraction.", flush=True)
        field = field_type(args.newrf_root, args.newrf_config, args.newrf_checkpoint,
                           original["newrf"]["model"], args.device)
        if field.provenance != original["newrf"] or field.feature_dim != channels:
            raise ValueError("NeWRF model/checkpoint/config/source differs from the reference RF cache")
    entries = []
    for i, (view, filename) in enumerate(zip(metadata, filenames)):
        path = directory / filename
        if path.is_file():
            record = load_tensor(path)
            check_shard(record, view["view_id"], signature, shape)
        else:
            origins, directions = cameras.camera_rays(document["views"][i], document,
                                                       sampling["width"], sampling["height"])
            values, diagnostics = extract_one_view(field, origins, directions, sampling, edges,
                                                   args.ray_chunk, args.point_chunk)
            features = values.permute(1, 2, 0).reshape(shape).contiguous()
            # Summing distance bins must reproduce the original extraction.
            collapsed = torch.cat((features[:, :-2].sum(0), features[:, -2].sum(0, keepdim=True)), dim=0)
            expected = old_x[i]
            error = float((collapsed - expected).square().mean().sqrt())
            scale = float(expected.square().mean().sqrt())
            relative_error = error / max(scale, 1e-12)
            if error > 1e-5 and relative_error > 0.005:
                raise ValueError(f"{view['view_id']}: distance-bin sum does not reproduce reference RF "
                                 f"features (relative RMS error {relative_error:.6g})")
            diagnostics["collapsed_reference_relative_rms_error"] = relative_error
            record = {"schema_version": 1, "kind": "rf_depth_bin_view", "view_id": view["view_id"],
                      "feature_signature": signature, "features": features, "diagnostics": diagnostics}
            check_shard(record, view["view_id"], signature, shape)
            base.save_checkpoint(path, record)
        entries.append({**view, "cache_file": filename, "sha256": base.sha256(path),
                        "diagnostics": record["diagnostics"]})
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(metadata):
            print(f"RF distance bins {i + 1}/{len(metadata)}: mass_nonzero="
                  f"{record['diagnostics']['mass_nonzero_fraction']:.3f}, shape={tuple(shape)}", flush=True)
    if not any(row["diagnostics"]["mass_max"] > 1e-8 for row in entries):
        raise ValueError("All RF rendering mass is zero; no alignment training was started")
    base.write_json(index_path, {**spec, "views": entries})
    del field
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return index_path


class DepthDataset(Dataset):
    def __init__(self, index_path, metadata, targets, cache_mode):
        self.index_path = Path(index_path)
        self.index = base.read_json(index_path)
        self.shape = self.index["feature_shape"]
        self.signature = self.index["feature_signature"]
        lookup = base.view_lookup(self.index["views"], [v["view_id"] for v in metadata], "Depth cache")
        self.entries = [self.index["views"][lookup[v["view_id"]]] for v in metadata]
        self.targets, self.memory = targets, None
        for actual, expected in zip(self.entries, metadata):
            base.match_metadata(actual, expected, "Depth cache")
        gib = len(metadata) * math.prod(self.shape) * 4 / 1024 ** 3
        print(f"RF feature storage: {gib:.2f} GiB float32; cache-mode={cache_mode}", flush=True)
        if cache_mode == "ram":
            self.memory = torch.empty((len(metadata), *self.shape), dtype=torch.float32)
        for i, entry in enumerate(self.entries):
            path = local_file(self.index_path.parent, entry["cache_file"])
            if base.sha256(path) != entry["sha256"]:
                raise ValueError(f"RF cache file changed: {path}")
            value = self.read(i)
            if self.memory is not None:
                self.memory[i].copy_(value)

    def read(self, index):
        entry = self.entries[index]
        path = local_file(self.index_path.parent, entry["cache_file"])
        return check_shard(load_tensor(path), entry["view_id"], self.signature, self.shape)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        value = self.memory[index] if self.memory is not None else self.read(index)
        return value, self.targets[index], index


def make_loader(dataset, indices, args, shuffle=False):
    return DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                      shuffle=shuffle, num_workers=args.workers,
                      generator=torch.Generator().manual_seed(args.seed),
                      pin_memory=str(args.device).startswith("cuda"))


def training_statistics(dataset, train, args):
    channels = dataset.shape[1]
    sums = torch.zeros((1, 1, channels, 1, 1), dtype=torch.float64)
    squares = torch.zeros_like(sums)
    total = 0
    for values, _, _ in make_loader(dataset, train, args):
        values = values.double()
        sums += values.sum((0, 1, 3, 4), keepdim=True)
        squares += values.square().sum((0, 1, 3, 4), keepdim=True)
        total += values.shape[0] * values.shape[1] * values.shape[3] * values.shape[4]
    mean = sums / total
    std = (squares / total - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    targets = dataset.targets[train].double()
    y_mean = targets.mean(0, keepdim=True)
    y_std = (targets - y_mean).square().mean((0, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
    return {"x_mean": mean.float(), "x_std": std.float(), "y_mean": y_mean.float(), "y_std": y_std.float()}


class DepthRFToMeanMap(nn.Module):
    def __init__(self, feature_channels, bin_centres, latent_shape):
        super().__init__()
        self.register_buffer("bin_centres", torch.tensor(bin_centres, dtype=torch.float32).reshape(-1, 1))
        self.bin_encoder = nn.Sequential(nn.Conv2d(feature_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(), base.ResidualBlock(64))
        self.distance_embedding = nn.Linear(1, 64)
        self.score = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.spatial = base.RFToMeanMap(64, latent_shape)

    def forward(self, normalized, mass, return_attention=False):
        batch, bins, channels, height, width = normalized.shape
        if bins != len(self.bin_centres):
            raise ValueError("Input distance-bin count differs from the model")
        features = self.bin_encoder(normalized.reshape(batch * bins, channels, height, width))
        features = features.reshape(batch, bins, 64, height, width)
        position = self.distance_embedding(self.bin_centres).reshape(1, bins, 64, 1, 1)
        features = features + position
        scores = self.score(features.reshape(batch * bins, 64, height, width)).reshape(batch, bins, height, width)
        valid = mass > 1e-8
        # All-zero rays fall back to a finite softmax over all bins.
        allowed = valid | ~valid.any(dim=1, keepdim=True)
        attention = scores.masked_fill(~allowed, -1e4).softmax(dim=1)
        selected = (features * attention[:, :, None]).sum(dim=1)
        result = self.spatial(selected)
        return (result, attention) if return_attention else result


@torch.no_grad()
def predict(model, dataset, indices, stats, args, with_attention=False):
    model.eval()
    gpu = {key: value.to(args.device) for key, value in stats.items()}
    predictions, attentions, ray_valid = [], [], []
    for values, _, _ in make_loader(dataset, indices, args):
        values = values.to(args.device)
        normalized = (values - gpu["x_mean"]) / gpu["x_std"]
        output, attention = model(normalized, values[:, :, -2], return_attention=True)
        mu = output * gpu["y_std"] + gpu["y_mean"]
        predictions.append(base.finite(mu.cpu(), "Predicted means"))
        if with_attention:
            attentions.append(attention.cpu())
            ray_valid.append((values[:, :, -2] > 1e-8).any(1).cpu())
    prediction = torch.cat(predictions)
    if with_attention:
        return prediction, torch.cat(attentions), torch.cat(ray_valid)
    return prediction


@torch.no_grad()
def save_comparison(vae, pred, target, mean, views, path, device, load_image):
    width = 360
    height = round(width * vae.config.image_height / vae.config.image_width)
    rows = min(4, len(views))
    sheet = Image.new("L", (4 * width, rows * (height + 24)), 255)
    draw = ImageDraw.Draw(sheet)
    mean_image = vae.decode(mean.to(device))
    for i in range(rows):
        tiles = (load_image(views[i], vae.config).unsqueeze(0).to(device),
                 vae.decode(target[i:i + 1].to(device)), mean_image,
                 vae.decode(pred[i:i + 1].to(device)))
        for j, (label, tensor) in enumerate(zip(("Input", "VAE target mu", "Train mean mu", "RF depth predicted mu"), tiles)):
            array = tensor[0, 0].detach().cpu().clamp(0, 1).numpy()
            image = Image.fromarray((array * 255).round().astype(np.uint8))
            top = i * (height + 24)
            draw.text((j * width + 4, top + 4), f"{label} #{i}", fill=0)
            sheet.paste(image.resize((width, height), Image.Resampling.LANCZOS), (j * width, top + 24))
    sheet.save(path)


def save_attention(attention, valid, edges, out):
    rows, bins = min(4, len(attention)), attention.shape[1]
    width, height = 288, 192
    sheet = Image.new("L", (bins * width, rows * (height + 30)), 255)
    draw = ImageDraw.Draw(sheet)
    for i in range(rows):
        for k in range(bins):
            array = attention[i, k].numpy()
            # Zero-mass rays are shown as black, not interpreted as measured depth.
            array = array * valid[i].numpy()
            tile = Image.fromarray((array.clip(0, 1) * 255).round().astype(np.uint8))
            top = i * (height + 30)
            draw.text((k * width + 3, top + 3), f"RF attention #{i}: [{edges[k]:.2f},{edges[k+1]:.2f}]", fill=0)
            sheet.paste(tile.resize((width, height), Image.Resampling.NEAREST), (k * width, top + 30))
    sheet.save(out / "val_rf_bin_attention.png")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workspace", default="/workspace")
    result.add_argument("--data", default="/workspace/temp/alignment_dataset/random2000_4dir_v2000")
    result.add_argument("--baseline-run", help="Default: DATA/runs/align_f32_c4_run001")
    result.add_argument("--reference-features", help="Original pooled RF cache; default from baseline arguments")
    result.add_argument("--vae-checkpoint")
    result.add_argument("--views")
    result.add_argument("--targets", help="Existing f32 mean-map directory or index.json")
    result.add_argument("--newrf-root")
    result.add_argument("--newrf-config")
    result.add_argument("--newrf-checkpoint")
    result.add_argument("--depth-bins", type=int, default=4)
    result.add_argument("--cache-dir")
    result.add_argument("--cache-mode", choices=("ram", "disk"), default="ram")
    result.add_argument("--output-dir")
    for name, kind in (("epochs", int), ("batch-size", int), ("lr", float),
                       ("weight-decay", float), ("patience", int), ("seed", int)):
        result.add_argument("--" + name, type=kind)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--ray-chunk", type=int, default=32)
    result.add_argument("--point-chunk", type=int, default=8192)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--extract-only", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    return result


def main(argv=None):
    options = parser()
    args = options.parse_args(argv)
    args.workspace = str(Path(args.workspace).expanduser().resolve())
    args.data = str(Path(args.data).expanduser().resolve())
    data = Path(args.data)
    # Paths recovered from the original NIST feature-extraction command.
    newrf_run = Path(args.workspace) / "dataset" / "NEWRF" / "rf3dgs_rx_cos0999_sameinit_100k"
    args.newrf_root = args.newrf_root or str(newrf_run / "source")
    args.newrf_config = args.newrf_config or str(newrf_run / "config" / "j4ru0_effective.yaml")
    args.newrf_checkpoint = args.newrf_checkpoint or str(newrf_run / "checkpoint" / "nist_rf3dgs_J4RU0" / "ckpt.pt")
    source = Path(args.baseline_run).expanduser().resolve() if args.baseline_run else data / "runs" / "align_f32_c4_run001"
    old_arguments = base.read_json(source / "arguments.json") if (source / "arguments.json").is_file() else {}
    defaults = {"epochs": 200, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4, "patience": 30, "seed": 0}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, old_arguments.get(key, value))
    if min(args.epochs, args.batch_size, args.patience, args.ray_chunk, args.point_chunk) < 1 or args.depth_bins < 2 or args.workers < 0:
        options.error("Positive training/chunk sizes, depth-bins >= 2, workers >= 0 are required")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        options.error("lr must be positive and weight-decay nonnegative")
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else data / "runs" / f"align_depth{args.depth_bins}_f32_c4_run001"
    args.output_dir, args.baseline_run = str(out), str(source)
    if out.exists() and not args.extract_only:
        raise FileExistsError(f"Choose a new --output-dir; run already exists: {out}")
    args.cache_dir = str(Path(args.cache_dir).expanduser().resolve()) if args.cache_dir else str(data / "rf" / f"depth{args.depth_bins}_run001")
    checkpoint = Path(args.vae_checkpoint or old_arguments.get("vae_checkpoint", data / "vae" / "map_f32_c4_run002" / "best.pt")).expanduser().resolve()
    args.vae_checkpoint = str(checkpoint)
    sys.path.insert(0, args.workspace)
    package = "scripts.newrf_latent_alignment"
    common = importlib.import_module(package + ".common")
    cameras = importlib.import_module(package + ".views.cameras")
    maps = importlib.import_module(package + ".spatial_vae.map_utils")
    encode = importlib.import_module(package + ".spatial_vae.encode_vae_map").main
    load_image = importlib.import_module(package + ".vae.vae").load_image
    args.views = args.views or old_arguments.get("views")
    if not args.views:
        args.views = base.read_json(checkpoint.parent / "arguments.json")["views"]
    args.views = str((Path(args.workspace) / Path(args.views).expanduser()).resolve())
    print(f"Loading frozen f32 VAE: {checkpoint}", flush=True)
    vae, saved = maps.load_map_checkpoint(checkpoint, "cpu")
    if vae.config.factor != 32 or tuple(vae.config.latent_shape) != (4, 25, 37):
        raise ValueError("This experiment requires the existing f32_c4 VAE with [4,25,37] means")
    print("Reading existing receiver-grouped splits...", flush=True)
    document, manifest_sha, splits = maps.load_split_manifest(args.views)
    if saved["manifest_sha256"] != manifest_sha:
        raise ValueError("Manifest differs from the f32 VAE training manifest")
    metadata = cameras.cache_views(document)
    reference_arg = args.reference_features or old_arguments.get("features")
    reference_path, reference = base.find_rf(data, reference_arg, load_tensor, metadata, manifest_sha)
    args.reference_features = str(reference_path)
    reference_sha, vae_sha = base.sha256(reference_path), base.sha256(checkpoint)
    if not args.targets:
        args.targets = old_arguments.get("targets")
    target_path = base.obtain_targets(args, checkpoint, vae_sha, manifest_sha, data, encode)
    args.targets = str(target_path)
    print("Checking reference RF, f32 means, camera correspondence and image identities...", flush=True)
    old_x, y, target_index, targets_sha = base.pair_inputs(
        reference, target_path, metadata, manifest_sha, vae_sha, vae.config, load_tensor)
    field_type = None
    if not args.train_only:
        field_type = importlib.import_module(package + ".rf.newrf").FrozenNeWRF
    index_path = prepare_depth_cache(args, document, metadata, manifest_sha, reference, old_x,
                                     reference_sha, cameras, common, field_type)
    del old_x, reference
    print(f"Completed RF distance-bin cache: {index_path}", flush=True)
    if args.extract_only:
        return
    dataset = DepthDataset(index_path, metadata, y, args.cache_mode)
    print("Computing train-only input and target statistics...", flush=True)
    stats = training_statistics(dataset, splits["train"], args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    edges = dataset.index["bin_edges"]
    centres = [(left + right) / (2 * edges[-1]) for left, right in zip(edges[:-1], edges[1:])]
    config = {"feature_channels": dataset.shape[1], "bin_centres": centres, "latent_shape": list(y.shape[1:])}
    model = DepthRFToMeanMap(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(dataset, splits["train"], args, shuffle=True)
    gpu = {key: value.to(args.device) for key, value in stats.items()}
    provenance = {"manifest_sha256": manifest_sha, "vae_sha256": vae_sha,
                  "vae_config": target_index["vae_config"], "targets_sha256": targets_sha,
                  "reference_features_sha256": reference_sha, "depth_index_sha256": base.sha256(index_path),
                  "feature_signature": dataset.signature, "feature_provenance": dataset.index["provenance"],
                  "script_sha256": base.sha256(__file__),
                  "helper_sha256": base.sha256(Path(base.__file__))}
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / "arguments.json", vars(args))
    base.write_json(out / "model_config.json", config)
    base.write_json(out / "splits.json", {name: [metadata[i]["view_id"] for i in indices] for name, indices in splits.items()})
    print(f"Learning RF {tuple(dataset.shape)} -> f32 mean map {tuple(y.shape[1:])}", flush=True)
    print(f"Output: {out}", flush=True)
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for values, target, _ in loader:
            values, target = values.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model((values - gpu["x_mean"]) / gpu["x_std"], values[:, :, -2])
            prediction = prediction * gpu["y_std"] + gpu["y_mean"]
            loss = base.finite(F.mse_loss(prediction, target), "Training MSE")
            loss.backward()
            base.finite(nn.utils.clip_grad_norm_(model.parameters(), 10.0), "Gradient norm")
            optimizer.step()
            total += float(loss.detach()) * len(target)
            count += len(target)
        val = predict(model, dataset, splits["val"], stats, args)
        metrics = base.latent_metrics(val, y[splits["val"]], stats["y_mean"])
        row = {"epoch": epoch, "train_mse": total / count, **metrics}
        history.append(row)
        if metrics["latent_mse"] < best:
            best, best_epoch, stale = metrics["latent_mse"], epoch, 0
            base.save_checkpoint(out / "best.pt", {"schema_version": 1, "kind": "depth_rf_to_mean_map",
                "epoch": epoch, "val_mse": best, "model_config": config,
                "model_state_dict": model.state_dict(), "statistics": stats,
                "views": metadata, "splits": splits, **provenance})
        else:
            stale += 1
        base.write_json(out / "history.json", history)
        print(f"Epoch {epoch}: train_MSE={row['train_mse']:.7f}, val_mu_MSE={metrics['latent_mse']:.7f}, "
              f"mean_baseline={metrics['mean_map_baseline_mse']:.7f}, best_epoch={best_epoch}", flush=True)
        if stale >= args.patience:
            print(f"Early stopping after {stale} epochs without improvement.", flush=True)
            break
    model.load_state_dict(load_tensor(out / "best.pt")["model_state_dict"], strict=True)
    predictions, report = {}, {"best_epoch": best_epoch, "splits": {}}
    attention, valid = None, None
    for name in ("train", "val"):
        indices = splits[name]
        if name == "val":
            pred, attention, valid = predict(model, dataset, indices, stats, args, with_attention=True)
        else:
            pred = predict(model, dataset, indices, stats, args)
        target = y[indices]
        report["splits"][name] = {"views": len(indices), **base.latent_metrics(pred, target, stats["y_mean"])}
        predictions[name] = {"pred_mu": pred, "target_mu": target, "views": [metadata[i] for i in indices]}
    common.save_new(out / "predictions.pt", {"schema_version": 1, "kind": "mean_map_predictions",
                                           "splits": predictions, **provenance})
    base.write_json(out / "metrics.json", report)
    val = predictions["val"]
    report["validation_images"] = base.decode_validation(
        vae, val["pred_mu"], val["target_mu"], val["views"], out, args.device, load_image)
    for name in ("train", "val"):
        part = predictions[name]
        save_comparison(vae, part["pred_mu"], part["target_mu"], stats["y_mean"], part["views"],
                        out / f"{name}_with_mean_baseline.png", args.device, load_image)
    save_attention(attention, valid, edges, out)
    valid_count = float(valid.sum())
    report["rf_attention"] = {"bin_edges": edges, "nonzero_mass_ray_fraction": float(valid.float().mean()),
        "mean_weights_on_nonzero_mass_rays": ((attention * valid[:, None]).sum((0, 2, 3)) / max(valid_count, 1)).tolist()}
    baseline_metrics = source / "metrics.json"
    baseline_checkpoint = source / "best.pt"
    if baseline_metrics.is_file() and baseline_checkpoint.is_file():
        baseline_saved = load_tensor(baseline_checkpoint)
        matches = (baseline_saved.get("manifest_sha256") == manifest_sha
                   and baseline_saved.get("vae_sha256") == vae_sha
                   and baseline_saved.get("features_sha256") == reference_sha)
        if matches:
            prior = base.read_json(baseline_metrics)
            report["reference_alignment"] = {"run": str(source), "metrics": prior,
                "val_latent_mse_change": report["splits"]["val"]["latent_mse"] - prior["splits"]["val"]["latent_mse"],
                "val_image_psnr_db_change": report["validation_images"]["rf_image_psnr_db"] - prior["validation_images"]["rf_image_psnr_db"]}
        else:
            report["reference_alignment"] = {"run": str(source), "comparison_skipped": "Input identities differ"}
    base.write_json(out / "metrics.json", report)
    print(f"Saved depth-aware alignment checkpoint: {out / 'best.pt'}", flush=True)
    print(f"Validation RF image PSNR: {report['validation_images']['rf_image_psnr_db']:.2f} dB", flush=True)
    print(f"Validation comparison: {out / 'best_val_preview.png'}", flush=True)
    if "val_image_psnr_db_change" in report.get("reference_alignment", {}):
        print(f"PSNR change from reference: {report['reference_alignment']['val_image_psnr_db_change']:+.2f} dB", flush=True)


if __name__ == "__main__":
    main()
