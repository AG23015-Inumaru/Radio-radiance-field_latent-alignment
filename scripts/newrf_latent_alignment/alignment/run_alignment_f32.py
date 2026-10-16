#!/usr/bin/env python3
"""Train RF features -> frozen f32 VAE means, then save validation images.

Run in the existing Docker checkout. No NeWRF/VAE weights are updated.
Existing RF caches are required. Matching extracted f32 means are reused;
otherwise the existing spatial-VAE exporter creates them once.
"""
import argparse
import hashlib
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
from torch.utils.data import DataLoader, TensorDataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def finite(tensor, label):
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise ValueError(f"{label} must be a floating-point tensor")
    for batch in tensor.reshape(-1).split(4_000_000):
        if not torch.isfinite(batch).all():
            raise ValueError(f"{label} contains NaN or infinity")
    return tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.layers(x)


class RFToMeanMap(nn.Module):
    """Keep spatial features throughout; output unbounded VAE means."""
    def __init__(self, in_channels, latent_shape):
        super().__init__()
        self.latent_shape = tuple(latent_shape)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(), ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(), ResidualBlock(128),
            ResidualBlock(128))
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(), ResidualBlock(64),
            nn.Conv2d(64, self.latent_shape[0], 1))

    def forward(self, x):
        features = self.encoder(x)
        features = F.interpolate(features, size=self.latent_shape[1:],
                                 mode="bilinear", align_corners=False)
        return self.head(features)


def find_rf(data, supplied, load, metadata, manifest_sha):
    if supplied:
        path = Path(supplied).expanduser().resolve()
        return path, load(path)
    # Search only the experiment's RF directory and RF-named root files.
    paths = set((data / "rf").rglob("*.pt")) | set(data.glob("rf*.pt"))
    candidates, seen = [], []
    expected_ids = {view["view_id"] for view in metadata}
    for path in sorted(paths):
        cache = load(path)
        if isinstance(cache, dict) and cache.get("kind") == "rf_features":
            shape = list(cache["features"].shape)
            seen.append((path, shape))
            ids = [view["view_id"] for view in cache.get("views", [])]
            if (cache.get("manifest_sha256") == manifest_sha and len(ids) == len(metadata)
                    and set(ids) == expected_ids):
                candidates.append((path, shape))
        del cache
    if len(candidates) != 1:
        detail = "\n".join(f"  {path}: {shape}" for path, shape in seen)
        raise ValueError("Specify --features with the extracted full RF cache. "
                         f"Found {len(candidates)} matching full caches. All RF caches:\n{detail}")
    path = candidates[0][0]
    return path, load(path)


def target_index_path(path):
    path = Path(path).expanduser().resolve()
    return path / "index.json" if path.is_dir() or path.suffix != ".json" else path


def obtain_targets(args, checkpoint, checkpoint_sha, manifest_sha, data, encode):
    if args.targets:
        index_path = target_index_path(args.targets)
        if not index_path.is_file():
            raise FileNotFoundError(f"Extracted spatial-mean index is missing: {index_path}")
        return index_path
    matches = []
    for path in sorted((data / "vae").rglob("index.json")):
        index = read_json(path)
        if (index.get("kind") == "visual_mean_maps"
                and index.get("vae_sha256") == checkpoint_sha
                and index.get("manifest_sha256") == manifest_sha):
            matches.append(path)
    if len(matches) == 1:
        print(f"Reusing f32 means: {matches[0]}", flush=True)
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Multiple f32 mean caches match; select one with --targets:\n"
                         + "\n".join(str(path) for path in matches))
    directory = checkpoint.parent / "alignment_mu_maps"
    print(f"Extracting deterministic f32 means: {directory}", flush=True)
    encode(["--views", str(args.views), "--vae-checkpoint", str(checkpoint),
            "--output-dir", str(directory), "--split", "all", "--device", args.device])
    return directory / "index.json"


def match_metadata(actual, expected, label):
    for key in ("view_id", "receiver_id", "camera_signature", "split", "image"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"{label} {expected['view_id']}: different or missing {key}")
    if (np.shape(actual.get("rf_origin")) != (3,)
            or not np.allclose(actual["rf_origin"], expected["rf_origin"], rtol=0, atol=1e-5)):
        raise ValueError(f"{label} {expected['view_id']}: different receiver position")


def view_lookup(views, expected_ids, label):
    ids = [v["view_id"] for v in views]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError(f"{label}: duplicate or mismatched view IDs; use the full matching cache")
    return {view_id: i for i, view_id in enumerate(ids)}


def pair_inputs(cache, index_path, metadata, manifest_sha, vae_sha, config, load):
    if cache.get("kind") != "rf_features" or cache.get("schema_version") != 1:
        raise ValueError("Expected a schema-1 RF feature cache")
    if cache.get("manifest_sha256") != manifest_sha:
        raise ValueError("RF cache and VAE training manifest differ")
    if not cache.get("feature_signature") or not cache.get("provenance"):
        raise ValueError("RF feature provenance is missing")
    x = finite(cache["features"], "RF features").float()
    fv = cache["views"]
    if x.ndim != 4 or len(x) != len(fv) or x.shape[1] < 2:
        raise ValueError("RF cache needs [N,C,H,W], with rendering mass in the last channel")
    if float(x[:, -1].max()) <= 1e-8 or (x[:, -1] < 0).any():
        raise ValueError("RF rendering mass is zero or negative; inspect the extracted RF cache")
    index = read_json(index_path)
    if index.get("kind") != "visual_mean_maps" or index.get("schema_version") != 1:
        raise ValueError("Expected spatial-mean index.json from encode_vae_map")
    if index.get("vae_sha256") != vae_sha or index.get("manifest_sha256") != manifest_sha:
        raise ValueError("Target cache uses a different VAE checkpoint or manifest")
    if tuple(index.get("latent_shape", [])) != tuple(config.latent_shape):
        raise ValueError("Target-cache latent shape differs from the f32 checkpoint")
    index_config = dict(index.get("vae_config", {}))
    index_config["channels"] = tuple(index_config.get("channels", []))
    if index_config != vars(config):
        raise ValueError("Target-cache VAE settings differ from the checkpoint")
    expected_ids = [v["view_id"] for v in metadata]
    fl = view_lookup(fv, expected_ids, "RF cache")
    tv = index["views"]
    tl = view_lookup(tv, expected_ids, "Target cache")
    ys, order = [], []
    digest = hashlib.sha256()
    digest.update(index_path.read_bytes())
    for view in metadata:
        fi, ti = fl[view["view_id"]], tl[view["view_id"]]
        match_metadata(fv[fi], view, "RF")
        match_metadata(tv[ti], view, "Target")
        image_sha = sha256(view["image"])
        if tv[ti].get("image_sha256") != image_sha:
            raise ValueError(f"Image changed since target extraction: {view['view_id']}")
        path = (index_path.parent / tv[ti]["latent_file"]).resolve()
        try:
            path.relative_to(index_path.parent.resolve())
        except ValueError:
            raise ValueError("Latent file lies outside the target-cache directory")
        target = load(path)
        if (target.get("kind") != "visual_mean_map" or target.get("schema_version") != 1
                or target.get("view_id") != view["view_id"]):
            raise ValueError(f"Wrong target record: {path}")
        mu = finite(target["mu"], "VAE mean").float()
        if tuple(mu.shape) != tuple(config.latent_shape):
            raise ValueError(f"Wrong latent shape in {path}: {tuple(mu.shape)}")
        digest.update(sha256(path).encode("ascii"))
        ys.append(mu)
        order.append(fi)
    # Keep the manifest order so its receiver-grouped splits remain valid.
    if order != list(range(len(order))):
        x = x[order]
    return x, torch.stack(ys), index, digest.hexdigest()


def statistics(x, y, train):
    # Channel statistics use train only; retain a per-location mean-map baseline.
    x_sum = torch.zeros((1, x.shape[1], 1, 1), dtype=torch.float64)
    x_squared = torch.zeros_like(x_sum)
    x_count = 0
    y_sum = torch.zeros((1, *y.shape[1:]), dtype=torch.float64)
    for start in range(0, len(train), 16):
        indices = train[start:start + 16]
        batch = x[indices].double()
        x_sum += batch.sum((0, 2, 3), keepdim=True)
        x_squared += batch.square().sum((0, 2, 3), keepdim=True)
        x_count += len(batch) * x.shape[2] * x.shape[3]
        y_sum += y[indices].double().sum(0, keepdim=True)
    x_mean = x_sum / x_count
    x_std = (x_squared / x_count - x_mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    y_mean = y_sum / len(train)
    residual_squared = torch.zeros((1, y.shape[1], 1, 1), dtype=torch.float64)
    for start in range(0, len(train), 16):
        residual = y[train[start:start + 16]].double() - y_mean
        residual_squared += residual.square().sum((0, 2, 3), keepdim=True)
    count = len(train) * y.shape[2] * y.shape[3]
    y_std = (residual_squared / count).sqrt().clamp_min(1e-6)
    return {"x_mean": x_mean.float(), "x_std": x_std.float(),
            "y_mean": y_mean.float(), "y_std": y_std.float()}


@torch.no_grad()
def predict(model, x, indices, stats, device, batch_size):
    model.eval()
    result = []
    gpu = {key: value.to(device) for key, value in stats.items()}
    for start in range(0, len(indices), batch_size):
        batch = x[indices[start:start + batch_size]].to(device)
        normalized = (batch - gpu["x_mean"]) / gpu["x_std"]
        mu = model(normalized) * gpu["y_std"] + gpu["y_mean"]
        result.append(finite(mu.cpu(), "Predicted mean"))
    return torch.cat(result)


def latent_metrics(prediction, target, mean):
    baseline = float((target - mean).square().mean())
    mse = float((prediction - target).square().mean())
    return {"latent_mse": mse, "latent_mae": float((prediction - target).abs().mean()),
            "mean_map_baseline_mse": baseline,
            "relative_mse_to_baseline": mse / baseline if baseline > 0 else None}


@torch.no_grad()
def decode_validation(vae, prediction, target, views, out, device, load_image):
    vae.to(device).eval().requires_grad_(False)
    width = 480
    height = round(width * vae.config.image_height / vae.config.image_width)
    rows = min(4, len(views))
    sheet = Image.new("L", (3 * width, rows * (height + 24)), color=255)
    draw = ImageDraw.Draw(sheet)
    total_pred, total_teacher = 0.0, 0.0
    native = out / "val_images"
    native.mkdir(exist_ok=False)
    for i, view in enumerate(views):
        image = load_image(view, vae.config).unsqueeze(0).to(device)
        teacher = finite(vae.decode(target[i:i + 1].to(device)), "Teacher reconstruction")
        decoded = finite(vae.decode(prediction[i:i + 1].to(device)), "RF reconstruction")
        total_teacher += float((teacher - image).square().mean())
        total_pred += float((decoded - image).square().mean())
        if i < rows:
            for j, (label, tensor) in enumerate(zip(
                    ("Input", "VAE target mu", "RF predicted mu"), (image, teacher, decoded))):
                pixels = tensor[0, 0].cpu().clamp(0, 1).numpy()
                tile = Image.fromarray((pixels * 255).round().astype(np.uint8))
                top = i * (height + 24)
                draw.text((j * width + 4, top + 4), f"{label} #{i}", fill=0)
                sheet.paste(tile.resize((width, height), Image.Resampling.LANCZOS),
                            (j * width, top + 24))
                tile.save(native / f"{i:02d}_{('input', 'target', 'prediction')[j]}.png")
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(views):
            print(f"Decoded validation {i + 1}/{len(views)}", flush=True)
    sheet.save(out / "best_val_preview.png")
    mse = total_pred / len(views)
    teacher_mse = total_teacher / len(views)
    return {"views": len(views), "rf_image_mse": mse,
            "rf_image_psnr_db": -10 * math.log10(max(mse, 1e-12)),
            "vae_target_image_mse": teacher_mse,
            "vae_target_image_psnr_db": -10 * math.log10(max(teacher_mse, 1e-12))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--data", default="/workspace/temp/alignment_dataset/random2000_4dir_v2000")
    parser.add_argument("--features", help="Existing schema-1 RF .pt cache; inferred only if unique")
    parser.add_argument("--targets", help="Existing f32 mean-map directory or index.json")
    parser.add_argument("--vae-checkpoint", help="Default: DATA/vae/map_f32_c4_run002/best.pt")
    parser.add_argument("--views", help="Default: views from the VAE run's arguments.json")
    parser.add_argument("--output-dir", help="Default: DATA/runs/align_f32_c4_run001")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if (min(args.epochs, args.batch_size, args.patience) < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("epochs, batch-size, patience, lr must be positive; weight-decay must be nonnegative")
    workspace = Path(args.workspace).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    checkpoint = Path(args.vae_checkpoint).expanduser().resolve() if args.vae_checkpoint else (
        data / "vae" / "map_f32_c4_run002" / "best.pt")
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        data / "runs" / "align_f32_c4_run001")
    if out.exists():
        raise FileExistsError(f"Choose a new --output-dir; this run already exists: {out}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"f32 VAE checkpoint is missing: {checkpoint}")
    sys.path.insert(0, str(workspace))
    base = "scripts.newrf_latent_alignment"
    common = importlib.import_module(base + ".common")
    cameras = importlib.import_module(base + ".views.cameras")
    maps = importlib.import_module(base + ".spatial_vae.map_utils")
    encode = importlib.import_module(base + ".spatial_vae.encode_vae_map").main
    load_image = importlib.import_module(base + ".vae.vae").load_image
    if not args.views:
        arguments = read_json(checkpoint.parent / "arguments.json")
        args.views = arguments["views"]
    args.views = str((workspace / Path(args.views).expanduser()).resolve())
    print(f"Loading frozen f32 VAE: {checkpoint}", flush=True)
    vae, saved = maps.load_map_checkpoint(checkpoint, "cpu")
    if vae.config.factor != 32 or tuple(vae.config.latent_shape) != (4, 25, 37):
        raise ValueError(f"This run requires f32_c4 with [4,25,37], got {vae.config.latent_shape}")
    print(f"Reading existing receiver-grouped splits: {args.views}", flush=True)
    document, manifest_sha, splits = maps.load_split_manifest(args.views)
    if saved["manifest_sha256"] != manifest_sha:
        raise ValueError("Manifest differs from the f32 VAE training manifest")
    metadata = cameras.cache_views(document)
    feature_path, cache = find_rf(data, args.features, common.load_tensor_file, metadata, manifest_sha)
    print(f"RF cache: {feature_path}; shape={tuple(cache['features'].shape)}", flush=True)
    checkpoint_sha = sha256(checkpoint)
    index_path = obtain_targets(args, checkpoint, checkpoint_sha, manifest_sha, data, encode)
    print("Checking paired IDs, cameras, splits, images, and f32 means...", flush=True)
    x, y, target_index, targets_sha = pair_inputs(
        cache, index_path, metadata, manifest_sha, checkpoint_sha, vae.config, common.load_tensor_file)
    stats = statistics(x, y, splits["train"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = {"in_channels": x.shape[1], "latent_shape": list(y.shape[1:])}
    model = RFToMeanMap(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(torch.tensor(splits["train"], dtype=torch.long)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    gpu = {key: value.to(args.device) for key, value in stats.items()}
    provenance = {"vae_sha256": checkpoint_sha, "vae_config": target_index["vae_config"],
                  "features_sha256": sha256(feature_path), "targets_sha256": targets_sha,
                  "manifest_sha256": manifest_sha, "feature_signature": cache["feature_signature"],
                  "feature_provenance": cache["provenance"], "script_sha256": sha256(__file__)}
    args.features, args.targets, args.vae_checkpoint = str(feature_path), str(index_path), str(checkpoint)
    args.output_dir = str(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "arguments.json", vars(args))
    write_json(out / "splits.json", {name: [metadata[i]["view_id"] for i in indices]
                                     for name, indices in splits.items()})
    write_json(out / "model_config.json", config)
    print(f"Learning RF {tuple(x.shape[1:])} -> mean map {tuple(y.shape[1:])}; "
          f"splits={ {name: len(indices) for name, indices in splits.items()} }", flush=True)
    print(f"Output: {out}", flush=True)
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        error, count = 0.0, 0
        for (indices,) in loader:
            batch, target = x[indices].to(args.device), y[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model((batch - gpu["x_mean"]) / gpu["x_std"])
            prediction = prediction * gpu["y_std"] + gpu["y_mean"]
            # Match deterministic mu in the original VAE coordinates.
            loss = F.mse_loss(prediction, target)
            finite(loss, "Training loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            finite(norm, "Gradient norm")
            optimizer.step()
            error += float(loss.detach()) * len(indices)
            count += len(indices)
        val = predict(model, x, splits["val"], stats, args.device, args.batch_size)
        metrics = latent_metrics(val, y[splits["val"]], stats["y_mean"])
        row = {"epoch": epoch, "train_mse": error / count, **metrics}
        history.append(row)
        if metrics["latent_mse"] < best:
            best, best_epoch, stale = metrics["latent_mse"], epoch, 0
            save_checkpoint(out / "best.pt", {"schema_version": 1, "kind": "rf_to_mean_map",
                "model_config": config, "model_state_dict": model.state_dict(),
                "statistics": stats, "epoch": epoch, "val_mse": best, "views": metadata,
                "splits": splits, **provenance})
        else:
            stale += 1
        write_json(out / "history.json", history)
        print(f"Epoch {epoch}: train_MSE={row['train_mse']:.7f}, "
              f"val_mu_MSE={metrics['latent_mse']:.7f}, "
              f"mean_baseline={metrics['mean_map_baseline_mse']:.7f}, best_epoch={best_epoch}",
              flush=True)
        if stale >= args.patience:
            print(f"Early stopping after {stale} epochs without improvement.", flush=True)
            break
    best_saved = common.load_tensor_file(out / "best.pt")
    model.load_state_dict(best_saved["model_state_dict"], strict=True)
    report, predictions = {"best_epoch": best_epoch, "splits": {}}, {}
    for name in ("train", "val"):
        indices = splits[name]
        pred = predict(model, x, indices, stats, args.device, args.batch_size)
        target = y[indices]
        report["splits"][name] = {"views": len(indices), **latent_metrics(pred, target, stats["y_mean"])}
        predictions[name] = {"pred_mu": pred, "target_mu": target,
                             "views": [metadata[i] for i in indices]}
    common.save_new(out / "predictions.pt", {"schema_version": 1, "kind": "mean_map_predictions",
                    "splits": predictions, "latent_shape": list(y.shape[1:]), **provenance})
    write_json(out / "metrics.json", report)
    val = predictions["val"]
    report["validation_images"] = decode_validation(
        vae, val["pred_mu"], val["target_mu"], val["views"], out, args.device, load_image)
    write_json(out / "metrics.json", report)
    print(f"Saved alignment checkpoint: {out / 'best.pt'}", flush=True)
    print(f"Validation RF image PSNR: {report['validation_images']['rf_image_psnr_db']:.2f} dB", flush=True)
    print(f"Validation comparison: {out / 'best_val_preview.png'}", flush=True)


if __name__ == "__main__":
    main()
