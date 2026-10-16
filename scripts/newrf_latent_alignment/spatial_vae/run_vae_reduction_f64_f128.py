#!/usr/bin/env python3
"""Prepare spatial latent reductions, then train the selected runs sequentially.

Run inside the original Docker checkout. The baseline's data manifest and
training settings are reused. Existing checkpoints are never resumed or
overwritten by this launcher. --prepare-only changes code/configs without
starting training; --summary-only reads completed runs without changing code.
Default experiments: f64_c4 and f128_c4. No tests are executed.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys


EXPERIMENT_DEPTHS = {"f16": 4, "f32": 5, "f64": 6, "f128": 7}


PATCHES = (
    (
        '        if self.image_height % self.factor or self.image_width % self.factor:\n'
        '            raise ValueError(f"Image height and width must be divisible by {self.factor}")',
        '        if self.image_height < self.factor or self.image_width < self.factor:\n'
        '            raise ValueError(f"Image height and width must be at least {self.factor}")',
    ),
    (
        '        cfg = self.config\n',
        '        cfg = self.config\n'
        '        spatial_sizes = [(cfg.image_height, cfg.image_width)]\n'
        '        for _ in cfg.channels:\n'
        '            height, width = spatial_sizes[-1]\n'
        '            spatial_sizes.append((height // 2, width // 2))\n'
        '        self.decoder_target_sizes = tuple(reversed(spatial_sizes[:-1]))\n',
    ),
    (
        '        for stage in self.decoder_stages:\n'
        '            features = stage(F.interpolate(features, scale_factor=2, mode="nearest"))',
        '        for stage, size in zip(self.decoder_stages, self.decoder_target_sizes):\n'
        '            features = stage(F.interpolate(features, size=size, mode="nearest"))',
    ),
)


def replace_known(source, before, after, label):
    if after in source:
        return source
    if source.count(before) != 1:
        raise ValueError(f"Unrecognized {label}; no files changed. Share the current vae_map.py before applying a different patch.")
    return source.replace(before, after, 1)


def detect_package(workspace):
    parent = workspace / "scripts" / "newrf_latent_alignment"
    found = [parent / name for name in ("spatial_vae", "sptialVAE", "spatialVAE")
             if (parent / name / "vae_map.py").is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected one spatial VAE package, found: {found}")
    return found[0]


def latent_shape(config):
    factor = 2 ** len(config["channels"])
    return [config["latent_channels"], config["image_height"] // factor, config["image_width"] // factor]


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def prepare(package, baseline_config, run_names):
    model_path = package / "vae_map.py"
    source = model_path.read_text(encoding="utf-8-sig")
    updated = source
    for before, after in PATCHES:
        updated = replace_known(updated, before, after, "spatial VAE architecture")
    ast.parse(updated, filename=str(model_path))
    pending = [(model_path, updated.encode("utf-8"))]
    if len(baseline_config["channels"]) != 3:
        raise ValueError("The reference must be the completed f8 model with three downsampling stages")
    if baseline_config["latent_channels"] != 4:
        raise ValueError("These compression experiments require a four-channel f8 baseline")
    configs = {}
    for name in run_names:
        depth = EXPERIMENT_DEPTHS[name]
        config = dict(baseline_config)
        config["channels"] = list(baseline_config["channels"]) + [256] * (depth - 3)
        shape = latent_shape(config)
        if min(shape) < 1:
            raise ValueError(f"Image is too small for {name}: {shape}")
        path = package / "configs" / f"vae.map_{name}.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8-sig")) != config:
            raise FileExistsError(f"Configuration already differs; no files changed: {path}")
        pending.append((path, json_bytes(config)))
        configs[name] = (path, config)
    # Back up existing modified files before writing; all backups stay in this package.
    changed = [(path, data) for path, data in pending if not path.exists() or path.read_bytes().replace(b"\r\n", b"\n") != data]
    for path, _ in changed:
        if path.exists():
            original = path.read_bytes()
            digest = hashlib.sha256(original).hexdigest()[:12]
            backup = package / "backups" / f"{path.name}.{digest}.bak"
            if backup.exists() and backup.read_bytes() != original:
                raise FileExistsError(f"Backup differs; no files changed: {backup}")
    for path, data in changed:
        if path.exists():
            original = path.read_bytes()
            digest = hashlib.sha256(original).hexdigest()[:12]
            backup = package / "backups" / f"{path.name}.{digest}.bak"
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                with backup.open("xb") as stream:
                    stream.write(original)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"Prepared: {path}", flush=True)
    return configs


def summarize(data, tag, compare_tag, run_names):
    rows = []
    for name in ("f8", *EXPERIMENT_DEPTHS):
        if name == "f8":
            candidates = ["run001"]
        elif name in run_names:
            candidates = [tag]
        else:
            candidates = list(dict.fromkeys((tag, compare_tag)))
        selected = None
        for run_tag in candidates:
            directory = data / "vae" / f"map_{name}_c4_{run_tag}"
            if not (directory / "history.json").is_file() or not (directory / "config.json").is_file():
                continue
            history = json.loads((directory / "history.json").read_text(encoding="utf-8-sig"))
            if history:
                selected = directory, run_tag, history
                break
        if selected is None:
            continue
        directory, run_tag, history = selected
        best = min(history, key=lambda row: row["val_mu_mse"])
        config = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
        shape = latent_shape(config)
        rows.append({"experiment": f"{name}_c4", "latent_shape": shape, "latent_elements": math.prod(shape),
                     "best_epoch": best["epoch"], "val_mu_mse": best["val_mu_mse"],
                     "val_sample_mse": best.get("val_sample_mse"),
                     "val_mu_psnr_db": best["val_mu_psnr_db"], "run_tag": run_tag,
                     "run_dir": str(directory)})
    path = data / "vae" / f"latent_reduction_summary_{tag}.json"
    path.write_bytes(json_bytes(rows))
    print("\nexperiment  latent       elements  best_epoch  val_mu_MSE  val_PSNR   run_tag", flush=True)
    for row in rows:
        shape = "x".join(map(str, row["latent_shape"]))
        print(f"{row['experiment']:10}  {shape:11}  {row['latent_elements']:8}  {row['best_epoch']:10}  "
              f"{row['val_mu_mse']:.6f}    {row['val_mu_psnr_db']:.2f} dB   {row['run_tag']}", flush=True)
    print(f"Summary: {path}", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--data", default="/workspace/temp/alignment_dataset/random2000_4dir_v2000")
    parser.add_argument("--run-tag", default="run003", help="Output tag for the new experiments")
    parser.add_argument("--compare-tag", default="run002", help="Tag of the previous f16/f32 comparison")
    parser.add_argument("--runs", nargs="+", choices=tuple(EXPERIMENT_DEPTHS), default=["f64", "f128"])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", tag) for tag in (args.run_tag, args.compare_tag)):
        parser.error("run-tag and compare-tag must contain only letters, digits, underscores and hyphens")
    if len(args.runs) != len(set(args.runs)):
        parser.error("Do not repeat an experiment in --runs")
    workspace = Path(args.workspace).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    baseline = data / "vae" / "map_f8_c4_run001"
    if args.summary_only:
        summarize(data, args.run_tag, args.compare_tag, args.runs)
        return
    package = detect_package(workspace)
    config = json.loads((baseline / "config.json").read_text(encoding="utf-8-sig"))
    training = json.loads((baseline / "arguments.json").read_text(encoding="utf-8-sig"))
    manifest = Path(training["views"]).expanduser()
    if not manifest.is_absolute():
        manifest = workspace / manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Baseline image manifest is unavailable: {manifest}")
    keys = ("epochs", "batch_size", "lr", "beta", "kl_reduction", "warmup_epochs", "recon_loss", "mode",
            "patience", "seed", "workers", "log_every", "device")
    missing = [key for key in keys if key not in training]
    if missing:
        raise ValueError(f"Baseline arguments are missing: {missing}")
    output_dirs = {name: data / "vae" / f"map_{name}_c4_{args.run_tag}" for name in args.runs}
    if not args.prepare_only:
        for directory in output_dirs.values():
            if directory.exists():
                raise FileExistsError(f"Run already exists: {directory}. Choose an unused --run-tag, such as run004, or use --runs to select only the experiments still needed.")
    configs = prepare(package, config, args.runs)
    module = "scripts.newrf_latent_alignment." + package.name
    for name in args.runs:
        shape = latent_shape(configs[name][1])
        print(f"{name}_c4: {shape}, {math.prod(shape):,} latent elements", flush=True)
    if args.prepare_only:
        print("Preparation complete. No training was started.", flush=True)
        return
    for name in args.runs:
        command = [sys.executable, "-u", "-m", module + ".train_vae_map", "--views", str(manifest),
                   "--vae-config", str(configs[name][0]), "--output-dir", str(output_dirs[name])]
        for key in keys:
            command.extend(["--" + key.replace("_", "-"), str(training[key])])
        print(f"\nStarting {name}_c4: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=workspace, check=True)
        summarize(data, args.run_tag, args.compare_tag, args.runs)


if __name__ == "__main__":
    main()
