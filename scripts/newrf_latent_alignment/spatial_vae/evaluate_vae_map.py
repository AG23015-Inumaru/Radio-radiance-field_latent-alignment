"""Evaluate a trained map VAE on an explicitly selected manifest split."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..common import write_json
from .map_utils import MapViewImages, load_map_checkpoint, load_split_manifest, save_preview, validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    out = Path(args.output_dir).expanduser()
    if out.exists():
        raise FileExistsError(f"Choose a new evaluation directory: {out}")
    model, saved = load_map_checkpoint(args.vae_checkpoint, args.device)
    document, digest, splits = load_split_manifest(args.views)
    if digest != saved["manifest_sha256"]:
        raise ValueError("Manifest differs from the checkpoint's training manifest")
    dataset = MapViewImages(document["views"], splits[args.split], model.config)
    mode = saved["training"]["mode"]
    metrics = validate(model, DataLoader(dataset, batch_size=args.batch_size), args.device, mode)
    metrics = {key.replace("val_", f"{args.split}_", 1): value for key, value in metrics.items()}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "metrics.json", {"split": args.split, "count": len(dataset), "mode": mode, **metrics})
    save_preview(model, dataset, out / "preview.png", args.device, mode, native_dir=out / "examples")
    print(metrics, flush=True)
    print(f"Saved metrics and reconstruction comparisons: {out}", flush=True)


if __name__ == "__main__":
    main()
