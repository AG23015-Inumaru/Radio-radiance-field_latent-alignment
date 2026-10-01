"""python -m scripts.newrf_latent_alignment.extract --help"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .cameras import cache_views, camera_rays, load_views
from .common import fingerprint, save_new, write_json
from .newrf import FrozenNeWRF


def main():
    parser = argparse.ArgumentParser(description="Extract camera-view NeWRF hidden-feature maps")
    parser.add_argument("--newrf-root", required=True)
    parser.add_argument("--config", required=True, help="YAML used to train this NeWRF checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--near", type=float, help="Default: training YAML sampling.near")
    parser.add_argument("--far", type=float, help="Default: training YAML sampling.far")
    parser.add_argument("--samples", type=int, help="Default: training YAML sampling.n_samples")
    parser.add_argument("--fine-samples", type=int, help="Default: training YAML sampling.n_samples_hierarchical")
    parser.add_argument("--inverse-depth", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--carrier-ghz", type=float, default=2.4)
    parser.add_argument("--ray-chunk", type=int, default=32)
    parser.add_argument("--point-chunk", type=int, default=8192)
    parser.add_argument("--limit", type=int, help="Use 1 for the first feature-map check")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"Choose a new output cache: {output}")
    document, manifest_sha = load_views(args.views)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        document["views"] = document["views"][:args.limit]
    field = FrozenNeWRF(args.newrf_root, args.config, args.checkpoint, args.model, args.device)
    for argument, key in {"near": "near", "far": "far", "samples": "n_samples",
                          "fine_samples": "n_samples_hierarchical", "inverse_depth": "inverse_depth"}.items():
        if getattr(args, argument) is None:
            if key not in field.sampling_config:
                raise ValueError(f"Missing sampling.{key} in training YAML; specify --{argument.replace('_', '-')}")
            setattr(args, argument, field.sampling_config[key])
    settings = {key: getattr(args, key) for key in ("width", "height", "near", "far", "samples",
                                                  "fine_samples", "inverse_depth", "carrier_ghz")}
    fields, diagnostics = [], []
    for index, view in enumerate(document["views"]):
        origins, directions = camera_rays(view, document, args.width, args.height)
        values, stats = field.extract(origins, directions, near=args.near, far=args.far,
                                      samples=args.samples, fine_samples=args.fine_samples,
                                      inverse_depth=args.inverse_depth, carrier_ghz=args.carrier_ghz,
                                      ray_chunk=args.ray_chunk, point_chunk=args.point_chunk)
        feature_map = values.T.reshape(-1, args.height, args.width).contiguous()
        fields.append(feature_map)
        diagnostics.append({"view_id": view["view_id"], **stats})
        print(f'[{index + 1}/{len(document["views"])}] {view["view_id"]}: '
              f'mass_mean={stats["mass_mean"]:.6g}, nonzero={stats["mass_nonzero_fraction"]:.3f}, '
              f'feature_norm={stats["pooled_feature_norm_mean"]:.6g}', flush=True)
        if index == 0:
            preview = output.with_suffix("").with_name(output.stem + "_preview")
            preview.mkdir(parents=True, exist_ok=True)
            mass = feature_map[-1].numpy()
            norm = feature_map[:-1].norm(dim=0).numpy()
            Image.fromarray((np.clip(mass, 0, 1) * 255).astype(np.uint8)).save(preview / "mass.png")
            scaled_norm = norm / max(float(norm.max()), 1e-12)
            Image.fromarray((scaled_norm * 255).astype(np.uint8)).save(preview / "feature_norm.png")
            write_json(preview / "preview.json", {"view_id": view["view_id"], "mass_display_range": [0, 1],
                                                  "feature_norm_display_range": [0, float(norm.max())]})
    provenance = {"newrf": field.provenance, "sampling": settings, "pooling": "sum(w*h),sum(w)"}
    save_new(output, {"schema_version": 1, "kind": "rf_features", "features": torch.stack(fields),
                      "views": cache_views(document), "manifest_sha256": manifest_sha,
                      "provenance": provenance, "feature_signature": fingerprint(provenance)})
    write_json(output.with_suffix(".diagnostics.json"), diagnostics)
    print(f"Saved {tuple(torch.stack(fields).shape)} to {output}")
    if max(row["mass_max"] for row in diagnostics) <= 1e-8:
        print("WARNING: all sampled rays have negligible mass. Inspect the checkpoint, source version, "
              "coordinate alignment and sampling before training an alignment model.")


if __name__ == "__main__":
    main()
