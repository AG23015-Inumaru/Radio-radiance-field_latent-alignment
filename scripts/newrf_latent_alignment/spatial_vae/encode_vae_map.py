"""Export deterministic means as individual spatial tensors, without flattening."""
import argparse
from pathlib import Path

import torch

from ..views.cameras import cache_views
from ..common import finite_tensor, save_new, sha256_file, write_json
from .map_utils import config_dict, load_map_checkpoint, load_split_manifest
from ..vae.vae import load_image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    out = Path(args.output_dir).expanduser()
    if out.exists():
        raise FileExistsError(f"Choose a new latent-cache directory: {out}")
    model, saved = load_map_checkpoint(args.vae_checkpoint, args.device)
    document, digest, splits = load_split_manifest(args.views)
    if digest != saved["manifest_sha256"]:
        raise ValueError("Manifest differs from the checkpoint's training manifest")
    indices = list(range(len(document["views"]))) if args.split == "all" else splits[args.split]
    if not indices:
        raise ValueError("Selected split is empty")
    metadata = cache_views(document)
    checkpoint_sha = sha256_file(Path(args.vae_checkpoint).expanduser())
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    with torch.no_grad():
        for progress, index in enumerate(indices, 1):
            view = document["views"][index]
            image = load_image(view, model.config).unsqueeze(0).to(args.device)
            mu, _ = model.encode(image)
            filename = f"{index:06d}.pt"
            save_new(out / filename, {"kind": "visual_mean_map", "schema_version": 1,
                     "mu": finite_tensor(mu[0], "spatial latent mean").cpu(), "view_id": view["view_id"]})
            entries.append({**metadata[index], "latent_file": filename, "image_sha256": sha256_file(view["image"])})
            if progress == 1 or progress % 50 == 0 or progress == len(indices):
                print(f"Encoded {progress}/{len(indices)}; latent={tuple(mu.shape[1:])}", flush=True)
    write_json(out / "index.json", {"schema_version": 1, "kind": "visual_mean_maps",
               "latent_shape": list(model.config.latent_shape), "vae_config": config_dict(model.config),
               "manifest_sha256": digest, "vae_sha256": checkpoint_sha,
               "preprocessing": saved["preprocessing"], "views": entries})
    print(f"Saved {len(entries)} spatial latent means: {out}", flush=True)


if __name__ == "__main__":
    main()
