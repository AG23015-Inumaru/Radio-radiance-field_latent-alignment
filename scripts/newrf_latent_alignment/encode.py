"""Cache the frozen visual VAE's deterministic mean, never a sampled z."""
import argparse
from pathlib import Path

import torch

from .cameras import cache_views, load_views
from .common import finite_tensor, save_new, sha256_file
from .vae import load_frozen_vae, load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--vae-config", help="JSON required if the checkpoint has no full configuration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"Choose a new target cache: {output}")
    document, manifest_sha = load_views(args.views)
    vae = load_frozen_vae(args.vae_checkpoint, args.vae_config, args.device)
    means = []
    metadata = cache_views(document)
    with torch.no_grad():
        for index, view in enumerate(document["views"]):
            image = load_image(view, vae.config).unsqueeze(0).to(args.device)
            mu, _ = vae.encode(image)
            means.append(finite_tensor(mu[0], "VAE mean").cpu())
            metadata[index]["image_sha256"] = sha256_file(view["image"])
            if index == 0 or (index + 1) % 50 == 0:
                print(f'Encoded {index + 1}/{len(metadata)}: {view["view_id"]}', flush=True)
    save_new(output, {"schema_version": 1, "kind": "visual_means", "mu": torch.stack(means),
                      "views": metadata, "manifest_sha256": manifest_sha,
                      "vae_sha256": sha256_file(Path(args.vae_checkpoint).expanduser()),
                      "vae_config": vars(vae.config), "preprocessing": "grayscale_L,Lanczos,0_to_1"})
    print(f"Saved {len(means)} visual latent means to {output}")


if __name__ == "__main__":
    main()
