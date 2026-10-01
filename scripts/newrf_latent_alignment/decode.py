"""Decode held-out predictions and compare teacher, VAE reconstruction and RF reconstruction."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .common import finite_tensor, load_tensor_file, sha256_file, write_json
from .vae import load_frozen_vae, load_image


def psnr(a, b):
    mse = float((a - b).square().mean())
    return float(-10 * np.log10(max(mse, 1e-12)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--vae-config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-images", type=int, default=8, help="Metrics use the entire selected split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cache = load_tensor_file(args.predictions)
    if cache["vae_sha256"] != sha256_file(Path(args.vae_checkpoint).expanduser()):
        raise ValueError("This is not the VAE checkpoint used to make the training targets")
    inference = cache.get("kind") == "predicted_visual_means"
    if not inference and args.split not in cache["splits"]:
        raise ValueError(f"No {args.split} predictions in this run")
    data = {"views": cache["views"], "pred_mu": cache["mu"]} if inference else cache["splits"][args.split]
    vae = load_frozen_vae(args.vae_checkpoint, args.vae_config, args.device)
    if vars(vae.config) != cache["vae_config"]:
        raise ValueError("VAE configuration differs from the target cache")
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.no_grad():
        for i, view in enumerate(data["views"]):
            if inference:
                predicted = finite_tensor(vae.decode(data["pred_mu"][i:i + 1].to(args.device))[0].cpu(), "RF image")
                name = f"{i:05d}.png"
                Image.fromarray((predicted[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(out / name)
                rows.append({"view_id": view["view_id"], "image": name})
                continue
            if sha256_file(view["image"]) != view["image_sha256"]:
                raise ValueError(f'Teacher image changed since encoding: {view["image"]}')
            teacher = load_image(view, vae.config)
            oracle = finite_tensor(vae.decode(data["target_mu"][i:i + 1].to(args.device))[0].cpu(), "VAE image")
            predicted = finite_tensor(vae.decode(data["pred_mu"][i:i + 1].to(args.device))[0].cpu(), "RF image")
            row = {"view_id": view["view_id"], "vae_mu_psnr": psnr(oracle, teacher),
                   "rf_mu_psnr": psnr(predicted, teacher)}
            if i < args.max_images:
                # Left: teacher grayscale, middle: decode(teacher mu), right: decode(RF mu).
                panel = torch.cat((teacher, oracle, predicted), dim=-1)[0].clamp(0, 1).numpy()
                row["panel"] = f"{i:05d}.png"
                Image.fromarray((panel * 255).round().astype(np.uint8)).save(out / row["panel"])
            rows.append(row)
    if inference:
        write_json(out / "images.json", {"views": rows})
        print(f"Decoded {len(rows)} RF-only predictions to {out}")
        return
    write_json(out / "metrics.json", {"split": args.split, "views": rows,
               "mean_vae_mu_psnr": float(np.mean([r["vae_mu_psnr"] for r in rows])),
               "mean_rf_mu_psnr": float(np.mean([r["rf_mu_psnr"] for r in rows])),
               "panel_order": ["teacher_grayscale", "decode_teacher_mu", "decode_rf_mu"]})
    print(f"Decoded {len(rows)} views; metrics and sample panels: {out}")


if __name__ == "__main__":
    main()
