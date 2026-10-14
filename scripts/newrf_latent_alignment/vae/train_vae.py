"""Train the grayscale VAE before caching means with encode.py.

Training: pixel-mean reconstruction MSE + beta * KL summed over latent dimensions.
Selection: validation reconstruction MSE from deterministic decode(mu).
The manifest must specify receiver-grouped train/val/test splits.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset

from ..alignment.alignment import split_views
from ..views.cameras import cache_views, load_views
from ..common import finite_tensor, load_tensor_file, write_json
from .vae import GrayVectorVAE, VAEConfig, load_image


class ViewImages(Dataset):
    def __init__(self, views, indices, config):
        self.views = [views[i] for i in indices]
        self.config = config

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        return load_image(self.views[index], self.config)


def loss_terms(reconstruction, image, mu, logvar):
    mse = (reconstruction - image).square().mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    return mse, kl


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mse_sum, kl_sum, count = 0.0, 0.0, 0
    for image in loader:
        image = image.to(device)
        reconstruction, mu, logvar = model(image, sample=False)
        mse, kl = loss_terms(reconstruction, image, mu, logvar)
        finite_tensor(mse, "validation reconstruction MSE")
        finite_tensor(kl, "validation KL")
        mse_sum += mse.item() * len(image)
        kl_sum += kl.item() * len(image)
        count += len(image)
    mse = mse_sum / count
    return {"val_mu_mse": mse, "val_kl": kl_sum / count,
            "val_mu_psnr_db": -10 * math.log10(max(mse, 1e-12))}


@torch.no_grad()
def save_preview(model, dataset, path, device):
    """Each row: grayscale input on the left, decode(mu) on the right."""
    model.eval()
    width = min(480, model.config.image_width)
    height = round(width * model.config.image_height / model.config.image_width)
    rows = min(4, len(dataset))
    sheet = Image.new("L", (2 * width, rows * (height + 20)), color=255)
    draw = ImageDraw.Draw(sheet)
    for i in range(rows):
        original = dataset[i]
        reconstructed, _, _ = model(original.unsqueeze(0).to(device), sample=False)
        for column, tensor in enumerate((original, reconstructed[0].cpu())):
            pixels = (tensor[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            tile = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
            sheet.paste(tile, (column * width, i * (height + 20) + 20))
        draw.text((4, i * (height + 20) + 3), f"Input #{i}", fill=0)
        draw.text((width + 4, i * (height + 20) + 3), "VAE decode(mu)", fill=0)
    sheet.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True)
    parser.add_argument("--vae-config", help="Architecture JSON; defaults to VAEConfig")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30, help="Epochs without improvement; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Resume last.pt in the same output directory")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 0:
        parser.error("epochs/batch-size must be positive; workers/patience must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.beta) or args.beta < 0:
        parser.error("lr must be finite and positive; beta must be finite and nonnegative")
    out = Path(args.output_dir).expanduser()
    if out.exists() and not args.resume:
        raise FileExistsError(f"Choose a new run directory or use --resume: {out}")
    saved = load_tensor_file(out / "last.pt") if args.resume else None
    settings = json.loads(Path(args.vae_config).expanduser().read_text()) if args.vae_config else (
        saved["config"] if saved else asdict(VAEConfig()))
    settings["channels"] = tuple(settings.get("channels", VAEConfig().channels))
    cfg = VAEConfig(**settings)
    if cfg.in_channels != 1:
        raise ValueError("This trainer uses grayscale input (in_channels=1)")
    if cfg.latent_dim < 1 or not cfg.channels or any(c < 1 or c % min(8, c) for c in cfg.channels):
        raise ValueError("Invalid latent dimension or GroupNorm channel configuration")

    document, manifest_sha = load_views(args.views)
    views = document["views"]
    if not all(v.get("split") in ("train", "val", "test") for v in views):
        raise ValueError("Specify train/val/test for every view; use the same split for VAE and alignment")
    print("Checking receiver groups and image paths...", flush=True)
    splits = split_views(cache_views(document))
    # Test images are neither opened nor used for training/model selection.
    for i in splits["train"] + splits["val"]:
        view = views[i]
        with Image.open(view["image"]) as image:
            expected = (view["intrinsics"]["width"], view["intrinsics"]["height"])
            if image.size != expected:
                raise ValueError(f'{view["view_id"]}: image size {image.size} != camera size {expected}')
            if not math.isclose(image.width / image.height, cfg.image_width / cfg.image_height, rel_tol=1e-6):
                raise ValueError("Images and VAE configuration must have the same aspect ratio")
    print(f'Splits: { {name: len(indices) for name, indices in splits.items()} }; device={args.device}',
          flush=True)

    training = {key: getattr(args, key) for key in ("batch_size", "lr", "beta", "seed")}
    if saved and (saved["manifest_sha256"] != manifest_sha or saved["config"] != asdict(cfg)
                  or saved["training"] != training):
        raise ValueError("Resume requires the same manifest, architecture, batch size, lr, beta and seed")
    if saved and args.epochs <= saved["epoch"]:
        raise ValueError("--epochs must exceed the epoch stored in last.pt")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = GrayVectorVAE(cfg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    datasets = {name: ViewImages(views, splits[name], cfg) for name in ("train", "val")}
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train",
                                num_workers=args.workers, generator=generator if name == "train" else None)
               for name, dataset in datasets.items()}
    history, best, best_epoch, start = [], float("inf"), 0, 1
    if saved:
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        history, best, best_epoch = saved["history"], saved["best_val_mu_mse"], saved["best_epoch"]
        start = saved["epoch"] + 1
        generator.set_state(saved["loader_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.cuda.is_available() and saved["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        write_json(out / "resume_arguments.json", vars(args))
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "arguments.json", vars(args))
        write_json(out / "config.json", asdict(cfg))
        write_json(out / "splits.json", {name: [views[i]["view_id"] for i in indices]
                                          for name, indices in splits.items()})

    for epoch in range(start, args.epochs + 1):
        model.train()
        mse_sum, kl_sum, count = 0.0, 0.0, 0
        for step, image in enumerate(loaders["train"], 1):
            image = image.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(image, sample=True)
            mse, kl = loss_terms(reconstruction, image, mu, logvar)
            loss = finite_tensor(mse + args.beta * kl, "VAE training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            mse_sum += mse.item() * len(image)
            kl_sum += kl.item() * len(image)
            count += len(image)
            if step == 1 or step % 50 == 0:
                print(f"Epoch {epoch} step {step}/{len(loaders['train'])}: "
                      f"MSE={mse.item():.6f} KL={kl.item():.6f}", flush=True)
        row = {"epoch": epoch, "train_mse": mse_sum / count, "train_kl": kl_sum / count,
               "train_loss": (mse_sum + args.beta * kl_sum) / count,
               **validate(model, loaders["val"], args.device)}
        history.append(row)
        improved = row["val_mu_mse"] < best
        if improved:
            best, best_epoch = row["val_mu_mse"], epoch
        payload = {"schema_version": 1, "kind": "gray_vae", "config": asdict(cfg),
                   "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                   "epoch": epoch, "best_epoch": best_epoch, "best_val_mu_mse": best,
                   "manifest_sha256": manifest_sha, "training": training,
                   "preprocessing": "grayscale_L,Lanczos,0_to_1"}
        if improved:
            save_checkpoint(out / "best.pt", payload)
            for name, dataset in datasets.items():
                save_preview(model, dataset, out / f"best_{name}_preview.png", args.device)
        save_checkpoint(out / "last.pt", {**payload, "optimizer_state": optimizer.state_dict(),
                        "history": history, "loader_rng_state": generator.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
        write_json(out / "history.json", history)
        print(f'Epoch {epoch}: train_MSE={row["train_mse"]:.6f}, '
              f'val_decode_mu_MSE={row["val_mu_mse"]:.6f}, '
              f'val_PSNR={row["val_mu_psnr_db"]:.2f} dB, best_epoch={best_epoch}', flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.", flush=True)
            break
    print(f"Saved VAE checkpoint: {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
