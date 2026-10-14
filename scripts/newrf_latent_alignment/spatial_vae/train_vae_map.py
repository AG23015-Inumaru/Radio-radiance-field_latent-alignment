"""Train the spatial VAE using the same views/preprocessing as train_vae.py."""
import argparse
import math
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader

from ..common import finite_tensor, load_tensor_file, write_json
from .map_utils import (MapViewImages, config_dict, load_split_manifest, read_config,
                        save_checkpoint, save_preview, validate)
from .vae_map import GrayMapVAE, loss_terms


def beta_at(epoch, step, steps, beta, warmup_epochs, mode):
    if mode == "ae":
        return 0.0
    progress = (epoch - 1 + step / steps) / warmup_epochs if warmup_epochs else 1.0
    return beta * min(1.0, progress)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True)
    parser.add_argument("--vae-config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--kl-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--recon-loss", choices=("mse", "l1"), default="mse")
    parser.add_argument("--mode", choices=("vae", "ae"), default="vae",
                        help="ae disables BOTH sampling and KL; beta=0 alone still samples in vae mode")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.log_every) < 1 or min(args.workers, args.patience, args.warmup_epochs) < 0:
        parser.error("epochs/batch-size/log-every must be positive; workers/patience/warmup must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.beta) or args.beta < 0:
        parser.error("lr must be finite and positive; beta must be finite and nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    out = Path(args.output_dir).expanduser()
    if out.exists() and not args.resume:
        raise FileExistsError(f"Choose a new run directory or use --resume: {out}")
    saved = load_tensor_file(out / "last.pt") if args.resume else None
    if saved and (saved.get("kind") != "gray_map_vae" or saved.get("schema_version") != 1):
        raise ValueError("Cannot resume a vector-VAE checkpoint as a map VAE")
    cfg = read_config(args.vae_config, saved["config"] if saved else None)
    document, manifest_sha, splits = load_split_manifest(args.views)
    views = document["views"]
    print("Checking receiver groups and train/val image paths...", flush=True)
    # Test images remain unopened during training and model selection.
    for index in splits["train"] + splits["val"]:
        view = views[index]
        with Image.open(view["image"]) as image:
            expected = (view["intrinsics"]["width"], view["intrinsics"]["height"])
            if image.size != expected:
                raise ValueError(f'{view["view_id"]}: image size {image.size} != camera size {expected}')
            if not math.isclose(image.width / image.height, cfg.image_width / cfg.image_height, rel_tol=1e-6):
                raise ValueError("Image and model aspect ratios must match")
    elements = math.prod(cfg.latent_shape)
    print(f'Splits: { {name: len(indices) for name, indices in splits.items()} }; device={device}', flush=True)
    print(f"Input: [B, 1, {cfg.image_height}, {cfg.image_width}]; latent: [B, {', '.join(map(str, cfg.latent_shape))}] "
          f"({elements:,} elements/image); f={cfg.factor}", flush=True)
    sum_coefficient = args.beta / elements if args.kl_reduction == "mean" else args.beta
    print(f"mode={args.mode}; recon={args.recon_loss}; KL={args.kl_reduction}; target beta={args.beta:g}; "
          f"equivalent coefficient on KL_sum={sum_coefficient if args.mode == 'vae' else 0:g}", flush=True)
    keys = ("batch_size", "lr", "beta", "kl_reduction", "warmup_epochs", "recon_loss", "mode", "seed", "workers", "device")
    training = {key: getattr(args, key) for key in keys}
    if saved and (saved["manifest_sha256"] != manifest_sha or saved["config"] != config_dict(cfg)
                  or saved["training"] != training):
        raise ValueError("Resume requires the same manifest, architecture and training settings")
    if saved and args.epochs <= saved["epoch"]:
        raise ValueError("--epochs must exceed the saved epoch")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = GrayMapVAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    datasets = {name: MapViewImages(views, splits[name], cfg) for name in ("train", "val")}
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train",
                               num_workers=args.workers, pin_memory=device.type == "cuda",
                               generator=generator if name == "train" else torch.Generator().manual_seed(args.seed + 1))
               for name, dataset in datasets.items()}
    history, best, best_epoch, start = [], float("inf"), 0, 1
    if saved:
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        history, best, best_epoch = saved["history"], saved["best_val_mu_mse"], saved["best_epoch"]
        start = saved["epoch"] + 1
        generator.set_state(saved["loader_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda" and saved["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        write_json(out / "resume_arguments.json", vars(args))
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "arguments.json", vars(args))
        write_json(out / "config.json", config_dict(cfg))
        write_json(out / "splits.json", {name: [views[i]["view_id"] for i in indices] for name, indices in splits.items()})
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
    for epoch in range(start, args.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in ("mse", "reconstruction", "kl_sum", "kl_mean", "weighted_kl", "loss")}
        count = 0
        for step, image in enumerate(loaders["train"], 1):
            image = image.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(image, sample=args.mode == "vae")
            rec, kl_sum, kl_mean = loss_terms(reconstruction, image, mu, logvar, args.recon_loss)
            weight = beta_at(epoch, step, len(loaders["train"]), args.beta, args.warmup_epochs, args.mode)
            chosen_kl = kl_mean if args.kl_reduction == "mean" else kl_sum
            weighted_kl = weight * chosen_kl if args.mode == "vae" else rec.new_zeros(())
            loss = finite_tensor(rec + weighted_kl, "map-VAE training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            values = {"mse": (reconstruction - image).square().mean(), "reconstruction": rec,
                      "kl_sum": kl_sum if args.mode == "vae" else rec.new_zeros(()),
                      "kl_mean": kl_mean if args.mode == "vae" else rec.new_zeros(()),
                      "weighted_kl": weighted_kl, "loss": loss}
            for key, value in values.items():
                totals[key] += finite_tensor(value, key).item() * len(image)
            count += len(image)
            if step == 1 or step % args.log_every == 0 or step == len(loaders["train"]):
                print(f"Epoch {epoch} step {step}/{len(loaders['train'])}: MSE={values['mse'].item():.6f} "
                      f"KL_mean={values['kl_mean'].item():.6f} KL_sum={values['kl_sum'].item():.2f} "
                      f"weighted_KL={weighted_kl.item():.6f} beta={weight:g}", flush=True)
        row = {"epoch": epoch, **{f"train_{key}": value / count for key, value in totals.items()},
               **validate(model, loaders["val"], device, args.mode)}
        history.append(row)
        if row["val_mu_mse"] < best:
            best, best_epoch = row["val_mu_mse"], epoch
            improved = True
        else:
            improved = False
        payload = {"schema_version": 1, "kind": "gray_map_vae", "config": config_dict(cfg),
                   "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                   "epoch": epoch, "best_epoch": best_epoch, "best_val_mu_mse": best,
                   "manifest_sha256": manifest_sha, "training": training,
                   "latent_shape": list(cfg.latent_shape), "preprocessing": "grayscale_L,Lanczos,0_to_1"}
        if improved:
            save_checkpoint(out / "best.pt", payload)
            save_preview(model, datasets["train"], out / "best_train_preview.png", device, args.mode)
            save_preview(model, datasets["val"], out / "best_val_preview.png", device, args.mode,
                         native_dir=out / "best_val_examples")
        save_checkpoint(out / "last.pt", {**payload, "optimizer_state": optimizer.state_dict(),
                        "history": history, "loader_rng_state": generator.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        write_json(out / "history.json", history)
        print(f"Epoch {epoch}: train_MSE={row['train_mse']:.6f}, val_decode_mu_MSE={row['val_mu_mse']:.6f}, "
              f"val_sample_MSE={row['val_sample_mse']:.6f}, val_PSNR={row['val_mu_psnr_db']:.2f} dB, "
              f"best_epoch={best_epoch}", flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.", flush=True)
            break
    print(f"Saved map VAE checkpoint: {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
