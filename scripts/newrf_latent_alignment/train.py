"""Train only the RF map CNN against cached means from a frozen VAE."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .alignment import RFMapEncoder, pair_caches, split_views, training_statistics
from .common import finite_tensor, load_tensor_file, save_new, sha256_file, write_json


@torch.no_grad()
def predict(model, x, stats, device, batch_size):
    if not len(x):
        return torch.empty((0, stats["y_mean"].shape[1]))
    predictions = []
    model.eval()
    for batch in x.split(batch_size):
        normalized = ((batch - stats["x_mean"]) / stats["x_std"]).to(device)
        mu = model(normalized).cpu() * stats["y_std"] + stats["y_mean"]
        predictions.append(finite_tensor(mu, "predicted mean"))
    return torch.cat(predictions)


def metrics(pred, target, mean):
    return {"latent_mse": float(nn.functional.mse_loss(pred, target)),
            "latent_mae": float(nn.functional.l1_loss(pred, target)),
            "mean_baseline_mse": float(nn.functional.mse_loss(mean.expand_as(target), target))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    features, targets = load_tensor_file(args.features), load_tensor_file(args.targets)
    x, y, views = pair_caches(features, targets)
    splits = split_views(views, args.seed, args.val_fraction, args.test_fraction)
    stats = training_statistics(x, y, splits["train"])
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "splits.json", {k: [views[i]["view_id"] for i in indices] for k, indices in splits.items()})
    write_json(out / "arguments.json", vars(args))
    config = {"in_channels": x.shape[1], "latent_dim": y.shape[1]}
    model = RFMapEncoder(**config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_train = (x[splits["train"]] - stats["x_mean"]) / stats["x_std"]
    loader = DataLoader(TensorDataset(x_train, y[splits["train"]]), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    y_mean, y_std = stats["y_mean"].to(args.device), stats["y_std"].to(args.device)
    features_sha = sha256_file(Path(args.features).expanduser())
    targets_sha = sha256_file(Path(args.targets).expanduser())
    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        squared_error, count = 0.0, 0
        for batch, target in loader:
            batch, target = batch.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            mu = model(batch) * y_std + y_mean
            # Plain MSE in the original VAE mean coordinates. No sampled latent.
            loss = nn.functional.mse_loss(mu, target)
            finite_tensor(loss, "training loss")
            loss.backward()
            optimizer.step()
            squared_error += float(loss.detach()) * len(batch)
            count += len(batch)
        val_pred = predict(model, x[splits["val"]], stats, args.device, args.batch_size)
        val_metrics = metrics(val_pred, y[splits["val"]], stats["y_mean"])
        row = {"epoch": epoch, "train_mse": squared_error / count, **val_metrics}
        history.append(row)
        print(f'Epoch {epoch}: train={row["train_mse"]:.6f}, val={row["latent_mse"]:.6f}, '
              f'mean_baseline={row["mean_baseline_mse"]:.6f}', flush=True)
        if row["latent_mse"] < best:
            best = row["latent_mse"]
            torch.save({"schema_version": 1, "model_state_dict": model.state_dict(), "model_config": config,
                        "statistics": stats, "epoch": epoch, "val_mse": best,
                        "feature_signature": features["feature_signature"],
                        "feature_provenance": features["provenance"],
                        "vae_sha256": targets["vae_sha256"], "vae_config": targets["vae_config"],
                        "features_sha256": features_sha,
                        "targets_sha256": targets_sha,
                        "splits": splits, "views": views}, out / "best.pt")
        write_json(out / "history.json", history)
    saved = load_tensor_file(out / "best.pt")
    model.load_state_dict(saved["model_state_dict"])
    report = {"best_epoch": saved["epoch"], "splits": {}}
    predictions = {}
    for name, indices in splits.items():
        if not indices:
            continue
        pred = predict(model, x[indices], stats, args.device, args.batch_size)
        report["splits"][name] = {"views": len(indices), **metrics(pred, y[indices], stats["y_mean"])}
        predictions[name] = {"pred_mu": pred, "target_mu": y[indices], "views": [views[i] for i in indices]}
    save_new(out / "predictions.pt", {"kind": "latent_predictions", "splits": predictions,
                                       "vae_sha256": targets["vae_sha256"], "vae_config": targets["vae_config"]})
    write_json(out / "metrics.json", report)
    print(f"Saved the best validation checkpoint and predictions to {out}")


if __name__ == "__main__":
    main()
