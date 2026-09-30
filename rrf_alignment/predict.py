"""Infer VAE means from additional camera-view RF feature maps."""
import argparse

import torch

from .alignment import RFMapEncoder
from .common import finite_tensor, load_tensor_file, save_new
from .train import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Alignment best.pt")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint, cache = load_tensor_file(args.checkpoint), load_tensor_file(args.features)
    if checkpoint["feature_signature"] != cache["feature_signature"]:
        raise ValueError("NeWRF checkpoint/source, model choice or feature sampling settings differ from training")
    x = finite_tensor(cache["features"], "RF features")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    model = RFMapEncoder(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    mu = predict(model, x, checkpoint["statistics"], args.device, args.batch_size)
    save_new(args.output, {"kind": "predicted_visual_means", "mu": mu, "views": cache["views"],
                           "vae_sha256": checkpoint["vae_sha256"], "vae_config": checkpoint["vae_config"]})
    print(f"Saved {len(mu)} predicted means to {args.output}")


if __name__ == "__main__":
    main()
