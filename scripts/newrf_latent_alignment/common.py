"""Small, explicit cache and checkpoint utilities."""
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_tensor_file(path):
    return torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_new(path, value):
    """Do not silently reuse or overwrite an earlier feature/target cache."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value, stream)


def finite_tensor(value, label):
    if not torch.is_tensor(value) or not torch.isfinite(value).all():
        raise ValueError(f"{label} must be a finite tensor")
    return value
