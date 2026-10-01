"""A small CNN and strict pairing/grouped splitting for the first experiment."""
import random

import numpy as np
import torch
from torch import nn

from .common import finite_tensor


class RFMapEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        # Retain coarse image-plane position instead of global average pooling.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
            nn.Linear(64 * 4 * 6, 128), nn.SiLU(), nn.Linear(128, latent_dim))

    def forward(self, x):
        return self.net(x)


def pair_caches(features, targets):
    if features.get("kind") != "rf_features" or targets.get("kind") != "visual_means":
        raise ValueError("Expected RF feature and visual-mean caches")
    if features.get("schema_version") != 1 or targets.get("schema_version") != 1:
        raise ValueError("Unsupported cache schema")
    x = finite_tensor(features["features"], "RF features").float()
    y = finite_tensor(targets["mu"], "visual means").float()
    fv, tv = features["views"], targets["views"]
    if x.ndim != 4 or y.ndim != 2 or len(x) != len(fv) or len(y) != len(tv):
        raise ValueError("Cache tensors must match their view metadata")
    fids, tids = [v["view_id"] for v in fv], [v["view_id"] for v in tv]
    if len(set(fids)) != len(fids) or len(set(tids)) != len(tids):
        raise ValueError("Duplicate view IDs in a cache")
    if set(fids) != set(tids):
        raise ValueError("Feature and target view IDs differ; re-extract both for the same views")
    lookup = {view_id: i for i, view_id in enumerate(tids)}
    order = [lookup[view_id] for view_id in fids]
    for f, ti in zip(fv, order):
        target = tv[ti]
        for key in ("receiver_id", "camera_signature", "image", "split"):
            if f[key] != target[key]:
                raise ValueError(f'{f["view_id"]}: incompatible {key} between the two caches')
    if x.shape[1] < 2 or float(x[:, -1].max()) <= 1e-8:
        raise ValueError("RF rendering mass is zero; diagnose the features before alignment training")
    if (x[:, -1] < 0).any():
        raise ValueError("The last feature channel must be nonnegative rendering mass")
    return x, y[order], [tv[i] for i in order]


def split_views(views, seed=0, val_fraction=0.2, test_fraction=0.1):
    """All views at the same Rx center stay together, even with aliased Rx IDs."""
    if not 0 < val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid validation/test fractions")
    parents = list(range(len(views)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, view in enumerate(views):
        for j in range(i):
            same_id = view["receiver_id"] == views[j]["receiver_id"]
            same_position = np.allclose(view["rf_origin"], views[j]["rf_origin"], rtol=0, atol=1e-5)
            if same_id or same_position:
                parents[root(i)] = root(j)
    groups = {}
    for i in range(len(views)):
        groups.setdefault(root(i), []).append(i)
    explicit = [v.get("split") for v in views]
    if any(explicit):
        if not all(explicit):
            raise ValueError("Specify split for every view or omit it for every view")
        for indices in groups.values():
            if len({explicit[i] for i in indices}) != 1:
                raise ValueError("The same receiver position appears in multiple splits")
        result = {name: [i for i, split in enumerate(explicit) if split == name]
                  for name in ("train", "val", "test")}
    else:
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        n_val = max(1, round(len(keys) * val_fraction))
        n_test = max(1, round(len(keys) * test_fraction)) if test_fraction else 0
        if len(keys) - n_val - n_test < 1:
            raise ValueError("Too few receiver groups; use more views or --test-fraction 0")
        assignments = {"val": keys[:n_val], "test": keys[n_val:n_val + n_test],
                       "train": keys[n_val + n_test:]}
        result = {name: [i for key in selected for i in groups[key]] for name, selected in assignments.items()}
    if not result["train"] or not result["val"]:
        raise ValueError("Both training and validation splits must be nonempty")
    return result


def training_statistics(x, y, train_indices):
    # These statistics never see validation/test rows.
    xt, yt = x[train_indices], y[train_indices]
    return {"x_mean": xt.mean((0, 2, 3), keepdim=True),
            "x_std": xt.std((0, 2, 3), correction=0, keepdim=True).clamp_min(1e-6),
            "y_mean": yt.mean(0, keepdim=True),
            "y_std": yt.std(0, correction=0, keepdim=True).clamp_min(1e-6)}
