"""Explicit camera/image correspondence; distances are in NeWRF coordinates."""
import json
import math
from pathlib import Path

import numpy as np

from .common import fingerprint


def _matrix(value, label, similarity=False):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label}: expected a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{label}: last row must be [0, 0, 0, 1]")
    linear = matrix[:3, :3]
    gram = linear.T @ linear
    scale2 = np.trace(gram) / 3 if similarity else 1.0
    if scale2 <= 0 or not np.allclose(gram, scale2 * np.eye(3), atol=1e-5):
        raise ValueError(f"{label}: expected {'a similarity transform' if similarity else 'a rigid pose'}")
    if not similarity and np.linalg.det(linear) < 0:
        raise ValueError(f"{label}: camera rotation must have determinant +1")
    return matrix


def load_views(path):
    """All views require a stable ID, receiver group, pose, intrinsics and image."""
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("views.json needs schema_version: 1")
    convention = document.get("camera_convention")
    if convention not in ("opencv", "opengl"):
        raise ValueError("camera_convention must explicitly be opencv or opengl")
    transform = _matrix(document["rf_from_world"], "rf_from_world", similarity=True)
    views = document.get("views", [])
    if not views:
        raise ValueError("No views in manifest")
    ids = set()
    origins = {}
    for view in views:
        view_id, receiver = view.get("view_id"), view.get("receiver_id")
        if not isinstance(view_id, str) or not view_id or view_id in ids:
            raise ValueError("view_id must be a unique, nonempty string")
        if not isinstance(receiver, str) or not receiver:
            raise ValueError(f"{view_id}: receiver_id must be a nonempty string")
        ids.add(view_id)
        pose = _matrix(view["camera_to_world"], view_id)
        origin = (transform @ pose)[:3, 3]
        if receiver in origins and not np.allclose(origins[receiver], origin, rtol=0, atol=1e-5):
            raise ValueError(f"{receiver}: multiple camera centers; check receiver IDs and poses")
        origins[receiver] = origin
        k = view["intrinsics"]
        if any(not np.isfinite(k[key]) for key in ("width", "height", "fx", "fy", "cx", "cy")):
            raise ValueError(f"{view_id}: nonfinite intrinsics")
        if any(k[key] <= 0 for key in ("width", "height", "fx", "fy")):
            raise ValueError(f"{view_id}: image size and focal length must be positive")
        if any(int(k[key]) != k[key] for key in ("width", "height")):
            raise ValueError(f"{view_id}: image width/height must be integers")
        if not isinstance(view.get("image"), str) or not view["image"]:
            raise ValueError(f"{view_id}: an explicit teacher image path is required")
        if view.get("split") not in (None, "train", "val", "test"):
            raise ValueError(f"{view_id}: invalid split")
    # Hash the authored manifest before resolving paths; identity is not file order.
    manifest_sha = fingerprint(document)
    for view in views:
        image_path = Path(view["image"]).expanduser()
        view["image"] = str((path.parent / image_path).resolve())
    return document, manifest_sha


def camera_rays(view, document, width, height):
    """Return unit RF-frame rays through the centers of the downsampled pixels.

    Intrinsics use image-edge coordinates: the first full-resolution pixel center
    is (0.5, 0.5). OpenCV cameras look along +Z; OpenGL cameras along -Z.
    """
    if width < 1 or height < 1:
        raise ValueError("Feature-map size must be positive")
    k = view["intrinsics"]
    if not math.isclose(width / height, k["width"] / k["height"], rel_tol=1e-6):
        raise ValueError("RF map and teacher image must have the same aspect ratio")
    u = (np.arange(width) + 0.5) * k["width"] / width
    v = (np.arange(height) + 0.5) * k["height"] / height
    xx, yy = np.meshgrid((u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"])
    directions = np.stack((xx, yy, np.ones_like(xx)), -1).reshape(-1, 3)
    if document["camera_convention"] == "opengl":
        directions *= [1, -1, -1]
    pose = np.asarray(view["camera_to_world"], dtype=np.float64)
    transform = np.asarray(document["rf_from_world"], dtype=np.float64)
    directions = directions @ (transform[:3, :3] @ pose[:3, :3]).T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origin = (transform @ pose)[:3, 3]
    return np.broadcast_to(origin, directions.shape).copy().astype(np.float32), directions.astype(np.float32)


def cache_views(document):
    transform = np.asarray(document["rf_from_world"])
    return [{"view_id": v["view_id"], "receiver_id": v["receiver_id"],
             "image": v["image"], "split": v.get("split"),
             "rf_origin": (transform @ np.asarray(v["camera_to_world"]))[:3, 3].tolist(),
             "camera_signature": fingerprint({"convention": document["camera_convention"],
                                                "rf_from_world": document["rf_from_world"],
                                                "pose": v["camera_to_world"], "intrinsics": v["intrinsics"]}),
             "intrinsics": v["intrinsics"]} for v in document["views"]]
