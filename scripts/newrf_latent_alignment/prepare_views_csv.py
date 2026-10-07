"""Convert the Blender teacher renderer's views.csv without opening any images.

Matches set_camera_pose() in render_teacher_images_blender.py: yaw 0 -> +X,
yaw 90 -> +Y, pitch up -> +Z, local -Z forward and +Y up. The renderer must
use square pixels, HORIZONTAL sensor fit and zero camera shift.
"""
#before->make_alignment_view_csvs.py
#.csv->.json convert
#next->prepare_views.py
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath


def _number(row, key, default=None):
    value = row.get(key, "")
    if value is None or value.strip() == "":
        if default is None:
            raise ValueError(f"Missing {key}")
        value = default
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _positive_integer(row, key):
    value = _number(row, key)
    if value < 1 or int(value) != value:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def blender_camera_to_world(x, y, z, yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """Equivalent to direction.to_track_quat('-Z', 'Y'), then world-axis roll.

    Vertical views need the renderer's actual matrix because its tracking
    convention has a singularity there; do not silently choose a roll for them.
    """
    if not -90.0 < pitch_deg < 90.0:
        raise ValueError("Require -90 < pitch_deg < 90; use Blender matrices for vertical views")
    yaw, pitch, roll = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    sy, cy, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    forward = [cp * cy, cp * sy, sp]
    right = [sy, -cy, 0.0]
    up = [-sp * cy, -sp * sy, cp]
    cr, sr = math.cos(roll), math.sin(roll)
    rolled_right = [cr * r - sr * u for r, u in zip(right, up)]
    rolled_up = [sr * r + cr * u for r, u in zip(right, up)]
    return [[rolled_right[i], rolled_up[i], -forward[i], position]
            for i, position in enumerate((x, y, z))] + [[0.0, 0.0, 0.0, 1.0]]


def build_manifest(views_csv, image_root, output, rf_from_world, rf_width=48, rf_height=32):
    """Preserve CSV row order, IDs and splits; paths are relative to output."""
    source = Path(views_csv).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if rf_width < 1 or rf_height < 1:
        raise ValueError("RF dimensions must be positive")
    views, ids, image_names, receivers = [], set(), set(), {}
    required = {"view_id", "RxID", "x", "y", "z", "split", "yaw_deg",
                "fov_x_deg", "rgb_w", "rgb_h", "rf_w", "rf_h", "image_relpath"}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            try:
                view_id, receiver = row["view_id"].strip(), row["RxID"].strip()
                if not view_id or view_id in ids or not receiver:
                    raise ValueError("view_id must be unique; view_id and RxID must be nonempty")
                split = row["split"].strip()
                if split not in ("train", "val", "test"):
                    raise ValueError("split must be train, val or test")
                position = [_number(row, key) for key in ("x", "y", "z")]
                if receiver in receivers:
                    previous_position, previous_split = receivers[receiver]
                    if split != previous_split or any(abs(a - b) > 1e-5 for a, b in zip(position, previous_position)):
                        raise ValueError(f"RxID {receiver} has multiple positions or splits")
                receivers[receiver] = (position, split)
                pose = blender_camera_to_world(
                    *position, _number(row, "yaw_deg"), _number(row, "pitch_deg", 0.0),
                    _number(row, "roll_deg", 0.0))
                if any(row.get(key, "").strip() for key in ("dir_x", "dir_y", "dir_z")):
                    direction = [_number(row, key) for key in ("dir_x", "dir_y", "dir_z")]
                    if any(abs(direction[i] + pose[i][2]) > 1e-5 for i in range(3)):
                        raise ValueError("dir_x/y/z disagree with the renderer's yaw/pitch direction")
                width, height = (_positive_integer(row, key) for key in ("rgb_w", "rgb_h"))
                if (_positive_integer(row, "rf_w"), _positive_integer(row, "rf_h")) != (rf_width, rf_height):
                    raise ValueError(f"CSV RF size differs from --rf-width {rf_width} --rf-height {rf_height}")
                if width * rf_height != height * rf_width:
                    raise ValueError("RGB image and RF map aspect ratios differ")
                fov_x = _number(row, "fov_x_deg")
                if not 0.0 < fov_x < 180.0:
                    raise ValueError("Require 0 < fov_x_deg < 180")
                focal = width / (2.0 * math.tan(math.radians(fov_x) / 2.0))
                fov_y = math.degrees(2.0 * math.atan(height / (2.0 * focal)))
                if row.get("fov_y_deg", "").strip() and not math.isclose(
                        _number(row, "fov_y_deg"), fov_y, abs_tol=1e-4):
                    raise ValueError("fov_y_deg disagrees with horizontal FOV and square pixels")
                name = row["image_relpath"].strip().replace("\\", "/")
                relative = PurePosixPath(name)
                if not name or relative.is_absolute() or ".." in relative.parts or ":" in name:
                    raise ValueError("image_relpath must be a relative image path under --image-root")
                # The supplied renderer forces PNG regardless of the CSV suffix.
                relative = relative.with_suffix(".png")
                if relative in image_names:
                    raise ValueError(f"Multiple views refer to image {relative}")
                image_path = image_root.joinpath(*relative.parts)
                views.append({
                    "view_id": view_id, "receiver_id": receiver, "split": split,
                    "image": Path(os.path.relpath(image_path, output.parent)).as_posix(),
                    "camera_to_world": pose,
                    "intrinsics": {"width": width, "height": height, "fx": focal, "fy": focal,
                                   "cx": width / 2.0, "cy": height / 2.0},
                })
                ids.add(view_id)
                image_names.add(relative)
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"CSV line {line}: {error}") from error
    if not views:
        raise ValueError("CSV has no views")
    return {"schema_version": 1, "camera_convention": "opengl",
            "rf_from_world": rf_from_world,
            "rf_map": {"width": rf_width, "height": rf_height},
            "source": {"format": "render_teacher_images_blender.views.csv",
                       "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                       "pixel_aspect": [1, 1], "sensor_fit": "HORIZONTAL", "camera_shift": [0, 0]},
            "views": views}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views-csv", required=True)
    parser.add_argument("--image-root", required=True, help="Planned Linux root containing train/val/test PNGs")
    parser.add_argument("--output", required=True)
    frame = parser.add_mutually_exclusive_group(required=True)
    frame.add_argument("--same-world", action="store_true", help="Explicitly use identical Blender/NeWRF coordinates and units")
    frame.add_argument("--rf-from-world", help="JSON containing a 4x4 Blender-world to NeWRF similarity matrix")
    parser.add_argument("--rf-width", type=int, default=48)
    parser.add_argument("--rf-height", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"Choose a new manifest filename: {output}")
    matrix = ([[float(i == j) for j in range(4)] for i in range(4)] if args.same_world
              else json.loads(Path(args.rf_from_world).expanduser().read_text(encoding="utf-8")))
    document = build_manifest(args.views_csv, args.image_root, output, matrix, args.rf_width, args.rf_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    views = document["views"]
    first, k = views[0], views[0]["intrinsics"]
    print(f"Saved {len(views)} views / {len({v['receiver_id'] for v in views})} receiver positions: {output}")
    print(f"Splits: {dict(Counter(v['split'] for v in views))}")
