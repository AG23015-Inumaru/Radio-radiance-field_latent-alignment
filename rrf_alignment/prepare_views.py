"""Convert Blender/NeRF-style GS transforms to an explicit image/pose manifest."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image

from .cameras import load_views
from .common import fingerprint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transforms", required=True, help="Blender-style camera_to_world transform_matrix JSON")
    parser.add_argument("--rf-from-world", required=True, help="JSON containing an explicit 4x4 similarity matrix")
    parser.add_argument("--image-map", help="JSON: source file_path -> actual GS rendered image path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.transforms).expanduser().resolve()
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(output)
    transforms = json.loads(source.read_text())
    mapping_path = Path(args.image_map).expanduser().resolve() if args.image_map else None
    mapping = json.loads(mapping_path.read_text()) if mapping_path else None
    matrix = json.loads(Path(args.rf_from_world).expanduser().read_text())
    views = []
    for frame in transforms["frames"]:
        name = frame["file_path"]
        if mapping is not None:
            if name not in mapping:
                raise ValueError(f"Missing explicit GS image mapping: {name}")
            image_path = mapping_path.parent / mapping[name]
        else:
            image_path = source.parent / name
            if not image_path.is_file() and not image_path.suffix:
                image_path = image_path.with_suffix(".png")
        image_path = image_path.resolve()
        with Image.open(image_path) as image:
            width, height = image.size
        settings = {**transforms, **frame}
        if any(key in settings and settings[key] != size for key, size in (("w", width), ("h", height))):
            raise ValueError(f"{name}: camera metadata and image size differ")
        if "fl_x" in settings:
            fx = float(settings["fl_x"])
        else:
            angle = float(settings["camera_angle_x"])
            if not 0 < angle < math.pi:
                raise ValueError("camera_angle_x must be in radians, between 0 and pi")
            fx = 0.5 * width / math.tan(0.5 * angle)
        if "fl_y" in settings:
            fy = float(settings["fl_y"])
        elif "camera_angle_y" in settings:
            angle_y = float(settings["camera_angle_y"])
            if not 0 < angle_y < math.pi:
                raise ValueError("camera_angle_y must be in radians, between 0 and pi")
            fy = 0.5 * height / math.tan(0.5 * angle_y)
        else:
            fy = fx  # Blender/GS's square-pixel, horizontal-FOV camera model.
        pose = frame["transform_matrix"]
        position = [round(pose[i][3], 6) for i in range(3)]
        views.append({"view_id": frame.get("view_id", name),
                      "receiver_id": str(frame.get("receiver_id", "rx_" + fingerprint(position)[:16])),
                      "image": str(image_path), "camera_to_world": pose,
                      "intrinsics": {"width": width, "height": height, "fx": fx, "fy": fy,
                                     "cx": settings.get("cx", width / 2), "cy": settings.get("cy", height / 2)},
                      **({"split": frame["split"]} if "split" in frame else {})})
    document = {"schema_version": 1, "camera_convention": "opengl", "rf_from_world": matrix, "views": views}
    write_json(output, document)
    load_views(output)
    print(f"Wrote {len(views)} explicit image/pose pairs to {output}")


if __name__ == "__main__":
    main()
