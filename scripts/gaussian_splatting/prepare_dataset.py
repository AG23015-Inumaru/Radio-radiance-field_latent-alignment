#.tif->.png
#dataset/RGB_gasussian-splatting/train
import argparse
import json
from pathlib import Path

from PIL import Image


def convert_split(root_in, root_out, json_name):
    json_in = root_in / json_name

    if not json_in.exists():
        raise FileNotFoundError(f"JSON not found: {json_in}")

    with open(json_in, "r") as f:
        data = json.load(f)

    new_data = dict(data)
    new_frames = []

    print(f"\nProcessing {json_name}")
    print(f"Frames: {len(data.get('frames', []))}")

    for i, frame in enumerate(data.get("frames", [])):
        src_rel = Path(frame["file_path"])
        src = root_in / src_rel

        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")

        # Example:
        # train/0001.tif -> train/0001.png
        dst_rel = src_rel.with_suffix(".png")
        dst = root_out / dst_rel

        dst.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src) as img:
            # Keep alpha if present, otherwise standard RGB
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            img.save(dst)

        # Official 3DGS adds ".png" itself.
        # train/0001.tif -> train/0001
        new_frame = dict(frame)
        new_frame["file_path"] = src_rel.with_suffix("").as_posix()
        new_frames.append(new_frame)

        if (i + 1) % 100 == 0 or (i + 1) == len(data["frames"]):
            print(f"  converted {i + 1}/{len(data['frames'])}")

    new_data["frames"] = new_frames

    json_out = root_out / json_name
    with open(json_out, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Saved: {json_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TIFF NeRF-style dataset for official Gaussian Splatting."
    )

    parser.add_argument(
        "--input",
        default="dataset/RGB_gaussian-splatting",
        help="Original dataset directory",
    )

    parser.add_argument(
        "--output",
        default="temp/gs_dataset",
        help="Output directory for Gaussian Splatting",
    )

    args = parser.parse_args()

    root_in = Path(args.input)
    root_out = Path(args.output)

    root_out.mkdir(parents=True, exist_ok=True)

    convert_split(
        root_in,
        root_out,
        "transforms_train.json",
    )

    convert_split(
        root_in,
        root_out,
        "transforms_test.json",
    )

    print("\nDataset preparation completed.")
    print(f"Input : {root_in}")
    print(f"Output: {root_out}")


if __name__ == "__main__":
    main()
