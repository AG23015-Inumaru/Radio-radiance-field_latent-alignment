"""Render NIST Lobby RGB teacher images at viewpoints defined in views.csv.

Run this script *inside Blender*, with `--` separating Blender and script options.
A scene (.blend) must already be loaded via `blender --background scene.blend`.

Camera convention for CSV yaw/pitch (degrees):
  yaw=0   -> +X
  yaw=90  -> +Y
  pitch=0 -> horizontal
Blender cameras look along local -Z with local +Y as up.
"""
#blender rendaling RGB figure
import argparse
import math
import sys
from pathlib import Path

import bpy
import pandas as pd
from mathutils import Matrix, Vector


def parse_args():
    args_for_script = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views_csv", required=True, help="CSV generated for alignment views")
    parser.add_argument("--output_root", required=True, help="Root directory for teacher PNGs")
    parser.add_argument("--camera_name", default="AlignmentCamera")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fov_x_deg", type=float, default=90.0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--max_bounces", type=int, default=6)
    parser.add_argument("--device", default="CUDA", help="Cycles device type, e.g. CUDA or CPU")
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--use_sky", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Render only the first N rows")
    return parser.parse_args(args_for_script)


def configure_render(scene, args):
    scene.render.engine = args.engine
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True

    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.max_bounces = args.max_bounces
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.use_denoising = True
        if args.device.upper() == "CPU":
            scene.cycles.device = "CPU"
        else:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = args.device.upper()
            prefs.get_devices()
            usable = [d for d in prefs.devices if d.type == args.device.upper()]
            if not usable:
                raise RuntimeError(
                    f"No available Cycles {args.device} devices. "
                    "Check the Blender build, drivers and GPU configuration."
                )
            for device in prefs.devices:
                device.use = device.type == args.device.upper()
            scene.cycles.device = "GPU"

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.exposure = args.exposure

    if args.use_sky:
        world = scene.world
        if world is None:
            world = bpy.data.worlds.new("AlignmentSky")
            scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        tex = nodes.new("ShaderNodeTexSky")
        tex.sky_type = "HOSEK_WILKIE"
        background = nodes.new("ShaderNodeBackground")
        output = nodes.new("ShaderNodeOutputWorld")
        links.new(tex.outputs["Color"], background.inputs["Color"])
        links.new(background.outputs["Background"], output.inputs["Surface"])


def get_camera(scene, camera_name):
    obj = bpy.data.objects.get(camera_name)
    if obj is None:
        camera_data = bpy.data.cameras.new(camera_name)
        obj = bpy.data.objects.new(camera_name, camera_data)
        scene.collection.objects.link(obj)
    if obj.type != "CAMERA":
        raise TypeError(f"'{camera_name}' exists but is not a camera")
    scene.camera = obj
    obj.data.type = "PERSP"
    obj.data.sensor_fit = "HORIZONTAL"
    obj.data.clip_start = 0.01
    obj.data.clip_end = 1000.0
    return obj


def set_camera(camera, row, fov_x_deg):
    x, y, z = (float(row[key]) for key in ("x", "y", "z"))
    yaw = math.radians(float(row.get("yaw_deg", 0.0)))
    pitch = math.radians(float(row.get("pitch_deg", 0.0)))
    roll = math.radians(float(row.get("roll_deg", 0.0)))
    direction = Vector((
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ))
    if direction.length < 1e-10:
        raise ValueError("Camera direction cannot have zero length")
    camera.location = (x, y, z)
    base_rotation = direction.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    roll_rotation = Matrix.Rotation(roll, 4, direction)
    camera.rotation_euler = (roll_rotation @ base_rotation).to_euler()

    if not 0.0 < fov_x_deg < 180.0:
        raise ValueError(f"Invalid horizontal FOV: {fov_x_deg}")
    camera.data.lens = camera.data.sensor_width / (2 * math.tan(math.radians(fov_x_deg) / 2))


def output_path(row, output_root):
    rel = row.get("image_relpath", "")
    if pd.notna(rel) and str(rel).strip():
        relative = Path(str(rel).strip().replace("\\", "/"))
    else:
        split = str(row.get("split", "train"))
        view_id = int(row.get("view_id", 0))
        rx_id = int(row.get("RxID", 0))
        yaw = round(float(row.get("yaw_deg", 0))) % 360
        relative = Path(split) / f"view_{view_id:06d}_rx{rx_id:04d}_yaw{yaw:03d}.png"
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"image_relpath must be relative to output_root: {relative}")
    path = output_root / relative
    return path.with_suffix(".png")


def main():
    args = parse_args()
    csv_path = Path(args.views_csv).expanduser()
    output_root = Path(args.output_root).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if args.width <= 0 or args.height <= 0 or args.samples <= 0:
        raise ValueError("width, height, and samples must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    rows = pd.read_csv(csv_path)
    required_columns = {"x", "y", "z", "yaw_deg"}
    missing = required_columns.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    if args.limit is not None:
        rows = rows.head(args.limit)
    scene = bpy.context.scene
    configure_render(scene, args)
    camera = get_camera(scene, args.camera_name)

    rendered = 0
    skipped = 0
    for row_number, (_, row) in enumerate(rows.iterrows(), start=1):
        width = int(row.get("rgb_w", args.width))
        height = int(row.get("rgb_h", args.height))
        fov = float(row.get("fov_x_deg", args.fov_x_deg))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions at CSV row {row_number}")
        path = output_path(row, output_root)
        if path.is_file() and not args.overwrite:
            skipped += 1
            print(f"[{row_number}/{len(rows)}] SKIP {path}", flush=True)
            continue

        set_camera(camera, row, fov)
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(path)
        print(f"[{row_number}/{len(rows)}] RENDER {path}", flush=True)
        bpy.ops.render.render(write_still=True)
        rendered += 1

    print(f"Finished: rendered={rendered}, skipped={skipped}, total={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
