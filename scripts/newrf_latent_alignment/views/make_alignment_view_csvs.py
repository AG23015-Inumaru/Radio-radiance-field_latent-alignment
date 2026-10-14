#!/usr/bin/env python3
#RX position ,angle -> .csv
#500position 4 direction
#output-> views.csv
#next->prepare_views_csv.py
import os
import math
import argparse
import numpy as np
import pandas as pd

YAW_CHOICES = [0, 90, 180, 270]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_points(mapping_csv):
    df = pd.read_csv(mapping_csv)

    required = ["RxID", "x", "y", "z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[["RxID", "x", "y", "z"]].copy()
    out["RxID"] = out["RxID"].astype(int)
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)
    out["z"] = out["z"].astype(float)
    return out

def make_point_split(points_df, seed=42):
    """
    Match the NeWRF fixed_val=True split.

    Random2000 dataset order:
        RxID 1-1600    : NeWRF train
        RxID 1601-2000 : NeWRF validation

    For latent alignment, divide the NeWRF validation part into:
        RxID 1601-1800 : alignment val
        RxID 1801-2000 : alignment test
    """
    points = points_df.copy().sort_values("RxID").reset_index(drop=True)

    n = len(points)
    assert n == 2000, f"Expected 2000 points, got {n}"

    expected = np.arange(1, 2001)
    actual = points["RxID"].to_numpy(dtype=int)

    if not np.array_equal(actual, expected):
        raise ValueError("RxID is not consecutive 1..2000")

    points["split"] = "train"
    points.loc[
        (points["RxID"] >= 1601) & (points["RxID"] <= 1800),
        "split"
    ] = "val"
    points.loc[
        (points["RxID"] >= 1801) & (points["RxID"] <= 2000),
        "split"
    ] = "test"

    return points

def add_common_view_columns(df, rgb_w=1200, rgb_h=800, rf_w=48, rf_h=32, fov_x_deg=90.0):
    df = df.copy()

    fov_y_deg = math.degrees(
        2.0 * math.atan((rf_h / rf_w) * math.tan(math.radians(fov_x_deg / 2.0)))
    )

    df["pitch_deg"] = 0.0
    df["roll_deg"] = 0.0
    df["fov_x_deg"] = float(fov_x_deg)
    df["fov_y_deg"] = float(fov_y_deg)
    df["rgb_w"] = int(rgb_w)
    df["rgb_h"] = int(rgb_h)
    df["rf_w"] = int(rf_w)
    df["rf_h"] = int(rf_h)

    rad = np.deg2rad(df["yaw_deg"].to_numpy(dtype=float))
    df["dir_x"] = np.cos(rad)
    df["dir_y"] = np.sin(rad)
    df["dir_z"] = 0.0

    return df

def create_onedir_dataset(points_split_df, out_csv, seed=42):
    rng = np.random.default_rng(seed)

    views = points_split_df.copy()
    views["yaw_deg"] = rng.choice(YAW_CHOICES, size=len(views), replace=True)

    views = add_common_view_columns(views)
    views = views.reset_index(drop=True)
    views["dataset_name"] = "random2000_1dir_v2000"
    views["view_id"] = np.arange(len(views), dtype=int)

    views["image_relpath"] = views.apply(
        lambda r: f"{r['split']}/view_{int(r['view_id']):06d}_rx{int(r['RxID']):04d}_yaw{int(r['yaw_deg']):03d}.png",
        axis=1
    )
    views["feature_relpath"] = views.apply(
        lambda r: f"{r['split']}/view_{int(r['view_id']):06d}_rx{int(r['RxID']):04d}_yaw{int(r['yaw_deg']):03d}.npy",
        axis=1
    )

    views.to_csv(out_csv, index=False)
    return views

def sample_500_points(points_split_df, seed=42):
    rng = np.random.default_rng(seed)

    want = {"train": 400, "val": 50, "test": 50}
    rows = []

    for split_name, n_take in want.items():
        sub = points_split_df[points_split_df["split"] == split_name].copy()
        idx = rng.choice(len(sub), size=n_take, replace=False)
        rows.append(sub.iloc[idx])

    sampled = pd.concat(rows, axis=0).reset_index(drop=True)
    sampled = sampled.sort_values(["split", "RxID"]).reset_index(drop=True)
    return sampled

def create_fourdir_dataset(points_split_df, out_csv, selected_points_csv, seed=42):
    sampled_points = sample_500_points(points_split_df, seed=seed)
    sampled_points.to_csv(selected_points_csv, index=False)

    rows = []
    for _, r in sampled_points.iterrows():
        for yaw in YAW_CHOICES:
            rows.append({
                "RxID": int(r["RxID"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z": float(r["z"]),
                "split": r["split"],
                "yaw_deg": float(yaw),
            })

    views = pd.DataFrame(rows)
    views = add_common_view_columns(views)
    views = views.reset_index(drop=True)
    views["dataset_name"] = "random2000_4dir_v2000"
    views["view_id"] = np.arange(len(views), dtype=int)

    views["image_relpath"] = views.apply(
        lambda r: f"{r['split']}/view_{int(r['view_id']):06d}_rx{int(r['RxID']):04d}_yaw{int(r['yaw_deg']):03d}.png",
        axis=1
    )
    views["feature_relpath"] = views.apply(
        lambda r: f"{r['split']}/view_{int(r['view_id']):06d}_rx{int(r['RxID']):04d}_yaw{int(r['yaw_deg']):03d}.npy",
        axis=1
    )

    views.to_csv(out_csv, index=False)
    return views, sampled_points

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping_csv",
        type=str,
        default=str(os.path.expanduser("~/sionna/sionna-0.19.1/nist_newrf_2_4GHz_random3d_fps2000_mapping.csv")),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(os.path.expanduser("~/Radio-radiance-field_latent-alignment/temp/alignment_dataset")),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.out_root)

    points_df = load_points(args.mapping_csv)
    points_split_df = make_point_split(points_df, seed=args.seed)

    point_split_csv = os.path.join(args.out_root, "random2000_point_split.csv")
    points_split_df.to_csv(point_split_csv, index=False)

    onedir_root = os.path.join(args.out_root, "random2000_1dir_v2000")
    ensure_dir(onedir_root)
    onedir_csv = os.path.join(onedir_root, "views.csv")
    onedir_views = create_onedir_dataset(points_split_df, onedir_csv, seed=args.seed)

    fourdir_root = os.path.join(args.out_root, "random2000_4dir_v2000")
    ensure_dir(fourdir_root)
    fourdir_csv = os.path.join(fourdir_root, "views.csv")
    selected_points_csv = os.path.join(fourdir_root, "selected_points.csv")
    fourdir_views, sampled_points = create_fourdir_dataset(
        points_split_df, fourdir_csv, selected_points_csv, seed=args.seed
    )

    print("============================================================")
    print("Done")
    print("============================================================")
    print("Point split CSV : ", point_split_csv)
    print("1dir views CSV  : ", onedir_csv, f"({len(onedir_views)} views)")
    print("4dir views CSV  : ", fourdir_csv, f"({len(fourdir_views)} views)")
    print("4dir point list : ", selected_points_csv, f"({len(sampled_points)} points)")
    print()

    print("1dir split counts")
    print(onedir_views["split"].value_counts().sort_index())
    print()

    print("4dir split counts")
    print(fourdir_views["split"].value_counts().sort_index())

if __name__ == "__main__":
    main()
