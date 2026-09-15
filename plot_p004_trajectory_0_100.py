#!/usr/bin/env python3
"""Run MAC-VO only on TartanAir-v2 House/P004 [0,100) and export a paper-ready top-down trajectory plot.

The script deliberately does NOT run ReSplat or the incremental GraphDECO backend.
It reuses the exact ``MacvoPoseFrontend`` implementation/configuration used by the
online pipeline, loads the left-camera GT poses from TartanAir, performs rigid
SE(3) alignment (no scale correction) of the estimated trajectory to GT, reports
ATE RMSE, and saves PNG/PDF/SVG plus raw/aligned trajectory data.

This is intended for qualitative/teaser visualization.  The current online
pipeline has no separate loop-closure trajectory, so the plot contains only:
  - Ground Truth
  - MAC-VO
Do not add a synthetic "Loop" curve unless an actual loop-corrected pose stream
is available.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from async_pipeline.macvo_runtime import (
    MacvoPoseFrontend,
    MacvoRuntimeConfig,
    _pose7_xyzw_to_matrix,
    _tartan_from_cv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100, help="exclusive end frame")
    p.add_argument(
        "--odom_config",
        type=Path,
        default=Path("Config/Experiment/MACVO/MACVO_Performant.yaml"),
    )
    p.add_argument(
        "--data_config",
        type=Path,
        default=Path("Config/Sequence/TartanAirV2_House_easy_P004.yaml"),
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P004"),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/p004_trajectory_0_100"),
    )
    p.add_argument(
        "--mark_frames",
        type=int,
        nargs="*",
        default=[],
        help="optional dataset frame indices to mark on the aligned trajectory",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no_title", action="store_true")
    return p.parse_args()


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gt_relative(gt_pose_file: Path, start: int, end: int) -> dict[int, torch.Tensor]:
    rows = np.loadtxt(gt_pose_file, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] < 7:
        raise ValueError(f"invalid GT pose file shape: {rows.shape}")
    if start < 0 or end > rows.shape[0] or end <= start:
        raise IndexError(f"requested [{start},{end}) outside GT pose rows={rows.shape[0]}")

    abs_poses: dict[int, torch.Tensor] = {}
    for frame_index in range(start, end):
        pose = _pose7_xyzw_to_matrix(torch.from_numpy(rows[frame_index, :7].copy()))
        abs_poses[frame_index] = pose @ _tartan_from_cv(dtype=torch.float64)

    T0_inv = torch.linalg.inv(abs_poses[start])
    return {
        frame_index: (T0_inv @ pose).float()
        for frame_index, pose in abs_poses.items()
    }


def run_macvo(start: int, end: int, odom_config: Path, data_config: Path) -> dict[int, torch.Tensor]:
    frontend = MacvoPoseFrontend(
        MacvoRuntimeConfig(
            repo=Path(__file__).resolve().parent,
            odom_config=odom_config.resolve(),
            data_config=data_config.resolve(),
            start_index=start,
            end_index=end,
            left_subdir="image_lcam_front",
            right_subdir="image_rcam_front",
            left_pattern="{index:06d}_lcam_front.png",
            right_pattern="{index:06d}_rcam_front.png",
            preload=False,
            pose_commit_policy="one_frame_delayed",
            dedicated_cuda_stream=False,
            pose_source="macvo",
        )
    )

    estimates: dict[int, torch.Tensor] = {}
    frontend.initialize()
    for descriptor, frame, _stereo_input, _observation in frontend.iter_frames():
        for estimate in frontend.process(descriptor, frame):
            if estimate.valid:
                estimates[int(estimate.descriptor.frame_index)] = (
                    estimate.T_world_from_left.detach().cpu().double()
                )
    for estimate in frontend.terminate():
        if estimate.valid:
            estimates[int(estimate.descriptor.frame_index)] = (
                estimate.T_world_from_left.detach().cpu().double()
            )
    return estimates


def rigid_align_se3(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find R,t minimizing ||R*est+t-gt|| with scale fixed to 1."""
    if est_xyz.shape != gt_xyz.shape or est_xyz.ndim != 2 or est_xyz.shape[1] != 3:
        raise ValueError(f"bad trajectory shapes est={est_xyz.shape} gt={gt_xyz.shape}")
    if est_xyz.shape[0] < 3:
        raise ValueError("need at least 3 matched poses for SE(3) alignment")

    mu_e = est_xyz.mean(axis=0)
    mu_g = gt_xyz.mean(axis=0)
    X = est_xyz - mu_e
    Y = gt_xyz - mu_g
    H = X.T @ Y
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_g - R @ mu_e
    aligned = (R @ est_xyz.T).T + t
    return aligned, R, t


def save_pose_json(path: Path, poses: dict[int, torch.Tensor]) -> None:
    payload = {
        str(frame): pose.detach().double().cpu().tolist()
        for frame, pose in sorted(poses.items())
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv(path: Path, frame_ids: list[int], gt_xyz: np.ndarray, est_xyz: np.ndarray, aligned_xyz: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_index",
            "gt_x", "gt_y", "gt_z",
            "est_x", "est_y", "est_z",
            "aligned_x", "aligned_y", "aligned_z",
        ])
        for i, frame in enumerate(frame_ids):
            w.writerow([frame, *gt_xyz[i].tolist(), *est_xyz[i].tolist(), *aligned_xyz[i].tolist()])


def main() -> None:
    args = parse_args()
    seed_all(0)

    root = Path(__file__).resolve().parent
    odom_config = (root / args.odom_config).resolve() if not args.odom_config.is_absolute() else args.odom_config.resolve()
    data_config = (root / args.data_config).resolve() if not args.data_config.is_absolute() else args.data_config.resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_pose_file = data_root / "pose_lcam_front.txt"
    if not gt_pose_file.is_file():
        raise FileNotFoundError(gt_pose_file)

    print(f"[trajectory] running MAC-VO only on P004 [{args.start},{args.end})", flush=True)
    est_poses = run_macvo(args.start, args.end, odom_config, data_config)
    gt_poses = load_gt_relative(gt_pose_file, args.start, args.end)

    matched = [
        frame for frame in range(args.start, args.end)
        if frame in est_poses and frame in gt_poses
    ]
    if len(matched) < 3:
        raise RuntimeError(f"too few matched valid poses: {len(matched)}")

    gt_xyz = np.stack([gt_poses[f][:3, 3].numpy() for f in matched], axis=0)
    est_xyz = np.stack([est_poses[f][:3, 3].numpy() for f in matched], axis=0)
    aligned_xyz, R_align, t_align = rigid_align_se3(est_xyz, gt_xyz)

    errors = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    ate_median = float(np.median(errors))

    metrics: dict[str, Any] = {
        "sequence": "TartanAir-v2 House/Data_easy/P004",
        "frame_range": [args.start, args.end],
        "num_requested_frames": args.end - args.start,
        "num_valid_matched_poses": len(matched),
        "alignment": "SE(3) rigid alignment, scale fixed to 1",
        "ate_rmse_m": ate_rmse,
        "ate_mean_m": ate_mean,
        "ate_median_m": ate_median,
        "R_est_to_gt": R_align.tolist(),
        "t_est_to_gt_m": t_align.tolist(),
        "top_down_plane": "TartanAir world X-Y",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_pose_json(output_dir / "gt_relative_twc.json", gt_poses)
    save_pose_json(output_dir / "macvo_relative_twc.json", est_poses)
    save_csv(output_dir / "trajectory_xyz.csv", matched, gt_xyz, est_xyz, aligned_xyz)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], linewidth=2.4, label="Ground Truth")
    ax.plot(aligned_xyz[:, 0], aligned_xyz[:, 1], linewidth=2.1, label="MAC-VO")

    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], s=42, marker="o", zorder=5)
    ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], s=55, marker="X", zorder=5)

    matched_to_pos = {frame: i for i, frame in enumerate(matched)}
    for frame in args.mark_frames:
        if frame not in matched_to_pos:
            print(f"[trajectory] warning: mark frame {frame} has no valid matched pose", flush=True)
            continue
        i = matched_to_pos[frame]
        ax.scatter(aligned_xyz[i, 0], aligned_xyz[i, 1], s=90, marker="s", facecolors="none", linewidths=2.0, zorder=6)
        ax.annotate(str(frame), (aligned_xyz[i, 0], aligned_xyz[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    if not args.no_title:
        ax.set_title(f"P004 [0,{args.end})  |  ATE RMSE = {ate_rmse:.3f} m")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()

    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"trajectory_topdown.{suffix}"
        fig.savefig(path, dpi=args.dpi if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    print("\n===== P004 trajectory =====", flush=True)
    print(f"frames requested : [{args.start},{args.end})", flush=True)
    print(f"valid poses      : {len(matched)}", flush=True)
    print(f"ATE RMSE (SE3)   : {ate_rmse:.6f} m", flush=True)
    print(f"output           : {output_dir}", flush=True)
    print(f"figure           : {output_dir / 'trajectory_topdown.pdf'}", flush=True)


if __name__ == "__main__":
    main()
