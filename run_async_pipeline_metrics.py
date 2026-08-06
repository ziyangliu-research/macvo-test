#!/usr/bin/env python3
"""Run the stable async pipeline and emit combined pose/render/runtime metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import run_async_pipeline as base
from async_pipeline.geometry import rotation_matrix_to_quaternion_xyzw
from async_pipeline.macvo_runtime import _pose7_xyzw_to_matrix, _tartan_from_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Config/Pipeline/MACVO_ReSplat_Async_Metrics_P000_0_50.yaml",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
    )
    return parser.parse_args()


def error_statistics(errors: np.ndarray) -> dict[str, float | int]:
    if errors.size == 0:
        return {"num_poses": 0}
    return {
        "num_poses": int(errors.size),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }


def umeyama_align(
    source: np.ndarray,
    target: np.ndarray,
    *,
    estimate_scale: bool,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Align source positions to target positions with SE(3) or Sim(3)."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"expected matching [N,3] trajectories, got {source.shape} and {target.shape}"
        )
    if source.shape[0] < 3:
        raise ValueError("at least three poses are required for trajectory alignment")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.T @ target_centered / source.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(Vt.T @ U.T) < 0:
        correction[-1, -1] = -1.0
    rotation = Vt.T @ correction @ U.T

    if estimate_scale:
        source_variance = np.mean(np.sum(source_centered**2, axis=1))
        if source_variance <= 1e-15:
            raise ValueError("predicted trajectory has zero translational variance")
        scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    else:
        scale = 1.0

    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale, rotation, translation


def trajectory_metric(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    alignment: str,
) -> dict[str, Any]:
    predicted_positions = predicted[:, :3, 3]
    gt_positions = ground_truth[:, :3, 3]

    if alignment == "raw":
        aligned = predicted_positions
        scale = 1.0
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
    elif alignment == "se3":
        aligned, scale, rotation, translation = umeyama_align(
            predicted_positions,
            gt_positions,
            estimate_scale=False,
        )
    elif alignment == "sim3":
        aligned, scale, rotation, translation = umeyama_align(
            predicted_positions,
            gt_positions,
            estimate_scale=True,
        )
    else:
        raise ValueError(f"unknown alignment mode {alignment}")

    errors = np.linalg.norm(aligned - gt_positions, axis=1)
    return {
        "alignment": alignment,
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "translation_error_m": error_statistics(errors),
    }


def load_gt_trajectory(
    resolved: dict[str, Any],
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    data_config_path = Path(resolved["paths"]["data_config"])
    data_config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    dataset_root = Path(data_config["args"]["root"]).expanduser().resolve()
    pose_path = dataset_root / "pose_lcam_front.txt"
    rows = np.loadtxt(pose_path, dtype=np.float64)

    start = int(resolved["sequence"]["start_index"])
    end = int(resolved["sequence"]["end_index"])
    frame_indices = np.arange(start, min(end, start + count), dtype=np.int64)
    if frame_indices.size == 0 or int(frame_indices[-1]) >= rows.shape[0]:
        raise IndexError(
            f"requested GT rows [{start},{end}) outside pose file with {rows.shape[0]} rows"
        )

    tartan_from_cv = _tartan_from_cv(dtype=torch.float64)
    absolute: list[np.ndarray] = []
    for index in frame_indices:
        pose = _pose7_xyzw_to_matrix(torch.from_numpy(rows[int(index)]))
        absolute.append((pose @ tartan_from_cv).numpy())
    absolute_array = np.stack(absolute, axis=0)
    first_inverse = np.linalg.inv(absolute_array[0])
    relative = first_inverse[None] @ absolute_array
    return frame_indices, relative


def collect_predicted_trajectory(runner, count: int) -> tuple[np.ndarray, np.ndarray]:
    frontend = runner.pose_frontend
    graph = frontend.system.graph
    available = min(count, len(frontend._descriptors))
    poses: list[np.ndarray] = []
    valid: list[bool] = []
    for index in range(available):
        pose = frontend._opencv_relative_pose(index).detach().cpu().double().numpy()
        need_interp = bool(
            graph.frames.data["need_interp"][index].detach().cpu().item()
        )
        poses.append(pose)
        valid.append(not need_interp)
    return np.stack(poses, axis=0), np.asarray(valid, dtype=bool)


def save_tum(path: Path, frame_indices: np.ndarray, poses: np.ndarray) -> None:
    lines: list[str] = []
    for frame_index, pose in zip(frame_indices, poses):
        rotation = torch.from_numpy(pose[:3, :3]).double()
        quaternion = rotation_matrix_to_quaternion_xyzw(rotation).cpu().numpy()
        translation = pose[:3, 3]
        timestamp = float(frame_index) * 0.1
        lines.append(
            f"{timestamp:.9f} "
            f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
            f"{quaternion[0]:.9f} {quaternion[1]:.9f} "
            f"{quaternion[2]:.9f} {quaternion[3]:.9f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_pose(runner, resolved: dict[str, Any], num_frames: int, output: Path) -> dict[str, Any]:
    predicted_all, valid_mask = collect_predicted_trajectory(runner, num_frames)
    frame_indices, gt_all = load_gt_trajectory(resolved, predicted_all.shape[0])

    predicted = predicted_all[valid_mask]
    ground_truth = gt_all[valid_mask]
    valid_indices = frame_indices[valid_mask]
    if predicted.shape[0] < 3:
        raise RuntimeError("fewer than three valid MAC-VO poses are available")

    report = {
        "coordinate_convention": "metric OpenCV c2w, first frame identity",
        "num_requested_poses": int(predicted_all.shape[0]),
        "num_valid_poses": int(predicted.shape[0]),
        "num_skipped_need_interp": int((~valid_mask).sum()),
        "raw": trajectory_metric(predicted, ground_truth, alignment="raw"),
        "se3": trajectory_metric(predicted, ground_truth, alignment="se3"),
        "sim3": trajectory_metric(predicted, ground_truth, alignment="sim3"),
    }

    np.savez_compressed(
        output / "macvo_async_trajectory.npz",
        selected_original_indices=frame_indices,
        valid_mask=valid_mask,
        T_pred_c2w_opencv=predicted_all,
        T_gt_c2w_opencv=gt_all,
    )
    save_tum(output / "macvo_pred_valid.tum", valid_indices, predicted)
    save_tum(output / "macvo_gt_valid.tum", valid_indices, ground_truth)
    (output / "pose_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = base.absolute(args.config, root)
    config = base.load_yaml(config_path)
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}")
        key, raw = item.split("=", 1)
        base.nested_set(config, key, base.normalize_override_value(key, raw))

    resolved = base.resolve(config, root)
    base.validate_paths(resolved)
    work_dir = Path(resolved["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "resolved_async_pipeline_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    runner = base.build_system(resolved)
    pipeline_summary = runner.run()
    pose_metrics = evaluate_pose(
        runner,
        resolved,
        int(pipeline_summary["num_frames"]),
        work_dir,
    )

    streaming_sec = float(pipeline_summary["streaming_wall_time_sec"])
    initialization_sec = float(pipeline_summary["initialization_sec"])
    num_frames = int(pipeline_summary["num_frames"])
    backend = pipeline_summary.get("backend") or {}
    comprehensive = {
        "pipeline": pipeline_summary["pipeline"],
        "pose": pose_metrics,
        "rendering": backend.get("final_metrics", {}),
        "map": {
            "num_train_packets": backend.get("num_train_packets"),
            "num_test_cameras": backend.get("num_test_cameras"),
            "total_iterations": backend.get("total_iterations"),
            "final_num_gaussians": backend.get("final_num_gaussians"),
        },
        "runtime": {
            "initialization_sec": initialization_sec,
            "streaming_wall_time_sec_excluding_initialization": streaming_sec,
            "end_to_end_sec_including_initialization": initialization_sec + streaming_sec,
            "streaming_fps_excluding_initialization": num_frames / streaming_sec,
            "end_to_end_fps_including_initialization": num_frames
            / (initialization_sec + streaming_sec),
            "sec_per_frame_streaming": streaming_sec / num_frames,
        },
        "artifacts": {
            "pipeline_summary": str(runner.config.summary_path.expanduser().resolve()),
            "backend_summary": str(
                work_dir
                / resolved["backend"]["output_name"]
                / "incremental_backend_summary.json"
            ),
            "metrics_log": str(
                work_dir
                / resolved["backend"]["output_name"]
                / "metrics_log.json"
            ),
            "trajectory_npz": str(work_dir / "macvo_async_trajectory.npz"),
            "pred_tum": str(work_dir / "macvo_pred_valid.tum"),
            "gt_tum": str(work_dir / "macvo_gt_valid.tum"),
        },
    }
    output = work_dir / "comprehensive_metrics.json"
    output.write_text(json.dumps(comprehensive, indent=2), encoding="utf-8")
    print("\n[comprehensive metrics]", flush=True)
    print(json.dumps(comprehensive, indent=2), flush=True)


if __name__ == "__main__":
    main()
