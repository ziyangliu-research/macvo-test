#!/usr/bin/env python3
"""Run MAC-VO and export a metric OpenCV c2w trajectory for ReSplat.

The output NPZ is compatible with ZipMap's generic
``run_pose_resplat_metric_packet_only.py`` interface.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


KNOWN_WARNING_FILTERS = [
    "ignore:instrumentor did not find the target function",
    "ignore:In 'main'",
    "ignore:The parameter 'pretrained' is deprecated since 0.13",
    "ignore:Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13",
]


def abs_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def subprocess_env(show_known_warnings: bool) -> dict[str, str]:
    env = os.environ.copy()
    if show_known_warnings:
        return env
    known = ",".join(KNOWN_WARNING_FILTERS)
    existing = env.get("PYTHONWARNINGS", "").strip(",")
    env["PYTHONWARNINGS"] = f"{existing},{known}" if existing else known
    return env


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError(f"Invalid near-zero quaternion: {q}")
    q /= norm
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_xyzw_to_matrix(pose7[3:7])
    out[:3, 3] = pose7[:3]
    return out


def tartan_from_cv() -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    return out


def newest_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"No {name} found below {root}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def load_need_interp(space: Path, count: int) -> np.ndarray:
    path = space / "tensor_map.npz"
    if not path.is_file():
        return np.zeros(count, dtype=bool)
    with np.load(path, allow_pickle=False) as data:
        candidates = [key for key in data.files if key.endswith("need_interp")]
        if not candidates:
            return np.zeros(count, dtype=bool)
        values = np.asarray(data[candidates[0]]).reshape(-1).astype(bool)
    if len(values) != count:
        return np.zeros(count, dtype=bool)
    return values


def umeyama_alignment(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    with_scale: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"Expected matching [N,3] trajectories, got {source.shape} and {target.shape}"
        )
    if len(source) < 2:
        raise ValueError("At least two matched poses are required for alignment")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = target_centered.T @ source_centered / len(source)
    U, singular_values, Vt = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        signs[-1] = -1.0
    rotation = U @ np.diag(signs) @ Vt

    if with_scale:
        variance = float(np.mean(np.sum(source_centered**2, axis=1)))
        if variance < 1e-15:
            raise ValueError("Estimated trajectory has near-zero translation variance")
        scale = float(np.dot(singular_values, signs) / variance)
    else:
        scale = 1.0

    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def ate_statistics(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def evaluate_against_reference(
    poses_path: Path,
    timestamps_ns: np.ndarray,
    estimated_pose7: np.ndarray,
) -> dict[str, object]:
    reference_path = poses_path.parent / "ref_poses.npy"
    if not reference_path.is_file():
        return {
            "available": False,
            "reason": "ref_poses.npy was not generated; enable gtPose in the MAC-VO data config",
        }

    reference = np.asarray(np.load(reference_path), dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 8:
        return {
            "available": False,
            "reason": f"Expected ref_poses.npy [N,8], got {reference.shape}",
            "reference_file": str(reference_path),
        }

    reference_by_time = {
        int(round(row[0])): row[1:8]
        for row in reference
    }
    matched_est: list[np.ndarray] = []
    matched_ref: list[np.ndarray] = []
    matched_timestamps: list[int] = []
    for timestamp, estimate in zip(timestamps_ns, estimated_pose7):
        reference_pose = reference_by_time.get(int(timestamp))
        if reference_pose is None:
            continue
        matched_est.append(np.asarray(estimate, dtype=np.float64))
        matched_ref.append(np.asarray(reference_pose, dtype=np.float64))
        matched_timestamps.append(int(timestamp))

    if len(matched_est) < 2:
        return {
            "available": False,
            "reason": "fewer than two timestamps matched between poses.npy and ref_poses.npy",
            "reference_file": str(reference_path),
            "matched_frames": len(matched_est),
        }

    estimated_xyz = np.stack(matched_est)[:, :3]
    reference_xyz = np.stack(matched_ref)[:, :3]
    result: dict[str, object] = {
        "available": True,
        "reference_file": str(reference_path),
        "matched_frames": len(matched_est),
        "matched_timestamps_ns": matched_timestamps,
    }

    for name, with_scale in (("se3", False), ("sim3", True)):
        scale, rotation, translation = umeyama_alignment(
            estimated_xyz, reference_xyz, with_scale=with_scale
        )
        aligned = (
            scale * (rotation @ estimated_xyz.T)
        ).T + translation[None]
        errors = np.linalg.norm(aligned - reference_xyz, axis=1)
        result[name] = {
            "alignment_scale": scale,
            "alignment_rotation": rotation.tolist(),
            "alignment_translation": translation.tolist(),
            "ate_m": ate_statistics(errors),
        }
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MAC-VO -> metric OpenCV c2w NPZ")
    p.add_argument("--macvo_repo", required=True)
    p.add_argument("--odom", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timing", action="store_true")
    p.add_argument("--reuse_existing", action="store_true")
    p.add_argument("--skip_ate", action="store_true")
    p.add_argument(
        "--show_known_warnings",
        action="store_true",
        help="Show known benign jaxtyping/Hydra/torchvision compatibility warnings.",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.stride != 1:
        raise ValueError("MAC-VO native sequence runner currently requires --stride 1")
    if args.end_index <= args.start_index:
        raise ValueError("--end_index must be larger than --start_index")

    repo = abs_path(args.macvo_repo)
    out = abs_path(args.output_dir)
    runtime_root = out / "macvo_runtime"
    out.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    run_sec = 0.0
    existing = list(runtime_root.rglob("poses.npy"))
    if not (args.reuse_existing and existing):
        cmd = [
            args.python, str(repo / "MACVO.py"),
            "--odom", str(abs_path(args.odom)),
            "--data", str(abs_path(args.data)),
            "--seq_from", str(args.start_index),
            "--seq_to", str(args.end_index),
            "--resultRoot", str(runtime_root),
            "--noeval",
        ]
        if args.timing:
            cmd.append("--timing")
        print("[MAC-VO]", " ".join(cmd), flush=True)
        start = time.perf_counter()
        subprocess.run(
            cmd,
            cwd=repo,
            check=True,
            env=subprocess_env(args.show_known_warnings),
        )
        run_sec = time.perf_counter() - start

    poses_path = newest_file(runtime_root, "poses.npy")
    space = poses_path.parent
    raw = np.asarray(np.load(poses_path), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 8:
        raise ValueError(f"Expected poses.npy [N,8], got {raw.shape}")

    timestamps_ns = np.rint(raw[:, 0]).astype(np.int64)
    pose7 = raw[:, 1:8]
    T_tartan = np.stack([pose7_to_matrix(row) for row in pose7])
    T_abs_cv = T_tartan @ tartan_from_cv()[None]
    T0_inv = np.linalg.inv(T_abs_cv[0])
    T_cv = np.stack([T0_inv @ pose for pose in T_abs_cv]).astype(np.float32)

    expected = args.end_index - args.start_index
    if len(T_cv) != expected:
        raise ValueError(
            f"MAC-VO returned {len(T_cv)} poses, expected {expected} for "
            f"[{args.start_index}, {args.end_index})"
        )

    indices = np.arange(args.start_index, args.end_index, dtype=np.int64)
    need_interp = load_need_interp(space, len(T_cv))
    valid = ~need_interp

    result_path = out / "macvo_pose_results.npz"
    np.savez_compressed(
        result_path,
        T_raw_accumulated_c2w_opencv=T_cv,
        T_macvo_c2w_tartanair=T_tartan.astype(np.float32),
        selected_original_indices=indices,
        timestamps_ns=timestamps_ns,
        need_interp=need_interp,
        valid=valid,
    )

    trajectory_path = out / "trajectory_c2w_opencv.txt"
    np.savetxt(trajectory_path, T_cv.reshape(len(T_cv), 16), fmt="%.9f")

    evaluation: dict[str, object]
    if args.skip_ate:
        evaluation = {"available": False, "reason": "disabled by --skip_ate"}
    else:
        evaluation = evaluate_against_reference(poses_path, timestamps_ns, pose7)
    (out / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )

    summary = {
        "source": "MAC-VO",
        "pose_file": str(poses_path),
        "output_npz": str(result_path),
        "coordinate_convention": "metric OpenCV camera-to-world, first frame identity",
        "frame_range": [args.start_index, args.end_index],
        "num_frames": len(T_cv),
        "num_need_interp": int(need_interp.sum()),
        "macvo_runtime_sec": run_sec,
        "postprocess_note": "poses.npy is the official trajectory saved after MAC-VO termination/post-processing",
        "evaluation": evaluation,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Done] pose NPZ: {result_path}")
    print(f"[Done] need_interp: {int(need_interp.sum())}/{len(need_interp)}")
    if evaluation.get("available"):
        se3 = evaluation["se3"]["ate_m"]["rmse"]
        sim3 = evaluation["sim3"]["ate_m"]["rmse"]
        print(f"[ATE] SE(3) RMSE={se3:.6f} m | Sim(3) RMSE={sim3:.6f} m")
    else:
        print(f"[ATE] unavailable: {evaluation.get('reason', 'unknown reason')}")


if __name__ == "__main__":
    main()
