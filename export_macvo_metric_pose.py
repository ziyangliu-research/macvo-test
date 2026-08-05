#!/usr/bin/env python3
"""Run MAC-VO and export a metric OpenCV c2w trajectory for ReSplat.

The output NPZ is compatible with ZipMap's generic
``run_pose_resplat_metric_packet_only.py`` interface.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def abs_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
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
        subprocess.run(cmd, cwd=repo, check=True)
        run_sec = time.perf_counter() - start

    poses_path = newest_file(runtime_root, "poses.npy")
    space = poses_path.parent
    raw = np.asarray(np.load(poses_path), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 8:
        raise ValueError(f"Expected poses.npy [N,8], got {raw.shape}")

    timestamps_ns = raw[:, 0].astype(np.int64)
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
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Done] pose NPZ: {result_path}")
    print(f"[Done] need_interp: {int(need_interp.sum())}/{len(need_interp)}")


if __name__ == "__main__":
    main()
