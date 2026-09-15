#!/usr/bin/env python3
"""Audit TartanAir V2 left/right camera pose files.

For each timestamp this script computes the exact stereo relative transform

    T_left_from_right = inv(Twc_left) @ Twc_right

in OpenCV camera axes, then reports baseline magnitude, translation direction,
relative rotation, and frame-to-frame variation.  It does not run MAC-VO,
ReSplat, or the Gaussian backend.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def quat_xyzw_to_matrix(values: list[float]) -> torch.Tensor:
    q = torch.tensor(values, dtype=torch.float64)
    q = q / torch.linalg.vector_norm(q).clamp_min(1e-12)
    x, y, z, w = q.unbind()
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ]
    ).reshape(3, 3)


def tartan_from_cv() -> torch.Tensor:
    out = torch.eye(4, dtype=torch.float64)
    out[:3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    return out


def load_pose_rows(path: Path) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    axis = tartan_from_cv()
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        values = [float(x) for x in raw.replace(",", " ").split()[:7]]
        if len(values) != 7:
            continue
        tx, ty, tz, qx, qy, qz, qw = values
        twc_tartan = torch.eye(4, dtype=torch.float64)
        twc_tartan[:3, :3] = quat_xyzw_to_matrix([qx, qy, qz, qw])
        twc_tartan[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float64)
        rows.append(twc_tartan @ axis)
    if not rows:
        raise RuntimeError(f"no valid pose rows in {path}")
    return rows


def rotation_angle_deg(R: torch.Tensor) -> float:
    c = float(((torch.trace(R) - 1.0) * 0.5).clamp(-1.0, 1.0).item())
    return math.degrees(math.acos(c))


def compute_audit(data_root: Path, max_frames: int | None = None) -> dict:
    left_path = data_root / "pose_lcam_front.txt"
    right_path = data_root / "pose_rcam_front.txt"
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(
            f"expected both pose_lcam_front.txt and pose_rcam_front.txt under {data_root}"
        )

    left = load_pose_rows(left_path)
    right = load_pose_rows(right_path)
    count = min(len(left), len(right))
    if max_frames is not None and max_frames > 0:
        count = min(count, max_frames)
    if count <= 0:
        raise RuntimeError("no overlapping left/right pose rows")

    relatives = [torch.linalg.inv(left[i]) @ right[i] for i in range(count)]
    first = relatives[0]

    baselines = []
    txs = []
    tys = []
    tzs = []
    rotations_from_identity = []
    translation_delta_from_first = []
    rotation_delta_from_first = []
    per_frame = []

    for i, rel in enumerate(relatives):
        t = rel[:3, 3]
        baseline = float(torch.linalg.vector_norm(t).item())
        rot_identity = rotation_angle_deg(rel[:3, :3])
        delta = torch.linalg.inv(first) @ rel
        trans_delta = float(torch.linalg.vector_norm(rel[:3, 3] - first[:3, 3]).item())
        rot_delta = rotation_angle_deg(delta[:3, :3])

        baselines.append(baseline)
        txs.append(float(t[0].item()))
        tys.append(float(t[1].item()))
        tzs.append(float(t[2].item()))
        rotations_from_identity.append(rot_identity)
        translation_delta_from_first.append(trans_delta)
        rotation_delta_from_first.append(rot_delta)
        per_frame.append(
            {
                "frame_index": i,
                "baseline_m": baseline,
                "translation_cv_m": [float(v) for v in t.tolist()],
                "rotation_from_identity_deg": rot_identity,
                "translation_delta_from_frame0_m": trans_delta,
                "rotation_delta_from_frame0_deg": rot_delta,
                "T_left_from_right_cv": rel.tolist(),
            }
        )

    def stats(values: list[float]) -> dict[str, float]:
        x = torch.tensor(values, dtype=torch.float64)
        return {
            "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=False).item()),
            "min": float(x.min().item()),
            "max": float(x.max().item()),
        }

    return {
        "data_root": str(data_root),
        "num_frames": count,
        "definition": "T_left_from_right = inv(Twc_left_cv) @ Twc_right_cv",
        "baseline_m": stats(baselines),
        "translation_cv_m": {
            "x": stats(txs),
            "y": stats(tys),
            "z": stats(tzs),
        },
        "rotation_from_identity_deg": stats(rotations_from_identity),
        "variation_vs_frame0": {
            "translation_delta_m": stats(translation_delta_from_first),
            "rotation_delta_deg": stats(rotation_delta_from_first),
        },
        "frame0_T_left_from_right_cv": first.tolist(),
        "per_frame": per_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P004"),
    )
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    audit = compute_audit(root, None if args.max_frames <= 0 else args.max_frames)
    out = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "stereo_pose_audit.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    b = audit["baseline_m"]
    t = audit["translation_cv_m"]
    v = audit["variation_vs_frame0"]
    r = audit["rotation_from_identity_deg"]
    print("===== TartanAir V2 stereo pose audit =====")
    print(f"root   : {root}")
    print(f"frames : {audit['num_frames']}")
    print(
        "baseline [m] "
        f"mean={b['mean']:.9f} std={b['std']:.3e} "
        f"min={b['min']:.9f} max={b['max']:.9f}"
    )
    print(
        "translation CV mean [m] "
        f"x={t['x']['mean']:.9f} y={t['y']['mean']:.9f} z={t['z']['mean']:.9f}"
    )
    print(
        "relative rotation from identity [deg] "
        f"mean={r['mean']:.6e} max={r['max']:.6e}"
    )
    print(
        "variation vs frame0 "
        f"translation_max={v['translation_delta_m']['max']:.3e}m "
        f"rotation_max={v['rotation_delta_deg']['max']:.3e}deg"
    )
    print(f"saved  : {out}")


if __name__ == "__main__":
    main()
