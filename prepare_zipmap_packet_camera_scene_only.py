#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the minimal GraphDECO camera scene required by the incremental
ReSplat/3DGS optimization scripts.

Inputs
------
- ReSplat/ZipMap packet directory. Each packet must contain target_extrinsics.
- Original RGB image directory.

Outputs
-------
output_scene/
  transforms_train.json
  transforms_test.json
  images/
  camera_scene_manifest.json

No fused Gaussian PLY is generated. The incremental training script creates
its own temporary dummy points3d.ply when constructing GraphDECO Camera objects.

Pose convention
---------------
Packet target_extrinsics is assumed to be an OpenCV-style camera pose:
  Twc: camera-to-world, +x right, +y down, +z forward
by default.

GraphDECO's Blender transforms reader expects an OpenGL/Blender c2w matrix
and internally flips columns 1 and 2. Therefore, this script writes:
  c2w_json[:, 1:3] *= -1
so that GraphDECO reconstructs the original OpenCV Twc.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def parse_range_spec(spec: str, n: int) -> List[int]:
    if not spec.strip():
        return list(range(n))
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a_i, b_i = int(a), int(b)
            if b_i < a_i:
                raise ValueError(f"Descending range is not allowed: {token}")
            out.extend(range(a_i, b_i + 1))
        else:
            out.append(int(token))
    out = sorted(set(out))
    bad = [i for i in out if i < 0 or i >= n]
    if bad:
        raise IndexError(f"Packet indices outside 0..{n-1}: {bad[:20]}")
    return out


def first_matrix(value: Any, shape=(4, 4)) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().float().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"Expected matrix {shape}, got {arr.shape}")
    return arr.astype(np.float64)


def first_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return int(default)
        return int(round(float(value.detach().cpu().reshape(-1)[0].item())))
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return int(default)
    return int(round(float(arr[0])))


def is_test_frame(
    local_pos: int,
    packet_index: int,
    frame_index: int,
    split_every: int,
    split_offset: int,
    split_index_mode: str,
) -> bool:
    if split_index_mode == "local_index":
        value = local_pos
    elif split_index_mode == "packet_index":
        value = packet_index
    elif split_index_mode == "frame_index":
        value = frame_index
    else:
        raise ValueError(split_index_mode)
    return (value - split_offset) % split_every == 0


def resolve_image(
    image_dir: Path,
    image_pattern: str,
    frame_index: int,
) -> Path:
    image_path = image_dir / image_pattern.format(index=frame_index)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for frame {frame_index}: {image_path}")
    return image_path


def install_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst):
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src.resolve(), dst)
    else:
        raise ValueError(mode)


def opencv_twc_to_graphdeco_json_c2w(twc_cv: np.ndarray) -> np.ndarray:
    """
    GraphDECO readCamerasFromTransforms() flips c2w columns 1 and 2:
        c2w[:3, 1:3] *= -1
    Write the inverse conversion here so the reconstructed pose equals twc_cv.
    """
    out = twc_cv.copy()
    out[:3, 1:3] *= -1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a minimal GraphDECO camera scene from ZipMap/ReSplat packet poses."
    )
    parser.add_argument("--packet_dir", type=Path, required=True)
    parser.add_argument("--packet_range_spec", type=str, default="")
    parser.add_argument("--dataset_start_index", type=int, default=0)

    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument(
        "--image_pattern",
        type=str,
        default="{index:06d}_lcam_front.png",
    )
    parser.add_argument("--image_mode", choices=["symlink", "copy", "hardlink"], default="symlink")

    parser.add_argument("--output_scene", type=Path, required=True)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fx", type=float, default=320.0)
    parser.add_argument("--fy", type=float, default=320.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=320.0)

    parser.add_argument(
        "--packet_extrinsic_type",
        choices=["Twc", "Tcw"],
        default="Twc",
        help="Meaning of packet['target_extrinsics'].",
    )
    parser.add_argument(
        "--packet_extrinsic_key",
        type=str,
        default="target_extrinsics",
    )

    parser.add_argument("--split_every", type=int, default=5)
    parser.add_argument("--split_offset", type=int, default=4)
    parser.add_argument(
        "--split_index_mode",
        choices=["local_index", "packet_index", "frame_index"],
        default="local_index",
    )
    args = parser.parse_args()

    if args.split_every <= 0:
        raise ValueError("--split_every must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Image dimensions must be positive")
    if args.fx <= 0 or args.fy <= 0:
        raise ValueError("Focal lengths must be positive")

    packet_paths = sorted(
        list(args.packet_dir.glob("*.pt")) + list(args.packet_dir.glob("*.pth")),
        key=natural_key,
    )
    if not packet_paths:
        raise FileNotFoundError(f"No .pt/.pth packets found in {args.packet_dir}")

    selected = parse_range_spec(args.packet_range_spec, len(packet_paths))

    output_scene = args.output_scene.expanduser().resolve()
    images_out = output_scene / "images"
    output_scene.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    camera_angle_x = 2.0 * math.atan(float(args.width) / (2.0 * float(args.fx)))
    camera_angle_y = 2.0 * math.atan(float(args.height) / (2.0 * float(args.fy)))

    train_frames: List[Dict[str, Any]] = []
    test_frames: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for local_pos, packet_index in enumerate(selected):
        packet_path = packet_paths[packet_index]
        packet = torch.load(packet_path, map_location="cpu")
        if not isinstance(packet, dict):
            raise TypeError(f"Packet is not a dict: {packet_path}")
        if args.packet_extrinsic_key not in packet:
            raise KeyError(
                f"{packet_path.name} lacks {args.packet_extrinsic_key!r}. "
                f"Available keys: {sorted(packet.keys())}"
            )

        ext = first_matrix(packet[args.packet_extrinsic_key], (4, 4))
        if args.packet_extrinsic_type == "Twc":
            twc_cv = ext
        else:
            twc_cv = np.linalg.inv(ext)

        if not np.isfinite(twc_cv).all():
            raise ValueError(f"Non-finite camera pose in {packet_path}")
        if abs(np.linalg.det(twc_cv[:3, :3])) < 1e-8:
            raise ValueError(f"Singular camera rotation in {packet_path}")

        default_frame = args.dataset_start_index + packet_index
        frame_index = first_int(packet.get("target_index"), default_frame)

        src_image = resolve_image(args.image_dir, args.image_pattern, frame_index)
        image_name = src_image.name
        dst_image = images_out / image_name
        install_image(src_image, dst_image, args.image_mode)

        c2w_json = opencv_twc_to_graphdeco_json_c2w(twc_cv)
        file_path = f"./images/{dst_image.stem}"

        frame_record = {
            "file_path": file_path,
            "transform_matrix": c2w_json.tolist(),
        }

        test = is_test_frame(
            local_pos=local_pos,
            packet_index=packet_index,
            frame_index=frame_index,
            split_every=args.split_every,
            split_offset=args.split_offset,
            split_index_mode=args.split_index_mode,
        )
        if test:
            test_frames.append(frame_record)
            split = "test"
        else:
            train_frames.append(frame_record)
            split = "train"

        manifest_rows.append(
            {
                "local_pos": int(local_pos),
                "packet_sorted_index": int(packet_index),
                "packet_file": packet_path.name,
                "frame_index": int(frame_index),
                "split": split,
                "image_source": str(src_image.resolve()),
                "image_output": str(dst_image),
                "camera_center": twc_cv[:3, 3].tolist(),
            }
        )

    if not train_frames:
        raise RuntimeError("No training frames were produced")
    if not test_frames:
        print("[warning] No test frames were produced")

    common = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": float(args.fx),
        "fl_y": float(args.fy),
        "cx": float(args.cx),
        "cy": float(args.cy),
        "w": int(args.width),
        "h": int(args.height),
    }

    train_json = dict(common)
    train_json["frames"] = train_frames
    test_json = dict(common)
    test_json["frames"] = test_frames

    (output_scene / "transforms_train.json").write_text(
        json.dumps(train_json, indent=2),
        encoding="utf-8",
    )
    (output_scene / "transforms_test.json").write_text(
        json.dumps(test_json, indent=2),
        encoding="utf-8",
    )

    centers = np.asarray([row["camera_center"] for row in manifest_rows], dtype=np.float64)
    step_norms = (
        np.linalg.norm(np.diff(centers, axis=0), axis=1)
        if len(centers) > 1
        else np.empty((0,), dtype=np.float64)
    )

    manifest = {
        "packet_dir": str(args.packet_dir.expanduser().resolve()),
        "output_scene": str(output_scene),
        "packet_extrinsic_key": args.packet_extrinsic_key,
        "packet_extrinsic_type": args.packet_extrinsic_type,
        "graphdeco_json_conversion": "OpenCV Twc -> negate c2w columns 1 and 2",
        "num_selected": len(selected),
        "num_train": len(train_frames),
        "num_test": len(test_frames),
        "split": {
            "every": args.split_every,
            "offset": args.split_offset,
            "index_mode": args.split_index_mode,
        },
        "intrinsics": {
            "width": args.width,
            "height": args.height,
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
            "camera_angle_x": camera_angle_x,
            "camera_angle_y": camera_angle_y,
        },
        "trajectory": {
            "center_min": centers.min(axis=0).tolist(),
            "center_max": centers.max(axis=0).tolist(),
            "median_step": float(np.median(step_norms)) if len(step_norms) else None,
            "max_step": float(np.max(step_norms)) if len(step_norms) else None,
        },
        "frames": manifest_rows,
    }
    (output_scene / "camera_scene_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"[done] output_scene: {output_scene}")
    print(f"[done] train frames: {len(train_frames)}")
    print(f"[done] test frames:  {len(test_frames)}")
    print(f"[done] image mode:  {args.image_mode}")
    print("[done] no fused Gaussian PLY was generated")


if __name__ == "__main__":
    main()
