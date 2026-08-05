#!/usr/bin/env python3
"""
Validate a local TartanAir v2 stereo sequence for MAC-VO and write a
single-sequence YAML config using MAC-VO's native TartanAirv2_NoIMU loader.

Expected dataset layout:
    <root>/
      image_lcam_front/
      image_rcam_front/
      pose_lcam_front.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png"}


def list_pngs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Missing image directory: {directory}")
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No PNG images found in: {directory}")
    return images


def image_shape(path: Path) -> tuple[int, int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV failed to read: {path}")
    return tuple(int(v) for v in image.shape)


def build_config(
    dataset_root: Path,
    sequence_name: str,
    gt_pose: bool,
    gt_depth: bool,
    gt_flow: bool,
) -> str:
    def yaml_bool(value: bool) -> str:
        return "true" if value else "false"

    return (
        "type: TartanAirv2_NoIMU\n"
        f"name: {sequence_name}\n"
        "args:\n"
        f"  root: {dataset_root}\n"
        "  compressed: true\n"
        f"  gtDepth: {yaml_bool(gt_depth)}\n"
        f"  gtPose: {yaml_bool(gt_pose)}\n"
        f"  gtFlow: {yaml_bool(gt_flow)}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--sequence_name", default="House_easy_P000")
    parser.add_argument("--expected_width", type=int, default=640)
    parser.add_argument("--expected_height", type=int, default=640)
    parser.add_argument("--gt_pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt_depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gt_flow", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    output_config = Path(args.output_config).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {root}")

    left_dir = root / "image_lcam_front"
    right_dir = root / "image_rcam_front"
    pose_file = root / "pose_lcam_front.txt"

    left_images = list_pngs(left_dir)
    right_images = list_pngs(right_dir)

    if len(left_images) != len(right_images):
        raise ValueError(
            f"Stereo image count mismatch: left={len(left_images)}, "
            f"right={len(right_images)}"
        )

    sample_indices = sorted(set([0, len(left_images) // 2, len(left_images) - 1]))

    for index in sample_indices:
        left_shape = image_shape(left_images[index])
        right_shape = image_shape(right_images[index])

        if left_shape != right_shape:
            raise ValueError(
                f"Stereo shape mismatch at index {index}: "
                f"left={left_shape}, right={right_shape}"
            )

        height, width = left_shape[:2]
        if height != args.expected_height or width != args.expected_width:
            raise ValueError(
                f"Unexpected image size at index {index}: "
                f"{width}x{height}; expected "
                f"{args.expected_width}x{args.expected_height}"
            )

    pose_count = None
    if args.gt_pose:
        if not pose_file.is_file():
            raise FileNotFoundError(f"Missing GT pose file: {pose_file}")

        poses = np.loadtxt(pose_file, dtype=np.float64)
        poses = np.atleast_2d(poses)

        if poses.shape[1] != 7:
            raise ValueError(
                f"Expected pose_lcam_front.txt with 7 columns "
                f"[tx ty tz qx qy qz qw], got shape {poses.shape}"
            )
        if poses.shape[0] < len(left_images):
            raise ValueError(
                f"GT pose count is smaller than image count: "
                f"poses={poses.shape[0]}, images={len(left_images)}"
            )
        if not np.isfinite(poses).all():
            raise ValueError("GT pose file contains NaN or Inf values.")
        pose_count = int(poses.shape[0])

    if args.gt_depth and not (root / "depth_lcam_front").is_dir():
        raise FileNotFoundError(
            f"--gt_depth was enabled but directory is missing: "
            f"{root / 'depth_lcam_front'}"
        )

    if args.gt_flow and not (root / "flow_lcam_front").is_dir():
        raise FileNotFoundError(
            f"--gt_flow was enabled but directory is missing: "
            f"{root / 'flow_lcam_front'}"
        )

    config_text = build_config(
        dataset_root=root,
        sequence_name=args.sequence_name,
        gt_pose=args.gt_pose,
        gt_depth=args.gt_depth,
        gt_flow=args.gt_flow,
    )

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(config_text, encoding="utf-8")

    print("Validation passed.")
    print(f"Dataset root : {root}")
    print(f"Left images  : {len(left_images)}")
    print(f"Right images : {len(right_images)}")
    print(f"Image size   : {args.expected_width}x{args.expected_height}")
    print(f"GT poses     : {pose_count if pose_count is not None else 'disabled'}")
    print(f"First left   : {left_images[0].name}")
    print(f"Last left    : {left_images[-1].name}")
    print(f"Config saved : {output_config}")
    print()
    print(config_text)


if __name__ == "__main__":
    main()
