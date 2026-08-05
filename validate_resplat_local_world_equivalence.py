#!/usr/bin/env python3
"""Gate local ReSplat inference before enabling pose/ReSplat parallelism.

For each requested timestamp the same persistent model runs twice:

  A. context poses = [I, fixed stereo rig], followed by explicit SE(3) alignment
  B. context poses = [T_world_left, T_world_left @ fixed stereo rig]

The script compares the complete minimal packet and renders both results from the
same world camera. A non-zero exit means the async coordinate contract must not
be used for production experiments yet.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from async_pipeline.contracts import FrameDescriptor, StereoFrameInput
from async_pipeline.geometry import align_local_packet_to_world
from async_pipeline.resplat_runtime import ResplatPacketGenerator, ResplatRuntimeConfig


def parse_int_ranges(spec: str) -> list[int]:
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", item)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(item))
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resplat_repo", required=True)
    parser.add_argument("--resplat_experiment", default="tartanair_p000_ft")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--left_image", default=None)
    parser.add_argument("--right_image", default=None)
    parser.add_argument("--left_subdir", default="image_lcam_front")
    parser.add_argument("--right_subdir", default="image_rcam_front")
    parser.add_argument("--left_pattern", default="{index:06d}_lcam_front.png")
    parser.add_argument("--right_pattern", default="{index:06d}_rcam_front.png")
    parser.add_argument("--pose_npz", required=True)
    parser.add_argument("--pose_key", default="T_raw_accumulated_c2w_opencv")
    parser.add_argument("--index_key", default="selected_original_indices")
    parser.add_argument("--pose_indices", default="1,10,20,30,40,49")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--input_mode",
        choices=["file_paths", "shared_tensors"],
        default="file_paths",
    )
    parser.add_argument("--output", default="outputs/local_world_equivalence.json")
    parser.add_argument("--max_mean_error", type=float, default=2e-4)
    parser.add_argument("--max_covariance_error", type=float, default=2e-4)
    parser.add_argument("--max_scale_error", type=float, default=2e-5)
    parser.add_argument("--max_opacity_error", type=float, default=2e-5)
    parser.add_argument("--max_harmonics_error", type=float, default=2e-4)
    parser.add_argument("--min_render_psnr", type=float, default=60.0)
    parser.add_argument("--fx", type=float, default=320.0)
    parser.add_argument("--fy", type=float, default=320.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=320.0)
    parser.add_argument("--baseline", type=float, default=0.25000006)
    return parser.parse_args()


def covariance(scales: torch.Tensor, rotations_xyzw: torch.Tensor) -> torch.Tensor:
    from src.model.encoder.common.gaussians import build_covariance

    return build_covariance(scales, rotations_xyzw)


def max_and_mean(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().abs().reshape(-1)
    return {"max": float(value.max().item()), "mean": float(value.mean().item())}


def packet_to_gaussians(packet, device: torch.device):
    from src.model.types import Gaussians

    scales = packet.scales.to(device).unsqueeze(0)
    rotations = packet.rotations_xyzw.to(device).unsqueeze(0)
    return Gaussians(
        means=packet.means.to(device).unsqueeze(0),
        covariances=covariance(scales, rotations),
        harmonics=packet.harmonics.to(device).unsqueeze(0),
        opacities=packet.opacities.to(device).unsqueeze(0),
        scales=scales,
        rotations=rotations,
        rotations_unnorm=rotations,
    )


@torch.inference_mode()
def render_left(generator, packet, world_batch):
    gaussians = packet_to_gaussians(packet, generator.device)
    context = world_batch["context"]
    shape = tuple(int(v) for v in context["image"].shape[-2:])
    return generator.model.decoder.forward(
        gaussians,
        context["extrinsics"][:, :1],
        context["intrinsics"][:, :1],
        context["near"][:, :1],
        context["far"][:, :1],
        shape,
        depth_mode=None,
    ).color[:, 0]


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB"))


def resolve_images(
    args: argparse.Namespace,
    local_pose_index: int,
    original_frame_index: int,
) -> tuple[Path, Path]:
    if args.dataset_root is not None:
        root = Path(args.dataset_root).expanduser().resolve()
        return (
            root
            / args.left_subdir
            / args.left_pattern.format(index=original_frame_index),
            root
            / args.right_subdir
            / args.right_pattern.format(index=original_frame_index),
        )
    if args.left_image is None or args.right_image is None:
        raise ValueError(
            "provide --dataset_root for multi-frame validation, or both "
            "--left_image/--right_image for a single pose index"
        )
    requested = parse_int_ranges(args.pose_indices)
    if len(requested) != 1:
        raise ValueError(
            "explicit --left_image/--right_image supports exactly one --pose_indices value"
        )
    return (
        Path(args.left_image).expanduser().resolve(),
        Path(args.right_image).expanduser().resolve(),
    )


def evaluate_one(
    args: argparse.Namespace,
    generator: ResplatPacketGenerator,
    T_world: torch.Tensor,
    local_pose_index: int,
    original_frame_index: int,
) -> dict[str, object]:
    left_path, right_path = resolve_images(
        args, local_pose_index, original_frame_index
    )
    descriptor = FrameDescriptor(
        sequence_index=local_pose_index,
        frame_index=original_frame_index,
        timestamp_ns=original_frame_index,
        left_path=left_path,
        right_path=right_path,
        is_test=False,
    )
    K = torch.tensor(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    frame_input = StereoFrameInput(
        descriptor=descriptor,
        left_image=load_rgb(left_path),
        right_image=load_rgb(right_path),
        intrinsic_pixel=K,
        baseline_m=args.baseline,
    )
    frame_input.validate(deep=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    local = generator.infer(frame_input, output_frame="left_camera_local")
    local_aligned = align_local_packet_to_world(local.packet, T_world)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    direct = generator.infer(
        frame_input,
        output_frame="world",
        T_world_from_left=T_world,
    )

    local_cov = covariance(
        local_aligned.scales.to(generator.device).unsqueeze(0),
        local_aligned.rotations_xyzw.to(generator.device).unsqueeze(0),
    )[0].cpu()
    direct_cov = covariance(
        direct.packet.scales.to(generator.device).unsqueeze(0),
        direct.packet.rotations_xyzw.to(generator.device).unsqueeze(0),
    )[0].cpu()
    metrics = {
        "means": max_and_mean(local_aligned.means - direct.packet.means),
        "covariances": max_and_mean(local_cov - direct_cov),
        "scales": max_and_mean(local_aligned.scales - direct.packet.scales),
        "opacities": max_and_mean(local_aligned.opacities - direct.packet.opacities),
        "harmonics": max_and_mean(local_aligned.harmonics - direct.packet.harmonics),
    }

    image_local = render_left(generator, local_aligned, direct.batch)
    image_direct = render_left(generator, direct.packet, direct.batch)
    mse = torch.mean((image_local - image_direct) ** 2).clamp_min(1e-12)
    render_psnr = float((-10.0 * torch.log10(mse)).item())
    render_max_abs = float((image_local - image_direct).abs().max().item())
    checks = {
        "means": metrics["means"]["max"] <= args.max_mean_error,
        "covariances": metrics["covariances"]["max"]
        <= args.max_covariance_error,
        "scales": metrics["scales"]["max"] <= args.max_scale_error,
        "opacities": metrics["opacities"]["max"] <= args.max_opacity_error,
        "harmonics": metrics["harmonics"]["max"] <= args.max_harmonics_error,
        "render_psnr": render_psnr >= args.min_render_psnr,
    }
    return {
        "local_pose_index": local_pose_index,
        "original_frame_index": original_frame_index,
        "left_image": str(left_path),
        "right_image": str(right_path),
        "num_gaussians": local.packet.num_gaussians,
        "metrics": metrics,
        "render": {"psnr": render_psnr, "max_abs": render_max_abs},
        "checks": checks,
        "passed": all(checks.values()),
        "local_inference_sec": local.inference_sec,
        "direct_world_inference_sec": direct.inference_sec,
    }


def main() -> None:
    args = parse_args()
    with np.load(args.pose_npz, allow_pickle=False) as data:
        poses = np.asarray(data[args.pose_key], dtype=np.float32)
        if args.index_key in data.files:
            original_indices = np.asarray(data[args.index_key], dtype=np.int64)
        else:
            original_indices = np.arange(len(poses), dtype=np.int64)
    requested = parse_int_ranges(args.pose_indices)
    if not requested:
        raise ValueError("--pose_indices produced no indices")
    for index in requested:
        if not 0 <= index < len(poses):
            raise IndexError(f"pose index {index} outside [0,{len(poses)-1}]")

    config = ResplatRuntimeConfig(
        repo=Path(args.resplat_repo),
        experiment=args.resplat_experiment,
        device=args.device,
        checkpoint=None if args.checkpoint is None else Path(args.checkpoint),
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        stereo_baseline=args.baseline,
        refine_steps=0,
        deterministic=False,
        pin_output_memory=False,
        input_mode=args.input_mode,
        handoff_mode="pinned_cpu",
        strict_validation=True,
    )
    generator = ResplatPacketGenerator(config)
    generator.initialize()
    results: list[dict[str, object]] = []
    for local_index in requested:
        results.append(
            evaluate_one(
                args,
                generator,
                torch.from_numpy(poses[local_index]),
                local_index,
                int(original_indices[local_index]),
            )
        )

    report = {
        "purpose": "gate local-frame ReSplat inference before asynchronous integration",
        "pose_npz": str(Path(args.pose_npz).expanduser().resolve()),
        "pose_indices": requested,
        "input_mode": args.input_mode,
        "rotation_convention": (
            "ReSplat xyzw; backend later converts aligned rotations to GraphDECO wxyz"
        ),
        "thresholds": {
            "max_mean_error": args.max_mean_error,
            "max_covariance_error": args.max_covariance_error,
            "max_scale_error": args.max_scale_error,
            "max_opacity_error": args.max_opacity_error,
            "max_harmonics_error": args.max_harmonics_error,
            "min_render_psnr": args.min_render_psnr,
        },
        "results": results,
        "passed": all(bool(result["passed"]) for result in results),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    generator.close()
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
