#!/usr/bin/env python3
"""Validate whether local-first ReSplat packets are safe for the async pipeline.

This test separates two questions that the original exact-equivalence validator
combined:

1. Rigid alignment correctness
   Render one local packet from its local camera, then rigidly align the exact
   same packet and render it from the corresponding world camera. These two
   images should be numerically equivalent.

2. ReSplat pose-frame sensitivity
   Run ReSplat once in the canonical left-camera frame and once with the MAC-VO
   world pose. The learned model is not guaranteed to be exactly SE(3)
   equivariant because its point transformer constructs discrete KNN
   neighborhoods from world-space points. Parameter equality is therefore kept
   as a diagnostic, while async safety is decided from alignment invariance and
   GT render-quality regression.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import validate_resplat_local_world_equivalence as exact
from async_pipeline.contracts import FrameDescriptor, StereoFrameInput
from async_pipeline.geometry import align_local_packet_to_world
from async_pipeline.resplat_runtime import ResplatPacketGenerator, ResplatRuntimeConfig


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
    parser.add_argument(
        "--output", default="outputs/local_alignment_quality_file_paths.json"
    )
    parser.add_argument("--fx", type=float, default=320.0)
    parser.add_argument("--fy", type=float, default=320.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=320.0)
    parser.add_argument("--baseline", type=float, default=0.25000006)

    # Alignment of the exact same packet should be nearly render invariant.
    parser.add_argument("--min_alignment_psnr", type=float, default=60.0)
    parser.add_argument("--max_alignment_abs_error", type=float, default=5e-3)

    # Local-first inference is accepted only when it does not materially reduce
    # packet self-render quality relative to the current direct-world path.
    parser.add_argument("--max_gt_psnr_drop", type=float, default=0.25)
    parser.add_argument("--max_gt_ssim_drop", type=float, default=0.002)

    # Exact pose equivariance remains diagnostic and is not required for safety.
    parser.add_argument("--max_mean_error", type=float, default=2e-4)
    parser.add_argument("--max_covariance_error", type=float, default=2e-4)
    parser.add_argument("--max_scale_error", type=float, default=2e-5)
    parser.add_argument("--max_opacity_error", type=float, default=2e-5)
    parser.add_argument("--max_harmonics_error", type=float, default=2e-4)
    parser.add_argument("--min_direct_similarity_psnr", type=float, default=60.0)
    return parser.parse_args()


def scalar_metric(value: torch.Tensor) -> float:
    return float(value.detach().float().reshape(-1).mean().item())


def render_metrics(gt: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    from src.evaluation.metrics import compute_psnr, compute_ssim

    return {
        "psnr": scalar_metric(compute_psnr(gt, prediction)),
        "ssim": scalar_metric(compute_ssim(gt, prediction)),
    }


def image_similarity(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    mse = torch.mean((a - b) ** 2).clamp_min(1e-12)
    return {
        "psnr": float((-10.0 * torch.log10(mse)).item()),
        "max_abs": float((a - b).abs().max().item()),
        "mean_abs": float((a - b).abs().mean().item()),
    }


def build_frame_input(
    args: argparse.Namespace,
    local_pose_index: int,
    original_frame_index: int,
) -> StereoFrameInput:
    left_path, right_path = exact.resolve_images(
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
        left_image=exact.load_rgb(left_path),
        right_image=exact.load_rgb(right_path),
        intrinsic_pixel=K,
        baseline_m=args.baseline,
    )
    frame_input.validate(deep=True)
    return frame_input


def parameter_equivariance(
    args: argparse.Namespace,
    generator: ResplatPacketGenerator,
    local_aligned,
    direct,
) -> tuple[dict[str, object], bool]:
    local_cov = exact.covariance(
        local_aligned.scales.to(generator.device).unsqueeze(0),
        local_aligned.rotations_xyzw.to(generator.device).unsqueeze(0),
    )[0].cpu()
    direct_cov = exact.covariance(
        direct.scales.to(generator.device).unsqueeze(0),
        direct.rotations_xyzw.to(generator.device).unsqueeze(0),
    )[0].cpu()
    metrics = {
        "means": exact.max_and_mean(local_aligned.means - direct.means),
        "covariances": exact.max_and_mean(local_cov - direct_cov),
        "scales": exact.max_and_mean(local_aligned.scales - direct.scales),
        "opacities": exact.max_and_mean(local_aligned.opacities - direct.opacities),
        "harmonics": exact.max_and_mean(local_aligned.harmonics - direct.harmonics),
    }
    checks = {
        "means": metrics["means"]["max"] <= args.max_mean_error,
        "covariances": metrics["covariances"]["max"]
        <= args.max_covariance_error,
        "scales": metrics["scales"]["max"] <= args.max_scale_error,
        "opacities": metrics["opacities"]["max"] <= args.max_opacity_error,
        "harmonics": metrics["harmonics"]["max"] <= args.max_harmonics_error,
    }
    return {"metrics": metrics, "checks": checks}, all(checks.values())


def evaluate_one(
    args: argparse.Namespace,
    generator: ResplatPacketGenerator,
    T_world: torch.Tensor,
    local_pose_index: int,
    original_frame_index: int,
) -> dict[str, object]:
    frame_input = build_frame_input(args, local_pose_index, original_frame_index)

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

    # The exact same local packet is rendered before and after rigid alignment.
    local_camera_render = exact.render_left(generator, local.packet, local.batch)
    aligned_world_render = exact.render_left(generator, local_aligned, direct.batch)
    direct_world_render = exact.render_left(generator, direct.packet, direct.batch)
    gt = direct.batch["context"]["image"][:, 0]

    alignment_similarity = image_similarity(
        local_camera_render, aligned_world_render
    )
    direct_similarity = image_similarity(aligned_world_render, direct_world_render)
    local_gt = render_metrics(gt, aligned_world_render)
    direct_gt = render_metrics(gt, direct_world_render)
    local_camera_gt = render_metrics(
        local.batch["context"]["image"][:, 0], local_camera_render
    )

    gt_delta = {
        "psnr_local_minus_direct": local_gt["psnr"] - direct_gt["psnr"],
        "ssim_local_minus_direct": local_gt["ssim"] - direct_gt["ssim"],
        "psnr_alignment_delta": local_gt["psnr"] - local_camera_gt["psnr"],
        "ssim_alignment_delta": local_gt["ssim"] - local_camera_gt["ssim"],
    }
    alignment_checks = {
        "render_psnr": alignment_similarity["psnr"] >= args.min_alignment_psnr,
        "render_max_abs": alignment_similarity["max_abs"]
        <= args.max_alignment_abs_error,
    }
    quality_checks = {
        "gt_psnr_regression": gt_delta["psnr_local_minus_direct"]
        >= -args.max_gt_psnr_drop,
        "gt_ssim_regression": gt_delta["ssim_local_minus_direct"]
        >= -args.max_gt_ssim_drop,
    }

    parameter_report, parameter_passed = parameter_equivariance(
        args, generator, local_aligned, direct.packet
    )
    exact_checks = dict(parameter_report["checks"])
    exact_checks["render_similarity"] = (
        direct_similarity["psnr"] >= args.min_direct_similarity_psnr
    )
    exact_equivariance_passed = parameter_passed and exact_checks["render_similarity"]
    alignment_passed = all(alignment_checks.values())
    quality_passed = all(quality_checks.values())
    safe_for_async = alignment_passed and quality_passed

    return {
        "local_pose_index": local_pose_index,
        "original_frame_index": original_frame_index,
        "left_image": str(frame_input.descriptor.left_path),
        "right_image": str(frame_input.descriptor.right_path),
        "num_gaussians": local.packet.num_gaussians,
        "alignment_invariance": {
            "similarity": alignment_similarity,
            "local_camera_gt": local_camera_gt,
            "aligned_world_gt": local_gt,
            "checks": alignment_checks,
            "passed": alignment_passed,
        },
        "local_first_vs_direct_world": {
            "render_similarity": direct_similarity,
            "local_first_gt": local_gt,
            "direct_world_gt": direct_gt,
            "gt_delta": gt_delta,
            "quality_checks": quality_checks,
            "quality_passed": quality_passed,
            "parameter_equivariance": parameter_report,
            "exact_checks": exact_checks,
            "exact_equivariance_passed": exact_equivariance_passed,
        },
        "safe_for_async": safe_for_async,
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

    requested = exact.parse_int_ranges(args.pose_indices)
    if not requested:
        raise ValueError("--pose_indices produced no indices")
    for index in requested:
        if not 0 <= index < len(poses):
            raise IndexError(f"pose index {index} outside [0,{len(poses)-1}]")

    generator = ResplatPacketGenerator(
        ResplatRuntimeConfig(
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
    )
    generator.initialize()
    results = [
        evaluate_one(
            args,
            generator,
            torch.from_numpy(poses[local_index]),
            local_index,
            int(original_indices[local_index]),
        )
        for local_index in requested
    ]

    report = {
        "purpose": (
            "separate rigid local-to-world alignment correctness from learned "
            "ReSplat pose-frame sensitivity"
        ),
        "pose_npz": str(Path(args.pose_npz).expanduser().resolve()),
        "pose_indices": requested,
        "input_mode": args.input_mode,
        "thresholds": {
            "min_alignment_psnr": args.min_alignment_psnr,
            "max_alignment_abs_error": args.max_alignment_abs_error,
            "max_gt_psnr_drop": args.max_gt_psnr_drop,
            "max_gt_ssim_drop": args.max_gt_ssim_drop,
            "min_direct_similarity_psnr": args.min_direct_similarity_psnr,
        },
        "interpretation": {
            "safe_for_async": (
                "alignment is render invariant and local-first packet quality does "
                "not regress beyond the configured GT thresholds"
            ),
            "exact_equivariance_passed": (
                "local-first+alignment and direct-world ReSplat runs produce "
                "numerically matching packets; this is diagnostic, not required"
            ),
        },
        "results": results,
        "alignment_passed": all(
            bool(result["alignment_invariance"]["passed"]) for result in results
        ),
        "quality_passed": all(
            bool(result["local_first_vs_direct_world"]["quality_passed"])
            for result in results
        ),
        "exact_equivariance_passed": all(
            bool(
                result["local_first_vs_direct_world"][
                    "exact_equivariance_passed"
                ]
            )
            for result in results
        ),
        "safe_for_async": all(bool(result["safe_for_async"]) for result in results),
    }
    # Keep a conventional top-level field for shell/CI use. Its meaning is now
    # explicitly async safety rather than exact learned-model equivariance.
    report["passed"] = report["safe_for_async"]

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    generator.close()
    if not report["safe_for_async"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
