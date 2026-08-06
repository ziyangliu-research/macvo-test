#!/usr/bin/env python3
"""Diagnose local-first ReSplat alignment and gate the final 0-50 A/B test.

The script reports three distinct properties instead of collapsing them into one
boolean:

1. strict_alignment_passed
   Pixel-level invariance of the same packet before/after a global rigid transform.
   CUDA rasterizers are not guaranteed to satisfy this at a 60 dB / max-pixel
   threshold after large world-coordinate changes, so this remains diagnostic.

2. practical_alignment_passed
   The same packet's PSNR/SSIM against the same GT image changes by less than the
   configured engineering tolerances after alignment.

3. quality_passed
   Local-first ReSplat followed by alignment does not regress materially against
   the current direct-world ReSplat path.

Passing this script means the implementation is a candidate for the controlled
full incremental 0-50 comparison. It does not establish exact SE(3) equivariance
and does not by itself approve a final research result.
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

    # Strict pixel-level diagnostic. These values intentionally remain severe.
    parser.add_argument("--min_alignment_psnr", type=float, default=60.0)
    parser.add_argument("--max_alignment_abs_error", type=float, default=5e-3)

    # Practical same-packet stability after changing the global coordinate frame.
    parser.add_argument("--max_alignment_gt_psnr_delta", type=float, default=0.25)
    parser.add_argument("--max_alignment_gt_ssim_delta", type=float, default=0.02)

    # Local-first versus the existing direct-world packet-generation path.
    parser.add_argument("--max_gt_psnr_drop", type=float, default=0.25)
    parser.add_argument("--max_gt_ssim_drop", type=float, default=0.02)

    # Exact learned-model equivariance remains diagnostic only.
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

    local_camera_render = exact.render_left(generator, local.packet, local.batch)
    aligned_world_render = exact.render_left(generator, local_aligned, direct.batch)
    direct_world_render = exact.render_left(generator, direct.packet, direct.batch)
    gt = direct.batch["context"]["image"][:, 0]

    alignment_similarity = image_similarity(local_camera_render, aligned_world_render)
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

    strict_alignment_checks = {
        "render_psnr": alignment_similarity["psnr"] >= args.min_alignment_psnr,
        "render_max_abs": alignment_similarity["max_abs"]
        <= args.max_alignment_abs_error,
    }
    practical_alignment_checks = {
        "gt_psnr_stability": abs(gt_delta["psnr_alignment_delta"])
        <= args.max_alignment_gt_psnr_delta,
        "gt_ssim_stability": abs(gt_delta["ssim_alignment_delta"])
        <= args.max_alignment_gt_ssim_delta,
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

    strict_alignment_passed = all(strict_alignment_checks.values())
    practical_alignment_passed = all(practical_alignment_checks.values())
    quality_passed = all(quality_checks.values())
    exact_equivariance_passed = parameter_passed and exact_checks["render_similarity"]
    ready_for_final_ab_test = practical_alignment_passed and quality_passed

    return {
        "local_pose_index": local_pose_index,
        "original_frame_index": original_frame_index,
        "left_image": str(frame_input.descriptor.left_path),
        "right_image": str(frame_input.descriptor.right_path),
        "num_gaussians": local.packet.num_gaussians,
        "alignment": {
            "similarity": alignment_similarity,
            "local_camera_gt": local_camera_gt,
            "aligned_world_gt": local_gt,
            "gt_delta": {
                "psnr": gt_delta["psnr_alignment_delta"],
                "ssim": gt_delta["ssim_alignment_delta"],
            },
            "strict_checks": strict_alignment_checks,
            "strict_passed": strict_alignment_passed,
            "practical_checks": practical_alignment_checks,
            "practical_passed": practical_alignment_passed,
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
        "ready_for_final_ab_test": ready_for_final_ab_test,
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

    strict_alignment_passed = all(
        bool(result["alignment"]["strict_passed"]) for result in results
    )
    practical_alignment_passed = all(
        bool(result["alignment"]["practical_passed"]) for result in results
    )
    quality_passed = all(
        bool(result["local_first_vs_direct_world"]["quality_passed"])
        for result in results
    )
    exact_equivariance_passed = all(
        bool(
            result["local_first_vs_direct_world"]["exact_equivariance_passed"]
        )
        for result in results
    )
    ready_for_final_ab_test = practical_alignment_passed and quality_passed

    psnr_direct_deltas = [
        float(result["local_first_vs_direct_world"]["gt_delta"]["psnr_local_minus_direct"])
        for result in results
    ]
    ssim_direct_deltas = [
        float(result["local_first_vs_direct_world"]["gt_delta"]["ssim_local_minus_direct"])
        for result in results
    ]
    psnr_alignment_deltas = [
        abs(float(result["alignment"]["gt_delta"]["psnr"])) for result in results
    ]
    ssim_alignment_deltas = [
        abs(float(result["alignment"]["gt_delta"]["ssim"])) for result in results
    ]

    report = {
        "purpose": (
            "separate strict rasterizer invariance from the practical gate for "
            "the controlled full incremental 0-50 comparison"
        ),
        "pose_npz": str(Path(args.pose_npz).expanduser().resolve()),
        "pose_indices": requested,
        "input_mode": args.input_mode,
        "thresholds": {
            "min_alignment_psnr": args.min_alignment_psnr,
            "max_alignment_abs_error": args.max_alignment_abs_error,
            "max_alignment_gt_psnr_delta": args.max_alignment_gt_psnr_delta,
            "max_alignment_gt_ssim_delta": args.max_alignment_gt_ssim_delta,
            "max_gt_psnr_drop": args.max_gt_psnr_drop,
            "max_gt_ssim_drop": args.max_gt_ssim_drop,
            "min_direct_similarity_psnr": args.min_direct_similarity_psnr,
        },
        "aggregate_worst_case": {
            "local_first_vs_direct_psnr_drop": max(0.0, -min(psnr_direct_deltas)),
            "local_first_vs_direct_ssim_drop": max(0.0, -min(ssim_direct_deltas)),
            "same_packet_alignment_abs_psnr_delta": max(psnr_alignment_deltas),
            "same_packet_alignment_abs_ssim_delta": max(ssim_alignment_deltas),
        },
        "results": results,
        "strict_alignment_passed": strict_alignment_passed,
        "practical_alignment_passed": practical_alignment_passed,
        "quality_passed": quality_passed,
        "exact_equivariance_passed": exact_equivariance_passed,
        "ready_for_final_ab_test": ready_for_final_ab_test,
        "passed": ready_for_final_ab_test,
        "interpretation": {
            "strict_alignment_passed": (
                "same packet is pixel-level invariant under the strict CUDA "
                "rasterizer threshold; diagnostic only"
            ),
            "ready_for_final_ab_test": (
                "sampled local-first packets are stable enough to justify the "
                "full 0-50 serial-versus-async incremental comparison"
            ),
            "final_acceptance": (
                "must be decided from the complete incremental metric curves, "
                "final test metrics, map size, and wall-clock timing"
            ),
        },
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    generator.close()
    if not ready_for_final_ab_test:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
