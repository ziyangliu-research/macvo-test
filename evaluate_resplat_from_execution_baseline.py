#!/usr/bin/env python3
"""Evaluate independent per-frame ReSplat packets using the corrected baseline path.

This diagnostic reuses the same system construction and frame loader as
`run_pipeline_execution_benchmark.py`, but intentionally does not run MAC-VO pose
estimation and does not initialize/run the GraphDECO backend:

    build_system(resolved)
      -> pose_frontend.iter_frames()        # frame loading only
      -> packet_generator.infer(..., output_frame="left_camera_local")
      -> ReSplat decoder at the same left target view
      -> PSNR / SSIM

Each timestamp is independent: no packet fusion, no world alignment, no opacity
reset, and no incremental optimization. The baseline ResplatPacketGenerator now
contains the per-batch CUDA-stream synchronization fix, so no diagnostic stream
override is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import run_async_pipeline as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Config/Pipeline/MACVO_ReSplat_Serial_Report_P000_0_50.yaml",
        help="Pipeline YAML providing dataset/camera/ReSplat settings.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override an existing pipeline key using the standard runner syntax.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <work_dir>/resplat_only_from_execution_baseline.",
    )
    parser.add_argument(
        "--include_test_frames",
        action="store_true",
        help="Also evaluate frames marked test by the pipeline split.",
    )
    parser.add_argument(
        "--save_gt",
        action="store_true",
        help="Save the exact post-shim left target image used for scoring.",
    )
    parser.add_argument(
        "--save_baseline_input",
        action="store_true",
        help="Save the exact left/right tensors yielded by pose_frontend.iter_frames().",
    )
    return parser.parse_args()


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    image = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def seed_all() -> int:
    seed = int(os.environ.get("PIPELINE_BENCHMARK_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed


def main() -> None:
    args = parse_args()
    seed = seed_all()

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

    work_dir = Path(resolved["paths"]["work_dir"]).expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else work_dir / "resplat_only_from_execution_baseline"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    template = base.build_system(resolved)
    pose_frontend = template.pose_frontend
    packet_generator = template.packet_generator

    # initialize() is required because iter_frames() owns the exact dataset loader.
    # process() is never called, so MAC-VO pose estimation is not executed.
    init_start = time.perf_counter()
    pose_frontend.initialize()
    packet_generator.initialize()
    initialization_sec = time.perf_counter() - init_start

    resplat_repo = Path(resolved["paths"]["resplat_repo"])
    if str(resplat_repo) not in sys.path:
        sys.path.insert(0, str(resplat_repo))
    from src.evaluation.metrics import compute_psnr, compute_ssim

    rows: list[dict[str, Any]] = []
    skipped_test = 0
    first_debug: dict[str, Any] | None = None
    total_start = time.perf_counter()

    for descriptor, _frame, stereo_input, _observation in pose_frontend.iter_frames():
        descriptor = replace(descriptor, is_test=template.config.is_test(descriptor))
        stereo_input.descriptor = descriptor

        if descriptor.is_test and not args.include_test_frames:
            skipped_test += 1
            continue

        if args.save_baseline_input:
            save_rgb(
                stereo_input.left_image,
                output_dir / "baseline_input_left" / f"{descriptor.frame_index:06d}.png",
            )
            save_rgb(
                stereo_input.right_image,
                output_dir / "baseline_input_right" / f"{descriptor.frame_index:06d}.png",
            )

        result = packet_generator.infer(
            stereo_input,
            output_frame="left_camera_local",
        )

        batch = result.batch
        target = batch["target"]
        h = int(target["image"].shape[-2])
        w = int(target["image"].shape[-1])

        render_start = time.perf_counter()
        with torch.inference_mode():
            render_out = packet_generator.model.decoder.forward(
                result.gaussians,
                target["extrinsics"],
                target["intrinsics"],
                target["near"],
                target["far"],
                (h, w),
                depth_mode=None,
            )
        torch.cuda.synchronize(packet_generator.device)
        render_sec = time.perf_counter() - render_start

        pred_all = render_out.color.float().clamp(0.0, 1.0)
        gt_all = target["image"].float().clamp(0.0, 1.0)
        pred_flat = pred_all.reshape(-1, *pred_all.shape[-3:])
        gt_flat = gt_all.reshape(-1, *gt_all.shape[-3:])

        psnr = float(compute_psnr(gt_flat, pred_flat).mean().item())
        ssim = float(compute_ssim(gt_flat, pred_flat).mean().item())

        pred = pred_all[0, 0]
        gt = gt_all[0, 0]
        save_rgb(pred, output_dir / "renders" / f"{descriptor.frame_index:06d}.png")
        if args.save_gt:
            save_rgb(gt, output_dir / "gt_post_shim" / f"{descriptor.frame_index:06d}.png")

        row = {
            "sequence_index": int(descriptor.sequence_index),
            "frame_index": int(descriptor.frame_index),
            "is_test": bool(descriptor.is_test),
            "num_gaussians": int(result.packet.num_gaussians),
            "psnr": psnr,
            "ssim": ssim,
            "resplat_inference_sec": float(result.inference_sec),
            "decoder_render_sec": float(render_sec),
            "input_h": int(stereo_input.left_image.shape[-2]),
            "input_w": int(stereo_input.left_image.shape[-1]),
            "target_h": h,
            "target_w": w,
            "baseline_m": float(stereo_input.baseline_m),
        }
        rows.append(row)

        if first_debug is None:
            first_debug = {
                "frame_index": int(descriptor.frame_index),
                "baseline_input_shape": list(stereo_input.left_image.shape),
                "baseline_intrinsic_pixel": stereo_input.intrinsic_pixel.detach().cpu().tolist(),
                "baseline_m": float(stereo_input.baseline_m),
                "post_shim_context_image_shape": list(batch["context"]["image"].shape),
                "post_shim_target_image_shape": list(batch["target"]["image"].shape),
                "post_shim_context_intrinsics": batch["context"]["intrinsics"].detach().cpu().tolist(),
                "post_shim_context_extrinsics": batch["context"]["extrinsics"].detach().cpu().tolist(),
                "resplat_image_shape": list(packet_generator.image_shape),
                "resplat_near": float(packet_generator.near),
                "resplat_far": float(packet_generator.far),
                "resplat_refine_steps": int(packet_generator.config.refine_steps),
                "resplat_input_mode": str(packet_generator.config.input_mode),
                "cuda_stream_fix_expected": True,
            }
            (output_dir / "first_frame_debug.json").write_text(
                json.dumps(first_debug, indent=2), encoding="utf-8"
            )

        print(
            f"[ReSplat-only {len(rows):03d}] "
            f"frame={descriptor.frame_index:04d} "
            f"G={result.packet.num_gaussians:,} PSNR={psnr:.2f} SSIM={ssim:.4f} "
            f"infer={result.inference_sec:.3f}s render={render_sec:.3f}s",
            flush=True,
        )

        del render_out, result, batch, target, pred_all, gt_all, pred_flat, gt_flat

    total_sec = time.perf_counter() - total_start
    packet_generator.close()

    if not rows:
        raise RuntimeError("No frames were evaluated")

    summary = {
        "protocol": (
            "Independent per-frame ReSplat packets from the corrected execution-baseline "
            "loader/runtime; no MAC-VO process(), no pose alignment, no fusion, no backend."
        ),
        "seed": seed,
        "config": str(config_path),
        "include_test_frames": bool(args.include_test_frames),
        "num_evaluated_frames": len(rows),
        "num_skipped_test_frames": skipped_test,
        "average_psnr": float(np.mean([r["psnr"] for r in rows])),
        "average_ssim": float(np.mean([r["ssim"] for r in rows])),
        "average_num_gaussians": float(np.mean([r["num_gaussians"] for r in rows])),
        "average_resplat_inference_sec": float(
            np.mean([r["resplat_inference_sec"] for r in rows])
        ),
        "average_decoder_render_sec": float(
            np.mean([r["decoder_render_sec"] for r in rows])
        ),
        "initialization_sec": initialization_sec,
        "total_eval_sec": total_sec,
        "first_frame_debug": first_debug,
    }
    write_csv(output_dir / "per_frame_metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n===== ReSplat-only summary =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
