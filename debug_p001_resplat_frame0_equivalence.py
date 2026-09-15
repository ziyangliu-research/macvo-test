#!/usr/bin/env python3
"""Exact frame-0 P001 ReSplat equivalence diagnostic.

This tool exists to compare the known-good ReSplat-only execution path with the
serial online execution ordering on *the same stereo frame and the same model
instance*.  It does not run the GraphDECO backend.

For frame 0 it performs:

  A) ReSplat inference before MAC-VO process()  (matches ReSplat-only evaluator)
  B) MAC-VO process(frame0), then ReSplat inference on the exact same StereoFrameInput

Before A and B, RNGs are reset to the same seed so any difference cannot be
explained by RNG consumption in MAC-VO.  ReSplat is forced onto the current /
default CUDA stream, matching evaluate_resplat_default_stream_repro.py.

For each result we render the raw ReSplat Gaussians three ways:
  - target_left: exactly the scoring path used by evaluate_resplat_from_execution_baseline.py
  - context_left_isolated: the first context camera rendered alone
  - context_both: both stereo context cameras in one decoder call

The script also reports whether target-left and context-left camera tensors are
actually identical after ReSplat's data_shim, and directly compares the Gaussian
tensors from A and B.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.utils import save_image

import run_async_pipeline as base


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred.float() - gt.float()) ** 2).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).item())


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if tuple(a.shape) != tuple(b.shape):
        return {"shape_equal": 0.0, "max_abs": float("inf"), "mean_abs": float("inf")}
    d = (a - b).abs()
    return {
        "shape_equal": 1.0,
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
    }


def render_diagnostics(packet_generator, result, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = result.batch
    target = batch["target"]
    context = batch["context"]
    model = packet_generator.model
    assert model is not None

    h = int(target["image"].shape[-2])
    w = int(target["image"].shape[-1])

    with torch.inference_mode():
        target_out = model.decoder.forward(
            result.gaussians,
            target["extrinsics"],
            target["intrinsics"],
            target["near"],
            target["far"],
            (h, w),
            depth_mode=None,
        )
        left_out = model.decoder.forward(
            result.gaussians,
            context["extrinsics"][:, 0:1],
            context["intrinsics"][:, 0:1],
            context["near"][:, 0:1],
            context["far"][:, 0:1],
            (h, w),
            depth_mode=None,
        )
        both_out = model.decoder.forward(
            result.gaussians,
            context["extrinsics"],
            context["intrinsics"],
            context["near"],
            context["far"],
            (h, w),
            depth_mode=None,
        )
    torch.cuda.synchronize(packet_generator.device)

    target_pred = target_out.color[0, 0].float().clamp(0, 1)
    target_gt = target["image"][0, 0].float().clamp(0, 1)
    left_pred = left_out.color[0, 0].float().clamp(0, 1)
    left_gt = context["image"][0, 0].float().clamp(0, 1)
    both_left = both_out.color[0, 0].float().clamp(0, 1)
    both_right = both_out.color[0, 1].float().clamp(0, 1)
    right_gt = context["image"][0, 1].float().clamp(0, 1)

    save_image(target_gt.detach().cpu(), out_dir / "gt_target_left.png")
    save_image(target_pred.detach().cpu(), out_dir / "render_target_left.png")
    save_image(left_pred.detach().cpu(), out_dir / "render_context_left_isolated.png")
    save_image(both_left.detach().cpu(), out_dir / "render_context_left_batched.png")
    save_image(right_gt.detach().cpu(), out_dir / "gt_context_right.png")
    save_image(both_right.detach().cpu(), out_dir / "render_context_right_batched.png")

    camera_equivalence = {
        "image": tensor_diff(target["image"], context["image"][:, 0:1]),
        "extrinsics": tensor_diff(target["extrinsics"], context["extrinsics"][:, 0:1]),
        "intrinsics": tensor_diff(target["intrinsics"], context["intrinsics"][:, 0:1]),
        "near": tensor_diff(target["near"], context["near"][:, 0:1]),
        "far": tensor_diff(target["far"], context["far"][:, 0:1]),
    }
    metrics = {
        "num_gaussians": int(result.packet.num_gaussians),
        "target_left_psnr_db": psnr(target_pred, target_gt),
        "context_left_isolated_psnr_db": psnr(left_pred, left_gt),
        "context_left_batched_psnr_db": psnr(both_left, left_gt),
        "context_right_batched_psnr_db": psnr(both_right, right_gt),
        "target_vs_context_left_isolated_render": tensor_diff(target_pred, left_pred),
        "target_vs_context_left_batched_render": tensor_diff(target_pred, both_left),
        "target_vs_context_left_camera_tensors": camera_equivalence,
        "context_extrinsics": context["extrinsics"][0].detach().double().cpu().tolist(),
        "context_intrinsics": context["intrinsics"][0].detach().double().cpu().tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def gaussian_diffs(a, b) -> dict[str, Any]:
    names = ["means", "covariances", "harmonics", "opacities", "scales", "rotations", "rotations_unnorm"]
    out: dict[str, Any] = {}
    for name in names:
        va = getattr(a.gaussians, name, None)
        vb = getattr(b.gaussians, name, None)
        if torch.is_tensor(va) and torch.is_tensor(vb):
            out[name] = tensor_diff(va, vb)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV2_P001_Teaser.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/p001_resplat_frame0_equivalence"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = base.absolute(args.config, root)
    config = base.load_yaml(config_path)
    # This diagnostic needs only the first frame, but keep the normal P001 loader contract.
    config["sequence"]["start_index"] = 0
    config["sequence"]["end_index"] = 1
    resolved = base.resolve(config, root)
    base.validate_paths(resolved)

    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    seed_all(args.seed)
    runner = base.build_system(resolved)
    pose_frontend = runner.pose_frontend
    packet_generator = runner.packet_generator

    pose_frontend.initialize()
    packet_generator.initialize()
    # Match evaluate_resplat_default_stream_repro.py exactly.
    packet_generator.stream = torch.cuda.current_stream(packet_generator.device)

    item = next(iter(pose_frontend.iter_frames()))
    descriptor, frame, stereo_input, _observation = item
    stereo_input.descriptor = descriptor

    # A: exact ReSplat-only ordering (no MAC-VO process()).
    seed_all(args.seed)
    result_before = packet_generator.infer(stereo_input, output_frame="left_camera_local")
    metrics_before = render_diagnostics(packet_generator, result_before, out_root / "A_before_macvo")

    # Now let MAC-VO process the exact same decoded frame, as the serial pipeline does.
    pose_frontend.process(descriptor, frame)

    # B: same input/model/seed after MAC-VO process().
    seed_all(args.seed)
    result_after = packet_generator.infer(stereo_input, output_frame="left_camera_local")
    metrics_after = render_diagnostics(packet_generator, result_after, out_root / "B_after_macvo")

    comparison = {
        "frame_index": int(descriptor.frame_index),
        "protocol_A": "ReSplat inference before MAC-VO process(); evaluator-equivalent ordering",
        "protocol_B": "MAC-VO process(frame0) then ReSplat inference; serial-online ordering",
        "same_rng_seed_before_each_inference": int(args.seed),
        "A": metrics_before,
        "B": metrics_after,
        "gaussian_tensor_diffs_A_vs_B": gaussian_diffs(result_before, result_after),
    }
    (out_root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n===== P001 frame-0 ReSplat equivalence =====")
    print(
        f"A before MAC-VO: target={metrics_before['target_left_psnr_db']:.3f} dB | "
        f"ctx-left-isolated={metrics_before['context_left_isolated_psnr_db']:.3f} dB | "
        f"ctx-left-batched={metrics_before['context_left_batched_psnr_db']:.3f} dB | "
        f"ctx-right={metrics_before['context_right_batched_psnr_db']:.3f} dB"
    )
    print(
        f"B after  MAC-VO: target={metrics_after['target_left_psnr_db']:.3f} dB | "
        f"ctx-left-isolated={metrics_after['context_left_isolated_psnr_db']:.3f} dB | "
        f"ctx-left-batched={metrics_after['context_left_batched_psnr_db']:.3f} dB | "
        f"ctx-right={metrics_after['context_right_batched_psnr_db']:.3f} dB"
    )
    print(f"saved: {out_root / 'comparison.json'}")

    packet_generator.close()
    # MAC-VO may hold CUDA state; terminate cleanly after the comparison.
    pose_frontend.terminate()


if __name__ == "__main__":
    main()
