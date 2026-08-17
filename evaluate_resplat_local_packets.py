#!/usr/bin/env python3
"""Evaluate ReSplat local packets independently of MAC-VO and the 3DGS backend.

For every selected stereo timestamp this script:
  1. reads the left/right images directly from the configured sequence;
  2. runs the same persistent ReSplat runtime/configuration used by the pipeline;
  3. keeps the packet in the current left-camera-local coordinate frame;
  4. renders that packet at the left target pose using ReSplat's own decoder;
  5. saves the rendered image and computes per-frame PSNR/SSIM (optional LPIPS).

No MAC-VO pose estimation, opacity reset, global fusion, GraphDECO optimization,
or map maintenance is involved. This is intended to isolate raw ReSplat packet
quality from downstream fusion/backend effects.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

import run_async_pipeline as base
from async_pipeline.contracts import FrameDescriptor, StereoFrameInput
from async_pipeline.resplat_runtime import (
    ResplatPacketGenerator,
    ResplatRuntimeConfig,
    build_pixel_intrinsic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "Config/Pipeline/"
            "MACVO_ReSplat_Serial_TartanAirV1_SH000_0_200_AllFrames.yaml"
        ),
        help="Pipeline YAML whose camera/ReSplat settings should be reproduced.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override an existing YAML key using the same syntax as pipeline runners.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <work_dir>/resplat_local_eval.",
    )
    parser.add_argument(
        "--with_lpips",
        action="store_true",
        help="Also compute LPIPS using ReSplat's evaluation implementation.",
    )
    parser.add_argument(
        "--save_packets",
        action="store_true",
        help="Save each raw local Gaussian packet as a .pt file (large output).",
    )
    parser.add_argument(
        "--save_gt",
        action="store_true",
        help="Also save the preprocessed left target image used for scoring.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB")).contiguous()


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


def save_packet(packet: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frame_index": int(packet.descriptor.frame_index),
            "sequence_index": int(packet.descriptor.sequence_index),
            "means": packet.means.detach().cpu().contiguous(),
            "scales": packet.scales.detach().cpu().contiguous(),
            "rotations_xyzw": packet.rotations_xyzw.detach().cpu().contiguous(),
            "harmonics": packet.harmonics.detach().cpu().contiguous(),
            "opacities": packet.opacities.detach().cpu().contiguous(),
            "context_intrinsics": packet.context_intrinsics.detach().cpu().contiguous(),
            "context_extrinsics": packet.context_extrinsics.detach().cpu().contiguous(),
            "coordinate_frame": packet.coordinate_frame,
            "metadata": dict(packet.metadata),
        },
        path,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = base.absolute(args.config, root)
    config = base.load_yaml(config_path)
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}")
        key, raw = item.split("=", 1)
        base.nested_set(config, key, base.normalize_override_value(key, raw))
    resolved = base.resolve(config, root)

    paths = resolved["paths"]
    sequence = resolved["sequence"]
    camera = resolved["camera"]
    resplat_cfg = resolved["resplat_frontend"]
    runtime = resolved["runtime"]

    resplat_repo = Path(paths["resplat_repo"])
    data_config_path = Path(paths["data_config"])
    if not resplat_repo.is_dir():
        raise NotADirectoryError(f"ReSplat repository not found: {resplat_repo}")
    if not data_config_path.is_file():
        raise FileNotFoundError(f"sequence config not found: {data_config_path}")

    data_cfg = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    dataset_root = Path(data_cfg["args"]["root"]).expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path(paths["work_dir"]) / "resplat_local_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ResplatPacketGenerator(
        ResplatRuntimeConfig(
            repo=resplat_repo,
            experiment=resplat_cfg["experiment"],
            device=resplat_cfg["device"],
            checkpoint=(
                None
                if not paths.get("resplat_checkpoint")
                else Path(paths["resplat_checkpoint"])
            ),
            overrides=tuple(resplat_cfg.get("overrides", [])),
            output_dir=output_dir / "runtime",
            fx=float(camera["fx"]),
            fy=float(camera["fy"]),
            cx=float(camera["cx"]),
            cy=float(camera["cy"]),
            stereo_baseline=float(camera["stereo_baseline"]),
            refine_steps=int(resplat_cfg["refine_steps"]),
            refine_use_target=bool(resplat_cfg["refine_use_target"]),
            deterministic=bool(resplat_cfg["deterministic"]),
            pin_output_memory=bool(runtime["pin_packet_memory"]),
            input_mode=resplat_cfg["input_mode"],
            handoff_mode=resplat_cfg["handoff_mode"],
            strict_validation=bool(resplat_cfg["strict_validation"]),
        )
    )
    generator.initialize()
    assert generator.model is not None

    if str(resplat_repo) not in sys.path:
        sys.path.insert(0, str(resplat_repo))
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    K = build_pixel_intrinsic(
        float(camera["fx"]),
        float(camera["fy"]),
        float(camera["cx"]),
        float(camera["cy"]),
    )
    start_index = int(sequence["start_index"])
    end_index = int(sequence["end_index"])
    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for sequence_index, frame_index in enumerate(range(start_index, end_index)):
        left_path = (
            dataset_root
            / camera["left_subdir"]
            / camera["left_pattern"].format(index=frame_index)
        )
        right_path = (
            dataset_root
            / camera["right_subdir"]
            / camera["right_pattern"].format(index=frame_index)
        )
        left = load_rgb(left_path)
        right = load_rgb(right_path)
        descriptor = FrameDescriptor(
            sequence_index=sequence_index,
            frame_index=frame_index,
            timestamp_ns=frame_index,
            left_path=left_path,
            right_path=right_path,
            is_test=False,
        )
        frame_input = StereoFrameInput(
            descriptor=descriptor,
            left_image=left,
            right_image=right,
            intrinsic_pixel=K.clone(),
            baseline_m=float(camera["stereo_baseline"]),
        )

        result = generator.infer(
            frame_input,
            output_frame="left_camera_local",
            keep_gpu_packet=False,
        )
        batch = result.batch
        target = batch["target"]
        h, w = int(target["image"].shape[-2]), int(target["image"].shape[-1])

        render_start = time.perf_counter()
        with torch.inference_mode():
            output = generator.model.decoder.forward(
                result.gaussians,
                target["extrinsics"],
                target["intrinsics"],
                target["near"],
                target["far"],
                (h, w),
                depth_mode=None,
            )
        torch.cuda.current_stream(generator.device).synchronize()
        render_sec = time.perf_counter() - render_start

        pred = output.color[0, 0].float().clamp(0.0, 1.0)
        gt = target["image"][0, 0].float().clamp(0.0, 1.0)
        pred_b = pred.unsqueeze(0)
        gt_b = gt.unsqueeze(0)
        psnr = float(compute_psnr(gt_b, pred_b).mean().item())
        ssim = float(compute_ssim(gt_b, pred_b).mean().item())
        lpips_value = (
            float(compute_lpips(gt_b, pred_b).mean().item())
            if args.with_lpips
            else None
        )

        packet = result.packet
        opacity = packet.opacities.detach().float().reshape(-1)
        scales = packet.scales.detach().float()
        row = {
            "sequence_index": sequence_index,
            "frame_index": frame_index,
            "num_gaussians": int(packet.num_gaussians),
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_value,
            "resplat_inference_sec": float(result.inference_sec),
            "decoder_render_sec": float(render_sec),
            "opacity_mean": float(opacity.mean().item()),
            "opacity_max": float(opacity.max().item()),
            "scale_mean": float(scales.mean().item()),
            "scale_max": float(scales.max().item()),
        }
        rows.append(row)

        save_rgb(pred, output_dir / "renders" / f"{frame_index:06d}.png")
        if args.save_gt:
            save_rgb(gt, output_dir / "gt" / f"{frame_index:06d}.png")
        if args.save_packets:
            save_packet(packet, output_dir / "packets" / f"{frame_index:06d}.pt")

        print(
            f"[ReSplat local {sequence_index + 1:03d}/{end_index - start_index:03d}] "
            f"frame={frame_index:04d} G={packet.num_gaussians:,} "
            f"PSNR={psnr:.2f} SSIM={ssim:.4f} "
            f"infer={result.inference_sec:.3f}s render={render_sec:.3f}s",
            flush=True,
        )

        del output, result, batch, target, pred, gt, pred_b, gt_b

    total_sec = time.perf_counter() - total_start
    psnr_values = [float(row["psnr"]) for row in rows]
    ssim_values = [float(row["ssim"]) for row in rows]
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "frame_range": [start_index, end_index - 1],
        "num_frames": len(rows),
        "protocol": (
            "Independent per-timestamp ReSplat stereo packet; render current left "
            "target pose with ReSplat decoder; no MAC-VO/global fusion/opacity reset/backend."
        ),
        "resplat_experiment": resplat_cfg["experiment"],
        "resplat_overrides": list(resplat_cfg.get("overrides", [])),
        "refine_steps": int(resplat_cfg["refine_steps"]),
        "camera": {
            "fx": float(camera["fx"]),
            "fy": float(camera["fy"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
            "baseline": float(camera["stereo_baseline"]),
        },
        "average_psnr": sum(psnr_values) / len(psnr_values),
        "average_ssim": sum(ssim_values) / len(ssim_values),
        "average_lpips": (
            sum(float(row["lpips"]) for row in rows) / len(rows)
            if args.with_lpips
            else None
        ),
        "average_num_gaussians": (
            sum(int(row["num_gaussians"]) for row in rows) / len(rows)
        ),
        "average_resplat_inference_sec": (
            sum(float(row["resplat_inference_sec"]) for row in rows) / len(rows)
        ),
        "average_decoder_render_sec": (
            sum(float(row["decoder_render_sec"]) for row in rows) / len(rows)
        ),
        "total_sec": total_sec,
    }
    write_csv(output_dir / "per_frame_metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n===== ReSplat local-packet summary =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    generator.close()


if __name__ == "__main__":
    main()
