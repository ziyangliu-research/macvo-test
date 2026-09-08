#!/usr/bin/env python3
"""Isolate ReSplat quality on EuRoC before any pose/fusion/3DGS backend effects.

For selected EuRoC stereo timestamps this script:
  1. loads the same synchronized, sensor.yaml-calibrated, rectified images used by
     the formal EuRoC pipeline;
  2. applies the same EuRoC-specific full-FoV ReSplat resize and dynamic K path;
  3. runs ReSplat once on the stereo pair (refine_steps=0);
  4. renders the produced local Gaussian packet back into BOTH context cameras
     using ReSplat's native decoder;
  5. reports PSNR/SSIM/LPIPS against the corresponding rectified input images.

There is deliberately NO MAC-VO pose estimation, global packet fusion, opacity
reset, GraphDECO optimization, pruning, historical replay, or global refinement.
Thus this is a feed-forward local-packet diagnostic, not an NVS benchmark.

Default sampling follows the formal temporal protocol only for frame selection:
valid synchronized/GT-covered EuRoC frames -> stride=5 -> uniformly pick 8
retained timestamps across the sequence. The strict 8:2 split is irrelevant here
because each tested stereo pair is evaluated only against its own two context
views.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from DataLoader import SequenceBase, StereoFrame
from Utility.Config import load_config
from async_pipeline.contracts import FrameDescriptor, StereoFrameInput
from async_pipeline.resplat_runtime import ResplatPacketGenerator, ResplatRuntimeConfig
from euroc_stride5_support import install_euroc_runtime_support


ROOT = Path(__file__).resolve().parent
SEQUENCE_CONFIGS = {
    "MH02": ROOT / "Config/Sequence/EuRoC_MH02_local.yaml",
    "V101": ROOT / "Config/Sequence/EuRoC_V101_local.yaml",
    "V201": ROOT / "Config/Sequence/EuRoC_V201_local.yaml",
    "MH05": ROOT / "Config/Sequence/EuRoC_MH05_local.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", choices=sorted(SEQUENCE_CONFIGS), default="MH02")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument(
        "--source_indices",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional explicit indices in the valid synchronized/GT-covered EuRoC "
            "sequence. If supplied, --num_samples is ignored."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: outputs/euroc_resplat_packet_diag/<sequence>",
    )
    return p.parse_args()


def select_indices(length: int, stride: int, num_samples: int) -> list[int]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    retained = list(range(0, length, stride))
    if not retained:
        raise RuntimeError("no retained EuRoC frames")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    n = min(num_samples, len(retained))
    positions = np.linspace(0, len(retained) - 1, n, dtype=np.int64)
    return [retained[int(i)] for i in positions]


def make_generator(device: str, output_dir: Path) -> ResplatPacketGenerator:
    # Match the formal EuRoC online experiment exactly on the ReSplat side.
    # The experiment itself supplies the trained checkpoint through Hydra.
    cfg = ResplatRuntimeConfig(
        repo=(ROOT / "../Resplat").resolve(),
        experiment="tartanair_p000_ft",
        device=device,
        checkpoint=None,
        overrides=("dataset.image_shape=[256,384]",),
        output_dir=output_dir / "resplat_runtime",
        fx=320.0,  # ignored by the EuRoC dynamic-K shared-tensor patch
        fy=320.0,
        cx=320.0,
        cy=240.0,
        stereo_baseline=0.1100778422,
        refine_steps=0,
        refine_use_target=False,
        deterministic=False,
        pin_output_memory=False,
        input_mode="shared_tensors",
        handoff_mode="gpu",
        strict_validation=False,
    )
    return ResplatPacketGenerator(cfg)


def to_float_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1)]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ReSplat")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / "outputs/euroc_resplat_packet_diag" / args.sequence
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Installs the same EuRoC-specific full-FoV resize + dynamic rectified-K path
    # used by the formal quality/timing runners.
    os.environ["PIPELINE_FRAME_STRIDE"] = str(args.stride)
    install_euroc_runtime_support()

    data_cfg_path = SEQUENCE_CONFIGS[args.sequence]
    data_cfg, _ = load_config(data_cfg_path)
    sequence = SequenceBase[StereoFrame].instantiate(data_cfg.type, data_cfg.args)
    if str(data_cfg.type) != "EuRoC_NoIMU":
        raise ValueError(f"expected EuRoC_NoIMU, got {data_cfg.type}")

    if args.source_indices is not None and len(args.source_indices) > 0:
        indices = list(args.source_indices)
        for idx in indices:
            if idx < 0 or idx >= len(sequence):
                raise IndexError(f"source index {idx} outside [0,{len(sequence)})")
    else:
        indices = select_indices(len(sequence), args.stride, args.num_samples)

    generator = make_generator(args.device, output_dir)
    generator.initialize()
    assert generator.model is not None

    # Use ReSplat's own metric implementation to avoid metric-definition drift.
    resplat_repo = generator.repo
    if str(resplat_repo) not in sys.path:
        sys.path.insert(0, str(resplat_repo))
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    rows: list[dict[str, float | int]] = []

    print("\n=== EuRoC standalone ReSplat packet diagnostic ===", flush=True)
    print(f"sequence          : {args.sequence}", flush=True)
    print(f"valid frames      : {len(sequence)}", flush=True)
    print(f"selection stride  : {args.stride}", flush=True)
    print(f"sample indices    : {indices}", flush=True)
    print("input domain      : rectified EuRoC stereo", flush=True)
    print("ReSplat resize    : full-FoV 752x480 -> 384x256", flush=True)
    print("backend/fusion    : NONE", flush=True)
    print("evaluation        : ReSplat native decoder, same stereo context views\n", flush=True)

    with torch.inference_mode():
        for sample_ordinal, source_index in enumerate(indices):
            frame = sequence[source_index]
            timestamp_ns = int(frame.stereo.frame_ns)

            # Paths are metadata only in shared_tensors mode. Use the actual
            # rectified-loader filenames where available for traceability.
            left_path = Path(sequence.ImageL.file_names[source_index])
            right_path = Path(sequence.ImageR.file_names[source_index])
            descriptor = FrameDescriptor(
                sequence_index=sample_ordinal,
                frame_index=source_index,
                timestamp_ns=timestamp_ns,
                left_path=left_path,
                right_path=right_path,
                is_test=False,
            )
            frame_input = StereoFrameInput(
                descriptor=descriptor,
                left_image=frame.stereo.imageL[0].detach().cpu().float().contiguous(),
                right_image=frame.stereo.imageR[0].detach().cpu().float().contiguous(),
                intrinsic_pixel=frame.stereo.frame_K.detach().cpu().float().contiguous(),
                baseline_m=float(frame.stereo.frame_baseline),
            )
            frame_input.validate(deep=True)

            result = generator.infer(
                frame_input,
                output_frame="left_camera_local",
                keep_gpu_packet=True,
            )
            context = result.batch["context"]
            target = context["image"][0].clamp(0.0, 1.0)  # [2,3,H,W]
            h, w = target.shape[-2:]

            decoded = generator.model.decoder.forward(
                result.gaussians,
                context["extrinsics"],
                context["intrinsics"],
                context["near"],
                context["far"],
                (h, w),
                depth_mode=None,
            )
            rendered = decoded.color[0].clamp(0.0, 1.0)  # [2,3,H,W]

            psnr = compute_psnr(target, rendered)
            ssim = compute_ssim(target, rendered)
            lpips = compute_lpips(target, rendered)
            p = to_float_list(psnr)
            s = to_float_list(ssim)
            l = to_float_list(lpips)

            row = {
                "sample_ordinal": sample_ordinal,
                "source_index": source_index,
                "timestamp_ns": timestamp_ns,
                "num_gaussians": int(result.packet.num_gaussians),
                "inference_sec": float(result.inference_sec),
                "left_psnr": p[0],
                "left_ssim": s[0],
                "left_lpips": l[0],
                "right_psnr": p[1],
                "right_ssim": s[1],
                "right_lpips": l[1],
                "stereo_mean_psnr": float((psnr.mean()).item()),
                "stereo_mean_ssim": float((ssim.mean()).item()),
                "stereo_mean_lpips": float((lpips.mean()).item()),
            }
            rows.append(row)

            frame_dir = output_dir / f"frame_{source_index:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            save_image(target[0], frame_dir / "left_gt.png")
            save_image(rendered[0], frame_dir / "left_render.png")
            save_image(target[1], frame_dir / "right_gt.png")
            save_image(rendered[1], frame_dir / "right_render.png")
            # One image for quick visual inspection: Lgt | Lrender | Rgt | Rrender.
            comparison = torch.cat(
                [target[0], rendered[0], target[1], rendered[1]], dim=-1
            )
            save_image(comparison, frame_dir / "comparison.png")

            print(
                f"idx={source_index:4d} G={row['num_gaussians']:7d} "
                f"infer={row['inference_sec']:.3f}s | "
                f"L {p[0]:.3f}/{s[0]:.4f}/{l[0]:.4f} | "
                f"R {p[1]:.3f}/{s[1]:.4f}/{l[1]:.4f}",
                flush=True,
            )

    generator.close()

    metric_keys = [
        "left_psnr", "left_ssim", "left_lpips",
        "right_psnr", "right_ssim", "right_lpips",
        "stereo_mean_psnr", "stereo_mean_ssim", "stereo_mean_lpips",
        "inference_sec",
    ]
    mean = {
        key: float(np.mean([float(row[key]) for row in rows])) for key in metric_keys
    }
    summary = {
        "sequence": args.sequence,
        "protocol": (
            "standalone ReSplat local packet; rectified EuRoC stereo; full-FoV "
            "384x256; native ReSplat decoder; context-view reconstruction only"
        ),
        "formal_resplat_experiment": "tartanair_p000_ft",
        "refine_steps": 0,
        "stride_for_sample_selection": args.stride,
        "valid_sequence_frames": len(sequence),
        "sample_source_indices": indices,
        "num_samples": len(rows),
        "mean": mean,
        "frames": rows,
    }

    json_path = output_dir / "packet_quality_summary.json"
    csv_path = output_dir / "packet_quality_frames.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Mean standalone packet quality ===", flush=True)
    print(
        "Left   P/S/L: "
        f"{mean['left_psnr']:.3f}/{mean['left_ssim']:.4f}/{mean['left_lpips']:.4f}",
        flush=True,
    )
    print(
        "Right  P/S/L: "
        f"{mean['right_psnr']:.3f}/{mean['right_ssim']:.4f}/{mean['right_lpips']:.4f}",
        flush=True,
    )
    print(
        "Stereo P/S/L: "
        f"{mean['stereo_mean_psnr']:.3f}/{mean['stereo_mean_ssim']:.4f}/"
        f"{mean['stereo_mean_lpips']:.4f}",
        flush=True,
    )
    print(f"Mean inference: {mean['inference_sec']:.3f}s", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"CSV : {csv_path}", flush=True)


if __name__ == "__main__":
    main()
