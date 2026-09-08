#!/usr/bin/env python3
"""Isolate ReSplat local-packet quality on TartanAir Challenge sequences.

For selected stereo timestamps this script:
  1. loads the same TartanAir stereo images/intrinsics used by the formal pipeline;
  2. applies the same ReSplat preprocessing (640x480 -> 320x240 full FoV);
  3. runs ReSplat once with refine_steps=0;
  4. renders the predicted local Gaussians back into BOTH stereo context cameras
     using ReSplat's native decoder;
  5. reports PSNR/SSIM/LPIPS against those same context images.

There is NO MAC-VO, world alignment, fusion, GraphDECO optimization, pruning,
historical replay, opacity reset, or global refinement. This is a context-view
reconstruction diagnostic, not a novel-view-synthesis benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
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


ROOT = Path(__file__).resolve().parent
SEQUENCE_CONFIGS = {
    "SE000": ROOT / "Config/Sequence/TartanAirV1_Challenge_SE000.yaml",
    "SE001": ROOT / "Config/Sequence/TartanAirV1_Challenge_SE001.yaml",
    "SE002": ROOT / "Config/Sequence/TartanAirV1_Challenge_SE002.yaml",
    "SE003": ROOT / "Config/Sequence/TartanAirV1_Challenge_SE003.yaml",
    "SH000": ROOT / "Config/Sequence/TartanAirV1_Challenge_SH000.yaml",
    "SH001": ROOT / "Config/Sequence/TartanAirV1_Challenge_SH001.yaml",
    "SH002": ROOT / "Config/Sequence/TartanAirV1_Challenge_SH002.yaml",
    "SH003": ROOT / "Config/Sequence/TartanAirV1_Challenge_SH003.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", choices=sorted(SEQUENCE_CONFIGS), default="SE000")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument(
        "--source_indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit frame indices. If supplied, --num_samples is ignored.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: outputs/tartanair_resplat_packet_diag/<sequence>",
    )
    return p.parse_args()


def select_indices(length: int, num_samples: int) -> list[int]:
    if length <= 0:
        raise RuntimeError("empty TartanAir sequence")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    n = min(num_samples, length)
    positions = np.linspace(0, length - 1, n, dtype=np.int64)
    return [int(i) for i in positions]


def make_generator(device: str, output_dir: Path) -> ResplatPacketGenerator:
    # Match the formal TartanAir Challenge ReSplat frontend.
    cfg = ResplatRuntimeConfig(
        repo=(ROOT / "../Resplat").resolve(),
        experiment="tartanair_p000_ft",
        device=device,
        checkpoint=None,
        overrides=("dataset.image_shape=[240,320]",),
        output_dir=output_dir / "resplat_runtime",
        fx=320.0,
        fy=320.0,
        cx=320.0,
        cy=240.0,
        stereo_baseline=0.25,
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

    data_cfg_path = SEQUENCE_CONFIGS[args.sequence]
    if not data_cfg_path.is_file():
        raise FileNotFoundError(f"missing sequence config: {data_cfg_path}")
    data_cfg, _ = load_config(data_cfg_path)
    sequence = SequenceBase[StereoFrame].instantiate(data_cfg.type, data_cfg.args)
    if str(data_cfg.type) != "TartanAir_NoIMU":
        raise ValueError(f"expected TartanAir_NoIMU, got {data_cfg.type}")

    if args.source_indices:
        indices = [int(i) for i in args.source_indices]
        for idx in indices:
            if idx < 0 or idx >= len(sequence):
                raise IndexError(f"source index {idx} outside [0,{len(sequence)})")
    else:
        indices = select_indices(len(sequence), args.num_samples)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / "outputs/tartanair_resplat_packet_diag" / args.sequence
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = make_generator(args.device, output_dir)
    generator.initialize()
    assert generator.model is not None

    resplat_repo = generator.repo
    if str(resplat_repo) not in sys.path:
        sys.path.insert(0, str(resplat_repo))
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    rows: list[dict[str, float | int]] = []

    print("\n=== TartanAir standalone ReSplat packet diagnostic ===", flush=True)
    print(f"sequence          : {args.sequence}", flush=True)
    print(f"sequence frames   : {len(sequence)}", flush=True)
    print(f"sample indices    : {indices}", flush=True)
    print("input domain      : native TartanAir stereo", flush=True)
    print("ReSplat resize    : full-FoV 640x480 -> 320x240", flush=True)
    print("backend/fusion    : NONE", flush=True)
    print("evaluation        : ReSplat native decoder, same stereo context views\n", flush=True)

    with torch.inference_mode():
        for sample_ordinal, source_index in enumerate(indices):
            frame = sequence[source_index]
            timestamp_ns = int(frame.stereo.frame_ns)

            left_path = Path(sequence.lcam_loader.file_names[source_index])
            right_path = Path(sequence.rcam_loader.file_names[source_index])
            descriptor = FrameDescriptor(
                sequence_index=sample_ordinal,
                frame_index=source_index,
                timestamp_ns=max(0, timestamp_ns),
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
            target = context["image"][0].clamp(0.0, 1.0)
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
            rendered = decoded.color[0].clamp(0.0, 1.0)

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
                "stereo_mean_psnr": float(psnr.mean().item()),
                "stereo_mean_ssim": float(ssim.mean().item()),
                "stereo_mean_lpips": float(lpips.mean().item()),
            }
            rows.append(row)

            frame_dir = output_dir / f"frame_{source_index:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            save_image(target[0], frame_dir / "left_gt.png")
            save_image(rendered[0], frame_dir / "left_render.png")
            save_image(target[1], frame_dir / "right_gt.png")
            save_image(rendered[1], frame_dir / "right_render.png")
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
            "standalone ReSplat local packet; native TartanAir stereo; full-FoV "
            "320x240; native ReSplat decoder; context-view reconstruction only"
        ),
        "formal_resplat_experiment": "tartanair_p000_ft",
        "refine_steps": 0,
        "sequence_frames": len(sequence),
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
