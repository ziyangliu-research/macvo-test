#!/usr/bin/env python3
"""One-command pipeline: MAC-VO stereo trajectory -> per-frame ReSplat packets."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def abs_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def run(cmd: list[str], stage: str, cwd: Path | None = None) -> float:
    print(f"\n[{stage}] {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, check=True)
    sec = time.perf_counter() - start
    print(f"[{stage}] completed in {sec:.3f}s", flush=True)
    return sec


def add(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd += [flag, str(value)]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MAC-VO -> ReSplat Gaussian packets")
    p.add_argument("--macvo_repo", required=True)
    p.add_argument("--odom", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--zipmap_repo", required=True,
                   help="Repository containing run_pose_resplat_metric_packet_only.py")
    p.add_argument("--resplat_repo", required=True)
    p.add_argument("--left_dir", required=True)
    p.add_argument("--right_dir", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--scene_name", required=True)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--stereo_baseline", type=float, default=0.25000006)
    p.add_argument("--resplat_experiment", required=True)
    p.add_argument("--resplat_checkpoint", default=None)
    p.add_argument("--resplat_override", action="append", default=[])
    p.add_argument("--resplat_packet_stage", choices=["init", "final", "both"], default="init")
    p.add_argument("--refine_steps", default="0")
    p.add_argument("--refine_use_target", default="false")
    p.add_argument("--resplat_target_camera", choices=["left", "right", "both"], default="left")
    p.add_argument("--resplat_target_offset", type=int, default=0)
    p.add_argument("--packet_out_name", default="packets")
    p.add_argument("--self_render_packets", action="store_true")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--reuse_macvo_pose", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    root = Path(__file__).resolve().parent
    macvo_repo = abs_path(args.macvo_repo)
    zipmap_repo = abs_path(args.zipmap_repo)
    work = abs_path(args.work_dir)
    pose_dir = work / "macvo_pose"
    work.mkdir(parents=True, exist_ok=True)

    pose_cmd = [
        sys.executable, str(root / "export_macvo_metric_pose.py"),
        "--macvo_repo", str(macvo_repo),
        "--odom", str(abs_path(args.odom)),
        "--data", str(abs_path(args.data)),
        "--output_dir", str(pose_dir),
        "--start_index", str(args.start_index),
        "--end_index", str(args.end_index),
        "--stride", str(args.stride),
    ]
    if args.timing:
        pose_cmd.append("--timing")
    if args.reuse_macvo_pose:
        pose_cmd.append("--reuse_existing")
    pose_sec = run(pose_cmd, "1/2 MAC-VO metric pose", cwd=macvo_repo)

    packet_script = zipmap_repo / "run_pose_resplat_metric_packet_only.py"
    if not packet_script.is_file():
        raise FileNotFoundError(
            f"Missing generic ReSplat packet runner: {packet_script}. "
            "Use the ZipMap branch containing the pairwise pose integration scripts."
        )

    packet_cmd = [
        sys.executable, str(packet_script),
        "--resplat_repo", str(abs_path(args.resplat_repo)),
        "--pose_npz", str(pose_dir / "macvo_pose_results.npz"),
        "--pose_source_name", "macvo_stereo",
        "--pose_key", "T_raw_accumulated_c2w_opencv",
        "--index_key", "selected_original_indices",
        "--scale_key", "",
        "--left_dir", str(abs_path(args.left_dir)),
        "--right_dir", str(abs_path(args.right_dir)),
        "--work_dir", str(work),
        "--scene_name", args.scene_name,
        "--start_index", str(args.start_index),
        "--end_index", str(args.end_index),
        "--stride", str(args.stride),
        "--stereo_baseline", str(args.stereo_baseline),
        "--resplat_experiment", args.resplat_experiment,
        "--resplat_packet_stage", args.resplat_packet_stage,
        "--refine_steps", args.refine_steps,
        "--refine_use_target", args.refine_use_target,
        "--resplat_target_camera", args.resplat_target_camera,
        "--resplat_target_offset", str(args.resplat_target_offset),
        "--packet_out_name", args.packet_out_name,
        "--device", args.device,
    ]
    add(packet_cmd, "--resplat_checkpoint", args.resplat_checkpoint)
    add(packet_cmd, "--fx", args.fx)
    add(packet_cmd, "--fy", args.fy)
    add(packet_cmd, "--cx", args.cx)
    add(packet_cmd, "--cy", args.cy)
    for override in args.resplat_override:
        packet_cmd += ["--resplat_override", override]
    if args.self_render_packets:
        packet_cmd.append("--self_render_packets")

    packet_sec = run(packet_cmd, "2/2 ReSplat packets", cwd=zipmap_repo)
    summary = {
        "pipeline": "MAC-VO stereo -> metric OpenCV c2w -> ReSplat packets",
        "frame_range": [args.start_index, args.end_index],
        "refine_steps": args.refine_steps,
        "pose_stage_sec": pose_sec,
        "packet_stage_sec": packet_sec,
        "total_sec": pose_sec + packet_sec,
        "pose_output": str(pose_dir / "macvo_pose_results.npz"),
        "packet_output": str(work / args.packet_out_name),
        "fusion_performed": False,
    }
    (work / "combined_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n[Done] packets: {work / args.packet_out_name}")


if __name__ == "__main__":
    main()
