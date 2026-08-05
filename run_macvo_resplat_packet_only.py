#!/usr/bin/env python3
"""MAC-VO stereo trajectory -> ReSplat packets -> optional backend camera scene."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


KNOWN_WARNING_FILTERS = [
    "ignore:instrumentor did not find the target function",
    "ignore:In 'main'",
    "ignore:The parameter 'pretrained' is deprecated since 0.13",
    "ignore:Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13",
]


def abs_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def subprocess_env(show_known_warnings: bool) -> dict[str, str]:
    env = os.environ.copy()
    if show_known_warnings:
        return env
    known = ",".join(KNOWN_WARNING_FILTERS)
    existing = env.get("PYTHONWARNINGS", "").strip(",")
    env["PYTHONWARNINGS"] = f"{existing},{known}" if existing else known
    return env


def run(
    cmd: list[str],
    stage: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> float:
    print(f"\n[{stage}] {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, check=True, env=env)
    sec = time.perf_counter() - start
    print(f"[{stage}] completed in {sec:.3f}s", flush=True)
    return sec


def add(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd += [flag, str(value)]


def split_packet_ids(
    num_packets: int,
    dataset_start_index: int,
    split_every: int,
    split_offset: int,
    split_index_mode: str,
) -> tuple[list[int], list[int]]:
    if split_every <= 0:
        raise ValueError("--split_every must be positive")
    train_ids: list[int] = []
    test_ids: list[int] = []
    for local_index in range(num_packets):
        packet_index = local_index
        frame_index = dataset_start_index + packet_index
        if split_index_mode == "local_index":
            value = local_index
        elif split_index_mode == "packet_index":
            value = packet_index
        elif split_index_mode == "frame_index":
            value = frame_index
        else:
            raise ValueError(f"Unknown split_index_mode: {split_index_mode}")
        is_test = (value - split_offset) % split_every == 0
        (test_ids if is_test else train_ids).append(packet_index)
    return train_ids, test_ids


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "MAC-VO -> ReSplat Gaussian packets, with optional strict-split "
            "camera-scene preparation for the incremental 3DGS backend"
        )
    )
    p.add_argument("--macvo_repo", required=True)
    p.add_argument("--odom", required=True)
    p.add_argument("--data", required=True)
    p.add_argument(
        "--zipmap_repo",
        required=True,
        help=(
            "Repository containing run_pose_resplat_metric_packet_only.py and, "
            "when requested, prepare_zipmap_packet_camera_scene_only.py"
        ),
    )
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
    p.add_argument(
        "--resplat_packet_stage",
        choices=["init", "final", "both"],
        default="init",
    )
    p.add_argument("--refine_steps", default="0")
    p.add_argument("--refine_use_target", default="false")
    p.add_argument(
        "--resplat_target_camera",
        choices=["left", "right", "both"],
        default="left",
    )
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
    p.add_argument("--skip_ate", action="store_true")
    p.add_argument(
        "--show_known_warnings",
        action="store_true",
        help="Show known benign jaxtyping/Hydra/torchvision compatibility warnings.",
    )

    # Optional third stage: camera-only GraphDECO scene for the backend.
    p.add_argument(
        "--prepare_camera_scene",
        action="store_true",
        help=(
            "After packet generation, run the strict-split packet camera-scene "
            "preparation script and write backend_input_manifest.json."
        ),
    )
    p.add_argument(
        "--scene_prepare_script",
        default=None,
        help=(
            "Path to prepare_zipmap_packet_camera_scene_only.py. Default: "
            "<zipmap_repo>/prepare_zipmap_packet_camera_scene_only.py"
        ),
    )
    p.add_argument(
        "--backend_packet_stage",
        default="refine_0",
        help="Packet subdirectory used by the backend, e.g. refine_0.",
    )
    p.add_argument(
        "--output_scene",
        default=None,
        help=(
            "Output camera scene. Default: "
            "<work_dir>/3dgs_camera_scene_macvo_strict_split"
        ),
    )
    p.add_argument("--image_pattern", default="{index:06d}_lcam_front.png")
    p.add_argument("--image_mode", default="symlink")
    p.add_argument("--packet_extrinsic_key", default="target_extrinsics")
    p.add_argument("--packet_extrinsic_type", default="Twc")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--split_every", type=int, default=5)
    p.add_argument("--split_offset", type=int, default=4)
    p.add_argument(
        "--split_index_mode",
        choices=["local_index", "packet_index", "frame_index"],
        default="local_index",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.stride != 1:
        raise ValueError(
            "This integrated MAC-VO pipeline currently requires --stride 1"
        )
    if args.end_index <= args.start_index:
        raise ValueError("--end_index must be larger than --start_index")

    root = Path(__file__).resolve().parent
    macvo_repo = abs_path(args.macvo_repo)
    zipmap_repo = abs_path(args.zipmap_repo)
    work = abs_path(args.work_dir)
    pose_dir = work / "macvo_pose"
    work.mkdir(parents=True, exist_ok=True)
    child_env = subprocess_env(args.show_known_warnings)

    total_stages = 3 if args.prepare_camera_scene else 2

    pose_cmd = [
        sys.executable,
        str(root / "export_macvo_metric_pose.py"),
        "--macvo_repo",
        str(macvo_repo),
        "--odom",
        str(abs_path(args.odom)),
        "--data",
        str(abs_path(args.data)),
        "--output_dir",
        str(pose_dir),
        "--start_index",
        str(args.start_index),
        "--end_index",
        str(args.end_index),
        "--stride",
        str(args.stride),
    ]
    if args.timing:
        pose_cmd.append("--timing")
    if args.reuse_macvo_pose:
        pose_cmd.append("--reuse_existing")
    if args.skip_ate:
        pose_cmd.append("--skip_ate")
    if args.show_known_warnings:
        pose_cmd.append("--show_known_warnings")
    pose_sec = run(
        pose_cmd,
        f"1/{total_stages} MAC-VO metric pose",
        cwd=macvo_repo,
        env=child_env,
    )

    packet_script = zipmap_repo / "run_pose_resplat_metric_packet_only.py"
    if not packet_script.is_file():
        raise FileNotFoundError(
            f"Missing generic ReSplat packet runner: {packet_script}. "
            "Use the ZipMap branch containing the pairwise pose integration scripts."
        )

    packet_cmd = [
        sys.executable,
        str(packet_script),
        "--resplat_repo",
        str(abs_path(args.resplat_repo)),
        "--pose_npz",
        str(pose_dir / "macvo_pose_results.npz"),
        "--pose_source_name",
        "macvo_stereo",
        "--pose_key",
        "T_raw_accumulated_c2w_opencv",
        "--index_key",
        "selected_original_indices",
        "--scale_key",
        "",
        "--left_dir",
        str(abs_path(args.left_dir)),
        "--right_dir",
        str(abs_path(args.right_dir)),
        "--work_dir",
        str(work),
        "--scene_name",
        args.scene_name,
        "--start_index",
        str(args.start_index),
        "--end_index",
        str(args.end_index),
        "--stride",
        str(args.stride),
        "--stereo_baseline",
        str(args.stereo_baseline),
        "--resplat_experiment",
        args.resplat_experiment,
        "--resplat_packet_stage",
        args.resplat_packet_stage,
        "--refine_steps",
        args.refine_steps,
        "--refine_use_target",
        args.refine_use_target,
        "--resplat_target_camera",
        args.resplat_target_camera,
        "--resplat_target_offset",
        str(args.resplat_target_offset),
        "--packet_out_name",
        args.packet_out_name,
        "--device",
        args.device,
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

    packet_sec = run(
        packet_cmd,
        f"2/{total_stages} ReSplat packets",
        cwd=zipmap_repo,
        env=child_env,
    )

    num_packets = args.end_index - args.start_index
    packet_range_spec = f"0-{num_packets - 1}"
    train_packet_ids, test_packet_ids = split_packet_ids(
        num_packets=num_packets,
        dataset_start_index=args.start_index,
        split_every=args.split_every,
        split_offset=args.split_offset,
        split_index_mode=args.split_index_mode,
    )

    scene_sec = 0.0
    output_scene: Path | None = None
    backend_packet_dir = work / args.packet_out_name / args.backend_packet_stage
    if args.prepare_camera_scene:
        required_intrinsics = {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
        }
        missing = [name for name, value in required_intrinsics.items() if value is None]
        if missing:
            raise ValueError(
                "Camera-scene preparation requires explicit intrinsics: "
                + ", ".join(missing)
            )
        if not backend_packet_dir.is_dir():
            raise FileNotFoundError(
                f"Backend packet stage does not exist: {backend_packet_dir}"
            )

        prepare_script = (
            abs_path(args.scene_prepare_script)
            if args.scene_prepare_script
            else zipmap_repo / "prepare_zipmap_packet_camera_scene_only.py"
        )
        if not prepare_script.is_file():
            raise FileNotFoundError(
                f"Missing camera-scene preparation script: {prepare_script}. "
                "Pass its local path with --scene_prepare_script."
            )
        output_scene = (
            abs_path(args.output_scene)
            if args.output_scene
            else work / "3dgs_camera_scene_macvo_strict_split"
        )

        scene_cmd = [
            sys.executable,
            str(prepare_script),
            "--packet_dir",
            str(backend_packet_dir),
            "--packet_range_spec",
            packet_range_spec,
            "--dataset_start_index",
            str(args.start_index),
            "--image_dir",
            str(abs_path(args.left_dir)),
            "--image_pattern",
            args.image_pattern,
            "--image_mode",
            args.image_mode,
            "--output_scene",
            str(output_scene),
            "--packet_extrinsic_key",
            args.packet_extrinsic_key,
            "--packet_extrinsic_type",
            args.packet_extrinsic_type,
            "--fx",
            str(args.fx),
            "--fy",
            str(args.fy),
            "--cx",
            str(args.cx),
            "--cy",
            str(args.cy),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--split_every",
            str(args.split_every),
            "--split_offset",
            str(args.split_offset),
            "--split_index_mode",
            args.split_index_mode,
        ]
        scene_sec = run(
            scene_cmd,
            f"3/{total_stages} strict-split camera scene",
            cwd=zipmap_repo,
            env=child_env,
        )

    pose_summary_path = pose_dir / "summary.json"
    pose_summary = (
        json.loads(pose_summary_path.read_text(encoding="utf-8"))
        if pose_summary_path.is_file()
        else None
    )
    summary = {
        "pipeline": (
            "MAC-VO stereo -> metric OpenCV c2w -> ReSplat packets"
            + (" -> strict-split backend camera scene" if args.prepare_camera_scene else "")
        ),
        "frame_range": [args.start_index, args.end_index],
        "num_packets": num_packets,
        "packet_range_spec": packet_range_spec,
        "refine_steps": args.refine_steps,
        "backend_packet_stage": args.backend_packet_stage,
        "pose_stage_sec": pose_sec,
        "packet_stage_sec": packet_sec,
        "camera_scene_stage_sec": scene_sec,
        "total_sec": pose_sec + packet_sec + scene_sec,
        "pose_output": str(pose_dir / "macvo_pose_results.npz"),
        "pose_evaluation": (
            None if pose_summary is None else pose_summary.get("evaluation")
        ),
        "packet_output": str(work / args.packet_out_name),
        "backend_packet_dir": str(backend_packet_dir),
        "camera_scene_prepared": bool(args.prepare_camera_scene),
        "camera_scene": None if output_scene is None else str(output_scene),
        "split": {
            "dataset_start_index": args.start_index,
            "split_every": args.split_every,
            "split_offset": args.split_offset,
            "split_index_mode": args.split_index_mode,
            "train_packet_ids": train_packet_ids,
            "test_packet_ids": test_packet_ids,
            "num_train_packets": len(train_packet_ids),
            "num_test_packets": len(test_packet_ids),
        },
        "fusion_performed": False,
    }
    (work / "combined_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if args.prepare_camera_scene:
        backend_manifest = {
            "source_scene": str(output_scene),
            "packet_dir": str(backend_packet_dir),
            "packet_range_spec": packet_range_spec,
            "dataset_start_index": args.start_index,
            "internal_split": True,
            "split_every": args.split_every,
            "split_offset": args.split_offset,
            "split_index_mode": args.split_index_mode,
            "train_packet_ids": train_packet_ids,
            "test_packet_ids": test_packet_ids,
            "test_packets_should_be_loaded": False,
            "note": (
                "Pass the same packet range and split parameters to the incremental "
                "backend. The source scene supplies cameras/images only."
            ),
        }
        (work / "backend_input_manifest.json").write_text(
            json.dumps(backend_manifest, indent=2), encoding="utf-8"
        )

    print(f"\n[Done] packets: {work / args.packet_out_name}")
    if output_scene is not None:
        print(f"[Done] camera scene: {output_scene}")
        print(f"[Done] backend manifest: {work / 'backend_input_manifest.json'}")


if __name__ == "__main__":
    main()
