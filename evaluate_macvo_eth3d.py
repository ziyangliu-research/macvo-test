#!/usr/bin/env python3
"""Run MAC-VO only on rectified ETH3D and report trajectory ATE.

No ReSplat inference, Gaussian fusion, GraphDECO optimization, or rendering is
executed.  The ETH3D loader supplies rectified stereo K/baseline and interpolated
rectified-left GT at the exact image timestamps.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import run_async_pipeline as base
from eth3d_rectified_support import evaluate_pose_eth3d, install_eth3d_runtime_support


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="Config/Pipeline/MACVO_ReSplat_Serial_ETH3D_mannequin_face_1_Smoke.yaml",
    )
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--output_dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed = int(os.environ.get("PIPELINE_BENCHMARK_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    install_eth3d_runtime_support()

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

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path(resolved["paths"]["work_dir"]).expanduser().resolve() / "macvo_only"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    # Build the common system object only to reuse the exact frontend construction.
    # The smoke config uses runtime.backend_mode=null, and packet_generator is never
    # initialized or invoked here.
    template = base.build_system(resolved)
    frontend = template.pose_frontend

    init_start = time.perf_counter()
    frontend.initialize()
    initialization_sec = time.perf_counter() - init_start

    frame_count = 0
    run_start = time.perf_counter()
    for descriptor, frame, _stereo_input, _observation in frontend.iter_frames():
        frontend.process(descriptor, frame)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"[MAC-VO] processed {frame_count} frames", flush=True)
    frontend.terminate()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    run_sec = time.perf_counter() - run_start

    metrics = evaluate_pose_eth3d(frontend, resolved, frame_count, output_dir)
    se3_rmse = metrics["se3"]["translation_error_m"]["rmse"]
    sim3_rmse = metrics["sim3"]["translation_error_m"]["rmse"]
    summary = {
        "frames": frame_count,
        "initialization_sec": initialization_sec,
        "macvo_wall_sec": run_sec,
        "macvo_fps_excluding_initialization": frame_count / run_sec,
        "se3_ate_rmse_m": se3_rmse,
        "sim3_ate_rmse_m": sim3_rmse,
        "pose_metrics": metrics,
    }
    (output_dir / "macvo_only_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== ETH3D MAC-VO only ===")
    print(f"frames        : {frame_count}")
    print(f"SE3 ATE RMSE  : {se3_rmse:.6f} m")
    print(f"Sim3 ATE RMSE : {sim3_rmse:.6f} m")
    print(f"MAC-VO FPS    : {frame_count / run_sec:.3f}")
    print(f"output        : {output_dir}")


if __name__ == "__main__":
    main()
