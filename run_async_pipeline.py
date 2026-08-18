#!/usr/bin/env python3
"""Run the class-based asynchronous MAC-VO + ReSplat + 3DGS system.

Models are initialized once and communicate through bounded in-memory queues.
No stage launches another Python script and no NPZ/PT/JSON file is required as a
runtime hand-off. Optional artifacts are written only for reproducibility.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from async_pipeline.macvo_runtime import MacvoPoseFrontend, MacvoRuntimeConfig
from async_pipeline.pipeline import (
    AsyncPipelineConfig,
    AsyncPipelineRunner,
    NullValidationBackend,
)
from async_pipeline.resplat_runtime import (
    ResplatPacketGenerator,
    ResplatRuntimeConfig,
)
from async_pipeline.streaming_backend import (
    BackendOptimizationConfig,
    StreamingBackendConfig,
    StreamingIncrementalBackend,
)


def absolute(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return value


def nested_set(config: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    current: Any = config
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"unknown config key {dotted}")
        current = current[key]
    if not isinstance(current, dict) or keys[-1] not in current:
        raise KeyError(f"unknown config key {dotted}")
    current[keys[-1]] = value


def normalize_override_value(dotted_key: str, raw: str) -> Any:
    """Parse CLI YAML values while preserving named string modes.

    YAML 1.1-style parsing treats unquoted ``off`` as boolean ``False`` and
    ``null`` as ``None``. In this pipeline those tokens are legitimate string
    values for backend configuration, so preserve them before generic parsing.
    """

    stripped = raw.strip()
    lowered = stripped.lower()
    if dotted_key == "backend.maintenance_mode" and lowered in {"off", "standard"}:
        return lowered
    if dotted_key == "runtime.backend_mode" and lowered == "null":
        return "null"

    parsed = yaml.safe_load(raw)
    if dotted_key == "runtime.backend_mode" and parsed is None:
        return "null"
    return parsed


def resolve(config: dict[str, Any], script_root: Path) -> dict[str, Any]:
    value = copy.deepcopy(config)
    paths = value["paths"]
    macvo_repo = absolute(paths["macvo_repo"], script_root)
    paths["macvo_repo"] = str(macvo_repo)
    paths["resplat_repo"] = str(absolute(paths["resplat_repo"], macvo_repo))
    paths["gs_repo"] = str(absolute(paths["gs_repo"], macvo_repo))
    paths["odom_config"] = str(absolute(paths["odom_config"], macvo_repo))
    paths["data_config"] = str(absolute(paths["data_config"], macvo_repo))
    paths["work_dir"] = str(absolute(paths["work_dir"], macvo_repo))
    checkpoint = paths.get("resplat_checkpoint")
    if checkpoint:
        paths["resplat_checkpoint"] = str(
            absolute(checkpoint, Path(paths["resplat_repo"]))
        )
    evaluation = value.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("gt_pose_file"):
        evaluation["gt_pose_file"] = str(
            absolute(evaluation["gt_pose_file"], macvo_repo)
        )
    return value


def validate_paths(config: dict[str, Any]) -> None:
    paths = config["paths"]
    required_dirs = ["macvo_repo", "resplat_repo", "gs_repo"]
    required_files = ["odom_config", "data_config"]
    for key in required_dirs:
        path = Path(paths[key])
        if not path.is_dir():
            raise NotADirectoryError(f"{key} not found: {path}")
    for key in required_files:
        path = Path(paths[key])
        if not path.is_file():
            raise FileNotFoundError(f"{key} not found: {path}")
    if not (Path(paths["macvo_repo"]) / "Odometry" / "MACVO.py").is_file():
        raise FileNotFoundError("MAC-VO Python API is missing")
    if not (Path(paths["resplat_repo"]) / "src" / "model").is_dir():
        raise FileNotFoundError("ReSplat Python package is missing")
    if not (Path(paths["gs_repo"]) / "gaussian_renderer").is_dir():
        raise FileNotFoundError("GraphDECO renderer is missing")


def build_system(config: dict[str, Any]):
    paths = config["paths"]
    sequence = config["sequence"]
    camera = config["camera"]
    pose_cfg = config["pose_frontend"]
    resplat_cfg = config["resplat_frontend"]
    backend_cfg = config["backend"]
    split = config["split"]
    runtime = config["runtime"]
    evaluation = config.get("evaluation") or {}
    work_dir = Path(paths["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    gt_pose_value = evaluation.get("gt_pose_file")
    pose_frontend = MacvoPoseFrontend(
        MacvoRuntimeConfig(
            repo=Path(paths["macvo_repo"]),
            odom_config=Path(paths["odom_config"]),
            data_config=Path(paths["data_config"]),
            start_index=int(sequence["start_index"]),
            end_index=int(sequence["end_index"]),
            left_subdir=camera["left_subdir"],
            right_subdir=camera["right_subdir"],
            left_pattern=camera["left_pattern"],
            right_pattern=camera["right_pattern"],
            preload=bool(runtime["preload"]),
            pose_commit_policy=pose_cfg["commit_policy"],
            dedicated_cuda_stream=bool(pose_cfg["dedicated_cuda_stream"]),
            pose_source=str(pose_cfg.get("source", "macvo")),
            gt_pose_file=(None if not gt_pose_value else Path(str(gt_pose_value))),
        )
    )
    packet_generator = ResplatPacketGenerator(
        ResplatRuntimeConfig(
            repo=Path(paths["resplat_repo"]),
            experiment=resplat_cfg["experiment"],
            device=resplat_cfg["device"],
            checkpoint=(
                None
                if not paths.get("resplat_checkpoint")
                else Path(paths["resplat_checkpoint"])
            ),
            overrides=tuple(resplat_cfg.get("overrides", [])),
            output_dir=work_dir / "resplat_runtime",
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

    backend_mode = runtime.get("backend_mode")
    if backend_mode is None:
        # Be robust to YAML ``null`` in a config file as well as the CLI form.
        backend_mode = "null"

    if backend_mode == "null":
        backend = NullValidationBackend()
    elif backend_mode == "incremental":
        optimization = BackendOptimizationConfig(**backend_cfg["optimization"])
        backend = StreamingIncrementalBackend(
            StreamingBackendConfig(
                gs_repo=Path(paths["gs_repo"]),
                output_dir=work_dir / backend_cfg["output_name"],
                device=backend_cfg["device"],
                sh_degree=int(backend_cfg["sh_degree"]),
                local_map_size=int(backend_cfg["local_map_size"]),
                iterations_per_packet=int(backend_cfg["iterations_per_packet"]),
                reset_new_packet_opacity=bool(
                    backend_cfg["reset_new_packet_opacity"]
                ),
                new_packet_reset_max_opacity=float(
                    backend_cfg["new_packet_reset_max_opacity"]
                ),
                maintenance_mode=backend_cfg["maintenance_mode"],
                maintenance_after_local_iteration=int(
                    backend_cfg["maintenance_after_local_iteration"]
                ),
                maintenance_grad_threshold=float(
                    backend_cfg["maintenance_grad_threshold"]
                ),
                maintenance_min_opacity=float(
                    backend_cfg["maintenance_min_opacity"]
                ),
                maintenance_max_screen_size=float(
                    backend_cfg["maintenance_max_screen_size"]
                ),
                spatial_lr_scale=float(backend_cfg["spatial_lr_scale"]),
                minimum_scene_extent=float(backend_cfg["minimum_scene_extent"]),
                white_background=bool(backend_cfg["white_background"]),
                antialiasing=bool(backend_cfg["antialiasing"]),
                eval_before_optimization=bool(
                    backend_cfg["eval_before_optimization"]
                ),
                eval_every_train_packets=int(
                    backend_cfg["eval_every_train_packets"]
                ),
                eval_max_views=int(backend_cfg["eval_max_views"]),
                save_every_train_packets=int(
                    backend_cfg["save_every_train_packets"]
                ),
                save_final_ply=bool(backend_cfg["save_final_ply"]),
                wandb_mode=backend_cfg["wandb_mode"],
                wandb_project=backend_cfg["wandb_project"],
                wandb_run_name=backend_cfg["wandb_run_name"],
                wandb_log_interval=int(backend_cfg["wandb_log_interval"]),
                dedicated_cuda_stream=bool(backend_cfg["dedicated_cuda_stream"]),
                invalid_pose_policy=backend_cfg["invalid_pose_policy"],
                write_runtime_artifacts=bool(backend_cfg["write_runtime_artifacts"]),
                evaluation_enabled=bool(backend_cfg["evaluation_enabled"]),
                optimization=optimization,
            )
        )
    else:
        raise ValueError(f"unknown backend_mode {backend_mode}")

    pipeline_config = AsyncPipelineConfig(
        split_every=int(split["split_every"]),
        split_offset=int(split["split_offset"]),
        split_index_mode=split["split_index_mode"],
        queue_size=int(runtime["queue_size"]),
        poll_timeout_sec=float(runtime["poll_timeout_sec"]),
        initialization_timeout_sec=float(runtime["initialization_timeout_sec"]),
        summary_path=work_dir / "async_pipeline_summary.json",
    )
    return AsyncPipelineRunner(
        pipeline_config,
        pose_frontend,
        packet_generator,
        backend,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Config/Pipeline/MACVO_ReSplat_Async_P000_0_50.yaml",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = absolute(args.config, root)
    config = load_yaml(config_path)
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}")
        key, raw = item.split("=", 1)
        nested_set(config, key, normalize_override_value(key, raw))
    resolved = resolve(config, root)
    validate_paths(resolved)
    work_dir = Path(resolved["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "resolved_async_pipeline_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        print("[dry-run] configuration and repository paths are valid")
        return
    runner = build_system(resolved)
    summary = runner.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
