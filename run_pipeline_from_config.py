#!/usr/bin/env python3
"""Run the MAC-VO -> ReSplat -> backend-input pipeline from one YAML file."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def abs_path(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def nested_get(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Unknown config key: {dotted}")
        current = current[key]
    return current


def nested_set(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    current: dict[str, Any] = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            raise KeyError(f"Unknown config key: {dotted}")
        current = current[key]
    if keys[-1] not in current:
        raise KeyError(f"Unknown config key: {dotted}")
    current[keys[-1]] = value


def parse_override(spec: str) -> tuple[str, Any]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid --set value {spec!r}; expected section.key=value"
        )
    key, raw = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Empty config key in --set")
    return key, yaml.safe_load(raw)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{label} not found: {path}")


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def resolve_config(config: dict[str, Any], script_root: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    paths = resolved["paths"]

    macvo_repo = abs_path(paths["macvo_repo"], script_root)
    paths["macvo_repo"] = str(macvo_repo)
    for key in [
        "odom",
        "data",
        "packet_generator_repo",
        "resplat_repo",
        "scene_prepare_script",
        "work_dir",
    ]:
        paths[key] = str(abs_path(paths[key], macvo_repo))

    data_cfg = load_yaml(Path(paths["data"]))
    data_args = data_cfg.get("args")
    if not isinstance(data_args, dict) or "root" not in data_args:
        raise KeyError(
            f"Dataset config must contain args.root: {paths['data']}"
        )
    dataset_root = abs_path(data_args["root"], Path(paths["data"]).parent)
    camera = resolved["camera"]
    camera["dataset_root"] = str(dataset_root)
    camera["left_dir"] = str(dataset_root / camera["left_subdir"])
    camera["right_dir"] = str(dataset_root / camera["right_subdir"])

    split = resolved["split"]
    split["output_scene"] = str(
        Path(paths["work_dir"]) / split["output_scene_name"]
    )
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    paths = config["paths"]
    camera = config["camera"]
    sequence = config["sequence"]
    split = config["split"]

    macvo_repo = Path(paths["macvo_repo"])
    packet_repo = Path(paths["packet_generator_repo"])
    resplat_repo = Path(paths["resplat_repo"])

    require_dir(macvo_repo, "MAC-VO repository")
    require_file(macvo_repo / "MACVO.py", "MAC-VO entrypoint")
    require_file(Path(paths["odom"]), "MAC-VO odometry config")
    require_file(Path(paths["data"]), "MAC-VO dataset config")
    require_dir(packet_repo, "Pose-to-ReSplat integration repository")
    require_file(
        packet_repo / "run_pose_resplat_metric_packet_only.py",
        "Generic pose-to-ReSplat runner",
    )
    require_dir(resplat_repo, "ReSplat repository")
    require_dir(Path(camera["left_dir"]), "Left image directory")
    require_dir(Path(camera["right_dir"]), "Right image directory")
    if split["prepare_camera_scene"]:
        require_file(
            Path(paths["scene_prepare_script"]),
            "Camera-scene preparation script",
        )

    start = int(sequence["start_index"])
    end = int(sequence["end_index"])
    stride = int(sequence["stride"])
    if start < 0 or end <= start:
        raise ValueError(f"Invalid frame range [{start}, {end})")
    if stride != 1:
        raise ValueError("Current MAC-VO integration supports stride=1 only")
    if int(split["split_every"]) <= 0:
        raise ValueError("split.split_every must be positive")


def add_value(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_command(config: dict[str, Any], script_root: Path) -> list[str]:
    paths = config["paths"]
    sequence = config["sequence"]
    camera = config["camera"]
    resplat = config["resplat"]
    split = config["split"]
    runtime = config["runtime"]

    cmd = [
        sys.executable,
        str(script_root / "run_macvo_resplat_packet_only.py"),
        "--macvo_repo", str(paths["macvo_repo"]),
        "--odom", str(paths["odom"]),
        "--data", str(paths["data"]),
        "--zipmap_repo", str(paths["packet_generator_repo"]),
        "--resplat_repo", str(paths["resplat_repo"]),
        "--left_dir", str(camera["left_dir"]),
        "--right_dir", str(camera["right_dir"]),
        "--work_dir", str(paths["work_dir"]),
        "--scene_name", str(sequence["scene_name"]),
        "--start_index", str(sequence["start_index"]),
        "--end_index", str(sequence["end_index"]),
        "--stride", str(sequence["stride"]),
        "--stereo_baseline", str(camera["stereo_baseline"]),
        "--resplat_experiment", str(resplat["experiment"]),
        "--resplat_packet_stage", str(resplat["packet_stage"]),
        "--refine_steps", str(resplat["refine_steps"]),
        "--refine_use_target", str(bool(resplat["refine_use_target"])).lower(),
        "--resplat_target_camera", str(resplat["target_camera"]),
        "--resplat_target_offset", str(resplat["target_offset"]),
        "--packet_out_name", str(resplat["packet_out_name"]),
        "--fx", str(camera["fx"]),
        "--fy", str(camera["fy"]),
        "--cx", str(camera["cx"]),
        "--cy", str(camera["cy"]),
        "--width", str(camera["width"]),
        "--height", str(camera["height"]),
        "--device", str(runtime["device"]),
    ]

    if resplat["self_render_packets"]:
        cmd.append("--self_render_packets")
    if runtime["timing"]:
        cmd.append("--timing")
    if runtime["reuse_macvo_pose"]:
        cmd.append("--reuse_macvo_pose")
    if runtime["skip_ate"]:
        cmd.append("--skip_ate")
    if runtime["show_known_warnings"]:
        cmd.append("--show_known_warnings")

    if split["prepare_camera_scene"]:
        cmd.extend([
            "--prepare_camera_scene",
            "--scene_prepare_script", str(paths["scene_prepare_script"]),
            "--backend_packet_stage", str(resplat["backend_packet_stage"]),
            "--output_scene", str(split["output_scene"]),
            "--image_pattern", str(camera["image_pattern"]),
            "--image_mode", str(camera["image_mode"]),
            "--packet_extrinsic_key", str(split["packet_extrinsic_key"]),
            "--packet_extrinsic_type", str(split["packet_extrinsic_type"]),
            "--split_every", str(split["split_every"]),
            "--split_offset", str(split["split_offset"]),
            "--split_index_mode", str(split["split_index_mode"]),
        ])
    return cmd


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a reproducible MAC-VO/ReSplat pipeline YAML"
    )
    p.add_argument(
        "--config",
        default="Config/Pipeline/MACVO_ReSplat_P000_0_50.yaml",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one existing YAML value. May be repeated.",
    )
    p.add_argument("--dry_run", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    script_root = Path(__file__).resolve().parent
    config_path = abs_path(args.config, script_root)
    require_file(config_path, "Pipeline config")

    config = load_yaml(config_path)
    for spec in args.overrides:
        key, value = parse_override(spec)
        nested_get(config, key)
        nested_set(config, key, value)

    resolved = resolve_config(config, script_root)
    validate_config(resolved)
    command = build_command(resolved, script_root)

    work_dir = Path(resolved["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "resolved_pipeline_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    command_text = shlex.join(command)
    (work_dir / "resolved_pipeline_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + command_text + "\n",
        encoding="utf-8",
    )

    print("[config]", config_path)
    print("[work_dir]", work_dir)
    print("[command]", command_text)
    if args.dry_run:
        print("[dry-run] validation passed; command was not executed")
        return

    subprocess.run(command, cwd=script_root, check=True)


if __name__ == "__main__":
    main()
