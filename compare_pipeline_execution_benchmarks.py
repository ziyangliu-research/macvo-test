#!/usr/bin/env python3
"""Compare matched serial/async benchmark outputs and generate paper-ready plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial_dir", required=True)
    parser.add_argument("--async_dir", required=True)
    parser.add_argument(
        "--output_dir",
        default="outputs/serial_async_comparison_P000_0_50",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(directory: Path) -> dict[str, Any]:
    summary = load_json(directory / "execution_benchmark_summary.json")
    timing = load_json(directory / "frame_timing_log.json")
    resolved = load_json(directory / "resolved_execution_benchmark_config.json")
    backend_dir = directory / resolved["backend"]["output_name"]
    metrics_path = backend_dir / "metrics_log.json"
    metrics = load_json(metrics_path) if metrics_path.is_file() else []
    return {
        "directory": directory,
        "summary": summary,
        "timing": timing,
        "resolved": resolved,
        "backend_dir": backend_dir,
        "metrics": metrics,
    }


def train_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in run["timing"]
        if not bool(row.get("is_test", False)) and "backend_end_sec" in row
    ]


def metric_after_by_frame(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for entry in run["metrics"]:
        if str(entry.get("stage", "")).endswith("_after"):
            output[int(entry["frame_index"])] = entry
    return output


def plot_two_lines(
    serial_x: list[float],
    serial_y: list[float],
    async_x: list[float],
    async_y: list[float],
    *,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(serial_x, serial_y, marker="o", markersize=3, label="Serial")
    plt.plot(async_x, async_y, marker="o", markersize=3, label="Asynchronous")
    plt.xlabel("Frame index / timestamp t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def write_timing_csv(
    serial_rows: list[dict[str, Any]],
    async_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    serial_map = {int(row["frame_index"]): row for row in serial_rows}
    async_map = {int(row["frame_index"]): row for row in async_rows}
    frames = sorted(set(serial_map) & set(async_map))
    fields = [
        "frame_index",
        "serial_completion_elapsed_sec",
        "async_completion_elapsed_sec",
        "serial_train_update_interval_sec",
        "async_train_update_interval_sec",
        "serial_backend_duration_sec",
        "async_backend_duration_sec",
        "serial_packet_duration_sec",
        "async_packet_duration_sec",
        "serial_pose_task_duration_sec",
        "async_pose_task_duration_sec",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            serial = serial_map[frame]
            asynchronous = async_map[frame]
            writer.writerow(
                {
                    "frame_index": frame,
                    "serial_completion_elapsed_sec": serial.get(
                        "map_completion_elapsed_sec"
                    ),
                    "async_completion_elapsed_sec": asynchronous.get(
                        "map_completion_elapsed_sec"
                    ),
                    "serial_train_update_interval_sec": serial.get(
                        "completion_interval_train_updates_sec"
                    ),
                    "async_train_update_interval_sec": asynchronous.get(
                        "completion_interval_train_updates_sec"
                    ),
                    "serial_backend_duration_sec": serial.get(
                        "backend_duration_sec"
                    ),
                    "async_backend_duration_sec": asynchronous.get(
                        "backend_duration_sec"
                    ),
                    "serial_packet_duration_sec": serial.get("packet_duration_sec"),
                    "async_packet_duration_sec": asynchronous.get(
                        "packet_duration_sec"
                    ),
                    "serial_pose_task_duration_sec": serial.get(
                        "pose_task_duration_sec"
                    ),
                    "async_pose_task_duration_sec": asynchronous.get(
                        "pose_task_duration_sec"
                    ),
                }
            )


def plot_quality(
    serial: dict[str, Any],
    asynchronous: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    serial_metrics = metric_after_by_frame(serial)
    async_metrics = metric_after_by_frame(asynchronous)
    common_frames = sorted(set(serial_metrics) & set(async_metrics))
    if not common_frames:
        return []

    outputs: list[str] = []
    for split, label in (
        ("train_inserted", "All inserted train views"),
        ("active_local_map", "Active local map"),
        ("test_seen", "Seen test views"),
    ):
        for metric_name, metric_label in (("psnr", "PSNR (dB)"), ("ssim", "SSIM")):
            valid_frames = [
                frame
                for frame in common_frames
                if metric_name in serial_metrics[frame].get(split, {})
                and metric_name in async_metrics[frame].get(split, {})
            ]
            if not valid_frames:
                continue
            serial_values = [
                float(serial_metrics[frame][split][metric_name])
                for frame in valid_frames
            ]
            async_values = [
                float(async_metrics[frame][split][metric_name])
                for frame in valid_frames
            ]
            filename = f"{split}_{metric_name}_after_update.png"
            plot_two_lines(
                [float(frame) for frame in valid_frames],
                serial_values,
                [float(frame) for frame in valid_frames],
                async_values,
                ylabel=metric_label,
                title=f"Serial vs asynchronous: {label} {metric_label}",
                output=output_dir / filename,
            )
            outputs.append(filename)
    return outputs


def main() -> None:
    args = parse_args()
    serial = load_run(Path(args.serial_dir).expanduser().resolve())
    asynchronous = load_run(Path(args.async_dir).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    serial_rows = train_rows(serial)
    async_rows = train_rows(asynchronous)
    serial_frames = [float(row["frame_index"]) for row in serial_rows]
    async_frames = [float(row["frame_index"]) for row in async_rows]

    plot_two_lines(
        serial_frames,
        [float(row["map_completion_elapsed_sec"]) for row in serial_rows],
        async_frames,
        [float(row["map_completion_elapsed_sec"]) for row in async_rows],
        ylabel="Cumulative completion time (s)",
        title="Map availability after each train timestamp",
        output=output_dir / "cumulative_map_completion_time.png",
    )

    serial_interval_rows = [
        row for row in serial_rows if "completion_interval_train_updates_sec" in row
    ]
    async_interval_rows = [
        row for row in async_rows if "completion_interval_train_updates_sec" in row
    ]
    plot_two_lines(
        [float(row["frame_index"]) for row in serial_interval_rows],
        [float(row["completion_interval_train_updates_sec"]) for row in serial_interval_rows],
        [float(row["frame_index"]) for row in async_interval_rows],
        [float(row["completion_interval_train_updates_sec"]) for row in async_interval_rows],
        ylabel="Time between completed map updates (s)",
        title="Per-timestamp map update interval",
        output=output_dir / "per_timestamp_update_interval.png",
    )

    plot_two_lines(
        serial_frames,
        [float(row["nominal_sensor_to_map_latency_sec"]) for row in serial_rows],
        async_frames,
        [float(row["nominal_sensor_to_map_latency_sec"]) for row in async_rows],
        ylabel="Nominal sensor-to-map latency (s)",
        title="Accumulated latency relative to dataset timestamps",
        output=output_dir / "nominal_sensor_to_map_latency.png",
    )

    write_timing_csv(serial_rows, async_rows, output_dir / "timing_comparison.csv")
    quality_plots = plot_quality(serial, asynchronous, output_dir)

    serial_summary = serial["summary"]
    async_summary = asynchronous["summary"]
    serial_sec = float(serial_summary["streaming_wall_time_sec"])
    async_sec = float(async_summary["streaming_wall_time_sec"])
    comparison = {
        "comparison_contract": {
            "serial": "same persistent components, strict order, no stage overlap",
            "async": "same persistent components, bounded queues and stage overlap",
            "initialization_excluded_from_speedup": True,
            "recommended_primary_plot": "cumulative_map_completion_time.png",
        },
        "runtime": {
            "serial_streaming_sec": serial_sec,
            "async_streaming_sec": async_sec,
            "serial_streaming_fps": float(serial_summary["num_frames"]) / serial_sec,
            "async_streaming_fps": float(async_summary["num_frames"]) / async_sec,
            "speedup_serial_over_async": serial_sec / async_sec,
            "time_reduction_percent": 100.0 * (serial_sec - async_sec) / serial_sec,
            "serial_initialization_sec": float(serial_summary["initialization_sec"]),
            "async_initialization_sec": float(async_summary["initialization_sec"]),
        },
        "final_rendering": {
            "serial": (serial_summary.get("backend") or {}).get("final_metrics", {}),
            "async": (async_summary.get("backend") or {}).get("final_metrics", {}),
        },
        "final_map": {
            "serial_num_gaussians": (serial_summary.get("backend") or {}).get(
                "final_num_gaussians"
            ),
            "async_num_gaussians": (async_summary.get("backend") or {}).get(
                "final_num_gaussians"
            ),
            "serial_total_iterations": (serial_summary.get("backend") or {}).get(
                "total_iterations"
            ),
            "async_total_iterations": (async_summary.get("backend") or {}).get(
                "total_iterations"
            ),
        },
        "pose_metrics": {
            "serial": serial_summary.get("pose_metrics", {}),
            "async": async_summary.get("pose_metrics", {}),
        },
        "artifacts": {
            "timing_csv": str(output_dir / "timing_comparison.csv"),
            "primary_plot": str(output_dir / "cumulative_map_completion_time.png"),
            "interval_plot": str(output_dir / "per_timestamp_update_interval.png"),
            "latency_plot": str(output_dir / "nominal_sensor_to_map_latency.png"),
            "quality_plots": quality_plots,
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
