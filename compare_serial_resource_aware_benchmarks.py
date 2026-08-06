#!/usr/bin/env python3
"""Compare matched serial and resource-aware asynchronous benchmark outputs."""
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
    parser.add_argument("--resource_aware_dir", required=True)
    parser.add_argument(
        "--output_dir",
        default="outputs/comparison_serial_resource_aware_P000_0_50",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(directory: Path) -> dict[str, Any]:
    return {
        "summary": load_json(directory / "execution_benchmark_summary.json"),
        "timing": load_json(directory / "frame_timing_log.json"),
    }


def train_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in run["timing"]
        if not bool(row.get("is_test", False)) and "backend_end_sec" in row
    ]


def plot_two_lines(
    serial_x: list[float],
    serial_y: list[float],
    resource_x: list[float],
    resource_y: list[float],
    *,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(serial_x, serial_y, marker="o", markersize=3, label="Serial")
    plt.plot(
        resource_x,
        resource_y,
        marker="o",
        markersize=3,
        label="Resource-aware Async",
    )
    plt.xlabel("Frame index / timestamp t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return None if not values else sum(values) / len(values)


def main() -> None:
    args = parse_args()
    serial_dir = Path(args.serial_dir).expanduser().resolve()
    resource_dir = Path(args.resource_aware_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    serial = load_run(serial_dir)
    resource = load_run(resource_dir)
    if resource["summary"].get("execution_mode") != "resource_aware_async":
        raise ValueError(
            "resource_aware_dir does not contain a resource_aware_async run"
        )

    serial_rows = train_rows(serial)
    resource_rows = train_rows(resource)
    serial_by_frame = {int(row["frame_index"]): row for row in serial_rows}
    resource_by_frame = {int(row["frame_index"]): row for row in resource_rows}
    frames = sorted(set(serial_by_frame) & set(resource_by_frame))
    if not frames:
        raise RuntimeError("no common completed train-frame updates")

    plot_two_lines(
        [float(frame) for frame in frames],
        [
            float(serial_by_frame[frame]["map_completion_elapsed_sec"])
            for frame in frames
        ],
        [float(frame) for frame in frames],
        [
            float(resource_by_frame[frame]["map_completion_elapsed_sec"])
            for frame in frames
        ],
        ylabel="Cumulative completion time (s)",
        title="Map availability after each train timestamp",
        output=output_dir / "cumulative_map_completion_time.png",
    )

    interval_frames = [
        frame
        for frame in frames
        if "completion_interval_train_updates_sec" in serial_by_frame[frame]
        and "completion_interval_train_updates_sec" in resource_by_frame[frame]
    ]
    plot_two_lines(
        [float(frame) for frame in interval_frames],
        [
            float(
                serial_by_frame[frame]["completion_interval_train_updates_sec"]
            )
            for frame in interval_frames
        ],
        [float(frame) for frame in interval_frames],
        [
            float(
                resource_by_frame[frame]["completion_interval_train_updates_sec"]
            )
            for frame in interval_frames
        ],
        ylabel="Time between completed map updates (s)",
        title="Per-timestamp map update interval",
        output=output_dir / "per_timestamp_update_interval.png",
    )

    lock_fields = [
        "pose_gpu_gate_wait_sec",
        "packet_gpu_gate_wait_sec",
        "backend_gpu_gate_wait_sec",
        "pose_gpu_gate_held_sec",
        "packet_gpu_gate_held_sec",
        "backend_gpu_gate_held_sec",
    ]
    with (output_dir / "timing_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "frame_index",
            "serial_completion_elapsed_sec",
            "resource_aware_completion_elapsed_sec",
            "serial_train_update_interval_sec",
            "resource_aware_train_update_interval_sec",
            *lock_fields,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            serial_row = serial_by_frame[frame]
            resource_row = resource_by_frame[frame]
            output = {
                "frame_index": frame,
                "serial_completion_elapsed_sec": serial_row.get(
                    "map_completion_elapsed_sec"
                ),
                "resource_aware_completion_elapsed_sec": resource_row.get(
                    "map_completion_elapsed_sec"
                ),
                "serial_train_update_interval_sec": serial_row.get(
                    "completion_interval_train_updates_sec"
                ),
                "resource_aware_train_update_interval_sec": resource_row.get(
                    "completion_interval_train_updates_sec"
                ),
            }
            for field in lock_fields:
                output[field] = resource_row.get(field)
            writer.writerow(output)

    serial_sec = float(serial["summary"]["streaming_wall_time_sec"])
    resource_sec = float(resource["summary"]["streaming_wall_time_sec"])
    speedup = serial_sec / resource_sec
    comparison = {
        "comparison_contract": {
            "serial": "same persistent components with no stage overlap",
            "resource_aware_async": (
                "asynchronous queues and workers with one mutually exclusive "
                "module-level GPU execution gate"
            ),
            "initialization_excluded_from_speedup": True,
        },
        "runtime": {
            "serial_streaming_sec": serial_sec,
            "resource_aware_streaming_sec": resource_sec,
            "speedup_serial_over_resource_aware": speedup,
            "time_reduction_percent": 100.0 * (serial_sec - resource_sec) / serial_sec,
        },
        "resource_aware_gpu_gate": resource["summary"].get("gpu_gate", {}),
        "mean_frame_values": {
            field: mean_present(resource_rows, field) for field in lock_fields
        },
        "consistency": {
            "serial_total_iterations": serial["summary"].get("backend", {}).get(
                "total_iterations"
            ),
            "resource_aware_total_iterations": resource["summary"]
            .get("backend", {})
            .get("total_iterations"),
            "serial_final_num_gaussians": serial["summary"]
            .get("backend", {})
            .get("final_num_gaussians"),
            "resource_aware_final_num_gaussians": resource["summary"]
            .get("backend", {})
            .get("final_num_gaussians"),
        },
        "artifacts": {
            "primary_plot": "cumulative_map_completion_time.png",
            "interval_plot": "per_timestamp_update_interval.png",
            "csv": "timing_comparison.csv",
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
