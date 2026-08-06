#!/usr/bin/env python3
"""Summarize a serial report run and generate compact report artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def nested_metric(mapping: dict[str, Any], *keys: str) -> float | int | None:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, (int, float)) else None


def plot_metric_curve(
    metrics: list[dict[str, Any]],
    split: str,
    metric: str,
    output: Path,
) -> bool:
    rows = [
        entry
        for entry in metrics
        if str(entry.get("stage", "")).endswith("_after")
        and metric in entry.get(split, {})
    ]
    if not rows:
        return False
    x = [int(entry["frame_index"]) for entry in rows]
    y = [float(entry[split][metric]) for entry in rows]
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(x, y, marker="o", markersize=3)
    plt.xlabel("Frame index / timestamp t")
    ylabel = "PSNR (dB)" if metric == "psnr" else metric.upper()
    plt.ylabel(ylabel)
    plt.title(f"Serial incremental reconstruction: {split} {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return True


def plot_gaussians(metrics: list[dict[str, Any]], output: Path) -> bool:
    rows = [
        entry
        for entry in metrics
        if str(entry.get("stage", "")).endswith("_after")
        and "num_gaussians" in entry
    ]
    if not rows:
        return False
    x = [int(entry["frame_index"]) for entry in rows]
    y = [int(entry["num_gaussians"]) for entry in rows]
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(x, y, marker="o", markersize=3)
    plt.xlabel("Frame index / timestamp t")
    plt.ylabel("Number of Gaussians")
    plt.title("Gaussian map size after each incremental update")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "report"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(run_dir / "execution_benchmark_summary.json")
    resolved = load_json(run_dir / "resolved_execution_benchmark_config.json")
    backend_dir = run_dir / resolved["backend"]["output_name"]
    backend_summary = load_json(backend_dir / "incremental_backend_summary.json")
    metrics = load_json(backend_dir / "metrics_log.json")

    pose = summary.get("pose_metrics", {})
    final_metrics = backend_summary.get("final_metrics", {})
    serial_sec = float(summary["streaming_wall_time_sec"])
    initialization_sec = float(summary["initialization_sec"])
    num_frames = int(summary["num_frames"])

    report = {
        "execution": {
            "mode": summary.get("execution_mode"),
            "num_frames": num_frames,
            "num_train_frames": summary.get("num_train_frames"),
            "num_test_frames": summary.get("num_test_frames"),
            "initialization_sec": initialization_sec,
            "quality_run_streaming_sec": serial_sec,
            "quality_run_streaming_fps": num_frames / serial_sec,
            "warning": (
                "This quality run includes rendering evaluation and must not be "
                "used as the pure speed benchmark."
            ),
        },
        "pose_ate_translation_m": {
            "raw_rmse": nested_metric(
                pose, "raw", "translation_error_m", "rmse"
            ),
            "se3_rmse": nested_metric(
                pose, "se3", "translation_error_m", "rmse"
            ),
            "sim3_rmse": nested_metric(
                pose, "sim3", "translation_error_m", "rmse"
            ),
            "sim3_scale": nested_metric(pose, "sim3", "scale"),
            "num_valid_poses": pose.get("num_valid_poses"),
            "num_skipped_need_interp": pose.get("num_skipped_need_interp"),
        },
        "final_rendering_metrics": final_metrics,
        "map": {
            "total_iterations": backend_summary.get("total_iterations"),
            "final_num_gaussians": backend_summary.get("final_num_gaussians"),
            "num_train_packets": backend_summary.get("num_train_packets"),
            "num_test_cameras": backend_summary.get("num_test_cameras"),
            "num_skipped_invalid_poses": backend_summary.get(
                "num_skipped_invalid_poses"
            ),
        },
    }

    (output_dir / "serial_report_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    flat_rows = [
        ("ATE Raw RMSE", report["pose_ate_translation_m"]["raw_rmse"], "m"),
        ("ATE SE3 RMSE", report["pose_ate_translation_m"]["se3_rmse"], "m"),
        ("ATE Sim3 RMSE", report["pose_ate_translation_m"]["sim3_rmse"], "m"),
        ("Sim3 scale", report["pose_ate_translation_m"]["sim3_scale"], ""),
        (
            "Train PSNR",
            nested_metric(final_metrics, "train_inserted", "psnr"),
            "dB",
        ),
        (
            "Train SSIM",
            nested_metric(final_metrics, "train_inserted", "ssim"),
            "",
        ),
        (
            "Train L1",
            nested_metric(final_metrics, "train_inserted", "l1"),
            "",
        ),
        (
            "Active local map PSNR",
            nested_metric(final_metrics, "active_local_map", "psnr"),
            "dB",
        ),
        (
            "Active local map SSIM",
            nested_metric(final_metrics, "active_local_map", "ssim"),
            "",
        ),
        (
            "Test PSNR",
            nested_metric(final_metrics, "test_all", "psnr"),
            "dB",
        ),
        (
            "Test SSIM",
            nested_metric(final_metrics, "test_all", "ssim"),
            "",
        ),
        (
            "Test L1",
            nested_metric(final_metrics, "test_all", "l1"),
            "",
        ),
        ("Final Gaussians", report["map"]["final_num_gaussians"], ""),
        ("Total iterations", report["map"]["total_iterations"], ""),
    ]
    with (output_dir / "serial_report_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value", "Unit"])
        writer.writerows(flat_rows)

    plots: list[str] = []
    for split in ("train_inserted", "active_local_map", "test_seen"):
        for metric in ("psnr", "ssim", "l1"):
            filename = f"{split}_{metric}_curve.png"
            if plot_metric_curve(metrics, split, metric, output_dir / filename):
                plots.append(filename)
    gaussian_plot = "num_gaussians_curve.png"
    if plot_gaussians(metrics, output_dir / gaussian_plot):
        plots.append(gaussian_plot)

    report["artifacts"] = {
        "table_csv": "serial_report_table.csv",
        "plots": plots,
    }
    (output_dir / "serial_report_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
