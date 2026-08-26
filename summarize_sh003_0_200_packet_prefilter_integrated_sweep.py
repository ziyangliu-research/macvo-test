#!/usr/bin/env python3
"""Summarize SH003 0-199 integrated local-packet prefilter experiments.

Reads the three quality runs (strict 80/20 held-out split) and, when available,
the three all-frame timing runs produced by
run_sh003_0_200_packet_prefilter_integrated_sweep.sh.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CONFIGS = (("0.03", "003"), ("0.05", "005"), ("0.07", "007"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def find_backend_timing(work_dir: Path) -> Path | None:
    matches = sorted(work_dir.glob("incremental_*/timing_log.json"))
    return matches[0] if matches else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    summary_rows: list[dict[str, Any]] = []

    for threshold, tag in CONFIGS:
        quality_dir = OUTPUTS / f"SH003_0_200_packet_prefilter_p{tag}_quality"
        timing_dir = OUTPUTS / f"SH003_0_200_packet_prefilter_p{tag}_timing"

        row: dict[str, Any] = {"threshold": float(threshold)}

        quality_summary_path = quality_dir / "execution_benchmark_summary.json"
        if quality_summary_path.is_file():
            q = load_json(quality_summary_path)
            backend = q.get("backend") or {}
            metrics = backend.get("final_metrics") or {}
            train = metrics.get("train_inserted") or {}
            test = metrics.get("test_all") or {}
            train_psnr = maybe_float(train.get("psnr"))
            test_psnr = maybe_float(test.get("psnr"))
            row.update(
                {
                    "train_psnr": train_psnr,
                    "train_ssim": maybe_float(train.get("ssim")),
                    "test_psnr": test_psnr,
                    "test_ssim": maybe_float(test.get("ssim")),
                    "generalization_gap_db": (
                        None
                        if train_psnr is None or test_psnr is None
                        else train_psnr - test_psnr
                    ),
                    "quality_final_gaussians": backend.get("final_num_gaussians"),
                    "quality_train_packets": backend.get("num_train_packets"),
                    "quality_test_cameras": backend.get("num_test_cameras"),
                }
            )
            pose = q.get("pose_metrics") or {}
            row["ate_se3_rmse_m"] = maybe_float(
                (((pose.get("se3") or {}).get("translation_error_m") or {}).get("rmse"))
            )

        timing_summary_path = timing_dir / "execution_benchmark_summary.json"
        if timing_summary_path.is_file():
            t = load_json(timing_summary_path)
            backend = t.get("backend") or {}
            row.update(
                {
                    "fps": maybe_float(t.get("streaming_fps_excluding_initialization")),
                    "streaming_wall_sec": maybe_float(t.get("streaming_wall_time_sec")),
                    "timing_final_gaussians": backend.get("final_num_gaussians"),
                    "peak_gpu_allocated_gb": maybe_float(
                        ((backend.get("gpu_memory") or {}).get("gpu_peak_memory_allocated_gb"))
                    ),
                }
            )

            fast_csv = timing_dir / "local_packet_prefilter_fast_summary.csv"
            if fast_csv.is_file():
                fast = csv_rows(fast_csv)
                prune = [float(x["pruned_ratio"]) for x in fast]
                out_g = [int(float(x["output_gaussians"])) for x in fast]
                total = [float(x["prefilter_total_sec"]) for x in fast]
                pre = [float(x["pre_optimization_sec"]) for x in fast]
                post = [float(x["post_optimization_sec"]) for x in fast]
                pruning = [float(x["prune_sec"]) for x in fast]
                row.update(
                    {
                        "local_pruned_mean_pct": 100.0 * mean(prune),
                        "local_pruned_median_pct": 100.0 * median(prune),
                        "local_output_gaussians_mean": mean(out_g),
                        "local_output_gaussians_median": median(out_g),
                        "local_prefilter_mean_sec": mean(total),
                        "local_preopt_mean_sec": mean(pre),
                        "local_prune_mean_sec": mean(pruning),
                        "local_postopt_mean_sec": mean(post),
                    }
                )

            timing_log_path = find_backend_timing(timing_dir)
            if timing_log_path is not None:
                events = load_json(timing_log_path)
                if events:
                    resplat = [float(x["resplat_inference_sec"]) for x in events]
                    backend_sec = [float(x["backend_total_sec"]) for x in events]
                    global_opt = [
                        float(x["local_optimization_and_maintenance_sec"]) for x in events
                    ]
                    row.update(
                        {
                            "resplat_mean_sec": mean(resplat),
                            "global_backend_mean_sec": mean(backend_sec),
                            "global_opt_maintenance_mean_sec": mean(global_opt),
                        }
                    )

        summary_rows.append(row)

    headers = [
        "Th",
        "Test PSNR",
        "Test SSIM",
        "Train PSNR",
        "Gap dB",
        "ATE SE3",
        "Quality G(k)",
        "Local prune %",
        "Local G(k)",
        "Local sec",
        "Global sec",
        "FPS",
        "Timing G(k)",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in summary_rows:
        values = [
            f"{row['threshold']:.2f}",
            fmt(row.get("test_psnr")),
            fmt(row.get("test_ssim"), 4),
            fmt(row.get("train_psnr")),
            fmt(row.get("generalization_gap_db")),
            fmt(row.get("ate_se3_rmse_m"), 4),
            fmt(
                None
                if row.get("quality_final_gaussians") is None
                else float(row["quality_final_gaussians"]) / 1000.0,
                1,
            ),
            fmt(row.get("local_pruned_mean_pct"), 2),
            fmt(
                None
                if row.get("local_output_gaussians_mean") is None
                else float(row["local_output_gaussians_mean"]) / 1000.0,
                1,
            ),
            fmt(row.get("local_prefilter_mean_sec"), 3),
            fmt(row.get("global_backend_mean_sec"), 3),
            fmt(row.get("fps"), 3),
            fmt(
                None
                if row.get("timing_final_gaussians") is None
                else float(row["timing_final_gaussians"]) / 1000.0,
                1,
            ),
        ]
        print(" | ".join(values))

    out = OUTPUTS / "SH003_0_200_packet_prefilter_integrated_sweep_summary.json"
    out.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
