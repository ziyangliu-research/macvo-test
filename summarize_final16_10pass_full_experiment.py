#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

SEQUENCES = [
    "SE000", "SE001", "SE002", "SE003", "SE004", "SE005", "SE006", "SE007",
    "SH000", "SH001", "SH002", "SH003", "SH004", "SH005", "SH006", "SH007",
]
SEEDS = [0, 1, 2]
ROOT = Path("outputs/final16_10pass_full")
OUT_DIR = Path("outputs")

QUALITY_METRICS = [
    "train_psnr", "train_ssim", "train_lpips",
    "test_psnr", "test_ssim", "test_lpips", "ate_se3_rmse_m",
]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def fmt_ms(mean: float | None, std: float | None, digits: int) -> str:
    if mean is None or std is None:
        return "-"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def find_single(root: Path, filename: str) -> Path | None:
    matches = list(root.glob(f"*/{filename}"))
    if len(matches) == 1:
        return matches[0]
    return None


def load_quality_run(seq: str, seed: int) -> dict[str, Any] | None:
    work = ROOT / seq / f"quality_seed{seed}"
    endpoint_path = find_single(
        work, "posthoc_global_refinement_endpoint_metrics.json"
    )
    summary_path = work / "execution_benchmark_summary.json"
    if endpoint_path is None or not summary_path.is_file():
        return None

    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pose = summary.get("pose_metrics") or {}
    ate = (
        ((pose.get("se3") or {}).get("translation_error_m") or {}).get("rmse")
    )
    if ate is None:
        return None

    online = endpoint["online"]
    refined = endpoint["global_refined"]
    return {
        "sequence": seq,
        "seed": seed,
        "online_train_psnr": online["train"]["psnr"],
        "online_train_ssim": online["train"]["ssim"],
        "online_train_lpips": online["train"]["lpips"],
        "online_test_psnr": online["test"]["psnr"],
        "online_test_ssim": online["test"]["ssim"],
        "online_test_lpips": online["test"]["lpips"],
        "refined_train_psnr": refined["train"]["psnr"],
        "refined_train_ssim": refined["train"]["ssim"],
        "refined_train_lpips": refined["train"]["lpips"],
        "refined_test_psnr": refined["test"]["psnr"],
        "refined_test_ssim": refined["test"]["ssim"],
        "refined_test_lpips": refined["test"]["lpips"],
        "ate_se3_rmse_m": ate,
        "num_train_views": endpoint.get("num_train_views"),
        "num_test_views": endpoint.get("num_test_views"),
    }


def load_timing(seq: str) -> dict[str, Any] | None:
    work = ROOT / seq / "timing"
    summary_path = work / "execution_benchmark_summary.json"
    frame_path = work / "frame_timing_log.json"
    timing_path = find_single(work, "posthoc_global_refinement_timing.json")
    if not summary_path.is_file() or not frame_path.is_file() or timing_path is None:
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = json.loads(frame_path.read_text(encoding="utf-8"))
    refine = json.loads(timing_path.read_text(encoding="utf-8"))

    backend_ends = [
        float(row["backend_end_sec"])
        for row in frames
        if row.get("backend_end_sec") is not None
    ]
    if not backend_ends:
        return None
    online_sec = max(backend_ends)
    num_frames = int(summary["num_frames"])
    fps = num_frames / online_sec
    refine_sec = float(refine["refinement_wall_time_sec"])

    return {
        "num_frames": num_frames,
        "num_train_frames": int(summary.get("num_train_frames", 0)),
        "num_test_frames": int(summary.get("num_test_frames", 0)),
        "online_fps": fps,
        "online_wall_time_sec": online_sec,
        "refinement_wall_time_sec": refine_sec,
        "online_plus_refinement_sec": online_sec + refine_sec,
        "initialization_sec": float(summary.get("initialization_sec", 0.0)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for seq in SEQUENCES:
        quality_runs: list[dict[str, Any]] = []
        for seed in SEEDS:
            row = load_quality_run(seq, seed)
            if row is None:
                missing.append(f"{seq} quality seed {seed}")
            else:
                quality_runs.append(row)
                raw_rows.append(row)

        timing = load_timing(seq)
        if timing is None:
            missing.append(f"{seq} timing")

        if len(quality_runs) != 3:
            print(f"[incomplete] {seq}: {len(quality_runs)}/3 quality runs")
            continue

        def aggregate(prefix: str, metric: str) -> tuple[float, float]:
            return mean_std([float(r[f"{prefix}_{metric}"]) for r in quality_runs])

        ate_mean, ate_std = mean_std(
            [float(r["ate_se3_rmse_m"]) for r in quality_runs]
        )

        for stage, prefix in (("Online", "online"), ("+GlobalRefine10p", "refined")):
            row: dict[str, Any] = {
                "sequence": seq,
                "stage": stage,
                "num_quality_runs": len(quality_runs),
                "ate_se3_rmse_m_mean": ate_mean,
                "ate_se3_rmse_m_std": ate_std,
            }
            for split in ("train", "test"):
                for metric in ("psnr", "ssim", "lpips"):
                    m, s = aggregate(prefix, f"{split}_{metric}")
                    row[f"{split}_{metric}_mean"] = m
                    row[f"{split}_{metric}_std"] = s

            if timing is not None:
                row["online_wall_time_sec"] = timing["online_wall_time_sec"]
                if stage == "Online":
                    row["fps"] = timing["online_fps"]
                    row["refinement_wall_time_sec"] = None
                    row["total_wall_time_sec"] = timing["online_wall_time_sec"]
                else:
                    row["fps"] = None
                    row["refinement_wall_time_sec"] = timing[
                        "refinement_wall_time_sec"
                    ]
                    row["total_wall_time_sec"] = timing[
                        "online_plus_refinement_sec"
                    ]
            else:
                row["fps"] = None
                row["online_wall_time_sec"] = None
                row["refinement_wall_time_sec"] = None
                row["total_wall_time_sec"] = None

            final_rows.append(row)

    raw_path = OUT_DIR / "final16_10pass_quality_runs_raw.csv"
    if raw_rows:
        with raw_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)

    summary_path = OUT_DIR / "final16_10pass_summary.csv"
    if final_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
            writer.writeheader()
            writer.writerows(final_rows)

    json_path = OUT_DIR / "final16_10pass_summary.json"
    json_path.write_text(
        json.dumps({"rows": final_rows, "missing": missing}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Final 16-sequence 10-pass experiment ===")
    print("Quality = mean ± sample std over 3 runs. ATE = SE3 RMSE.")
    print("Timing run contains no quality/pose evaluation; times exclude initialization.")
    header = (
        f"{'Seq':<6} {'Stage':<18} | {'Train P/S/L':<34} | "
        f"{'Test P/S/L':<34} | {'ATE(m)':<15} | {'FPS':>7} | "
        f"{'Online(s)':>9} {'Refine(s)':>9} {'Total(s)':>9}"
    )
    print(header)
    print("-" * len(header))

    for row in final_rows:
        train = (
            f"{fmt_ms(row['train_psnr_mean'], row['train_psnr_std'], 3)}/"
            f"{fmt_ms(row['train_ssim_mean'], row['train_ssim_std'], 4)}/"
            f"{fmt_ms(row['train_lpips_mean'], row['train_lpips_std'], 4)}"
        )
        test = (
            f"{fmt_ms(row['test_psnr_mean'], row['test_psnr_std'], 3)}/"
            f"{fmt_ms(row['test_ssim_mean'], row['test_ssim_std'], 4)}/"
            f"{fmt_ms(row['test_lpips_mean'], row['test_lpips_std'], 4)}"
        )
        ate = fmt_ms(
            row["ate_se3_rmse_m_mean"], row["ate_se3_rmse_m_std"], 4
        )
        fps = "-" if row.get("fps") is None else f"{row['fps']:.3f}"
        online = (
            "-" if row.get("online_wall_time_sec") is None
            else f"{row['online_wall_time_sec']:.1f}"
        )
        refine = (
            "-" if row.get("refinement_wall_time_sec") is None
            else f"{row['refinement_wall_time_sec']:.1f}"
        )
        total = (
            "-" if row.get("total_wall_time_sec") is None
            else f"{row['total_wall_time_sec']:.1f}"
        )
        print(
            f"{row['sequence']:<6} {row['stage']:<18} | {train:<34} | "
            f"{test:<34} | {ate:<15} | {fps:>7} | "
            f"{online:>9} {refine:>9} {total:>9}"
        )

    if missing:
        print("\nMissing/incomplete runs:")
        for item in missing:
            print(f"  - {item}")
    else:
        print("\nAll 64 requested runs are complete (48 quality + 16 timing).")

    print(f"\nRaw quality runs : {raw_path}")
    print(f"Final summary    : {summary_path}")
    print(f"Summary JSON     : {json_path}")


if __name__ == "__main__":
    main()
