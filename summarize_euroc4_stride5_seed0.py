#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

SEQUENCES = ["MH02", "V101", "V201", "MH05"]
ROOT = Path("outputs/euroc4_stride5_10pass")
OUT_DIR = Path("outputs")


def find_single(root: Path, filename: str) -> Path | None:
    matches = list(root.glob(f"*/{filename}"))
    return matches[0] if len(matches) == 1 else None


def online_time_and_fps(work: Path) -> tuple[float, float, int] | None:
    summary_path = work / "execution_benchmark_summary.json"
    frame_path = work / "frame_timing_log.json"
    if not summary_path.is_file() or not frame_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = json.loads(frame_path.read_text(encoding="utf-8"))
    ends = [
        float(row["backend_end_sec"])
        for row in frames
        if row.get("backend_end_sec") is not None
    ]
    if not ends:
        return None
    sec = max(ends)
    n = int(summary["num_frames"])
    return sec, n / sec, n


def load_quality(seq: str) -> dict[str, Any] | None:
    work = ROOT / seq / "quality_seed0"
    endpoint_path = find_single(work, "posthoc_global_refinement_endpoint_metrics.json")
    summary_path = work / "execution_benchmark_summary.json"
    if endpoint_path is None or not summary_path.is_file():
        return None

    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pose = summary.get("pose_metrics") or {}
    ate = (((pose.get("se3") or {}).get("translation_error_m") or {}).get("rmse"))
    if ate is None:
        return None

    backend = summary.get("backend") or {}
    g = backend.get("final_num_gaussians")
    if g is None:
        return None

    qtiming = online_time_and_fps(work)
    online = endpoint["online"]
    refined = endpoint["global_refined"]
    return {
        "num_frames": int(summary["num_frames"]),
        "num_train_frames": int(summary.get("num_train_frames", 0)),
        "num_test_frames": int(summary.get("num_test_frames", 0)),
        "gaussians": int(g),
        "ate_se3_rmse_m": float(ate),
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
        "quality_online_sec_check": None if qtiming is None else qtiming[0],
        "quality_online_fps_check": None if qtiming is None else qtiming[1],
    }


def load_timing(seq: str) -> dict[str, Any] | None:
    work = ROOT / seq / "timing_seed0"
    timing_path = find_single(work, "posthoc_global_refinement_timing.json")
    summary_path = work / "execution_benchmark_summary.json"
    base_timing = online_time_and_fps(work)
    if timing_path is None or not summary_path.is_file() or base_timing is None:
        return None

    refine = json.loads(timing_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    online_sec, fps, n = base_timing
    backend = summary.get("backend") or {}
    g = backend.get("final_num_gaussians")
    refine_sec = float(refine["refinement_wall_time_sec"])
    return {
        "num_frames": n,
        "online_fps": fps,
        "online_wall_time_sec": online_sec,
        "refinement_wall_time_sec": refine_sec,
        "total_wall_time_sec": online_sec + refine_sec,
        "gaussians": None if g is None else int(g),
    }


def f(value: Any, digits: int) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    warnings: list[str] = []

    for seq in SEQUENCES:
        q = load_quality(seq)
        t = load_timing(seq)
        if q is None:
            missing.append(f"{seq} quality_seed0")
            continue
        if t is None:
            missing.append(f"{seq} timing_seed0")

        if t is not None:
            qfps = q.get("quality_online_fps_check")
            if qfps and qfps > 0:
                rel = abs(float(t["online_fps"]) - float(qfps)) / float(qfps)
                if rel > 0.25:
                    warnings.append(
                        f"{seq}: timing FPS {t['online_fps']:.4f} differs from "
                        f"quality-run online FPS {qfps:.4f} by {rel*100:.1f}%"
                    )
            tg = t.get("gaussians")
            if tg and q["gaussians"] > 0:
                relg = abs(int(tg) - int(q["gaussians"])) / int(q["gaussians"])
                if relg > 0.20:
                    warnings.append(
                        f"{seq}: timing G={int(tg)} differs from quality G={q['gaussians']} "
                        f"by {relg*100:.1f}%"
                    )

        for stage, prefix in (("Online", "online"), ("+GlobalRefine10p", "refined")):
            row: dict[str, Any] = {
                "sequence": seq,
                "stage": stage,
                "seed": 0,
                "stride": 5,
                "num_frames": q["num_frames"],
                "num_train_frames": q["num_train_frames"],
                "num_test_frames": q["num_test_frames"],
                "train_psnr": q[f"{prefix}_train_psnr"],
                "train_ssim": q[f"{prefix}_train_ssim"],
                "train_lpips": q[f"{prefix}_train_lpips"],
                "test_psnr": q[f"{prefix}_test_psnr"],
                "test_ssim": q[f"{prefix}_test_ssim"],
                "test_lpips": q[f"{prefix}_test_lpips"],
                "ate_se3_rmse_m": q["ate_se3_rmse_m"],
                "gaussians": q["gaussians"],
                "fps": None,
                "online_wall_time_sec": None,
                "refinement_wall_time_sec": None,
                "total_wall_time_sec": None,
                # Diagnostic only: the quality run has a clean online boundary too.
                "quality_run_online_fps_check": q["quality_online_fps_check"],
            }
            if t is not None:
                row["online_wall_time_sec"] = t["online_wall_time_sec"]
                if stage == "Online":
                    row["fps"] = t["online_fps"]
                    row["total_wall_time_sec"] = t["online_wall_time_sec"]
                else:
                    row["refinement_wall_time_sec"] = t["refinement_wall_time_sec"]
                    row["total_wall_time_sec"] = t["total_wall_time_sec"]
            rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "euroc4_stride5_seed0_summary.csv"
    json_path = OUT_DIR / "euroc4_stride5_seed0_summary.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(
        json.dumps({"rows": rows, "missing": missing, "warnings": warnings}, indent=2),
        encoding="utf-8",
    )

    print("\n=== EuRoC 4-sequence | stride=5 -> strict8:2 | seed0 ===")
    header = (
        f"{'Seq':<5} {'Stage':<18} | {'Train P/S/L':<24} | {'Test P/S/L':<24} | "
        f"{'ATE(m)':>8} {'G(k)':>10} {'FPS':>8} {'Online(s)':>10} "
        f"{'Refine(s)':>10} {'Total(s)':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        train = f"{f(r['train_psnr'],3)}/{f(r['train_ssim'],4)}/{f(r['train_lpips'],4)}"
        test = f"{f(r['test_psnr'],3)}/{f(r['test_ssim'],4)}/{f(r['test_lpips'],4)}"
        gk = float(r["gaussians"]) / 1000.0
        print(
            f"{r['sequence']:<5} {r['stage']:<18} | {train:<24} | {test:<24} | "
            f"{f(r['ate_se3_rmse_m'],4):>8} {gk:>10.1f} {f(r['fps'],3):>8} "
            f"{f(r['online_wall_time_sec'],1):>10} {f(r['refinement_wall_time_sec'],1):>10} "
            f"{f(r['total_wall_time_sec'],1):>10}"
        )

    if warnings:
        print("\nTiming consistency warnings:")
        for item in warnings:
            print(f"  - {item}")
    if missing:
        print("\nMissing/incomplete:")
        for item in missing:
            print(f"  - {item}")
    if not warnings and not missing:
        print("\nAll four sequences are complete; timing/quality consistency checks passed.")

    print(f"\nCSV  : {csv_path}")
    print(f"JSON : {json_path}")


if __name__ == "__main__":
    main()
