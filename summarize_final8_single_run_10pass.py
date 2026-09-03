#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

SEQUENCES = ["SE000", "SE001", "SE002", "SE003", "SH000", "SH001", "SH002", "SH003"]
ROOT = Path("outputs/final16_10pass_full")
OUT_DIR = Path("outputs")
SEED = 0


def find_single(root: Path, filename: str) -> Path | None:
    matches = list(root.glob(f"*/{filename}"))
    return matches[0] if len(matches) == 1 else None


def load_quality(seq: str) -> dict[str, Any] | None:
    work = ROOT / seq / f"quality_seed{SEED}"
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

    online = endpoint["online"]
    refined = endpoint["global_refined"]
    return {
        "sequence": seq,
        "seed": SEED,
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

    backend_ends = [float(r["backend_end_sec"]) for r in frames if r.get("backend_end_sec") is not None]
    if not backend_ends:
        return None
    online_sec = max(backend_ends)
    num_frames = int(summary["num_frames"])
    refine_sec = float(refine["refinement_wall_time_sec"])
    return {
        "online_fps": num_frames / online_sec,
        "online_wall_time_sec": online_sec,
        "refinement_wall_time_sec": refine_sec,
        "total_wall_time_sec": online_sec + refine_sec,
    }


def f(v: Any, digits: int) -> str:
    return "-" if v is None else f"{float(v):.{digits}f}"


def main() -> None:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for seq in SEQUENCES:
        q = load_quality(seq)
        t = load_timing(seq)
        if q is None:
            missing.append(f"{seq} quality_seed0")
            continue
        if t is None:
            missing.append(f"{seq} timing")

        for stage, prefix in (("Online", "online"), ("+GlobalRefine10p", "refined")):
            row = {
                "sequence": seq,
                "stage": stage,
                "seed": SEED,
                "train_psnr": q[f"{prefix}_train_psnr"],
                "train_ssim": q[f"{prefix}_train_ssim"],
                "train_lpips": q[f"{prefix}_train_lpips"],
                "test_psnr": q[f"{prefix}_test_psnr"],
                "test_ssim": q[f"{prefix}_test_ssim"],
                "test_lpips": q[f"{prefix}_test_lpips"],
                "ate_se3_rmse_m": q["ate_se3_rmse_m"],
                "fps": None,
                "online_wall_time_sec": None,
                "refinement_wall_time_sec": None,
                "total_wall_time_sec": None,
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
    csv_path = OUT_DIR / "final8_single_run_10pass_summary.csv"
    json_path = OUT_DIR / "final8_single_run_10pass_summary.json"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps({"rows": rows, "missing": missing}, indent=2), encoding="utf-8")

    print("\n=== Prioritized 8-sequence single-run result (seed0) ===")
    print("SE000-SE003 + SH000-SH003 | ATE=SE3 RMSE | Global refinement=10 passes")
    header = (
        f"{'Seq':<6} {'Stage':<18} | {'Train P/S/L':<24} | {'Test P/S/L':<24} | "
        f"{'ATE(m)':>8} {'FPS':>8} {'Online(s)':>10} {'Refine(s)':>10} {'Total(s)':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        train = f"{f(r['train_psnr'],3)}/{f(r['train_ssim'],4)}/{f(r['train_lpips'],4)}"
        test = f"{f(r['test_psnr'],3)}/{f(r['test_ssim'],4)}/{f(r['test_lpips'],4)}"
        print(
            f"{r['sequence']:<6} {r['stage']:<18} | {train:<24} | {test:<24} | "
            f"{f(r['ate_se3_rmse_m'],4):>8} {f(r['fps'],3):>8} "
            f"{f(r['online_wall_time_sec'],1):>10} {f(r['refinement_wall_time_sec'],1):>10} "
            f"{f(r['total_wall_time_sec'],1):>10}"
        )

    if missing:
        print("\nMissing/incomplete:")
        for item in missing:
            print(f"  - {item}")
    else:
        print("\nAll 8 sequences have one quality result and one timing result.")

    print(f"\nCSV  : {csv_path}")
    print(f"JSON : {json_path}")


if __name__ == "__main__":
    main()
