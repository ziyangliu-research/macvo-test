#!/usr/bin/env python3
"""Summarize ETH3D mannequin_face_1 online + multi-checkpoint refinement.

Defaults reproduce the original 10/15/20/25/30-pass summary, but both the
output root and checkpoint list can be overridden with environment variables:
  ETH3D_SUMMARY_ROOT
  ETH3D_SUMMARY_CHECKPOINTS
Incomplete runs are reported as MISSING rather than treated as zeros.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("ETH3D_SUMMARY_ROOT", "outputs/eth3d_mannequin_full30"))
SEED = int(os.environ.get("ETH3D_SUMMARY_SEED", "0"))
CHECKPOINTS = tuple(
    sorted(
        {
            int(x.strip())
            for x in os.environ.get(
                "ETH3D_SUMMARY_CHECKPOINTS", "10,15,20,25,30"
            ).split(",")
            if x.strip()
        }
    )
)
if not CHECKPOINTS or any(x <= 0 for x in CHECKPOINTS):
    raise ValueError(f"invalid ETH3D_SUMMARY_CHECKPOINTS: {CHECKPOINTS}")

QUALITY_NAME = f"incremental_ETH3D_mannequin_quality_seed{SEED}"
TIMING_NAME = f"incremental_ETH3D_mannequin_timing_seed{SEED}"


def load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}")
        return None


def online_wall_from_frame_log(path: Path) -> float | None:
    rows = load_json(path)
    if not isinstance(rows, list):
        return None
    ends = [float(row["backend_end_sec"]) for row in rows if "backend_end_sec" in row]
    return max(ends) if ends else None


def metric_triplet(metrics: dict[str, Any] | None) -> tuple[float | None, float | None, float | None]:
    if not isinstance(metrics, dict) or int(metrics.get("num_views", 0)) <= 0:
        return None, None, None
    return (
        float(metrics["psnr"]),
        float(metrics["ssim"]),
        float(metrics["lpips"]),
    )


def fmt(value: float | None, digits: int = 3) -> str:
    return "MISSING" if value is None else f"{value:.{digits}f}"


def fmt_triplet(values: tuple[float | None, float | None, float | None]) -> str:
    p, s, l = values
    if p is None or s is None or l is None:
        return "MISSING"
    return f"{p:.3f}/{s:.4f}/{l:.4f}"


def main() -> None:
    quality_dir = ROOT / f"quality_seed{SEED}"
    timing_dir = ROOT / f"timing_seed{SEED}"

    quality_metrics = load_json(
        quality_dir / QUALITY_NAME / "posthoc_global_refinement_checkpoint_metrics.json"
    )
    quality_summary = load_json(quality_dir / "execution_benchmark_summary.json")
    timing_metrics = load_json(
        timing_dir / TIMING_NAME / "posthoc_global_refinement_timing_checkpoints.json"
    )
    timing_summary = load_json(timing_dir / "execution_benchmark_summary.json")

    missing: list[str] = []
    if quality_metrics is None:
        missing.append("quality checkpoint metrics")
    if quality_summary is None:
        missing.append("quality execution summary")
    if timing_metrics is None:
        missing.append("timing checkpoint metrics")
    if timing_summary is None:
        missing.append("timing execution summary")

    num_frames = None
    if isinstance(timing_summary, dict):
        num_frames = int(timing_summary.get("num_frames", 0)) or None
    if num_frames is None and isinstance(quality_summary, dict):
        num_frames = int(quality_summary.get("num_frames", 0)) or None

    online_wall = online_wall_from_frame_log(timing_dir / "frame_timing_log.json")
    online_fps = (
        float(num_frames) / online_wall
        if num_frames is not None and online_wall is not None and online_wall > 0
        else None
    )
    quality_online_wall = online_wall_from_frame_log(quality_dir / "frame_timing_log.json")
    quality_online_fps = (
        float(num_frames) / quality_online_wall
        if num_frames is not None and quality_online_wall is not None and quality_online_wall > 0
        else None
    )

    ate = None
    g_quality = None
    if isinstance(quality_summary, dict):
        try:
            ate = float(
                quality_summary["pose_metrics"]["se3"]["translation_error_m"]["rmse"]
            )
        except Exception:
            pass
        try:
            g_quality = int(quality_summary["backend"]["final_num_gaussians"])
        except Exception:
            pass

    g_timing = None
    if isinstance(timing_summary, dict):
        try:
            g_timing = int(timing_summary["backend"]["final_num_gaussians"])
        except Exception:
            pass

    if quality_online_fps is not None and online_fps is not None:
        relative = abs(online_fps - quality_online_fps) / max(quality_online_fps, 1e-12)
        if relative > 0.25:
            print(
                f"[warn] timing-vs-quality online FPS differs by {relative*100:.1f}%: "
                f"timing={online_fps:.4f}, quality={quality_online_fps:.4f}"
            )
    if g_quality is not None and g_timing is not None:
        relative = abs(g_timing - g_quality) / max(g_quality, 1)
        if relative > 0.20:
            print(
                f"[warn] timing-vs-quality Gaussian count differs by {relative*100:.1f}%: "
                f"timing={g_timing}, quality={g_quality}"
            )

    rows: list[dict[str, Any]] = []
    online_train = online_test = None
    if isinstance(quality_metrics, dict):
        online = quality_metrics.get("online") or {}
        online_train = online.get("train")
        online_test = online.get("test")

    rows.append(
        {
            "stage": "Online",
            "pass": 0,
            "train": metric_triplet(online_train),
            "test": metric_triplet(online_test),
            "ate_m": ate,
            "gaussians": g_quality,
            "fps": online_fps,
            "online_wall_sec": online_wall,
            "refine_sec": None,
            "total_sec": online_wall,
        }
    )

    checkpoints_payload = (
        quality_metrics.get("checkpoints", {}) if isinstance(quality_metrics, dict) else {}
    )
    timing_payload = (
        timing_metrics.get("checkpoint_cumulative_wall_time_sec", {})
        if isinstance(timing_metrics, dict)
        else {}
    )
    for checkpoint in CHECKPOINTS:
        q = checkpoints_payload.get(str(checkpoint)) if isinstance(checkpoints_payload, dict) else None
        refine_sec = None
        if isinstance(timing_payload, dict) and str(checkpoint) in timing_payload:
            refine_sec = float(timing_payload[str(checkpoint)])
        rows.append(
            {
                "stage": f"+GlobalRefine{checkpoint}p",
                "pass": checkpoint,
                "train": metric_triplet(q.get("train") if isinstance(q, dict) else None),
                "test": metric_triplet(q.get("test") if isinstance(q, dict) else None),
                "ate_m": ate,
                "gaussians": g_quality,
                "fps": None,
                "online_wall_sec": online_wall,
                "refine_sec": refine_sec,
                "total_sec": (
                    online_wall + refine_sec
                    if online_wall is not None and refine_sec is not None
                    else None
                ),
            }
        )

    print(f"\n=== ETH3D mannequin_face_1 | full | strict8:2 | seed{SEED} ===")
    print(f"checkpoints: {list(CHECKPOINTS)}")
    print(
        f"{'Stage':22s} | {'Train P/S/L':24s} | {'Test P/S/L':24s} | "
        f"{'ATE(m)':>8s} {'G(k)':>9s} {'FPS':>8s} {'Online(s)':>10s} {'Refine(s)':>10s} {'Total(s)':>10s}"
    )
    print("-" * 132)
    for row in rows:
        gk = None if row["gaussians"] is None else row["gaussians"] / 1000.0
        print(
            f"{row['stage']:22s} | {fmt_triplet(row['train']):24s} | {fmt_triplet(row['test']):24s} | "
            f"{fmt(row['ate_m'],4):>8s} {fmt(gk,1):>9s} {fmt(row['fps'],3):>8s} "
            f"{fmt(row['online_wall_sec'],1):>10s} {fmt(row['refine_sec'],1):>10s} {fmt(row['total_sec'],1):>10s}"
        )

    csv_path = ROOT / f"summary_seed{SEED}.csv"
    json_path = ROOT / f"summary_seed{SEED}.json"
    ROOT.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage", "pass", "train_psnr", "train_ssim", "train_lpips",
                "test_psnr", "test_ssim", "test_lpips", "se3_ate_rmse_m",
                "gaussians", "online_fps", "online_wall_sec", "refine_sec", "total_sec",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["stage"], row["pass"], *row["train"], *row["test"],
                    row["ate_m"], row["gaussians"], row["fps"],
                    row["online_wall_sec"], row["refine_sec"], row["total_sec"],
                ]
            )

    json_payload = {
        "protocol": "ETH3D mannequin_face_1 full, strict8:2, seed0, W20/rho.30/B100/M50/Th.10",
        "checkpoint_passes": list(CHECKPOINTS),
        "num_frames": num_frames,
        "quality_online_fps_diagnostic": quality_online_fps,
        "missing": missing,
        "rows": rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    if missing:
        print("\nMISSING/incomplete:")
        for item in missing:
            print(f"  - {item}")
    print(f"\nCSV : {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
