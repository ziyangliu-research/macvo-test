#!/usr/bin/env python3
"""Summarize completed SH003 component-ablation runs; show MISSING otherwise.

By default this inspects the extra-seed runs under
outputs/SH003_0_200_component_ablation_3seed/seed1 and seed2.

A run is considered complete only when execution_benchmark_summary.json exists.
Incomplete/interrupted runs are reported as MISSING and are never guessed from
partial logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASES = (
    ("ff_only", "FF insertion only"),
    ("incremental", "Incremental optimization"),
    ("strong_prune", "Inc. + strong pruning"),
    ("historical", "Inc. + historical views"),
    ("full", "Full"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def online_fps(work_dir: Path, summary: dict[str, Any]) -> float | None:
    timing_path = work_dir / "frame_timing_log.json"
    if not timing_path.is_file():
        return None
    rows = load_json(timing_path)
    ends = [
        float(row["backend_end_sec"])
        for row in rows
        if isinstance(row, dict) and row.get("backend_end_sec") is not None
    ]
    if not ends:
        return None
    wall = max(ends)
    nframes = int(summary.get("num_frames", len(rows)))
    if wall <= 0.0 or nframes <= 0:
        return None
    return nframes / wall


def fmt(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/SH003_0_200_component_ablation_3seed"),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Seed directories to inspect; default: 1 2",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    records: list[dict[str, Any]] = []

    print("\nSH003 [0,200) component-ablation progress")
    print("strict 8:2 | W20/B100/M50 | cap=.01 | completed iff execution_benchmark_summary.json exists")
    print()
    print(
        f"{'Seed':>4s}  {'Configuration':28s}  {'Train P/S':>18s}  {'Test P/S':>18s}  "
        f"{'G(M)':>9s}  {'FPS':>8s}  {'Status':>8s}"
    )
    print("-" * 109)

    for seed in args.seeds:
        for mode, label in CASES:
            work = root / f"seed{seed}" / mode
            summary_path = work / "execution_benchmark_summary.json"
            if not summary_path.is_file():
                row = {
                    "seed": seed,
                    "mode": mode,
                    "label": label,
                    "status": "MISSING",
                    "work_dir": str(work),
                }
                records.append(row)
                print(
                    f"{seed:>4d}  {label:28s}  {'-':>18s}  {'-':>18s}  "
                    f"{'-':>9s}  {'-':>8s}  {'MISSING':>8s}"
                )
                continue

            summary = load_json(summary_path)
            backend = summary.get("backend") or {}
            final_metrics = backend.get("final_metrics") or {}
            train = final_metrics.get("train_inserted") or {}
            test = final_metrics.get("test_all") or {}
            gaussians = backend.get("final_num_gaussians")
            fps = online_fps(work, summary)

            train_text = f"{fmt(train.get('psnr'), 3)}/{fmt(train.get('ssim'), 4)}"
            test_text = f"{fmt(test.get('psnr'), 3)}/{fmt(test.get('ssim'), 4)}"
            g_m = None if gaussians is None else float(gaussians) / 1_000_000.0

            row = {
                "seed": seed,
                "mode": mode,
                "label": label,
                "status": "OK",
                "train_psnr": train.get("psnr"),
                "train_ssim": train.get("ssim"),
                "test_psnr": test.get("psnr"),
                "test_ssim": test.get("ssim"),
                "gaussians": gaussians,
                "gaussians_m": g_m,
                "online_fps": fps,
                "work_dir": str(work),
            }
            records.append(row)
            print(
                f"{seed:>4d}  {label:28s}  {train_text:>18s}  {test_text:>18s}  "
                f"{fmt(g_m, 3):>9s}  {fmt(fps, 4):>8s}  {'OK':>8s}"
            )

        print()

    out = root / "component_ablation_progress_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")

    complete = sum(r["status"] == "OK" for r in records)
    total = len(records)
    print(f"Completed: {complete}/{total}")
    print(f"Saved: {out}")
    print("FPS = num input frames / max(frame_timing_log.backend_end_sec)")
    print("      (excludes initialization and final endpoint metric rendering)")


if __name__ == "__main__":
    main()
