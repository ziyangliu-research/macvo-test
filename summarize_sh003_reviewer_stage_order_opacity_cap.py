#!/usr/bin/env python3
"""Summarize the reviewer-requested SH003 stage-order ablation.

Online FPS is reconstructed from frame_timing_log.json as

    num_input_frames / max(backend_end_sec)

which excludes component initialization and final Train/Test metric rendering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASES = (
    ("baseline_two_stage_cap001", "Two-stage + cap .01"),
    ("mixed_from_start_cap001", "Mixed from start + cap .01"),
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _online_fps(work_dir: Path, summary: dict[str, Any]) -> tuple[float | None, float | None]:
    timing_path = work_dir / "frame_timing_log.json"
    if not timing_path.is_file():
        return None, None
    rows = _load_json(timing_path)
    ends = [
        float(row["backend_end_sec"])
        for row in rows
        if isinstance(row, dict) and "backend_end_sec" in row
    ]
    if not ends:
        return None, None
    wall = max(ends)
    nframes = int(summary.get("num_frames", len(rows)))
    if wall <= 0 or nframes <= 0:
        return None, wall
    return nframes / wall, wall


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/SH003_0_200_reviewer_stage_order_opacity_cap"),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    records: list[dict[str, Any]] = []
    for case, label in CASES:
        work = root / case
        summary_path = work / "execution_benchmark_summary.json"
        if not summary_path.is_file():
            records.append({"case": case, "label": label, "status": "MISSING"})
            continue

        summary = _load_json(summary_path)
        backend = summary.get("backend") or {}
        final_metrics = backend.get("final_metrics") or {}
        train = final_metrics.get("train_inserted") or {}
        test = final_metrics.get("test_all") or {}
        fps, wall = _online_fps(work, summary)
        records.append(
            {
                "case": case,
                "label": label,
                "status": "OK",
                "num_frames": int(summary.get("num_frames", 0)),
                "num_train": int(summary.get("num_train_frames", 0)),
                "num_test": int(summary.get("num_test_frames", 0)),
                "train_psnr": train.get("psnr"),
                "train_ssim": train.get("ssim"),
                "test_psnr": test.get("psnr"),
                "test_ssim": test.get("ssim"),
                "gaussians": backend.get("final_num_gaussians"),
                "online_fps": fps,
                "online_wall_sec": wall,
            }
        )

    print("\nSH003 [0,200) stage-order ablation | strict 8:2 | W20/R30/B100/M50/Th.10 | cap=.01")
    print(
        f"{'Variant':31s} {'Train P/S':>19s} {'Test P/S':>19s} "
        f"{'G':>12s} {'FPS':>9s}"
    )
    print("-" * 96)
    for row in records:
        if row["status"] != "OK":
            print(f"{row['label']:31s} {'MISSING':>19s}")
            continue
        train_text = f"{_fmt(row['train_psnr'],3)}/{_fmt(row['train_ssim'],4)}"
        test_text = f"{_fmt(row['test_psnr'],3)}/{_fmt(row['test_ssim'],4)}"
        g = "-" if row["gaussians"] is None else f"{int(row['gaussians']):,}"
        print(
            f"{row['label']:31s} {train_text:>19s} {test_text:>19s} "
            f"{g:>12s} {_fmt(row['online_fps'],4):>9s}"
        )

    output = root / "reviewer_ablation_summary.json"
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nSaved: {output}")
    print("FPS definition: num_frames / max(frame_timing_log.backend_end_sec); ")
    print("excludes initialization and final endpoint metric rendering.")


if __name__ == "__main__":
    main()
