#!/usr/bin/env python3
"""Summarize reviewer-requested SH003 stage-order ablation across seeds 0/1/2.

Directory contract:
  seed 0:
    <root>/baseline_two_stage_cap001/
    <root>/mixed_from_start_cap001/
  seed 1/2:
    <root>/seed1/<case>/
    <root>/seed2/<case>/

The already-completed no-insertion-cap run, if present at
<root>/two_stage_no_insertion_cap/, is printed separately as a single-run
reference and is NOT included in the 3-seed stage-order statistics.

Online FPS is reconstructed from frame_timing_log.json as

    num_input_frames / max(backend_end_sec)

which excludes component initialization and final Train/Test metric rendering.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


CASES = (
    ("baseline_two_stage_cap001", "Two-stage + cap .01"),
    ("mixed_from_start_cap001", "Mixed from start + cap .01"),
)
SEEDS = (0, 1, 2)
METRICS = (
    "train_psnr",
    "train_ssim",
    "test_psnr",
    "test_ssim",
    "gaussians",
    "online_fps",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _work_dir(root: Path, seed: int, case: str) -> Path:
    return root / case if seed == 0 else root / f"seed{seed}" / case


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


def _read_run(work: Path, case: str, label: str, seed: int | str) -> dict[str, Any]:
    summary_path = work / "execution_benchmark_summary.json"
    if not summary_path.is_file():
        return {
            "case": case,
            "label": label,
            "seed": seed,
            "status": "MISSING",
            "work_dir": str(work),
        }

    summary = _load_json(summary_path)
    backend = summary.get("backend") or {}
    final_metrics = backend.get("final_metrics") or {}
    train = final_metrics.get("train_inserted") or {}
    test = final_metrics.get("test_all") or {}
    fps, wall = _online_fps(work, summary)
    return {
        "case": case,
        "label": label,
        "seed": seed,
        "status": "OK",
        "work_dir": str(work),
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


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "OK"]
    result: dict[str, Any] = {
        "num_available_seeds": len(ok),
        "seeds": [r["seed"] for r in ok],
    }
    for key in METRICS:
        vals = [float(r[key]) for r in ok if r.get(key) is not None]
        result[key] = {
            "mean": statistics.mean(vals) if vals else None,
            # Population standard deviation across the actually-run seeds.
            "std": statistics.pstdev(vals) if len(vals) >= 2 else (0.0 if len(vals) == 1 else None),
        }
    return result


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_pm(stat: dict[str, Any], digits: int) -> str:
    mean = stat.get("mean")
    std = stat.get("std")
    if mean is None:
        return "-"
    if std is None:
        return _fmt(mean, digits)
    return f"{float(mean):.{digits}f}±{float(std):.{digits}f}"


def _print_run(row: dict[str, Any], prefix: str = "") -> None:
    label = f"{prefix}{row['label']}"
    if row.get("status") != "OK":
        print(f"{label:39s} {'MISSING':>21s}")
        return
    train_text = f"{_fmt(row['train_psnr'],3)}/{_fmt(row['train_ssim'],4)}"
    test_text = f"{_fmt(row['test_psnr'],3)}/{_fmt(row['test_ssim'],4)}"
    g = "-" if row.get("gaussians") is None else f"{int(row['gaussians']):,}"
    print(
        f"{label:39s} {train_text:>21s} {test_text:>21s} "
        f"{g:>13s} {_fmt(row.get('online_fps'),4):>10s}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/SH003_0_200_reviewer_stage_order_opacity_cap"),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    all_runs: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, Any] = {}

    print("\nSH003 [0,200) stage-order ablation | strict 8:2 | W20/R30/B100/M50/Th.10 | cap=.01")
    print("Seeds: 0, 1, 2")
    print(
        f"{'Variant':39s} {'Train PSNR/SSIM':>21s} {'Test PSNR/SSIM':>21s} "
        f"{'Gaussians':>13s} {'FPS':>10s}"
    )
    print("-" * 112)

    for case, label in CASES:
        rows = [
            _read_run(_work_dir(root, seed, case), case, label, seed)
            for seed in SEEDS
        ]
        all_runs[case] = rows
        for row in rows:
            _print_run(row, prefix=f"seed{row['seed']} | ")

        agg = _aggregate(rows)
        aggregates[case] = agg
        train_text = (
            f"{_fmt_pm(agg['train_psnr'],3)}/"
            f"{_fmt_pm(agg['train_ssim'],4)}"
        )
        test_text = (
            f"{_fmt_pm(agg['test_psnr'],3)}/"
            f"{_fmt_pm(agg['test_ssim'],4)}"
        )
        g_text = _fmt_pm(agg["gaussians"], 0)
        fps_text = _fmt_pm(agg["online_fps"], 4)
        print(
            f"{'MEAN±STD | ' + label:39s} {train_text:>21s} {test_text:>21s} "
            f"{g_text:>13s} {fps_text:>10s}"
        )
        print()

    # Keep the already-completed opacity-cap ablation visible, but do not mix
    # its single run into the 3-seed stage-order statistics.
    no_cap_work = root / "two_stage_no_insertion_cap"
    no_cap = _read_run(
        no_cap_work,
        "two_stage_no_insertion_cap",
        "Two-stage + no cap",
        "single",
    )
    if no_cap.get("status") == "OK":
        print("Single-run opacity-cap reference (not included in 3-seed statistics):")
        _print_run(no_cap)
        print()

    payload = {
        "protocol": {
            "sequence": "SH003[0,200)",
            "split": "strict 8:2 (160 mapping / 40 held-out)",
            "W": 20,
            "rho": 0.30,
            "B": 100,
            "M": 50,
            "prune_threshold": 0.10,
            "insertion_opacity_cap": 0.01,
            "stage_order_seeds": list(SEEDS),
            "std_definition": "population standard deviation across available seeds",
            "fps_definition": "num_frames / max(frame_timing_log.backend_end_sec)",
        },
        "runs": all_runs,
        "aggregate": aggregates,
        "single_run_no_cap_reference": no_cap,
    }
    output = root / "reviewer_stage_order_3seed_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    print("FPS excludes component initialization and final endpoint metric rendering.")


if __name__ == "__main__":
    main()
