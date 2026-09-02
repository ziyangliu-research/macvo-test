#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

OUTPUT_ROOT = Path("outputs")
WINDOWS = [5, 10, 20]
REPLAYS = [0, 5, 10, 20, 30, 40, 50]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def work_dir_for(window: int, replay: int) -> Path:
    if replay == 0:
        return OUTPUT_ROOT / f"SH003_0_200_ablation_rho0_W{window}_th010"
    if window == 10 and replay == 30:
        # Legacy cell reused by the original W x replay grid.
        return OUTPUT_ROOT / "SH003_0_200_novelview_8to2_replay30_th010"
    return OUTPUT_ROOT / f"SH003_0_200_novelview_8to2_W{window}_R{replay}_th010"


def metric_eval_time_sec(work_dir: Path) -> float:
    """Return only in-stream metric-rendering time recorded by metrics_log.json.

    Final Train/Test evaluation is not in metrics_log and occurs after the last
    backend_end_sec, so it is already excluded from the FPS clock below.
    """
    paths = list(work_dir.rglob("metrics_log.json"))
    if not paths:
        return 0.0
    total = 0.0
    for path in paths:
        data = load_json(path)
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict) and row.get("eval_time_sec") is not None:
                    total += float(row["eval_time_sec"])
    return total


def clean_fps(work_dir: Path, num_frames: int) -> tuple[float | None, float | None, float]:
    """FPS from first streamed frame to last backend completion, minus metric eval.

    Initialization is excluded because TimingRecorder's origin is created after
    persistent component initialization.  Final metric rendering is after the
    last backend_end_sec.  Intermediate metric rendering is explicitly removed
    using metrics_log eval_time_sec.
    """
    timing_path = work_dir / "frame_timing_log.json"
    if not timing_path.is_file():
        return None, None, 0.0
    rows = load_json(timing_path)
    ends = [
        float(row["backend_end_sec"])
        for row in rows
        if isinstance(row, dict) and row.get("backend_end_sec") is not None
    ]
    if not ends:
        return None, None, 0.0
    raw_elapsed = max(ends)
    eval_sec = metric_eval_time_sec(work_dir)
    clean_elapsed = max(raw_elapsed - eval_sec, 1e-12)
    return float(num_frames) / clean_elapsed, clean_elapsed, eval_sec


def nested(value: Any, *keys: str) -> Any:
    cur = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def collect(window: int, replay: int) -> dict[str, Any]:
    work_dir = work_dir_for(window, replay)
    summary_path = work_dir / "execution_benchmark_summary.json"
    base = {
        "window": window,
        "replay_percent": replay,
        "replay_ratio": replay / 100.0,
        "history_steps": replay,
        "recent_steps": 100 - replay,
        "work_dir": str(work_dir),
    }
    if not summary_path.is_file():
        return {**base, "status": "missing"}

    summary = load_json(summary_path)
    backend = summary.get("backend") or {}
    metrics = backend.get("final_metrics") or {}
    train = metrics.get("train_inserted") or {}
    test = metrics.get("test_all") or {}
    num_frames = int(summary.get("num_frames") or 200)
    num_updates = int(summary.get("num_backend_updates") or 0)
    fps, clean_elapsed, eval_sec = clean_fps(work_dir, num_frames)

    return {
        **base,
        "status": "complete",
        "num_frames": num_frames,
        "train_views": train.get("num_views"),
        "test_views": test.get("num_views"),
        "max_map_pct": None if num_frames <= 0 else 100.0 * num_updates / num_frames,
        "train_psnr": train.get("psnr"),
        "train_ssim": train.get("ssim"),
        "test_psnr": test.get("psnr"),
        "test_ssim": test.get("ssim"),
        "final_gaussians": backend.get("final_num_gaussians"),
        "fps_no_metric_render": fps,
        "stream_elapsed_no_metric_sec": clean_elapsed,
        "subtracted_intermediate_eval_sec": eval_sec,
        "ate_se3_rmse_m": nested(
            summary.get("pose_metrics") or {},
            "se3",
            "translation_error_m",
            "rmse",
        ),
    }


def fnum(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    rows = [collect(w, r) for w in WINDOWS for r in REPLAYS]

    print("\nSH003 [0,200), strict 8:2 = 160 mapping/train + 40 held-out test")
    print("Fixed: Th=.10, B=100, M=50")
    print("FPS excludes initialization, final metric rendering, and recorded intermediate metric rendering.\n")

    header = (
        f"{'W':>3} {'rho':>6} {'H':>4} {'Recent':>6} "
        f"{'TrainP':>8} {'TrainS':>8} {'TestP':>8} {'TestS':>8} "
        f"{'G(k)':>9} {'FPS':>8} {'ATE(m)':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        if row["status"] != "complete":
            print(
                f"{row['window']:>3} {row['replay_percent']:>5}% "
                f"{'MISSING':>8}  {row['work_dir']}"
            )
            continue
        gk = None if row.get("final_gaussians") is None else int(row["final_gaussians"]) / 1000.0
        print(
            f"{row['window']:>3} {row['replay_percent']:>5}% "
            f"{row['history_steps']:>4} {row['recent_steps']:>6} "
            f"{fnum(row.get('train_psnr')):>8} {fnum(row.get('train_ssim'),4):>8} "
            f"{fnum(row.get('test_psnr')):>8} {fnum(row.get('test_ssim'),4):>8} "
            f"{fnum(gk,1):>9} {fnum(row.get('fps_no_metric_render'),4):>8} "
            f"{fnum(row.get('ate_se3_rmse_m'),4):>8}"
        )

    print("\nHeld-out Test PSNR matrix:")
    print("       " + " ".join(f"{r:>8}%" for r in REPLAYS))
    for w in WINDOWS:
        values = []
        for r in REPLAYS:
            row = next(x for x in rows if x["window"] == w and x["replay_percent"] == r)
            values.append(fnum(row.get("test_psnr"), 3).rjust(9))
        print(f"W{w:<2}   " + "".join(values))

    print("\nFPS matrix (metric rendering excluded):")
    print("       " + " ".join(f"{r:>8}%" for r in REPLAYS))
    for w in WINDOWS:
        values = []
        for r in REPLAYS:
            row = next(x for x in rows if x["window"] == w and x["replay_percent"] == r)
            values.append(fnum(row.get("fps_no_metric_render"), 4).rjust(9))
        print(f"W{w:<2}   " + "".join(values))

    out_csv = OUTPUT_ROOT / "SH003_0_200_WxReplay_full_grid_with_fps.csv"
    out_json = OUTPUT_ROOT / "SH003_0_200_WxReplay_full_grid_with_fps.json"

    fields = [
        "window", "replay_percent", "replay_ratio", "history_steps", "recent_steps",
        "status", "num_frames", "train_views", "test_views", "max_map_pct",
        "train_psnr", "train_ssim", "test_psnr", "test_ssim",
        "final_gaussians", "fps_no_metric_render", "stream_elapsed_no_metric_sec",
        "subtracted_intermediate_eval_sec", "ate_se3_rmse_m", "work_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})

    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
