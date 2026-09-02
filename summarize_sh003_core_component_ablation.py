#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path("outputs")

CASES = [
    {
        "name": "Feed-forward aggregation only",
        "incremental_optimization": False,
        "aggressive_pruning": False,
        "historical_replay": False,
        "work_dir": ROOT / "SH003_0_200_core_feedforward_aggregation_only",
    },
    {
        "name": "Incremental optimization",
        "incremental_optimization": True,
        "aggressive_pruning": False,
        "historical_replay": False,
        "work_dir": ROOT / "SH003_0_200_core_W20_R0_B100_th0005",
    },
    {
        "name": "+ Aggressive opacity pruning",
        "incremental_optimization": True,
        "aggressive_pruning": True,
        "historical_replay": False,
        "work_dir": ROOT / "SH003_0_200_ablation_rho0_W20_th010",
    },
    {
        "name": "+ Historical replay (standard pruning)",
        "incremental_optimization": True,
        "aggressive_pruning": False,
        "historical_replay": True,
        "work_dir": ROOT / "SH003_0_200_core_W20_R30_B100_th0005",
    },
    {
        "name": "Full method",
        "incremental_optimization": True,
        "aggressive_pruning": True,
        "historical_replay": True,
        "work_dir": ROOT / "SH003_0_200_novelview_8to2_W20_R30_th010",
    },
]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def intermediate_eval_sec(work_dir: Path) -> float:
    # Older ablation runs may contain one intermediate metric evaluation.  It is
    # outside backend_end_sec for that update but delays all subsequent updates,
    # so subtract it from cumulative streaming elapsed time. Final evaluation is
    # never part of metrics_log and already occurs after the last backend_end_sec.
    paths = list(work_dir.glob("*/metrics_log.json"))
    total = 0.0
    for path in paths:
        try:
            rows = load(path)
        except Exception:
            continue
        if isinstance(rows, list):
            for row in rows:
                value = row.get("eval_time_sec") if isinstance(row, dict) else None
                if value is not None:
                    total += float(value)
    return total


def fps_excluding_metric_render(work_dir: Path, num_frames: int) -> float | None:
    path = work_dir / "frame_timing_log.json"
    if not path.is_file():
        return None
    rows = load(path)
    ends = [
        float(row["backend_end_sec"])
        for row in rows
        if isinstance(row, dict) and row.get("backend_end_sec") is not None
    ]
    if not ends:
        return None
    elapsed = max(ends) - intermediate_eval_sec(work_dir)
    if elapsed <= 0:
        return None
    return float(num_frames) / elapsed


def mark(v: bool) -> str:
    return "Y" if v else "-"


def fmt(v: Any, digits: int = 3) -> str:
    return "-" if v is None else f"{float(v):.{digits}f}"


def main() -> None:
    rows: list[dict[str, Any]] = []

    for case in CASES:
        wd: Path = case["work_dir"]
        p = wd / "execution_benchmark_summary.json"
        if not p.is_file():
            rows.append({**case, "status": "missing"})
            continue

        s = load(p)
        backend = s.get("backend") or {}
        fm = backend.get("final_metrics") or {}
        train = fm.get("train_inserted") or {}
        test = fm.get("test_all") or {}
        n = int(s.get("num_frames") or 0)

        rows.append(
            {
                **case,
                "status": "complete",
                "train_psnr": train.get("psnr"),
                "train_ssim": train.get("ssim"),
                "test_psnr": test.get("psnr"),
                "test_ssim": test.get("ssim"),
                "final_gaussians": backend.get("final_num_gaussians"),
                "fps": fps_excluding_metric_render(wd, n),
                "num_frames": n,
            }
        )

    print("\n=== SH003 Core Component Ablation ===")
    print("[0,200), strict 8:2 | component rows use W20/B100; replay rows rho=.30")
    print("FPS excludes initialization and metric-evaluation rendering.\n")

    header = (
        f"{'Configuration':<40} {'Opt':>4} {'Prune':>6} {'Replay':>7} "
        f"{'TrainP':>8} {'TrainS':>8} {'TestP':>8} {'TestS':>8} "
        f"{'G(k)':>10} {'FPS':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        if row.get("status") != "complete":
            print(f"{row['name']:<40} MISSING  {row['work_dir']}")
            continue
        gk = None if row.get("final_gaussians") is None else int(row["final_gaussians"]) / 1000.0
        print(
            f"{row['name']:<40} "
            f"{mark(row['incremental_optimization']):>4} "
            f"{mark(row['aggressive_pruning']):>6} "
            f"{mark(row['historical_replay']):>7} "
            f"{fmt(row.get('train_psnr')):>8} "
            f"{fmt(row.get('train_ssim'),4):>8} "
            f"{fmt(row.get('test_psnr')):>8} "
            f"{fmt(row.get('test_ssim'),4):>8} "
            f"{fmt(gk,1):>10} "
            f"{fmt(row.get('fps'),4):>8}"
        )

    serializable = []
    for row in rows:
        item = dict(row)
        item["work_dir"] = str(item["work_dir"])
        serializable.append(item)

    out_json = ROOT / "SH003_core_component_ablation_summary.json"
    out_csv = ROOT / "SH003_core_component_ablation_summary.csv"
    out_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    fields = [
        "name",
        "incremental_optimization",
        "aggressive_pruning",
        "historical_replay",
        "status",
        "train_psnr",
        "train_ssim",
        "test_psnr",
        "test_ssim",
        "final_gaussians",
        "fps",
        "num_frames",
        "work_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in serializable:
            writer.writerow({key: row.get(key) for key in fields})

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
