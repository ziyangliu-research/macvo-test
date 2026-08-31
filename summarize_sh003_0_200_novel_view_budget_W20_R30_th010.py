#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CASES = [
    (50, 25, 15, Path("outputs/SH003_0_200_novelview_8to2_W20_R30_B50_th010"), "incremental_SH003_0_200_novelview_8to2_W20_R30_B50_th010"),
    # Reuse the completed W20/R30/B100/Th=.10 result.
    (100, 50, 30, Path("outputs/SH003_0_200_novelview_8to2_W20_R30_th010"), "incremental_SH003_0_200_novelview_8to2_W20_R30_th010"),
    (200, 100, 60, Path("outputs/SH003_0_200_novelview_8to2_W20_R30_B200_th010"), "incremental_SH003_0_200_novelview_8to2_W20_R30_B200_th010"),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: Any, *keys: str) -> Any:
    cur = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fnum(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    rows: list[dict[str, Any]] = []

    for budget, maintenance, history_slots, work_dir, output_name in CASES:
        summary_path = work_dir / "execution_benchmark_summary.json"
        if not summary_path.is_file():
            rows.append({
                "budget": budget,
                "maintenance": maintenance,
                "history_slots": history_slots,
                "status": "missing",
                "work_dir": str(work_dir),
            })
            continue

        summary = load_json(summary_path)
        backend = summary.get("backend") or {}
        metrics = backend.get("final_metrics") or {}
        train = metrics.get("train_inserted") or {}
        test = metrics.get("test_all") or {}
        recent10 = metrics.get("fixed_recent_10") or {}

        timing_path = work_dir / output_name / "timing_log.json"
        timing = load_json(timing_path) if timing_path.is_file() else []
        backend_times = [
            float(row["backend_total_sec"])
            for row in timing
            if row.get("backend_total_sec") is not None
        ]
        tail = backend_times[-50:]

        train_psnr = train.get("psnr")
        test_psnr = test.get("psnr")
        rows.append({
            "budget": budget,
            "maintenance": maintenance,
            "history_slots": history_slots,
            "historical_ratio": 0.30,
            "status": "complete",
            "train_psnr": train_psnr,
            "test_psnr": test_psnr,
            "gap_db": None if train_psnr is None or test_psnr is None else float(train_psnr) - float(test_psnr),
            "train_ssim": train.get("ssim"),
            "test_ssim": test.get("ssim"),
            "fixed_recent10_psnr": recent10.get("psnr"),
            "final_gaussians": backend.get("final_num_gaussians"),
            "wall_sec": summary.get("streaming_wall_time_sec"),
            "quality_protocol_fps": summary.get("streaming_fps_excluding_initialization"),
            "backend_mean_sec": (sum(backend_times) / len(backend_times)) if backend_times else None,
            "backend_last50_mean_sec": (sum(tail) / len(tail)) if tail else None,
            "ate_se3_rmse_m": nested(summary.get("pose_metrics") or {}, "se3", "translation_error_m", "rmse"),
            "work_dir": str(work_dir),
        })

    print("\n=== SH003 0-199 strict 8:2 | W20 rho=.30 Th=.10 | optimization-budget sweep ===\n")
    header = (
        f"{'B':>4} {'M':>4} {'Hist':>5} {'TrainP':>8} {'TestP':>8} {'Gap':>7} "
        f"{'TestS':>8} {'Recent10':>9} {'G(k)':>9} {'Wall(s)':>9} {'Backend':>8} {'Last50':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        if row["status"] != "complete":
            print(f"{row['budget']:>4} {row['maintenance']:>4} {row['history_slots']:>5} {'missing':>8}")
            continue
        gk = None if row.get("final_gaussians") is None else int(row["final_gaussians"]) / 1000.0
        print(
            f"{row['budget']:>4} {row['maintenance']:>4} {row['history_slots']:>5} "
            f"{fnum(row.get('train_psnr')):>8} {fnum(row.get('test_psnr')):>8} "
            f"{fnum(row.get('gap_db')):>7} {fnum(row.get('test_ssim'),4):>8} "
            f"{fnum(row.get('fixed_recent10_psnr')):>9} {fnum(gk,1):>9} "
            f"{fnum(row.get('wall_sec'),1):>9} {fnum(row.get('backend_mean_sec'),3):>8} "
            f"{fnum(row.get('backend_last50_mean_sec'),3):>8}"
        )

    complete = [r for r in rows if r.get("status") == "complete" and r.get("test_psnr") is not None]
    if complete:
        best = max(complete, key=lambda r: float(r["test_psnr"]))
        print(f"\nBest held-out Test PSNR: B={best['budget']}, PSNR={best['test_psnr']:.3f} dB")

        by_budget = {int(r["budget"]): r for r in complete}
        if 50 in by_budget and 100 in by_budget:
            print(
                f"B50 -> B100: Test {float(by_budget[100]['test_psnr']) - float(by_budget[50]['test_psnr']):+.3f} dB, "
                f"wall {float(by_budget[100]['wall_sec']) - float(by_budget[50]['wall_sec']):+.1f} s"
            )
        if 100 in by_budget and 200 in by_budget:
            print(
                f"B100 -> B200: Test {float(by_budget[200]['test_psnr']) - float(by_budget[100]['test_psnr']):+.3f} dB, "
                f"wall {float(by_budget[200]['wall_sec']) - float(by_budget[100]['wall_sec']):+.1f} s"
            )

    out_json = Path("outputs/SH003_0_200_novelview_8to2_W20_R30_budget_summary.json")
    out_csv = Path("outputs/SH003_0_200_novelview_8to2_W20_R30_budget_summary.csv")
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fields = [
        "budget", "maintenance", "history_slots", "historical_ratio", "status",
        "train_psnr", "test_psnr", "gap_db", "train_ssim", "test_ssim",
        "fixed_recent10_psnr", "final_gaussians", "wall_sec", "quality_protocol_fps",
        "backend_mean_sec", "backend_last50_mean_sec", "ate_se3_rmse_m", "work_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
