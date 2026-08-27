#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RUNS = [
    (0, "baseline_no_replay"),
    (10, "replay10"),
    (20, "replay20"),
    (30, "replay30"),
    (40, "replay40"),
    (50, "replay50"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: dict[str, Any], *keys: str):
    cur: Any = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fnum(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def main() -> None:
    rows: list[dict[str, Any]] = []
    for replay_percent, name in RUNS:
        work_dir = Path(f"outputs/SH003_0_200_novelview_8to2_{name}")
        summary_path = work_dir / "execution_benchmark_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"missing {summary_path}; run the replay-budget sweep first"
            )

        summary = load_json(summary_path)
        backend = summary["backend"]
        metrics = backend.get("final_metrics", {})
        train = metrics.get("train_inserted", {})
        test = metrics.get("test_all", {})
        active = metrics.get("active_local_map", {})
        train_psnr = train.get("psnr")
        test_psnr = test.get("psnr")
        pose = summary.get("pose_metrics", {})
        ate = nested(pose, "se3", "translation_error_m", "rmse")

        rows.append(
            {
                "replay_percent": replay_percent,
                "name": name,
                "train_views": train.get("num_views"),
                "test_views": test.get("num_views"),
                "train_psnr": train_psnr,
                "train_ssim": train.get("ssim"),
                "test_psnr": test_psnr,
                "test_ssim": test.get("ssim"),
                "active_psnr": active.get("psnr"),
                "active_ssim": active.get("ssim"),
                "generalization_gap_db": (
                    None
                    if train_psnr is None or test_psnr is None
                    else float(train_psnr) - float(test_psnr)
                ),
                "final_gaussians": backend.get("final_num_gaussians"),
                "ate_se3_rmse_m": ate,
                "streaming_wall_sec": summary.get("streaming_wall_time_sec"),
                "quality_protocol_fps": summary.get(
                    "streaming_fps_excluding_initialization"
                ),
            }
        )

    print("\n=== SH003 0-199 strict 8:2 held-out novel-view replay-budget sweep ===\n")
    header = (
        f"{'Replay':>6} {'Train PSNR':>10} {'Test PSNR':>10} {'Gap':>8} "
        f"{'Active':>9} {'Train SSIM':>10} {'Test SSIM':>10} "
        f"{'G(k)':>10} {'Wall(s)':>10} {'ATE(m)':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        gk = (
            None
            if row["final_gaussians"] is None
            else float(row["final_gaussians"]) / 1000.0
        )
        print(
            f"{row['replay_percent']:>5}% "
            f"{fnum(row['train_psnr']):>10} {fnum(row['test_psnr']):>10} "
            f"{fnum(row['generalization_gap_db']):>8} "
            f"{fnum(row['active_psnr']):>9} "
            f"{fnum(row['train_ssim'],4):>10} {fnum(row['test_ssim'],4):>10} "
            f"{fnum(gk,1):>10} {fnum(row['streaming_wall_sec'],1):>10} "
            f"{fnum(row['ate_se3_rmse_m'],4):>9}"
        )

    valid_test = [r for r in rows if r["test_psnr"] is not None]
    if valid_test:
        best = max(valid_test, key=lambda r: float(r["test_psnr"]))
        print(
            "\nBest held-out Test PSNR: "
            f"replay={best['replay_percent']}% -> {float(best['test_psnr']):.4f} dB"
        )

    out_json = Path("outputs/SH003_0_200_novelview_8to2_replay_budget_summary.json")
    out_csv = Path("outputs/SH003_0_200_novelview_8to2_replay_budget_summary.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
