#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RUNS = [
    (0.005, Path("outputs/SH003_0_200_novelview_8to2_replay30")),
    (0.010, Path("outputs/SH003_0_200_novelview_8to2_replay30_th001")),
    (0.020, Path("outputs/SH003_0_200_novelview_8to2_replay30_th002")),
    (0.030, Path("outputs/SH003_0_200_novelview_8to2_replay30_th003")),
    (0.050, Path("outputs/SH003_0_200_novelview_8to2_replay30_th005")),
    (0.100, Path("outputs/SH003_0_200_novelview_8to2_replay30_th010")),
]


def load_json(path: Path) -> Any:
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


def count_empty_maintenance(work_dir: Path, backend_output_name: str | None) -> int:
    if not backend_output_name:
        return 0
    path = work_dir / backend_output_name / "maintenance_log.json"
    if not path.is_file():
        return 0
    events = load_json(path)
    if not isinstance(events, list):
        return 0
    return sum(1 for event in events if int(event.get("count_after", -1)) == 0)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for threshold, work_dir in RUNS:
        summary_path = work_dir / "execution_benchmark_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing {summary_path}")
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

        resolved_path = work_dir / "resolved_execution_benchmark_config.json"
        output_name = None
        if resolved_path.is_file():
            resolved = load_json(resolved_path)
            output_name = nested(resolved, "backend", "output_name")

        row = {
            "threshold": threshold,
            "train_views": train.get("num_views"),
            "test_views": test.get("num_views"),
            "train_psnr": train_psnr,
            "test_psnr": test_psnr,
            "generalization_gap_db": (
                None if train_psnr is None or test_psnr is None
                else float(train_psnr) - float(test_psnr)
            ),
            "active_psnr": active.get("psnr"),
            "train_ssim": train.get("ssim"),
            "test_ssim": test.get("ssim"),
            "active_ssim": active.get("ssim"),
            "final_gaussians": backend.get("final_num_gaussians"),
            "wall_sec": summary.get("streaming_wall_time_sec"),
            "quality_protocol_fps": summary.get("streaming_fps_excluding_initialization"),
            "ate_se3_rmse_m": ate,
            "empty_maintenance_events": count_empty_maintenance(work_dir, output_name),
        }
        rows.append(row)

    print("\n=== SH003 0-199 strict 8:2 | replay30 | global prune threshold sweep ===\n")
    header = (
        f"{'Th':>6} {'Train PSNR':>10} {'Test PSNR':>10} {'Gap':>8} {'Active':>9} "
        f"{'Train SSIM':>10} {'Test SSIM':>10} {'G(k)':>10} {'Wall(s)':>10} "
        f"{'Empty':>6} {'ATE(m)':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        gk = None if row["final_gaussians"] is None else float(row["final_gaussians"]) / 1000.0
        print(
            f"{row['threshold']:>6.3f} "
            f"{fnum(row['train_psnr']):>10} {fnum(row['test_psnr']):>10} "
            f"{fnum(row['generalization_gap_db']):>8} {fnum(row['active_psnr']):>9} "
            f"{fnum(row['train_ssim'],4):>10} {fnum(row['test_ssim'],4):>10} "
            f"{fnum(gk,1):>10} {fnum(row['wall_sec'],1):>10} "
            f"{str(row['empty_maintenance_events']):>6} {fnum(row['ate_se3_rmse_m'],4):>9}"
        )

    valid = [r for r in rows if r["test_psnr"] is not None]
    if valid:
        best = max(valid, key=lambda r: float(r["test_psnr"]))
        print(
            "\nBest held-out Test PSNR: "
            f"Th={best['threshold']:.3f}, PSNR={float(best['test_psnr']):.3f} dB, "
            f"G={int(best['final_gaussians']) if best['final_gaussians'] is not None else '-'}"
        )

    baseline = rows[0]
    print("\n=== Delta vs Th=0.005 ===")
    for row in rows[1:]:
        dg = None
        dp = None
        if baseline["final_gaussians"] and row["final_gaussians"] is not None:
            dg = int(row["final_gaussians"]) - int(baseline["final_gaussians"])
            dp = 100.0 * (float(row["final_gaussians"]) / float(baseline["final_gaussians"]) - 1.0)
        dtest = None if row["test_psnr"] is None else float(row["test_psnr"]) - float(baseline["test_psnr"])
        print(
            f"Th={row['threshold']:.3f}: "
            f"TestPSNR {dtest:+.3f} dB" if dtest is not None else f"Th={row['threshold']:.3f}: TestPSNR -",
            end=""
        )
        if dg is not None and dp is not None:
            print(f", G {dg:+d} ({dp:+.2f}%)")
        else:
            print()

    out_json = Path("outputs/SH003_0_200_novelview_8to2_replay30_prune_threshold_summary.json")
    out_csv = Path("outputs/SH003_0_200_novelview_8to2_replay30_prune_threshold_summary.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
