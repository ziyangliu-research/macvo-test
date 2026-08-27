#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUNS = [
    ("baseline_no_replay", Path("outputs/SH003_0_200_novelview_8to2_baseline_no_replay")),
    ("replay20", Path("outputs/SH003_0_200_novelview_8to2_replay20")),
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
    for name, work_dir in RUNS:
        summary_path = work_dir / "execution_benchmark_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing {summary_path}")
        summary = load_json(summary_path)
        backend = summary["backend"]
        metrics = backend.get("final_metrics", {})
        train = metrics.get("train_inserted", {})
        test = metrics.get("test_all", {})
        train_psnr = train.get("psnr")
        test_psnr = test.get("psnr")
        pose = summary.get("pose_metrics", {})
        ate = nested(pose, "se3", "translation_error_m", "rmse")
        row = {
            "name": name,
            "train_views": train.get("num_views"),
            "test_views": test.get("num_views"),
            "train_psnr": train_psnr,
            "train_ssim": train.get("ssim"),
            "test_psnr": test_psnr,
            "test_ssim": test.get("ssim"),
            "generalization_gap_db": (
                None
                if train_psnr is None or test_psnr is None
                else float(train_psnr) - float(test_psnr)
            ),
            "final_gaussians": backend.get("final_num_gaussians"),
            "ate_se3_rmse_m": ate,
            "quality_wall_sec": summary.get("streaming_wall_time_sec"),
            "quality_protocol_fps": summary.get("streaming_fps_excluding_initialization"),
            "num_train_packets": backend.get("num_train_packets"),
            "num_test_cameras": backend.get("num_test_cameras"),
        }
        rows.append(row)

    print("\n=== SH003 0-199 strict 8:2 held-out novel-view baseline ===\n")
    header = (
        f"{'System':<20} {'Train':>5} {'Test':>5} "
        f"{'Train PSNR':>10} {'Test PSNR':>10} {'Gap':>8} "
        f"{'Train SSIM':>10} {'Test SSIM':>10} {'G(k)':>10} {'ATE(m)':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        gk = None if row["final_gaussians"] is None else float(row["final_gaussians"]) / 1000.0
        print(
            f"{row['name']:<20} "
            f"{str(row['train_views']):>5} {str(row['test_views']):>5} "
            f"{fnum(row['train_psnr']):>10} {fnum(row['test_psnr']):>10} "
            f"{fnum(row['generalization_gap_db']):>8} "
            f"{fnum(row['train_ssim'],4):>10} {fnum(row['test_ssim'],4):>10} "
            f"{fnum(gk,1):>10} {fnum(row['ate_se3_rmse_m'],4):>9}"
        )

    if len(rows) == 2:
        base, replay = rows
        print("\n=== Replay20 delta vs baseline ===")
        for key, label in [
            ("train_psnr", "Train PSNR"),
            ("test_psnr", "Test PSNR"),
            ("generalization_gap_db", "Generalization gap"),
            ("train_ssim", "Train SSIM"),
            ("test_ssim", "Test SSIM"),
        ]:
            a, b = base[key], replay[key]
            if a is not None and b is not None:
                print(f"{label:<20}: {float(b)-float(a):+.4f}")
        if base["final_gaussians"] and replay["final_gaussians"] is not None:
            delta = int(replay["final_gaussians"]) - int(base["final_gaussians"])
            ratio = float(replay["final_gaussians"]) / float(base["final_gaussians"])
            print(f"{'Final Gaussians':<20}: {delta:+d} ({(ratio-1.0)*100:+.2f}%)")

    out = Path("outputs/SH003_0_200_novelview_8to2_baseline_replay_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
