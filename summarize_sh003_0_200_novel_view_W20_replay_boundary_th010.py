#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPLAYS = [5, 10, 20, 30, 40, 50]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: Any, *keys: str) -> Any:
    cur = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fnum(v: Any, digits: int = 3) -> str:
    return "-" if v is None else f"{float(v):.{digits}f}"


def main() -> None:
    rows: list[dict[str, Any]] = []

    for r in REPLAYS:
        work_dir = Path(f"outputs/SH003_0_200_novelview_8to2_W20_R{r}_th010")
        summary_path = work_dir / "execution_benchmark_summary.json"
        if not summary_path.is_file():
            rows.append({"replay": r, "status": "missing", "work_dir": str(work_dir)})
            continue

        s = load_json(summary_path)
        backend = s.get("backend") or {}
        fm = backend.get("final_metrics") or {}
        train = fm.get("train_inserted") or {}
        test = fm.get("test_all") or {}
        recent10 = fm.get("fixed_recent_10") or {}
        active = fm.get("active_local_map") or {}
        train_p = train.get("psnr")
        test_p = test.get("psnr")

        output_name = f"incremental_SH003_0_200_novelview_8to2_W20_R{r}_th010"
        timing_path = work_dir / output_name / "timing_log.json"
        timing = load_json(timing_path) if timing_path.is_file() else []
        backend_times = [
            float(x["backend_total_sec"])
            for x in timing
            if x.get("backend_total_sec") is not None
        ]

        rows.append(
            {
                "replay": r,
                "replay_ratio": r / 100.0,
                "status": "complete",
                "train_psnr": train_p,
                "test_psnr": test_p,
                "gap_db": (
                    None
                    if train_p is None or test_p is None
                    else float(train_p) - float(test_p)
                ),
                "test_ssim": test.get("ssim"),
                "fixed_recent10_psnr": recent10.get("psnr"),
                "active_psnr": active.get("psnr"),
                "final_gaussians": backend.get("final_num_gaussians"),
                "wall_sec": s.get("streaming_wall_time_sec"),
                "backend_mean_sec": (
                    sum(backend_times) / len(backend_times) if backend_times else None
                ),
                "ate_se3_rmse_m": nested(
                    s.get("pose_metrics") or {},
                    "se3",
                    "translation_error_m",
                    "rmse",
                ),
                "work_dir": str(work_dir),
            }
        )

    print("\n=== SH003 0-199 strict 8:2 | W=20 | Th=.10 | replay-boundary sweep ===\n")
    header = (
        f"{'R':>3} {'TrainP':>8} {'TestP':>8} {'Gap':>7} {'TestSSIM':>9} "
        f"{'Recent10':>9} {'G(k)':>9} {'Wall(s)':>9} {'Backend':>8}"
    )
    print(header)
    print("-" * len(header))

    complete: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] != "complete":
            print(f"{row['replay']:>3} {'missing':>8}")
            continue
        complete.append(row)
        gk = (
            None
            if row.get("final_gaussians") is None
            else int(row["final_gaussians"]) / 1000.0
        )
        print(
            f"{row['replay']:>3} "
            f"{fnum(row.get('train_psnr')):>8} "
            f"{fnum(row.get('test_psnr')):>8} "
            f"{fnum(row.get('gap_db')):>7} "
            f"{fnum(row.get('test_ssim'),4):>9} "
            f"{fnum(row.get('fixed_recent10_psnr')):>9} "
            f"{fnum(gk,1):>9} "
            f"{fnum(row.get('wall_sec'),1):>9} "
            f"{fnum(row.get('backend_mean_sec'),3):>8}"
        )

    if complete:
        best = max(complete, key=lambda x: float(x.get("test_psnr", float("-inf"))))
        print(
            f"\nBest held-out Test PSNR: R={best['replay']}%, "
            f"PSNR={best['test_psnr']:.3f} dB"
        )

        ordered = sorted(complete, key=lambda x: int(x["replay"]))
        print("\nMarginal Test-PSNR gain:")
        for a, b in zip(ordered, ordered[1:]):
            if a.get("test_psnr") is None or b.get("test_psnr") is None:
                continue
            delta = float(b["test_psnr"]) - float(a["test_psnr"])
            print(f"R{a['replay']:>2} -> R{b['replay']:>2}: {delta:+.3f} dB")

        r30 = next((x for x in complete if x["replay"] == 30), None)
        if r30 is not None:
            print("\nDelta vs R=30:")
            for row in complete:
                if row["replay"] <= 30 or row.get("test_psnr") is None:
                    continue
                dtest = float(row["test_psnr"]) - float(r30["test_psnr"])
                dtrain = float(row["train_psnr"]) - float(r30["train_psnr"])
                print(
                    f"R={row['replay']}: Test {dtest:+.3f} dB, "
                    f"Train {dtrain:+.3f} dB"
                )

    out_json = Path("outputs/SH003_0_200_novelview_8to2_W20_replay_boundary_th010_summary.json")
    out_csv = Path("outputs/SH003_0_200_novelview_8to2_W20_replay_boundary_th010_summary.csv")
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fields = [
        "replay",
        "replay_ratio",
        "status",
        "train_psnr",
        "test_psnr",
        "gap_db",
        "test_ssim",
        "fixed_recent10_psnr",
        "active_psnr",
        "final_gaussians",
        "wall_sec",
        "backend_mean_sec",
        "ate_se3_rmse_m",
        "work_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
