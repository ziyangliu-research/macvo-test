#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

OUTPUT_ROOT = Path("outputs")

THRESHOLDS = [
    (0.005, "0005"),
    (0.010, "001"),
    (0.020, "002"),
    (0.030, "003"),
    (0.050, "005"),
    (0.100, "010"),
]
WINDOWS = [5, 10, 20]
REPLAY_RATIOS = [0, 5, 10, 20, 30]
REPLAY_BOUNDARY = [0, 5, 10, 20, 30, 40, 50]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def gk(value: Any) -> str:
    if value is None:
        return "-"
    return f"{int(value) / 1000.0:.1f}"


def summary_row(work_dir: Path, **metadata: Any) -> dict[str, Any]:
    path = work_dir / "execution_benchmark_summary.json"
    if not path.is_file():
        return {
            **metadata,
            "status": "missing",
            "work_dir": str(work_dir),
        }

    s = load_json(path)
    backend = s.get("backend") or {}
    fm = backend.get("final_metrics") or {}
    train = fm.get("train_inserted") or {}
    test = fm.get("test_all") or {}

    return {
        **metadata,
        "status": "complete",
        "train_views": train.get("num_views"),
        "test_views": test.get("num_views"),
        "train_psnr": train.get("psnr"),
        "train_ssim": train.get("ssim"),
        "test_psnr": test.get("psnr"),
        "test_ssim": test.get("ssim"),
        "final_gaussians": backend.get("final_num_gaussians"),
        "wall_sec": s.get("streaming_wall_time_sec"),
        "work_dir": str(work_dir),
    }


def rho0_dir(window: int, tag: str = "010") -> Path:
    return OUTPUT_ROOT / f"SH003_0_200_ablation_rho0_W{window}_th{tag}"


def replay_dir(window: int, replay_percent: int) -> Path:
    if replay_percent == 0:
        return rho0_dir(window)
    if window == 10 and replay_percent == 30:
        # This cell predates the W x rho naming convention and is reused.
        return OUTPUT_ROOT / "SH003_0_200_novelview_8to2_replay30_th010"
    return OUTPUT_ROOT / f"SH003_0_200_novelview_8to2_W{window}_R{replay_percent}_th010"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def print_missing(row: dict[str, Any], prefix: str) -> bool:
    if row.get("status") == "complete":
        return False
    print(f"{prefix}  MISSING  {row.get('work_dir')}")
    return True


def table_threshold() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for th, tag in THRESHOLDS:
        rows.append(
            summary_row(
                rho0_dir(10, tag),
                threshold=th,
                window=10,
                replay_ratio=0.0,
                history_steps=0,
                budget=100,
                maintenance=50,
            )
        )

    baseline_g = next(
        (
            int(r["final_gaussians"])
            for r in rows
            if r.get("status") == "complete"
            and abs(float(r["threshold"]) - 0.005) < 1e-12
            and r.get("final_gaussians") is not None
        ),
        None,
    )
    for row in rows:
        g = row.get("final_gaussians")
        row["gaussian_reduction_vs_0005_pct"] = (
            None
            if baseline_g is None or g is None or baseline_g <= 0
            else 100.0 * (1.0 - int(g) / baseline_g)
        )

    print("\n=== Table A. Opacity pruning threshold (NO replay) ===")
    print("SH003 [0,200): 160 mapping / 40 held-out | W=10, rho=0, B=100, M=50\n")
    header = (
        f"{'Th':>7} {'TrainP':>8} {'TrainS':>8} {'TestP':>8} {'TestS':>8} "
        f"{'G(k)':>9} {'G-red.':>8} {'Wall(s)':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if print_missing(row, f"{row['threshold']:>7.3f}"):
            continue
        red = row.get("gaussian_reduction_vs_0005_pct")
        red_text = "-" if red is None else f"{float(red):.1f}%"
        print(
            f"{row['threshold']:>7.3f} "
            f"{fnum(row.get('train_psnr')):>8} {fnum(row.get('train_ssim'),4):>8} "
            f"{fnum(row.get('test_psnr')):>8} {fnum(row.get('test_ssim'),4):>8} "
            f"{gk(row.get('final_gaussians')):>9} {red_text:>8} "
            f"{fnum(row.get('wall_sec'),1):>9}"
        )
    return rows


def table_window_replay() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        for replay in REPLAY_RATIOS:
            rows.append(
                summary_row(
                    replay_dir(window, replay),
                    window=window,
                    replay_percent=replay,
                    replay_ratio=replay / 100.0,
                    history_steps=replay,  # B=100 in this table.
                    recent_steps=100 - replay,
                    threshold=0.10,
                    budget=100,
                    maintenance=50,
                )
            )

    print("\n=== Table B. Recent working set x historical replay ===")
    print("SH003 [0,200): 160 mapping / 40 held-out | Th=.10, B=100, M=50\n")
    header = (
        f"{'W':>3} {'rho':>6} {'H':>4} {'TrainP':>8} {'TrainS':>8} "
        f"{'TestP':>8} {'TestS':>8} {'G(k)':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        prefix = f"{row['window']:>3} {100*row['replay_ratio']:>5.0f}%"
        if print_missing(row, prefix):
            continue
        print(
            f"{row['window']:>3} {100*row['replay_ratio']:>5.0f}% "
            f"{row['history_steps']:>4} "
            f"{fnum(row.get('train_psnr')):>8} {fnum(row.get('train_ssim'),4):>8} "
            f"{fnum(row.get('test_psnr')):>8} {fnum(row.get('test_ssim'),4):>8} "
            f"{gk(row.get('final_gaussians')):>9}"
        )

    print("\nHeld-out Test PSNR matrix (rows=W, cols=rho):")
    print("          0%       5%      10%      20%      30%")
    for window in WINDOWS:
        vals: list[str] = []
        for replay in REPLAY_RATIOS:
            row = next(
                (r for r in rows if r["window"] == window and r["replay_percent"] == replay),
                None,
            )
            vals.append(
                "   -   "
                if row is None or row.get("test_psnr") is None
                else f"{float(row['test_psnr']):7.3f}"
            )
        print(f"W{window:<2}   " + "  ".join(vals))
    return rows


def table_replay_boundary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replay in REPLAY_BOUNDARY:
        rows.append(
            summary_row(
                replay_dir(20, replay),
                window=20,
                replay_percent=replay,
                replay_ratio=replay / 100.0,
                history_steps=replay,
                recent_steps=100 - replay,
                threshold=0.10,
                budget=100,
                maintenance=50,
            )
        )

    print("\n=== Table C. Replay-ratio boundary ===")
    print("SH003 [0,200): 160 mapping / 40 held-out | W=20, Th=.10, B=100, M=50\n")
    header = (
        f"{'rho':>6} {'H':>4} {'Recent':>6} {'TrainP':>8} {'TrainS':>8} "
        f"{'TestP':>8} {'TestS':>8} {'G(k)':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        prefix = f"{100*row['replay_ratio']:>5.0f}%"
        if print_missing(row, prefix):
            continue
        print(
            f"{100*row['replay_ratio']:>5.0f}% {row['history_steps']:>4} {row['recent_steps']:>6} "
            f"{fnum(row.get('train_psnr')):>8} {fnum(row.get('train_ssim'),4):>8} "
            f"{fnum(row.get('test_psnr')):>8} {fnum(row.get('test_ssim'),4):>8} "
            f"{gk(row.get('final_gaussians')):>9}"
        )

    complete = [r for r in rows if r.get("test_psnr") is not None]
    if complete:
        best = max(complete, key=lambda r: float(r["test_psnr"]))
        print(
            f"\nBest held-out Test PSNR: rho={100*best['replay_ratio']:.0f}% "
            f"(H={best['history_steps']}), {best['test_psnr']:.3f} dB"
        )
    return rows


def table_budget() -> list[dict[str, Any]]:
    cases = [
        (
            50,
            25,
            15,
            OUTPUT_ROOT / "SH003_0_200_novelview_8to2_W20_R30_B50_th010",
        ),
        (
            100,
            50,
            30,
            OUTPUT_ROOT / "SH003_0_200_novelview_8to2_W20_R30_th010",
        ),
        (
            200,
            100,
            60,
            OUTPUT_ROOT / "SH003_0_200_novelview_8to2_W20_R30_B200_th010",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for budget, maintenance, history_steps, work_dir in cases:
        rows.append(
            summary_row(
                work_dir,
                window=20,
                replay_ratio=0.30,
                budget=budget,
                maintenance=maintenance,
                history_steps=history_steps,
                recent_steps=budget - history_steps,
                threshold=0.10,
            )
        )

    print("\n=== Table D. Optimization budget ===")
    print("SH003 [0,200): 160 mapping / 40 held-out | W=20, rho=.30, Th=.10, M=B/2\n")
    header = (
        f"{'B':>4} {'M':>4} {'H':>4} {'TrainP':>8} {'TrainS':>8} "
        f"{'TestP':>8} {'TestS':>8} {'G(k)':>9} {'Wall(s)':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if print_missing(row, f"B={row['budget']}"):
            continue
        print(
            f"{row['budget']:>4} {row['maintenance']:>4} {row['history_steps']:>4} "
            f"{fnum(row.get('train_psnr')):>8} {fnum(row.get('train_ssim'),4):>8} "
            f"{fnum(row.get('test_psnr')):>8} {fnum(row.get('test_ssim'),4):>8} "
            f"{gk(row.get('final_gaussians')):>9} {fnum(row.get('wall_sec'),1):>9}"
        )
    return rows


def main() -> None:
    print(
        "\nSH003 ablation protocol: [0,200), strict 8:2 = "
        "160 mapping/train views + 40 held-out test views"
    )

    threshold = table_threshold()
    window_replay = table_window_replay()
    boundary = table_replay_boundary()
    budget = table_budget()

    out = OUTPUT_ROOT / "SH003_ablation_paper_tables.json"
    out.write_text(
        json.dumps(
            {
                "protocol": {
                    "sequence": "SH003",
                    "start_index": 0,
                    "end_index_exclusive": 200,
                    "num_frames": 200,
                    "mapping_train_views": 160,
                    "heldout_test_views": 40,
                    "split": "strict 8:2; local indices 4,9,14,...,199 are held out from ReSplat/map supervision",
                },
                "table_A_no_replay_threshold": threshold,
                "table_B_window_x_replay": window_replay,
                "table_C_replay_boundary": boundary,
                "table_D_budget": budget,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    common = [
        "status",
        "train_views",
        "test_views",
        "train_psnr",
        "train_ssim",
        "test_psnr",
        "test_ssim",
        "final_gaussians",
        "wall_sec",
        "work_dir",
    ]
    write_csv(
        OUTPUT_ROOT / "SH003_ablation_table_A_no_replay_threshold.csv",
        threshold,
        [
            "threshold",
            "window",
            "replay_ratio",
            "history_steps",
            "budget",
            "maintenance",
            "gaussian_reduction_vs_0005_pct",
            *common,
        ],
    )
    write_csv(
        OUTPUT_ROOT / "SH003_ablation_table_B_window_x_replay.csv",
        window_replay,
        [
            "window",
            "replay_percent",
            "replay_ratio",
            "history_steps",
            "recent_steps",
            "threshold",
            "budget",
            "maintenance",
            *common,
        ],
    )
    write_csv(
        OUTPUT_ROOT / "SH003_ablation_table_C_replay_boundary.csv",
        boundary,
        [
            "window",
            "replay_percent",
            "replay_ratio",
            "history_steps",
            "recent_steps",
            "threshold",
            "budget",
            "maintenance",
            *common,
        ],
    )
    write_csv(
        OUTPUT_ROOT / "SH003_ablation_table_D_budget.csv",
        budget,
        [
            "window",
            "replay_ratio",
            "budget",
            "maintenance",
            "history_steps",
            "recent_steps",
            "threshold",
            *common,
        ],
    )

    print(f"\nSaved combined JSON: {out}")
    print("Saved paper-table CSVs:")
    print("  outputs/SH003_ablation_table_A_no_replay_threshold.csv")
    print("  outputs/SH003_ablation_table_B_window_x_replay.csv")
    print("  outputs/SH003_ablation_table_C_replay_boundary.csv")
    print("  outputs/SH003_ablation_table_D_budget.csv")


if __name__ == "__main__":
    main()
