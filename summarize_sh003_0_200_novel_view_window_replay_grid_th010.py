#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

CASES = [
    (10, 10, Path("outputs/SH003_0_200_novelview_8to2_W10_R10_th010"), "incremental_SH003_0_200_novelview_8to2_W10_R10_th010"),
    (10, 20, Path("outputs/SH003_0_200_novelview_8to2_W10_R20_th010"), "incremental_SH003_0_200_novelview_8to2_W10_R20_th010"),
    # Reuse the already-completed replay30 / threshold=.10 cell.
    (10, 30, Path("outputs/SH003_0_200_novelview_8to2_replay30_th010"), "incremental_SH003_0_200_novelview_8to2_replay30_th010"),
    (20, 10, Path("outputs/SH003_0_200_novelview_8to2_W20_R10_th010"), "incremental_SH003_0_200_novelview_8to2_W20_R10_th010"),
    (20, 20, Path("outputs/SH003_0_200_novelview_8to2_W20_R20_th010"), "incremental_SH003_0_200_novelview_8to2_W20_R20_th010"),
    (20, 30, Path("outputs/SH003_0_200_novelview_8to2_W20_R30_th010"), "incremental_SH003_0_200_novelview_8to2_W20_R30_th010"),
]

EMPTY_RE = re.compile(r"\[empty-map guard\].*?frame=(?P<frame>\d+)")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(value: Any, *keys: str) -> Any:
    cur = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def empty_frames(path: Path) -> list[int]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [int(m.group("frame")) for m in EMPTY_RE.finditer(text)]


def fnum(v: Any, digits: int = 3) -> str:
    return "-" if v is None else f"{float(v):.{digits}f}"


def main() -> None:
    rows: list[dict[str, Any]] = []
    for window, replay, work_dir, output_name in CASES:
        p = work_dir / "execution_benchmark_summary.json"
        if not p.is_file():
            rows.append({"window": window, "replay": replay, "status": "missing", "work_dir": str(work_dir)})
            continue
        s = load_json(p)
        backend = s.get("backend") or {}
        metrics = backend.get("final_metrics") or {}
        train = metrics.get("train_inserted") or {}
        test = metrics.get("test_all") or {}
        active = metrics.get("active_local_map") or {}
        train_p = train.get("psnr")
        test_p = test.get("psnr")
        log_path = work_dir / "run.log"
        timing_path = work_dir / output_name / "timing_log.json"
        timing = load_json(timing_path) if timing_path.is_file() else []
        backend_times = [float(x["backend_total_sec"]) for x in timing if x.get("backend_total_sec") is not None]
        row = {
            "window": window,
            "replay": replay,
            "replay_ratio": replay / 100.0,
            "status": "complete",
            "train_psnr": train_p,
            "test_psnr": test_p,
            "gap_db": None if train_p is None or test_p is None else float(train_p) - float(test_p),
            "active_psnr": active.get("psnr"),
            "train_ssim": train.get("ssim"),
            "test_ssim": test.get("ssim"),
            "final_gaussians": backend.get("final_num_gaussians"),
            "wall_sec": s.get("streaming_wall_time_sec"),
            "backend_mean_sec": sum(backend_times) / len(backend_times) if backend_times else None,
            "empty_frames": empty_frames(log_path),
            "ate_se3_rmse_m": nested(s.get("pose_metrics") or {}, "se3", "translation_error_m", "rmse"),
            "work_dir": str(work_dir),
        }
        rows.append(row)

    print("\n=== SH003 0-199 strict 8:2 | Th=.10 | W x historical replay budget ===\n")
    header = (
        f"{'W':>3} {'R':>3} {'TrainP':>8} {'TestP':>8} {'Gap':>7} {'Active':>8} "
        f"{'TestSSIM':>9} {'G(k)':>9} {'Wall(s)':>9} {'Backend':>8} {'Empty':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["status"] != "complete":
            print(f"{r['window']:>3} {r['replay']:>3} {'missing':>8}")
            continue
        gk = None if r.get("final_gaussians") is None else int(r["final_gaussians"]) / 1000.0
        print(
            f"{r['window']:>3} {r['replay']:>3} "
            f"{fnum(r.get('train_psnr')):>8} {fnum(r.get('test_psnr')):>8} "
            f"{fnum(r.get('gap_db')):>7} {fnum(r.get('active_psnr')):>8} "
            f"{fnum(r.get('test_ssim'),4):>9} {fnum(gk,1):>9} "
            f"{fnum(r.get('wall_sec'),1):>9} {fnum(r.get('backend_mean_sec'),3):>8} "
            f"{len(r.get('empty_frames', [])):>5}"
        )

    complete = [r for r in rows if r.get("status") == "complete" and r.get("test_psnr") is not None]
    if complete:
        best = max(complete, key=lambda r: float(r["test_psnr"]))
        print(
            f"\nBest held-out Test PSNR: W={best['window']}, R={best['replay']}, "
            f"PSNR={best['test_psnr']:.3f} dB"
        )

    print("\nTest PSNR matrix (rows=W, cols=R):")
    print("          R10      R20      R30")
    for w in (10, 20):
        vals = []
        for rr in (10, 20, 30):
            match = next((x for x in rows if x["window"] == w and x["replay"] == rr), None)
            vals.append("   -   " if not match or match.get("test_psnr") is None else f"{float(match['test_psnr']):7.3f}")
        print(f"W{w:<2}   " + "  ".join(vals))

    print("\nActive PSNR matrix (rows=W, cols=R):")
    print("          R10      R20      R30")
    for w in (10, 20):
        vals = []
        for rr in (10, 20, 30):
            match = next((x for x in rows if x["window"] == w and x["replay"] == rr), None)
            vals.append("   -   " if not match or match.get("active_psnr") is None else f"{float(match['active_psnr']):7.3f}")
        print(f"W{w:<2}   " + "  ".join(vals))

    out_json = Path("outputs/SH003_0_200_novelview_8to2_WxR_th010_summary.json")
    out_csv = Path("outputs/SH003_0_200_novelview_8to2_WxR_th010_summary.csv")
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = [
        "window", "replay", "replay_ratio", "status", "train_psnr", "test_psnr",
        "gap_db", "active_psnr", "train_ssim", "test_ssim", "final_gaussians",
        "wall_sec", "backend_mean_sec", "empty_frames", "ate_se3_rmse_m", "work_dir",
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
