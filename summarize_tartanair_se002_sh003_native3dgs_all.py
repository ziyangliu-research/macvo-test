#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/tartanair_se002_sh003_native3dgs_refine100k")
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEQS = ("SE002", "SE003", "SH000", "SH001", "SH002", "SH003")
CHECKPOINTS = (30000, 50000, 100000)


def load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt_metric(m: Any) -> str:
    if not isinstance(m, dict) or int(m.get("num_views", 0)) <= 0:
        return "MISSING"
    try:
        return f"{float(m['psnr']):.3f}/{float(m['ssim']):.4f}/{float(m['lpips']):.4f}"
    except Exception:
        return "MISSING"


def first_json_glob(work: Path, pattern: str) -> tuple[Path | None, Any | None]:
    files = sorted(work.glob(pattern))
    if not files:
        return None, None
    return files[0], load_json(files[0])


def extract_ate(summary: Any, work: Path) -> float | None:
    sources = []
    if isinstance(summary, dict):
        sources.append(summary.get("pose_metrics"))
    sources.append(load_json(work / "pose_metrics.json"))
    for pm in sources:
        if not isinstance(pm, dict):
            continue
        try:
            return float(pm["se3"]["translation_error_m"]["rmse"])
        except Exception:
            continue
    return None


def recover_online_timing(work: Path) -> tuple[float | None, float | None, int | None]:
    rows = load_json(work / "frame_timing_log.json")
    if not isinstance(rows, list) or not rows:
        return None, None, None
    completed = [r for r in rows if isinstance(r, dict) and "backend_end_sec" in r]
    if not completed:
        return None, None, None
    try:
        wall = max(float(r["backend_end_sec"]) for r in completed)
    except Exception:
        return None, None, None
    frames = len(rows)
    if wall <= 0 or frames <= 0:
        return None, wall, frames
    return float(frames) / wall, wall, frames


def online_progress_from_log(log: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not log.is_file():
        return out
    text = log.read_text(errors="replace")
    lines = [x for x in text.splitlines() if "[backend] packet " in x]
    if lines:
        line = lines[-1]
        out["last_backend_line"] = line
        m = re.search(
            r"packet\s+(\d+)/(\d+)\s+frame=(\d+)\s+iter=(\d+)/(\d+)\s+G=([\d,]+)",
            line,
        )
        if m:
            out.update(
                {
                    "packet_done": int(m.group(1)),
                    "packet_total": int(m.group(2)),
                    "last_frame": int(m.group(3)),
                    "online_iter": int(m.group(4)),
                    "online_iter_total": int(m.group(5)),
                    "gaussians": int(m.group(6).replace(",", "")),
                }
            )
    return out


rows_out: list[dict[str, Any]] = []

for seq in SEQS:
    work = ROOT / seq / f"quality_seed{SEED}"
    progress = online_progress_from_log(work / "run.log")
    _, metrics = first_json_glob(work, "*/posthoc_global_refinement_native3dgs_metrics.json")
    summary = load_json(work / "execution_benchmark_summary.json")
    ate = extract_ate(summary, work)
    fps, online_wall, online_frames = recover_online_timing(work)

    if isinstance(metrics, dict):
        online = metrics.get("online") or {}
        rows_out.append(
            {
                "sequence": seq,
                "stage": "Online",
                "train": fmt_metric(online.get("train")),
                "test": fmt_metric(online.get("test")),
                "ate": ate,
                "fps": fps,
                "online_wall_sec": online_wall,
                "online_num_frames": online_frames,
                "gaussians": metrics.get("gaussians_online"),
                "status": "DONE",
            }
        )
        cps = metrics.get("checkpoints") or {}
        for cp in CHECKPOINTS:
            q = cps.get(str(cp)) if isinstance(cps, dict) else None
            rows_out.append(
                {
                    "sequence": seq,
                    "stage": f"{cp//1000}k",
                    "train": fmt_metric(q.get("train") if isinstance(q, dict) else None),
                    "test": fmt_metric(q.get("test") if isinstance(q, dict) else None),
                    "ate": ate,
                    "fps": None,
                    "online_wall_sec": online_wall,
                    "online_num_frames": online_frames,
                    "gaussians": q.get("num_gaussians") if isinstance(q, dict) else None,
                    "status": "DONE" if isinstance(q, dict) else "MISSING",
                }
            )
    else:
        status = "MISSING"
        if progress.get("packet_done") is not None:
            status = f"{progress['packet_done']}/{progress['packet_total']} packets"
        rows_out.append(
            {
                "sequence": seq,
                "stage": "Online",
                "train": "MISSING",
                "test": "MISSING",
                "ate": ate,
                "fps": fps,
                "online_wall_sec": online_wall,
                "online_num_frames": online_frames,
                "gaussians": progress.get("gaussians"),
                "status": status,
            }
        )
        for cp in CHECKPOINTS:
            rows_out.append(
                {
                    "sequence": seq,
                    "stage": f"{cp//1000}k",
                    "train": "MISSING",
                    "test": "MISSING",
                    "ate": ate,
                    "fps": None,
                    "online_wall_sec": online_wall,
                    "online_num_frames": online_frames,
                    "gaussians": None,
                    "status": "MISSING",
                }
            )

print("\n=== TartanAir SE002-SH003 | Online + Native-3DGS Global Refinement ===")
print(f"{'Seq':6s} {'Stage':7s} {'Train P/S/L':24s} {'Test P/S/L':24s} {'ATE(m)':>9s} {'FPS':>8s} {'G(k)':>10s} {'Status':>18s}")
print("-" * 122)
for r in rows_out:
    ate_s = "MISSING" if r["ate"] is None else f"{r['ate']:.4f}"
    fps_s = "-" if r["stage"] != "Online" else ("MISSING" if r["fps"] is None else f"{r['fps']:.4f}")
    g_s = "MISSING" if r["gaussians"] is None else f"{int(r['gaussians'])/1000:.1f}"
    print(
        f"{r['sequence']:6s} {r['stage']:7s} {r['train']:24s} {r['test']:24s} "
        f"{ate_s:>9s} {fps_s:>8s} {g_s:>10s} {str(r['status']):>18s}"
    )

print("\n=== Online timing recovered from frame_timing_log.json ===")
for seq in SEQS:
    work = ROOT / seq / f"quality_seed{SEED}"
    fps, wall, frames = recover_online_timing(work)
    if fps is None:
        print(f"{seq}: MISSING")
    else:
        print(f"{seq}: FPS={fps:.4f} | frames={frames} | wall={wall:.3f}s")

print("\n=== Current online progress ===")
for seq in SEQS:
    p = online_progress_from_log(ROOT / seq / f"quality_seed{SEED}" / "run.log")
    print(f"{seq}: {p.get('last_backend_line', 'MISSING')}")

ROOT.mkdir(parents=True, exist_ok=True)
out_csv = ROOT / f"summary_seed{SEED}_partial.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "sequence", "stage", "train_psnr_ssim_lpips", "test_psnr_ssim_lpips",
        "se3_ate_rmse_m", "online_fps", "online_wall_sec", "online_num_frames",
        "gaussians", "status"
    ])
    for r in rows_out:
        w.writerow([
            r["sequence"], r["stage"], r["train"], r["test"], r["ate"], r["fps"],
            r["online_wall_sec"], r["online_num_frames"], r["gaussians"], r["status"]
        ])

print(f"\nCSV: {out_csv}")
