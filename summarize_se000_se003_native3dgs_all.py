#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/se000_se003_native3dgs_refine100k")
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEQS = ("SE000", "SE001", "SE002", "SE003")
CHECKPOINTS = tuple(range(10000, 100001, 10000))


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
    if "launch timed out and was terminated" in text or "cudaErrorLaunchTimeout" in text:
        out["status"] = "CUDA_TIMEOUT"
    elif "Traceback (most recent call last):" in text:
        out["status"] = "FAILED_OR_RUNNING"
    else:
        out["status"] = "RUNNING_OR_OK"
    return out


rows: list[dict[str, Any]] = []

for seq in SEQS:
    work = ROOT / seq / f"quality_seed{SEED}"
    log = work / "run.log"
    progress = online_progress_from_log(log)

    _, metrics = first_json_glob(work, "*/posthoc_global_refinement_native3dgs_metrics.json")
    summary = load_json(work / "execution_benchmark_summary.json")

    ate = None
    if isinstance(summary, dict):
        try:
            ate = float(summary["pose_metrics"]["se3"]["translation_error_m"]["rmse"])
        except Exception:
            pass

    if isinstance(metrics, dict):
        online = metrics.get("online") or {}
        g_online = metrics.get("gaussians_online")
        rows.append(
            {
                "sequence": seq,
                "stage": "Online",
                "train": fmt_metric(online.get("train")),
                "test": fmt_metric(online.get("test")),
                "ate": ate,
                "gaussians": g_online,
                "status": "DONE",
            }
        )
        cps = metrics.get("checkpoints") or {}
        for cp in CHECKPOINTS:
            q = cps.get(str(cp)) if isinstance(cps, dict) else None
            rows.append(
                {
                    "sequence": seq,
                    "stage": f"{cp//1000}k",
                    "train": fmt_metric(q.get("train") if isinstance(q, dict) else None),
                    "test": fmt_metric(q.get("test") if isinstance(q, dict) else None),
                    "ate": ate,
                    "gaussians": q.get("num_gaussians") if isinstance(q, dict) else None,
                    "status": "DONE" if isinstance(q, dict) else "MISSING",
                }
            )
    else:
        prog = "MISSING"
        if progress.get("packet_done") is not None:
            prog = f"{progress['packet_done']}/{progress['packet_total']} packets"
        rows.append(
            {
                "sequence": seq,
                "stage": "Online",
                "train": "MISSING",
                "test": "MISSING",
                "ate": ate,
                "gaussians": progress.get("gaussians"),
                "status": prog,
            }
        )
        for cp in CHECKPOINTS:
            rows.append(
                {
                    "sequence": seq,
                    "stage": f"{cp//1000}k",
                    "train": "MISSING",
                    "test": "MISSING",
                    "ate": ate,
                    "gaussians": None,
                    "status": "MISSING",
                }
            )

print("\n=== SE000-SE003 | Online + Native-3DGS Global Refinement ===")
print(f"{'Seq':6s} {'Stage':8s} {'Train P/S/L':24s} {'Test P/S/L':24s} {'ATE(m)':>9s} {'G(k)':>10s} {'Status':>16s}")
print("-" * 110)
for r in rows:
    ate_s = "MISSING" if r["ate"] is None else f"{r['ate']:.4f}"
    g_s = "MISSING" if r["gaussians"] is None else f"{int(r['gaussians'])/1000:.1f}"
    print(
        f"{r['sequence']:6s} {r['stage']:8s} {r['train']:24s} {r['test']:24s} "
        f"{ate_s:>9s} {g_s:>10s} {str(r['status']):>16s}"
    )

print("\n=== Current online progress ===")
for seq in SEQS:
    work = ROOT / seq / f"quality_seed{SEED}"
    p = online_progress_from_log(work / "run.log")
    if p.get("last_backend_line"):
        print(f"{seq}: {p['last_backend_line']}")
    else:
        print(f"{seq}: MISSING")

ROOT.mkdir(parents=True, exist_ok=True)
out_csv = ROOT / f"summary_seed{SEED}_partial.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sequence", "stage", "train_psnr_ssim_lpips", "test_psnr_ssim_lpips", "se3_ate_rmse_m", "gaussians", "status"])
    for r in rows:
        w.writerow([r["sequence"], r["stage"], r["train"], r["test"], r["ate"], r["gaussians"], r["status"]])

print(f"\nCSV: {out_csv}")
