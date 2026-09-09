#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/se000_se003_native3dgs_refine100k")
SEQ = sys.argv[2] if len(sys.argv) > 2 else "SE000"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

work = ROOT / SEQ / f"quality_seed{SEED}"
metrics_files = list(work.glob("*/posthoc_global_refinement_native3dgs_metrics.json"))
if not metrics_files:
    raise SystemExit(f"missing native3dgs metrics under {work}")
metrics_path = metrics_files[0]
d = json.loads(metrics_path.read_text())


def fmt(m):
    if not isinstance(m, dict) or not m:
        return "MISSING"
    return f"{m.get('psnr', float('nan')):.3f}/{m.get('ssim', float('nan')):.4f}/{m.get('lpips', float('nan')):.4f}"

completed = d.get("completed_refinement_iterations", d.get("total_refinement_iterations"))
print(f"=== {SEQ} native3dgs refinement partial summary ===")
print(f"metrics file : {metrics_path}")
print(f"completed    : {completed}")
print(f"G online     : {d.get('gaussians_online', 'MISSING')}")
print(f"G current    : {d.get('gaussians_current', d.get('gaussians_final', 'MISSING'))}")

online = d.get("online") or {}
print(f"Online Train : {fmt(online.get('train'))}")
print(f"Online Test  : {fmt(online.get('test'))}")
print("\nIter      Train P/S/L              Test P/S/L               G")
print("---------------------------------------------------------------------")
checkpoints = d.get("checkpoints") or {}
for key in sorted(checkpoints, key=lambda x: int(x)):
    q = checkpoints[key]
    print(f"{int(key):6d}    {fmt(q.get('train')):24s} {fmt(q.get('test')):24s} {q.get('num_gaussians','MISSING')}")

event_files = list(work.glob("*/posthoc_global_refinement_native3dgs_events.json"))
if event_files:
    events = json.loads(event_files[0].read_text())
    dp = [e for e in events if e.get("type") == "densify_and_prune"]
    resets = [e for e in events if e.get("type") in {"opacity_reset_initial", "opacity_reset"}]
    print("\nMaintenance:")
    print("  resets       :", [e.get("refinement_iteration") for e in resets])
    if dp:
        print(f"  densify/prune: {len(dp)} events; last iter={dp[-1].get('refinement_iteration')} G={dp[-1].get('count_before')}->{dp[-1].get('count_after')} max_screen={dp[-1].get('max_screen_size')}")

summary = work / "execution_benchmark_summary.json"
print(f"\nexecution summary exists: {summary.is_file()}")
log = work / "run.log"
if log.is_file():
    text = log.read_text(errors="replace")
    if "launch timed out and was terminated" in text or "cudaErrorLaunchTimeout" in text:
        print("run status: CUDA LAUNCH TIMEOUT detected")
