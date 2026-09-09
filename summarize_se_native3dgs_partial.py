#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/se000_se003_native3dgs_refine100k")
SEQ = sys.argv[2] if len(sys.argv) > 2 else "SE000"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

work = ROOT / SEQ / f"quality_seed{SEED}"


def fmt(m):
    if not isinstance(m, dict) or not m:
        return "MISSING"
    return f"{m.get('psnr', float('nan')):.3f}/{m.get('ssim', float('nan')):.4f}/{m.get('lpips', float('nan')):.4f}"


def summarize_online_partial() -> bool:
    timing_files = sorted(work.glob("*/timing_log.json"))
    if not timing_files:
        return False
    timing_path = timing_files[0]
    rows = json.loads(timing_path.read_text())
    if not isinstance(rows, list) or not rows:
        return False

    last = rows[-1]
    durations = [float(r.get("backend_total_sec", 0.0)) for r in rows if float(r.get("backend_total_sec", 0.0)) > 0]
    last20 = durations[-20:]

    print(f"=== {SEQ} ONLINE partial summary ===")
    print(f"timing file        : {timing_path}")
    print(f"completed packets  : {len(rows)}")
    print(f"last frame         : {last.get('frame_index', 'MISSING')}")
    print(f"last train packet  : {last.get('train_packet_count', 'MISSING')}")
    if isinstance(last.get('train_packet_count'), int):
        print(f"global iteration   : {int(last['train_packet_count']) * 100}")
    print(f"Gaussians          : {last.get('num_gaussians', 'MISSING')}")
    print(f"last backend sec   : {float(last.get('backend_total_sec', float('nan'))):.3f}")
    if durations:
        print(f"backend sec median : {statistics.median(durations):.3f}")
        print(f"backend sec max    : {max(durations):.3f}")
    if last20:
        print(f"last20 mean sec    : {statistics.mean(last20):.3f}")
        print(f"last20 max sec     : {max(last20):.3f}")

    for key in (
        "gpu_memory_allocated_gb",
        "gpu_memory_reserved_gb",
        "gpu_peak_memory_allocated_gb",
        "gpu_free_memory_gb",
    ):
        if key in last:
            print(f"{key:20s}: {float(last[key]):.2f}")

    maintenance_files = sorted(work.glob("*/maintenance_log.json"))
    if maintenance_files:
        events = json.loads(maintenance_files[0].read_text())
        if isinstance(events, list) and events:
            e = events[-1]
            print("\nlast online maintenance:")
            print(f"  frame/global iter : {e.get('frame_index')} / {e.get('global_iteration')}")
            print(f"  G                 : {e.get('count_before')} -> {e.get('count_after')}")
            print(f"  min opacity       : {e.get('min_opacity')}")
            print(f"  max screen        : {e.get('max_screen_size')}")

    log = work / "run.log"
    if log.is_file():
        text = log.read_text(errors="replace")
        if "launch timed out and was terminated" in text or "cudaErrorLaunchTimeout" in text:
            print("\nrun status: CUDA LAUNCH TIMEOUT detected during ONLINE stage")
    return True


metrics_files = sorted(work.glob("*/posthoc_global_refinement_native3dgs_metrics.json"))
if not metrics_files:
    if summarize_online_partial():
        raise SystemExit(0)
    raise SystemExit(f"no online timing_log or native3dgs metrics under {work}")

metrics_path = metrics_files[0]
d = json.loads(metrics_path.read_text())
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

event_files = sorted(work.glob("*/posthoc_global_refinement_native3dgs_events.json"))
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
