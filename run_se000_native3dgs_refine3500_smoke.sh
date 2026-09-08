#!/usr/bin/env bash
set -euo pipefail

cd /home/shiyo/Desktop/MAC-VO

GPU="${GPU:-0}"
SEED="${SEED:-0}"
OUT="${OUT:-outputs/SE000_first50_native3dgs_refine3500_smoke}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SE000.yaml"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SE000.txt"
RUNNER="run_pipeline_execution_benchmark_repro_posthoc_global_refine_native3dgs_lpips.py"
NAME="incremental_SE000_first50_native3dgs_refine3500"

rm -rf "$OUT"
mkdir -p "$OUT"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONHASHSEED="$SEED" \
PIPELINE_BENCHMARK_SEED="$SEED" \
PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
PIPELINE_GLOBAL_REFINE_TOTAL_ITERS=3500 \
PIPELINE_GLOBAL_REFINE_ITER_CHECKPOINTS=3500 \
PIPELINE_GLOBAL_REFINE_SEED="$SEED" \
CUDA_VISIBLE_DEVICES="$GPU" \
TORCH_COMPILE_DISABLE=1 \
python "$RUNNER" \
  --mode serial \
  --with_pose_metrics \
  --config "$CONFIG" \
  --set paths.data_config="$DATA_CONFIG" \
  --set evaluation.gt_pose_file="$GT" \
  --set pose_frontend.source=macvo \
  --set sequence.start_index=0 \
  --set sequence.end_index=50 \
  --set 'resplat_frontend.overrides=["dataset.image_shape=[240,320]"]' \
  --set split.split_every=5 \
  --set split.split_offset=4 \
  --set split.split_index_mode=local_index \
  --set backend.local_map_size=20 \
  --set backend.iterations_per_packet=100 \
  --set backend.reset_new_packet_opacity=true \
  --set backend.new_packet_reset_max_opacity=0.01 \
  --set backend.maintenance_mode=standard \
  --set backend.maintenance_after_local_iteration=50 \
  --set backend.maintenance_min_opacity=0.10 \
  --set backend.optimization.iterations=4000 \
  --set backend.eval_before_optimization=false \
  --set backend.eval_every_train_packets=1000000 \
  --set backend.evaluation_enabled=true \
  --set backend.eval_max_views=0 \
  --set backend.save_every_train_packets=0 \
  --set backend.save_final_ply=false \
  --set backend.wandb_mode=disabled \
  --set backend.write_runtime_artifacts=true \
  --set paths.work_dir="$OUT" \
  --set backend.output_name="$NAME" \
  2>&1 | tee "$OUT/run.log"

EVENTS="$OUT/$NAME/posthoc_global_refinement_native3dgs_events.json"
METRICS="$OUT/$NAME/posthoc_global_refinement_native3dgs_metrics.json"

python - "$EVENTS" "$METRICS" <<'PY'
import json, sys
from pathlib import Path

events_path, metrics_path = map(Path, sys.argv[1:])
if not events_path.is_file():
    raise SystemExit(f"[FAIL] missing events: {events_path}")
if not metrics_path.is_file():
    raise SystemExit(f"[FAIL] missing metrics: {metrics_path}")

events = json.loads(events_path.read_text())
metrics = json.loads(metrics_path.read_text())

initial = [e for e in events if e.get("type") == "opacity_reset_initial"]
dp = [e for e in events if e.get("type") == "densify_and_prune"]
resets = [e for e in events if e.get("type") == "opacity_reset"]

expected_dp = list(range(600, 3501, 100))
actual_dp = [int(e["refinement_iteration"]) for e in dp]
actual_resets = [int(e["refinement_iteration"]) for e in resets]

assert len(initial) == 1 and int(initial[0]["refinement_iteration"]) == 0, initial
assert actual_dp == expected_dp, (actual_dp[:5], actual_dp[-5:], len(actual_dp))
assert actual_resets == [3000], actual_resets
by_iter = {int(e["refinement_iteration"]): e for e in dp}
assert by_iter[3000]["max_screen_size"] is None, by_iter[3000]
assert float(by_iter[3100]["max_screen_size"]) == 20.0, by_iter[3100]
assert int(metrics["total_refinement_iterations"]) == 3500
assert "3500" in metrics["checkpoints"]

print("\n=== native 3DGS smoke verification ===")
print("initial opacity reset : PASS @ iter 0")
print(f"densify/prune schedule: PASS ({len(dp)} events, 600..3500 every 100)")
print("native opacity reset  : PASS @ iter 3000")
print("large-size pruning    : PASS (None @3000, max_screen=20 @3100+)")
print(f"G online -> final     : {metrics['gaussians_online']:,} -> {metrics['gaussians_final']:,}")
q = metrics["checkpoints"]["3500"]
print(f"3500 Train P/S/L      : {q['train']['psnr']:.3f}/{q['train']['ssim']:.4f}/{q['train']['lpips']:.4f}")
print(f"3500 Test  P/S/L      : {q['test']['psnr']:.3f}/{q['test']['ssim']:.4f}/{q['test']['lpips']:.4f}")
print("SMOKE TEST: PASS")
PY
