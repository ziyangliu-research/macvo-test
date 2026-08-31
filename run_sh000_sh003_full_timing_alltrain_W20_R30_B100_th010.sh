#!/usr/bin/env bash
set -u

# Formal runtime/FPS benchmark for the frozen incremental backend policy.
#
# Protocol:
#   - SH000-SH003 full sequences
#   - ALL frames are mapping/train frames (no held-out split)
#   - backend evaluation disabled
#   - no pose-metric evaluation in this timing run
#   - no intermediate/final PLY saving
#   - serial execution, same persistent MAC-VO + ReSplat + GraphDECO classes
#
# Frozen policy:
#   W = 20 recent cameras
#   rho = 0.30 historical replay ratio
#   B = 100 optimizer steps / packet
#   M = 50 maintenance iteration
#   opacity prune threshold = 0.10
#   new packet opacity cap = 0.01
#
# Reported FPS is num_frames / streaming_wall_time_sec. Initialization is
# excluded; final backend evaluation is disabled, so streaming wall time measures
# the actual full mapping pipeline rather than the strict-8:2 quality protocol.

cd /home/shiyo/Desktop/MAC-VO || exit 1

PIPELINE_CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo"
RUNNER="run_pipeline_execution_benchmark_repro_empty_safe.py"

count_frames() {
  local seq="$1"
  python - "$DATA_ROOT/$seq" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
left = root / "image_left"
if not left.is_dir():
    raise SystemExit(f"image_left directory not found: {left}")
exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
files = [p for p in left.iterdir() if p.is_file() and p.suffix.lower() in exts]
if not files:
    raise SystemExit(f"no images found in: {left}")
print(len(files))
PY
}

run_case() {
  local seq="$1"
  local data_config="Config/Sequence/TartanAirV1_Challenge_${seq}.yaml"
  local num_frames
  num_frames=$(count_frames "$seq") || return 2

  local bookkeeping_iters=$(( num_frames * 100 ))
  local name="${seq}_full_timing_alltrain_W20_R30_B100_th010"
  local work_dir="outputs/${name}"
  local output_name="incremental_${name}"
  local summary="$work_dir/execution_benchmark_summary.json"

  if [[ -f "$summary" ]]; then
    echo "[skip complete] $seq -> $summary"
    return 0
  fi

  mkdir -p "$work_dir"

  echo
  echo "================================================================"
  echo "$seq full runtime benchmark | frames=$num_frames | ALL frames map"
  echo "W=20 | rho=.30 | B=100 | M=50 | Th=.10 | evaluation OFF"
  echo "work_dir=$work_dir"
  echo "================================================================"

  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED=0 \
  PIPELINE_BENCHMARK_SEED=0 \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  CUDA_VISIBLE_DEVICES=0 \
  TORCH_COMPILE_DISABLE=1 \
  python "$RUNNER" \
    --mode serial \
    --config "$PIPELINE_CONFIG" \
    --set paths.data_config="$data_config" \
    --set pose_frontend.source=macvo \
    --set sequence.start_index=0 \
    --set sequence.end_index="$num_frames" \
    --set 'resplat_frontend.overrides=["dataset.image_shape=[240,320]"]' \
    --set split.split_every=1000000 \
    --set split.split_offset=999999 \
    --set split.split_index_mode=local_index \
    --set backend.local_map_size=20 \
    --set backend.iterations_per_packet=100 \
    --set backend.reset_new_packet_opacity=true \
    --set backend.new_packet_reset_max_opacity=0.01 \
    --set backend.maintenance_mode=standard \
    --set backend.maintenance_after_local_iteration=50 \
    --set backend.maintenance_min_opacity=0.10 \
    --set backend.optimization.iterations="$bookkeeping_iters" \
    --set backend.eval_before_optimization=false \
    --set backend.eval_every_train_packets=1000000 \
    --set backend.evaluation_enabled=false \
    --set backend.write_runtime_artifacts=true \
    --set backend.save_every_train_packets=0 \
    --set backend.save_final_ply=false \
    --set backend.wandb_mode=disabled \
    --set paths.work_dir="$work_dir" \
    --set backend.output_name="$output_name" \
    2>&1 | tee "$work_dir/run.log"

  local status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$work_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED] $seq exit=$status; continuing to next sequence" >&2
  fi
}

for seq in SH000 SH001 SH002 SH003; do
  run_case "$seq"
done

echo
echo "Runtime benchmark finished. Summarize with:"
echo "  python summarize_sh000_sh003_full_timing_alltrain_W20_R30_B100_th010.py"
