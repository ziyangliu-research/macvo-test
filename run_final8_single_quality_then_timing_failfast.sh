#!/usr/bin/env bash
set -u

# Prioritized final subset with CUDA-timeout fail-fast handling:
#   SE000-SE003 + SH000-SH003
#
# Phase 1: one seed0 quality run for missing targets SE002 SE003 SH000-SH003.
#          Existing completed seed0 outputs are skipped.
# Phase 2: one timing-only run for all eight sequences.
#          Existing completed timing outputs are skipped.
#
# If a child process prints cudaErrorLaunchTimeout / torch.AcceleratorError and
# then hangs during CUDA cleanup, kill its entire process group and continue.

cd /home/shiyo/Desktop/MAC-VO || exit 1

PIPELINE_CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo"
GT_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt"
QUALITY_RUNNER="run_pipeline_execution_benchmark_repro_posthoc_global_refine_endpoint_lpips.py"
TIMING_RUNNER="run_pipeline_execution_benchmark_repro_posthoc_global_refine_timing_only.py"
OUTPUT_ROOT="outputs/final16_10pass_full"
GPU="${GPU:-0}"

QUALITY_TARGETS=(SE002 SE003 SH000 SH001 SH002 SH003)
TIMING_TARGETS=(SE000 SE001 SE002 SE003 SH000 SH001 SH002 SH003)

mkdir -p "$OUTPUT_ROOT"

python - <<'PY'
from lpips import LPIPS
print("[preflight] LPIPS import OK")
PY
if [[ $? -ne 0 ]]; then
  echo "[fatal] lpips import failed in the current environment" >&2
  exit 2
fi

# Run one command in its own session/process group while mirroring output to a log.
# If a CUDA launch-timeout signature appears and the process does not terminate,
# terminate the whole group so the outer experiment can move on.
run_cuda_failfast() {
  local log="$1"
  shift

  : > "$log"
  setsid "$@" > >(tee "$log") 2>&1 &
  local pid=$!
  local fatal_seen=0

  while kill -0 "$pid" 2>/dev/null; do
    if tail -n 160 "$log" 2>/dev/null | grep -Eq \
      'cudaErrorLaunchTimeout|CUDA error: the launch timed out|torch\.AcceleratorError: CUDA error'; then
      fatal_seen=1
      echo "[fail-fast] detected CUDA launch timeout; terminating process group PGID=$pid" | tee -a "$log"
      kill -TERM -- "-$pid" 2>/dev/null || true

      for _ in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 1
      done

      if kill -0 "$pid" 2>/dev/null; then
        echo "[fail-fast] process group did not exit after TERM; sending KILL" | tee -a "$log"
        kill -KILL -- "-$pid" 2>/dev/null || true
      fi
      break
    fi
    sleep 3
  done

  wait "$pid" 2>/dev/null
  local status=$?
  if [[ "$fatal_seen" -eq 1 ]]; then
    return 86
  fi
  return "$status"
}

count_frames() {
  local seq="$1"
  python - "$DATA_ROOT/$seq" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]) / "image_left"
exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
if not files:
    raise SystemExit(f"no images found in {root}")
print(len(files))
PY
}

common_args() {
  local seq="$1"
  local n="$2"
  local bookkeeping_iters="$3"
  local work_dir="$4"
  local output_name="$5"

  printf '%s\n' \
    --mode serial \
    --config "$PIPELINE_CONFIG" \
    --set "paths.data_config=Config/Sequence/TartanAirV1_Challenge_${seq}.yaml" \
    --set "evaluation.gt_pose_file=$GT_ROOT/${seq}.txt" \
    --set pose_frontend.source=macvo \
    --set sequence.start_index=0 \
    --set "sequence.end_index=$n" \
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
    --set "backend.optimization.iterations=$bookkeeping_iters" \
    --set backend.eval_before_optimization=false \
    --set backend.eval_every_train_packets=1000000 \
    --set backend.eval_max_views=0 \
    --set backend.save_every_train_packets=0 \
    --set backend.save_final_ply=false \
    --set backend.wandb_mode=disabled \
    --set backend.write_runtime_artifacts=true \
    --set "paths.work_dir=$work_dir" \
    --set "backend.output_name=$output_name"
}

run_quality_seed0() {
  local seq="$1"
  local n="$2"
  local ntrain="$3"
  local bookkeeping_iters="$4"

  local work_dir="$OUTPUT_ROOT/$seq/quality_seed0"
  local output_name="incremental_${seq}_quality_seed0"
  local endpoint="$work_dir/$output_name/posthoc_global_refinement_endpoint_metrics.json"
  local summary="$work_dir/execution_benchmark_summary.json"
  mkdir -p "$work_dir"

  if [[ -f "$endpoint" && -f "$summary" ]]; then
    echo "[skip quality seed0 complete] $seq"
    return 0
  fi

  echo
  echo "================================================================"
  echo "QUALITY ONCE | $seq | seed=0 | frames=$n | train=$ntrain"
  echo "Online endpoint -> opacity reset -> 10 full train passes -> final endpoint"
  echo "================================================================"

  mapfile -t args < <(common_args "$seq" "$n" "$bookkeeping_iters" "$work_dir" "$output_name")

  set +e
  run_cuda_failfast "$work_dir/run.log" env \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONHASHSEED=0 \
    PIPELINE_BENCHMARK_SEED=0 \
    PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
    PIPELINE_GLOBAL_REFINE_PASSES=10 \
    PIPELINE_GLOBAL_REFINE_SEED=0 \
    CUDA_VISIBLE_DEVICES="$GPU" \
    TORCH_COMPILE_DISABLE=1 \
    python "$QUALITY_RUNNER" \
      --with_pose_metrics \
      "${args[@]}"
  local status=$?
  set -e

  echo "$status" > "$work_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED quality] $seq exit=$status; continuing" >&2
    sleep 5
  fi
  return 0
}

run_timing_once() {
  local seq="$1"
  local n="$2"
  local ntrain="$3"
  local bookkeeping_iters="$4"

  local work_dir="$OUTPUT_ROOT/$seq/timing"
  local output_name="incremental_${seq}_timing"
  local timing_json="$work_dir/$output_name/posthoc_global_refinement_timing.json"
  local summary="$work_dir/execution_benchmark_summary.json"
  mkdir -p "$work_dir"

  if [[ -f "$timing_json" && -f "$summary" ]]; then
    echo "[skip timing complete] $seq"
    return 0
  fi

  echo
  echo "================================================================"
  echo "TIMING ONCE | $seq | frames=$n | train=$ntrain"
  echo "No quality/pose metrics; online timing + 10-pass refinement timing"
  echo "================================================================"

  mapfile -t args < <(common_args "$seq" "$n" "$bookkeeping_iters" "$work_dir" "$output_name")

  set +e
  run_cuda_failfast "$work_dir/run.log" env \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONHASHSEED=0 \
    PIPELINE_BENCHMARK_SEED=0 \
    PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
    PIPELINE_GLOBAL_REFINE_PASSES=10 \
    PIPELINE_GLOBAL_REFINE_SEED=0 \
    CUDA_VISIBLE_DEVICES="$GPU" \
    TORCH_COMPILE_DISABLE=1 \
    python "$TIMING_RUNNER" \
      "${args[@]}" \
      --set backend.evaluation_enabled=false
  local status=$?
  set -e

  echo "$status" > "$work_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED timing] $seq exit=$status; continuing" >&2
    sleep 5
  fi
  return 0
}

echo "=== PHASE 1/2: single quality run for SE002, SE003, SH000-SH003 ==="
for seq in "${QUALITY_TARGETS[@]}"; do
  n=$(count_frames "$seq") || {
    echo "[FAILED dataset check] $seq; skipping quality" >&2
    continue
  }
  ntest=$(( n / 5 ))
  ntrain=$(( n - ntest ))
  bookkeeping_iters=$(( ntrain * 100 ))
  run_quality_seed0 "$seq" "$n" "$ntrain" "$bookkeeping_iters"
done

echo
echo "=== PHASE 2/2: timing-only run for SE000-SE003 + SH000-SH003 ==="
for seq in "${TIMING_TARGETS[@]}"; do
  n=$(count_frames "$seq") || {
    echo "[FAILED dataset check] $seq; skipping timing" >&2
    continue
  }
  ntest=$(( n / 5 ))
  ntrain=$(( n - ntest ))
  bookkeeping_iters=$(( ntrain * 100 ))
  run_timing_once "$seq" "$n" "$ntrain" "$bookkeeping_iters"
done

echo
echo "================================================================"
echo "Prioritized 8-sequence subset attempted. Building seed0 summary..."
echo "================================================================"
python summarize_final8_single_run_10pass.py || true

echo
echo "Re-run this same script to resume any failed/missing run."
