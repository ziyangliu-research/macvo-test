#!/usr/bin/env bash
set -u

# Four-sequence EuRoC evaluation:
#   MH02 V101 V201 MH05
# Protocol:
#   existing EuRoC valid stereo/GT interval -> temporal stride=5 -> strict 8:2
#   W20 / replay30% / B100 / M50 / Th=.10
#   seed0 quality endpoint run + independent timing-only run
#   post-hoc global refinement = 10 full train passes
#
# Restart-safe: complete runs are skipped.
# CUDA launch timeouts are fail-fast so one failed sequence does not block all.

cd /home/shiyo/Desktop/MAC-VO || exit 1

PIPELINE_CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_EuRoC_Stride5_Final.yaml"
QUALITY_RUNNER="run_pipeline_execution_benchmark_repro_euroc_endpoint_lpips.py"
TIMING_RUNNER="run_pipeline_execution_benchmark_repro_euroc_timing_only.py"
OUTPUT_ROOT="outputs/euroc4_stride5_10pass"
SEQUENCES=(MH02 V101 V201 MH05)
STRIDE=5
SEED=0
GPU="${GPU:-0}"

mkdir -p "$OUTPUT_ROOT"

python - <<'PY'
from lpips import LPIPS
print("[preflight] LPIPS import OK")
PY
if [[ $? -ne 0 ]]; then
  echo "[fatal] lpips import failed in current environment" >&2
  exit 2
fi

config_for() {
  case "$1" in
    MH02) echo "Config/Sequence/EuRoC_MH02_local.yaml" ;;
    V101) echo "Config/Sequence/EuRoC_V101_local.yaml" ;;
    V201) echo "Config/Sequence/EuRoC_V201_local.yaml" ;;
    MH05) echo "Config/Sequence/EuRoC_MH05_local.yaml" ;;
    *) echo "unknown EuRoC sequence: $1" >&2; return 2 ;;
  esac
}

# Count the actual sequence AFTER the existing EuRoC loader has synchronized
# cam0/cam1 and masked camera timestamps to the GT-covered interval.
count_valid_frames() {
  local data_config="$1"
  python - "$data_config" <<'PY'
from pathlib import Path
import sys
from DataLoader import SequenceBase, StereoFrame
from Utility.Config import load_config

cfg, _ = load_config(Path(sys.argv[1]).resolve())
seq = SequenceBase[StereoFrame].instantiate(cfg.type, cfg.args)
print(len(seq))
PY
}

run_cuda_failfast() {
  local log="$1"
  shift
  : > "$log"
  setsid "$@" > >(tee "$log") 2>&1 &
  local pid=$!
  local fatal_seen=0

  while kill -0 "$pid" 2>/dev/null; do
    if tail -n 180 "$log" 2>/dev/null | grep -Eq \
      'cudaErrorLaunchTimeout|CUDA error: the launch timed out|torch\.AcceleratorError: CUDA error'; then
      fatal_seen=1
      echo "[fail-fast] CUDA launch timeout; terminating PGID=$pid" | tee -a "$log"
      kill -TERM -- "-$pid" 2>/dev/null || true
      for _ in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then break; fi
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || true
      fi
      break
    fi
    sleep 3
  done

  wait "$pid" 2>/dev/null
  local status=$?
  if [[ "$fatal_seen" -eq 1 ]]; then return 86; fi
  return "$status"
}

common_args() {
  local seq="$1"
  local data_config="$2"
  local valid_n="$3"
  local bookkeeping_iters="$4"
  local work_dir="$5"
  local output_name="$6"

  printf '%s\n' \
    --mode serial \
    --config "$PIPELINE_CONFIG" \
    --set "paths.data_config=$data_config" \
    --set sequence.start_index=0 \
    --set "sequence.end_index=$valid_n" \
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

run_quality() {
  local seq="$1"
  local data_config="$2"
  local valid_n="$3"
  local retained_n="$4"
  local ntrain="$5"
  local ntest="$6"
  local bookkeeping_iters="$7"

  local work_dir="$OUTPUT_ROOT/$seq/quality_seed0"
  local output_name="incremental_${seq}_quality_seed0"
  local endpoint="$work_dir/$output_name/posthoc_global_refinement_endpoint_metrics.json"
  local summary="$work_dir/execution_benchmark_summary.json"
  mkdir -p "$work_dir"

  if [[ -f "$endpoint" && -f "$summary" ]]; then
    echo "[skip quality complete] $seq"
    return 0
  fi

  echo
  echo "================================================================"
  echo "EUROC QUALITY | $seq | seed=0"
  echo "valid=$valid_n -> stride5=$retained_n -> train/test=$ntrain/$ntest"
  echo "Online endpoint -> reset -> 10 full train passes -> final endpoint"
  echo "================================================================"

  mapfile -t args < <(
    common_args "$seq" "$data_config" "$valid_n" "$bookkeeping_iters" "$work_dir" "$output_name"
  )

  set +e
  run_cuda_failfast "$work_dir/run.log" env \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONHASHSEED="$SEED" \
    PIPELINE_BENCHMARK_SEED="$SEED" \
    PIPELINE_FRAME_STRIDE="$STRIDE" \
    PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
    PIPELINE_GLOBAL_REFINE_PASSES=10 \
    PIPELINE_GLOBAL_REFINE_SEED="$SEED" \
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

run_timing() {
  local seq="$1"
  local data_config="$2"
  local valid_n="$3"
  local retained_n="$4"
  local ntrain="$5"
  local ntest="$6"
  local bookkeeping_iters="$7"

  local work_dir="$OUTPUT_ROOT/$seq/timing_seed0"
  local output_name="incremental_${seq}_timing_seed0"
  local timing_json="$work_dir/$output_name/posthoc_global_refinement_timing.json"
  local summary="$work_dir/execution_benchmark_summary.json"
  mkdir -p "$work_dir"

  if [[ -f "$timing_json" && -f "$summary" ]]; then
    echo "[skip timing complete] $seq"
    return 0
  fi

  echo
  echo "================================================================"
  echo "EUROC TIMING | $seq | seed=0"
  echo "valid=$valid_n -> stride5=$retained_n -> train/test=$ntrain/$ntest"
  echo "No quality/pose evaluation; Online timing + 10-pass refinement timing"
  echo "================================================================"

  mapfile -t args < <(
    common_args "$seq" "$data_config" "$valid_n" "$bookkeeping_iters" "$work_dir" "$output_name"
  )

  set +e
  run_cuda_failfast "$work_dir/run.log" env \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONHASHSEED="$SEED" \
    PIPELINE_BENCHMARK_SEED="$SEED" \
    PIPELINE_FRAME_STRIDE="$STRIDE" \
    PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
    PIPELINE_GLOBAL_REFINE_PASSES=10 \
    PIPELINE_GLOBAL_REFINE_SEED="$SEED" \
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

# Resolve actual valid counts once, before any long GPU work.
declare -A VALID RETAINED TRAIN TEST BOOKKEEPING CONFIG

echo "=== EuRoC stride-5 dataset preflight ==="
for seq in "${SEQUENCES[@]}"; do
  cfg=$(config_for "$seq") || exit 2
  if [[ ! -f "$cfg" ]]; then
    echo "[fatal] missing config: $cfg" >&2
    exit 2
  fi
  n=$(count_valid_frames "$cfg") || {
    echo "[fatal] failed to load EuRoC sequence $seq" >&2
    exit 2
  }
  retained=$(( (n + STRIDE - 1) / STRIDE ))
  ntest=$(( retained / 5 ))
  ntrain=$(( retained - ntest ))
  bookkeeping=$(( ntrain * 100 ))

  CONFIG[$seq]="$cfg"
  VALID[$seq]="$n"
  RETAINED[$seq]="$retained"
  TRAIN[$seq]="$ntrain"
  TEST[$seq]="$ntest"
  BOOKKEEPING[$seq]="$bookkeeping"

  printf "%-5s valid=%-5d stride5=%-4d train=%-4d test=%-4d\n" \
    "$seq" "$n" "$retained" "$ntrain" "$ntest"
done

echo
echo "=== PHASE 1/2: quality seed0 for MH02 V101 V201 MH05 ==="
for seq in "${SEQUENCES[@]}"; do
  run_quality "$seq" "${CONFIG[$seq]}" "${VALID[$seq]}" \
    "${RETAINED[$seq]}" "${TRAIN[$seq]}" "${TEST[$seq]}" "${BOOKKEEPING[$seq]}"
done

echo
echo "=== PHASE 2/2: timing seed0 for MH02 V101 V201 MH05 ==="
for seq in "${SEQUENCES[@]}"; do
  run_timing "$seq" "${CONFIG[$seq]}" "${VALID[$seq]}" \
    "${RETAINED[$seq]}" "${TRAIN[$seq]}" "${TEST[$seq]}" "${BOOKKEEPING[$seq]}"
done

echo
echo "=== Building EuRoC summary ==="
python summarize_euroc4_stride5_seed0.py || true

echo "Re-run the same launcher to resume any missing/failed run."
