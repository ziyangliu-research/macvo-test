#!/usr/bin/env bash
set -uo pipefail

# Full-sequence SH003 strict 8:2 held-out novel-view scaling experiment.
#
# Four system versions:
#   V0: replay=0%,  global prune threshold=0.005
#   V1: replay=30%, global prune threshold=0.005
#   V2: replay=30%, global prune threshold=0.05
#   V3: replay=30%, global prune threshold=0.10
#
# Everything else is fixed:
#   - ReSplat refine_steps=0
#   - newest packet opacity cap/reset=0.01
#   - local_map_size=10
#   - 100 global optimizer iterations per mapping packet
#   - standard maintenance at local iteration 50
#   - strict 8:2 mapping/test split: frames 4,9,14,... are held out from mapping
#   - held-out frames still participate in MAC-VO tracking
#
# The script detects the actual SH003 length from image_left at runtime.
# It continues to later cases even if one case fails/OOMs, and skips cases whose
# execution_benchmark_summary.json already exists.
#
# Usage:
#   bash run_sh003_full_novel_view_scaling_sweep.sh
#   bash run_sh003_full_novel_view_scaling_sweep.sh v2 v3
#

cd /home/shiyo/Desktop/MAC-VO || exit 1

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
DATA_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo/SH003"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt"
RUNNER="run_pipeline_execution_benchmark_repro_empty_safe.py"

NUM_FRAMES=$(python - <<'PY'
from pathlib import Path
root = Path('/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo/SH003/image_left')
files = sorted(root.glob('*_left.png'))
if not files:
    raise SystemExit(f'no *_left.png files found in {root}')
print(len(files))
PY
) || exit 1

NUM_TEST=$(( NUM_FRAMES / 5 ))
NUM_TRAIN=$(( NUM_FRAMES - NUM_TEST ))
OPT_ITERS=$(( NUM_TRAIN * 100 ))

printf '\n============================================================\n'
printf 'Detected full SH003: %d frames\n' "$NUM_FRAMES"
printf 'Strict 8:2 protocol: %d mapping / %d held-out\n' "$NUM_TRAIN" "$NUM_TEST"
printf 'Backend bookkeeping iterations: %d\n' "$OPT_ITERS"
printf '============================================================\n\n'

# Default order intentionally runs compact systems first.  This makes sure the
# most important V2/V3 long-sequence results are obtained even if V0/V1 later
# become prohibitively large or OOM.
if (( $# > 0 )); then
  CASES=("$@")
else
  CASES=(v2 v3 v1 v0)
fi

run_case() {
  local key="$1"
  local name replay threshold
  case "$key" in
    v0)
      name="v0_baseline_r0_th0005"; replay="0"; threshold="0.005" ;;
    v1)
      name="v1_replay30_th0005"; replay="0.30"; threshold="0.005" ;;
    v2)
      name="v2_replay30_th005"; replay="0.30"; threshold="0.05" ;;
    v3)
      name="v3_replay30_th010"; replay="0.30"; threshold="0.10" ;;
    *)
      echo "Unknown case '$key' (expected v0/v1/v2/v3)" >&2
      return 2 ;;
  esac

  local work_dir="outputs/SH003_full_novelview_8to2_${name}"
  local output_name="incremental_SH003_full_novelview_8to2_${name}"
  local summary="$work_dir/execution_benchmark_summary.json"
  local log="$work_dir/run.log"
  mkdir -p "$work_dir"

  echo
  echo "================================================================"
  echo "FULL SH003 | ${key^^} | ${name}"
  echo "frames=${NUM_FRAMES} train=${NUM_TRAIN} test=${NUM_TEST}"
  echo "replay=${replay} threshold=${threshold}"
  echo "work_dir=${work_dir}"
  echo "================================================================"

  if [[ -s "$summary" ]]; then
    echo "[skip] completed summary already exists: $summary"
    return 0
  fi

  local -a env_args=(
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "PYTHONHASHSEED=0"
    "PIPELINE_BENCHMARK_SEED=0"
    "CUDA_VISIBLE_DEVICES=0"
    "TORCH_COMPILE_DISABLE=1"
  )

  local status
  if [[ "$replay" == "0" ]]; then
    env -u PIPELINE_HISTORICAL_REPLAY_FRACTION "${env_args[@]}" \
      python "$RUNNER" \
        --mode serial \
        --with_pose_metrics \
        --config "$CONFIG" \
        --set paths.data_config="$DATA_CONFIG" \
        --set evaluation.gt_pose_file="$GT" \
        --set pose_frontend.source=macvo \
        --set sequence.start_index=0 \
        --set sequence.end_index="$NUM_FRAMES" \
        --set 'resplat_frontend.overrides=["dataset.image_shape=[240,320]"]' \
        --set split.split_every=5 \
        --set split.split_offset=4 \
        --set split.split_index_mode=local_index \
        --set backend.local_map_size=10 \
        --set backend.iterations_per_packet=100 \
        --set backend.reset_new_packet_opacity=true \
        --set backend.new_packet_reset_max_opacity=0.01 \
        --set backend.maintenance_mode=standard \
        --set backend.maintenance_after_local_iteration=50 \
        --set backend.maintenance_min_opacity="$threshold" \
        --set backend.optimization.iterations="$OPT_ITERS" \
        --set backend.eval_before_optimization=false \
        --set backend.eval_every_train_packets=1000000 \
        --set backend.evaluation_enabled=true \
        --set backend.save_every_train_packets=0 \
        --set backend.save_final_ply=false \
        --set backend.wandb_mode=disabled \
        --set paths.work_dir="$work_dir" \
        --set backend.output_name="$output_name" \
        2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
  else
    env "${env_args[@]}" "PIPELINE_HISTORICAL_REPLAY_FRACTION=${replay}" \
      python "$RUNNER" \
        --mode serial \
        --with_pose_metrics \
        --config "$CONFIG" \
        --set paths.data_config="$DATA_CONFIG" \
        --set evaluation.gt_pose_file="$GT" \
        --set pose_frontend.source=macvo \
        --set sequence.start_index=0 \
        --set sequence.end_index="$NUM_FRAMES" \
        --set 'resplat_frontend.overrides=["dataset.image_shape=[240,320]"]' \
        --set split.split_every=5 \
        --set split.split_offset=4 \
        --set split.split_index_mode=local_index \
        --set backend.local_map_size=10 \
        --set backend.iterations_per_packet=100 \
        --set backend.reset_new_packet_opacity=true \
        --set backend.new_packet_reset_max_opacity=0.01 \
        --set backend.maintenance_mode=standard \
        --set backend.maintenance_after_local_iteration=50 \
        --set backend.maintenance_min_opacity="$threshold" \
        --set backend.optimization.iterations="$OPT_ITERS" \
        --set backend.eval_before_optimization=false \
        --set backend.eval_every_train_packets=1000000 \
        --set backend.evaluation_enabled=true \
        --set backend.save_every_train_packets=0 \
        --set backend.save_final_ply=false \
        --set backend.wandb_mode=disabled \
        --set paths.work_dir="$work_dir" \
        --set backend.output_name="$output_name" \
        2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
  fi

  if (( status == 0 )); then
    echo "[done] $key completed successfully"
  else
    echo "[FAILED] $key exited with status $status; continuing to next case" >&2
    echo "$status" > "$work_dir/exit_status.txt"
  fi
  return 0
}

for key in "${CASES[@]}"; do
  run_case "$key"
done

echo
echo "Full SH003 sweep finished/attempted. Summarize with:"
echo "  python summarize_sh003_full_novel_view_scaling.py"
