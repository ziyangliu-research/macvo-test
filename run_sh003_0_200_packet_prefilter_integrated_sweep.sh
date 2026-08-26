#!/usr/bin/env bash
set -euo pipefail

# Integrated 200-frame SH003 sweep for local ReSplat packet compression.
#
# Usage:
#   bash run_sh003_0_200_packet_prefilter_integrated_sweep.sh quality
#   bash run_sh003_0_200_packet_prefilter_integrated_sweep.sh timing
#   bash run_sh003_0_200_packet_prefilter_integrated_sweep.sh both
#
# Fixed local policy:
#   opacity reset 0.01 -> 150 alternating-stereo pre-opt -> prune -> 50 post-opt
#   no local densification
# Threshold sweep:
#   0.03 / 0.05 / 0.07
#
# Fixed global policy:
#   preserve locally learned opacity at insertion (no second reset)
#   100 global iterations / train packet
#   maintenance at local iteration 50
#   global maintenance min opacity 0.05
#   historical replay fraction 0.20

MODE="${1:-quality}"
if [[ "$MODE" != "quality" && "$MODE" != "timing" && "$MODE" != "both" ]]; then
  echo "usage: $0 {quality|timing|both}" >&2
  exit 2
fi

cd /home/shiyo/Desktop/MAC-VO

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt"
RUNNER="run_pipeline_execution_benchmark_packet_prefilter_fast.py"

run_one() {
  local run_mode="$1"
  local th="$2"
  local tag
  case "$th" in
    0.03) tag="003" ;;
    0.05) tag="005" ;;
    0.07) tag="007" ;;
    *) echo "unsupported threshold: $th" >&2; exit 3 ;;
  esac

  local work_dir="outputs/SH003_0_200_packet_prefilter_p${tag}_${run_mode}"
  local output_name="incremental_SH003_0_200_packet_prefilter_p${tag}_${run_mode}"

  local split_every
  local split_offset
  local eval_enabled
  local opt_iterations
  local -a pose_metric_arg=()

  if [[ "$run_mode" == "quality" ]]; then
    # Strict held-out test timestamps: 4,9,14,...,199.
    # Test timestamps do not generate ReSplat packets and never supervise either
    # the local packet optimizer or the global backend.
    split_every=5
    split_offset=4
    eval_enabled=true
    # 160 train packets * 100 global iterations.
    opt_iterations=16000
    # ATE is evaluated after the streaming benchmark and does not contaminate
    # streaming_wall_time_sec.
    pose_metric_arg=(--with_pose_metrics)
  else
    # All 200 frames are train packets so timing represents continuous online
    # processing rather than a split in which 20% of timestamps skip ReSplat.
    split_every=1000000
    split_offset=999999
    eval_enabled=false
    opt_iterations=20000
  fi

  echo
  echo "================================================================"
  echo "SH003 0-199 | mode=${run_mode} | local prune threshold=${th}"
  echo "work_dir=${work_dir}"
  echo "================================================================"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED=0 \
  PIPELINE_BENCHMARK_SEED=0 \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.2 \
  PIPELINE_PACKET_FAST_PRE_ITERS=150 \
  PIPELINE_PACKET_FAST_POST_ITERS=50 \
  PIPELINE_PACKET_FAST_RESET_OPACITY=0.01 \
  PIPELINE_PACKET_FAST_PRUNE_THRESHOLD="$th" \
  PIPELINE_PACKET_FAST_BOUNDARY_EVAL=0 \
  CUDA_VISIBLE_DEVICES=0 \
  TORCH_COMPILE_DISABLE=1 \
  python "$RUNNER" \
    --mode serial \
    "${pose_metric_arg[@]}" \
    --config "$CONFIG" \
    --set paths.data_config="$DATA_CONFIG" \
    --set evaluation.gt_pose_file="$GT" \
    --set pose_frontend.source=macvo \
    --set sequence.start_index=0 \
    --set sequence.end_index=200 \
    --set 'resplat_frontend.overrides=["dataset.image_shape=[240,320]"]' \
    --set split.split_every="$split_every" \
    --set split.split_offset="$split_offset" \
    --set backend.local_map_size=10 \
    --set backend.iterations_per_packet=100 \
    --set backend.maintenance_mode=standard \
    --set backend.maintenance_after_local_iteration=50 \
    --set backend.maintenance_min_opacity=0.05 \
    --set backend.reset_new_packet_opacity=false \
    --set backend.optimization.iterations="$opt_iterations" \
    --set backend.eval_every_train_packets=1000000 \
    --set backend.evaluation_enabled="$eval_enabled" \
    --set backend.save_every_train_packets=0 \
    --set backend.save_final_ply=false \
    --set backend.wandb_mode=disabled \
    --set paths.work_dir="$work_dir" \
    --set backend.output_name="$output_name"
}

run_group() {
  local run_mode="$1"
  for th in 0.03 0.05 0.07; do
    run_one "$run_mode" "$th"
  done
}

if [[ "$MODE" == "quality" ]]; then
  run_group quality
elif [[ "$MODE" == "timing" ]]; then
  run_group timing
else
  run_group quality
  run_group timing
fi
