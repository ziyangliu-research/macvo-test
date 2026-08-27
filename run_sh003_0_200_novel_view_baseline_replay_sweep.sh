#!/usr/bin/env bash
set -euo pipefail

# SH003 [0,200) strict held-out novel-view evaluation for the pre-replay baseline
# and the same system with fixed-budget historical replay.
#
# Protocol:
#   train/mapping timestamps: 0,1,2,3,5,6,7,8,...
#   held-out timestamps:      4,9,14,...,199
# Held-out timestamps still go through MAC-VO tracking but generate no ReSplat
# packet and never supervise local/global Gaussian optimization.
#
# V0 baseline:
#   newest-packet opacity reset 0.01
#   100 global iterations / train packet
#   local_map_size 10
#   standard maintenance at local iteration 50
#   maintenance_min_opacity 0.005
#   no historical replay
#
# V1 replay20:
#   identical to V0, but 20/100 optimizer iterations per packet are assigned to
#   randomly sampled historical train cameras after maintenance when history exists.

cd /home/shiyo/Desktop/MAC-VO

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt"
RUNNER="run_pipeline_execution_benchmark_repro.py"

run_case() {
  local name="$1"
  local replay="$2"
  local work_dir="outputs/SH003_0_200_novelview_8to2_${name}"
  local output_name="incremental_SH003_0_200_novelview_8to2_${name}"

  echo
  echo "================================================================"
  echo "SH003 0-199 | strict 8:2 held-out | ${name}"
  echo "replay=${replay}"
  echo "work_dir=${work_dir}"
  echo "================================================================"

  local -a env_args=(
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "PYTHONHASHSEED=0"
    "PIPELINE_BENCHMARK_SEED=0"
    "CUDA_VISIBLE_DEVICES=0"
    "TORCH_COMPILE_DISABLE=1"
  )

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
        --set sequence.end_index=200 \
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
        --set backend.maintenance_min_opacity=0.005 \
        --set backend.optimization.iterations=16000 \
        --set backend.eval_before_optimization=false \
        --set backend.eval_every_train_packets=1000000 \
        --set backend.evaluation_enabled=true \
        --set backend.save_every_train_packets=0 \
        --set backend.save_final_ply=false \
        --set backend.wandb_mode=disabled \
        --set paths.work_dir="$work_dir" \
        --set backend.output_name="$output_name"
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
        --set sequence.end_index=200 \
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
        --set backend.maintenance_min_opacity=0.005 \
        --set backend.optimization.iterations=16000 \
        --set backend.eval_before_optimization=false \
        --set backend.eval_every_train_packets=1000000 \
        --set backend.evaluation_enabled=true \
        --set backend.save_every_train_packets=0 \
        --set backend.save_final_ply=false \
        --set backend.wandb_mode=disabled \
        --set paths.work_dir="$work_dir" \
        --set backend.output_name="$output_name"
  fi
}

run_case baseline_no_replay 0
run_case replay20 0.2

echo
echo "Both runs finished. Summarize with:"
echo "  python summarize_sh003_0_200_novel_view_baseline_replay.py"
