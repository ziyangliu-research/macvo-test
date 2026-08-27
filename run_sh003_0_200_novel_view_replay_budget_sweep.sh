#!/usr/bin/env bash
set -euo pipefail

# Complete the SH003 [0,200) strict 8:2 replay-budget sweep.
# Existing results for replay=0 and replay=0.20 are intentionally reused.
# This script runs only replay=0.10/0.30/0.40/0.50.
#
# Fixed protocol/system:
#   held-out timestamps: 4,9,14,...,199 (tracking only; no ReSplat/map supervision)
#   160 mapping packets / 40 held-out test cameras
#   newest-packet opacity reset = 0.01
#   global optimization = 100 iterations / mapping packet
#   local_map_size = 10
#   maintenance at local iteration 50
#   global maintenance opacity threshold = 0.005
#   total optimizer budget always 100; replay only reallocates post-maintenance slots
#
# Replay definition:
#   replay10: 50 recent before maintenance + (40 recent + 10 historical)
#   replay30: 50 recent before maintenance + (20 recent + 30 historical)
#   replay40: 50 recent before maintenance + (10 recent + 40 historical)
#   replay50: 50 recent before maintenance + ( 0 recent + 50 historical)

cd /home/shiyo/Desktop/MAC-VO

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt"
RUNNER="run_pipeline_execution_benchmark_repro.py"

run_case() {
  local percent="$1"
  local fraction="$2"
  local name="replay${percent}"
  local work_dir="outputs/SH003_0_200_novelview_8to2_${name}"
  local output_name="incremental_SH003_0_200_novelview_8to2_${name}"

  echo
  echo "================================================================"
  echo "SH003 0-199 | strict 8:2 held-out | replay=${percent}%"
  echo "fraction=${fraction} | work_dir=${work_dir}"
  echo "================================================================"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED=0 \
  PIPELINE_BENCHMARK_SEED=0 \
  PIPELINE_HISTORICAL_REPLAY_FRACTION="$fraction" \
  CUDA_VISIBLE_DEVICES=0 \
  TORCH_COMPILE_DISABLE=1 \
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
}

run_case 10 0.10
run_case 30 0.30
run_case 40 0.40
run_case 50 0.50

echo
echo "Replay-budget sweep finished. Summarize all 0/10/20/30/40/50 results with:"
echo "  python summarize_sh003_0_200_novel_view_replay_budget.py"
