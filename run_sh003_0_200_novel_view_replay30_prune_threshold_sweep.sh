#!/usr/bin/env bash
set -euo pipefail

# SH003 [0,200) strict 8:2 held-out novel-view pruning-threshold sweep.
# Replay budget is fixed to 30% (50 recent pre-maintenance + 20 recent/30 history post-maintenance).
# Existing threshold=0.005 replay30 result is intentionally reused.
# This script runs only 0.01/0.02/0.03/0.05/0.10.
#
# Fixed protocol/system:
#   held-out timestamps: 4,9,14,...,199 (tracking only; no ReSplat/map supervision)
#   160 mapping packets / 40 held-out test cameras
#   newest-packet opacity reset = 0.01
#   global optimization = 100 iterations / mapping packet
#   local_map_size = 10
#   maintenance at local iteration 50
#   historical replay fraction = 0.30
#   total optimizer budget always 100
#
# The empty-map-safe runner changes nothing while Gaussians exist. If a high
# threshold prunes the map to zero, it skips only the remaining invalid backward
# passes for that packet; it does not rescue Gaussians or weaken the threshold.

cd /home/shiyo/Desktop/MAC-VO

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt"
RUNNER="run_pipeline_execution_benchmark_repro_empty_safe.py"

run_case() {
  local th="$1"
  local tag
  case "$th" in
    0.01) tag="001" ;;
    0.02) tag="002" ;;
    0.03) tag="003" ;;
    0.05) tag="005" ;;
    0.10) tag="010" ;;
    *) echo "unsupported threshold: $th" >&2; exit 3 ;;
  esac

  local name="replay30_th${tag}"
  local work_dir="outputs/SH003_0_200_novelview_8to2_${name}"
  local output_name="incremental_SH003_0_200_novelview_8to2_${name}"

  echo
  echo "================================================================"
  echo "SH003 0-199 | strict 8:2 | replay=30% | global prune Th=${th}"
  echo "work_dir=${work_dir}"
  echo "================================================================"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED=0 \
  PIPELINE_BENCHMARK_SEED=0 \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
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
    --set backend.maintenance_min_opacity="$th" \
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

for th in 0.01 0.02 0.03 0.05 0.10; do
  run_case "$th"
done

echo
echo "Pruning-threshold sweep finished. Summarize 0.005/0.01/0.02/0.03/0.05/0.10 with:"
echo "  python summarize_sh003_0_200_novel_view_replay30_prune_threshold.py"
