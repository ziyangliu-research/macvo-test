#!/usr/bin/env bash
set -euo pipefail

cd /home/shiyo/Desktop/MAC-VO

WORK_DIR="outputs/SH003_0_200_final_W20_R30_B100_th010_global_refine5x"
OUTPUT_NAME="incremental_SH003_0_200_final_W20_R30_B100_th010_global_refine5x"
mkdir -p "$WORK_DIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONHASHSEED=0 \
PIPELINE_BENCHMARK_SEED=0 \
PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
PIPELINE_GLOBAL_REFINE_PASSES=5 \
PIPELINE_GLOBAL_REFINE_EVAL_EVERY=100 \
PIPELINE_GLOBAL_REFINE_RESET_OPACITY_MAX=0.01 \
PIPELINE_GLOBAL_REFINE_SEED=0 \
CUDA_VISIBLE_DEVICES=0 \
TORCH_COMPILE_DISABLE=1 \
python run_pipeline_execution_benchmark_repro_posthoc_global_refine.py \
  --mode serial \
  --with_pose_metrics \
  --config Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml \
  --set paths.data_config=Config/Sequence/TartanAirV1_Challenge_SH003.yaml \
  --set evaluation.gt_pose_file=/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt \
  --set pose_frontend.source=macvo \
  --set sequence.start_index=0 \
  --set sequence.end_index=200 \
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
  --set backend.optimization.iterations=16000 \
  --set backend.eval_before_optimization=false \
  --set backend.eval_every_train_packets=1000000 \
  --set backend.evaluation_enabled=true \
  --set backend.eval_max_views=0 \
  --set backend.save_every_train_packets=0 \
  --set backend.save_final_ply=true \
  --set backend.wandb_mode=disabled \
  --set backend.write_runtime_artifacts=true \
  --set paths.work_dir="$WORK_DIR" \
  --set backend.output_name="$OUTPUT_NAME" \
  2>&1 | tee "$WORK_DIR/run.log"

python summarize_sh003_posthoc_global_refinement.py "$WORK_DIR"
