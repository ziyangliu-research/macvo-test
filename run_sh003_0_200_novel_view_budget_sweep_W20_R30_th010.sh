#!/usr/bin/env bash
set -u

# SH003 [0,200) strict 8:2 held-out novel-view optimization-budget ablation.
# Fixed system parameters:
#   W = 20 recent-view working set
#   historical replay ratio rho = 0.30
#   global opacity threshold = 0.10
#   newest-packet opacity reset = 0.01
#   strict held-out split: 160 mapping / 40 test
#
# Sweep total optimization budget B while keeping phase proportions fixed:
#   B=50  -> maintenance M=25, historical slots R=15
#   B=100 -> maintenance M=50, historical slots R=30 (existing result reused)
#   B=200 -> maintenance M=100, historical slots R=60
#
# Thus maintenance is always at B/2 and 30% of the total optimizer budget is
# allocated to historical supervision.  Only B changes.

cd /home/shiyo/Desktop/MAC-VO || exit 1

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt"
RUNNER="run_pipeline_execution_benchmark_repro_fixed_recent10_eval.py"

run_case() {
  local budget="$1"
  local maintenance=$((budget / 2))
  local total_backend_iters=$((160 * budget))
  local name="W20_R30_B${budget}_th010"
  local work_dir="outputs/SH003_0_200_novelview_8to2_${name}"
  local output_name="incremental_SH003_0_200_novelview_8to2_${name}"
  local summary="${work_dir}/execution_benchmark_summary.json"

  if [[ -f "$summary" ]]; then
    echo "[skip complete] ${name}"
    return 0
  fi

  mkdir -p "$work_dir"
  echo
  echo "================================================================"
  echo "SH003 0-199 | strict 8:2 | W=20 | rho=.30 | B=${budget} | M=${maintenance} | Th=.10"
  echo "work_dir=${work_dir}"
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
    --set backend.local_map_size=20 \
    --set backend.iterations_per_packet="$budget" \
    --set backend.reset_new_packet_opacity=true \
    --set backend.new_packet_reset_max_opacity=0.01 \
    --set backend.maintenance_mode=standard \
    --set backend.maintenance_after_local_iteration="$maintenance" \
    --set backend.maintenance_min_opacity=0.10 \
    --set backend.optimization.iterations="$total_backend_iters" \
    --set backend.eval_before_optimization=false \
    --set backend.eval_every_train_packets=1000000 \
    --set backend.evaluation_enabled=true \
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
    echo "[FAILED] ${name}, exit=${status}; continuing" >&2
  fi
}

# B=100 already exists as:
# outputs/SH003_0_200_novelview_8to2_W20_R30_th010
run_case 50
run_case 200

echo
echo "Optimization-budget sweep finished. Summarize B=50/100/200 with:"
echo "  python summarize_sh003_0_200_novel_view_budget_W20_R30_th010.py"
