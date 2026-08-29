#!/usr/bin/env bash
set -u

# SH003 [0,200) strict 8:2 held-out replay-boundary extension.
# W is fixed to the best tested recent working-set size (20), and only the
# historical replay allocation is extended beyond the previous R=30 boundary.
#
# Fixed:
#   W = 20 recent mapping cameras
#   B = 100 optimizer steps / mapping packet
#   maintenance = 50 (half-budget)
#   global opacity threshold = 0.10
#   newest-packet opacity reset = 0.01
#   strict held-out split = 160 mapping / 40 test
#
# New cells:
#   R=40: 50 recent pre-maintenance + 10 recent / 40 historical post-maintenance
#   R=50: 50 recent pre-maintenance +  0 recent / 50 historical post-maintenance
#
# The fixed_recent_10 evaluator is kept for protocol consistency with the prior
# W x R sweep; optimization behavior is unchanged.

cd /home/shiyo/Desktop/MAC-VO || exit 1

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt"
RUNNER="run_pipeline_execution_benchmark_repro_fixed_recent10_eval.py"

run_case() {
  local replay_slots="$1"
  local fraction
  fraction=$(python - <<PY
print(${replay_slots}/100.0)
PY
)

  local name="W20_R${replay_slots}_th010"
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
  echo "SH003 0-199 | strict 8:2 | W=20 | R=${replay_slots}/100 | Th=.10"
  echo "B=100 | M=50 | work_dir=${work_dir}"
  echo "================================================================"

  set +e
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

run_case 40
run_case 50

echo
echo "Replay-boundary extension finished. Summarize W=20, R=5..50 with:"
echo "  python summarize_sh003_0_200_novel_view_W20_replay_boundary_th010.py"
