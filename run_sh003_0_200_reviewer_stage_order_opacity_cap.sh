#!/usr/bin/env bash
set -u

# Reviewer-requested stage-order ablation on SH003 [0,200), strict held-out 8:2.
# Common protocol for both cases:
#   W=20, rho=.30, B=100, M=50, Th=.10, refine_steps=0
#   insertion opacity cap=.01
#   160 mapping/train + 40 held-out test timestamps
#   serial execution, ReSplat default CUDA stream, seed fixed
#   no intermediate metric rendering, no global/post-hoc refinement
#
# Two fresh runs:
#   A baseline_two_stage_cap001:
#       recent-only through M=50 -> prune -> mixed replay
#   B mixed_from_start_cap001:
#       exact same 70 recent + 30 history updates, but history spread from iter 1
#
# Only stage order changes between A and B.

cd /home/shiyo/Desktop/MAC-VO || exit 1

GPU="${GPU:-0}"
SEED="${SEED:-0}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
RUNNER="run_pipeline_execution_benchmark_repro_reviewer_ablation.py"
ROOT="${ROOT:-outputs/SH003_0_200_reviewer_stage_order_opacity_cap}"

mkdir -p "$ROOT"

run_case() {
  local case_name="$1"
  local replay_order="$2"
  local work_dir="$ROOT/$case_name"
  local output_name="incremental_${case_name}"
  local summary="$work_dir/execution_benchmark_summary.json"

  if [[ -f "$summary" ]]; then
    echo "[skip complete] $case_name"
    return 0
  fi

  mkdir -p "$work_dir"
  cat > "$work_dir/protocol.txt" <<EOF
sequence=SH003[0,200)
split=strict8to2(split_every=5,split_offset=4)
W=20
rho=0.30
B=100
M=50
prune_threshold=0.10
replay_order=$replay_order
reset_new_packet_opacity=true
new_packet_reset_max_opacity=0.01
seed=$SEED
EOF

  echo
  echo "======================================================================"
  echo "Reviewer stage-order ablation: $case_name"
  echo "SH003 [0,200) | strict 8:2 | W20/R30/B100/M50/Th.10 | cap=.01"
  echo "replay_order=$replay_order"
  echo "======================================================================"

  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED="$SEED" \
  PIPELINE_BENCHMARK_SEED="$SEED" \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  PIPELINE_HISTORICAL_REPLAY_ORDER="$replay_order" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TORCH_COMPILE_DISABLE=1 \
  python "$RUNNER" \
    --mode serial \
    --config "$CONFIG" \
    --set paths.data_config="$DATA_CONFIG" \
    --set pose_frontend.source=macvo \
    --set sequence.start_index=0 \
    --set sequence.end_index=200 \
    --set 'resplat_frontend.overrides=["dataset.image_shape=[240,320]"]' \
    --set resplat_frontend.refine_steps=0 \
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
    --set backend.eval_max_views=0 \
    --set backend.evaluation_enabled=true \
    --set backend.save_every_train_packets=0 \
    --set backend.save_final_ply=false \
    --set backend.wandb_mode=disabled \
    --set backend.write_runtime_artifacts=true \
    --set paths.work_dir="$work_dir" \
    --set backend.output_name="$output_name" \
    2>&1 | tee "$work_dir/run.log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$work_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED] $case_name exit=$status; continuing to next case" >&2
  fi
}

run_case baseline_two_stage_cap001 post_maintenance
run_case mixed_from_start_cap001 from_start

echo
echo "======================================================================"
echo "Stage-order ablation runs finished."
echo "Summarize with:"
echo "  python summarize_sh003_reviewer_stage_order_opacity_cap.py --root $ROOT"
echo "======================================================================"
