#!/usr/bin/env bash
set -u

# Two additional independent runs for the SH003 stage-order ablation only.
# Existing seed-0 results are preserved. This launcher always executes fresh
# seed-1 and seed-2 runs and never skips because a summary already exists.
#
# Common protocol:
#   SH003 [0,200), strict 8:2 = 160 mapping + 40 held-out
#   W=20, rho=.30, B=100, M=50, Th=.10
#   insertion opacity cap=.01
#   ReSplat refine_steps=0, serial/default CUDA stream
#   no intermediate evaluation, no global refinement
#
# Variants:
#   A Two-stage:         50 recent -> maintenance -> 20 recent + 30 history
#   B Mixed-from-start:  exact same 70 recent + 30 history total, distributed
#                        from iteration 1; maintenance remains at iteration 50

cd /home/shiyo/Desktop/MAC-VO || exit 1

GPU="${GPU:-0}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
RUNNER="run_pipeline_execution_benchmark_repro_reviewer_ablation.py"
ROOT="${ROOT:-outputs/SH003_0_200_reviewer_stage_order_opacity_cap}"

run_case() {
  local seed="$1"
  local case_name="$2"
  local replay_order="$3"
  local work_dir="$ROOT/seed${seed}/$case_name"
  local output_name="incremental_${case_name}_seed${seed}"

  # This script is explicitly for fresh reruns. Remove any stale/incomplete or
  # even previously-complete directory for this seed/case rather than skipping.
  rm -rf "$work_dir"
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
seed=$seed
EOF

  echo
  echo "======================================================================"
  echo "Fresh stage-order run: seed=$seed | $case_name"
  echo "SH003 [0,200) | strict 8:2 | W20/R30/B100/M50/Th.10 | cap=.01"
  echo "replay_order=$replay_order"
  echo "work_dir=$work_dir"
  echo "======================================================================"

  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED="$seed" \
  PIPELINE_BENCHMARK_SEED="$seed" \
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
    echo "[FAILED] seed=$seed $case_name exit=$status; continuing" >&2
  fi
}

for seed in 1 2; do
  run_case "$seed" baseline_two_stage_cap001 post_maintenance
  run_case "$seed" mixed_from_start_cap001 from_start
done

echo
echo "======================================================================"
echo "Finished two additional seeds for both stage-order variants."
echo "Existing seed-0 results were not touched."
echo "New results: $ROOT/seed1 and $ROOT/seed2"
echo "======================================================================"
