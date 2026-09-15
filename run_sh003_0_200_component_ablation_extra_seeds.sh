#!/usr/bin/env bash
set -u

# Two additional seeds (1,2) for the five-row SH003 component ablation.
# Existing seed-0 results are preserved and are NOT rerun here.
# Completed seed1/seed2 cases are skipped when execution_benchmark_summary.json exists.
# Incomplete cases are rerun in-place without deleting their work directories.
#
# Common protocol:
#   SH003 [0,200), strict 8:2 = 160 mapping + 40 held-out
#   W=20, B=100, M=50, insertion opacity cap=.01
#   ReSplat refine_steps=0, serial/default CUDA stream
#   no intermediate metric rendering, no global refinement
#
# Rows:
#   ff_only       : FF insertion only; no optimization, no maintenance, no replay
#   incremental   : IncOpt + weak/default pruning Th=.005, no replay
#   strong_prune  : IncOpt + strong pruning Th=.10, no replay
#   historical    : IncOpt + weak/default pruning Th=.005 + replay rho=.30
#   full          : IncOpt + strong pruning Th=.10 + replay rho=.30

cd /home/shiyo/Desktop/MAC-VO || exit 1

GPU="${GPU:-0}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
RUNNER="run_pipeline_execution_benchmark_repro_component_ablation.py"
ROOT="${ROOT:-outputs/SH003_0_200_component_ablation_3seed}"

run_case() {
  local seed="$1"
  local mode="$2"
  local label="$3"
  local threshold="$4"
  local replay="$5"
  local maintenance_mode="$6"

  local work_dir="$ROOT/seed${seed}/${mode}"
  local output_name="incremental_component_${mode}_seed${seed}"
  local summary="$work_dir/execution_benchmark_summary.json"

  if [[ -f "$summary" ]]; then
    echo "[skip complete] seed=$seed | $label | $summary"
    return 0
  fi

  mkdir -p "$work_dir"

  cat > "$work_dir/protocol.txt" <<EOF
sequence=SH003[0,200)
split=strict8to2(split_every=5,split_offset=4)
component_mode=$mode
label=$label
W=20
B=100
M=50
maintenance_mode=$maintenance_mode
maintenance_min_opacity=$threshold
historical_replay_fraction=$replay
reset_new_packet_opacity=true
new_packet_reset_max_opacity=0.01
seed=$seed
EOF

  echo
  echo "================================================================================"
  echo "Component ablation run | seed=$seed | $label"
  echo "SH003 [0,200) | strict 8:2 | W20/B100/M50 | Th=$threshold | replay=$replay"
  echo "work_dir=$work_dir"
  echo "================================================================================"

  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED="$seed" \
  PIPELINE_BENCHMARK_SEED="$seed" \
  PIPELINE_COMPONENT_ABLATION_MODE="$mode" \
  PIPELINE_HISTORICAL_REPLAY_FRACTION="$replay" \
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
    --set backend.maintenance_mode="$maintenance_mode" \
    --set backend.maintenance_after_local_iteration=50 \
    --set backend.maintenance_min_opacity="$threshold" \
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
    echo "[FAILED] seed=$seed mode=$mode exit=$status; continuing" >&2
  fi
}

for seed in 1 2; do
  run_case "$seed" ff_only      "FF insertion only"         0.005 0.00 off
  run_case "$seed" incremental  "Incremental optimization" 0.005 0.00 standard
  run_case "$seed" strong_prune "Inc. + strong pruning"     0.100 0.00 standard
  run_case "$seed" historical   "Inc. + historical views"  0.005 0.30 standard
  run_case "$seed" full         "Full"                      0.100 0.30 standard
done

echo
echo "================================================================================"
echo "Finished component-ablation seed1/seed2 pass."
echo "Completed cases were skipped; incomplete cases were rerun without deleting work dirs."
echo "Existing seed-0 results were not touched."
echo "Results: $ROOT/seed1 and $ROOT/seed2"
echo "================================================================================"
