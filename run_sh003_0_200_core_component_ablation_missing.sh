#!/usr/bin/env bash
set -u

# Complete the three missing SH003 core-component ablation runs.
# Common protocol: SH003 [0,200), strict 8:2, serial, final evaluation only.
#
# Existing/reused cells:
#   Aggressive pruning only : W20 / rho0  / B100 / Th=.10
#   Full method             : W20 / rho30 / B100 / Th=.10
#
# New runs here:
#   A. Feed-forward aggregation only (no optimization/maintenance/replay)
#   B. Incremental optimization + standard pruning: W20 / rho0 / B100 / Th=.005
#   C. Incremental optimization + replay + standard pruning: W20 / rho30 / B100 / Th=.005

cd /home/shiyo/Desktop/MAC-VO || exit 1

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt"

COMMON_ARGS=(
  --mode serial
  --with_pose_metrics
  --config "$CONFIG"
  --set paths.data_config="$DATA_CONFIG"
  --set evaluation.gt_pose_file="$GT"
  --set pose_frontend.source=macvo
  --set sequence.start_index=0
  --set sequence.end_index=200
  --set 'resplat_frontend.overrides=["dataset.image_shape=[240,320]"]'
  --set split.split_every=5
  --set split.split_offset=4
  --set split.split_index_mode=local_index
  --set backend.local_map_size=20
  --set backend.eval_before_optimization=false
  --set backend.eval_every_train_packets=1000000
  --set backend.evaluation_enabled=true
  --set backend.save_every_train_packets=0
  --set backend.save_final_ply=false
  --set backend.wandb_mode=disabled
  --set backend.write_runtime_artifacts=true
)

run_ff_only() {
  local work_dir="outputs/SH003_0_200_core_feedforward_aggregation_only"
  local output_name="incremental_SH003_0_200_core_feedforward_aggregation_only"
  if [[ -f "$work_dir/execution_benchmark_summary.json" ]]; then
    echo "[skip complete] feed-forward aggregation only"
    return 0
  fi
  mkdir -p "$work_dir"
  echo
  echo "================================================================"
  echo "A | Feed-forward aggregation only"
  echo "MAC-VO + ReSplat + world transform + append; no optimization/maintenance/replay"
  echo "================================================================"
  set +e
  env -u PIPELINE_HISTORICAL_REPLAY_FRACTION \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONHASHSEED=0 PIPELINE_BENCHMARK_SEED=0 \
    CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
    python run_pipeline_execution_benchmark_feedforward_aggregation_only.py \
      "${COMMON_ARGS[@]}" \
      --set backend.iterations_per_packet=100 \
      --set backend.reset_new_packet_opacity=false \
      --set backend.maintenance_mode=off \
      --set backend.optimization.iterations=16000 \
      --set paths.work_dir="$work_dir" \
      --set backend.output_name="$output_name" \
      2>&1 | tee "$work_dir/run.log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$work_dir/exit_status.txt"
  [[ "$status" -eq 0 ]] || echo "[FAILED] feed-forward aggregation only, exit=$status" >&2
}

run_standard() {
  local replay="$1"
  local tag
  local fraction
  if [[ "$replay" == "0" ]]; then
    tag="R0"
    fraction="0"
  else
    tag="R30"
    fraction="0.30"
  fi
  local work_dir="outputs/SH003_0_200_core_W20_${tag}_B100_th0005"
  local output_name="incremental_SH003_0_200_core_W20_${tag}_B100_th0005"
  if [[ -f "$work_dir/execution_benchmark_summary.json" ]]; then
    echo "[skip complete] W20 ${tag} B100 Th=.005"
    return 0
  fi
  mkdir -p "$work_dir"
  echo
  echo "================================================================"
  echo "W20 | replay=${replay}% | B100 | M50 | standard pruning Th=.005"
  echo "================================================================"
  set +e
  if [[ "$replay" == "0" ]]; then
    env -u PIPELINE_HISTORICAL_REPLAY_FRACTION \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONHASHSEED=0 PIPELINE_BENCHMARK_SEED=0 \
      CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
      python run_pipeline_execution_benchmark_repro_final_eval_only.py \
        "${COMMON_ARGS[@]}" \
        --set backend.iterations_per_packet=100 \
        --set backend.reset_new_packet_opacity=true \
        --set backend.new_packet_reset_max_opacity=0.01 \
        --set backend.maintenance_mode=standard \
        --set backend.maintenance_after_local_iteration=50 \
        --set backend.maintenance_min_opacity=0.005 \
        --set backend.optimization.iterations=16000 \
        --set paths.work_dir="$work_dir" \
        --set backend.output_name="$output_name" \
        2>&1 | tee "$work_dir/run.log"
  else
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONHASHSEED=0 PIPELINE_BENCHMARK_SEED=0 \
      PIPELINE_HISTORICAL_REPLAY_FRACTION="$fraction" \
      CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
      python run_pipeline_execution_benchmark_repro_final_eval_only.py \
        "${COMMON_ARGS[@]}" \
        --set backend.iterations_per_packet=100 \
        --set backend.reset_new_packet_opacity=true \
        --set backend.new_packet_reset_max_opacity=0.01 \
        --set backend.maintenance_mode=standard \
        --set backend.maintenance_after_local_iteration=50 \
        --set backend.maintenance_min_opacity=0.005 \
        --set backend.optimization.iterations=16000 \
        --set paths.work_dir="$work_dir" \
        --set backend.output_name="$output_name" \
        2>&1 | tee "$work_dir/run.log"
  fi
  local status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$work_dir/exit_status.txt"
  [[ "$status" -eq 0 ]] || echo "[FAILED] W20 ${tag} B100 Th=.005, exit=$status" >&2
}

run_ff_only
run_standard 0
run_standard 30

echo
echo "Core-component missing runs attempted. Summarize with:"
echo "  python summarize_sh003_core_component_ablation.py"
