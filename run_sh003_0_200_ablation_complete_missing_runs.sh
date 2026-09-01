#!/usr/bin/env bash
set -u

# One-click missing ablation experiments for the paper/professor report.
#
# Protocol shared by all NEW runs:
#   Dataset: SH003 [0,200), strict 8:2 held-out
#   Mapping/train views: 160; held-out test views: 40
#   B = 100 optimization steps / mapping packet
#   M = 50 maintenance iteration
#   Historical replay disabled (rho = 0)
#   New-packet opacity cap = 0.01
#
# Part A: no-replay opacity-threshold sweep using the original W=10 baseline
#   Th = 0.005, 0.01, 0.02, 0.03, 0.05, 0.10       (6 runs)
#
# Part B: complete rho=0 recent-window column at Th=0.10
#   W = 5 and W = 20                                 (2 runs)
#   W = 10 / rho=0 / Th=.10 is reused from Part A.
#
# Total NEW runs: 8.
# Existing replay/window/budget experiments are NOT rerun.  At the end this
# script calls summarize_sh003_ablation_paper_tables.py, which combines the new
# results with the already-completed W x replay, replay-boundary, and B sweeps.

cd /home/shiyo/Desktop/MAC-VO || exit 1

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_SH003.yaml"
GT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt/SH003.txt"
RUNNER="run_pipeline_execution_benchmark_repro_no_replay_empty_safe.py"

run_case() {
  local window="$1"
  local th="$2"
  local tag="$3"

  local name="SH003_0_200_ablation_rho0_W${window}_th${tag}"
  local work_dir="outputs/${name}"
  local output_name="incremental_${name}"
  local summary="${work_dir}/execution_benchmark_summary.json"

  if [[ -f "$summary" ]]; then
    echo "[skip complete] W=${window} rho=0 Th=${th} -> ${summary}"
    return 0
  fi

  mkdir -p "$work_dir"

  echo
  echo "================================================================"
  echo "SH003 [0,200) | strict 8:2 | W=${window} | rho=0 | B=100 | M=50 | Th=${th}"
  echo "160 mapping views / 40 held-out test views"
  echo "work_dir=${work_dir}"
  echo "================================================================"

  set +e
  env -u PIPELINE_HISTORICAL_REPLAY_FRACTION \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONHASHSEED=0 \
    PIPELINE_BENCHMARK_SEED=0 \
    CUDA_VISIBLE_DEVICES=0 \
    TORCH_COMPILE_DISABLE=1 \
    python "$RUNNER" \
      --mode serial \
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
      --set backend.local_map_size="$window" \
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
      --set backend.output_name="$output_name" \
      2>&1 | tee "$work_dir/run.log"
  local status=${PIPESTATUS[0]}
  set -e

  echo "$status" > "$work_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED] W=${window} rho=0 Th=${th}, exit=${status}; continuing" >&2
  fi
}

# ---------------------------------------------------------------------------
# Part A: opacity threshold without replay, original recent-only W=10 baseline.
# ---------------------------------------------------------------------------
run_case 10 0.005 0005
run_case 10 0.01  001
run_case 10 0.02  002
run_case 10 0.03  003
run_case 10 0.05  005
run_case 10 0.10  010

# ---------------------------------------------------------------------------
# Part B: add rho=0 cells needed for W={5,10,20} comparison at Th=.10.
# W10/rho0/Th=.10 above is reused automatically.
# ---------------------------------------------------------------------------
run_case 5  0.10 010
run_case 20 0.10 010

echo
echo "================================================================"
echo "All missing ablation runs attempted. Generating paper-ready tables..."
echo "================================================================"
python summarize_sh003_ablation_paper_tables.py
