#!/usr/bin/env bash
set -u

# Final large-scale experiment:
#   SE000-SE007 + SH000-SH007 (16 sequences)
#   - 3 quality runs per sequence (seeds 0/1/2)
#       * strict 8:2 online mapping, frozen W20/R30/B100/M50/Th=.10
#       * endpoint metrics only: Online and +10-pass Global Refinement
#       * PSNR / SSIM / LPIPS(VGG) + SE3 ATE
#   - 1 timing-only run per sequence
#       * no image metric rendering, no pose-metric evaluation
#       * online FPS/time from frame timing until last online backend update
#       * post-hoc 10-pass refinement wall time measured separately
#
# The script is restart-safe and failure-tolerant: completed runs are skipped;
# one failed run does not abort the remaining 63 jobs.

cd /home/shiyo/Desktop/MAC-VO || exit 1

PIPELINE_CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo"
GT_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt"
QUALITY_RUNNER="run_pipeline_execution_benchmark_repro_posthoc_global_refine_endpoint_lpips.py"
TIMING_RUNNER="run_pipeline_execution_benchmark_repro_posthoc_global_refine_timing_only.py"
OUTPUT_ROOT="outputs/final16_10pass_full"
GPU="${GPU:-0}"

SEQUENCES=(
  SE000 SE001 SE002 SE003 SE004 SE005 SE006 SE007
  SH000 SH001 SH002 SH003 SH004 SH005 SH006 SH007
)
SEEDS=(0 1 2)

mkdir -p "$OUTPUT_ROOT"

# Fail early before launching dozens of long jobs if LPIPS is unavailable.
python - <<'PY'
from lpips import LPIPS
print("[preflight] LPIPS import OK")
PY
if [[ $? -ne 0 ]]; then
  echo "[fatal] lpips import failed in the current environment" >&2
  exit 2
fi

count_frames() {
  local seq="$1"
  python - "$DATA_ROOT/$seq" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]) / "image_left"
exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
if not files:
    raise SystemExit(f"no images found in {root}")
print(len(files))
PY
}

common_args() {
  local seq="$1"
  local n="$2"
  local bookkeeping_iters="$3"
  local work_dir="$4"
  local output_name="$5"

  printf '%s\n' \
    --mode serial \
    --config "$PIPELINE_CONFIG" \
    --set "paths.data_config=Config/Sequence/TartanAirV1_Challenge_${seq}.yaml" \
    --set "evaluation.gt_pose_file=$GT_ROOT/${seq}.txt" \
    --set pose_frontend.source=macvo \
    --set sequence.start_index=0 \
    --set "sequence.end_index=$n" \
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
    --set "backend.optimization.iterations=$bookkeeping_iters" \
    --set backend.eval_before_optimization=false \
    --set backend.eval_every_train_packets=1000000 \
    --set backend.eval_max_views=0 \
    --set backend.save_every_train_packets=0 \
    --set backend.save_final_ply=false \
    --set backend.wandb_mode=disabled \
    --set backend.write_runtime_artifacts=true \
    --set "paths.work_dir=$work_dir" \
    --set "backend.output_name=$output_name"
}

run_quality() {
  local seq="$1"
  local seed="$2"
  local n="$3"
  local ntrain="$4"
  local bookkeeping_iters="$5"

  local work_dir="$OUTPUT_ROOT/$seq/quality_seed${seed}"
  local output_name="incremental_${seq}_quality_seed${seed}"
  local endpoint="$work_dir/$output_name/posthoc_global_refinement_endpoint_metrics.json"
  local summary="$work_dir/execution_benchmark_summary.json"
  mkdir -p "$work_dir"

  if [[ -f "$endpoint" && -f "$summary" ]]; then
    echo "[skip quality complete] $seq seed=$seed"
    return 0
  fi

  echo
  echo "================================================================"
  echo "QUALITY | $seq | seed=$seed | frames=$n | train=$ntrain"
  echo "Online W20/R30/B100/M50/Th=.10 -> reset -> 10 passes"
  echo "Metrics only at Online endpoint and Pass10 endpoint"
  echo "================================================================"

  mapfile -t args < <(common_args "$seq" "$n" "$bookkeeping_iters" "$work_dir" "$output_name")

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED="$seed" \
  PIPELINE_BENCHMARK_SEED="$seed" \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  PIPELINE_GLOBAL_REFINE_PASSES=10 \
  PIPELINE_GLOBAL_REFINE_SEED="$seed" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TORCH_COMPILE_DISABLE=1 \
  python "$QUALITY_RUNNER" \
    --with_pose_metrics \
    "${args[@]}" \
    2>&1 | tee "$work_dir/run.log"
  local status=${PIPESTATUS[0]}

  echo "$status" > "$work_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED quality] $seq seed=$seed exit=$status; continuing" >&2
  fi
  return 0
}

run_timing() {
  local seq="$1"
  local n="$2"
  local ntrain="$3"
  local bookkeeping_iters="$4"

  local work_dir="$OUTPUT_ROOT/$seq/timing"
  local output_name="incremental_${seq}_timing"
  local timing_json="$work_dir/$output_name/posthoc_global_refinement_timing.json"
  local summary="$work_dir/execution_benchmark_summary.json"
  mkdir -p "$work_dir"

  if [[ -f "$timing_json" && -f "$summary" ]]; then
    echo "[skip timing complete] $seq"
    return 0
  fi

  echo
  echo "================================================================"
  echo "TIMING ONLY | $seq | frames=$n | train=$ntrain"
  echo "No PSNR/SSIM/LPIPS/ATE evaluation in this run"
  echo "Online endpoint timing + reset + 10-pass refinement timing"
  echo "================================================================"

  mapfile -t args < <(common_args "$seq" "$n" "$bookkeeping_iters" "$work_dir" "$output_name")

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED=0 \
  PIPELINE_BENCHMARK_SEED=0 \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  PIPELINE_GLOBAL_REFINE_PASSES=10 \
  PIPELINE_GLOBAL_REFINE_SEED=0 \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TORCH_COMPILE_DISABLE=1 \
  python "$TIMING_RUNNER" \
    "${args[@]}" \
    --set backend.evaluation_enabled=false \
    2>&1 | tee "$work_dir/run.log"
  local status=${PIPESTATUS[0]}

  echo "$status" > "$work_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED timing] $seq exit=$status; continuing" >&2
  fi
  return 0
}

for seq in "${SEQUENCES[@]}"; do
  n=$(count_frames "$seq") || {
    echo "[FAILED dataset check] $seq; skipping sequence" >&2
    continue
  }
  ntest=$(( n / 5 ))
  ntrain=$(( n - ntest ))
  bookkeeping_iters=$(( ntrain * 100 ))

  for seed in "${SEEDS[@]}"; do
    run_quality "$seq" "$seed" "$n" "$ntrain" "$bookkeeping_iters"
  done
  run_timing "$seq" "$n" "$ntrain" "$bookkeeping_iters"
done

echo
echo "================================================================"
echo "All 16 sequences attempted. Building aggregate summary..."
echo "================================================================"
python summarize_final16_10pass_full_experiment.py || true

echo
echo "Done. Re-run this same script to resume any failed/missing runs."
