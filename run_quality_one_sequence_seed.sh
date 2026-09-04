#!/usr/bin/env bash
set -u

# Flexible single quality run for one TartanAir Stereo Challenge sequence and seed.
# Usage:
#   bash run_quality_one_sequence_seed.sh SE001 10
#   bash run_quality_one_sequence_seed.sh SH003 2
#
# This runs the frozen online protocol and endpoint-only 10-pass global refinement:
#   strict 8:2 | W20/R30/B100/M50/Th=.10
#   Online endpoint PSNR/SSIM/LPIPS + SE3 ATE
#   opacity reset -> continue online optimizer -> 10 full train passes
#   refined endpoint PSNR/SSIM/LPIPS
# Existing complete output for the same sequence+seed is skipped unless FORCE=1.

cd /home/shiyo/Desktop/MAC-VO || exit 1

SEQ="${1:-}"
SEED="${2:-}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"

if [[ -z "$SEQ" || -z "$SEED" ]]; then
  echo "Usage: bash $0 <SEQ> <SEED>" >&2
  echo "Example: bash $0 SE001 10" >&2
  exit 2
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "[fatal] seed must be a non-negative integer: $SEED" >&2
  exit 2
fi

PIPELINE_CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo"
GT_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt"
DATA_CONFIG="Config/Sequence/TartanAirV1_Challenge_${SEQ}.yaml"
QUALITY_RUNNER="run_pipeline_execution_benchmark_repro_posthoc_global_refine_endpoint_lpips.py"
OUTPUT_ROOT="outputs/final16_10pass_full"

if [[ ! -f "$DATA_CONFIG" ]]; then
  echo "[fatal] data config not found: $DATA_CONFIG" >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT/$SEQ/image_left" ]]; then
  echo "[fatal] dataset not found: $DATA_ROOT/$SEQ/image_left" >&2
  exit 2
fi
if [[ ! -f "$GT_ROOT/$SEQ.txt" ]]; then
  echo "[fatal] GT pose not found: $GT_ROOT/$SEQ.txt" >&2
  exit 2
fi

n=$(python - "$DATA_ROOT/$SEQ/image_left" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
if not files:
    raise SystemExit(f"no images found in {root}")
print(len(files))
PY
) || exit 2

ntest=$(( n / 5 ))
ntrain=$(( n - ntest ))
bookkeeping_iters=$(( ntrain * 100 ))

work_dir="$OUTPUT_ROOT/$SEQ/quality_seed${SEED}"
output_name="incremental_${SEQ}_quality_seed${SEED}"
endpoint="$work_dir/$output_name/posthoc_global_refinement_endpoint_metrics.json"
summary="$work_dir/execution_benchmark_summary.json"
mkdir -p "$work_dir"

if [[ "$FORCE" != "1" && -f "$endpoint" && -f "$summary" ]]; then
  echo "[skip complete] $SEQ seed=$SEED"
  echo "  $endpoint"
  exit 0
fi

python - <<'PY'
from lpips import LPIPS
print("[preflight] LPIPS import OK")
PY
if [[ $? -ne 0 ]]; then
  echo "[fatal] lpips import failed in current environment" >&2
  exit 2
fi

echo "================================================================"
echo "QUALITY | $SEQ | seed=$SEED | frames=$n | train=$ntrain | test=$ntest"
echo "Online: strict8:2 W20/R30/B100/M50/Th=.10"
echo "Global refine: opacity reset + continue optimizer + 10 train passes"
echo "Output: $work_dir"
echo "================================================================"

set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONHASHSEED="$SEED" \
PIPELINE_BENCHMARK_SEED="$SEED" \
PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
PIPELINE_GLOBAL_REFINE_PASSES=10 \
PIPELINE_GLOBAL_REFINE_SEED="$SEED" \
CUDA_VISIBLE_DEVICES="$GPU" \
TORCH_COMPILE_DISABLE=1 \
python "$QUALITY_RUNNER" \
  --mode serial \
  --with_pose_metrics \
  --config "$PIPELINE_CONFIG" \
  --set paths.data_config="$DATA_CONFIG" \
  --set evaluation.gt_pose_file="$GT_ROOT/$SEQ.txt" \
  --set pose_frontend.source=macvo \
  --set sequence.start_index=0 \
  --set sequence.end_index="$n" \
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
  --set backend.optimization.iterations="$bookkeeping_iters" \
  --set backend.eval_before_optimization=false \
  --set backend.eval_every_train_packets=1000000 \
  --set backend.evaluation_enabled=true \
  --set backend.eval_max_views=0 \
  --set backend.save_every_train_packets=0 \
  --set backend.save_final_ply=false \
  --set backend.wandb_mode=disabled \
  --set backend.write_runtime_artifacts=true \
  --set paths.work_dir="$work_dir" \
  --set backend.output_name="$output_name" \
  2>&1 | tee "$work_dir/run.log"
status=${PIPESTATUS[0]}
set -e

echo "$status" > "$work_dir/exit_status.txt"
if [[ "$status" -ne 0 ]]; then
  echo "[FAILED] $SEQ seed=$SEED exit=$status" >&2
  exit "$status"
fi

echo "[DONE] $SEQ seed=$SEED"
echo "Endpoint metrics: $endpoint"
echo "Run summary     : $summary"
