#!/usr/bin/env bash
set -u

# Full mannequin_face_1 benchmark, seed0.
# Online: all frames, strict 8:2, W20/rho=.30/B100/M50/Th=.10.
# Quality: online endpoint + ONE continuous post-hoc refinement trajectory with
#          PSNR/SSIM/LPIPS checkpoints at passes 10/15/20/25/30.
# Timing : separate no-metric run, with cumulative refinement wall time at the
#          same pass checkpoints.

cd /home/shiyo/Desktop/MAC-VO || exit 1

GPU="${GPU:-0}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
CHECKPOINTS="${CHECKPOINTS:-10,15,20,25,30}"

CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_ETH3D_mannequin_face_1_Full.yaml"
DATA_CONFIG="Config/Sequence/ETH3D_mannequin_face_1_rectified.yaml"
DATA_ROOT="/home/shiyo/Desktop/Datasets/ETH3D_rectified/mannequin_face_1"
OUTPUT_ROOT="outputs/eth3d_mannequin_full30"
QUALITY_RUNNER="run_pipeline_execution_benchmark_repro_eth3d_refine_checkpoints_lpips.py"
TIMING_RUNNER="run_pipeline_execution_benchmark_repro_eth3d_timing_checkpoints.py"

for required in "$CONFIG" "$DATA_CONFIG" "$QUALITY_RUNNER" "$TIMING_RUNNER"; do
  if [[ ! -f "$required" ]]; then
    echo "[fatal] missing: $required" >&2
    exit 2
  fi
done
for required in "$DATA_ROOT/image_left" "$DATA_ROOT/image_right"; do
  if [[ ! -d "$required" ]]; then
    echo "[fatal] missing directory: $required" >&2
    exit 2
  fi
done
for required in "$DATA_ROOT/calibration.json" "$DATA_ROOT/groundtruth_left.txt" "$DATA_ROOT/timestamps.txt"; do
  if [[ ! -f "$required" ]]; then
    echo "[fatal] missing rectified ETH3D artifact: $required" >&2
    exit 2
  fi
done

n=$(python - "$DATA_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
left = sorted((root / "image_left").glob("*.png"))
right = sorted((root / "image_right").glob("*.png"))
ts = [x.strip() for x in (root / "timestamps.txt").read_text().splitlines() if x.strip()]
if not left or len(left) != len(right) or len(left) != len(ts):
    raise SystemExit(f"inconsistent ETH3D counts: left={len(left)} right={len(right)} timestamps={len(ts)}")
if [p.name for p in left] != [p.name for p in right]:
    raise SystemExit("left/right filenames do not match")
print(len(left))
PY
) || exit 2

ntest=$(( n / 5 ))
ntrain=$(( n - ntest ))
bookkeeping_iters=$(( ntrain * 100 ))

python - <<'PY'
from lpips import LPIPS
print("[preflight] LPIPS import OK")
PY
if [[ $? -ne 0 ]]; then
  echo "[fatal] lpips import failed" >&2
  exit 2
fi

echo "======================================================================"
echo "ETH3D mannequin_face_1 FULL | seed=$SEED | frames=$n"
echo "strict8:2 -> train=$ntrain test=$ntest"
echo "online: W20/R30/B100/M50/Th=.10"
echo "global refinement checkpoints: $CHECKPOINTS (single continuous run)"
echo "======================================================================"

quality_dir="$OUTPUT_ROOT/quality_seed${SEED}"
quality_name="incremental_ETH3D_mannequin_quality_seed${SEED}"
quality_checkpoint="$quality_dir/$quality_name/posthoc_global_refinement_checkpoint_metrics.json"
quality_summary="$quality_dir/execution_benchmark_summary.json"
mkdir -p "$quality_dir"

if [[ "$FORCE" == "1" || ! -f "$quality_checkpoint" || ! -f "$quality_summary" ]]; then
  echo "[run] QUALITY"
  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED="$SEED" \
  PIPELINE_BENCHMARK_SEED="$SEED" \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  PIPELINE_GLOBAL_REFINE_CHECKPOINT_PASSES="$CHECKPOINTS" \
  PIPELINE_GLOBAL_REFINE_SEED="$SEED" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TORCH_COMPILE_DISABLE=1 \
  python "$QUALITY_RUNNER" \
    --mode serial \
    --with_pose_metrics \
    --config "$CONFIG" \
    --set paths.data_config="$DATA_CONFIG" \
    --set sequence.start_index=0 \
    --set sequence.end_index="$n" \
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
    --set paths.work_dir="$quality_dir" \
    --set backend.output_name="$quality_name" \
    2>&1 | tee "$quality_dir/run.log"
  status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$quality_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED] quality exit=$status" >&2
    exit "$status"
  fi
else
  echo "[skip complete] QUALITY: $quality_checkpoint"
fi

timing_dir="$OUTPUT_ROOT/timing_seed${SEED}"
timing_name="incremental_ETH3D_mannequin_timing_seed${SEED}"
timing_checkpoint="$timing_dir/$timing_name/posthoc_global_refinement_timing_checkpoints.json"
timing_summary="$timing_dir/execution_benchmark_summary.json"
mkdir -p "$timing_dir"

if [[ "$FORCE" == "1" || ! -f "$timing_checkpoint" || ! -f "$timing_summary" ]]; then
  echo "[run] TIMING"
  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED="$SEED" \
  PIPELINE_BENCHMARK_SEED="$SEED" \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  PIPELINE_GLOBAL_REFINE_CHECKPOINT_PASSES="$CHECKPOINTS" \
  PIPELINE_GLOBAL_REFINE_SEED="$SEED" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TORCH_COMPILE_DISABLE=1 \
  python "$TIMING_RUNNER" \
    --mode serial \
    --config "$CONFIG" \
    --set paths.data_config="$DATA_CONFIG" \
    --set sequence.start_index=0 \
    --set sequence.end_index="$n" \
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
    --set backend.evaluation_enabled=false \
    --set backend.eval_max_views=0 \
    --set backend.save_every_train_packets=0 \
    --set backend.save_final_ply=false \
    --set backend.wandb_mode=disabled \
    --set backend.write_runtime_artifacts=true \
    --set paths.work_dir="$timing_dir" \
    --set backend.output_name="$timing_name" \
    2>&1 | tee "$timing_dir/run.log"
  status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$timing_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED] timing exit=$status" >&2
    exit "$status"
  fi
else
  echo "[skip complete] TIMING: $timing_checkpoint"
fi

python summarize_eth3d_mannequin_full30.py || true
