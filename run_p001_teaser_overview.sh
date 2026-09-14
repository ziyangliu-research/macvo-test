#!/usr/bin/env bash
set -euo pipefail

cd /home/shiyo/Desktop/MAC-VO

GPU="${GPU:-0}"
SEED="${SEED:-0}"
FULL="${FULL:-0}"
SAVE_FINAL_PLY="${SAVE_FINAL_PLY:-1}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV2_P001_Teaser.yaml"
RUNNER="run_pipeline_execution_benchmark_repro_p001_teaser.py"
DATA_ROOT="${DATA_ROOT:-/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P001}"
PROBE_POSE="${PROBE_POSE:-/home/shiyo/Desktop/Resplat/assets/custom_poses/p001_render.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/p001_teaser_overview}"

for p in "$CONFIG" "$RUNNER" "$DATA_ROOT/pose_lcam_front.txt" "$PROBE_POSE"; do
  if [[ ! -f "$p" ]]; then
    echo "[fatal] missing: $p" >&2
    exit 2
  fi
done
if [[ ! -d "$DATA_ROOT/image_lcam_front" || ! -d "$DATA_ROOT/image_rcam_front" ]]; then
  echo "[fatal] missing P001 stereo image directories under $DATA_ROOT" >&2
  exit 2
fi

case "$SAVE_FINAL_PLY" in
  1|true|TRUE|yes|YES) SAVE_FINAL_PLY_VALUE=true ;;
  0|false|FALSE|no|NO) SAVE_FINAL_PLY_VALUE=false ;;
  *)
    echo "[fatal] SAVE_FINAL_PLY must be 0/1 or false/true, got: $SAVE_FINAL_PLY" >&2
    exit 2
    ;;
esac

if [[ "$FULL" == "1" ]]; then
  END_INDEX=$(python - "$DATA_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
left = sorted((root / "image_lcam_front").glob("*.png"))
right = sorted((root / "image_rcam_front").glob("*.png"))
if not left or len(left) != len(right):
    raise SystemExit(f"invalid P001 stereo counts left={len(left)} right={len(right)}")
print(len(left))
PY
  )
  TAG="full"
else
  # END_INDEX follows Python slicing semantics: [0, END_INDEX).
  # Default preserves the old ReSplat inclusive --packet_ranges 0-20 -> 21 frames.
  END_INDEX="${END_INDEX:-21}"
  TAG="0_$((END_INDEX - 1))"
fi

if (( END_INDEX <= 0 )); then
  echo "[fatal] invalid END_INDEX=$END_INDEX" >&2
  exit 2
fi

WORK="$OUTPUT_ROOT/$TAG"
NAME="incremental_P001_teaser_${TAG}"
BOOKKEEPING_ITERS=$(( END_INDEX * 100 ))
mkdir -p "$WORK"

echo "======================================================================"
echo "P001 teaser overview | tag=$TAG | frames=[0,$END_INDEX)"
echo "Visualization protocol: all selected frames map/insert"
echo "Online: W20 / rho=.30 / B100 / M50 / Th=.10 / new opacity<=.01"
echo "No post-hoc refinement; no quantitative final evaluation"
echo "Legacy fixed camera: $PROBE_POSE"
echo "Overview render: 960x540 | K_norm=(0.25,0.35,0.5,0.5)"
echo "Save final Gaussian PLY: $SAVE_FINAL_PLY_VALUE"
echo "Output: $WORK/$NAME/teaser_overview/render.png"
echo "======================================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONHASHSEED="$SEED" \
PIPELINE_BENCHMARK_SEED="$SEED" \
PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
PIPELINE_TEASER_PROBE_POSE_JSON="$PROBE_POSE" \
PIPELINE_TEASER_GT_POSE_FILE="$DATA_ROOT/pose_lcam_front.txt" \
PIPELINE_TEASER_RENDER_H=540 \
PIPELINE_TEASER_RENDER_W=960 \
PIPELINE_TEASER_FX_NORM=0.25 \
PIPELINE_TEASER_FY_NORM=0.35 \
PIPELINE_TEASER_CX_NORM=0.5 \
PIPELINE_TEASER_CY_NORM=0.5 \
PIPELINE_TEASER_NEAR=0.1 \
PIPELINE_TEASER_FAR=50.0 \
CUDA_VISIBLE_DEVICES="$GPU" \
TORCH_COMPILE_DISABLE=1 \
python "$RUNNER" \
  --mode serial \
  --config "$CONFIG" \
  --set paths.data_config=Config/Sequence/TartanAirV2_House_easy_P001.yaml \
  --set sequence.start_index=0 \
  --set sequence.end_index="$END_INDEX" \
  --set split.split_every=1000000 \
  --set split.split_offset=999999 \
  --set split.split_index_mode=local_index \
  --set backend.local_map_size=20 \
  --set backend.iterations_per_packet=100 \
  --set backend.reset_new_packet_opacity=true \
  --set backend.new_packet_reset_max_opacity=0.01 \
  --set backend.maintenance_mode=standard \
  --set backend.maintenance_after_local_iteration=50 \
  --set backend.maintenance_min_opacity=0.10 \
  --set backend.optimization.iterations="$BOOKKEEPING_ITERS" \
  --set backend.evaluation_enabled=false \
  --set backend.eval_before_optimization=false \
  --set backend.eval_every_train_packets=1000000 \
  --set backend.save_every_train_packets=0 \
  --set backend.save_final_ply="$SAVE_FINAL_PLY_VALUE" \
  --set backend.wandb_mode=disabled \
  --set backend.write_runtime_artifacts=true \
  --set paths.work_dir="$WORK" \
  --set backend.output_name="$NAME" \
  2>&1 | tee "$WORK/run.log"

RUN_DIR="$WORK/$NAME"
RENDER="$RUN_DIR/teaser_overview/render.png"
META="$RUN_DIR/teaser_overview/metadata.json"
PLY=""
if [[ "$SAVE_FINAL_PLY_VALUE" == "true" && -d "$RUN_DIR/point_cloud" ]]; then
  PLY=$(find "$RUN_DIR/point_cloud" -type f -name point_cloud.ply | sort | tail -n 1 || true)
fi

echo
echo "======================================================================"
if [[ -f "$RENDER" ]]; then
  echo "[done] teaser render: $RENDER"
  echo "[done] metadata     : $META"
else
  echo "[warning] run finished but teaser render is missing: $RENDER" >&2
  exit 3
fi
if [[ "$SAVE_FINAL_PLY_VALUE" == "true" ]]; then
  if [[ -n "$PLY" && -f "$PLY" ]]; then
    echo "[done] final Gaussians: $PLY"
  else
    echo "[warning] SAVE_FINAL_PLY=true but final point_cloud.ply was not found under $RUN_DIR/point_cloud" >&2
    exit 4
  fi
fi
echo "======================================================================"
