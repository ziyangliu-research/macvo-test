#!/usr/bin/env bash
set -euo pipefail

cd /home/shiyo/Desktop/MAC-VO

GPU="${GPU:-0}"
SEED="${SEED:-0}"
DATA_ROOT="${DATA_ROOT:-/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P004}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV2_P004_Teaser.yaml"
RUNNER="run_pipeline_execution_benchmark_repro_p004_teaser.py"
WORK="${WORK:-outputs/p004_teaser_0_50}"
NAME="${NAME:-incremental_P004_teaser_0_50}"
START_INDEX=0
END_INDEX=50
TOTAL_ITERS=$(((END_INDEX - START_INDEX) * 100))
OUT="$WORK/$NAME"

for p in "$CONFIG" "$RUNNER"; do
  if [[ ! -f "$p" ]]; then
    echo "[fatal] missing: $p" >&2
    exit 2
  fi
done
for d in "$DATA_ROOT/image_lcam_front" "$DATA_ROOT/image_rcam_front"; do
  if [[ ! -d "$d" ]]; then
    echo "[fatal] missing directory: $d" >&2
    exit 2
  fi
done

LEFT_COUNT=$(find "$DATA_ROOT/image_lcam_front" -maxdepth 1 -type f -name '*.png' | wc -l)
RIGHT_COUNT=$(find "$DATA_ROOT/image_rcam_front" -maxdepth 1 -type f -name '*.png' | wc -l)
if [[ "$LEFT_COUNT" -lt "$END_INDEX" || "$RIGHT_COUNT" -lt "$END_INDEX" ]]; then
  echo "[fatal] P004 does not contain frames [0,$END_INDEX): left=$LEFT_COUNT right=$RIGHT_COUNT" >&2
  exit 2
fi

mkdir -p "$WORK" "$OUT"

cat <<EOF
======================================================================
TartanAir V2 House/P004 | qualitative teaser | first 50 frames
Frames       : [0,50) = 50 frames
Mapping      : all 50 frames (no held-out split in this qualitative run)
Camera       : 640x640, fx=fy=cx=cy=320, stereo baseline=0.25 m
ReSplat      : 320x320, refine_steps=0, default CUDA stream in serial repro
Online map   : W20 / rho=.30 / B100 / M50 / Th=.10
Global refine: NONE

Artifacts:
  $OUT/resplat_packet_renders/<frame>/
      gt_left.png / gt_right.png
      render_left.png / render_right.png
      metrics.json
  $OUT/final_map_input_renders/<frame>/
      gt.png / render.png / metrics.json
  $OUT/point_cloud/iteration_*/point_cloud.ply
  $OUT/teaser_overview/render.png
  $OUT/teaser_overview/probe_pose_current_relative.json

Initial teaser camera:
  pose        : exact first mapping/input-camera pose
  render size : 960x540
  normalized K: fx=.25 fy=.35 cx=.5 cy=.5
======================================================================
EOF

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONHASHSEED="$SEED" \
PIPELINE_BENCHMARK_SEED="$SEED" \
PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
PIPELINE_TEASER_ARTIFACT_ROOT="$OUT" \
PIPELINE_TEASER_RENDER_H=540 \
PIPELINE_TEASER_RENDER_W=960 \
PIPELINE_TEASER_FX_NORM=0.25 \
PIPELINE_TEASER_FY_NORM=0.35 \
PIPELINE_TEASER_NEAR=0.1 \
PIPELINE_TEASER_FAR=50.0 \
CUDA_VISIBLE_DEVICES="$GPU" \
TORCH_COMPILE_DISABLE=1 \
python "$RUNNER" \
  --mode serial \
  --config "$CONFIG" \
  --set sequence.start_index="$START_INDEX" \
  --set sequence.end_index="$END_INDEX" \
  --set split.split_every=1000000 \
  --set split.split_offset=999999 \
  --set split.split_index_mode=local_index \
  --set resplat_frontend.refine_steps=0 \
  --set backend.local_map_size=20 \
  --set backend.iterations_per_packet=100 \
  --set backend.reset_new_packet_opacity=true \
  --set backend.new_packet_reset_max_opacity=0.01 \
  --set backend.maintenance_mode=standard \
  --set backend.maintenance_after_local_iteration=50 \
  --set backend.maintenance_min_opacity=0.10 \
  --set backend.optimization.iterations="$TOTAL_ITERS" \
  --set backend.evaluation_enabled=false \
  --set backend.eval_before_optimization=false \
  --set backend.eval_every_train_packets=1000000 \
  --set backend.save_every_train_packets=0 \
  --set backend.save_final_ply=true \
  --set backend.wandb_mode=disabled \
  --set backend.write_runtime_artifacts=true \
  --set paths.work_dir="$WORK" \
  --set backend.output_name="$NAME" \
  2>&1 | tee "$WORK/run.log"

echo
echo "======================================================================"
echo "[done] P004 first-50 teaser run: $OUT"
echo "Web pose tuner (VS Code Remote SSH):"
echo "  CUDA_VISIBLE_DEVICES=$GPU python tune_p001_teaser_pose.py --ui web --run_dir $OUT"
echo
find "$OUT" -maxdepth 4 -type f \
  \( -name 'summary.json' -o -name 'manifest.json' -o -name 'point_cloud.ply' -o \
     -name 'render.png' -o -name 'probe_pose_current_relative.json' \) \
  -print | sort | tail -40 || true
echo "======================================================================"
