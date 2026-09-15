#!/usr/bin/env bash
set -euo pipefail

cd /home/shiyo/Desktop/MAC-VO

GPU="${GPU:-0}"
SEED="${SEED:-0}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV2_P001_Teaser.yaml"
RUNNER="run_pipeline_execution_benchmark_repro_p001_v2_diagnostic.py"
DATA_ROOT="${DATA_ROOT:-/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P001}"
PROBE_POSE="${PROBE_POSE:-/home/shiyo/Desktop/Resplat/assets/custom_poses/p001_render.json}"
WORK="${WORK:-outputs/p001_v2_diagnostic_0_20}"
NAME="${NAME:-incremental_P001_v2_diagnostic_0_20}"
END_INDEX=20

for p in "$CONFIG" "$RUNNER" \
         "$DATA_ROOT/pose_lcam_front.txt" "$DATA_ROOT/pose_rcam_front.txt" \
         "$PROBE_POSE"; do
  if [[ ! -f "$p" ]]; then
    echo "[fatal] missing: $p" >&2
    exit 2
  fi
done
if [[ ! -d "$DATA_ROOT/image_lcam_front" || ! -d "$DATA_ROOT/image_rcam_front" ]]; then
  echo "[fatal] missing P001 V2 stereo image directories under $DATA_ROOT" >&2
  exit 2
fi

OUT="$WORK/$NAME"
mkdir -p "$WORK" "$OUT"

cat <<EOF
======================================================================
P001 / TartanAir V2 diagnostic | first 20 frames = [0,20)
Camera contract under test:
  native image : 640x640
  K            : fx=320 fy=320 cx=320 cy=320
  stereo       : baseline=0.25 m
ReSplat:
  experiment   : tartanair_p000_ft (base checkpoint; no per-P001 fine-tune)
  input domain : 320x320
  refine_steps : 0
Online backend:
  W20 / rho=.30 / B100 / M50 / Th=.10 / new opacity<=.01
  no post-hoc/global refinement
Diagnostics saved under:
  $OUT/camera_audit.json
  $OUT/resplat_packet_renders/<frame>/
  $OUT/final_map_renders/<frame>/
  $OUT/point_cloud/iteration_*/point_cloud.ply
  $OUT/teaser_overview/render.png
======================================================================
EOF

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONHASHSEED="$SEED" \
PIPELINE_BENCHMARK_SEED="$SEED" \
PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
PIPELINE_P001_DIAG_OUTPUT="$OUT" \
PIPELINE_P001_DIAG_DATA_ROOT="$DATA_ROOT" \
PIPELINE_P001_DIAG_MAX_FRAMES="$END_INDEX" \
PIPELINE_P001_DIAG_BASELINE=0.25 \
PIPELINE_P001_DIAG_FX=320 \
PIPELINE_P001_DIAG_FY=320 \
PIPELINE_P001_DIAG_CX=320 \
PIPELINE_P001_DIAG_CY=320 \
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
  --set camera.stereo_baseline=0.25 \
  --set camera.fx=320.0 \
  --set camera.fy=320.0 \
  --set camera.cx=320.0 \
  --set camera.cy=320.0 \
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
  --set backend.optimization.iterations=2000 \
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
echo "[done] diagnostic root: $OUT"
find "$OUT" -maxdepth 3 -type f \
  \( -name 'camera_audit.json' -o -name 'summary.json' -o \
     -name 'manifest.json' -o -name 'point_cloud.ply' -o \
     -name 'render.png' \) -print | sort || true
echo
echo "Web pose tuner for this 20-frame map:"
echo "  CUDA_VISIBLE_DEVICES=$GPU python tune_p001_teaser_pose.py --ui web --run_dir $OUT"
echo "======================================================================"
