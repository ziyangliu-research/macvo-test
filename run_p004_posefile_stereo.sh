#!/usr/bin/env bash
set -euo pipefail

cd /home/shiyo/Desktop/MAC-VO

GPU="${GPU:-0}"
SEED="${SEED:-0}"
DATA_ROOT="${DATA_ROOT:-/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P004}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV2_P004_Teaser.yaml"
RUNNER="run_pipeline_execution_benchmark_repro_p004_posefile_stereo.py"
FULL="${FULL:-0}"
END_INDEX="${END_INDEX:-20}"

for p in "$CONFIG" "$RUNNER" "$DATA_ROOT/pose_lcam_front.txt" "$DATA_ROOT/pose_rcam_front.txt"; do
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
if [[ "$LEFT_COUNT" -ne "$RIGHT_COUNT" ]]; then
  echo "[fatal] stereo frame-count mismatch: left=$LEFT_COUNT right=$RIGHT_COUNT" >&2
  exit 2
fi
if [[ "$FULL" == "1" ]]; then
  END_INDEX="$LEFT_COUNT"
fi
if [[ "$END_INDEX" -le 0 || "$END_INDEX" -gt "$LEFT_COUNT" ]]; then
  echo "[fatal] END_INDEX=$END_INDEX outside [1,$LEFT_COUNT]" >&2
  exit 2
fi

TAG="0_$((END_INDEX - 1))"
if [[ "$END_INDEX" -eq "$LEFT_COUNT" ]]; then
  TAG="full"
fi
WORK="${WORK:-outputs/p004_posefile_stereo_${TAG}}"
NAME="${NAME:-incremental_P004_posefile_stereo_${TAG}}"
OUT="$WORK/$NAME"
TOTAL_ITERS=$((END_INDEX * 100))
mkdir -p "$WORK" "$OUT"

cat <<EOF
======================================================================
TartanAir V2 House/P004 | exact pose-file stereo ReSplat test
Frames       : [0,$END_INDEX) = $END_INDEX frames
Stereo input : exact per-frame T_left_from_right = inv(Twc_left) @ Twc_right
               from pose_lcam_front.txt + pose_rcam_front.txt
ReSplat      : 320x320, refine_steps=0, serial default CUDA stream
Online map   : W20 / rho=.30 / B100 / M50 / Th=.10
Global refine: NONE

Artifacts:
  $OUT/stereo_pose_audit.json
  $OUT/resplat_packet_renders/<frame>/
  $OUT/final_map_input_renders/<frame>/
  $OUT/point_cloud/iteration_*/point_cloud.ply
  $OUT/teaser_overview/render.png
  $OUT/teaser_overview/probe_pose_current_relative.json
======================================================================
EOF

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONHASHSEED="$SEED" \
PIPELINE_BENCHMARK_SEED="$SEED" \
PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
PIPELINE_P004_DATA_ROOT="$DATA_ROOT" \
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
  --set sequence.start_index=0 \
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
echo "[done] $OUT"
echo "audit: $OUT/stereo_pose_audit.json"
echo "packet summary: $OUT/resplat_packet_renders/summary.json"
echo "Web pose tuner:"
echo "  CUDA_VISIBLE_DEVICES=$GPU python tune_p001_teaser_pose.py --ui web --run_dir $OUT"
echo "======================================================================"
