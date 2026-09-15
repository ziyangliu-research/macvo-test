#!/usr/bin/env bash
set -euo pipefail

cd /home/shiyo/Desktop/MAC-VO

GPU="${GPU:-0}"
SEED="${SEED:-0}"
DATA_ROOT="${DATA_ROOT:-/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P004}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV2_P004_Teaser.yaml"
RUNNER="run_pipeline_execution_benchmark_repro_p004_teaser_gr10k.py"
WORK="${WORK:-outputs/p004_teaser_full_gr10k}"
NAME="${NAME:-incremental_P004_teaser_full_gr10k}"

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
if [[ "$LEFT_COUNT" -le 0 || "$RIGHT_COUNT" -le 0 ]]; then
  echo "[fatal] no P004 stereo PNG frames found" >&2
  exit 2
fi
if [[ "$LEFT_COUNT" -ne "$RIGHT_COUNT" ]]; then
  echo "[fatal] P004 stereo frame-count mismatch: left=$LEFT_COUNT right=$RIGHT_COUNT" >&2
  exit 2
fi

END_INDEX="$LEFT_COUNT"
TOTAL_ONLINE_ITERS=$((END_INDEX * 100))
OUT="$WORK/$NAME"
mkdir -p "$WORK" "$OUT"

cat <<EOF
======================================================================
TartanAir V2 House/P004 | FULL teaser + GR10k
Frames        : [0,$END_INDEX) = $END_INDEX frames
Mapping       : all frames
ReSplat       : 320x320, refine_steps=0
Online        : W20 / rho=.30 / B100 / M50 / Th=.10
Post-hoc GR   : native GraphDECO schedule, 10,000 optimizer updates
  extra opacity reset @ GR iter 0
  densify/prune after 500, every 100, while iter < 15000
  opacity reset every 3000 inside that window
  fixed camera poses; all mapping cameras shuffled without replacement

Artifacts:
  ONLINE PLY / renders are saved first.
  Refined PLY is then saved at the larger point_cloud iteration.
  $OUT/teaser_overview_refined_10k/render.png

Web UI automatically loads the latest (= refined) PLY:
  CUDA_VISIBLE_DEVICES=$GPU python tune_p001_teaser_pose.py --ui web --run_dir $OUT
======================================================================
EOF

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONHASHSEED="$SEED" \
PIPELINE_BENCHMARK_SEED="$SEED" \
PIPELINE_GLOBAL_REFINE_SEED="$SEED" \
PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
PIPELINE_GLOBAL_REFINE_ITER_CHECKPOINTS=10000 \
PIPELINE_GLOBAL_REFINE_TOTAL_ITERS=10000 \
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
  --set backend.optimization.iterations="$TOTAL_ONLINE_ITERS" \
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
echo "[done] P004 full + GR10k: $OUT"
echo
find "$OUT" -maxdepth 4 -type f \
  \( -name 'point_cloud.ply' -o -name 'render.png' -o -name 'metadata.json' -o \
     -name 'posthoc_global_refinement_native3dgs_metrics.json' \) \
  -print | sort | tail -50 || true

echo
echo "Web pose tuner (loads latest/refined PLY):"
echo "  CUDA_VISIBLE_DEVICES=$GPU python tune_p001_teaser_pose.py --ui web --run_dir $OUT"
echo "======================================================================"
