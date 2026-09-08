#!/usr/bin/env bash
set -u

# Four full ETH3D RGB-stereo sequences.
# Online protocol: strict8:2, W20/rho=.30/B100/M50/Th=.10, serial/default stream.
# Post-hoc quality: ONE continuous trajectory with BOTH:
#   - full-pass checkpoints, default 50..150 every 10 passes;
#   - exact update checkpoints, default 20k/30k/40k/50k optimizer updates.
#
# This exploratory sweep intentionally does NOT duplicate the entire run for a
# separate timing-only refinement trajectory. Online timing remains available
# from frame_timing_log.json. Once a final fixed refinement budget is selected,
# run a dedicated timing-only benchmark only at that chosen budget.

cd /home/shiyo/Desktop/MAC-VO || exit 1

GPU="${GPU:-0}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
PASS_CHECKPOINTS="${PASS_CHECKPOINTS:-50,60,70,80,90,100,110,120,130,140,150}"
ITER_CHECKPOINTS="${ITER_CHECKPOINTS:-20000,30000,40000,50000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eth3d4_refine_sweep}"
RAW_ROOT="/home/shiyo/Desktop/Datasets/ETH3D/sequences"
RECT_ROOT="/home/shiyo/Desktop/Datasets/ETH3D_rectified"
BASE_CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_ETH3D_mannequin_face_1_Full.yaml"
QUALITY_RUNNER="run_pipeline_execution_benchmark_repro_eth3d_refine_dual_checkpoints_lpips.py"

SEQS=(mannequin_face_1 einstein_1 sofa_3 plant_scene_3)

if [[ ! -f "$BASE_CONFIG" || ! -f "$QUALITY_RUNNER" || ! -f rectify_eth3d_stereo.py ]]; then
  echo "[fatal] required benchmark files are missing; git pull first" >&2
  exit 2
fi

python - <<'PY'
from lpips import LPIPS
print('[preflight] LPIPS import OK')
PY
if [[ $? -ne 0 ]]; then
  echo "[fatal] lpips import failed" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$RECT_ROOT"

echo "======================================================================"
echo "ETH3D4 full refinement sweep | seed=$SEED | GPU=$GPU"
echo "Pass checkpoints : ${PASS_CHECKPOINTS:-<disabled>}"
echo "Iter checkpoints : ${ITER_CHECKPOINTS:-<disabled>}"
echo "Output root      : $OUTPUT_ROOT"
echo "======================================================================"

for seq in "${SEQS[@]}"; do
  raw="$RAW_ROOT/$seq/$seq"
  rect="$RECT_ROOT/$seq"
  data_cfg="Config/Sequence/ETH3D_${seq}_rectified.yaml"

  if [[ ! -d "$raw/rgb" || ! -d "$raw/rgb2" ]]; then
    echo "[fatal] raw ETH3D sequence missing: $raw" >&2
    exit 2
  fi
  if [[ ! -f "$data_cfg" ]]; then
    echo "[fatal] data config missing: $data_cfg" >&2
    exit 2
  fi

  # Rectify only when the complete processed dataset is not already present.
  if [[ ! -d "$rect/image_left" || ! -d "$rect/image_right" || \
        ! -f "$rect/calibration.json" || ! -f "$rect/groundtruth_left.txt" || \
        ! -f "$rect/timestamps.txt" ]]; then
    echo "[rectify] $seq"
    rm -rf "$rect"
    python rectify_eth3d_stereo.py \
      --input "$raw" \
      --output "$rect" \
      --alpha 0 || exit $?
  else
    echo "[rectified exists] $seq"
  fi

  n=$(python - "$rect" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
left = sorted((root / 'image_left').glob('*.png'))
right = sorted((root / 'image_right').glob('*.png'))
ts = [x.strip() for x in (root / 'timestamps.txt').read_text().splitlines() if x.strip()]
if not left or len(left) != len(right) or len(left) != len(ts):
    raise SystemExit(f'inconsistent rectified counts: left={len(left)} right={len(right)} ts={len(ts)}')
if [p.name for p in left] != [p.name for p in right]:
    raise SystemExit('left/right filenames differ')
print(len(left))
PY
  ) || exit 2

  ntest=$(( n / 5 ))
  ntrain=$(( n - ntest ))
  bookkeeping_iters=$(( ntrain * 100 ))
  max150=$(( ntrain * 150 ))

  work="$OUTPUT_ROOT/$seq/quality_seed${SEED}"
  name="incremental_ETH3D_${seq}_quality_seed${SEED}"
  metrics="$work/$name/posthoc_global_refinement_dual_checkpoint_metrics.json"
  summary="$work/execution_benchmark_summary.json"
  mkdir -p "$work"

  complete=0
  if [[ "$FORCE" != "1" && -f "$metrics" && -f "$summary" ]]; then
    python - "$metrics" "$PASS_CHECKPOINTS" "$ITER_CHECKPOINTS" <<'PY'
import json, sys
path, pass_raw, iter_raw = sys.argv[1:]
d = json.load(open(path))
want_p = {int(x) for x in pass_raw.split(',') if x.strip()}
want_i = {int(x) for x in iter_raw.split(',') if x.strip()}
have_p = {int(x) for x in (d.get('checkpoints_by_pass') or {}).keys()}
have_i = {int(x) for x in (d.get('checkpoints_by_iteration') or {}).keys()}
raise SystemExit(0 if want_p <= have_p and want_i <= have_i else 1)
PY
    [[ $? -eq 0 ]] && complete=1
  fi

  echo
  echo "----------------------------------------------------------------------"
  echo "$seq | frames=$n train=$ntrain test=$ntest"
  echo "online updates=$bookkeeping_iters | 150-pass refinement updates=$max150"
  echo "----------------------------------------------------------------------"

  if [[ "$complete" == "1" ]]; then
    echo "[skip complete] $seq"
    continue
  fi

  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED="$SEED" \
  PIPELINE_BENCHMARK_SEED="$SEED" \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  PIPELINE_GLOBAL_REFINE_CHECKPOINT_PASSES="$PASS_CHECKPOINTS" \
  PIPELINE_GLOBAL_REFINE_CHECKPOINT_ITERS="$ITER_CHECKPOINTS" \
  PIPELINE_GLOBAL_REFINE_SEED="$SEED" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TORCH_COMPILE_DISABLE=1 \
  python "$QUALITY_RUNNER" \
    --mode serial \
    --with_pose_metrics \
    --config "$BASE_CONFIG" \
    --set paths.data_config="$data_cfg" \
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
    --set paths.work_dir="$work" \
    --set backend.output_name="$name" \
    2>&1 | tee "$work/run.log"
  status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$work/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED] $seq exit=$status" >&2
    exit "$status"
  fi
  echo "[DONE] $seq"
done

python summarize_eth3d4_refine_sweep.py --root "$OUTPUT_ROOT" --seed "$SEED" || true
