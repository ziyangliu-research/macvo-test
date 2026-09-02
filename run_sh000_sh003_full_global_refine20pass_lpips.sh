#!/usr/bin/env bash
set -u

# Full SH000-SH003 quality experiment.
# Online stage remains the frozen W20/R30/B100/M50/Th=.10 protocol.
# After online mapping:
#   pass 0: evaluate all Train/Test views with PSNR/SSIM/LPIPS(VGG)
#   opacity reset once
#   continue online Gaussian optimizer
#   20 shuffled full passes over all Train views
#   evaluate all Train/Test views after every completed pass
# No FPS/wall-time reporting is used for this experiment.

cd /home/shiyo/Desktop/MAC-VO || exit 1

PIPELINE_CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo"
GT_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt"
RUNNER="run_pipeline_execution_benchmark_repro_posthoc_global_refine_pass_lpips.py"

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

run_case() {
  local seq="$1"
  local data_config="Config/Sequence/TartanAirV1_Challenge_${seq}.yaml"
  local gt="$GT_ROOT/${seq}.txt"
  local n
  n=$(count_frames "$seq") || return 2

  local ntest=$(( n / 5 ))
  local ntrain=$(( n - ntest ))
  local bookkeeping_iters=$(( ntrain * 100 ))

  local name="${seq}_full_final_W20_R30_B100_th010_global_refine20pass_lpips"
  local work_dir="outputs/${name}"
  local output_name="incremental_${name}"
  local curve="$work_dir/$output_name/posthoc_global_refinement_pass_metrics.json"

  if [[ -f "$curve" ]]; then
    echo "[skip complete] $seq -> $curve"
    return 0
  fi

  mkdir -p "$work_dir"
  echo
  echo "================================================================"
  echo "$seq | frames=$n | nominal train=$ntrain | nominal test=$ntest"
  echo "Online: strict 8:2 | W20 R30 B100 M50 Th=.10"
  echo "Post-hoc: reset opacity | continue optimizer | 20 full Train passes"
  echo "Metrics at pass 0..20: PSNR / SSIM / LPIPS(VGG)"
  echo "================================================================"

  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED=0 \
  PIPELINE_BENCHMARK_SEED=0 \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  PIPELINE_GLOBAL_REFINE_PASSES=20 \
  PIPELINE_GLOBAL_REFINE_SEED=0 \
  CUDA_VISIBLE_DEVICES=0 \
  TORCH_COMPILE_DISABLE=1 \
  python "$RUNNER" \
    --mode serial \
    --with_pose_metrics \
    --config "$PIPELINE_CONFIG" \
    --set paths.data_config="$data_config" \
    --set evaluation.gt_pose_file="$gt" \
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

  local status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$work_dir/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "[FAILED] $seq exit=$status; continuing" >&2
  fi
}

for seq in SH000 SH001 SH002 SH003; do
  run_case "$seq"
done

echo
echo "All requested sequences attempted. Summarize with:"
echo "  python summarize_sh000_sh003_global_refine20pass_lpips.py"
