#!/usr/bin/env bash
set -euo pipefail

cd /home/shiyo/Desktop/MAC-VO

GPU="${GPU:-0}"
SEED="${SEED:-0}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
RUNNER="run_pipeline_execution_benchmark_repro_test_psnr_only.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sh003_0_200_opacity_003_01_test_psnr}"

mkdir -p "$OUTPUT_ROOT"

for TH in 0.03 0.10; do
  TAG=$(python - "$TH" <<'PY'
import sys
x=float(sys.argv[1])
print(f"{int(round(x*1000)):03d}")
PY
)
  WORK="$OUTPUT_ROOT/th_${TAG}"
  NAME="incremental_SH003_0_200_W20_R30_B100_M50_Th${TH}_testpsnr"
  mkdir -p "$WORK"

  echo "======================================================================"
  echo "SH003 [0,200) | strict8:2 | W20 R30 B100 M50 | Th=$TH | seed=$SEED"
  echo "Final metric: held-out Test PSNR only | no post-hoc refinement"
  echo "Output: $WORK"
  echo "======================================================================"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED="$SEED" \
  PIPELINE_BENCHMARK_SEED="$SEED" \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TORCH_COMPILE_DISABLE=1 \
  python "$RUNNER" \
    --mode serial \
    --config "$CONFIG" \
    --set sequence.start_index=0 \
    --set sequence.end_index=200 \
    --set split.split_every=5 \
    --set split.split_offset=4 \
    --set split.split_index_mode=local_index \
    --set backend.local_map_size=20 \
    --set backend.iterations_per_packet=100 \
    --set backend.reset_new_packet_opacity=true \
    --set backend.new_packet_reset_max_opacity=0.01 \
    --set backend.maintenance_mode=standard \
    --set backend.maintenance_after_local_iteration=50 \
    --set backend.maintenance_min_opacity="$TH" \
    --set backend.optimization.iterations=16000 \
    --set backend.evaluation_enabled=false \
    --set backend.eval_before_optimization=false \
    --set backend.eval_every_train_packets=1000000 \
    --set backend.save_every_train_packets=0 \
    --set backend.save_final_ply=false \
    --set backend.wandb_mode=disabled \
    --set backend.write_runtime_artifacts=true \
    --set paths.work_dir="$WORK" \
    --set backend.output_name="$NAME" \
    2>&1 | tee "$WORK/run.log"

done

python - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
print("\n=== SH003 opacity threshold | Test PSNR only ===")
print(f"{'Th':>6} {'Test PSNR(dB)':>15} {'Test views':>11} {'Gaussians':>12}")
print('-'*50)
for th, tag in [(0.03,'030'), (0.10,'100')]:
    files=sorted((root/f'th_{tag}').glob('*/test_psnr_only.json'))
    if not files:
        print(f"{th:6.2f} {'MISSING':>15} {'MISSING':>11} {'MISSING':>12}")
        continue
    d=json.load(open(files[0]))
    m=d.get('test') or {}
    psnr=m.get('psnr')
    p='MISSING' if psnr is None else f'{float(psnr):.6f}'
    print(f"{th:6.2f} {p:>15} {m.get('num_views','MISSING'):>11} {d.get('num_gaussians','MISSING'):>12}")
PY
