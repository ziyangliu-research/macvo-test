#!/usr/bin/env bash
set -euo pipefail

cd /home/shiyo/Desktop/MAC-VO

GPU="${GPU:-0}"
N_RESPLAT="${N_RESPLAT:-20}"
N_MACVO="${N_MACVO:-100}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_ETH3D_mannequin_face_1_Smoke.yaml"
DATA_ROOT="/home/shiyo/Desktop/Datasets/ETH3D_rectified/mannequin_face_1"
OUT_ROOT="outputs/eth3d_mannequin_face_1_smoke"

for required in \
  "$DATA_ROOT/calibration.json" \
  "$DATA_ROOT/groundtruth_left.txt" \
  "$DATA_ROOT/timestamps.txt"; do
  if [[ ! -f "$required" ]]; then
    echo "[fatal] missing $required" >&2
    exit 2
  fi
done
if [[ ! -d "$DATA_ROOT/image_left" || ! -d "$DATA_ROOT/image_right" ]]; then
  echo "[fatal] missing rectified image_left/image_right" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"

echo "============================================================"
echo "ETH3D mannequin_face_1 smoke"
echo "ReSplat-only frames : $N_RESPLAT"
echo "MAC-VO-only frames  : $N_MACVO"
echo "GPU                 : $GPU"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONHASHSEED=0 \
PIPELINE_BENCHMARK_SEED=0 \
CUDA_VISIBLE_DEVICES="$GPU" \
TORCH_COMPILE_DISABLE=1 \
python evaluate_resplat_default_stream_eth3d_repro.py \
  --config "$CONFIG" \
  --set sequence.start_index=0 \
  --set sequence.end_index="$N_RESPLAT" \
  --output_dir "$OUT_ROOT/resplat_first${N_RESPLAT}" \
  --save_gt \
  2>&1 | tee "$OUT_ROOT/resplat_first${N_RESPLAT}.log"

PYTHONHASHSEED=0 \
PIPELINE_BENCHMARK_SEED=0 \
CUDA_VISIBLE_DEVICES="$GPU" \
TORCH_COMPILE_DISABLE=1 \
python evaluate_macvo_eth3d.py \
  --config "$CONFIG" \
  --set sequence.start_index=0 \
  --set sequence.end_index="$N_MACVO" \
  --output_dir "$OUT_ROOT/macvo_first${N_MACVO}" \
  2>&1 | tee "$OUT_ROOT/macvo_first${N_MACVO}.log"

python - "$OUT_ROOT" "$N_RESPLAT" "$N_MACVO" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
nr = sys.argv[2]
nm = sys.argv[3]
r = json.loads((root / f"resplat_first{nr}" / "summary.json").read_text())
m = json.loads((root / f"macvo_first{nm}" / "macvo_only_summary.json").read_text())
print("\n================ ETH3D smoke summary ================")
print(f"ReSplat frames : {r['num_evaluated_frames']}")
print(f"ReSplat PSNR   : {r['average_psnr']:.3f} dB")
print(f"ReSplat SSIM   : {r['average_ssim']:.4f}")
print(f"Gaussians/frame: {r['average_num_gaussians']:.1f}")
print(f"MAC-VO frames  : {m['frames']}")
print(f"SE3 ATE RMSE   : {m['se3_ate_rmse_m']:.6f} m")
print(f"Sim3 ATE RMSE  : {m['sim3_ate_rmse_m']:.6f} m")
print("======================================================")
PY
