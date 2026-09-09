#!/usr/bin/env bash
set -uo pipefail

# Full TartanAir Stereo Challenge SE000-SE003 benchmark.
# Online protocol: strict8:2, W20/rho=.30/B100/M50/Th=.10, serial/default ReSplat stream.
# Post-hoc refinement: one extra opacity reset, then the native GraphDECO
# densification/pruning/reset schedule, with exact metric checkpoints every 10k
# optimizer updates from 10k through 100k.
#
# Native densification can make one rasterization kernel substantially longer
# than the fixed-topology refinement path.  A GPU that is also driving X/Wayland
# may therefore trip NVIDIA's kernel execution watchdog (cudaErrorLaunchTimeout).
# By default this launcher refuses a long run when that watchdog is enabled,
# preserving the native optimization protocol instead of weakening densification.

cd /home/shiyo/Desktop/MAC-VO || exit 1

GPU="${GPU:-0}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
WATCHDOG_POLICY="${WATCHDOG_POLICY:-require_disabled}"  # require_disabled | warn | ignore
ITER_CHECKPOINTS="${ITER_CHECKPOINTS:-10000,20000,30000,40000,50000,60000,70000,80000,90000,100000}"
TOTAL_ITERS="${TOTAL_ITERS:-100000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/se000_se003_native3dgs_refine100k}"
CONFIG="Config/Pipeline/MACVO_ReSplat_Serial_TartanAirV1_SH003_0_200_AllFrames.yaml"
DATA_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo"
GT_ROOT="/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt"
RUNNER="run_pipeline_execution_benchmark_repro_posthoc_global_refine_native3dgs_lpips.py"
SEQS=(SE000 SE001 SE002 SE003)

python - <<'PY'
from lpips import LPIPS
print('[preflight] LPIPS import OK')
PY
if [[ $? -ne 0 ]]; then
  echo "[fatal] LPIPS import failed" >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  watchdog=$(nvidia-smi -q -i "$GPU" 2>/dev/null | awk -F: '/Kernel Execution Timeout/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')
  if [[ -n "$watchdog" ]]; then
    echo "[preflight] GPU $GPU Kernel Execution Timeout: $watchdog"
    if [[ "$watchdog" == "Enabled" && "$WATCHDOG_POLICY" == "require_disabled" ]]; then
      cat >&2 <<'EOF'
[fatal] NVIDIA kernel execution watchdog is enabled on the selected GPU.
Native 3DGS densification can create rasterization kernels long enough to hit
cudaErrorLaunchTimeout.  Do not change the densification/pruning protocol to
work around this.  Run the experiment on a non-display/headless GPU, or stop the
display manager from SSH/TTY first, then rerun this launcher.

Check:
  nvidia-smi -q -i 0 | grep -A2 "Kernel Execution Timeout"

Typical Ubuntu headless step (run from SSH/TTY; local GUI will disappear):
  sudo systemctl stop display-manager

Restore GUI after experiments:
  sudo systemctl start display-manager

To bypass only this preflight check (not recommended):
  WATCHDOG_POLICY=warn bash run_se000_se003_native3dgs_refine100k_seed0.sh
EOF
      exit 90
    elif [[ "$watchdog" == "Enabled" && "$WATCHDOG_POLICY" == "warn" ]]; then
      echo "[warn] CUDA watchdog is enabled; cudaErrorLaunchTimeout remains possible" >&2
    fi
  fi
fi

mkdir -p "$OUTPUT_ROOT"

echo "======================================================================"
echo "SE000-SE003 | native GraphDECO global refinement | seed=$SEED GPU=$GPU"
echo "Metric checkpoints: $ITER_CHECKPOINTS"
echo "Total refinement : $TOTAL_ITERS updates"
echo "Native schedule  : densify >500 every100 until <15000; opacity<.005"
echo "                   reset every3000 inside densification window;"
echo "                   max_screen=20 after iter>3000"
echo "======================================================================"

for seq in "${SEQS[@]}"; do
  data_cfg="Config/Sequence/TartanAirV1_Challenge_${seq}.yaml"
  seq_root="$DATA_ROOT/$seq"
  gt="$GT_ROOT/$seq.txt"

  for p in "$CONFIG" "$data_cfg" "$gt" "$RUNNER"; do
    if [[ ! -f "$p" ]]; then
      echo "[fatal] missing: $p" >&2
      exit 2
    fi
  done
  if [[ ! -d "$seq_root/image_left" || ! -d "$seq_root/image_right" ]]; then
    echo "[fatal] missing stereo images: $seq_root" >&2
    exit 2
  fi

  n=$(python - "$seq_root" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
left = [p for p in (root/'image_left').iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'}]
right = [p for p in (root/'image_right').iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'}]
if not left or len(left) != len(right):
    raise SystemExit(f'invalid stereo counts left={len(left)} right={len(right)}')
print(len(left))
PY
  ) || exit 2

  ntest=$(( n / 5 ))
  ntrain=$(( n - ntest ))
  online_bookkeeping_iters=$(( ntrain * 100 ))

  work="$OUTPUT_ROOT/$seq/quality_seed${SEED}"
  name="incremental_${seq}_native3dgs_refine100k_seed${SEED}"
  metrics="$work/$name/posthoc_global_refinement_native3dgs_metrics.json"
  summary="$work/execution_benchmark_summary.json"
  mkdir -p "$work"

  if [[ "$FORCE" != "1" && -f "$metrics" && -f "$summary" ]]; then
    complete=$(python - "$metrics" "$TOTAL_ITERS" "$ITER_CHECKPOINTS" <<'PY'
import json, sys
p, total, raw = sys.argv[1:]
d = json.load(open(p))
want = {int(x) for x in raw.split(',') if x.strip()}
have = {int(x) for x in (d.get('checkpoints') or {}).keys()}
ok = int(d.get('total_refinement_iterations', -1)) == int(total) and want <= have
print(1 if ok else 0)
PY
    )
    if [[ "$complete" == "1" ]]; then
      echo "[skip complete] $seq"
      continue
    fi
  fi

  echo ""
  echo "======================================================================"
  echo "$seq | frames=$n train=$ntrain test=$ntest"
  echo "Online iterations=$online_bookkeeping_iters | Offline=$TOTAL_ITERS"
  echo "Output=$work"
  echo "======================================================================"

  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONHASHSEED="$SEED" \
  PIPELINE_BENCHMARK_SEED="$SEED" \
  PIPELINE_HISTORICAL_REPLAY_FRACTION=0.30 \
  PIPELINE_GLOBAL_REFINE_TOTAL_ITERS="$TOTAL_ITERS" \
  PIPELINE_GLOBAL_REFINE_ITER_CHECKPOINTS="$ITER_CHECKPOINTS" \
  PIPELINE_GLOBAL_REFINE_SEED="$SEED" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TORCH_COMPILE_DISABLE=1 \
  python "$RUNNER" \
    --mode serial \
    --with_pose_metrics \
    --config "$CONFIG" \
    --set paths.data_config="$data_cfg" \
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
    --set backend.optimization.iterations="$online_bookkeeping_iters" \
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
    python summarize_se_native3dgs_partial.py "$OUTPUT_ROOT" "$seq" "$SEED" || true
    if grep -qE 'launch timed out and was terminated|cudaErrorLaunchTimeout' "$work/run.log"; then
      echo "[fatal] CUDA launch timeout detected.  The CUDA context is no longer reliable." >&2
      echo "        Disable the display-GPU watchdog / use a headless GPU, then rerun." >&2
      exit 70
    fi
    exit "$status"
  fi

done

echo "[DONE] SE000-SE003 native3dgs refinement sweep"
