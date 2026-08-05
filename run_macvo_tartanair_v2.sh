#!/usr/bin/env bash
set -euo pipefail

MACVO_ROOT="${MACVO_ROOT:-/home/shiyo/Desktop/MAC-VO}"
DATA_CONFIG="${DATA_CONFIG:-${MACVO_ROOT}/Config/Sequence/TartanAirV2_House_easy_P000.yaml}"
ODOM_CONFIG="${ODOM_CONFIG:-${MACVO_ROOT}/Config/Experiment/MACVO/MACVO_Performant.yaml}"
RESULT_ROOT="${RESULT_ROOT:-${MACVO_ROOT}/Results_P000}"
GPU_ID="${GPU_ID:-0}"
SEQ_FROM="${SEQ_FROM:-0}"
SEQ_TO="${SEQ_TO:-50}"

cd "${MACVO_ROOT}"

echo "MACVO_ROOT : ${MACVO_ROOT}"
echo "ODOM_CONFIG: ${ODOM_CONFIG}"
echo "DATA_CONFIG: ${DATA_CONFIG}"
echo "RESULT_ROOT: ${RESULT_ROOT}"
echo "GPU_ID     : ${GPU_ID}"
echo "FRAME RANGE: [${SEQ_FROM}, ${SEQ_TO})"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python MACVO.py \
  --odom "${ODOM_CONFIG}" \
  --data "${DATA_CONFIG}" \
  --seq_from "${SEQ_FROM}" \
  --seq_to "${SEQ_TO}" \
  --resultRoot "${RESULT_ROOT}" \
  --timing
