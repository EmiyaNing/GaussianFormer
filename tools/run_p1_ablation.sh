#!/usr/bin/env bash
# Launch one P1 ablation run with exclusivity and environment guards.
#
#   tools/run_p1_ablation.sh <config-path> <work-dir> <seed>
#
# Guards:
#  * refuses to start while another train.py is running, because two concurrent
#    runs exhaust the 24 GiB card and the resulting OOM is reported through an
#    NVML path that is broken on this host (driver/library mismatch), which
#    replaces the real error with an INTERNAL ASSERT;
#  * disables PyTorch's NVML-based device check for the same reason.
set -euo pipefail

CONFIG="${1:?usage: run_p1_ablation.sh <config> <work-dir> <seed>}"
WORK_DIR="${2:?missing work dir}"
SEED="${3:-42}"

cd "$(dirname "$0")/.."

if pgrep -f "[t]rain\.py --py-config" > /dev/null; then
  echo "refusing to start: another train.py is already running" >&2
  exit 1
fi

export PYTORCH_NVML_BASED_CUDA_CHECK=0
LOG="/tmp/p1_$(basename "${WORK_DIR}").log"
mkdir -p "${WORK_DIR}"

echo "config=${CONFIG} work_dir=${WORK_DIR} seed=${SEED} log=${LOG}"
nohup python3.10 train.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --seed "${SEED}" > "${LOG}" 2>&1 &
echo "pid=$! log=${LOG}"
