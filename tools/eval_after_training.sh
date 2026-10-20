#!/usr/bin/env bash
# Wait for the A5 run to produce its final checkpoint, then run the cap-sweep
# evaluation.  Chaining avoids polling and keeps the GPU busy without ever
# running two jobs at once (concurrent runs exhaust the card and the ensuing
# OOM is reported through a broken NVML path on this host).
set -uo pipefail

cd "$(dirname "$0")/.."

CKPT="${1:-out/p1_a5_gaussian/epoch_40.pth}"
shift || true
VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(a5_cap0 a5_cap8 a5_cap32 a5_cap64 a5_cap128)
fi

echo "waiting for ${CKPT} ..."
while [ ! -f "${CKPT}" ]; do
  if ! pgrep -f "[t]rain\.py --py-config" > /dev/null; then
    echo "training stopped before producing ${CKPT}" >&2
    exit 1
  fi
  sleep 30
done

# Give the finishing process a moment to release the GPU.
sleep 20

echo "found ${CKPT}; evaluating: ${VARIANTS[*]}"
python3.10 benchmarks/p1/eval_matrix.py --variants "${VARIANTS[@]}"
echo "cap sweep done"
