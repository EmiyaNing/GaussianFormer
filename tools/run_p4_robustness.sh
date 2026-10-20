#!/usr/bin/env bash
# Second-stage GPU queue: the P4.3 accuracy-side robustness probe.
#
# Runs strictly after the P1 follow-up queue (tile sweep + a2_knn_max), because
# both need the whole 24-GiB card and the OOM path on this host reports an NVML
# INTERNAL ASSERT instead of the real error.  Waiting on the queue *script* (not
# on train.py) matters: the queue also runs evaluations, during which no
# train.py exists but the GPU is still busy.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the P1 follow-up queue ==="
while pgrep -f "[r]un_p1_followup_queue\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[r]un_p1_a3_and_seeds\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
sleep 20
echo "=== GPU handed over at $(date -Is) ==="

if [ ! -f out/p1_a5_gaussian/epoch_40.pth ]; then
  echo "FATAL: frozen A5 adapter checkpoint missing; cannot probe robustness" >&2
  exit 1
fi

echo "=== P4.3 accuracy-side robustness probe ==="
python3.10 benchmarks/p4/robustness_probe.py || exit 1
echo "=== robustness probe complete at $(date -Is) ==="
