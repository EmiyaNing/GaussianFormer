#!/usr/bin/env bash
# Eighth-stage queue: the EFFECTIVE-field-matched KNN control (A2'').
#
# A5's Gaussian weighting is not uniform: its effective sample size at the
# learned scale 0.53 is 59.3 (median 38.6), not the nominal 215.2 accepted
# neighbours.  A uniform mean over k neighbours has ESS = k, so k = 59 is the
# control that isolates the WEIGHTING at matched effective field, while k = 215
# (the seventh stage) tests whether uniform weighting survives a 3.6x larger
# effective field.
#
# Runs after the k=215 control so the two never touch the card together; the
# OOM path on this host reports an NVML INTERNAL ASSERT rather than the real
# error, so concurrency is not survivable.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the k=215 control ==="
while pgrep -f "[r]un_p1_a2k215\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

# k is far above anything trained before, so prove the CUDA k-NN path
# and the index-tensor size work on the real card first (seconds vs the
# 50 minutes a mid-run failure would cost).
python3.10 benchmarks/p1/smoke_knn.py --config config/opus/ablations/p1_a2_knn_k59.py || exit 1

if ! tools/run_p1_ablation.sh config/opus/ablations/p1_a2_knn_k59.py out/a2_knn_k59 42; then
  echo "FATAL: could not launch A2-double-prime training" >&2
  exit 1
fi
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
sleep 15
if [ ! -f out/a2_knn_k59/epoch_40.pth ]; then
  echo "FATAL: A2-double-prime produced no epoch_40.pth; refusing to evaluate" >&2
  exit 1
fi
python3.10 benchmarks/p1/eval_matrix.py --variants a2_knn_k59 || exit 1
python3.10 benchmarks/p1/eval_matrix.py --reparse || exit 1
python3.10 benchmarks/p1/summarize_p1.py benchmarks/results/p1_eval.json \
  > benchmarks/results/p1_table_a2k215.md || exit 1
echo "=== effective-field-matched control complete at $(date -Is) ==="
