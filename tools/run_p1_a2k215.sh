#!/usr/bin/env bash
# Seventh-stage queue: the receptive-field-matched KNN control (A2').
#
# P1's verdict compares A5 (Gaussian weighting over ~215 accepted neighbours)
# against A2 (uniform mean over 8), so it confounds the weighting with the
# receptive field and the aggregation arithmetic.  This run changes only the
# neighbour count, to A5's final trained receptive field.
#
# Memory note: gathering k=215 neighbours instead of 8 makes the index 59 MB per
# (batch item, horizon) rather than 2 MB.  A gather's backward needs the indices,
# not the gathered values, so this stays within the card's headroom; the run
# aborts loudly rather than being evaluated if it does not produce epoch_40.pth.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the full-val protocol check ==="
while pgrep -f "[r]un_p1_fullval\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

# k is far above anything trained before, so prove the CUDA k-NN path
# and the index-tensor size work on the real card first (seconds vs the
# 50 minutes a mid-run failure would cost).
python3.10 benchmarks/p1/smoke_knn.py --config config/opus/ablations/p1_a2_knn_k215.py || exit 1

if ! tools/run_p1_ablation.sh config/opus/ablations/p1_a2_knn_k215.py out/a2_knn_k215 42; then
  echo "FATAL: could not launch A2-prime training" >&2
  exit 1
fi
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
sleep 15
if [ ! -f out/a2_knn_k215/epoch_40.pth ]; then
  echo "FATAL: A2-prime produced no epoch_40.pth; refusing to evaluate" >&2
  exit 1
fi
python3.10 benchmarks/p1/eval_matrix.py --variants a2_knn_k215 || exit 1
python3.10 benchmarks/p1/eval_matrix.py --reparse || exit 1
python3.10 benchmarks/p1/summarize_p1.py benchmarks/results/p1_eval.json \
  > benchmarks/results/p1_table_a2k215.md || exit 1
echo "=== A2-prime complete at $(date -Is) ==="
