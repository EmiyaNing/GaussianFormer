#!/usr/bin/env bash
# Eleventh-stage queue: the weighting-vs-selection control (A2-gauss).
#
# P1-6 section 4.2.1 established that A5's thin-class deficit is not explained by
# the receptive field, but left open whether it is the *weight profile* or the
# *neighbour selection*: at a matched k of 8 the two rules pick almost disjoint
# sets (overlap 0.177) at very different distances (0.399 m vs 0.978 m).
#
# This run changes ONLY the weight profile -- A2's exact k-NN selection with a
# fixed isotropic Gaussian instead of uniform 1/k -- so
#   A2 (uniform, k-NN)  vs A2-gauss (Gaussian, k-NN)   isolates the weighting
#   A5 cap=8 (Gaussian, shell) vs A2-gauss (Gaussian, k-NN) isolates the selection
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the sub-Gaussian diagnostics ==="
while pgrep -f "[r]un_p4_subgaussian\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

if ! tools/run_p1_ablation.sh config/opus/ablations/p1_a2_knn_gaussian8.py out/a2_knn_gaussian8 42; then
  echo "FATAL: could not launch A2-gauss training" >&2
  exit 1
fi
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
sleep 15
if [ ! -f out/a2_knn_gaussian8/epoch_40.pth ]; then
  echo "FATAL: A2-gauss produced no epoch_40.pth; refusing to evaluate" >&2
  exit 1
fi
python3.10 benchmarks/p1/eval_matrix.py --variants a2_knn_gaussian8 || exit 1
python3.10 benchmarks/p1/eval_matrix.py --reparse || exit 1
python3.10 benchmarks/p1/summarize_p1.py benchmarks/results/p1_eval.json \
  > benchmarks/results/p1_table_a2gauss.md || exit 1
echo "=== A2-gauss complete at $(date -Is) ==="
