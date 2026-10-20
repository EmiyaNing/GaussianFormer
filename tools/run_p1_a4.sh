#!/usr/bin/env bash
# Fifth-stage queue: train and evaluate P1's missing A4 control.
#
# A4 is the Gaussian kernel with FIXED isotropic scale/opacity (plan T1.1).  It
# closes the one empty cell of the P1 2x2:
#   A4 vs A2  -> Gaussian distance weighting vs uniform KNN-8 averaging
#   A5 vs A4  -> learnable per-point geometry vs fixed isotropic geometry
# Without it, "A5 finished last" is consistent with both "the kernel form adds
# nothing" and "the kernel helps but the parameterisation was the mistake".
#
# Same budget, split, warm start and matched parameter-count protocol as every
# other P1 variant, so it is directly comparable.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the paired horizon probe ==="
while pgrep -f "[r]un_p3_horizon_paired\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

if ! tools/run_p1_ablation.sh config/opus/ablations/p1_a4_fixed_gaussian.py out/a4_fixed_gaussian 42; then
  echo "FATAL: could not launch A4 training" >&2
  exit 1
fi
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
sleep 15
if [ ! -f out/a4_fixed_gaussian/epoch_40.pth ]; then
  echo "FATAL: A4 produced no epoch_40.pth; refusing to evaluate" >&2
  exit 1
fi
python3.10 benchmarks/p1/eval_matrix.py --variants a4_fixed_gaussian || exit 1
python3.10 benchmarks/p1/eval_matrix.py --reparse || exit 1
python3.10 benchmarks/p1/summarize_p1.py benchmarks/results/p1_eval.json \
  > benchmarks/results/p1_table_a4.md || exit 1
echo "=== A4 complete at $(date -Is) ==="
