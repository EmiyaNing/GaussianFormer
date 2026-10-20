#!/usr/bin/env bash
# Does the density cap's accuracy cost belong to the cap, or to the bin size?
#
# The cap keeps the first K neighbours found while walking cell shells outward,
# so the selected subset is a function of cell_size as well as K: at K=8, cell
# 0.6 keeps only 0.1% of the same neighbours as cell 1.8 (mean Jaccard 0.100),
# and cell 3.6 keeps ~7% of the rows exactly (Jaccard 0.407).  P1-1/P0-4 report
# "cap=8 costs -0.26 mIoU" from cell 1.8 only, so this stage re-evaluates the
# frozen A5 adapter at cap=8 with the bin edge bracketed (0.9 and 3.6).
#
# Appended to the very end of the pipeline: it waits on the last stage
# (run_p1_a2gauss.sh) and on the GPU being idle.  Inference only -- no training,
# no new weights -- so it cannot invalidate anything upstream.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the A2-gauss stage (last of the queue) ==="
while pgrep -f "[r]un_p1_a2gauss\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 20
echo "=== GPU handed over at $(date -Is) ==="

if [ ! -f out/p1_a5_gaussian/epoch_40.pth ]; then
  echo "FATAL: frozen A5 checkpoint missing; cannot probe the cap/cell interaction" >&2
  exit 1
fi

echo "=== cap=8 at cell 0.9 and 3.6 (frozen A5 weights) ==="
python3.10 benchmarks/p1/eval_matrix.py \
  --variants a5_cap8_cell0p9 a5_cap8_cell3p6 || exit 1
echo "=== cap/cell probe complete at $(date -Is) ==="
