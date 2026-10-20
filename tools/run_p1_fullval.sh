#!/usr/bin/env bash
# Sixth-stage GPU queue: verify the published SparseWorldStrict anchors on the
# FULL val split, and re-test the P1 A5-vs-A2 gap on 37x more eval samples.
#
# Plan T3.1 makes anchor reproduction a hard prerequisite for trusting A0, and
# it was never done: tools/make_reduced_split.py keeps the first N scenes in
# sorted order, so every P1 number so far rests on 4 of 150 val scenes.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the A4 control ==="
while pgrep -f "[r]un_p1_a4\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

# A0 first: it is the anchor check and the only variant that needs no adapter.
# Then the P1 trio on the same larger sample.  Each variant is ~35 min.
python3.10 benchmarks/p1/fullval_anchor.py \
  --variants a0_base a2_knn_mean a5_cap0 a4_fixed_gaussian || exit 1
echo "=== full-val protocol check complete at $(date -Is) ==="
