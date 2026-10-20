#!/usr/bin/env bash
# Does the density-cap cost curve hold on the other training seeds?
#
# P1-1's headline cap result -- "cap=8 costs -0.267 mIoU while cap=32 costs only
# -0.053, i.e. concave marginal gain" and "cap=32 discards 31.5% of the weight
# mass for -0.06 mIoU" -- was measured on the seed-42 checkpoint only.  P1-5 then
# showed that single-run comparisons in this project can be wrong by 0.2-0.9
# mIoU purely from the metric's class-divisor artifact, so a one-seed curve is
# not something to lean on.
#
# This stage re-evaluates the same four caps on the seed-43 and seed-44
# checkpoints.  Only the checkpoint changes: the config files are byte-identical
# to the ones the seed-42 points used (`tests/test_control_configs.py` already
# pins cap8/32/64/128 to differ from A5 only in `max_neighbours`), so any spread
# across seeds is training randomness.  Eight evaluations at ~1 min each.
#
# Appended after the cap/cell probe, i.e. last in the pipeline.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the cap/cell interaction probe ==="
while pgrep -f "[r]un_p4_cap_cell_probe\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

for seed in s43 s44; do
  if [ ! -f "out/a5_${seed}/epoch_40.pth" ]; then
    echo "FATAL: out/a5_${seed}/epoch_40.pth missing; cannot check the cap curve" >&2
    exit 1
  fi
done

echo "=== cap 8/32/64/128 on A5 seeds 43 and 44 ==="
python3.10 benchmarks/p1/eval_matrix.py --variants \
  a5_cap8_s43 a5_cap32_s43 a5_cap64_s43 a5_cap128_s43 \
  a5_cap8_s44 a5_cap32_s44 a5_cap64_s44 a5_cap128_s44 || exit 1
python3.10 benchmarks/p1/eval_matrix.py --reparse || exit 1
echo "=== cap seed check complete at $(date -Is) ==="
