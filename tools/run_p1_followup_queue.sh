#!/usr/bin/env bash
# Queue the GPU work that must follow the P1 seed chain, strictly serially.
#
# Everything here shares one 24-GiB card with the seed chain, and the OOM path
# on this host is broken (driver/library mismatch 595.84 vs 595.91 turns an
# out-of-memory into an NVML INTERNAL ASSERT), so two concurrent jobs do not
# fail cleanly — they crash.  This script therefore *waits* for the chain
# process itself to disappear before touching the GPU, and never assumes the
# card is free just because train.py is momentarily absent (the chain is also
# running eval between runs).
#
# Steps, in order of decision value:
#   1. tile-size sweep (2/4/8) for the block G2V design space   -> P4-8
#   2. train + evaluate the missing a2_knn_max control          -> completes P1's
#      aggregation-control family (currently mean + attention only)
#   3. re-aggregate the multi-seed P1 comparison                -> P1-5
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the P1 seed chain to finish ==="
while pgrep -f "[r]un_p1_a3_and_seeds\.sh" > /dev/null; do sleep 60; done
# The chain's own train.py may still be draining after the script exits.
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
sleep 20
echo "=== GPU handed over at $(date -Is) ==="

# --- 1. tile-size sweep -------------------------------------------------
# The sweep rebuilds the extension per tile edge and restores tile 4 in a
# finally block, so a failure here still leaves the tree in the default state.
echo "=== tile-size sweep ==="
if ! python3.10 benchmarks/p4/sweep_tile_size.py --tiles 2 4 8 --reps 15 \
     2>&1 | tee benchmarks/results/p4_tile_sweep.log; then
  echo "WARN: tile sweep failed; continuing with the P1 control run" >&2
fi

# --- 2. the missing a2_knn_max control ----------------------------------
echo "=== training a2_knn_max ==="
if ! tools/run_p1_ablation.sh config/opus/ablations/p1_a2_knn_max.py out/a2_knn_max 42; then
  echo "FATAL: could not launch a2_knn_max" >&2
  exit 1
fi
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
sleep 15
if [ ! -f out/a2_knn_max/epoch_40.pth ]; then
  echo "FATAL: a2_knn_max produced no epoch_40.pth; refusing to evaluate" >&2
  exit 1
fi
python3.10 benchmarks/p1/eval_matrix.py --variants a2_knn_max || exit 1

# --- 3. multi-seed aggregate --------------------------------------------
echo "=== multi-seed aggregate ==="
python3.10 benchmarks/p1/summarize_seeds.py --json benchmarks/results/p1_seeds.json
echo "=== queue complete at $(date -Is) ==="
