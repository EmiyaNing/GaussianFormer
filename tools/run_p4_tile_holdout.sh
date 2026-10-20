#!/usr/bin/env bash
# Holdout test for the three-term tile cost model, on two unseen tile sizes.
#
# P4-10 found that a two-parameter model (load + scan) over-predicts every tile and
# that the error grows with the tile edge (+9% / +18% / +30.5%).  Reading the
# kernel explains why: the inner loop cheaply REJECTS a (voxel, union-entry) pair
# with a per-axis AABB test before doing the full evaluation, so "one scan" has
# two prices.  The rejection fraction is exact and grows with the tile (26% / 55%
# / 77%), matching the residual shape.
#
# Fitting the three-term model (U*l + L*s_full + (S-L)*s_rej) on tiles 2/4/8 is an
# exact fit with three unknowns -- self-consistent, but not evidence.  This stage
# measures tiles 3 and 6, which take no part in the fit, so they are a real
# holdout:  six data points, two of them untouched.
#
# Appended after the cap-seed check, i.e. last in the pipeline.  Each tile rebuild
# is verified with tile_edge() and the default tile is restored in a finally block.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the cap-seed check ==="
while pgrep -f "[r]un_p1_cap_seed_check\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

echo "=== timing tiles 3 and 6 (holdout) ==="
python3.10 benchmarks/p4/sweep_tile_size.py --tiles 3 6 --reps 15 \
  --json benchmarks/results/p4_tile_sweep_holdout.json 2>&1 | tail -20 || exit 1

echo "=== holdout check ==="
python3.10 benchmarks/p4/tile_holdout.py || exit 1
echo "=== tile holdout complete at $(date -Is) ==="
