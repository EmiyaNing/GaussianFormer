#!/usr/bin/env bash
# Tenth-stage queue: the sub-Gaussian redundancy diagnostic (plan P4.2 item 3).
#
# The flat real-points dump cannot support this: it applies a boolean mask, so
# only 138,124 of 4x34,560 points survive and the 48-child boundaries are lost.
# This stage re-dumps WITH the grouping (and the validity mask) and then runs the
# CPU diagnostic that measures, per query, the distinct-voxel ratio and the
# pairwise child distances.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the precision/recall re-evaluation ==="
while pgrep -f "[r]un_p1_precision_recall\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

if [ ! -f out/p1_a5_gaussian/epoch_40.pth ]; then
  echo "FATAL: frozen A5 adapter missing; cannot dump grouped points" >&2
  exit 1
fi
python3.10 benchmarks/p4/dump_real_points.py --grouped \
  --config config/opus/ablations/p1_a5_gaussian.py \
  --checkpoint out/p1_a5_gaussian/epoch_40.pth \
  --samples 4 --out benchmarks/results/real_points.pt \
  --json benchmarks/results/p4_real_workload_grouped.json || exit 1

if [ ! -f benchmarks/results/real_points_grouped.pt ]; then
  echo "FATAL: grouped dump not produced" >&2
  exit 1
fi
python3.10 benchmarks/p4/subgaussian_redundancy.py --voxels 0.4 1.8 || exit 1
# Items 2 and 4 from the same dump: the realised scale/opacity distribution and
# the semantic-aliasing / unused-Gaussian statistics.  No model change is needed
# because the segmentor's result dict already carries gaussian_scales,
# gaussian_opacities and the fused per-child logits.
python3.10 benchmarks/p4/gaussian_attribute_distribution.py --voxels 0.4 1.8 || exit 1
echo "=== sub-Gaussian redundancy diagnostic complete at $(date -Is) ==="
