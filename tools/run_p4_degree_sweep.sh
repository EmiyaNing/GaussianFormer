#!/usr/bin/env bash
# Third-stage GPU queue: the accepted-vs-candidate neighbour count sweep.
#
# The candidate count recorded in p0_matrix_sweeps.jsonl is set by max_scale
# alone and is identical for support_sigma 2.0/3.0/4.0, yet latency changes
# ~2.3x across them.  This sweep measures the box-accepted count, which is the
# missing explanatory variable.
#
# The measurement is a COUNT, not a timing, so contention cannot corrupt it; it
# is queued behind training only because it allocates a sample x G x 3 temporary
# and this host reports OOM through a broken NVML assertion instead of an error.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the robustness probe ==="
while pgrep -f "[r]un_p4_robustness\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

python3.10 benchmarks/p4/accepted_degree_sweep.py || exit 1
echo "=== degree sweep complete at $(date -Is) ==="
