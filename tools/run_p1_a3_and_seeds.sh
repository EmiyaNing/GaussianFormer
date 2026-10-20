#!/usr/bin/env bash
# Finish the P1 control matrix (A3) and then produce the multi-seed repeats of
# the decisive pair (A5 Gaussian vs A2 KNN mean).
#
# Serial by construction: two concurrent runs exhaust the 24-GiB card and the
# resulting OOM is reported through an NVML path that is broken on this host
# (driver/library mismatch 595.84 vs 595.91), replacing the real error with an
# INTERNAL ASSERT.  Every step aborts loudly rather than silently skipping, so
# an unevaluated variant can never be mistaken for a finished one.
set -uo pipefail

cd "$(dirname "$0")/.."

run_one() {
  local name="$1" config="$2" seed="${3:-42}"
  local work="out/${name}"
  echo "=== training ${name} (seed ${seed}) ==="
  if ! tools/run_p1_ablation.sh "${config}" "${work}" "${seed}"; then
    echo "FATAL: could not launch training for ${name}" >&2
    exit 1
  fi
  while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
  sleep 15
  if [ ! -f "${work}/epoch_40.pth" ]; then
    echo "FATAL: ${name} produced no epoch_40.pth; refusing to evaluate" >&2
    exit 1
  fi
  echo "=== evaluating ${name} ==="
  python3.10 benchmarks/p1/eval_matrix.py --variants "${name}"
}

run_one a3_graph_attention config/opus/ablations/p1_a3_graph_attention.py 42
run_one a5_s43 config/opus/ablations/p1_a5_gaussian.py 43
run_one a2_s43 config/opus/ablations/p1_a2_knn_mean.py 43
run_one a5_s44 config/opus/ablations/p1_a5_gaussian.py 44
run_one a2_s44 config/opus/ablations/p1_a2_knn_mean.py 44
echo "A3 + seed chain complete"
