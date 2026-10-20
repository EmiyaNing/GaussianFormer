#!/usr/bin/env bash
# Train and evaluate the P1 control adapters (A2, A1, A3) back to back.
#
# Runs strictly serially: two concurrent runs exhaust the 24-GiB card, and the
# resulting OOM is reported through an NVML path that is broken on this host
# (driver/library mismatch 595.84 vs 595.91), replacing the real error with an
# INTERNAL ASSERT.  Each variant trains for the same 40-epoch budget as A5 so
# the comparison is matched, then is evaluated and merged into p1_eval.json.
set -uo pipefail

cd "$(dirname "$0")/.."

run_one() {
  local name="$1" config="$2" seed="${3:-42}"
  local work="out/${name}"
  echo "=== training ${name} ==="
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

run_one a2_knn_mean config/opus/ablations/p1_a2_knn_mean.py
run_one a1_pointwise config/opus/ablations/p1_a1_pointwise.py
run_one a3_graph_attention config/opus/ablations/p1_a3_graph_attention.py
echo "P1 control chain complete"
