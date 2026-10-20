#!/usr/bin/env bash
# Multi-seed repeats of the decisive pair (A5 Gaussian vs A2 KNN control).
#
# The single-seed difference is ~0.05 mIoU, which sits inside the likely noise
# floor of a 113-sample evaluation.  The plan requires ">= 3 seeds, mean +- std"
# for the final key row, so this chain produces seeds 43 and 44 for both
# variants with otherwise identical data, budget and warm start.
#
# Waits until the A1/A3 control chain has finished before touching the GPU:
# two concurrent runs exhaust the card and the OOM is reported through a broken
# NVML path on this host.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "waiting for the A1/A3 control chain to finish ..."
while true; do
  if python3.10 - <<'PY'
import json, sys
from pathlib import Path
path = Path('benchmarks/results/p1_eval.json')
if not path.exists():
    sys.exit(1)
variants = {row.get('variant') for row in json.loads(path.read_text())}
sys.exit(0 if 'a3_graph_attention' in variants else 1)
PY
  then
    break
  fi
  sleep 60
done
echo "control chain done; starting multi-seed runs"

run_one() {
  local name="$1" config="$2" seed="$3"
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

run_one a5_s43 config/opus/ablations/p1_a5_gaussian.py 43
run_one a2_s43 config/opus/ablations/p1_a2_knn_mean.py 43
run_one a5_s44 config/opus/ablations/p1_a5_gaussian.py 44
run_one a2_s44 config/opus/ablations/p1_a2_knn_mean.py 44
echo "seed chain complete"
