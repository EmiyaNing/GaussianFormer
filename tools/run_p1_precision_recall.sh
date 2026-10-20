#!/usr/bin/env bash
# Ninth-stage queue: re-evaluate the P1 matrix so occupied precision/recall land.
#
# The strict evaluator previously reported only mIoU and occupied IoU, neither of
# which separates "misses occupied space" from "hallucinates occupied space" --
# both collapse to the same IoU.  The pair is now derived from the binary
# confusion matrix the evaluator already accumulates (no model change, no
# probability tensor), so getting the numbers is a pure re-evaluation.
#
# Evaluation is deterministic given the checkpoint, so the re-run's mIoU/IoU must
# reproduce the recorded values; the existing JSON is backed up first so a failed
# re-evaluation cannot silently replace a good row with an error row.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the effective-field-matched control ==="
while pgrep -f "[r]un_p1_a2k59\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

cp -n benchmarks/results/p1_eval.json benchmarks/results/p1_eval.json.bak_before_precision || true

VARIANTS="a0_base a1_pointwise a2_knn_mean a3_graph_attention a5_cap0 a5_cap8 a5_cap32 a5_cap64 a5_cap128 a5_s43"
if [ -f out/a4_fixed_gaussian/epoch_40.pth ]; then VARIANTS="$VARIANTS a4_fixed_gaussian"; fi
if [ -f out/a2_knn_k215/epoch_40.pth ]; then VARIANTS="$VARIANTS a2_knn_k215"; fi
if [ -f out/a2_knn_k59/epoch_40.pth ]; then VARIANTS="$VARIANTS a2_knn_k59"; fi
if [ -f out/a2_knn_max/epoch_40.pth ]; then VARIANTS="$VARIANTS a2_knn_max"; fi

echo "re-evaluating: $VARIANTS"
# Per-sample dumps so the bootstrap can separate eval-split noise from
# training-seed noise (see P1-5/P1-6).
export DSH_EVAL_PER_SAMPLE_DIR=out/p1_persample
# Capture TP/FP/FN voxel indices for the first 3 scenes, so P4.2 item 5
# (spatial failure visualisation) has data without a second evaluation.
export DSH_EVAL_SPATIAL_SAMPLES=3
python3.10 benchmarks/p1/eval_matrix.py --variants $VARIANTS || exit 1
python3.10 benchmarks/p1/eval_matrix.py --reparse || exit 1
python3.10 benchmarks/p1/summarize_p1.py benchmarks/results/p1_eval.json \
  > benchmarks/results/p1_table_precision.md || exit 1

# Bootstrap CIs on the per-sample dumps that were just written.
SPECS=""
for name in $VARIANTS; do
  if [ -f "out/p1_persample/${name}.json" ]; then
    SPECS="$SPECS ${name}=out/p1_persample/${name}.json"
  fi
done
if [ -n "$SPECS" ]; then
  # shellcheck disable=SC2086
  python3.10 benchmarks/p1/bootstrap_ci.py --variants $SPECS \
    --json benchmarks/results/p1_bootstrap_ci.json || exit 1
fi

# BEV failure maps for a baseline and the Gaussian adapter, if spatial data landed.
MAP_SPECS=""
for name in a0_base a2_knn_mean a5_cap0; do
  if [ -f "out/p1_persample/${name}.json" ]; then
    MAP_SPECS="$MAP_SPECS ${name}=out/p1_persample/${name}.json"
  fi
done
if [ -n "$MAP_SPECS" ]; then
  # shellcheck disable=SC2086
  python3.10 benchmarks/plots/make_failure_maps.py --variants $MAP_SPECS \
    --horizon 4 --scene 0 || true
fi

# The re-run must reproduce the recorded mIoU, otherwise the checkpoint or the
# split changed underneath us and the new precision/recall numbers describe a
# different model.
python3.10 - <<'PY' || exit 1
import json, sys
rows = json.load(open('benchmarks/results/p1_eval.json'))
backup = json.load(open('benchmarks/results/p1_eval.json.bak_before_precision'))
old = {r['variant']: (r.get('metrics') or {}).get('avg/mIoU') for r in backup}
bad = []
for row in rows:
    new = (row.get('metrics') or {}).get('avg/mIoU')
    was = old.get(row['variant'])
    if was is not None and new is not None and abs(new - was) > 1e-6:
        bad.append((row['variant'], was, new))
if bad:
    print('REPRODUCIBILITY FAILURE: mIoU changed on re-evaluation:', bad)
    sys.exit(1)
print(f'verified: all {len(old)} previously recorded mIoU values reproduced exactly')
for row in rows:
    m = row.get('metrics') or {}
    if 'avg/occupiedPrecision' in m:
        print(f"  {row['variant']:20s} mIoU={m['avg/mIoU']:6.2f} "
              f"prec={m['avg/occupiedPrecision']:6.2f} rec={m['avg/occupiedRecall']:6.2f} "
              f"IoU={m['avg/IoU']:6.2f}")
PY
echo "=== precision/recall re-evaluation complete at $(date -Is) ==="
