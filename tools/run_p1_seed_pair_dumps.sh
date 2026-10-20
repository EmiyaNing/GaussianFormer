#!/usr/bin/env bash
# Per-sample dumps for the three paired seeds, then the clustered paired bootstrap.
#
# P1-6's §6 criterion is: "if the paired interval excludes zero for every seed
# while the sign still flips, the noise that makes the comparison unresolvable is
# training randomness, not evaluation sampling".  That needs per-sample dumps for
# BOTH arms of EVERY seed pair.  The precision/recall stage writes dumps for
# `a5_s43` but not for `a2_s43`, `a5_s44` or `a2_s44` (it was written before the
# seed-44 variants existed), so the pairing can currently only be done for seed
# 42.  This stage fills exactly those gaps.
#
# It also exercises the new occupancy precision/recall output of
# `bootstrap_ci.py`, which reports the pair separating "misses occupied space"
# from "hallucinates occupied space" with the same scene-clustered treatment.
#
# Appended after the tile holdout, i.e. last in the pipeline.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== waiting for the tile-size holdout ==="
while pgrep -f "[r]un_p4_tile_holdout\.sh" > /dev/null; do sleep 60; done
while pgrep -f "[t]rain\.py --py-config" > /dev/null; do sleep 30; done
while pgrep -f "[e]val\.py" > /dev/null; do sleep 30; done
sleep 15
echo "=== GPU handed over at $(date -Is) ==="

export DSH_EVAL_PER_SAMPLE_DIR=out/p1_persample
PAIRS="42:a5_cap0:a2_knn_mean 43:a5_s43:a2_s43 44:a5_s44:a2_s44"

# Evaluate only the variants whose dump is missing, so a re-run is cheap.
MISSING=""
for spec in $PAIRS; do
  for name in "$(echo "$spec" | cut -d: -f2)" "$(echo "$spec" | cut -d: -f3)"; do
    if [ ! -f "out/p1_persample/${name}.json" ]; then
      case " $MISSING " in
        *" $name "*) ;;
        *) MISSING="$MISSING $name" ;;
      esac
    fi
  done
done

if [ -n "$MISSING" ]; then
  echo "=== evaluating (dump only):$MISSING ==="
  # shellcheck disable=SC2086
  python3.10 benchmarks/p1/eval_matrix.py --variants $MISSING || exit 1
  python3.10 benchmarks/p1/eval_matrix.py --reparse || exit 1
else
  echo "=== every paired dump already present ==="
fi

for spec in $PAIRS; do
  seed="$(echo "$spec" | cut -d: -f1)"
  a5="$(echo "$spec" | cut -d: -f2)"
  a2="$(echo "$spec" | cut -d: -f3)"
  if [ ! -f "out/p1_persample/${a5}.json" ] || [ ! -f "out/p1_persample/${a2}.json" ]; then
    echo "WARN: seed ${seed} dumps missing (${a5} / ${a2}); skipping" >&2
    continue
  fi
  echo "=== paired clustered bootstrap, seed ${seed}: ${a5} vs ${a2} ==="
  # The first named variant is the base, so the printed paired difference is A5-A2.
  python3.10 benchmarks/p1/bootstrap_ci.py \
    --variants "${a5}=out/p1_persample/${a5}.json" \
               "${a2}=out/p1_persample/${a2}.json" \
    --json "benchmarks/results/p1_bootstrap_ci_s${seed}.json" || exit 1
done

# Combined report over all six paired variants, for the record.
SPECS=""
for spec in $PAIRS; do
  for name in "$(echo "$spec" | cut -d: -f2)" "$(echo "$spec" | cut -d: -f3)"; do
    if [ -f "out/p1_persample/${name}.json" ]; then
      case " $SPECS " in
        *" ${name}="*) ;;
        *) SPECS="$SPECS ${name}=out/p1_persample/${name}.json" ;;
      esac
    fi
  done
done
if [ -n "$SPECS" ]; then
  echo "=== combined bootstrap over all paired variants ==="
  # shellcheck disable=SC2086
  python3.10 benchmarks/p1/bootstrap_ci.py --variants $SPECS \
    --json benchmarks/results/p1_bootstrap_ci.json || exit 1
fi

echo "=== seed-pair bootstrap complete at $(date -Is) ==="
