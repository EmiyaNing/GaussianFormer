"""The density-cap cost curve with a training-seed dimension (plan P1-11).

P1-1 measured the cap cost on one checkpoint: cap=8 costs -0.267 mIoU, cap=32
-0.053, and the marginal gain per unit of recovered weight mass is concave.  A
single-seed curve is exactly the kind of measurement this project has been burned
by -- P1-5 showed a metric divisor artifact can move one run's mIoU by 0.2-0.9 --
so P1-11 re-evaluates the same four caps on the seed-43 and seed-44 checkpoints.

Two design points this script enforces:

* **The cost is paired within a seed.**  Each cap is compared against the *same
  seed's* uncapped number (``a5_cap0`` / ``a5_s43`` / ``a5_s44``), so the seed's
  overall level cancels; the across-seed spread is then the quantity of interest,
  not a confound.
* **Fixed divisors.**  Every mIoU is recomputed on the class set common to all
  compared variants.  Using the recorded ``avg/mIoU`` would let the artifact
  masquerade as a cap effect.

    python3.10 benchmarks/p1/cap_seed_curve.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

CAPS = (8, 32, 64, 128)
SEEDS = ('s42', 's43', 's44')

# Uncapped reference per seed: the plain A5 config evaluated on that seed's
# checkpoint.  s42 is the checkpoint every earlier report used.
BASELINE = {'s42': 'a5_cap0', 's43': 'a5_s43', 's44': 'a5_s44'}


def variant_name(cap, seed):
    return f'a5_cap{cap}' if seed == 's42' else f'a5_cap{cap}_{seed}'


def needed_variants():
    names = [BASELINE[seed] for seed in SEEDS]
    names += [variant_name(cap, seed) for cap in CAPS for seed in SEEDS]
    return names


def fixed_divisor_values(names):
    """``({variant: fixed avg mIoU}, {variant: [per-horizon]})`` or ``({}, {})``.

    The scope is the intersection over exactly ``names``: the fixed class set
    depends on what is compared (P1-5 4.6), so it has to be stated rather than
    inherited from whatever happens to be in the artifact.
    """
    from benchmarks.p1.common_class_miou import comparable_for, load
    logs = load(names)
    present = [name for name in names if name in logs]
    if not present:
        return {}, {}
    table, _ = comparable_for(logs, present)
    return ({name: table[name]['avg'] for name in present},
            {name: table[name]['per_horizon'] for name in present})


def cap_curve(row_by_variant):
    """``[(seed, cap, baseline, capped, cost)]`` from fixed-divisor values."""
    rows = []
    for seed in SEEDS:
        base_name = BASELINE[seed]
        if base_name not in row_by_variant:
            continue
        base = row_by_variant[base_name]
        for cap in CAPS:
            name = variant_name(cap, seed)
            if name not in row_by_variant:
                continue
            capped = row_by_variant[name]
            rows.append((seed, cap, base, capped, capped - base))
    return rows


def summarise(rows):
    """Per-cap seed statistics: mean cost, spread, and how many seeds agree."""
    out = {}
    for cap in CAPS:
        costs = [row[4] for row in rows if row[1] == cap]
        if not costs:
            continue
        out[cap] = {
            'seeds': len(costs),
            'costs': [round(value, 4) for value in costs],
            'mean': round(sum(costs) / len(costs), 4),
            'spread': round(max(costs) - min(costs), 4),
            # A cap cost is "seed-stable in sign" when every seed pays the same
            # way; a sign flip means the single-seed curve cannot be quoted.
            'all_negative': all(value < 0 for value in costs),
            'all_positive': all(value > 0 for value in costs),
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',
                        default='benchmarks/results/p1_cap_seed_curve.json')
    parser.add_argument('--eval', default='benchmarks/results/p1_eval.json')
    args = parser.parse_args(argv)

    names = needed_variants()
    values, per_horizon = fixed_divisor_values(names)
    recorded = {row['variant']: (row.get('metrics') or {}).get('avg/mIoU')
                for row in json.loads(
                    (ROOT / args.eval).read_text(encoding='utf-8'))}

    missing = [name for name in names if name not in values]
    rows = cap_curve(values)
    summary = summarise(rows)

    print(f'{len(values)}/{len(names)} variants have a fixed-divisor value; '
          f'missing: {missing or "none"}')
    print()
    print('| seed | cap | uncapped | capped | cost (fixed divisor) |'
          ' recorded cost |')
    print('| --- | ---: | ---: | ---: | ---: | ---: |')
    for seed, cap, base, capped, cost in rows:
        recorded_cost = None
        base_rec = recorded.get(BASELINE[seed])
        cap_rec = recorded.get(variant_name(cap, seed))
        if base_rec is not None and cap_rec is not None:
            recorded_cost = cap_rec - base_rec
        shown = '—' if recorded_cost is None else f'{recorded_cost:+.3f}'
        print(f'| {seed} | {cap} | {base:.3f} | {capped:.3f} | {cost:+.3f} | '
              f'{shown} |')
    print()
    print('| cap | seeds | mean cost | spread | sign stable | costs |')
    print('| ---: | ---: | ---: | ---: | --- | --- |')
    for cap, info in sorted(summary.items()):
        stable = 'yes' if (info['all_negative'] or info['all_positive']) else 'NO'
        print(f"| {cap} | {info['seeds']} | {info['mean']:+.3f} | "
              f"{info['spread']:.3f} | {stable} | {info['costs']} |")

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'script': 'benchmarks/p1/cap_seed_curve.py',
        'baseline': BASELINE,
        'caps': list(CAPS),
        'values': {name: round(value, 4) for name, value in values.items()},
        'per_horizon': {name: [round(v, 3) for v in series]
                        for name, series in per_horizon.items()},
        'rows': [{'seed': seed, 'cap': cap, 'uncapped': base, 'capped': capped,
                  'cost': cost} for seed, cap, base, capped, cost in rows],
        'summary': summary,
        'missing': missing,
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
