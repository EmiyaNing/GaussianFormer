"""Decompose the A5 - A2 gap into *weighting* and *selection* (P1-6 4.2.1).

P1-6 4.2.1 shows the "over-smoothing grows with the receptive field" hypothesis
is refuted by the cap sweep: tightening the cap (fewer neighbours) makes the thin
classes *worse*, not better.  What remains unresolved is which of the two things
A5 changes relative to A2 actually matters:

* **weighting** -- A2 averages the 8 nearest uniformly, A5 weights neighbours by
  a learned Gaussian;
* **selection** -- A2 takes the 8 *nearest*, A5 admits everything passing the
  per-axis support box and (at ``cap``) keeps a shell-ordered subset.

The variant ``a2_knn_gaussian8`` (``mode='gaussian'``, ``num_neighbors=8``) is the
cell that separates them: it keeps A2's selection and swaps in Gaussian weights.
This script walks the chain of **single-factor** steps and reports which one
carries the gap, at one fixed-divisor scope and on the same thin/structural class
groups P1-6 uses:

    A2 KNN-8 uniform  --weighting-->  A2-gauss KNN-8  --selection-->  A5 cap=8
                                                                         |
                                                                      cap |
                                                                         v
                                                                   A5 cap=0

It is deliberately a *chain*, not a 2x2: there is no uniform-weight shell-cap-8
variant, so the interaction is **not identifiable** and is not reported as one.

    python3.10 benchmarks/p1/weight_selection_decomposition.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

# name -> (label, what it fixes)
CHAIN = {
    'a2_knn_mean': ('A2 KNN-8 uniform', '8 nearest, uniform mean'),
    'a2_knn_gaussian8': ('A2-gauss KNN-8', '8 nearest, Gaussian weights'),
    'a5_cap8': ('A5 cap=8', 'shell-ordered 8, Gaussian weights'),
    'a5_cap0': ('A5 cap=0', 'support box (~215), Gaussian weights'),
}
ORDER = ['a2_knn_mean', 'a2_knn_gaussian8', 'a5_cap8', 'a5_cap0']

# The single-factor steps between consecutive entries of ORDER.
STEPS = [
    ('weighting', 'a2_knn_mean', 'a2_knn_gaussian8',
     'same 8 nearest: uniform mean -> Gaussian weights'),
    ('selection', 'a2_knn_gaussian8', 'a5_cap8',
     'same Gaussian weights: KNN-8 -> shell-ordered cap 8'),
    ('cap', 'a5_cap8', 'a5_cap0',
     'same Gaussian weights and order: cap 8 -> unbounded'),
]

# The cell a full factorial would need; its absence is why this is a chain.
MISSING_CELL = 'uniform weights x shell-ordered cap 8'


def step_deltas(values):
    """``{step: later - earlier}`` for whatever steps are fully measured."""
    out = {}
    for name, earlier, later, _ in STEPS:
        if values.get(earlier) is None or values.get(later) is None:
            continue
        out[name] = values[later] - values[earlier]
    return out


# The two factors the P1-6 4.2.1 question is about.  The ``cap`` step is *inside*
# A5 (cap 8 -> unbounded), already quantified in P1-1/P1-6, and is reported in the
# table without being allowed to name the cause of the A2-vs-A5 difference: with
# the count matched (A2's 8 nearest vs A5 cap=8) the confound is exactly
# weighting + selection.
ROUTING_FACTORS = ('weighting', 'selection')


def decomposition_verdict(values, tolerance=0.05):
    """``(verdict, dominant, deltas)`` for the weighting-vs-selection question.

    ``tolerance`` is the measured same-configuration noise (P1-6: seed spread
    ~0.05 avg).  A step inside it is not treated as the carrier.  If no routing
    step clears the tolerance the difference is not attributable at this scope,
    which is a result in itself -- the honest verdict is ``UNRESOLVED``, not a
    ranking of noise.
    """
    deltas = step_deltas(values)
    if any(factor not in deltas for factor in ROUTING_FACTORS):
        return 'INCOMPLETE', None, deltas
    material = {step: delta for step, delta in deltas.items()
                if step in ROUTING_FACTORS and abs(delta) > tolerance}
    if not material:
        return 'UNRESOLVED', None, deltas
    dominant = min(material, key=lambda step: material[step])
    verdict = ('WEIGHTING' if dominant == 'weighting' else 'SELECTION') \
        if len(material) == 1 else 'BOTH'
    # The matched-count total (A5 cap=8 - A2) is weighting + selection.  Both legs
    # can be large while cancelling: then there is no difference left to explain.
    matched = deltas['weighting'] + deltas['selection']
    if abs(matched) <= tolerance:
        verdict = 'CANCELLING'
    return verdict, dominant, deltas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', default='benchmarks/results/p1_eval.json')
    parser.add_argument('--json',
                        default='benchmarks/results/p1_weight_selection.json')
    parser.add_argument('--tolerance', type=float, default=0.05)
    args = parser.parse_args(argv)

    from benchmarks.p1 import per_class_analysis as classes  # noqa: PLC0415
    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415

    recorded = {}
    for row in json.loads((ROOT / args.eval).read_text(encoding='utf-8')):
        if row.get('metrics'):
            recorded[row['variant']] = row['metrics']
    available = [name for name in ORDER if name in recorded]
    missing = [name for name in ORDER if name not in recorded]
    print(f'{len(available)}/{len(ORDER)} chain members available; '
          f'missing: {missing or "none"}')

    logs = load(available)
    table, common = comparable_for(logs, available) if logs else ({}, {})
    if common:
        print('# fixed-divisor scope: '
              + ', '.join(f'{horizon}s={len(common[horizon])}'
                          for horizon in sorted(common)))
    else:
        print('# fixed-divisor scope: unavailable (logs missing)')

    small = large = keys = None
    if logs:
        for name in available:
            variety = classes.finite_keys(logs[name], classes.CLASSES[:17])
            keys = variety if keys is None else (keys & variety)
        small = {key for key in keys if key[1] in classes.SMALL}
        large = {key for key in keys if key[1] in classes.LARGE}
        print(f'# class groups: {len(small)} thin + {len(large)} structural '
              f'(horizon, class) pairs')

    def group(name, subset):
        if not logs or name not in logs or not subset:
            return None
        return classes.group_mean(logs[name], classes.CLASSES[:17], keys=subset)

    values, thin_values, structural_values = {}, {}, {}
    rows = []
    for name in available:
        label, fixes = CHAIN[name]
        fixed = table.get(name, {}).get('avg')
        thin, structural = group(name, small), group(name, large)
        values[name] = fixed
        thin_values[name] = thin
        structural_values[name] = structural
        rows.append({'variant': name, 'label': label, 'fixes': fixes,
                     'recorded_mIoU': recorded[name].get('avg/mIoU'),
                     'fixed_mIoU': None if fixed is None else round(fixed, 4),
                     'thin': None if thin is None else round(thin, 3),
                     'structural': (None if structural is None
                                    else round(structural, 3))})

    print()
    print('| step | factor | fixed mIoU | Δ avg | Δ thin | Δ structural |')
    print('| --- | --- | ---: | ---: | ---: | ---: |')
    for name, earlier, later, fixes in STEPS:
        if values.get(earlier) is None or values.get(later) is None:
            print(f'| {earlier} -> {later} | {name} | — | — | — | — |')
            continue
        d = values[later] - values[earlier]
        dt = (None if thin_values.get(earlier) is None
              or thin_values.get(later) is None
              else thin_values[later] - thin_values[earlier])
        ds = (None if structural_values.get(earlier) is None
              or structural_values.get(later) is None
              else structural_values[later] - structural_values[earlier])
        print(f'| {earlier} -> {later} | **{name}** | {values[earlier]:.2f} -> '
              f'{values[later]:.2f} | {d:+.3f} | '
              f'{"—" if dt is None else f"{dt:+.3f}"} | '
              f'{"—" if ds is None else f"{ds:+.3f}"} |')

    verdict, dominant, deltas = decomposition_verdict(values, args.tolerance)
    total = (None if values.get('a2_knn_mean') is None
             or values.get('a5_cap0') is None
             else values['a5_cap0'] - values['a2_knn_mean'])
    print()
    if verdict == 'INCOMPLETE':
        print(f'VERDICT: INCOMPLETE -- the separating variant '
              f'({", ".join(missing)}) has not been evaluated, so weighting and '
              f'selection remain confounded.  Re-run after that training lands.')
    else:
        print(f'A5 - A2 total (fixed divisor): '
              f'{"—" if total is None else f"{total:+.3f}"} mIoU')
        print(f'VERDICT: {verdict}; dominant step = {dominant} '
              f'(tolerance {args.tolerance:.2f} mIoU = measured seed spread).')
        print(f'NOTE: this is a chain, not a 2x2 -- "{MISSING_CELL}" is not '
              f'measured, so the interaction is not identifiable.')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'script': 'benchmarks/p1/weight_selection_decomposition.py',
        'chain': {name: CHAIN[name][0] for name in ORDER},
        'available': available, 'missing': missing,
        'rows': rows,
        'steps': {name: {'earlier': earlier, 'later': later, 'fixes': fixes,
                         'delta_fixed': (None if name not in deltas
                                         else round(deltas[name], 4))}
                  for name, earlier, later, fixes in STEPS},
        'total_fixed': None if total is None else round(total, 4),
        'tolerance': args.tolerance,
        'missing_cell': MISSING_CELL,
        'verdict': verdict, 'dominant': dominant,
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
