"""Is the cap=8 accuracy cost a property of ``cap`` or of ``(cap, cell)``?

P1-1 measures the density cap's accuracy cost at ``cell_size = 1.8`` (the trained
value): cap=8 costs -0.267 mIoU, cap=32 only -0.053.  P1-4 2.1.1 then shows that
with ``max_neighbours > 0`` the *selected* subset is strongly cell-dependent --
the operator keeps a shell-ordered subset, and the shell order is decided by the
bin size (at cap=8 only 0.1% of rows select the identical set at cell 0.6 vs
1.8).  So the cost measured at one bin size need not transfer.

This script asks that question directly, using the two frozen-weight probes
``a5_cap8_cell0p9`` / ``a5_cap8_cell3p6``:

    cost(cell) = mIoU(cap=0, cell) - mIoU(cap=8, cell)

Only the cap=8 arms exist at cell 0.9/3.6, so the cap=0 reference is a single
cell-1.8 number -- which is legitimate precisely because P4-8 8.2 measured the
unbounded operator to be **bitwise identical** across cell 0.9/1.8/3.6 on the
same frozen weights (the admitted set does not depend on the bin size).  That
fact is asserted here rather than assumed: if it ever stops holding, the
reference is invalid and this script must refuse to compute a cost.

Tolerance is the **same-weights evaluation floor** (0.03 mIoU, P4-8 8.1), not the
0.05 training-seed spread: all cap probes run inference-time on one frozen
checkpoint, so no retraining variance enters.

    python3.10 benchmarks/p1/cap_cell_interaction.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

UNCAPPED = 'a5_cap0'
CAP8 = 'a5_cap8'
CELL_VARIANTS = {0.9: 'a5_cap8_cell0p9', 1.8: CAP8, 3.6: 'a5_cap8_cell3p6'}
TRAINED_CELL = 1.8
# Same frozen checkpoint, same command shape, inference-time knobs only.
EVAL_FLOOR = 0.03
# The robustness probe's bitwise cell-invariance result (P4-8 8.2).
CELL_INVARIANCE_ARTIFACT = 'benchmarks/results/p4_robustness.json'


def cell_invariance_holds(path=None):
    """Do the two cell variants agree on every reported metric?

    This is the licence for using one cap=0 reference across bin sizes.  A
    ``False`` (or a missing artifact) must stop the analysis rather than quietly
    produce a cost from mismatched references.
    """
    path = Path(path) if path else ROOT / CELL_INVARIANCE_ARTIFACT
    if not path.exists():
        return None
    rows = {row['variant']: row.get('metrics') or {}
            for row in json.loads(path.read_text(encoding='utf-8'))}
    if 'cell0p9' not in rows or 'cell3p6' not in rows:
        return None
    return rows['cell0p9'] == rows['cell3p6']


def interaction_verdict(costs, tolerance=EVAL_FLOOR):
    """``(verdict, spread, worst_cell)`` for ``{cell: cost}``.

    ``CAP_PROPERTY`` means every cell's cost agrees within the evaluation floor,
    so the cost transfers and can be quoted without naming a bin size.
    ``CELL_DEPENDENT`` means it does not: the number P1-1 reports is then a
    statement about the (cap, cell) pair.
    """
    values = {cell: cost for cell, cost in costs.items() if cost is not None}
    if len(values) < 2:
        return 'INCOMPLETE', float('nan'), None
    spread = max(values.values()) - min(values.values())
    worst = max(values, key=values.get)
    verdict = 'CAP_PROPERTY' if spread <= tolerance else 'CELL_DEPENDENT'
    return verdict, spread, worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', default='benchmarks/results/p1_eval.json')
    parser.add_argument('--json',
                        default='benchmarks/results/p1_cap_cell.json')
    parser.add_argument('--tolerance', type=float, default=EVAL_FLOOR)
    args = parser.parse_args(argv)

    from benchmarks.p1 import per_class_analysis as classes  # noqa: PLC0415
    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415

    recorded = {}
    for row in json.loads((ROOT / args.eval).read_text(encoding='utf-8')):
        if row.get('metrics'):
            recorded[row['variant']] = row['metrics']

    invariant = cell_invariance_holds()
    print(f'cell-invariance licence ({CELL_INVARIANCE_ARTIFACT}): '
          f'{invariant}')
    if invariant is None:
        print('WARNING: the cell-invariance artifact is missing, so the single '
              'cap=0 reference cannot be justified.  Costs are printed for '
              'inspection but the verdict is withheld.')
    elif not invariant:
        raise SystemExit('cell variants disagree on the frozen checkpoint; the '
                         'cap=0 reference is not valid across bin sizes')

    names = [name for name in [UNCAPPED, *CELL_VARIANTS.values()]
             if name in recorded]
    missing = [Cell for Cell, name in CELL_VARIANTS.items()
               if name not in recorded]
    if UNCAPPED not in recorded:
        missing.insert(0, 'cap=0 reference')
    print(f'{len(names)}/4 arms available; missing: {missing or "none"}')

    logs = load(names)
    table, common = comparable_for(logs, names) if logs else ({}, {})
    if common:
        print('# fixed-divisor scope: '
              + ', '.join(f'{horizon}s={len(common[horizon])}'
                          for horizon in sorted(common)))

    keys = small = large = None
    if logs:
        for name in names:
            variety = classes.finite_keys(logs[name], classes.CLASSES[:17])
            keys = variety if keys is None else (keys & variety)
        small = {key for key in keys if key[1] in classes.SMALL}
        large = {key for key in keys if key[1] in classes.LARGE}

    def group(name, subset):
        if not logs or name not in logs or not subset:
            return None
        return classes.group_mean(logs[name], classes.CLASSES[:17], keys=subset)

    def fixed(name):
        return table.get(name, {}).get('avg')

    reference = fixed(UNCAPPED)
    uncapped_thin, uncapped_structural = group(UNCAPPED, small), group(
        UNCAPPED, large)

    print()
    print('| cell | cap=8 fixed mIoU | cap cost (cap=0 - cap=8) | thin cost | '
          'structural cost |')
    print('| ---: | ---: | ---: | ---: | ---: |')
    rows, costs = [], {}
    for cell in sorted(CELL_VARIANTS):
        name = CELL_VARIANTS[cell]
        value = fixed(name)
        cost = None if value is None or reference is None else reference - value
        thin = group(name, small)
        structural = group(name, large)
        thin_cost = (None if thin is None or uncapped_thin is None
                     else uncapped_thin - thin)
        structural_cost = (None if structural is None or uncapped_structural is None
                           else uncapped_structural - structural)
        costs[cell] = cost
        rows.append({'cell': cell, 'variant': name,
                     'fixed_mIoU': None if value is None else round(value, 4),
                     'cap_cost': None if cost is None else round(cost, 4),
                     'thin_cost': None if thin_cost is None else round(thin_cost, 3),
                     'structural_cost': (None if structural_cost is None
                                         else round(structural_cost, 3))})
        print(f'| {cell} | {"—" if value is None else f"{value:.2f}"} | '
              f'{"—" if cost is None else f"{cost:+.3f}"} | '
              f'{"—" if thin_cost is None else f"{thin_cost:+.3f}"} | '
              f'{"—" if structural_cost is None else f"{structural_cost:+.3f}"} |')

    verdict, spread, worst = interaction_verdict(costs, args.tolerance)
    if invariant is None:
        verdict = 'INCOMPLETE'
    print()
    if verdict == 'INCOMPLETE':
        print('VERDICT: INCOMPLETE -- either an arm is missing or the '
              'cell-invariance licence is unavailable, so no cost is claimed.')
    else:
        print(f'cap=8 cost spread across bin sizes: {spread:.3f} mIoU '
              f'(tolerance {args.tolerance:.2f} = same-weights evaluation floor)')
        if verdict == 'CAP_PROPERTY':
            print('VERDICT: CAP_PROPERTY -- the cap=8 cost agrees across cell '
                  '0.9/1.8/3.6 within the evaluation floor, so it transfers and '
                  'need not be qualified by the bin size.')
        else:
            print(f'VERDICT: CELL_DEPENDENT -- the cost differs by {spread:.3f} '
                  f'mIoU (largest at cell {worst}), so P1-1\'s number is a '
                  f'statement about the (cap, cell) pair, not about cap alone.')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'script': 'benchmarks/p1/cap_cell_interaction.py',
        'uncapped_reference': UNCAPPED,
        'cell_variants': {str(cell): name for cell, name in CELL_VARIANTS.items()},
        'trained_cell': TRAINED_CELL,
        'cell_invariance_licence': invariant,
        'licence_artifact': CELL_INVARIANCE_ARTIFACT,
        'tolerance': args.tolerance,
        'rows': rows,
        'costs': {str(cell): (None if cost is None else round(cost, 4))
                  for cell, cost in costs.items()},
        'spread': None if spread != spread else round(spread, 4),
        'worst_cell': worst,
        'verdict': verdict,
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
