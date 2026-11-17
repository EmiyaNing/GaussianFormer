"""Per-class analysis: is there a small-object gain the macro mIoU hides?

The parent analysis' gate 3 asks for ">= 0.5 mIoU" **or** "a clear boundary /
small-object gain".  P1 measured the macro mIoU (and P4-7 the boundary IoU); the
per-class breakdown was never analysed beyond the divisor artifact, even though
every eval log carries 18 per-class IoUs at four horizons.

Two reasons this is the better lens for that gate:

* **Per-class IoU is immune to the ``nanmean`` divisor artifact** (P1-5).  Each
  class's IoU is computed independently, so a class being absent in one run does
  not rescale any other class.  The macro mean is exactly where that artifact
  lives.
* The macro mean averages 17 classes of wildly different support, so a gain
  concentrated in thin classes (bicycle, pedestrian, traffic cone) is diluted by
  the structural ones.

The grouping is a judgement call, so it is declared explicitly and the raw
per-class numbers are printed for every horizon; the grouped means are a
summary of those, not a substitute.

    python3.10 benchmarks/p1/per_class_analysis.py
    python3.10 benchmarks/p1/per_class_analysis.py --groups
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402
from benchmarks.p1.common_class_miou import CLASSES, HORIZONS, load  # noqa: E402

# Thin / small-support classes: the ones a neighbourhood operator would have to
# help if "small-object gain" is to mean anything.  Declared, not derived.
SMALL = ['bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
# Structural classes: large area, dominated by flat surfaces and vegetation.
LARGE = ['car', 'truck', 'bus', 'construction_vehicle', 'trailer',
         'driveable_surface', 'sidewalk', 'terrain', 'manmade', 'vegetation',
         'others', 'other_flat']
# 'free' (index 17) is excluded by the official metric; assert the split covers it.

# The decisive pair plus the references.  Seed 42 uses the original runs.
VARIANTS = {
    'A0 baseline': 'a0_base',
    'A1 point-wise': 'a1_pointwise',
    'A2 KNN-8 mean (s42)': 'a2_knn_mean',
    'A2 KNN-8 mean (s43)': 'a2_s43',
    'A2 KNN-8 mean (s44)': 'a2_s44',
    'A3 attention': 'a3_graph_attention',
    'A5 Gaussian (s42)': 'a5_cap0',
    'A5 Gaussian (s43)': 'a5_s43',
    'A5 Gaussian (s44)': 'a5_s44',
}


def finite_keys(values, names):
    """The (horizon, class) pairs that are finite for this variant."""
    return {(h, name) for name in names for h in HORIZONS
            if values.get(f'{h}s/{name}') == values.get(f'{h}s/{name}')}


def group_mean(values, names, keys=None):
    """Mean IoU over a FIXED (horizon, class) set.

    Passing the intersection over all compared variants is mandatory, not
    optional: ``motorcycle`` is NaN in several runs, so averaging whatever is
    finite per variant would compare different class sets and manufacture a
    difference -- the same failure mode as the ``nanmean`` divisor artifact in
    P1-5, one level up.
    """
    keys = keys if keys is not None else finite_keys(values, names)
    keys = {k for k in keys if k[1] in names}
    if not keys:
        return float('nan')
    return sum(values[f'{h}s/{name}'] for h, name in sorted(keys)) / len(keys)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', action='store_true',
                        help='print the per-class breakdown for every horizon')
    parser.add_argument('--json', default='benchmarks/results/p1_per_class.json')
    args = parser.parse_args(argv)

    assert set(SMALL) | set(LARGE) <= set(CLASSES[:17]), 'class split must cover the scored 17'
    assert not (set(SMALL) & set(LARGE)), 'the two groups must be disjoint'
    missing = set(CLASSES[:17]) - set(SMALL) - set(LARGE)
    assert not missing, f'classes left out of both groups: {sorted(missing)}'

    data = load(list(VARIANTS.values()))
    present = [name for name in VARIANTS.values() if name in data]
    # Fixed evaluation sets: the intersection over every compared variant.
    small_keys = set.intersection(*(finite_keys(data[n], SMALL) for n in present))
    large_keys = set.intersection(*(finite_keys(data[n], LARGE) for n in present))
    all_keys = set.intersection(*(finite_keys(data[n], CLASSES[:17]) for n in present))
    print(f'fixed evaluation sets (finite in ALL {len(present)} variants): '
          f'small {len(small_keys)} pairs, structural {len(large_keys)}, '
          f'all-17 {len(all_keys)} of {len(HORIZONS) * 17}')
    small_missing = sorted({n for n in SMALL} - {k[1] for k in small_keys})
    if small_missing:
        print(f'  dropped from the small-object group (never finite in all runs): '
              f'{small_missing}')
    official = {row['variant']: (row.get('metrics') or {}).get('avg/mIoU')
                for row in json.loads(
                    (ROOT / 'benchmarks/results/p1_eval.json').read_text(encoding='utf-8'))}
    rows = {}
    print('| variant | small-object mean IoU | structural mean IoU | '
          'official avg/mIoU |')
    print('| --- | ---: | ---: | ---: |')
    for label, variant in VARIANTS.items():
        if variant not in data:
            print(f'| {label} | (missing) | | |')
            continue
        small = group_mean(data[variant], SMALL, small_keys)
        large = group_mean(data[variant], LARGE, large_keys)
        # The official macro value is quoted from the recorded row, NOT
        # recomputed here: a third estimator with the same label would be exactly
        # the kind of drift this report set keeps auditing for.
        recorded = official.get(variant)
        rows[variant] = {'small': small, 'large': large,
                         'official_avg_mIoU': recorded, 'label': label}
        shown = 'missing' if recorded is None else f'{recorded:.2f}'
        print(f'| {label} | {small:.2f} | {large:.2f} | {shown} |')

    if args.groups:
        for variant in ('a5_cap0', 'a2_knn_mean'):
            if variant not in data:
                continue
            print(f'\nper-class IoU, {variant}:')
            print('| class | ' + ' | '.join(f'{h}s' for h in HORIZONS) + ' |')
            print('| --- | ' + ' | '.join('---:' for _ in HORIZONS) + ' |')
            for name in CLASSES[:17]:
                values = [data[variant].get(f'{h}s/{name}') for h in HORIZONS]
                # A class absent from every target is NaN and the metric skips it.
                cells = ' | '.join(
                    'NaN' if v is None or v != v else f'{v:.2f}' for v in values)
                marker = ' **small**' if name in SMALL else ''
                print(f'| {name}{marker} | {cells} |')

    # Paired per-group differences, seed by seed: the whole point is whether any
    # group shows an A5 advantage that survives every seed.  The pair list is
    # data-driven -- hardcoding 42/43 made the verdict below keep saying "in both
    # seeds" after seed 44 had landed.
    candidate_pairs = [('seed 42', 'a5_cap0', 'a2_knn_mean'),
                       ('seed 43', 'a5_s43', 'a2_s43'),
                       ('seed 44', 'a5_s44', 'a2_s44')]
    pairs = [pair for pair in candidate_pairs if pair[1] in data and pair[2] in data]
    payload = {'groups': {'small': SMALL, 'large': LARGE},
               'fixed_sets': {'small_pairs': sorted(f'{h}s/{n}' for h, n in small_keys),
                              'large_pairs': sorted(f'{h}s/{n}' for h, n in large_keys)},
               'rows': rows, 'paired': {}}
    print('\npaired A5 - A2, by group and seed (positive = A5 better):')
    print('| seed | small-object delta | structural delta | all-17 pooled delta |')
    print('| --- | ---: | ---: | ---: |')
    for label, a5, a2 in pairs:
        if a5 not in data or a2 not in data:
            print(f'| {label} | (missing) | | |')
            continue
        deltas = {
            'small': (group_mean(data[a5], SMALL, small_keys)
                      - group_mean(data[a2], SMALL, small_keys)),
            'large': (group_mean(data[a5], LARGE, large_keys)
                      - group_mean(data[a2], LARGE, large_keys)),
            'all17': (group_mean(data[a5], CLASSES[:17], all_keys)
                      - group_mean(data[a2], CLASSES[:17], all_keys)),
        }
        payload['paired'][label] = deltas
        print(f"| {label} | {deltas['small']:+.2f} | {deltas['large']:+.2f} | "
              f"{deltas['all17']:+.2f} |")

    # Verdict: a small-object gain is only credible if the sign repeats across
    # EVERY available seed.  The wording is derived from the pairs actually used,
    # so it cannot go stale the way "both seeds" did.
    if len(payload['paired']) >= 2:
        small = [value['small'] for value in payload['paired'].values()]
        rendered = ', '.join(f'{value:+.2f}' for value in small)
        count = len(small)
        if all(value > 0 for value in small):
            print(f'\nVERDICT: A5 is better on small objects in all {count} seeds '
                  f'({rendered}) -> a small-object gain is plausible and is the '
                  f'signal the macro mean hides.')
        elif all(value < 0 for value in small):
            print(f'\nVERDICT: A5 is WORSE on small objects in all {count} seeds '
                  f'({rendered}) -> no small-object gain; the macro mean is not '
                  'hiding one.')
        else:
            print(f'\nVERDICT: the small-object delta changes sign across the '
                  f'{count} seeds ({rendered}) -> no resolvable small-object '
                  'effect, matching the macro-mIoU verdict.')
        payload['small_object_sign_repeats'] = bool(
            all(value > 0 for value in small) or all(value < 0 for value in small))
        payload['small_object_seeds'] = count

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')


if __name__ == '__main__':
    main()
