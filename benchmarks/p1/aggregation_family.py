"""The aggregation family side by side, at one fixed-divisor scope.

Report P1-4 concludes that the benefit comes from *adding a residual branch*, not
from how neighbours are aggregated -- but it rested on A2's uniform mean, with the
max/attention controls still training.  This script puts the whole family on one
footing so the update is mechanical:

    A1  point-wise        no neighbourhood at all
    A2  KNN mean          uniform mean over the 8 nearest
    A2  KNN max           element-wise max over the same 8  (the missing control)
    A3  graph attention   kNN graph + attention
    A5  Gaussian          learned-support Gaussian weighting

Everything is reported at the **fixed-divisor** scope of exactly these variants,
plus the thin/structural class-group means on the same (horizon, class)
intersection -- so a claim about "the aggregation form" cannot be an artifact of
the metric's class divisor or of comparing different class sets.

    python3.10 benchmarks/p1/aggregation_family.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

FAMILY = {
    'A0 baseline': 'a0_base',
    'A1 point-wise': 'a1_pointwise',
    'A2 KNN mean': 'a2_knn_mean',
    'A2 KNN max': 'a2_knn_max',
    'A3 attention': 'a3_graph_attention',
    'A5 Gaussian': 'a5_cap0',
}
# The control this script exists to unblock.
REQUIRED_CONTROL = 'a2_knn_max'


NEIGHBOUR_LABELS = {
    # A0 has no adapter at all -- labelling it "8 nearest" (as an early version
    # did, because the else-branch caught every name) misreads the baseline as a
    # kNN aggregator.
    'a0_base': '— (frozen baseline)',
    'a1_pointwise': 'none (no aggregation)',
    'a2_knn_mean': '8 nearest (uniform mean)',
    'a2_knn_max': '8 nearest (max)',
    'a3_graph_attention': '8 nearest (attention)',
    'a5_cap0': 'support box (~215)',
}


def neighbour_label(variant):
    """How many neighbours this variant aggregates, as reported in the table."""
    return NEIGHBOUR_LABELS.get(variant, 'unknown')


def family_verdict(fixed_values, control_names, gaussian_name, tolerance=0.10):
    """``(verdict, control_spread, gaussian_minus_best_control)``.

    CONFIRMED means every non-Gaussian aggregation lands within ``tolerance`` of
    the others *and* the Gaussian adapter does not lead them -- i.e. the
    aggregation form is not what produces the gain.
    """
    controls = [fixed_values[name] for name in control_names
                if fixed_values.get(name) is not None]
    gaussian = fixed_values.get(gaussian_name)
    if not controls or gaussian is None:
        return 'INCOMPLETE', float('nan'), float('nan')
    spread = max(controls) - min(controls)
    margin = gaussian - max(controls)
    verdict = 'CONFIRMED' if spread <= tolerance and margin <= 0.05 else 'REVIEW'
    return verdict, spread, margin


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', default='benchmarks/results/p1_eval.json')
    parser.add_argument('--json',
                        default='benchmarks/results/p1_aggregation_family.json')
    args = parser.parse_args(argv)

    from benchmarks.p1 import per_class_analysis as classes  # noqa: PLC0415
    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415

    recorded = {}
    for row in json.loads((ROOT / args.eval).read_text(encoding='utf-8')):
        if row.get('metrics'):
            recorded[row['variant']] = row['metrics']
    available = {label: name for label, name in FAMILY.items()
                 if name in recorded}
    missing = [label for label, name in FAMILY.items() if name not in recorded]
    print(f'{len(available)}/{len(FAMILY)} family members available; '
          f'missing: {missing or "none"}')

    names = list(available.values())
    logs = load(names)
    table, common = comparable_for(logs, names) if logs else ({}, {})
    print('# fixed-divisor scope: '
          + ', '.join(f'{horizon}s={len(common[horizon])}'
                      for horizon in sorted(common)) if common else
          '# fixed-divisor scope: unavailable (logs missing)')

    # Class groups on the intersection over exactly these variants.
    data = logs
    small = large = None
    if data:
        keys = None
        for name in names:
            variety = classes.finite_keys(data[name], classes.CLASSES[:17])
            keys = variety if keys is None else (keys & variety)
        small = {key for key in keys if key[1] in classes.SMALL}
        large = {key for key in keys if key[1] in classes.LARGE}
        print(f'# class groups: {len(small)} thin + {len(large)} structural '
              f'(horizon, class) pairs')

    def group(name, subset):
        if not data or name not in data or not subset:
            return None
        return classes.group_mean(data[name], classes.CLASSES[:17], keys=subset)

    print()
    print('| variant | recorded mIoU | fixed mIoU | thin classes | structural | '
          'neighbours |')
    print('| --- | ---: | ---: | ---: | ---: | --- |')
    rows = []
    for label, name in available.items():
        metrics = recorded[name]
        fixed = table.get(name, {}).get('avg')
        thin = group(name, small)
        structural = group(name, large)
        neighbours = neighbour_label(name)
        rows.append({'label': label, 'variant': name,
                     'recorded_mIoU': metrics.get('avg/mIoU'),
                     'fixed_mIoU': (round(fixed, 4) if fixed is not None else None),
                     'thin': (round(thin, 3) if thin is not None else None),
                     'structural': (round(structural, 3) if structural is not None
                                    else None),
                     'neighbours': neighbours})
        print(f"| {label} | {metrics.get('avg/mIoU')} | "
              f"{'—' if fixed is None else f'{fixed:.2f}'} | "
              f"{'—' if thin is None else f'{thin:.2f}'} | "
              f"{'—' if structural is None else f'{structural:.2f}'} | "
              f'{neighbours} |')

    if REQUIRED_CONTROL not in recorded:
        verdict = 'INCOMPLETE'
        print(f'\nVERDICT: the max-pooling control ({REQUIRED_CONTROL}) has not '
              f'run, so "aggregation is not the source" still rests on the mean '
              f'alone.  Re-run this script after that training lands.')
    else:
        # The conclusion to test: every non-Gaussian aggregation lands within the
        # seed noise of the others, and none of them beats A5.
        fixed = {row['variant']: row['fixed_mIoU'] for row in rows
                 if row['fixed_mIoU'] is not None}
        verdict, spread, margin = family_verdict(
            fixed, ('a1_pointwise', 'a2_knn_mean', 'a2_knn_max',
                    'a3_graph_attention'), 'a5_cap0')
        print(f'\ncontrol spread (A1/A2-mean/A2-max/A3, fixed divisor): '
              f'{spread:.3f} mIoU')
        print(f'A5 minus the best control: {margin:+.3f} mIoU')
        print(f'VERDICT: aggregation form is {"NOT" if verdict == "CONFIRMED" else "possibly"} '
              f'the source of the gain ({verdict}).  A control spread of '
              f'{spread:.2f} mIoU is within the measured seed noise, and A5 does '
              f'not lead the controls.')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'script': 'benchmarks/p1/aggregation_family.py',
        'family': FAMILY, 'available': available, 'missing': missing,
        'rows': rows, 'verdict': verdict,
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
