"""Re-derive the P1 mIoU over a class set that is common to every variant.

``SparseWorldStrictMetric.results`` computes ``miou = nanmean(semantic[:17])``.
The NaN filter makes the **divisor depend on the run**: a class that happens to
be absent from the target in every evaluated scene is excluded, while the same
class in another run can land on a handful of voxels, score ~0.00, and be
*included* — adding a zero to the numerator while also growing the denominator.

That is not a rounding detail.  For the two A5 seeds at 6 s the recorded mIoU is
13.28 vs 12.53, a 0.75 gap, yet the per-class IoU values move by at most 0.48.
The whole gap is the divisor:

    seed 42: 'trailer' NaN   -> 15 finite classes, sum 199.21 -> 13.28
    seed 43: 'trailer' 0.00  -> 16 finite classes, sum 200.50 -> 12.53

Since every P1 conclusion lives in the 0.05-0.75 range, a metric whose divisor
changes between runs cannot support them.  This script recomputes the same
numbers over the **intersection** of classes that are finite in *all* variants
at a given horizon, so the divisor is constant by construction.

Input is the recorded per-class IoU in ``out/p1_eval_*/eval.log`` — no GPU, no
re-training, and the raw per-class numbers are printed so the arithmetic is
auditable by hand.

    python3.10 benchmarks/p1/common_class_miou.py
    python3.10 benchmarks/p1/common_class_miou.py --variants a0_base a2_knn_mean a5_cap0 a5_s43
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _display(path):
    """Path for messages: repo-relative when possible, absolute otherwise."""
    try:
        return Path(path).relative_to(ROOT)
    except ValueError:
        return Path(path)

# Order matters: index 17 ('free') is excluded by the official metric's [:17].
CLASSES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
           'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
           'vegetation', 'free']
SCORED = CLASSES[:17]
HORIZONS = (0, 2, 4, 6)
NUMBER = r'([-+]?\d+\.?\d*|nan)'

DEFAULT_VARIANTS = ['a0_base', 'a1_pointwise', 'a2_knn_mean', 'a3_graph_attention',
                    'a5_cap0', 'a5_cap8', 'a5_cap32', 'a5_cap64', 'a5_cap128',
                    'a5_s43', 'a2_s43', 'a5_s44', 'a2_s44', 'a4_fixed_gaussian',
                    'a2_knn_max', 'a2_knn_k215', 'a2_knn_k59']

# Variants that share a training seed, used for the paired comparisons below.
SEED_42 = {'a5': 'a5_cap0', 'a2': 'a2_knn_mean', 'a0': 'a0_base',
           'a1': 'a1_pointwise', 'a3': 'a3_graph_attention'}
SEED_43 = {'a5': 'a5_s43'}


def parse_log(text):
    """Per-class IoU plus the official aggregates, straight from one eval log."""
    out = {}
    for horizon in HORIZONS:
        for name in CLASSES:
            match = re.search(rf"'{horizon}s/{re.escape(name)}':\s*{NUMBER}", text)
            if match:
                raw = match.group(1)
                out[f'{horizon}s/{name}'] = float('nan') if raw == 'nan' else float(raw)
        for key in ('mIoU', 'IoU'):
            match = re.search(rf"'{horizon}s/{key}':\s*{NUMBER}", text)
            if match:
                out[f'{horizon}s/{key}'] = float(match.group(1))
    for key in ('avg/mIoU', 'avg/IoU', 'avg/boundaryIoU', 'num_samples',
                'boundary_samples_per_horizon'):
        match = re.search(rf"'{re.escape(key)}':\s*{NUMBER}", text)
        if match:
            out[key] = float(match.group(1))
    return out


def load(variants):
    data = {}
    for name in variants:
        path = ROOT / 'out' / f'p1_eval_{name}' / 'eval.log'
        if path.exists():
            data[name] = parse_log(path.read_text(encoding='utf-8', errors='replace'))
    return data


def comparable_for(data, names=None):
    """Fixed-divisor mIoU for every variant, on a class set shared by all of them.

    Returns ``(rows, common)`` where ``rows[name]`` has ``per_horizon`` (one value
    per horizon, in ``HORIZONS`` order) and ``avg``, and ``common[horizon]`` is
    the class list used.  The class set is the intersection over the variants
    passed in, so callers that only care about a subset (e.g. the multi-seed A5
    vs A2 pair) get a wider -- and therefore more sensitive -- common set than a
    caller spanning every ablation.

    This exists so downstream aggregation never mixes divisors: consuming the
    recorded ``avg/mIoU`` across runs is exactly the bug this module documents.
    """
    names = list(names) if names else sorted(data)
    names = [n for n in names if n in data]
    if not names:
        raise ValueError('no variants with per-class data')
    common = {}
    for horizon in HORIZONS:
        common[horizon] = [c for c in SCORED
                           if all(data[n].get(f'{horizon}s/{c}', float('nan')) ==
                                  data[n].get(f'{horizon}s/{c}', float('nan'))
                                  for n in names)]
    rows = {}
    for name in names:
        per_horizon = [sum(data[name][f'{h}s/{c}'] for c in common[h]) / len(common[h])
                       for h in HORIZONS]
        rows[name] = {'per_horizon': per_horizon,
                      'avg': sum(per_horizon) / len(HORIZONS),
                      'recorded_avg_mIoU': data[name].get('avg/mIoU'),
                      'finite_counts': {
                          h: sum(1 for c in SCORED
                                 if data[name].get(f'{h}s/{c}', float('nan')) ==
                                 data[name].get(f'{h}s/{c}', float('nan')))
                          for h in HORIZONS}}
    return rows, common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='*', default=None)
    parser.add_argument('--json', default='benchmarks/results/p1_common_class_miou.json')
    args = parser.parse_args(argv)

    names = args.variants or [n for n in DEFAULT_VARIANTS
                              if (ROOT / 'out' / f'p1_eval_{n}' / 'eval.log').exists()]
    data = load(names)
    if not data:
        raise SystemExit('no eval logs found')

    # --- the divisor problem, made explicit ------------------------------
    print('finite-class count per variant and horizon (the divisor of nanmean):')
    print(f"{'variant':20s}" + ''.join(f'{h}s'.rjust(6) for h in HORIZONS)
          + '   avg/mIoU')
    for name in names:
        counts = []
        for horizon in HORIZONS:
            finite = sum(1 for c in SCORED
                         if data[name].get(f'{horizon}s/{c}', float('nan')) ==
                         data[name].get(f'{horizon}s/{c}', float('nan')))
            counts.append(finite)
        print(f'{name:20s}' + ''.join(str(c).rjust(6) for c in counts)
              + f"   {data[name].get('avg/mIoU')}")

    # --- recompute over the class set common to every variant ------------
    payload = {'variants': names, 'horizons': {}}
    comparable_miou = {name: [] for name in names}
    for horizon in HORIZONS:
        common = [c for c in SCORED
                  if all(data[name].get(f'{horizon}s/{c}', float('nan')) ==
                         data[name].get(f'{horizon}s/{c}', float('nan'))
                         for name in names)]
        payload['horizons'][horizon] = {'common_classes': common}
        print(f'\n=== {horizon} s: {len(common)} of 17 classes finite in every '
              f'variant ===')
        print('  ' + ', '.join(common))

    print('\n| variant | recorded mIoU (avg) | comparable mIoU (avg) | difference |')
    print('| --- | ---: | ---: | ---: |')
    rows = {}
    for name in names:
        for horizon in HORIZONS:
            common = [c for c in SCORED
                      if all(data[n].get(f'{horizon}s/{c}', float('nan')) ==
                             data[n].get(f'{horizon}s/{c}', float('nan'))
                             for n in names)]
            mean = sum(data[name][f'{horizon}s/{c}'] for c in common) / len(common)
            comparable_miou[name].append(mean)
        recorded = data[name].get('avg/mIoU')
        comparable = sum(comparable_miou[name]) / len(HORIZONS)
        rows[name] = {'recorded_avg_mIoU': recorded,
                      'comparable_avg_mIoU': round(comparable, 3),
                      'per_horizon_comparable': [round(v, 3) for v in comparable_miou[name]],
                      # The divisor per horizon: without it a reader cannot tell
                      # whether a row was affected by the nanmean artifact.
                      'finite_classes': {
                          str(horizon): sum(
                              1 for c in SCORED
                              if data[name].get(f'{horizon}s/{c}', float('nan')) ==
                              data[name].get(f'{horizon}s/{c}', float('nan')))
                          for horizon in HORIZONS},
                      'per_horizon_recorded': [data[name].get(f'{h}s/mIoU')
                                               for h in HORIZONS]}
        print(f'| {name} | {recorded} | {comparable:.3f} | '
              f'{comparable - recorded:+.3f} |')
    payload['rows'] = rows

    # --- do the paired conclusions survive? ------------------------------
    print('\n=== paired differences, recorded vs comparable (seed 42) ===')
    for label, a, b in (('A5 - A2', 'a5', 'a2'), ('A5 - A0', 'a5', 'a0'),
                        ('A1 - A0', 'a1', 'a0'), ('A2 - A0', 'a2', 'a0')):
        va, vb = SEED_42[a], SEED_42[b]
        if va not in rows or vb not in rows:
            continue
        rec = rows[va]['recorded_avg_mIoU'] - rows[vb]['recorded_avg_mIoU']
        comp = (rows[va]['comparable_avg_mIoU'] - rows[vb]['comparable_avg_mIoU'])
        print(f'  {label}: recorded {rec:+.3f}   comparable {comp:+.3f}')

    # --- the seed spread that motivated this script ----------------------
    #
    # Per horizon, for both arms.  This is the quantity P1-6 quotes, and it has
    # to be taken twice over: the recorded column is what the metric printed
    # (divisor varies run to run), the comparable column is what the same runs
    # say on the common class set.  An earlier version hardcoded the 42-vs-43
    # pair, so it silently ignored seed 44 once that landed.
    seed_groups = {'A5': ['a5_cap0', 'a5_s43', 'a5_s44'],
                   'A2': ['a2_knn_mean', 'a2_s43', 'a2_s44']}
    for label, members in seed_groups.items():
        present_seeds = [name for name in members if name in rows]
        if len(present_seeds) < 2:
            continue
        print(f'\n=== {label} seed spread per horizon: '
              f'{", ".join(present_seeds)} ===')
        for horizon in HORIZONS:
            idx = HORIZONS.index(horizon)
            recorded = [data[name][f'{horizon}s/mIoU'] for name in present_seeds]
            comparable = [rows[name]['per_horizon_comparable'][idx]
                          for name in present_seeds]
            rec_spread = max(recorded) - min(recorded)
            cmp_spread = max(comparable) - min(comparable)
            print(f'  {horizon} s: recorded '
                  f'{min(recorded):6.2f}-{max(recorded):6.2f} '
                  f'(spread {rec_spread:.2f})   comparable '
                  f'{min(comparable):6.2f}-{max(comparable):6.2f} '
                  f'(spread {cmp_spread:.2f})')
        if label == 'A5' and len(present_seeds) >= 2:
            print('  NOTE: the recorded spread is NOT the training-seed noise; '
                  'the comparable spread is.')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print(f'\nwrote {_display(target)}')


if __name__ == '__main__':
    main()
