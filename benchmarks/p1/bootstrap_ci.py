"""Bootstrap confidence intervals from per-sample evaluation dumps.

Three reports (P1-5, P1-6, P4-7) had to state that no confidence interval was
possible, because ``SparseWorldStrictMetric`` accumulated the per-scene
histograms into one aggregate and discarded the detail.  ``DSH_EVAL_PER_SAMPLE``
makes the evaluator write those histograms out, so the eval-side uncertainty can
finally be quantified -- and, crucially, separated from the training-seed
uncertainty that P1-6 measured.

**The resampling unit matters.**  The reduced split has 113 evaluated samples
drawn from only **4 scenes**, and samples inside a scene come from overlapping
temporal windows on the same scene.  Resampling *samples* therefore treats 113
correlated draws as 113 independent ones -- pseudo-replication -- and returns an
interval that is too narrow.  The dumps now carry a ``scene`` label (taken from
``scene_token`` or the camera filename), so this tool resamples **scenes**
(cluster bootstrap, all samples of a drawn scene kept together) as the headline
number, and prints the sample-level interval next to it so the gap between the
two is visible rather than hidden.  With four clusters the cluster interval is
coarse -- that is the honest state of the evidence, not a defect of the method.

What this buys:

* **single-variant CI** -- resample the evaluated *scenes* with replacement and
  report the percentile interval of the mean mIoU.  This is the uncertainty from
  *which scenes happened to be in the split*, not from training.
* **paired difference CI** -- resample scenes *jointly* for two variants, so the
  per-scene component cancels.  If this interval excludes zero for both seeds
  while the sign still flips between seeds, then training randomness -- not the
  evaluation -- is what makes the comparison unresolvable.  That distinction is
  the honest form of the P1 verdict and was previously unavailable.

Per-sample mIoU is recomputed from each sample's 18x18 confusion matrix with the
same ``nanmean(semantic[:17])`` the official metric uses, so the bootstrap mean
matches the reported value (modulo that metric's own divisor caveat, see P1-5).

    python3.10 benchmarks/p1/bootstrap_ci.py \
        --variants a5_cap0=out/p1_eval_a5_cap0/per_sample.json \
                   a2_knn_mean=out/p1_eval_a2_knn_mean/per_sample.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

SCORED = 17          # the official metric averages nanmean(semantic[:17])
FREE = 17


def per_sample_miou(semantic_hist):
    """mIoU for one sample, using the metric's own nanmean-over-17 rule."""
    hist = np.asarray(semantic_hist, dtype=np.float64)
    denominator = hist.sum(0) + hist.sum(1) - np.diag(hist)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / denominator
    return float(np.nanmean(iou[:SCORED]) * 100.0)


def per_sample_occupied(binary_hist):
    hist = np.asarray(binary_hist, dtype=np.float64)
    denominator = hist.sum(0) + hist.sum(1) - np.diag(hist)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / denominator
    return float(iou[1] * 100.0)


def per_sample_occupancy_pair(binary_hist):
    """``(precision, recall)`` for the occupied class, in percent.

    Both come from the same 2x2 binary matrix the dump already stores, so the
    pair separating "misses occupied space" from "hallucinates occupied space"
    gets the same clustered treatment as mIoU -- which matters because the two
    failure modes collapse to one IoU number.  ``NaN`` when the denominator is
    zero (nothing predicted, or nothing occupied in the target).
    """
    hist = np.asarray(binary_hist, dtype=np.float64)
    # The metric builds its 2x2 with ``_hist(pred, target, 2)``, i.e.
    # ``hist[target, prediction]``: row 1 is "actually occupied" and column 1 is
    # "predicted occupied" (which is also why ``per_sample_occupied`` uses
    # ``iou[1]``).  Getting this orientation wrong silently swaps precision and
    # recall, and a symmetric test case would not catch it.
    true_positive = hist[1, 1]
    predicted_positive = hist[:, 1].sum()
    actual_positive = hist[1, :].sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = true_positive / predicted_positive * 100.0 \
            if predicted_positive else float('nan')
        recall = true_positive / actual_positive * 100.0 \
            if actual_positive else float('nan')
    return float(precision), float(recall)


def load_scores(path: Path):
    """Return ``(ids, scenes, mIoU, occupied IoU)``, one entry per sample.

    ``scenes`` is an object array; entries are ``None`` for dumps written before
    the scene label existed, which is why every caller must handle that.
    """
    payload = json.loads(path.read_text(encoding='utf-8'))
    horizons = payload['horizons']
    by_sample = {}
    for entry in payload['per_sample']:
        by_sample.setdefault(entry['sample'], {})[entry['horizon']] = entry
    ids = sorted(by_sample)
    miou, occupied, precision, recall, scenes = [], [], [], [], []
    for sample in ids:
        present = [h for h in horizons if h in by_sample[sample]]
        miou.append(np.mean([per_sample_miou(by_sample[sample][h]['semantic'])
                             for h in present]))
        occupied.append(np.mean([per_sample_occupied(by_sample[sample][h]['binary'])
                                 for h in present]))
        pairs = [per_sample_occupancy_pair(by_sample[sample][h]['binary'])
                 for h in present]
        precision.append(_nanmean([value for value, _ in pairs]))
        recall.append(_nanmean([value for _, value in pairs]))
        labels = {by_sample[sample][h].get('scene') for h in present}
        scenes.append(labels.pop() if len(labels) == 1 else None)
    return (np.array(ids), np.array(scenes, dtype=object),
            np.array(miou), np.array(occupied),
            np.array(precision), np.array(recall))


def _nanmean(values):
    finite = [value for value in values if value == value]
    return float(np.mean(finite)) if finite else float('nan')


def bootstrap_ci(values, repetitions, seed, alpha=0.05):
    """Percentile CI resampling *samples* (anti-conservative when clustered)."""
    generator = np.random.default_rng(seed)
    n = values.size
    draws = values[generator.integers(0, n, size=(repetitions, n))].mean(axis=1)
    return (float(values.mean()),
            float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)),
            float(draws.std(ddof=1)))


def cluster_members(groups):
    """Map each distinct cluster label to the indices belonging to it."""
    members = {}
    for index, label in enumerate(groups):
        members.setdefault(label, []).append(index)
    return {label: np.array(indices) for label, indices in members.items()}


def _draw_clusters(members, labels, repetitions, generator):
    """``[repetitions, n_samples]`` index matrix drawn cluster-wise."""
    count = len(labels)
    draws = np.empty(repetitions, dtype=object)
    for repetition in range(repetitions):
        picked = generator.integers(0, count, size=count)
        draws[repetition] = np.concatenate(
            [members[labels[i]] for i in picked])
    return draws


def cluster_bootstrap_ci(values, groups, repetitions, seed, alpha=0.05):
    """Percentile CI resampling whole *scenes* with replacement."""
    members = cluster_members(groups)
    labels = list(members)
    generator = np.random.default_rng(seed)
    draws = np.array([values[index].mean() for index in
                      _draw_clusters(members, labels, repetitions, generator)])
    return (float(values.mean()),
            float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)),
            float(draws.std(ddof=1)))


def paired_bootstrap_ci(a, b, repetitions, seed, alpha=0.05):
    """Joint sample-level resample; returns the CI of mean(a) - mean(b)."""
    assert a.size == b.size, 'paired bootstrap needs the same samples'
    generator = np.random.default_rng(seed)
    n = a.size
    index = generator.integers(0, n, size=(repetitions, n))
    differences = a[index].mean(axis=1) - b[index].mean(axis=1)
    return _interval(a, b, differences, alpha)


def paired_cluster_bootstrap_ci(a, b, groups, repetitions, seed, alpha=0.05):
    """Joint *scene-level* resample; returns the CI of mean(a) - mean(b).

    Drawing the same scenes for both variants makes the per-scene component
    cancel, which is what lets the paired interval be tighter than either
    single-variant interval without pretending samples are independent.
    """
    assert a.size == b.size, 'paired bootstrap needs the same samples'
    assert len(groups) == a.size, (len(groups), a.size)
    members = cluster_members(groups)
    labels = list(members)
    generator = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for repetition, index in enumerate(
            _draw_clusters(members, labels, repetitions, generator)):
        differences[repetition] = a[index].mean() - b[index].mean()
    return _interval(a, b, differences, alpha)


def _interval(a, b, differences, alpha):
    observed = float(a.mean() - b.mean())
    return (observed, float(np.quantile(differences, alpha / 2)),
            float(np.quantile(differences, 1 - alpha / 2)),
            float(differences.std(ddof=1)))


def _shared(first, second):
    """Restrict two variants to their common sample ids, keeping scene labels."""
    common = np.intersect1d(first['ids'], second['ids'])
    take = [np.searchsorted(first['ids'], common),
            np.searchsorted(second['ids'], common)]
    values = [first['miou'][take[0]], second['miou'][take[1]]]
    groups = first['scenes'][take[0]]
    if not np.array_equal(groups, second['scenes'][take[1]]):
        groups = None
    return common, values, groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', required=True,
                        help='name=path pairs pointing at per-sample dumps')
    parser.add_argument('--repetitions', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', default='benchmarks/results/p1_bootstrap_ci.json')
    args = parser.parse_args(argv)

    loaded = {}
    for spec in args.variants:
        name, _, path = spec.partition('=')
        target = Path(path)
        if not target.is_absolute():
            target = ROOT / path
        if not target.exists():
            print(f'skip {name}: {path} missing')
            continue
        ids, scenes, miou, occupied, precision, recall = load_scores(target)
        labelled = scenes.size and all(isinstance(s, str) for s in scenes)
        entry = {'ids': ids, 'scenes': scenes, 'miou': miou,
                 'occupied': occupied, 'occupied_precision': precision,
                 'occupied_recall': recall}
        if labelled:
            scene_count = len({s for s in scenes})
            mean, low, high, stderr = cluster_bootstrap_ci(
                miou, scenes, args.repetitions, args.seed)
            sample = bootstrap_ci(miou, args.repetitions, args.seed)
            entry['unit'] = 'scene'
            entry['ci'] = {'mean': mean, 'low': low, 'high': high,
                           'stderr': stderr, 'unit': 'scene',
                           'clusters': scene_count}
            entry['ci_sample_level'] = {
                'mean': sample[0], 'low': sample[1], 'high': sample[2],
                'stderr': sample[3], 'unit': 'sample'}
            print(f'{name:16s} samples={miou.size:3d} scenes={scene_count}  '
                  f'mean mIoU {mean:7.3f}  '
                  f'95% CI (scene) [{low:7.3f}, {high:7.3f}] SE {stderr:.3f}  |  '
                  f'(sample) [{sample[1]:7.3f}, {sample[2]:7.3f}] '
                  f'SE {sample[3]:.3f}')
            # Occupied precision/recall get the same clustered treatment: they
            # separate "misses occupied space" from "hallucinates occupied space",
            # which a single IoU number cannot.
            for label, values in (('occupied precision', precision),
                                  ('occupied recall', recall)):
                if not np.isfinite(values).any():
                    continue
                p_mean, p_low, p_high, p_se = cluster_bootstrap_ci(
                    values, scenes, args.repetitions, args.seed)
                entry.setdefault('occupancy_ci', {})[label] = {
                    'mean': p_mean, 'low': p_low, 'high': p_high,
                    'stderr': p_se, 'unit': 'scene'}
                print(f'{"":16s} {label:19s} {p_mean:7.3f}  '
                      f'95% CI (scene) [{p_low:7.3f}, {p_high:7.3f}] '
                      f'SE {p_se:.3f}')
        else:
            mean, low, high, stderr = bootstrap_ci(miou, args.repetitions,
                                                   args.seed)
            entry['unit'] = 'sample'
            entry['ci'] = {'mean': mean, 'low': low, 'high': high,
                           'stderr': stderr, 'unit': 'sample'}
            print(f'{name:16s} samples={miou.size:3d} scenes=?  '
                  f'mean mIoU {mean:7.3f}  95% CI (sample) [{low:7.3f}, '
                  f'{high:7.3f}] SE {stderr:.3f}')
            print('      WARNING: dump carries no scene label, so the interval '
                  'resamples correlated samples (anti-conservative); '
                  're-evaluate with DSH_EVAL_PER_SAMPLE to get scene labels')
        loaded[name] = entry

    names = list(loaded)
    if len(names) >= 2:
        print('\npaired difference (jointly resampled, so the shared component '
              'cancels):')
        base = names[0]
        for other in names[1:]:
            a, b = loaded[base], loaded[other]
            common, values, groups = _shared(a, b)
            if common.size == 0:
                print(f'  {base} vs {other}: no shared samples, skipped')
                continue
            av, bv = values
            if groups is not None and all(isinstance(g, str) for g in groups):
                observed, low, high, stderr = paired_cluster_bootstrap_ci(
                    av, bv, groups, args.repetitions, args.seed)
                unit = 'scene'
            else:
                observed, low, high, stderr = paired_bootstrap_ci(
                    av, bv, args.repetitions, args.seed)
                unit = 'sample'
            excludes_zero = (low > 0) or (high < 0)
            print(f'  {base} - {other}: {observed:+.3f}  '
                  f'95% CI [{low:+.3f}, {high:+.3f}] SE {stderr:.3f}  ({unit})  '
                  f'{common.size} shared samples  '
                  f'{"EXCLUDES 0" if excludes_zero else "includes 0"}')
            loaded[base].setdefault('paired', {})[other] = {
                'observed': observed, 'low': low, 'high': high,
                'stderr': stderr, 'excludes_zero': bool(excludes_zero),
                'shared_samples': int(common.size), 'unit': unit}

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(
        {name: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in value.items()} for name, value in loaded.items()},
        indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')


if __name__ == '__main__':
    main()
