"""Aggregate the multi-seed repeats of the decisive P1 pair (A5 vs A2).

Plan reference: ``plans/sparse_render_plans/P1_frozen_backbone_semantic_validation.md``.

The single-seed control matrix put A5 Gaussian at 14.32 and A2 KNN-mean at 14.37
mIoU — a 0.05 gap on 113 samples, which is far below any plausible noise floor.
The plan therefore requires >= 3 seeds with mean +- std before the A5-vs-A2
comparison can be called either way.  This script reads the recorded rows of
``benchmarks/results/p1_eval.json`` and produces that aggregate.

It refuses to invent numbers: variants whose run has not finished are reported as
MISSING and no mean/std is printed for an incomplete group, so a partial chain
can never be mistaken for a completed multi-seed result.

    python3.10 benchmarks/p1/summarize_seeds.py
    python3.10 benchmarks/p1/summarize_seeds.py --json benchmarks/results/p1_seeds.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _display(path):
    """Path for messages: repo-relative when possible, absolute otherwise."""
    try:
        return Path(path).relative_to(ROOT)
    except ValueError:
        return Path(path)
sys.path.insert(0, str(ROOT))

# Seed-42 rows reuse the original single-seed runs; 43/44 are the new repeats.
# Task seed and torch seed are both set to this value by run_p1_ablation.sh, so
# one number identifies a repeat.
GROUPS = {
    'A5 Gaussian': {42: 'a5_cap0', 43: 'a5_s43', 44: 'a5_s44'},
    'A2 KNN-mean': {42: 'a2_knn_mean', 43: 'a2_s43', 44: 'a2_s44'},
}

# Reported for context; these are single-seed and must not be aggregated.
REFERENCE = {
    'A0 baseline': 'a0_base',
    'A1 point-wise': 'a1_pointwise',
    'A3 attention': 'a3_graph_attention',
}

HORIZONS = (0, 2, 4, 6)
SEEDS = (42, 43, 44)

# Two-sided 95% t multipliers by degrees of freedom, for the seed counts this
# script can plausibly see.  Kept explicit rather than imported so the report's
# arithmetic is auditable without a statistics dependency.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
# Smallest n for which a two-sided sign test can reach p < 0.05 when every pair
# agrees: 2 * (1/2)^n < 0.05.
MIN_SIGN_N = 6


def load_rows(path: Path) -> dict[str, dict]:
    """Metrics per variant, with mIoU recomputed on a **fixed divisor**.

    The recorded ``avg/mIoU`` is ``nanmean`` over 17 classes, so its divisor
    depends on which classes happen to be finite in that particular run.  For
    the seed comparison that is fatal: the A5 seed-43 run has 16 finite classes
    at 6 s where every other run has 15, which alone accounts for a 0.75 mIoU
    swing (see ``P1-5_metric_divisor_artifact.md``).  This loader therefore
    recomputes mIoU over the class set shared by *all* variants in the
    comparison and keeps the recorded value only for display.
    """
    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415

    recorded = {}
    rows = json.loads(path.read_text(encoding='utf-8'))
    for row in rows:
        metrics = row.get('metrics') or {}
        if metrics:
            recorded[row['variant']] = metrics

    needed = sorted({v for variants in GROUPS.values() for v in variants.values()}
                    | set(REFERENCE.values()))
    logs = load(needed)
    if logs:
        table, common = comparable_for(logs, needed)
        for name, values in table.items():
            base = dict(recorded.get(name, {}))
            base['avg/mIoU_recorded'] = base.get('avg/mIoU')
            base['avg/mIoU'] = round(values['avg'], 4)
            for horizon, value in zip(HORIZONS, values['per_horizon']):
                base[f'{horizon}s/mIoU_recorded'] = base.get(f'{horizon}s/mIoU')
                base[f'{horizon}s/mIoU'] = round(value, 4)
            base['finite_classes'] = values['finite_counts']
            base['divisor_fixed'] = True
            recorded[name] = base
        print('# mIoU 已按固定类别集重算（分母不随运行变化）')
        print('# 每个 horizon 的公共类别数: '
              + ', '.join(f'{h}s={len(common[h])}' for h in HORIZONS))
        # A *partial* log set is the dangerous case: the variants that do have a
        # log get fixed-divisor mIoU while the others keep their recorded value,
        # so a paired difference mixes the two conventions.  That is how a
        # simulated seed with no log produced a spurious +0.34 delta here.
        fallback = [name for name in needed
                    if name in recorded and not recorded[name].get('divisor_fixed')]
        if fallback:
            print(f'# ⚠️ 警告：以下 {len(fallback)} 个变体没有逐类日志，仍使用**记录值**'
                  f'（分母与其它变体不可比）：' + ', '.join(fallback))
            print('# ⚠️ 若它们参与配对比较，结论不可用——先补评估日志再重算。')
        print()
    else:
        print('# 警告：找不到逐类日志，退回使用记录值（分母可能不可比）')
        print()
    return recorded


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float('nan')
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def fmt(mean: float, std: float) -> str:
    if math.isnan(std):
        return f'{mean:.2f} (n=1, no std)'
    return f'{mean:.2f} ± {std:.2f}'


def paired_statistics(diffs):
    """Summarise the per-seed paired differences A5 - A2.

    Three separate things are needed to decide whether a direction is real, and
    conflating them is how an n = 3 study gets overclaimed:

    * the **t interval**, because the normal approximation ``2 x SE`` is 2.15x
      too narrow at n = 3 (df = 2);
    * the **sign test**, which only uses the direction and is therefore robust
      to the heavy tails that three mIoU values hide -- and which also shows the
      ceiling of the plan's ">= 3 seeds" rule, since a sign test cannot reach
      p < 0.05 with fewer than 6 agreeing pairs;
    * the **seeds needed** to resolve the observed effect at 2 x SE.

    Hand-computed expectations live in ``tests/test_p1_seed_stats.py``.
    """
    n = len(diffs)
    dmean, dstd = mean_std(diffs)
    se = dstd / math.sqrt(n) if n > 1 else float('nan')
    t_critical = T_975.get(n - 1)
    halves = t_critical * se if t_critical is not None and se == se else float('nan')
    positives = sum(1 for d in diffs if d > 0)
    tail = sum(math.comb(n, k)
               for k in range(max(positives, n - positives), n + 1))
    # max(positives, negatives) already makes this two-sided; doubling is wrong
    # when every pair agrees in the same direction only if the tail is not
    # counted twice, hence the min(1.0, ...) guard.
    sign_p = min(1.0, 2 * tail / 2 ** n)
    needed = None
    if dstd > 0 and dmean != 0:
        needed = math.ceil((2 * dstd / abs(dmean)) ** 2)
    return {'n': n, 'mean': dmean, 'std': dstd, 'se': se,
            't_critical': t_critical, 'ci_low': dmean - halves,
            'ci_high': dmean + halves, 'positives': positives,
            'sign_p_two_sided': sign_p, 'min_sign_n': MIN_SIGN_N,
            'seeds_needed_normal_approx': needed}


def collect(rows: dict[str, dict], group: str, metric: str):
    """Return (values, missing_seeds) for one metric over the three seeds."""
    values, missing = [], []
    for seed in SEEDS:
        variant = GROUPS[group][seed]
        if variant in rows and metric in rows[variant]:
            values.append(rows[variant][metric])
        else:
            missing.append(seed)
    return values, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', default='benchmarks/results/p1_seeds.json')
    parser.add_argument('--eval', default='benchmarks/results/p1_eval.json')
    args = parser.parse_args(argv)

    rows = load_rows(ROOT / args.eval)
    complete = all(v in rows for name in GROUPS for v in GROUPS[name].values())

    print('# Multi-seed aggregate: A5 Gaussian vs A2 KNN-mean')
    print()
    print(f'recorded variants: {len(rows)}')
    for name, variants in GROUPS.items():
        have = [s for s in SEEDS if variants[s] in rows]
        absent = [s for s in SEEDS if variants[s] not in rows]
        status = 'complete' if not absent else f'MISSING seeds {absent}'
        # Only eval rows are printed; a training-only run has no metrics row.
        print(f'  {name:14s} seeds {have} -> {status}')

    payload = {'complete': complete, 'groups': {}}
    print()

    # --- headline metric -------------------------------------------------
    print('| variant | avg/mIoU | avg/IoU | num_samples |')
    print('| --- | --- | --- | --- |')
    for name in GROUPS:
        values, missing = collect(rows, name, 'avg/mIoU')
        ious, _ = collect(rows, name, 'avg/IoU')
        if not values:
            print(f'| {name} | MISSING | MISSING | - |')
            continue
        n = rows[GROUPS[name][SEEDS[0]]].get('num_samples', '?')
        print(f'| {name} | {fmt(*mean_std(values))} | {fmt(*mean_std(ious))} | {n} |')
        payload['groups'][name] = {
            'mIoU': values, 'mIoU_mean': mean_std(values)[0],
            'mIoU_std': mean_std(values)[1],
            'IoU': ious, 'missing_seeds': missing,
        }

    # --- per-horizon stability -------------------------------------------
    # The single-seed story was "worse at 0 s, better at 2/4/6 s".  Averaging
    # over seeds is the only way to tell that pattern from noise.
    print()
    print('| horizon | A5 Gaussian | A2 KNN-mean | delta (A5 - A2) |')
    print('| --- | --- | --- | --- |')
    for horizon in HORIZONS:
        key = f'{horizon}s/mIoU'
        a5, missing5 = collect(rows, 'A5 Gaussian', key)
        a2, missing2 = collect(rows, 'A2 KNN-mean', key)
        row_deltas = []
        for seed in SEEDS:
            v5, v2 = GROUPS['A5 Gaussian'][seed], GROUPS['A2 KNN-mean'][seed]
            if v5 in rows and v2 in rows and key in rows[v5] and key in rows[v2]:
                row_deltas.append(rows[v5][key] - rows[v2][key])
        delta = fmt(*mean_std(row_deltas)) if row_deltas else 'MISSING'
        print(f'| {horizon} s | {fmt(*mean_std(a5)) if a5 else "MISSING"} | '
              f'{fmt(*mean_std(a2)) if a2 else "MISSING"} | {delta} |')
        payload.setdefault('per_horizon', {})[horizon] = {
            'a5': a5, 'a2': a2, 'delta': row_deltas,
            'missing_seeds_a5': missing5, 'missing_seeds_a2': missing2,
        }

    # --- reference row ----------------------------------------------------
    print()
    print('| reference (single seed 42) | avg/mIoU |')
    print('| --- | --- |')
    for label, variant in REFERENCE.items():
        value = rows.get(variant, {}).get('avg/mIoU')
        print(f'| {label} | {"MISSING" if value is None else f"{value:.2f}"} |')

    # --- verdict ----------------------------------------------------------
    # Paired difference is the right statistic: both variants share the exact
    # same eval split and backbone, so seed-to-seed eval noise cancels.
    print()
    diffs = []
    for seed in SEEDS:
        v5, v2 = GROUPS['A5 Gaussian'][seed], GROUPS['A2 KNN-mean'][seed]
        if v5 in rows and v2 in rows:
            diffs.append(rows[v5]['avg/mIoU'] - rows[v2]['avg/mIoU'])
    payload['paired_differences'] = diffs
    if len(diffs) < 2:
        print('VERDICT: cannot decide yet — fewer than 2 paired seeds recorded.')
    else:
        stats = paired_statistics(diffs)
        dmean, dstd, n, se = (stats['mean'], stats['std'], stats['n'], stats['se'])
        print(f'paired delta (A5 - A2) per seed: '
              f'{", ".join(f"{d:+.2f}" for d in diffs)}')
        print(f'paired delta mean {dmean:+.3f}, std {dstd:.3f}, '
              f'standard error {se:.3f} over {n} seeds')

        # A t interval, not 2 x SE: with n = 3 the correct two-sided 95%
        # multiplier is t(0.975, df=2) = 4.303, i.e. 2.15x wider than the normal
        # approximation.  Using 2 x SE here would overstate the resolution.
        if stats['t_critical'] is not None:
            print(f"  two-sided 95% CI (t, df={n - 1}, t={stats['t_critical']}): "
                  f"[{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}]")

        # Sign test: the honest floor for "is the direction real", and also the
        # ceiling of the plan's ">= 3 seeds" requirement.
        print(f"  sign test: {stats['positives']}/{n} seeds have A5 > A2, "
              f"two-sided p = {stats['sign_p_two_sided']:.3f}")
        print(f"  NOTE: a sign test needs n >= {MIN_SIGN_N} pairs for two-sided "
              f"p < 0.05 even if every single seed agrees, so n = 3 can bound "
              f"the noise but cannot establish significance by itself.")
        payload.update(stats)

        if abs(dmean) < 0.05:
            print('VERDICT: the A5-vs-A2 gap is NOT resolvable — the '
                  '"Gaussian beats KNN" direction is unsupported.')
        elif stats['t_critical'] is not None and abs(dmean) <= stats['t_critical'] * se:
            print(f'VERDICT: point estimate {dmean:+.3f} is inside the 95% CI '
                  f'around zero at n = {n}; unsupported.')
        else:
            direction = 'better' if dmean > 0 else 'worse'
            print(f'VERDICT: A5 is {direction} than A2 by {abs(dmean):.2f} mIoU '
                  f'({abs(dmean) / se:.1f} x standard error).')

        # How many seeds would it take?  Using the observed paired SD, the
        # normal approximation n >= (2*SD/effect)^2 (with the t correction this
        # is a lower bound, reported as such).
        if stats['seeds_needed_normal_approx'] is not None:
            print(f"  to resolve |delta| = {abs(dmean):.3f} at 2 x SE with the "
                  f"observed SD {dstd:.3f} would need n >= "
                  f"{stats['seeds_needed_normal_approx']} seeds "
                  f"(normal approximation; a t interval needs more).")

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1), encoding='utf-8')
        print(f'\nwrote {_display(out)}')


if __name__ == '__main__':
    main()
