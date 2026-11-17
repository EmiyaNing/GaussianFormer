"""Reproduce the active-set loss's unbiasedness claim with the real implementation.

``P2-prep``'s headline methodological result is that the importance-weighted
active-set loss is an unbiased estimator of the dense mean loss (0.1% error) while
the plain sample mean is biased by 157% when the loss correlates with the
sampling probability.  Those numbers appeared **only in prose**: no script
produced them and no test pinned them, so the claim could go stale silently if
``loss/active_set_loss.py`` changed.

This script re-derives them through the real ``ActiveSetOccupancyLoss`` (not a
re-implementation) on the setup the report documents:

* 2,000 dense locations;
* the first quarter carries much larger loss;
* a sampling distribution strongly biased towards that quarter (~40x);
* 200 sampled locations per repetition, 400 repetitions.

The estimator is judged statistically, not by matching digits.  A single Monte
Carlo run gives a *realisation* of the bias, so the meaningful test is

    |bias| < 3 * standard error of the estimate

which fails if the estimator is genuinely biased and passes for an unbiased one
regardless of the seed.  The exact value printed by the original ad-hoc run
(0.0108) is reported for comparison but is not the pass criterion.

    python3.10 benchmarks/p2/active_set_unbiasedness.py
    python3.10 benchmarks/p2/active_set_unbiasedness.py --repetitions 1000
"""
from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

from loss.active_set_loss import (  # noqa: E402
    ActiveSetOccupancyLoss, active_set_focal_loss)

# The value quoted in P2-prep section 3.2, kept for comparison only.
REPORTED_IMPORTANCE_BIAS = 0.0108
REPORTED_COUNT_BIAS = 16.904


def build_problem(locations=2000, high_fraction=0.25, classes=17, seed=0):
    """Engineer logits/labels whose focal loss is much larger in the first quarter.

    Targets are taken correctly in the low-loss region (so the loss is small but
    non-zero) and inverted in the high-loss region, which is what gives the loss
    its correlation with the sampling probability -- without that correlation any
    weighting looks unbiased.
    """
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (locations,), generator=generator)
    high = int(locations * high_fraction)
    # 18 columns so the empty-label convention keeps every location (num_classes
    # + 1 => has_empty_column), which keeps the arithmetic in section 2.3 exact.
    logits = torch.full((locations, classes + 1), -6.0)
    logits[torch.arange(locations), labels] = 6.0
    logits[:high, :] = -6.0
    logits[:high, labels[:high]] = -6.0          # wrong in the high-loss quarter
    return logits, labels, high


def sampling_probability(locations, high, generator, odds=40.0):
    weights = torch.ones(locations)
    weights[:high] = odds
    return weights / weights.sum()


def reported_setup(locations=2000, samples=200, repetitions=400, odds=40.0,
                   contrast=6.0, truth=10.739433, seed=1):
    """Reproduce P2-prep's own experiment: CONSTRUCTED loss values, x6 contrast.

    The report says "构造一个 2,000 个位置的稠密网格，其中前 1/4 的损失显著更大
    （乘以 6 倍）" -- i.e. the loss values were constructed, not produced by the
    module, which is why its dense mean (10.739433) is far above what a focal BCE
    produces (focal losses live near [0, 1.5]).  The two modes are complementary:
    the module mode above shows the *implementation* is unbiased; this mode shows
    the *reported numbers* follow from the documented setup.

    The base loss level is solved from the reported dense mean rather than
    guessed, so a mismatch in the biases is a mismatch in the statistics, not in
    an unknown scale.
    """
    high = int(locations * 0.25)
    low = locations - high
    # truth = (high*contrast*b + low*b) / locations  ->  solve for b
    base = truth * locations / (high * contrast + low)
    values = torch.empty(locations)
    values[:high] = contrast * base
    values[high:] = base

    weights = torch.ones(locations)
    weights[:high] = odds
    probability = weights / weights.sum()

    generator = torch.Generator().manual_seed(seed)
    importance, count = [], []
    for _ in range(repetitions):
        index = torch.multinomial(probability, samples, replacement=True,
                                  generator=generator)
        drawn = values[index]
        prob = probability[index]
        importance.append(float((drawn / prob).sum() / (samples * locations)))
        count.append(float(drawn.mean()))

    def summarise(estimates):
        tensor = torch.tensor(estimates)
        mean = float(tensor.mean())
        stderr = float(tensor.std(unbiased=True) / len(estimates) ** 0.5)
        return mean, mean - float(values.mean()), stderr

    return {
        'base_loss': base, 'truth': float(values.mean()), 'contrast': contrast,
        'importance': summarise(importance), 'count': summarise(count),
        'analytical_count_mean': float((probability * values).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--locations', type=int, default=2000)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--repetitions', type=int, default=400)
    parser.add_argument('--odds', type=float, default=40.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', default='benchmarks/results/p2_unbiasedness.json')
    args = parser.parse_args(argv)

    logits, labels, high = build_problem(args.locations, seed=args.seed)
    dense = ActiveSetOccupancyLoss(num_classes=17, normalizer='count',
                                   warmup_iters=0)
    weighted = ActiveSetOccupancyLoss(num_classes=17, normalizer='importance',
                                      warmup_iters=0)

    # Ground truth: the plain mean over ALL locations (normalizer='count' on the
    # full set is exactly the dense mean).
    # ``forward`` takes ``inputs`` positionally even when the tensors are passed
    # explicitly, so the registry-style call is ``model(None, logits=..., ...)``.
    truth = float(dense(None, logits=logits, labels=labels)[0])
    per_location = active_set_focal_loss(logits, labels)
    high_loss = float(per_location[:high].mean())
    low_loss = float(per_location[high:].mean())

    generator = torch.Generator().manual_seed(args.seed + 1)
    probability = sampling_probability(args.locations, high, generator, args.odds)
    importance, count = [], []
    for _ in range(args.repetitions):
        index = torch.multinomial(probability, args.samples, replacement=True,
                                  generator=generator)
        sampled_logits = logits[index]
        sampled_labels = labels[index]
        prob = probability[index]
        importance.append(float(weighted(
            None, logits=sampled_logits, labels=sampled_labels,
            sample_prob=prob, dense_voxel_count=args.locations)[0]))
        count.append(float(dense(
            None, logits=sampled_logits, labels=sampled_labels)[0]))

    def summarise(values):
        tensor = torch.tensor(values)
        mean = float(tensor.mean())
        # Standard error of the estimator itself, from the repetition spread.
        stderr = float(tensor.std(unbiased=True) / len(values) ** 0.5)
        return mean, mean - truth, stderr

    imp_mean, imp_bias, imp_se = summarise(importance)
    cnt_mean, cnt_bias, cnt_se = summarise(count)

    print(f'true dense mean            : {truth:.6f}')
    print(f'per-location loss contrast : high {high_loss:.4f} vs low {low_loss:.4f} '
          f'({high_loss / low_loss:.1f}x) over the first {high}/{args.locations} locations')
    print(f'sampling odds              : {args.odds:.1f}x on that quarter')
    print()
    print('| estimator | mean over repetitions | bias | std. error | bias/SE |')
    print('| --- | ---: | ---: | ---: | ---: |')
    print(f'| importance-weighted | {imp_mean:.6f} | {imp_bias:+.6f} | {imp_se:.6f} | '
          f'{imp_bias / imp_se if imp_se else float("nan"):+.2f} |')
    print(f'| plain sample mean | {cnt_mean:.6f} | {cnt_bias:+.6f} | {cnt_se:.6f} | '
          f'{cnt_bias / cnt_se if cnt_se else float("nan"):+.2f} |')
    print()
    print(f'P2-prep quoted {REPORTED_IMPORTANCE_BIAS} and {REPORTED_COUNT_BIAS} for '
          f'these two rows; the quantities that must hold are the statistical ones:')
    ok = True
    if abs(imp_bias) > 3 * imp_se:
        print(f'  FAIL: importance-weighted bias {imp_bias:+.6f} exceeds 3 SE '
              f'({3 * imp_se:.6f}) -- the estimator is not unbiased in this setup')
        ok = False
    else:
        print(f'  ok: importance-weighted |bias| {abs(imp_bias):.6f} < 3 SE '
              f'({3 * imp_se:.6f})')
    relative = cnt_bias / truth
    if relative < 1.0:
        print(f'  FAIL: plain mean bias is only {relative:.0%} of the truth; the '
              f'setup no longer demonstrates a large bias')
        ok = False
    else:
        print(f'  ok: plain mean bias {cnt_bias:+.4f} = {relative:.0%} of the truth')

    print('\n--- P2-prep\'s own setup (constructed losses, x6 contrast, '
          'dense mean fixed at the reported value) ---')
    rep = reported_setup(locations=args.locations, samples=args.samples,
                         repetitions=args.repetitions, odds=args.odds)
    i_mean, i_bias, i_se = rep['importance']
    c_mean, c_bias, c_se = rep['count']
    print(f"base loss level           : {rep['base_loss']:.6f} "
          f"(solved so the dense mean is {rep['truth']:.6f})")
    print('| estimator | mean | bias | std. error | bias/SE |')
    print('| --- | ---: | ---: | ---: | ---: |')
    print(f'| importance-weighted | {i_mean:.6f} | {i_bias:+.6f} | {i_se:.6f} | '
          f'{i_bias / i_se:+.2f} |')
    print(f'| plain sample mean | {c_mean:.6f} | {c_bias:+.6f} | {c_se:.6f} | '
          f'{c_bias / c_se:+.2f} |')
    print(f"analytical expectation of the plain mean: "
          f"{rep['analytical_count_mean']:.6f} (bias "
          f"{rep['analytical_count_mean'] - rep['truth']:+.6f})")
    print(f"P2-prep quoted bias {REPORTED_IMPORTANCE_BIAS} (importance, "
          f"{REPORTED_IMPORTANCE_BIAS / rep['truth']:.2%} of truth) and "
          f"{REPORTED_COUNT_BIAS} (plain, "
          f"{REPORTED_COUNT_BIAS / rep['truth']:.0%} of truth); this run gives "
          f"{i_bias / rep['truth']:.2%} and {c_bias / rep['truth']:.0%}.")

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'reported_setup': {
            'base_loss': rep['base_loss'], 'truth': rep['truth'],
            'contrast': rep['contrast'],
            'importance_mean': i_mean, 'importance_bias': i_bias,
            'importance_stderr': i_se,
            'count_mean': c_mean, 'count_bias': c_bias, 'count_stderr': c_se,
            'analytical_count_mean': rep['analytical_count_mean'],
        },
        'locations': args.locations, 'samples': args.samples,
        'repetitions': args.repetitions, 'odds': args.odds,
        'truth': truth, 'loss_contrast': high_loss / low_loss,
        'importance_mean': imp_mean, 'importance_bias': imp_bias,
        'importance_stderr': imp_se,
        'count_mean': cnt_mean, 'count_bias': cnt_bias, 'count_stderr': cnt_se,
        'count_bias_relative': relative,
        'reported_importance_bias': REPORTED_IMPORTANCE_BIAS,
        'reported_count_bias': REPORTED_COUNT_BIAS,
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
