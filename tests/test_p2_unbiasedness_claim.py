"""Regression tests for P2-prep's unbiasedness claim.

``P2-prep``'s headline methodological result was quoted only in prose: the
importance-weighted active-set loss is unbiased (0.1% error) while the plain
sample mean is biased by ~157% when the loss correlates with the sampling
probability.  Nothing produced or pinned those numbers, so the claim could go
stale silently.

These tests run the reproduction in ``benchmarks/p2/active_set_unbiasedness.py``
and assert the **statistical** content rather than the quoted digits: a single
Monte Carlo run returns a realisation of the bias, so the right criterion is
``|bias| < 3 * standard error`` (which fails for a genuinely biased estimator and
passes for an unbiased one whatever the seed).

Kept small on purpose -- the full-size run is a few seconds, and the property
being checked does not need 2,000 locations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p2.active_set_unbiasedness import (  # noqa: E402
    build_problem, reported_setup)


def test_importance_weighting_is_unbiased_through_the_real_loss_module():
    """The module's importance path must be unbiased in expectation.

    This is the property that makes T2.6's "sampled loss estimates the dense
    loss" claim true, so it is checked through ``ActiveSetOccupancyLoss`` rather
    than through a re-implementation of the formula.
    """
    from loss.active_set_loss import ActiveSetOccupancyLoss, active_set_focal_loss
    import torch

    logits, labels, high = build_problem(locations=600, seed=0)
    dense = ActiveSetOccupancyLoss(num_classes=17, normalizer='count',
                                   warmup_iters=0)
    weighted = ActiveSetOccupancyLoss(num_classes=17, normalizer='importance',
                                      warmup_iters=0)
    truth = float(dense(None, logits=logits, labels=labels)[0])
    assert truth > 0
    # The setup is only meaningful if the loss really is larger in the region the
    # sampler oversamples; otherwise any weighting looks unbiased.
    per_location = active_set_focal_loss(logits, labels)
    assert float(per_location[:high].mean()) > float(per_location[high:].mean())

    probability = torch.ones(600)
    probability[:high] = 40.0
    probability = probability / probability.sum()
    generator = torch.Generator().manual_seed(7)
    estimates = []
    for _ in range(120):
        index = torch.multinomial(probability, 50, replacement=True,
                                  generator=generator)
        estimates.append(float(weighted(
            None, logits=logits[index], labels=labels[index],
            sample_prob=probability[index], dense_voxel_count=600)[0]))
    tensor = torch.tensor(estimates)
    bias = float(tensor.mean()) - truth
    stderr = float(tensor.std(unbiased=True) / len(estimates) ** 0.5)
    assert abs(bias) <= 3 * stderr, (
        f'importance-weighted estimator looks biased: bias {bias:.6f} vs '
        f'3 SE {3 * stderr:.6f}')


def test_plain_sample_mean_is_strongly_biased_in_the_same_setup():
    """The contrast case: without weighting the estimate is inflated."""
    from loss.active_set_loss import ActiveSetOccupancyLoss
    import torch

    logits, labels, high = build_problem(locations=600, seed=0)
    dense = ActiveSetOccupancyLoss(num_classes=17, normalizer='count',
                                   warmup_iters=0)
    truth = float(dense(None, logits=logits, labels=labels)[0])
    probability = torch.ones(600)
    probability[:high] = 40.0
    probability = probability / probability.sum()
    generator = torch.Generator().manual_seed(7)
    estimates = []
    for _ in range(120):
        index = torch.multinomial(probability, 50, replacement=True,
                                  generator=generator)
        estimates.append(float(dense(
            None, logits=logits[index], labels=labels[index])[0]))
    mean = float(torch.tensor(estimates).mean())
    assert mean - truth > truth, (
        f'expected a >100% upward bias, got {mean - truth:.4f} on truth {truth:.4f}')


def test_reported_setup_matches_the_documented_statistics():
    """P2-prep's constructed-loss experiment, x6 contrast and 40x sampling odds."""
    result = reported_setup(locations=800, samples=100, repetitions=150)
    i_mean, i_bias, i_se = result['importance']
    c_mean, c_bias, c_se = result['count']
    # Unbiased: the bias must be inside a few standard errors.
    assert abs(i_bias) <= 3 * i_se, (
        f'importance bias {i_bias:.4f} exceeds 3 SE {3 * i_se:.4f}')
    # Biased: the plain mean must be inflated by more than the true value.
    assert c_bias > result['truth']
    # The Monte Carlo result must agree with the analytic expectation of the
    # plain mean under the sampling distribution, which is the strongest check
    # that the sampling itself is implemented as documented.
    analytic = result['analytical_count_mean']
    assert abs(c_mean - analytic) <= 4 * c_se + 0.05 * analytic
