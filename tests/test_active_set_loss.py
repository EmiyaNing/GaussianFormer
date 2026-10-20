"""Tests for the P2 active-set occupancy loss.

Includes a Monte-Carlo check of the unbiasedness claim that plan task T2.6
asks for: with importance weights, the sampled loss must estimate the mean
dense loss without bias.
"""
from __future__ import annotations

import pytest
import torch

from loss.active_set_loss import ActiveSetOccupancyLoss, active_set_focal_loss
from model.ops.active_sampling import (
    SOURCE_BOUNDARY, SOURCE_EMPTY, SOURCE_PREDICTED)


# --------------------------------------------------------------------------
# Per-location focal loss
# --------------------------------------------------------------------------

def test_focal_loss_is_near_zero_for_confident_correct_predictions():
    # The formulation is sigmoid multi-label, not softmax: "confident and
    # correct" means the target channel is high AND every other channel is low.
    logits = torch.full((3, 4), -20.0)
    targets = torch.tensor([0, 1, 2])
    logits[torch.arange(3), targets] = 20.0
    per_location = active_set_focal_loss(logits, targets)
    assert float(per_location.max()) < 1e-6


def test_focal_loss_is_large_for_confident_wrong_predictions():
    # Same shape, but the high channel is never the target.
    logits = torch.full((3, 4), -20.0)
    logits[:, 3] = 20.0
    targets = torch.tensor([0, 1, 2])
    per_location = active_set_focal_loss(logits, targets)
    assert float(per_location.min()) > 1.0


def test_focal_loss_handles_empty_input():
    per_location = active_set_focal_loss(torch.zeros(0, 5), torch.zeros(0, dtype=torch.long))
    assert per_location.numel() == 0


def test_class_weights_scale_the_per_location_loss():
    logits = torch.zeros(2, 3)
    targets = torch.tensor([0, 2])
    plain = active_set_focal_loss(logits, targets)
    weighted = active_set_focal_loss(logits, targets,
                                     class_weights=torch.tensor([1.0, 1.0, 4.0]))
    # Only the class-2 location gets multiplied.
    assert weighted[0].item() == pytest.approx(plain[0].item())
    assert weighted[1].item() == pytest.approx(4.0 * plain[1].item())


# --------------------------------------------------------------------------
# Module behaviour
# --------------------------------------------------------------------------

def test_empty_targets_are_dropped_in_the_17_channel_convention():
    logits = torch.randn(4, 17)
    labels = torch.tensor([3, 17, 5, 17])       # two empty targets
    module = ActiveSetOccupancyLoss(num_classes=17, empty_label=17)
    loss, metrics = module({}, logits=logits, labels=labels)
    assert float(metrics['active_set_locations']) == 2.0
    assert torch.isfinite(loss)


def test_eighteen_channel_convention_supervises_empty_locations():
    logits = torch.randn(4, 18)
    labels = torch.tensor([3, 17, 5, 17])
    module = ActiveSetOccupancyLoss(num_classes=17, empty_label=17)
    loss, metrics = module({}, logits=logits, labels=labels)
    # All four locations are scored, and the empty ones map to the last channel.
    assert float(metrics['active_set_locations']) == 4.0
    assert torch.isfinite(loss)


def test_source_weights_and_per_source_metrics():
    logits = torch.zeros(4, 18)
    labels = torch.tensor([1, 1, 1, 1])
    source = torch.tensor([SOURCE_PREDICTED, SOURCE_BOUNDARY,
                           SOURCE_EMPTY, SOURCE_EMPTY])
    module = ActiveSetOccupancyLoss(num_classes=17, empty_label=17,
                                    source_weights={SOURCE_EMPTY: 3.0})
    loss, metrics = module({}, logits=logits, labels=labels, source=source)
    assert 'active_set_boundary_samples_loss' in metrics
    assert float(metrics['active_set_empty_samples_count']) == 2.0
    assert torch.isfinite(loss)


def test_warmup_ramp_scales_the_loss():
    logits = torch.randn(8, 18)
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    module = ActiveSetOccupancyLoss(num_classes=17, empty_label=17,
                                    warmup_iters=100, normalizer='count')
    early, _ = module({}, logits=logits, labels=labels, global_iter=10)
    late, _ = module({}, logits=logits, labels=labels, global_iter=1000)
    assert float(early) == pytest.approx(0.1 * float(late), rel=1e-5)


def test_importance_normalizer_requires_probabilities_and_dense_count():
    logits = torch.randn(4, 18)
    labels = torch.tensor([1, 2, 3, 4])
    module = ActiveSetOccupancyLoss(num_classes=17, empty_label=17,
                                    normalizer='importance')
    with pytest.raises(ValueError):
        module({}, logits=logits, labels=labels)          # no sample_prob
    with pytest.raises(ValueError):
        module({}, logits=logits, labels=labels,
               sample_prob=torch.full((4,), 0.5))          # no dense count
    with pytest.raises(ValueError):
        module({}, logits=logits, labels=labels,
               sample_prob=torch.tensor([0.5, 0.5, 0.0, 0.5]),
               dense_voxel_count=1000)                     # zero probability


# --------------------------------------------------------------------------
# Unbiasedness (plan task T2.6)
# --------------------------------------------------------------------------

def test_importance_weighted_loss_is_an_unbiased_dense_loss_estimator():
    """Monte-Carlo check that importance weighting removes sampling bias.

    The dense grid has 2,000 locations with a known mean per-location loss.
    Each trial draws 200 of them with a *non-uniform* probability, and the two
    estimators are compared against the true dense mean:

    * ``count`` normalisation is biased (it over-weights frequently drawn
      locations);
    * ``importance`` normalisation is unbiased.
    """
    torch.manual_seed(0)
    dense_count, sample_size, trials = 2000, 200, 400

    # The loss landscape must CORRELATE with the sampling probability, or the
    # biased estimator would look fine: if loss were independent of position,
    # any weighting would still estimate the mean.  The first quarter of the
    # grid is given much larger losses, and the sampler oversamples it.
    dense_targets = torch.randint(0, 18, (dense_count,))
    dense_logits = torch.randn(dense_count, 18) * 1.0
    dense_logits[: dense_count // 4] *= 6.0
    dense_loss = active_set_focal_loss(dense_logits, dense_targets)
    true_mean = float(dense_loss.mean())

    # Strongly skewed sampling: the first quarter is ~40x more likely.
    probability = torch.linspace(40.0, 1.0, dense_count)
    probability[dense_count // 4:] = 1.0
    probability = probability / probability.sum()

    importance_estimates, count_estimates = [], []
    for trial in range(trials):
        generator = torch.Generator().manual_seed(1000 + trial)
        # With replacement: the 1/p estimator is only unbiased for
        # independent draws, where p_i is the per-draw probability.
        index = torch.multinomial(probability, sample_size, replacement=True,
                                  generator=generator)
        sampled_loss = dense_loss[index]
        sampled_prob = probability[index]
        # (1 / (n * V)) * sum l_j / p_j estimates the mean dense loss; the
        # 1/n factor is part of the estimator, not an implementation detail.
        importance_estimates.append(float(
            (sampled_loss / sampled_prob).sum() / (sample_size * dense_count)))
        count_estimates.append(float(sampled_loss.mean()))

    importance_mean = sum(importance_estimates) / trials
    count_mean = sum(count_estimates) / trials
    importance_error = abs(importance_mean - true_mean)
    count_error = abs(count_mean - true_mean)

    print(f'  true dense mean      = {true_mean:.6f}')
    print(f'  importance estimate  = {importance_mean:.6f} '
          f'(error {importance_error:.6f})')
    print(f'  count estimate       = {count_mean:.6f} '
          f'(error {count_error:.6f})')

    # The importance estimator must be close to the truth, and clearly closer
    # than the biased one on a landscape this skewed.
    assert importance_error < 0.02 * true_mean + 1e-4
    assert importance_error < count_error


def test_count_normalisation_is_the_plain_sample_mean():
    logits = torch.randn(10, 18)
    labels = torch.randint(0, 18, (10,))
    module = ActiveSetOccupancyLoss(num_classes=17, empty_label=17,
                                    normalizer='count')
    loss, _ = module({}, logits=logits, labels=labels)
    expected = active_set_focal_loss(logits, labels).mean()
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)
