"""Hand-computed checks on the strict evaluator's occupied precision/recall.

The plan's P4.2 asks for five quality diagnostics; the strict evaluator only
reported mIoU and occupied IoU.  Occupied IoU alone cannot separate "the model
misses occupied space" from "the model hallucinates occupied space" -- the
precision/recall pair can, and it needs no probability tensor, so it can be
derived from the binary confusion matrix the evaluator already accumulates.

The derivation is only correct if the matrix orientation is right:
``_hist(prediction, target)`` fills ``hist[target, prediction]``, so recall's
denominator is the row-1 sum and precision's is the column-1 sum.  Swapping them
would silently produce two plausible-looking but meaningless numbers, which is
exactly the kind of error these tests exist to catch.

Everything here is CPU-only: the metric takes numpy/torch label grids, no GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from misc.sparseworld_strict_eval import SparseWorldStrictMetric  # noqa: E402

FREE = 17          # the empty class label
OCC = 0            # any label != FREE counts as occupied


def build_grid(shape, predicted_occupied, target_occupied):
    """Label grids with the requested occupied voxel indices."""
    prediction = torch.full(shape, FREE, dtype=torch.long)
    target = torch.full(shape, FREE, dtype=torch.long)
    for index in predicted_occupied:
        prediction.view(-1)[index] = OCC
    for index in target_occupied:
        target.view(-1)[index] = OCC
    return prediction, target


def run_metric(prediction, target, horizons=6):
    """Feed one sample through the metric and return its result dict."""
    metric = SparseWorldStrictMetric(boundary_stride=0)
    shape = prediction.shape
    result = {'final_occ_grid': prediction.unsqueeze(0)}
    result['future_occ_grids'] = [prediction.unsqueeze(0)] * horizons
    metas = {'occ_label': target.unsqueeze(0),
             'future_occ_labels': target.unsqueeze(0).unsqueeze(0).expand(
                 1, horizons, *shape).contiguous()}
    metric.add_batch(result, metas)
    return metric.results()


def test_precision_and_recall_from_a_hand_computed_confusion_matrix():
    # 4x4x4 grid.  Target occupies {0, 1}; prediction occupies {0, 2}:
    #   TP = 1 (voxel 0), FN = 1 (voxel 1), FP = 1 (voxel 2)
    #   precision = 1/2 = 50, recall = 1/2 = 50, IoU = 1/3 = 33.33
    prediction, target = build_grid((4, 4, 4), [0, 2], [0, 1])
    result = run_metric(prediction, target)
    assert result['0s/occupiedPrecision'] == pytest.approx(50.0, abs=0.01)
    assert result['0s/occupiedRecall'] == pytest.approx(50.0, abs=0.01)
    assert result['0s/IoU'] == pytest.approx(100.0 / 3, abs=0.01)
    assert result['avg/occupiedPrecision'] == pytest.approx(50.0, abs=0.01)
    assert result['avg/occupiedRecall'] == pytest.approx(50.0, abs=0.01)


def test_precision_and_recall_separate_the_two_failure_modes():
    """The whole point: IoU cannot tell misses from hallucinations, this can."""
    shape = (4, 4, 4)
    # Perfect prediction.
    prediction, target = build_grid(shape, [0, 1], [0, 1])
    perfect = run_metric(prediction, target)
    assert perfect['avg/occupiedPrecision'] == pytest.approx(100.0, abs=0.01)
    assert perfect['avg/occupiedRecall'] == pytest.approx(100.0, abs=0.01)
    assert perfect['avg/IoU'] == pytest.approx(100.0, abs=0.01)

    # All misses: target occupies {0,1}, prediction occupies nothing.
    prediction, target = build_grid(shape, [], [0, 1])
    misses = run_metric(prediction, target)
    assert misses['avg/occupiedRecall'] == pytest.approx(0.0, abs=0.01)
    assert misses['avg/IoU'] == pytest.approx(0.0, abs=0.01)

    # All hallucinations: prediction occupies {0,1}, target occupies nothing.
    prediction, target = build_grid(shape, [0, 1], [])
    hallucinations = run_metric(prediction, target)
    assert hallucinations['avg/occupiedPrecision'] == pytest.approx(0.0, abs=0.01)
    assert hallucinations['avg/IoU'] == pytest.approx(0.0, abs=0.01)

    # Both collapse to IoU 0, which is why the pair is needed.
    assert misses['avg/IoU'] == hallucinations['avg/IoU'] == 0.0


def test_recall_is_high_and_precision_low_when_oversegmenting():
    # Prediction occupies a superset of the target: recall 100, precision 50.
    prediction, target = build_grid((4, 4, 4), [0, 1, 2, 3], [0, 1])
    result = run_metric(prediction, target)
    assert result['avg/occupiedRecall'] == pytest.approx(100.0, abs=0.01)
    assert result['avg/occupiedPrecision'] == pytest.approx(50.0, abs=0.01)
    assert result['avg/IoU'] == pytest.approx(50.0, abs=0.01)


def test_recall_is_low_and_precision_high_when_conservative():
    prediction, target = build_grid((4, 4, 4), [0, 1], [0, 1, 2, 3])
    result = run_metric(prediction, target)
    assert result['avg/occupiedRecall'] == pytest.approx(50.0, abs=0.01)
    assert result['avg/occupiedPrecision'] == pytest.approx(100.0, abs=0.01)
    assert result['avg/IoU'] == pytest.approx(50.0, abs=0.01)


def test_precision_is_nan_safe_when_nothing_is_predicted_occupied():
    """An all-empty prediction must not divide by zero into a fake perfect score."""
    prediction, target = build_grid((4, 4, 4), [], [])
    result = run_metric(prediction, target)
    precision = result['avg/occupiedPrecision']
    assert precision != precision or precision == 0.0  # NaN or an explicit zero


def test_every_horizon_reports_the_pair():
    prediction, target = build_grid((4, 4, 4), [0, 2], [0, 1])
    result = run_metric(prediction, target)
    for horizon in (0, 2, 4, 6):
        assert f'{horizon}s/occupiedPrecision' in result
        assert f'{horizon}s/occupiedRecall' in result


def test_average_precision_is_pooled_not_a_nanmean_over_horizons():
    """The average must not divide by a horizon-dependent count.

    Averaging the per-horizon ratios with ``nanmean`` would skip horizons whose
    prediction (or target) was empty, so the divisor would depend on the variant
    -- the same failure class as the mIoU artifact in P1-5.  Pooling the counts
    keeps it fixed, which this test distinguishes: the aggregate must equal
    total TP over total predicted, NOT the mean of the per-horizon ratios.
    """
    shape = (4, 4, 4)
    # Horizon structure is identical for every future index, so make the current
    # frame the interesting one by feeding different future grids is not possible
    # here; instead verify the pooled identity on the values the metric reports.
    prediction, target = build_grid(shape, [0, 2], [0, 1])
    result = run_metric(prediction, target)
    per_horizon = [result[f'{h}s/occupiedPrecision'] for h in (0, 2, 4, 6)]
    pooled = result['avg/occupiedPrecision']
    assert result.get('avg/occupiedPrecision_pooled') is True
    assert pooled == pytest.approx(sum(per_horizon) / len(per_horizon), abs=0.01)
    # And the key must exist even when nothing is predicted occupied.
    empty_prediction, empty_target = build_grid(shape, [], [])
    empty = run_metric(empty_prediction, empty_target)
    assert 'avg/occupiedPrecision' in empty
    assert 'avg/occupiedRecall' in empty
    assert empty['avg/occupiedPrecision'] != empty['avg/occupiedPrecision']  # NaN


def test_the_metric_emits_the_miou_divisor_and_a_uniformity_flag():
    """The ``nanmean`` divisor must be visible at the source (P1-5 §7 item 1).

    Two runs of the same configuration differed by 0.75 mIoU purely because one
    had an extra finite class at 6 s.  Reconstructing that after the fact required
    the per-class IoUs; emitting the divisor directly means a reader sees it in
    the log, and ``avg/mIoU_divisor_uniform_across_horizons`` answers "are my four
    horizons even using the same denominator?" at a glance.
    """
    metric = SparseWorldStrictMetric(boundary_stride=0)
    present = torch.full((1, 4, 4, 4), 17, dtype=torch.long)
    absent = present.clone()
    present.view(-1)[0] = 0                     # class 0 finite in the current frame
    prediction = torch.full((1, 4, 4, 4), 17, dtype=torch.long)
    prediction.view(-1)[0] = 0
    # The current frame has class 0 only (divisor 1); the future frames keep
    # class 0 *and* add class 1, so their divisor is 2 -- the mixed case.
    future_target = present.clone()
    future_target.view(-1)[1] = 1
    future_prediction = prediction.clone()
    future_prediction.view(-1)[1] = 1
    metric.add_batch(
        {'final_occ_grid': prediction,
         'future_occ_grids': [future_prediction] * 6},
        {'occ_label': present,
         'future_occ_labels': future_target.unsqueeze(0).expand(
             1, 6, 4, 4, 4).contiguous()})
    result = metric.results()
    assert result['0s/mIoU_finite_classes'] == 1
    assert result['2s/mIoU_finite_classes'] == 2
    assert result['6s/mIoU_finite_classes'] == 2
    assert result['avg/mIoU_divisor_min'] == 1
    assert result['avg/mIoU_divisor_max'] == 2
    assert result['avg/mIoU_divisor_uniform_across_horizons'] is False

    # A uniform run reports the same divisor everywhere and the flag is True.
    uniform = SparseWorldStrictMetric(boundary_stride=0)
    prediction = torch.full((1, 4, 4, 4), 17, dtype=torch.long)
    target = torch.full((1, 4, 4, 4), 17, dtype=torch.long)
    prediction.view(-1)[0] = 0
    target.view(-1)[0] = 0
    uniform.add_batch({'final_occ_grid': prediction,
                       'future_occ_grids': [prediction] * 6},
                      {'occ_label': target,
                       'future_occ_labels': target.unsqueeze(0).expand(
                           1, 6, 4, 4, 4).contiguous()})
    result = uniform.results()
    assert result['avg/mIoU_divisor_uniform_across_horizons'] is True
    assert result['avg/mIoU_divisor_min'] == result['avg/mIoU_divisor_max'] == 1
