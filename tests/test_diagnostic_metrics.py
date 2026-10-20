"""Unit tests for the P1 diagnostic metrics.

Every case has a hand-computed expected value, so a silently wrong metric
cannot pass.  See ``misc/diagnostic_metrics.py`` for the definitions.
"""
from __future__ import annotations

import math

import pytest
import torch

from misc.diagnostic_metrics import (
    boundary_band, boundary_iou, expected_calibration_error,
    occupancy_precision_recall)


# --------------------------------------------------------------------------
# Expected calibration error
# --------------------------------------------------------------------------

def test_ece_is_zero_for_a_perfectly_calibrated_certain_predictor():
    probabilities = torch.tensor([1.0, 1.0, 0.0, 0.0])
    correct = torch.tensor([1, 1, 0, 0])
    ece, counts = expected_calibration_error(probabilities, correct, num_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)
    assert int(counts.sum()) == 4


def test_ece_matches_hand_computation():
    # Eight predictions at confidence 1.0 of which only four are correct, and
    # eight at confidence 0.0 that are all correct-as-negative.
    probabilities = torch.tensor([1.0] * 8 + [0.0] * 8)
    correct = torch.tensor([1] * 4 + [0] * 4 + [0] * 8)
    ece, _ = expected_calibration_error(probabilities, correct, num_bins=10)
    # Bin at 1.0: confidence 1.0, accuracy 0.5 -> gap 0.5, weight 0.5 -> 0.25
    # Bin at 0.0: confidence 0.0, accuracy 0.0 -> gap 0.0, weight 0.5 -> 0.0
    assert ece == pytest.approx(0.25, abs=1e-6)


def test_ece_is_zero_when_confidence_equals_accuracy_in_every_bin():
    # Confidence 0.5, half correct: perfectly calibrated.
    probabilities = torch.full((100,), 0.5)
    correct = torch.cat([torch.ones(50), torch.zeros(50)])
    ece, _ = expected_calibration_error(probabilities, correct, num_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_ece_handles_empty_input():
    ece, counts = expected_calibration_error(torch.zeros(0), torch.zeros(0))
    assert ece == 0.0
    assert int(counts.sum()) == 0


# --------------------------------------------------------------------------
# Boundary band and boundary IoU
# --------------------------------------------------------------------------

def test_boundary_band_of_a_single_voxel_is_its_full_3x3x3_neighbourhood():
    mask = torch.zeros(1, 1, 5, 5, 5, dtype=torch.bool)
    mask[..., 2, 2, 2] = True
    band = boundary_band(mask, radius=1)
    # dilate = the 3x3x3 cube (27); erode = empty (no voxel has all neighbours
    # set); so the band is the whole cube.  The voxel itself is inside the band
    # because the band is two-sided.
    assert int(band.sum().item()) == 27
    assert bool(band[..., 2, 2, 2])


def test_boundary_band_of_a_solid_block_covers_its_shell():
    mask = torch.zeros(1, 1, 8, 8, 8, dtype=torch.bool)
    mask[..., 2:6, 2:6, 2:6] = True          # 4x4x4 solid block
    band = boundary_band(mask, radius=1)
    # 6x6x6 minus the 2x2x2 core that survives a one-voxel erosion.
    assert int(band.sum().item()) == 6 ** 3 - 2 ** 3
    assert bool(band[..., 2, 2, 2])          # shell voxel
    assert not bool(band[..., 3, 3, 3])      # deep interior


def test_boundary_iou_is_one_for_identical_masks():
    target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    target[..., 2:6, 2:6, 2:6] = 3
    value, _ = boundary_iou(target.clone(), target, num_classes=5, radius=1,
                            ignore_label=0)
    assert value == pytest.approx(100.0, abs=1e-4)


def test_boundary_iou_is_zero_for_disjoint_masks():
    prediction = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    prediction[..., 1:3, 1:3, 1:3] = 2
    target[..., 5:7, 5:7, 5:7] = 2
    # ignore_label=0 keeps the large matching empty region out of the average,
    # which would otherwise hide the disagreement entirely.
    value, _ = boundary_iou(prediction, target, num_classes=3, radius=1,
                            ignore_label=0)
    assert value == pytest.approx(0.0, abs=1e-6)


def test_boundary_iou_ignores_classes_absent_from_the_target():
    target = torch.zeros(1, 1, 6, 6, 6, dtype=torch.long)
    target[..., 2:4, 2:4, 2:4] = 1
    prediction = target.clone()
    prediction[..., 0, 0, 0] = 2          # a class the target never contains
    value, per_class = boundary_iou(prediction, target, num_classes=4,
                                    radius=1, ignore_label=0,
                                    per_class=True)
    # Class 2 is skipped because the target has no such voxel, so the mean is
    # taken over class 1 alone and stays perfect.
    assert value == pytest.approx(100.0, abs=1e-4)
    assert len(per_class) == 1


def test_boundary_iou_is_more_pessimistic_than_voxel_iou_for_a_thin_error():
    """A one-voxel-thick error hurts boundary IoU much more than voxel IoU."""
    target = torch.zeros(1, 1, 10, 10, 10, dtype=torch.long)
    target[..., 3:7, 3:7, 3:7] = 1
    prediction = target.clone()
    prediction[..., 3, 3:7, 3:7] = 0       # carve one face off

    boundary_value, _ = boundary_iou(prediction, target, num_classes=2,
                                     radius=1, ignore_label=0)

    def voxel_iou(pred, tgt):
        intersection = int(((pred == 1) & (tgt == 1)).sum().item())
        union = int(((pred == 1) | (tgt == 1)).sum().item())
        return 100.0 * intersection / union

    assert boundary_value < voxel_iou(prediction, target)


# --------------------------------------------------------------------------
# Occupancy precision / recall
# --------------------------------------------------------------------------

def test_occupancy_precision_recall_hand_computed():
    target = torch.tensor([[[[1, 1], [0, 0]]]])       # two occupied voxels
    prediction = torch.tensor([[[[1, 0], [1, 0]]]])   # TP=1, FP=1, FN=1
    precision, recall = occupancy_precision_recall(
        prediction, target, ignore_label=0)
    assert precision == pytest.approx(50.0)
    assert recall == pytest.approx(50.0)


def test_occupancy_precision_recall_perfect():
    target = torch.tensor([[[[1, 0], [1, 0]]]])
    precision, recall = occupancy_precision_recall(
        target.clone(), target, ignore_label=0)
    assert precision == pytest.approx(100.0)
    assert recall == pytest.approx(100.0)
