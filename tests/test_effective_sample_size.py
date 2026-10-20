"""Hand-computed tests for the Gaussian effective-sample-size diagnostic.

ESS decides whether A5's large nominal receptive field (215 accepted neighbours)
is really large, which in turn decides whether P1's A5-vs-A2 comparison is
confounded by field size.  The formula is small enough to pin every case by hand:

    ESS = (sum w)^2 / sum w^2
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p4.effective_sample_size import ess_from_weights  # noqa: E402


def test_uniform_weights_recover_the_neighbour_count():
    # ESS must equal k for uniform weights, otherwise it would not be comparable
    # with A2's neighbour count at all.  float64 keeps 1/215 exact enough to
    # assert tightly; float32 rounding alone costs ~1.5e-5 here.
    for k in (2, 8, 215):
        weight = torch.full((1, k), 1.0 / k, dtype=torch.float64)
        assert ess_from_weights(weight).item() == pytest.approx(k, abs=1e-9)


def test_float32_is_accurate_to_float32_precision():
    # The real pipeline is float32, so the slack must be small but non-zero.
    weight = torch.full((1, 215), 1.0 / 215)
    assert ess_from_weights(weight).item() == pytest.approx(215, abs=1e-3)


def test_single_nonzero_weight_gives_one():
    weight = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert ess_from_weights(weight).item() == pytest.approx(1.0, abs=1e-9)


def test_two_equal_weights_give_two_regardless_of_magnitude():
    # ESS is invariant to a global scale factor, so the measurement does not
    # depend on opacity's absolute level.
    for scale in (1e-3, 1.0, 1e3):
        weight = torch.tensor([[scale, scale, 0.0]], dtype=torch.float64)
        assert ess_from_weights(weight).item() == pytest.approx(2.0, abs=1e-9)


def test_hand_computed_three_weight_case():
    # w = [3, 1, 0]: (4)^2 / (9 + 1) = 1.6
    weight = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float64)
    assert ess_from_weights(weight).item() == pytest.approx(1.6, abs=1e-12)


def test_ess_is_bounded_by_the_number_of_nonzero_weights():
    weight = torch.tensor([[0.9, 0.05, 0.03, 0.02, 0.0, 0.0]])
    value = ess_from_weights(weight).item()
    assert 1.0 < value < 4.0


def test_all_zero_row_is_guarded():
    # An empty accepted set would divide zero by zero; the clamp keeps it finite
    # so a single degenerate query cannot poison the mean.
    weight = torch.zeros(1, 4)
    assert torch.isfinite(ess_from_weights(weight)).all()


def test_rows_are_independent():
    weight = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    result = ess_from_weights(weight)
    assert result[0].item() == pytest.approx(2.0, abs=1e-9)
    assert result[1].item() == pytest.approx(1.0, abs=1e-9)
