"""Tests for the receptive-field-matched controls (report P1-8).

P1-3 §5 left a falsifiable prediction for A2'' (k=59, effective-field match) and
the classification has four pre-registered branches.  Getting the mapping wrong
would silently turn "the Gaussian buys field size, not accuracy" into "the
Gaussian is refuted" (or the reverse), so the branches are pinned here -- including
the tolerance boundaries, since the tolerance is the measured seed spread.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p1.receptive_field_controls import classify  # noqa: E402

TOL = 0.054


def test_all_four_preregistered_branches():
    # A2'' indistinguishable from the k=8 control -> the field size is irrelevant.
    assert classify(0.01, -0.06, 0.07, TOL) == 'size-irrelevant'
    # A2'' clearly beats A5 -> a uniform mean over a large field is better.
    assert classify(-0.30, -0.06, 0.30, TOL) == 'uniform-better'
    # A2'' collapses while A5 holds near the k=8 control -> the Gaussian is what
    # survives a large field (the only branch that partially exculpates it).
    assert classify(-0.35, 0.02, -0.37, TOL) == 'gaussian-needed'
    # Anything else needs the k trend.
    assert classify(-0.15, -0.06, -0.09, TOL) == 'intermediate'
    # A missing control never claims a branch.
    assert classify(None, -0.06, None, TOL) == 'INCOMPLETE'


def test_tolerance_is_the_seed_spread_and_boundaries_are_inclusive():
    # Exactly at tolerance counts as "equal" on both sides of every comparison.
    assert classify(TOL, -0.06, 2 * TOL, TOL) == 'size-irrelevant'
    assert classify(2 * TOL, -0.06, 2.1 * TOL, TOL) == 'uniform-better'
    # A gaussian-needed case just inside the boundary.
    assert classify(-3 * TOL, TOL, -4 * TOL, TOL) == 'gaussian-needed'
    # ... and just outside it becomes intermediate.
    assert classify(-3 * TOL, 2 * TOL, -5 * TOL, TOL) == 'intermediate'


def test_artifact_records_the_ceiling_and_the_computable_deltas():
    path = ROOT / 'benchmarks/results/p1_receptive_field_controls.json'
    if not path.exists():
        pytest.skip('receptive-field artifact not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    # The uniform-approximation ceiling comes from the ESS artifact.
    ess = json.loads((ROOT / 'benchmarks/results/p4_ess.json')
                     .read_text(encoding='utf-8'))['rows'][0]
    assert payload['uniform_approximation_gap'] == pytest.approx(
        ess['uniform_approx_error']['64'], abs=1e-9)
    assert payload['uniform_approximation_gap'] == pytest.approx(0.147, abs=0.005)
    # While the controls are missing the verdict must be INCOMPLETE and the
    # A5-vs-k8 delta must still be recorded (it is available and matches P1-6).
    if payload['missing']:
        assert payload['branch'] == 'INCOMPLETE'
        assert payload['deltas']['gaussian_minus_small'] == pytest.approx(
            -0.058, abs=0.005)
