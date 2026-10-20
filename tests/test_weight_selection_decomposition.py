"""Tests for the weighting-vs-selection decomposition (P1-6 4.2.1).

P1-6 4.2.1 refutes the "over-smoothing grows with the receptive field" mechanism
with the cap sweep, but leaves unresolved *which* of the two things A5 changes
relative to A2 -- the weighting or the selection -- carries the gap.  The
``a2_knn_gaussian8`` variant separates them; this file pins the logic that turns
the chain of single-factor steps into a verdict, so the moment that variant lands
the answer is computed rather than argued.

Two properties are load-bearing:

* the decomposition is a **chain**, not a 2x2 -- there is no uniform-weight
  shell-cap-8 variant, so the interaction must not be reported as identifiable;
* a step inside the measured seed noise may not be named the "cause".  Ranking
  noise is how a 0.01 mIoU difference becomes a mechanism story.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p1.weight_selection_decomposition import (  # noqa: E402
    MISSING_CELL, STEPS, decomposition_verdict, step_deltas)

BASE = {'a2_knn_mean': 14.375, 'a2_knn_gaussian8': 14.375,
        'a5_cap8': 14.050, 'a5_cap0': 14.317}


def test_step_deltas_are_single_factor_differences():
    deltas = step_deltas(BASE)
    assert set(deltas) == {'weighting', 'selection', 'cap'}
    assert deltas['weighting'] == pytest.approx(0.0, abs=1e-9)
    assert deltas['selection'] == pytest.approx(-0.325, abs=1e-9)
    assert deltas['cap'] == pytest.approx(0.267, abs=1e-9)
    # A missing member removes only the steps that touch it.
    partial = step_deltas({k: v for k, v in BASE.items()
                           if k != 'a2_knn_gaussian8'})
    assert set(partial) == {'cap'}


def test_each_verdict_branch():
    # Weighting is material and selection is not -> weighting carries the gap.
    # (A5 cap=8 is held at the same value as A2-gauss so the selection leg is 0.)
    weighting = dict(BASE, a2_knn_gaussian8=14.05, a5_cap8=14.05)
    assert decomposition_verdict(weighting)[:2] == ('WEIGHTING', 'weighting')
    # Selection is material and weighting is not -> selection carries it.
    selection = dict(BASE, a2_knn_gaussian8=14.375, a5_cap8=14.05)
    assert decomposition_verdict(selection)[:2] == ('SELECTION', 'selection')
    # Both material -> not attributable to one factor.
    both = dict(BASE, a2_knn_gaussian8=14.20, a5_cap8=14.00)
    assert decomposition_verdict(both)[0] == 'BOTH'
    # Nothing clears the noise floor -> the honest answer is "unresolved",
    # NOT a ranking of 0.01-sized differences.
    noise = dict(BASE, a2_knn_gaussian8=14.36, a5_cap8=14.34)
    assert decomposition_verdict(noise)[0] == 'UNRESOLVED'
    # Steps can be large and cancel: the matched-count difference is zero, so
    # there is no A2-vs-A5 difference left to explain.
    cancelling = dict(BASE, a2_knn_gaussian8=14.075, a5_cap8=14.375)
    assert decomposition_verdict(cancelling)[0] == 'CANCELLING'
    # The cap leg is inside A5 and must not be able to name the cause: with
    # weighting/selection both at zero, +0.267 of cap alone is still UNRESOLVED.
    cap_only = dict(BASE, a2_knn_gaussian8=14.375, a5_cap8=14.375)
    assert decomposition_verdict(cap_only)[0] == 'UNRESOLVED'
    # Without the separating variant the verdict cannot be computed at all.
    assert decomposition_verdict({k: v for k, v in BASE.items()
                                  if k != 'a2_knn_gaussian8'})[0] == \
        'INCOMPLETE'


def test_the_chain_is_not_a_factorial():
    """The interaction must stay unidentifiable until the cell is measured."""
    steps = {name for name, _, _, _ in STEPS}
    assert steps == {'weighting', 'selection', 'cap'}
    assert 'uniform' in MISSING_CELL and 'cap 8' in MISSING_CELL


def test_cap_step_reproduces_the_documented_cap8_damage():
    """The one step already measurable must match P1-6's cap=8 numbers.

    P1-6 4.2.1 reports cap=8 costing -0.20 on thin classes and -0.29 on
    structural ones; the chain's cap step is the same difference with the
    opposite sign, and it is re-derived here from the script's artifact.
    """
    path = ROOT / 'benchmarks/results/p1_weight_selection.json'
    if not path.exists():
        pytest.skip('p1_weight_selection.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    steps = payload['steps']
    assert steps['cap']['earlier'] == 'a5_cap8'
    assert steps['cap']['later'] == 'a5_cap0'
    assert steps['cap']['delta_fixed'] == pytest.approx(0.267, abs=0.001)
    rows = {row['variant']: row for row in payload['rows']}
    assert rows['a5_cap0']['thin'] - rows['a5_cap8']['thin'] == pytest.approx(
        0.200, abs=0.001)
    assert (rows['a5_cap0']['structural'] - rows['a5_cap8']['structural']
            == pytest.approx(0.286, abs=0.001))
    # Non-selective damage: the structural classes lose at least as much as the
    # thin ones -- this is the observation that refuted the over-smoothing story.
    assert (rows['a5_cap0']['structural'] - rows['a5_cap8']['structural']) >= \
        (rows['a5_cap0']['thin'] - rows['a5_cap8']['thin'])
    # The verdict must stay INCOMPLETE until the separating variant lands, and
    # must be one of the known values afterwards.
    if 'a2_knn_gaussian8' in payload['missing']:
        assert payload['verdict'] == 'INCOMPLETE'
    else:
        assert payload['verdict'] in {'WEIGHTING', 'SELECTION', 'BOTH',
                                      'UNRESOLVED', 'CANCELLING'}
    # The report's framework and the script must agree on the class groups.
    assert len(payload['rows']) == len(payload['available'])
