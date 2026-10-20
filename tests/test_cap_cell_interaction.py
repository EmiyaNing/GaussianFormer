"""Tests for the cap-vs-cell interaction analysis (P1-10).

P1-1 measures the cap=8 accuracy cost at the trained bin size (1.8 m).  P1-4
2.1.1 shows the *selected* neighbour subset at ``cap>0`` is strongly cell
dependent, so that cost need not transfer to another bin size.  Two things must
therefore hold before a cost may be computed across cells:

* the single cap=0 reference is only valid because the **unbounded** operator was
  measured bitwise identical across cell 0.9/1.8/3.6 (P4-8 8.2).  The script must
  refuse to run if that stops being true -- a mismatched reference would produce
  a plausible-looking number out of two different baselines;
* the verdict must compare the per-cell costs against the **same-weights
  evaluation floor** (0.03, P4-8 8.1), not the training-seed spread, because every
  cap probe runs inference-time on one frozen checkpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p1.cap_cell_interaction import (  # noqa: E402
    EVAL_FLOOR, cell_invariance_holds, interaction_verdict)


def write_robustness(path, cell0p9, cell3p6):
    path.write_text(json.dumps([
        {'variant': 'cell0p9', 'metrics': cell0p9},
        {'variant': 'cell3p6', 'metrics': cell3p6},
    ]), encoding='utf-8')


def test_cell_invariance_licence_is_read_from_the_artifact(tmp_path):
    same = {'avg/mIoU': 14.32, '2s/mIoU': 14.96}
    good = tmp_path / 'good.json'
    write_robustness(good, same, dict(same))
    assert cell_invariance_holds(good) is True

    bad = tmp_path / 'bad.json'
    write_robustness(bad, same, {'avg/mIoU': 14.20, '2s/mIoU': 14.96})
    assert cell_invariance_holds(bad) is False, (
        'a differing cell pair must invalidate the shared cap=0 reference')

    assert cell_invariance_holds(tmp_path / 'absent.json') is None


def test_verdict_branches():
    tol = EVAL_FLOOR
    assert tol == pytest.approx(0.03)
    # Costs that agree within the floor transfer -> cap is the property.
    verdict, spread, worst = interaction_verdict(
        {0.9: 0.26, 1.8: 0.27, 3.6: 0.28}, tol)
    assert verdict == 'CAP_PROPERTY'
    assert spread == pytest.approx(0.02, abs=1e-9)
    assert worst == 3.6
    # A cost that moves by more than the floor is a (cap, cell) statement.
    verdict, spread, worst = interaction_verdict(
        {0.9: 0.10, 1.8: 0.27, 3.6: 0.28}, tol)
    assert verdict == 'CELL_DEPENDENT'
    assert spread == pytest.approx(0.18, abs=1e-9)
    assert worst == 3.6, 'the largest cost, not the largest deviation'
    # Fewer than two measured cells cannot support a verdict.
    assert interaction_verdict({1.8: 0.27}, tol)[0] == 'INCOMPLETE'
    assert interaction_verdict({0.9: None, 1.8: None}, tol)[0] == 'INCOMPLETE'


def test_it_refuses_when_the_reference_is_not_cell_invariant(tmp_path, monkeypatch):
    """A mismatched reference must abort, not silently produce a cost."""
    from benchmarks.p1 import cap_cell_interaction as mod

    bad = tmp_path / 'robustness.json'
    write_robustness(bad, {'avg/mIoU': 14.32}, {'avg/mIoU': 14.20})
    monkeypatch.setattr(mod, 'CELL_INVARIANCE_ARTIFACT', str(bad))
    with pytest.raises(SystemExit, match='not valid across bin sizes'):
        mod.main(['--json', str(tmp_path / 'out.json')])


def test_delivered_artifact_records_the_licence_and_the_cap_cost():
    path = ROOT / 'benchmarks/results/p1_cap_cell.json'
    if not path.exists():
        pytest.skip('p1_cap_cell.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['licence_artifact'] == 'benchmarks/results/p4_robustness.json'
    assert payload['cell_invariance_licence'] is True
    assert payload['tolerance'] == pytest.approx(EVAL_FLOOR)
    by_cell = {row['cell']: row for row in payload['rows']}
    # The trained cell is the only arm with data so far, and its cost must equal
    # P1-1's documented -0.267 mIoU (opposite sign: this is cap=0 minus cap=8).
    assert by_cell[1.8]['cap_cost'] == pytest.approx(0.267, abs=0.001)
    assert by_cell[1.8]['thin_cost'] == pytest.approx(0.200, abs=0.001)
    assert by_cell[1.8]['structural_cost'] == pytest.approx(0.286, abs=0.001)
    # Non-selective damage: structural classes lose at least as much.
    assert by_cell[1.8]['structural_cost'] >= by_cell[1.8]['thin_cost']
    if payload['verdict'] == 'INCOMPLETE':
        assert len([r for r in payload['rows'] if r['cap_cost'] is not None]) < 2
    else:
        assert payload['verdict'] in {'CAP_PROPERTY', 'CELL_DEPENDENT'}


def test_two_independent_scripts_agree_on_the_cap_cost():
    """``p1_cap_cell.json`` and ``p1_weight_selection.json`` must agree.

    The cap=8 cost is computed twice through different code paths: as the final
    step of the weighting/selection chain, and as ``cap=0 - cap=8`` in the
    interaction script.  They rest on the same eval logs but on different class
    intersections and different scope bookkeeping, so a disagreement would mean
    one of them is comparing the wrong pair.
    """
    chain_path = ROOT / 'benchmarks/results/p1_weight_selection.json'
    cell_path = ROOT / 'benchmarks/results/p1_cap_cell.json'
    if not chain_path.exists() or not cell_path.exists():
        pytest.skip('one of the two cap artifacts is missing')
    chain = json.loads(chain_path.read_text(encoding='utf-8'))
    cell = json.loads(cell_path.read_text(encoding='utf-8'))
    step = chain['steps']['cap']
    row = {r['cell']: r for r in cell['rows']}[1.8]
    assert step['delta_fixed'] == pytest.approx(row['cap_cost'], abs=5e-4)
    chain_rows = {r['variant']: r for r in chain['rows']}
    assert chain_rows['a5_cap0']['thin'] - chain_rows['a5_cap8']['thin'] == \
        pytest.approx(row['thin_cost'], abs=5e-4)
    assert (chain_rows['a5_cap0']['structural']
            - chain_rows['a5_cap8']['structural']) == pytest.approx(
                row['structural_cost'], abs=5e-4)
