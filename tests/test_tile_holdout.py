"""Tests for the three-term tile model and its holdout check (report P4-10).

The kernel cheaply rejects a (voxel, union-entry) pair with a per-axis AABB test
before the full evaluation, so a "scan" has two prices.  Because the accepted
count is exactly ``L`` for every tile (each Gaussian counted once per voxel it
covers), the rejection fraction is exact and grows with the tile edge -- which is
what the two-parameter model's growing over-prediction looks like from the
outside.  These tests pin:

* the rejection fraction identity ``accepted == L`` on the real artifact;
* that the fitted three-term model is an *exact* fit (three unknowns, three
  points) and therefore must not be presented as validation;
* that the holdout script refuses to report a verdict without holdout data, and
  reports errors when the data is there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p4.tile_holdout import (  # noqa: E402
    FIT_TILES, design, fit, fit_two_term, load_rows)


def real_rows():
    rows = load_rows('benchmarks/results/p4_tile_sweep.json')
    if not all(tile in rows for tile in FIT_TILES):
        pytest.skip('tile sweep artifact missing')
    return rows


def test_accepted_scan_count_is_the_tile_independent_L():
    """``accepted == L`` for every tile -- the rejection fraction is exact."""
    rows = real_rows()
    accepted = {rows[tile]['L_voxel_loads'] for tile in FIT_TILES}
    assert len(accepted) == 1, accepted
    for tile in FIT_TILES:
        row = rows[tile]
        assert row['U_tile_loads'] == row['block_scan_iterations'] // (tile ** 3)
        rejected = row['block_scan_iterations'] - row['L_voxel_loads']
        assert rejected > 0, tile
    # The rejection fraction grows with the tile edge, matching the residual shape.
    fractions = [(rows[tile]['block_scan_iterations']
                  - rows[tile]['L_voxel_loads'])
                 / rows[tile]['block_scan_iterations'] for tile in FIT_TILES]
    assert fractions == sorted(fractions), fractions
    assert fractions[0] == pytest.approx(0.260, abs=0.005)
    assert fractions[1] == pytest.approx(0.548, abs=0.005)
    assert fractions[2] == pytest.approx(0.772, abs=0.005)


def test_three_term_fit_is_exact_and_physically_plausible():
    rows = real_rows()
    per_load, per_full, per_reject = fit(rows)
    # Exact by construction: three unknowns, three equations.
    residuals = [abs(float(np.dot(design(rows[tile]),
                                  [per_load, per_full, per_reject]))
                     - rows[tile]['p50_ms']) for tile in FIT_TILES]
    assert max(residuals) < 1e-9, residuals
    # Plausible in sign and ordering: a rejected scan is cheaper than a full one,
    # and both are positive.
    assert 0 < per_reject < per_full
    assert per_reject / per_full == pytest.approx(0.32, abs=0.05)
    assert per_load > per_full, (per_load, per_full)


def test_ignoring_the_rejection_branch_costs_accuracy_on_a_heldout_style_fit():
    """A two-term fit has a systematic, tile-growing error.

    Fitting ``U*l + L*s`` on 2/8 (rejection term forced to zero) and predicting
    the middle tile is the cheapest available check of whether that term earns
    its place; on the real artifacts it misses by ~28%.
    """
    rows = real_rows()
    # Fit on 2 and 8 only, then predict 4 (which is not in that fit).
    subset = {tile: rows[tile] for tile in (2, 8)}
    matrix = np.array([[subset[tile]['U_tile_loads'],
                        subset[tile]['L_voxel_loads']] for tile in (2, 8)])
    times = np.array([subset[tile]['p50_ms'] for tile in (2, 8)])
    coefficients = np.linalg.lstsq(matrix, times, rcond=None)[0]
    predicted = float(np.dot([rows[4]['U_tile_loads'], rows[4]['L_voxel_loads']],
                             coefficients))
    error = (predicted - rows[4]['p50_ms']) / rows[4]['p50_ms']
    assert error > 0.2, error
    # The full three-term fit on 2/4/8 reproduces tile 4 by construction.
    per_load, per_full, per_reject = fit(rows)
    assert float(np.dot(design(rows[4]), [per_load, per_full, per_reject])) \
        == pytest.approx(rows[4]['p50_ms'], abs=1e-9)


def test_holdout_script_requires_holdout_data(tmp_path):
    """No holdout data => a clear refusal, not a fabricated verdict."""
    import subprocess
    completed = subprocess.run(
        [sys.executable, 'benchmarks/p4/tile_holdout.py', '--holdout',
         str(tmp_path / 'absent.json'), '--json', str(tmp_path / 'out.json')],
        cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode != 0
    assert 'holdout tiles missing' in completed.stdout + completed.stderr


def test_p4_10_reports_the_rejection_fractions_and_the_holdout_plan():
    text = (ROOT / 'analysis/sparse_render/results/P4-10_tile_size_sweep.md'
            ).read_text(encoding='utf-8')
    for token in ('26.0%', '54.8%', '77.2%', '0.32', '+28.0%',
                  'run_p4_tile_holdout.sh', '拒绝'):
        assert token in text, f'P4-10 lost {token!r}'
