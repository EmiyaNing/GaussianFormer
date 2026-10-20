"""Tests for the tile-cost model's measured-vs-predicted comparison.

The model exists to test one pre-registered claim: the 4x4x4x4 tile is optimal,
tile 2 costs ~2.10x and tile 8 ~1.38x at the sweep's workload.  Two properties
have to hold for that test to mean anything:

* the measured rows must be compared against predictions at the **same
  workload** -- the sweep runs at G=34,560 / scale 0.53, while the fitted (l, s)
  are calibrated at G=76,800 / scale 0.35, and the two workloads predict
  different absolute times (0.813 ms vs 1.152 ms for tile 4).  An earlier
  version of this comparison attached the sweep rows to the calibration table.
* only the ``block`` operator's row is the kernel being modelled; a naive row
  must be ignored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p4.sweep_tile_size import bound_for  # noqa: E402
from benchmarks.p4.tile_cost_model import (  # noqa: E402
    BLOCK4_MS, CALIBRATION, NAIVE_MS, fit, load_measured)

SWEEP_WORKLOAD = {'gaussians': 34560, 'scale': 0.53}
POINTS = 'benchmarks/results/real_points.pt'


def test_load_measured_keeps_only_block_rows(tmp_path):
    path = tmp_path / 'sweep.json'
    path.write_text(json.dumps({'timings': [
        {'tile_edge': 2, 'operator': 'naive', 'p50_ms': 9.0},
        {'tile_edge': 2, 'operator': 'block', 'p50_ms': 1.7},
        {'tile_edge': 4, 'operator': 'block', 'p50_ms': 0.8},
        {'tile_edge': 8, 'operator': 'block', 'p50_ms': 1.1},
    ]}), encoding='utf-8')
    measured = load_measured(str(path))
    assert measured == {2: 1.7, 4: 0.8, 8: 1.1}
    # Missing file degrades to "no measured rows", not an exception.
    assert load_measured(str(tmp_path / 'absent.json')) == {}
    # A malformed file must not take the whole script down either.
    (tmp_path / 'bad.json').write_text('{not json', encoding='utf-8')
    assert load_measured(str(tmp_path / 'bad.json')) == {}


def test_predictions_are_taken_at_the_sweep_workload_not_the_calibration_one():
    """The comparison's two sides must come from the same workload.

    At the calibration workload (G=76,800, scale=0.35) the model predicts
    2.009 / 1.152 / 1.824 ms; at the sweep's workload (G=34,560, scale=0.53) it
    predicts 1.710 / 0.813 / 1.120 ms.  The sweep measures the latter, so that is
    the table the measured column belongs to.
    """
    if not (ROOT / POINTS).exists():
        pytest.skip('real point cloud not present')
    calib_row = bound_for(CALIBRATION['tile'], POINTS, CALIBRATION['gaussians'],
                          CALIBRATION['scale'], 0)
    per_load, per_scan = fit(NAIVE_MS, BLOCK4_MS, calib_row)

    def predict(workload):
        out = {}
        for tile in (2, 4, 8):
            row = bound_for(tile, POINTS, workload['gaussians'],
                            workload['scale'], 0)
            out[tile] = (row['U_tile_loads'] * per_load
                         + row['block_scan_iterations'] * per_scan)
        return out

    sweep = predict(SWEEP_WORKLOAD)
    calibration = predict({'gaussians': CALIBRATION['gaussians'],
                           'scale': CALIBRATION['scale']})
    assert sweep[4] == pytest.approx(0.813, abs=0.005)
    assert calibration[4] == pytest.approx(1.152, abs=0.005)
    # The two workloads are far apart in absolute time: quoting one against the
    # other's measurement is a >40% error before any modelling error.
    assert abs(sweep[4] - calibration[4]) / sweep[4] > 0.4

    # The pre-registered ratios the report quotes come from the sweep workload.
    assert sweep[2] / sweep[4] == pytest.approx(2.10, abs=0.02)
    assert sweep[8] / sweep[4] == pytest.approx(1.38, abs=0.02)
    assert min(sweep, key=sweep.get) == 4
    # And the out-of-sample check against P4-6's independent 0.795 ms.
    assert abs(sweep[4] - 0.795) / 0.795 < 0.05


def test_bound_only_never_overwrites_the_sweep_artifact():
    """A CPU diagnostic must not destroy the measured rows.

    ``sweep_tile_size.py --bound-only`` used to write the same default json as
    the GPU sweep, so re-running the analytic bound after the sweep would replace
    the measured timings with ``timings: None`` -- silently deleting the data the
    P4-10 report is built from.  An explicit --json still wins.
    """
    from benchmarks.p4.sweep_tile_size import DEFAULT_JSON, bound_only_target
    assert bound_only_target(DEFAULT_JSON) == 'benchmarks/results/p4_tile_bounds.json'
    assert bound_only_target(DEFAULT_JSON) != DEFAULT_JSON
    explicit = 'benchmarks/results/somewhere_else.json'
    assert bound_only_target(explicit) == explicit


def test_load_measured_reads_the_bounds_shape_the_sweep_actually_writes(tmp_path):
    """The real artifact merges timings INTO the bounds rows.

    The first version of ``load_measured`` only looked at ``timings``/``rows``, so
    it reported "not present yet" for a sweep that had in fact run -- the
    measured p50 sits inside ``bounds`` (with ``verified_tile_edge`` and
    ``speedup_vs_tile4``).  Pinning the real shape here is what keeps the P4-10
    comparison from silently degrading to "prediction only".
    """
    path = tmp_path / 'sweep.json'
    path.write_text(json.dumps({
        'gaussians': 34560, 'scale': 0.53, 'reps': 15,
        'bounds': [
            {'tile_edge': 2, 'U_tile_loads': 3430855, 'p50_ms': 1.454,
             'verified_tile_edge': 2, 'speedup_vs_tile4': 0.513},
            {'tile_edge': 4, 'U_tile_loads': 701445, 'p50_ms': 0.746,
             'verified_tile_edge': 4, 'speedup_vs_tile4': 1.0},
            {'tile_edge': 8, 'U_tile_loads': 173750, 'p50_ms': 0.858,
             'verified_tile_edge': 8, 'speedup_vs_tile4': 0.869},
        ]}), encoding='utf-8')
    assert load_measured(str(path)) == {2: 1.454, 4: 0.746, 8: 0.858}
    # A bounds row without a timing (the --bound-only payload) yields nothing.
    path.write_text(json.dumps({'bounds': [{'tile_edge': 4, 'p50_ms': None}]}),
                    encoding='utf-8')
    assert load_measured(str(path)) == {}


def test_p4_10_numbers_match_the_sweep_artifact():
    """P4-10's curve, prediction and ratios must come from the artifacts."""
    sweep_path = ROOT / 'benchmarks/results/p4_tile_sweep.json'
    model_path = ROOT / 'benchmarks/results/p4_tile_model.json'
    if not (sweep_path.exists() and model_path.exists()):
        pytest.skip('tile sweep/model artifacts not present')
    sweep = json.loads(sweep_path.read_text(encoding='utf-8'))
    rows = {row['tile_edge']: row for row in sweep['bounds']}
    if not all(rows.get(tile, {}).get('p50_ms') for tile in (2, 4, 8)):
        pytest.skip('measured rows not present')
    # Every row must carry a runtime verification of the tile it was built with.
    for tile, row in rows.items():
        if row.get('p50_ms'):
            assert row['verified_tile_edge'] == tile, (tile, row)
    measured = {tile: rows[tile]['p50_ms'] for tile in (2, 4, 8)}
    assert measured[2] == pytest.approx(1.454, abs=0.01)
    assert measured[4] == pytest.approx(0.746, abs=0.01)
    assert measured[8] == pytest.approx(0.858, abs=0.01)
    assert min(measured, key=measured.get) == 4, 'tile 4 must stay the optimum'

    model = json.loads(model_path.read_text(encoding='utf-8'))
    # The sweep-workload predictions live under ``model_point_predictions`` (the
    # ``predicted_p50_ms`` key is the CALIBRATION workload, 1.152 ms for tile 4 --
    # comparing the measured rows against that one would be a 40%+ error before
    # any modelling error).
    predictions = {int(k): v
                   for k, v in model['model_point_predictions'].items()}
    assert model['predicted_optimum_tile'] == 4
    assert predictions[4] == pytest.approx(0.813, abs=0.005)
    assert model['predicted_p50_ms']['4'] == pytest.approx(1.152, abs=0.005)
    # The measured comparison is recorded in the artifact, so P4-10's error
    # figures are re-derivable from JSON and not only from stdout.
    assert model['measured'] == {str(t): pytest.approx(measured[t], abs=0.01)
                                 for t in (2, 4, 8)}
    assert model['measured_optimum_tile'] == 4
    assert model['measured_agrees_with_prediction'] is True
    assert model['model_incomplete_at_this_workload'] is True
    assert model['measured_errors']['8'] == pytest.approx(0.305, abs=0.02)
    assert model['measured_errors']['4'] == pytest.approx(0.090, abs=0.01)
    # The model over-predicts every tile, worst at tile 8 -- the pre-registered
    # "missing third cost term" signature.
    errors = {tile: (predictions[tile] - measured[tile]) / measured[tile]
              for tile in (2, 4, 8)}
    assert all(error > 0 for error in errors.values()), errors
    assert errors[8] > errors[4] and errors[2] > errors[4], errors
    assert errors[8] == pytest.approx(0.305, abs=0.02)
    ratios = {tile: measured[tile] / measured[4] for tile in (2, 8)}
    assert ratios[2] == pytest.approx(1.95, abs=0.02)
    assert ratios[8] == pytest.approx(1.15, abs=0.02)

    text = (ROOT / 'analysis/sparse_render_result/P4-10_tile_size_sweep.md'
            ).read_text(encoding='utf-8')
    for token in ('1.454', '0.746', '0.858', '+30.5%', '1.95×', '1.15×',
                  'verified_tile_edge'):
        assert token in text, f'P4-10 lost {token!r}'
