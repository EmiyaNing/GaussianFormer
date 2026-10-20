"""Tests for the spatial failure capture and the BEV failure map (P4.2 item 5).

The confusion matrices elsewhere say *how much* is missed or hallucinated; this
path says *where*.  Three things must hold:

* the evaluator records exactly the TP/FP/FN voxel sets implied by its own inputs
  (so the picture describes the same errors as the numbers);
* the BEV column map marks a column for the requested category only -- separate
  panels per category exist precisely because a single priority-encoded map would
  hide a true positive behind a false positive in the same column;
* flat-index decoding must invert the ``reshape(-1)`` layout (z fastest, then y,
  then x), which is easy to get wrong and silently flips the map.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.plots.make_failure_maps import column_flags, summarise  # noqa: E402
from misc.sparseworld_strict_eval import SparseWorldStrictMetric  # noqa: E402

FREE = 17


def test_flat_index_decoding_inverts_reshape():
    """Index i in a [D,H,W] grid reshaped to (-1) is (z, y, x) = i fastest in z."""
    shape = (4, 5, 6)
    flat = np.arange(4 * 5 * 6)
    z = flat % 4
    rest = flat // 4
    y = rest % 5
    x = rest // 5
    assert (z[7], y[7], x[7]) == (3, 1, 0)
    # And a round trip through the rasteriser's own expression.
    entry = {'grid_shape': list(shape), 'fp': [7]}
    canvas = column_flags(entry, 'fp')
    assert canvas[1, 0]
    assert int(canvas.sum()) == 1


def test_separate_panels_do_not_hide_each_other():
    """One column holds both a TP and an FP; both panels must mark it."""
    entry = {'grid_shape': [2, 3, 3], 'tp': [0], 'fp': [1], 'fn': []}
    # index 0 -> (z=0, y=0, x=0); index 1 -> (z=1, y=0, x=0): both column (0,0).
    assert column_flags(entry, 'tp')[0, 0]
    assert column_flags(entry, 'fp')[0, 0]
    assert not column_flags(entry, 'fn').any()


def test_evaluator_capture_matches_the_confusion_counts(tmp_path):
    """The captured sets must have the sizes the confusion matrix implies."""
    shape = (4, 4, 4)
    prediction = torch.full(shape, FREE, dtype=torch.long)
    target = torch.full(shape, FREE, dtype=torch.long)
    # TP at flat 0, FN at flat 1, FP at flat 16.
    prediction.view(-1)[0] = 0
    target.view(-1)[0] = 0
    target.view(-1)[1] = 0
    prediction.view(-1)[16] = 0

    dump = tmp_path / 'per_sample.json'
    metric = SparseWorldStrictMetric(boundary_stride=0, per_sample_path=dump,
                                     spatial_samples=1)
    metad = {'occ_label': target.unsqueeze(0),
             'future_occ_labels': target.unsqueeze(0).unsqueeze(0).expand(
                 1, 6, *shape).contiguous()}
    metric.add_batch({'final_occ_grid': prediction.unsqueeze(0),
                      'future_occ_grids': [prediction.unsqueeze(0)] * 6}, metad)
    metric.results()

    payload = json.loads(dump.read_text(encoding='utf-8'))
    assert len(payload['spatial']) == 4        # one per horizon
    entry = payload['spatial'][0]
    assert entry['tp'] == [0]
    assert entry['fn'] == [1]
    assert entry['fp'] == [16]
    # And the counts agree with the binary histogram the metric accumulates.
    binary = np.asarray(payload['per_sample'][0]['binary'])
    tp, fp, fn = len(entry['tp']), len(entry['fp']), len(entry['fn'])
    assert binary[1, 1] == tp
    assert binary[0, 1] == fp
    assert binary[1, 0] == fn


def test_spatial_capture_is_off_by_default(tmp_path):
    """A plain evaluation must not silently grow its dump with voxel indices."""
    shape = (4, 4, 4)
    prediction = torch.zeros(shape, dtype=torch.long)
    target = torch.zeros(shape, dtype=torch.long)
    dump = tmp_path / 'per_sample.json'
    metric = SparseWorldStrictMetric(boundary_stride=0, per_sample_path=dump)
    metad = {'occ_label': target.unsqueeze(0),
             'future_occ_labels': target.unsqueeze(0).unsqueeze(0).expand(
                 1, 6, *shape).contiguous()}
    metric.add_batch({'final_occ_grid': prediction.unsqueeze(0),
                      'future_occ_grids': [prediction.unsqueeze(0)] * 6}, metad)
    metric.results()
    payload = json.loads(dump.read_text(encoding='utf-8'))
    assert payload['spatial'] == []


def test_summarise_pools_counts_over_scenes():
    payload = {'spatial': [
        {'horizon': 4, 'tp': [0, 1], 'fp': [16], 'fn': [2]},
        {'horizon': 4, 'tp': [0], 'fp': [], 'fn': [3, 4]},
        {'horizon': 0, 'tp': [0], 'fp': [], 'fn': []},
    ]}
    totals = summarise(payload)
    assert totals[4] == {'tp': 3, 'fp': 1, 'fn': 3, 'scenes': 2}
    assert totals[0]['tp'] == 1


def test_cli_renders_a_figure_from_a_synthetic_dump(tmp_path):
    shape = [4, 4, 4]
    entries = [{'sample': 0, 'horizon': h, 'grid_shape': shape,
                'tp': [0, 1], 'fp': [16], 'fn': [2]} for h in (0, 2, 4, 6)]
    dump = tmp_path / 'spatial.json'
    dump.write_text(json.dumps({'samples': 1, 'horizons': [0, 2, 4, 6],
                                'class_names': [], 'per_sample': [],
                                'spatial': entries}), encoding='utf-8')
    out = tmp_path / 'figs'
    completed = subprocess.run(
        [sys.executable, 'benchmarks/plots/make_failure_maps.py', '--variants',
         f'syn={dump}', '--horizon', '4', '--scene', '0', '--out', str(out)],
        cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-400:]
    figures = list(out.glob('*.png'))
    assert len(figures) == 1 and figures[0].stat().st_size > 0
    # Hand-computed: TP 2, FP 1, FN 1 -> precision = recall = 2/3.
    assert 'precision 0.667, recall 0.667' in completed.stdout
