"""Hand-computed tests for the P4.2 items 2 and 4 diagnostics.

Item 2 asks whether the realised Gaussian geometry degenerates to isotropic, and
item 4 asks for the unused-Gaussian ratio and semantic aliasing.  Both consume
the grouped dump's per-child ``scales``/``opacities``/``logits``.

The synthetic case below has every expected value computable by hand:

* four children with scales (0.5, 0.5, 0.5) and four with (0.2, 0.5, 0.8):
  per-axis means 0.35 / 0.50 / 0.65 -> axis ratio 1.857;
  per-child max/min is 1.0 for the first four and 4.0 for the rest -> mean 2.5;
* one child of query 0 has a different top-1 class, and all four of that query's
  children sit in one voxel -> 1 aliased voxel of 1 multi-child voxel (100%),
  1 conflicting child of 4 (25%);
* every child is valid -> unused ratio 0.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
CHILDREN = 4
CLASSES = 17


def synthetic_dump(path: Path, valid_all=True):
    points = torch.zeros(1, 2 * CHILDREN, 3)
    points[0, 0:4] = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                                   [0.0, 0.1, 0.0], [0.05, 0.05, 0.0]])
    points[0, 4:8] = torch.tensor([[0.0, 3.0, 0.0], [3.0, 3.0, 0.0],
                                   [6.0, 3.0, 0.0], [9.0, 3.0, 0.0]])
    scales = torch.zeros(1, 2 * CHILDREN, 3)
    scales[0, 0:4] = torch.tensor([[0.5, 0.5, 0.5]]).expand(4, 3)
    scales[0, 4:8] = torch.tensor([[0.2, 0.5, 0.8]]).expand(4, 3)
    opacities = torch.full((1, 2 * CHILDREN), 0.02)
    logits = torch.zeros(1, 2 * CHILDREN, CLASSES)
    logits[..., 0] = 5.0
    logits[0, 1, 0] = 0.0
    logits[0, 1, 1] = 5.0
    valid = torch.ones(1, 2 * CHILDREN, dtype=torch.bool)
    if not valid_all:
        valid[0, 2:4] = False
    torch.save({'points': points, 'valid': valid, 'children_per_query': CHILDREN,
                'pc_range': PC_RANGE, 'config': 'synthetic', 'checkpoint': 'synthetic',
                'scales': scales, 'opacities': opacities, 'logits': logits}, path)


def run(dumped: Path, out: Path, voxels=('0.4',)):
    # ``--voxels`` is nargs='*', so each value must be its own argv entry; a
    # single "0.4 1.8" string reaches argparse as one non-float token.
    completed = subprocess.run(
        [sys.executable, 'benchmarks/p4/gaussian_attribute_distribution.py',
         '--dumped', str(dumped), '--voxels', *voxels, '--json', str(out),
         '--base-attributes', str(out) + '.missing'],
        cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-500:]
    return json.loads(out.read_text(encoding='utf-8'))


def test_realised_scale_axes_and_anisotropy(tmp_path):
    dumped = tmp_path / 'grouped.pt'
    synthetic_dump(dumped)
    payload = run(dumped, tmp_path / 'out.json')
    axis = payload['scale_per_axis']
    assert axis['0']['mean'] == pytest.approx(0.35, abs=1e-6)
    assert axis['1']['mean'] == pytest.approx(0.50, abs=1e-6)
    assert axis['2']['mean'] == pytest.approx(0.65, abs=1e-6)
    assert payload['scale_realised_axis_ratio'] == pytest.approx(1.8571, abs=1e-3)
    # Per-child max/min: four children at 1.0 and four at 4.0.
    assert payload['scale_anisotropy']['mean'] == pytest.approx(2.5, abs=1e-6)
    assert payload['scale_anisotropy']['p95'] == pytest.approx(4.0, abs=1e-6)
    assert payload['opacity']['mean'] == pytest.approx(0.02, abs=1e-9)


def test_semantic_aliasing_and_unused_ratio(tmp_path):
    dumped = tmp_path / 'grouped.pt'
    synthetic_dump(dumped)
    payload = run(dumped, tmp_path / 'out.json', voxels=('0.4', '1.8'))
    assert payload['unused_gaussian_ratio'] == pytest.approx(0.0, abs=1e-9)
    for row in payload['aliasing']:
        assert row['multi_child_voxels'] == 1, row
        assert row['aliased_voxel_fraction'] == pytest.approx(1.0, abs=1e-6)
        assert row['conflicting_child_fraction'] == pytest.approx(0.25, abs=1e-6)
        assert row['class_consistent_support_fraction'] == pytest.approx(0.0, abs=1e-6)


def test_unused_ratio_counts_masked_children(tmp_path):
    dumped = tmp_path / 'grouped.pt'
    synthetic_dump(dumped, valid_all=False)
    payload = run(dumped, tmp_path / 'out.json')
    assert payload['unused_gaussian_ratio'] == pytest.approx(0.25, abs=1e-6)
    assert payload['valid_children'] == 6
    assert payload['total_children'] == 8


def test_missing_dump_fails_loudly(tmp_path):
    completed = subprocess.run(
        [sys.executable, 'benchmarks/p4/gaussian_attribute_distribution.py',
         '--dumped', str(tmp_path / 'nope.pt'), '--json', str(tmp_path / 'o.json')],
        cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode != 0
    assert 'grouped dump missing' in (completed.stdout + completed.stderr)
