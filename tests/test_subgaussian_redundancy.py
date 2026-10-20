"""Hand-computed tests for the sub-Gaussian redundancy diagnostic (P4.2 item 3).

The diagnostic answers "are a query's 48 children covering distinct occupied
voxels, or are they duplicates?".  Two things must be right or the number is
meaningless:

* the **grouping** — the original flat dump applied a boolean mask that dropped
  116 of 4x34,560 points, so 48-child boundaries cannot be recovered from it;
  this diagnostic consumes a *grouped* dump instead;
* the **voxel origin** — voxels are indexed from ``pc_range[:3]``, which is
  offset (-40, -40, -1).  Forgetting the offset shifts every index and changes
  which children collide.

Both are pinned below with hand-computed values, including the 1.8 m case where
the offset decides whether two children share a voxel.
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

from benchmarks.p4.subgaussian_redundancy import (  # noqa: E402
    pairwise_distances, unique_voxel_ratio)

PC_LOWER = torch.tensor([-40.0, -40.0, -1.0])


def test_unique_voxel_ratio_counts_distinct_voxels():
    # Four children inside one 0.4 m voxel: one distinct voxel out of four.
    clustered = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                              [0.0, 0.1, 0.0], [0.05, 0.05, 0.0]])
    assert unique_voxel_ratio(clustered, 0.4, PC_LOWER) == pytest.approx(0.25)
    # Four children 1 m apart: all distinct.
    spread = torch.tensor([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0],
                           [7.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    assert unique_voxel_ratio(spread, 0.4, PC_LOWER) == pytest.approx(1.0)


def test_voxel_origin_offset_changes_the_answer():
    """The origin decides whether (0.0, 1.4) collide at 1.8 m.

    With the real origin ``pc_lower = -40``: ``floor(40/1.8) = 22`` and
    ``floor(41.4/1.8) = 23`` -> distinct voxels (ratio 1.0).
    Without the offset: ``floor(0/1.8) = 0`` and ``floor(1.4/1.8) = 0`` -> the
    same voxel (ratio 0.5).  So forgetting the offset silently changes which
    children are considered duplicates.  (The pair was found by search, not by
    hand: an earlier hand-picked pair turned out not to discriminate.)
    """
    pair = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    assert unique_voxel_ratio(pair, 1.8, PC_LOWER) == pytest.approx(1.0)
    assert unique_voxel_ratio(pair, 1.8, torch.zeros(3)) == pytest.approx(0.5)


def test_pairwise_distances_are_hand_checkable():
    triangle = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    distances = sorted(pairwise_distances(triangle).tolist())
    assert distances == pytest.approx([3.0, 4.0, 5.0])


def test_single_child_has_no_pairs():
    assert pairwise_distances(torch.zeros(1, 3)).numel() == 0


def test_end_to_end_on_a_synthetic_grouped_dump(tmp_path):
    """The script must read a grouped dump and reproduce a hand-computed mean."""
    pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    points = torch.zeros(1, 8, 3)
    points[0, 0:4] = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                                   [0.0, 0.1, 0.0], [0.05, 0.05, 0.0]])
    points[0, 4:8] = torch.tensor([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0],
                                   [7.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    dumped = tmp_path / 'grouped.pt'
    torch.save({'points': points, 'valid': torch.ones(1, 8, dtype=torch.bool),
                'children_per_query': 4, 'pc_range': pc_range,
                'config': 'synthetic', 'checkpoint': 'synthetic'}, dumped)
    out = tmp_path / 'redundancy.json'
    completed = subprocess.run(
        [sys.executable, 'benchmarks/p4/subgaussian_redundancy.py',
         '--dumped', str(dumped), '--voxels', '0.4', '--json', str(out)],
        cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-400:]
    payload = json.loads(out.read_text(encoding='utf-8'))
    row = payload['rows'][0]
    # Query 0 gives 1/4, query 1 gives 4/4 -> mean 0.625.
    assert row['unique_voxel_ratio_mean'] == pytest.approx(0.625, abs=1e-6)
    assert row['queries_analysed'] == 2
    # Distances are pooled ACROSS queries, so this median mixes the clustered
    # query's small separations with the spread query's 1-3 m ones.  Pinning the
    # pooled value makes the pooling explicit rather than accidental.
    assert row['child_distance_p50_m'] == pytest.approx(0.5707, abs=0.001)
    assert row['child_distance_max_m'] == pytest.approx(3.0, abs=1e-4)
    # Writing outside the repository must not crash the message formatting.
    assert out.exists()
