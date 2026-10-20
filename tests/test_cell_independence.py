"""The accepted neighbour set must be independent of the spatial-hash cell size.

``cell_size`` is only a *candidate lookup* structure: the operator scans the
``(2k+1)^3`` cells around a query with ``k = ceil(support_sigma * max_scale /
cell)``, then admits a candidate only if it passes the per-axis support box test.
The window is sized to cover that box, so the admitted set is cell-independent
**by construction** -- yet the existing numerics tests only compare each cell size
against the brute-force reference, never the outputs *across* cell sizes, so a
cell-dependent bug that the reference reproduced would have passed.

Two consequences make this worth pinning:

* it is a correctness invariant of the operator, and
* it makes the robustness probe's ``cell_size`` variants (0.9 and 3.6 against the
  trained 1.8) a **consistency check on the whole evaluation path**: identical
  metrics are the expected outcome, and any difference means either an operator
  bug or that something other than ``cell_size`` changed between the configs --
  it must not be reported as "robustness to the bin size".

The invariant has a precondition -- ``max_scale`` must really bound the scales --
which is *not* re-tested here: ``test_semantic_gaussian_localagg_numerics.
test_binning_completeness_requires_adequate_max_scale`` already pins it at the
operator level (an undersized ``max_scale`` silently drops true neighbours).  The
adapter satisfies it by deriving ``max_scale`` from the configured ``scale_range``
upper bound, which the learned scales cannot leave.

CPU-only: it exercises the candidate-window logic directly, which is the logic
``cell_size`` controls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.bench_operators import PC_RANGE  # noqa: E402
from benchmarks.reference.g2q_bruteforce import (  # noqa: E402
    cell_coordinates, neighbour_cells)

CELLS = (0.6, 1.2, 1.8, 2.4, 3.6)

# A hand-built cloud with clusters at different scales, so the box test actually
# rejects candidates (a uniform cloud would make the test insensitive).
MAX_SCALE = 0.6


def build_cloud(count=400, seed=0):
    generator = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0., 0., 0.], [5., 0., 0.], [0., 5., 0.],
                            [5., 5., 0.], [-4., 2., 1.]])
    which = torch.randint(0, centres.shape[0], (count,), generator=generator)
    points = centres[which] + torch.randn(count, 3, generator=generator) * 0.7
    # Scales must respect the operator's contract that ``max_scale`` is a true
    # upper bound: the cell window is sized from it.  The first version of this
    # test let the jitter push scales to ~0.70 above MAX_SCALE, and the window
    # then genuinely missed neighbours -- the invariant failed, correctly, for an
    # input the operator never promises to handle.
    scales = torch.full((count, 3), 0.35) * (
        1.0 + 0.3 * torch.randn(count, 1, generator=generator)).clamp_min(0.5)
    return points, scales.clamp(1e-4, MAX_SCALE)


def admitted(points, scales, query, cell, support_sigma, max_scale):
    """Admitted neighbours for one query via the cell window plus the box test."""
    coord, shape = cell_coordinates(points, PC_RANGE, cell)
    shape = tuple(int(v) for v in shape)
    radius = neighbour_cells(support_sigma, max_scale, cell)
    members = {}
    for index in range(points.shape[0]):
        members.setdefault(tuple(int(v) for v in coord[index]), []).append(index)
    base = tuple(int(v) for v in coord[query])
    found = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                key = (base[0] + dx, base[1] + dy, base[2] + dz)
                if not (0 <= key[0] < shape[0] and 0 <= key[1] < shape[1]
                        and 0 <= key[2] < shape[2]):
                    continue
                for other in members.get(key, ()):
                    delta = (points[query] - points[other]).abs()
                    if bool((delta <= support_sigma * scales[other]).all()):
                        found.add(other)
    return found


@pytest.mark.parametrize('support_sigma', [2.0, 3.0, 4.0])
def test_admitted_set_is_identical_across_cell_sizes(support_sigma):
    points, scales = build_cloud()
    max_scale = MAX_SCALE
    queries = [0, 7, 33, 101, 250]
    reference = {q: admitted(points, scales, q, 1.8, support_sigma, max_scale)
                 for q in queries}
    for cell in CELLS:
        for query in queries:
            got = admitted(points, scales, query, cell, support_sigma, max_scale)
            assert got == reference[query], (
                f'cell {cell} admits a different set for query {query} than cell '
                f'1.8: symmetric difference {sorted(got ^ reference[query])[:5]}')


def test_the_window_always_covers_the_support_box():
    """The invariant holds because k grows as the cell shrinks.

    Pinning the reason matters: if ``neighbour_cells`` ever stopped growing with
    a shrinking cell, the previous test would start failing for a reason nobody
    would guess from its name.
    """
    for cell in CELLS:
        radius = neighbour_cells(3.0, 0.6, cell)
        assert radius * cell >= 3.0 * 0.6 - 1e-9, (cell, radius)
