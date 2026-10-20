"""The A2-gauss control's scale must be chosen by relative concentration.

The first draft used 0.35 (A5's initialisation scale).  Measured over the actual
8 nearest neighbours -- which sit 0.399 m away on average -- that gives a weight
ESS of 6.97 out of 8, i.e. a Gaussian barely distinguishable from a uniform mean,
so a null result would have been uninformative.  The scale is instead fixed by
matching **relative concentration** to A5's kernel (ESS/accepted = 0.278).

These tests pin that criterion, not the particular number: the chosen scale must
land in the same concentration regime as A5, and the ESS of normalised weights
can never exceed k (a larger value means the kernel underflowed -- which is
exactly the bug this measurement had, producing an impossible ESS of 1e34).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p4.selection_overlap import scale_sweep  # noqa: E402
from benchmarks.common_paths import ROOT as BENCH_ROOT  # noqa: E402

A5_CONCENTRATION = 0.278          # ESS/accepted, measured in p4_ess.json


def declared_scale():
    text = (ROOT / 'config/opus/ablations/p1_a2_knn_gaussian8.py').read_text(encoding='utf-8')
    match = re.search(r'gaussian_scale=([0-9.]+)', text)
    assert match, 'the config must declare gaussian_scale'
    return float(match.group(1))


def test_declared_scale_is_not_the_vacuous_one():
    scale = declared_scale()
    assert scale < 0.2, (
        f'scale {scale} is in the regime where the Gaussian is nearly uniform '
        'over the k-NN set; the control would be vacuous')


@pytest.mark.skipif(not (ROOT / 'benchmarks/results/real_points.pt').exists(),
                    reason='real points not dumped')
def test_declared_scale_matches_a5_relative_concentration():
    from benchmarks.bench_operators import make_cloud

    cloud = make_cloud('real', 34560, scale=0.53, device='cpu',
                       real_points_path=str(ROOT / 'benchmarks/results/real_points.pt'),
                       seed=0)
    points = cloud.points.float()
    generator = torch.Generator().manual_seed(0)
    queries = torch.randperm(points.shape[0], generator=generator)[:60]
    rows = scale_sweep(points, queries, 8, [declared_scale()])
    row = rows[0]
    # Same concentration regime as A5, within a generous band.
    assert abs(row['ess_over_k'] - A5_CONCENTRATION) < 0.08, row
    # And never more concentrated than uniform: the sanity bound that caught the
    # underflow bug in the first version of this measurement.
    assert row['ess'] <= 8 + 1e-6, row
