"""Tests for the ``mode='gaussian'`` weighting control.

A2 (uniform weights over the 8 Euclidean nearest) versus A5 (Gaussian weights
over a shell-selected set) confounds the weight profile with the neighbour
selection.  ``mode='gaussian'`` holds the selection at A2's exact k-NN and swaps
only the weights, so A2-vs-this isolates the weighting and A5-cap8-vs-this
isolates the selection.

The weight formula must therefore be a *normalised* fixed-isotropic Gaussian of
the query-to-neighbour displacement, equal weights must reproduce A2 exactly, and
the parameter count must stay identical to A2's (no learned scale) -- get any of
those wrong and the control would no longer isolate what it claims to.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.head.ablations.knn_residual import KNNResidualAggregator  # noqa: E402

PC_RANGE = [-40., -40., -1., 40., 40., 5.4]


def build(mode, **kwargs):
    torch.manual_seed(0)
    return KNNResidualAggregator(
        embed_dims=8, num_children=3, num_classes=5, pc_range=PC_RANGE,
        num_neighbors=3, include_self=True, mode=mode, query_chunk_size=4, **kwargs)


def run(head, points):
    token = torch.randn(1, points.shape[0], 8, generator=torch.Generator().manual_seed(1))
    valid = torch.ones(1, points.shape[0], dtype=torch.bool)
    return head(token, points.unsqueeze(0), valid)


def test_gaussian_mode_is_accepted_and_default_is_unchanged():
    head = build('gaussian')
    assert head.mode == 'gaussian'
    assert head.gaussian_scale == pytest.approx(0.35)
    with pytest.raises(ValueError):
        build('gaussian', gaussian_scale=0.0)
    with pytest.raises(ValueError):
        KNNResidualAggregator(embed_dims=8, num_children=3, num_classes=5,
                              pc_range=PC_RANGE, mode='nonsense')


def test_vanishing_scale_concentrates_on_the_self_neighbour():
    """scale -> 0 must reproduce ``mode='self'``.

    The self neighbour sits at distance zero, so as the kernel narrows its weight
    tends to 1 and the mixture tends to the query's own seed -- which is exactly
    what ``mode='self'`` returns.  This pins the profile's concentration on the
    *nearest* neighbour without needing to control the seeds.
    """
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]]])
    self_mode = build('self')
    narrow = build('gaussian', gaussian_scale=1e-4)
    with torch.no_grad():
        self_mode.residual_head.weight.normal_(0, 1)
        self_mode.residual_head.bias.normal_(0, 1)
    narrow.load_state_dict(self_mode.state_dict(), strict=False)
    a = run(self_mode, points)['residual']
    b = run(narrow, points)['residual']
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_gaussian_output_stays_a_convex_combination():
    """A weighted mean cannot leave the range of the values it averages.

    Checked against the two extremes the other modes provide: the output must lie
    between the query's own seed (``self``) and the uniform mean over the same
    k-NN set (``mean``), because those bracket the neighbours' seeds.
    """
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]]])
    mean_mode = build('mean')
    self_mode = build('self')
    mid = build('gaussian', gaussian_scale=0.4)
    with torch.no_grad():
        mean_mode.residual_head.weight.normal_(0, 1)
        mean_mode.residual_head.bias.normal_(0, 1)
    self_mode.load_state_dict(mean_mode.state_dict(), strict=False)
    mid.load_state_dict(mean_mode.state_dict(), strict=False)
    mean_out = run(mean_mode, points)['residual']
    self_out = run(self_mode, points)['residual']
    mid_out = run(mid, points)['residual']
    low = torch.minimum(mean_out, self_out) - 1e-5
    high = torch.maximum(mean_out, self_out) + 1e-5
    assert bool(((mid_out >= low) & (mid_out <= high)).all()), 'left the convex hull'


def test_equal_weights_reduce_to_the_uniform_mean():
    """A very large scale makes the kernel flat, i.e. A2's uniform mean.

    This is the check that the new mode really is A2 plus a weight profile: with
    exp(-d^2/2s^2) -> 1 every member gets 1/k, so the outputs must coincide.
    """
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]]])
    uniform = build('mean')
    huge = build('gaussian', gaussian_scale=1e6)
    # Share the parameters so only the aggregation differs.
    huge.load_state_dict(uniform.state_dict(), strict=False)
    with torch.no_grad():
        uniform.residual_head.weight.normal_(0, 1)
        uniform.residual_head.bias.normal_(0, 1)
        huge.load_state_dict(uniform.state_dict(), strict=False)
    a = run(uniform, points)['residual']
    b = run(huge, points)['residual']
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_gaussian_mode_adds_no_parameters_beyond_a_constant():
    """Plan T1.2: the control must match A2's parameter count exactly."""
    uniform = build('mean')
    gaussian = build('gaussian')
    assert sum(p.numel() for p in uniform.parameters()) == \
        sum(p.numel() for p in gaussian.parameters())


def test_smoke_script_covers_the_large_k_configs():
    """The queue's pre-training smoke test must at least run end to end.

    ``A2'`` uses k=215, far above anything trained before; the smoke script is
    what catches a CUDA k-NN limitation or an index-tensor blow-up in seconds
    instead of 50 minutes into a queued run.  Exercised here on CPU (the k-NN
    falls back to a chunked cdist), which still validates the wiring, the config
    parsing and the shape/finiteness assertions.
    """
    import subprocess
    config = ROOT / 'config/opus/ablations/p1_a2_knn_k215.py'
    assert config.exists()
    completed = subprocess.run(
        [sys.executable, 'benchmarks/p1/smoke_knn.py', '--config', str(config),
         '--device', 'cpu', '--queries', '16'],
        cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-500:]
    assert 'SMOKE OK' in completed.stdout
    assert 'num_neighbors : 215' in completed.stdout
    # The full-scale index estimate must be reported so the run's memory cost is
    # visible before the fact.
    assert 'MiB per horizon' in completed.stdout
