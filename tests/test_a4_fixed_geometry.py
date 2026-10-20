"""P1/A4 control: the Gaussian kernel with geometry frozen to a fixed isotropic.

A4 exists to separate two hypotheses that the A5-vs-A2 comparison confounds:

* **A4 vs A2** — Gaussian distance weighting vs uniform KNN-8 averaging, both
  with fixed geometry.  Isolates the *kernel form*.
* **A5 vs A4** — learnable per-point, per-axis geometry vs fixed isotropic
  geometry.  Isolates the *learnable parameters*.

The implementation relies on a property of the A5 head that is easy to break
silently: ``scale_head``/``opacity_head`` are initialised with a **zero weight**,
so freezing that weight is what makes every point share one geometry.  Note what
this does *not* freeze: the biases stay trainable, so A4 learns the shared scale
and opacity rather than holding them at 0.35/0.02 (measured 0.403/0.405/0.407 and
0.0186 -- see P1-7 and ``test_p1_7_a4_geometry_is_global_not_frozen``).  If
someone later changes the initialisation (or the head gains a non-zero weight),
A4 would quietly stop being "global geometry" while still reporting as A4 --
these tests fail loudly in that case.

CPU-only: everything here runs on the base (non-CUDA) class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.head.point_anchored_semantic_gaussian import PointAnchoredSemanticGaussian  # noqa: E402

PC_RANGE = (-40., -40., -1., 40., 40., 5.4)


def build(freeze_geometry, embed_dims=8, children=3, classes=5, seed=0):
    torch.manual_seed(seed)
    return PointAnchoredSemanticGaussian(
        embed_dims=embed_dims, num_children=children, num_classes=classes,
        pc_range=PC_RANGE, initial_scale=.35, initial_opacity=.02,
        num_neighbors=8, include_self=True, query_chunk_size=4,
        freeze_geometry=freeze_geometry)


def attributes(head, batch=1, queries=2):
    children = head.num_children
    token = torch.randn(batch, queries, head.embed_dims)
    points = torch.zeros(batch, queries, children, 3)
    valid = torch.ones(batch, queries, dtype=torch.bool)
    return head._predict_attributes(token, points, valid)


def test_frozen_geometry_heads_have_zero_weight_and_no_gradient():
    head = build(freeze_geometry=True)
    assert head.freeze_geometry is True
    assert torch.count_nonzero(head.scale_head.weight) == 0
    assert torch.count_nonzero(head.opacity_head.weight) == 0
    assert head.scale_head.weight.requires_grad is False
    assert head.opacity_head.weight.requires_grad is False
    # The biases stay, so the frozen value is the *initial* value, not zero.
    assert head.scale_head.bias.requires_grad is True
    assert head.opacity_head.bias.requires_grad is True


def test_frozen_geometry_still_trains_the_semantic_path():
    head = build(freeze_geometry=True)
    trainable = {name for name, p in head.named_parameters() if p.requires_grad}
    assert 'residual_head.weight' in trainable
    assert 'scale_head.weight' not in trainable
    assert 'opacity_head.weight' not in trainable
    # The position encoder and trunk must remain trainable, otherwise A4 would
    # be a different experiment (a frozen trunk) rather than frozen geometry.
    assert any(name.startswith('trunk.') for name in trainable)
    assert any(name.startswith('position_encoder.') for name in trainable)


def test_frozen_geometry_pins_isotropic_scale_and_constant_opacity():
    """The functional claim: every point gets the same isotropic geometry."""
    head = build(freeze_geometry=True)
    values = attributes(head)
    scales = values['scales']
    opacities = values['opacities']
    assert scales.shape[-1] == 3
    assert torch.allclose(scales, torch.full_like(scales, 0.35), atol=1e-6)
    assert torch.allclose(scales[..., 0], scales[..., 1])
    assert torch.allclose(scales[..., 1], scales[..., 2])
    assert torch.allclose(opacities, torch.full_like(opacities, 0.02), atol=1e-7)


def test_unfrozen_head_is_not_isotropic_once_its_weight_moves():
    """A5's head must actually be able to break isotropy, or A5 vs A4 is empty."""
    head = build(freeze_geometry=False)
    assert head.scale_head.weight.requires_grad is True
    with torch.no_grad():
        head.scale_head.weight.normal_(0.0, 1.0)
    values = attributes(head)
    scales = values['scales']
    per_point_spread = (scales.max(dim=-1).values - scales.min(dim=-1).values)
    assert per_point_spread.max() > 1e-3, 'A5 head failed to produce anisotropy'


def test_freezing_removes_parameters_from_the_optimiser_set():
    frozen = build(freeze_geometry=True)
    free = build(freeze_geometry=False)
    frozen_count = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    free_count = sum(p.numel() for p in free.parameters() if p.requires_grad)
    expected_drop = (frozen.scale_head.weight.numel()
                     + frozen.opacity_head.weight.numel())
    assert free_count - frozen_count == expected_drop


def test_default_is_unfrozen_so_a5_is_unchanged():
    """A4 must be opt-in: the A5 runs already in the report set have no flag."""
    torch.manual_seed(0)
    head = PointAnchoredSemanticGaussian(
        embed_dims=8, num_children=3, num_classes=5, pc_range=PC_RANGE,
        initial_scale=.35, initial_opacity=.02, num_neighbors=8,
        include_self=True, query_chunk_size=4)
    assert head.freeze_geometry is False
    assert head.scale_head.weight.requires_grad is True
    assert head.opacity_head.weight.requires_grad is True
    # At initialisation A5 is isotropic too (zero weight); the difference only
    # appears after training, which is why A4 needs its own training run.
    values = attributes(head)
    assert torch.allclose(values['scales'],
                          torch.full_like(values['scales'], 0.35), atol=1e-6)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
