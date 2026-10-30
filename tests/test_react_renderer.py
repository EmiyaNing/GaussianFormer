"""Exact support, autograd and cache contracts for the tiled React renderer."""

import pytest
import torch
from mmengine import Config

from model.head.react_tiled_aggregate import TiledReactAggregator
from model.head.react_local_aggregate import PreparedReactAggregator
from model.head.gaussian_head import GaussianHead
from model.encoder.gaussian_encoder.utils import GaussianPrediction


def settings():
    # Include partial tiles on all axes.
    return dict(H=7, W=6, D=5, pc_min=[-.4, -.4, -.4], grid_size=.4,
                scale_multiplier=3, validate_inputs=False)


def inputs(count=11, duplicate=False):
    torch.manual_seed(172)
    size = (7, 6, 5)
    cells = torch.cartesian_prod(*(torch.arange(n, device='cuda') for n in size)).int()
    # Permutation/subsets exercise map lookup rather than canonical indexing.
    cells = cells[torch.randperm(len(cells), device='cuda')[:109]]
    if duplicate:
        cells = torch.cat([cells, cells[:5]])
    points = (cells.float() * .4 - .4 + .2)[None]
    points[:, -5:] += .015  # multiple different points can share a voxel
    means = (torch.rand(1, count, 3, device='cuda') *
             torch.tensor(size, device='cuda') * .4 - .4).requires_grad_()
    scales = torch.rand(1, count, 3, device='cuda') * .55 + .08
    opacity = torch.rand(1, count, device='cuda', requires_grad=True)
    semantics = torch.randn(1, count, 18, device='cuda', requires_grad=True)
    matrix = torch.randn(1, count, 3, 3, device='cuda') * .3
    inverse = (matrix.transpose(-1, -2) @ matrix +
               torch.eye(3, device='cuda') * 2).detach().requires_grad_()
    return [points, means, opacity, semantics, scales, inverse], cells


def reference(values, cells):
    pts, means, opacity, semantics, scales, inverse = values
    config = settings()
    centers = ((means.detach()[0] - means.new_tensor(config['pc_min'])) / .4).int()
    radii = torch.ceil(scales.detach()[0] * 3 / .4).int()
    support = ((cells[None] >= centers[:, None] - radii[:, None]) &
               (cells[None] <= centers[:, None] + radii[:, None])).all(-1)
    delta = means[0, :, None] - pts[0, None]
    # Match the renderer's packed upper-triangular quadratic convention.
    dx, dy, dz = delta.unbind(-1)
    cov = inverse[0]
    power = -.5 * (cov[:, None, 0, 0]*dx.square() + cov[:, None, 1, 1]*dy.square() +
                   cov[:, None, 2, 2]*dz.square())
    power -= (cov[:, None, 0, 1]*dx*dy + cov[:, None, 1, 2]*dy*dz + cov[:, None, 0, 2]*dx*dz)
    weights = power.exp() * support * opacity[0, :, None]
    return weights.T @ semantics[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='renderer requires CUDA')
@pytest.mark.parametrize('tile,hint,duplicate', [(1, 5, False), (2, 5, False),
                                              (4, 5, False), (8, 5, False),
                                              (2, 0, True), (4, 0, True)])
def test_support_and_all_gradients_against_independent_reference(tile, hint, duplicate):
    values, cells = inputs(duplicate=duplicate)
    backend = TiledReactAggregator(**settings(), tile_size=tile, radius_hint=hint).cuda()
    variables = [values[index] for index in (1, 2, 3, 5)]
    expected = reference(values, cells)
    actual = backend(*values, points_int=cells)
    if hint == 0:
        from model.head.react_tiled_aggregate import extension
        with torch.no_grad():
            prepared = backend._prepare_inputs(*values, points_int=cells)
            capacity = backend.capacity_for(values[1].shape[1])
            offsets, _ = extension().tile_bins(prepared[3], prepared[6], backend.H,
                backend.W, backend.D, tile, capacity)
            assert int(offsets[-1]) > capacity  # exercise the exact fallback
    probe = torch.randn_like(expected).T.contiguous().T  # non-contiguous gradient
    expected_grad = torch.autograd.grad(expected, variables, probe)
    actual_grad = torch.autograd.grad(actual, variables, probe)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=4e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='renderer requires CUDA')
def test_matches_legacy_with_unique_sparse_queries():
    values, cells = inputs()
    legacy = PreparedReactAggregator(**settings()).cuda()
    tiled = TiledReactAggregator(**settings()).cuda()
    expected = legacy(*values, points_int=cells)
    actual = tiled(*values, points_int=cells)
    variables = [values[index] for index in (1, 2, 3, 5)]
    probe = torch.randn_like(actual)
    eg = torch.autograd.grad(expected, variables, probe)
    ag = torch.autograd.grad(actual, variables, probe)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for a, b in zip(ag, eg):
        torch.testing.assert_close(a, b, atol=5e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='renderer requires CUDA')
def test_query_cache_invalidates_and_preserves_live_backward_graphs():
    backend = TiledReactAggregator(**settings()).cuda()
    values, cells = inputs(duplicate=True)
    first = backend(*values, points_int=cells)
    heads = backend._query_heads
    assert backend.prepare_query_map(cells)[0] is heads
    cells.copy_(cells.flip(0))
    second_values = [value.detach().clone().requires_grad_(value.requires_grad) for value in values]
    second_values[0] = (cells.float() * .4 - .4 + .2)[None]
    second = backend(*second_values, points_int=cells)
    assert backend._query_heads is not heads
    backend.load_state_dict(backend.state_dict(), strict=True)
    assert backend._query_heads.numel() == 0
    assert not any('_query' in key for key in backend.state_dict())
    # Old contexts retain their own topology, independent of current buffers.
    first.sum().backward()
    second.sum().backward()
    assert torch.isfinite(values[1].grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='renderer requires CUDA')
@pytest.mark.parametrize('empty_queries,empty_gaussians', [(True, False), (False, True), (True, True)])
def test_empty_inputs(empty_queries, empty_gaussians):
    values, cells = inputs(count=0 if empty_gaussians else 2)
    if empty_queries:
        values[0], cells = values[0][:, :0], cells[:0]
    backend = TiledReactAggregator(**settings()).cuda()
    output = backend(*values, points_int=cells)
    assert output.shape == (len(cells), 18)
    assert output.count_nonzero() == 0
    output.sum().backward()
    for index in (1, 2, 3, 5):
        assert values[index].grad.count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='renderer requires CUDA')
def test_current_stream_and_inference_mode():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values, cells = inputs(duplicate=True)
        backend = TiledReactAggregator(**settings()).cuda()
        actual = backend(*values, points_int=cells)
        expected = reference(values, cells)
        actual.sum().backward()
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    with torch.inference_mode():
        value = backend(*[v.detach().clone() for v in values], points_int=cells.clone())
    torch.testing.assert_close(value, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='renderer requires CUDA')
def test_head_background_incremental_render_and_rotation_scale_gradients():
    torch.manual_seed(39)
    config = dict(apply_loss_type='all', num_classes=18, with_empty=True,
                  empty_args=dict(mean=[.8,.8,.8], scale=[.5,.7,.9]),
                  use_localagg_react=True, analytic_covariance=True,
                  separate_empty=True, incremental_densify=True, cache_render_grid=True,
                  cuda_kwargs=dict(scale_multiplier=3, H=4, W=4, D=4,
                                   pc_min=[0.,0.,0.], grid_size=.4, validate_inputs=False))
    legacy = GaussianHead(**config).cuda().train()
    tiled = GaussianHead(**config, react_renderer='tiled').cuda().train()
    tiled.load_state_dict(legacy.state_dict(), strict=True)

    def gaussian(count):
        values = [torch.rand(1,count,3,device='cuda') + .2,
                  torch.rand(1,count,3,device='cuda') * .25 + .15,
                  torch.randn(1,count,4,device='cuda'),
                  torch.rand(1,count,1,device='cuda'),
                  torch.rand(1,count,17,device='cuda')]
        return GaussianPrediction(*(x.detach().requires_grad_() for x in values))

    parent, child = gaussian(9), gaussian(2)
    final = GaussianPrediction(*(torch.cat([a,b],1) for a,b in zip(parent[:5],child[:5])))
    states = [dict(gaussian=parent), dict(gaussian=final, delta_gaussian=child,
                                        densify_parent_index=0)]
    axis = torch.arange(4,device='cuda') * .4 + .2
    xyz = torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1)[None]
    metas = dict(occ_xyz=xyz, occ_label=torch.zeros(1,4,4,4,device='cuda',dtype=torch.long))
    expected, actual = [head(states,metas)['pred_occ'] for head in (legacy,tiled)]
    probes = [torch.randn_like(value) for value in expected]
    variables = parent[:5] + child[:5]
    eg = torch.autograd.grad(sum((a*b).sum() for a,b in zip(expected,probes)),
                             variables + (legacy.empty_scalar,), retain_graph=True)
    ag = torch.autograd.grad(sum((a*b).sum() for a,b in zip(actual,probes)),
                             variables + (tiled.empty_scalar,))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    for a,b in zip(ag,eg):
        torch.testing.assert_close(a,b,atol=2e-4,rtol=3e-4)


def test_config_preserves_second_round_data_budget_and_objective():
    v2 = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast_v2.py')
    v3 = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast_v3.py')
    for key in ('loss', 'optimizer', 'train_dataset_config', 'val_dataset_config',
                'train_loader', 'val_loader', 'input_shape', 'amp', 'load_from'):
        assert v2.get(key) == v3.get(key)
    for key in ('lifter', 'encoder', 'img_backbone', 'img_neck'):
        assert v2.model[key] == v3.model[key]
    assert v3.model.head.react_renderer == 'tiled'
