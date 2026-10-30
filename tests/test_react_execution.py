"""Numerical and gradient checks for opt-in React execution optimizations."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from mmengine import Config

from model.head.gaussian_execution import inverse_gaussian_covariance
from model.head.gaussian_head import GaussianHead
from model.utils.utils import get_rotation_matrix
from model.encoder.gaussian_encoder.topk_module.topk_densify_v2 import DensifyOnly
from model.encoder.gaussian_encoder.utils import GaussianPrediction
from model.encoder.gaussian_encoder.deformable_module import DeformableFeatureAggregation, DAF
from model.lifter.voxel_gaussian_lifter import GaussianVoxelLearnear


def _gaussians(count, device='cpu', dtype=torch.float32):
    values = [torch.rand(1, count, 3, device=device, dtype=dtype) * 1.2 + .2,
              torch.rand(1, count, 3, device=device, dtype=dtype) * .25 + .15,
              torch.randn(1, count, 4, device=device, dtype=dtype),
              torch.rand(1, count, 1, device=device, dtype=dtype),
              torch.rand(1, count, 17, device=device, dtype=dtype)]
    return GaussianPrediction(*(value.detach().requires_grad_() for value in values))


def _concat(parent, delta):
    return GaussianPrediction(*(torch.cat([a, b], dim=1)
                                for a, b in zip(parent[:5], delta[:5])))


def test_covariance_formula_matches_inverse_and_parameter_gradients():
    torch.manual_seed(12)
    gaussian = _gaussians(32, dtype=torch.float64)
    rotation = get_rotation_matrix(gaussian.rotations)
    factor = torch.diag_embed(gaussian.scales) @ rotation
    reference = torch.linalg.inv(factor.transpose(-1, -2) @ factor)
    actual = inverse_gaussian_covariance(gaussian.scales, gaussian.rotations)
    torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)
    probe = torch.randn_like(actual)
    targets = (gaussian.scales, gaussian.rotations)
    expected_grad = torch.autograd.grad((reference * probe).sum(), targets, retain_graph=True)
    actual_grad = torch.autograd.grad((actual * probe).sum(), targets)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-10)


def test_terminal_densify_preserves_gaussians_and_gradients():
    torch.manual_seed(8)
    module = DensifyOnly(feat_embed_dim=8, semantic_dim=17, topk_count=3,
                         pc_range=(0., 0., 0., 2., 2., 2.),
                         scale_range=(.08, .64), unit_xyz=(4., 4., 1.))
    features = torch.randn(1, 7, 8, requires_grad=True)
    anchor = torch.randn(1, 7, 28)
    gaussian = _gaussians(7)
    _, expected, _ = module(features, anchor, gaussian)
    actual, delta = module(features, anchor, gaussian, terminal=True)
    for a, b in zip(actual[:5], expected[:5]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, p, d in zip(actual[:5], gaussian[:5], delta[:5]):
        torch.testing.assert_close(a, torch.cat([p, d], 1), atol=0, rtol=0)
    targets = (features,) + gaussian[:5] + tuple(module.parameters())
    probes = [torch.randn_like(value) for value in actual[:5]]
    expected_grad = torch.autograd.grad(sum((v * p).sum() for v, p in zip(expected[:5], probes)),
                                        targets, retain_graph=True)
    actual_grad = torch.autograd.grad(sum((v * p).sum() for v, p in zip(actual[:5], probes)), targets)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


class _TorchReact(nn.Module):
    """Independent additive renderer including the CUDA AABB support rule."""
    def __init__(self):
        super().__init__()
        self.H = self.W = self.D = 4
        self.grid_size = .4
        self.scale_multiplier = 3.
        self.register_buffer('pc_min', torch.zeros(1, 3))

    def forward(self, pts, means, opacity, semantics, scales, inverse, points_int=None):
        points = pts[0]
        delta = means[0, :, None] - points[None]
        power = -.5 * torch.einsum('gni,gij,gnj->gn', delta, inverse[0], delta)
        point_cells = (points / self.grid_size).int()
        mean_cells = (means.detach()[0] / self.grid_size).int()
        radius = torch.ceil(scales.detach()[0] * self.scale_multiplier / self.grid_size).int()
        lower = (mean_cells - radius).clamp_min(0)
        upper = torch.minimum(mean_cells + radius + 1, radius.new_tensor([4, 4, 4]))
        mask = ((point_cells[None] >= lower[:, None]) &
                (point_cells[None] < upper[:, None])).all(-1)
        weights = power.exp() * opacity[0, :, None] * mask
        return weights.T @ semantics[0]


@pytest.mark.parametrize('backend', ['torch', 'cuda'])
def test_background_and_incremental_render_preserve_logits_and_all_gradients(backend):
    if backend == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    device = 'cuda' if backend == 'cuda' else 'cpu'
    torch.manual_seed(31)
    args = dict(apply_loss_type='all', num_classes=18, with_empty=True,
                empty_args=dict(mean=[.8, .8, .8], scale=[.5, .7, .9]),
                use_localagg_react=True, analytic_covariance=True,
                cuda_kwargs=dict(scale_multiplier=3, H=4, W=4, D=4,
                                 pc_min=[0., 0., 0.], grid_size=.4))
    baseline = GaussianHead(**args).to(device).train()
    fast = GaussianHead(**args, separate_empty=True, incremental_densify=True,
                         cache_render_grid=True).to(device).train()
    fast.load_state_dict(baseline.state_dict(), strict=True)
    if backend == 'torch':
        baseline.aggregator = _TorchReact().to(device)
        fast.aggregator = _TorchReact().to(device)
    first, parent, delta = (_gaussians(n, device) for n in (5, 6, 2))
    final = _concat(parent, delta)
    states = [dict(gaussian=first), dict(gaussian=parent),
              dict(gaussian=final, delta_gaussian=delta, densify_parent_index=1)]
    axis = torch.arange(4, device=device) * .4 + .2
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1)[None]
    metas = dict(occ_xyz=xyz, occ_label=torch.zeros(1, 4, 4, 4, device=device, dtype=torch.long))
    expected = baseline(states, metas)['pred_occ']
    actual = fast(states, metas)['pred_occ']
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=2e-5)
    probes = [torch.randn_like(value) for value in expected]
    variables = first[:5] + parent[:5] + delta[:5]
    expected_grad = torch.autograd.grad(sum((v * p).sum() for v, p in zip(expected, probes)),
                                        variables + (baseline.empty_scalar,), retain_graph=True)
    actual_grad = torch.autograd.grad(sum((v * p).sum() for v, p in zip(actual, probes)),
                                      variables + (fast.empty_scalar,))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=5e-4)
    assert fast._cached_points_int.shape == (64, 3)
    # The integer/empty-field cache must be reset when loading another state.
    fast.load_state_dict(fast.state_dict())
    assert fast._cached_empty_weights.numel() == 0


def test_fast_config_preserves_model_budget_data_and_objective():
    base = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d.py')
    fast = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast.py')
    for key in ('loss', 'optimizer', 'train_dataset_config', 'val_dataset_config',
                'train_loader', 'val_loader', 'input_shape', 'load_from'):
        assert base[key] == fast[key]
    for key in ('img_backbone', 'img_neck', 'lifter'):
        assert base.model[key] == fast.model[key]
    assert base.model.encoder.operation_order == fast.model.encoder.operation_order
    assert fast.model.head.apply_loss_type == 'all'


def test_existing_voxel_lifter_accepts_ragged_numpy_and_tensor_points():
    # The current collate function preserves NumPy LiDAR arrays as a ragged
    # list; the old lifter unconditionally called .cpu() on each array.
    lifter = GaussianVoxelLearnear.__new__(GaussianVoxelLearnear)
    nn.Module.__init__(lifter)
    lifter.device_marker = nn.Parameter(torch.zeros(1))

    def voxelize(points):
        assert points.dtype == np.float32 and points.flags.c_contiguous
        return points[None].copy(), np.zeros((1, 3), dtype=np.int32), np.array([len(points)], dtype=np.int32)

    lifter.lidar_processor = SimpleNamespace(generate=voxelize)
    points = np.arange(20, dtype=np.float32).reshape(5, 4)
    numpy_result = lifter.process_lidar_data(dict(lidar_points=[points]))
    tensor_result = lifter.process_lidar_data(dict(lidar_points=[torch.from_numpy(points)]))
    for key in ('voxels', 'voxel_num_points', 'voxel_coords'):
        torch.testing.assert_close(numpy_result[key], tensor_result[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='DAF requires CUDA')
@pytest.mark.parametrize('fuse_point_sum', [False, True])
def test_preformatted_image_features_preserve_sampling_and_gradients(fuse_point_sum):
    torch.manual_seed(71)
    module = DeformableFeatureAggregation(
        embed_dims=8, num_groups=2, num_levels=2, num_cams=2, attn_drop=.15,
        use_deformable_func=True, use_camera_embed=True,
        kps_generator=dict(type='SparseGaussian3DKeyPointsGenerator', embed_dims=8,
                           num_learnable_pts=2, fix_scale=[[0., 0., 0.]],
                           pc_range=[0., 0., 0., 2., 2., 2.], scale_range=[.08, .64]),
    ).cuda().train()
    features = torch.randn(1, 7, 8, device='cuda', requires_grad=True)
    embedding = torch.randn_like(features, requires_grad=True)
    anchor = torch.randn(1, 7, 28, device='cuda', requires_grad=True)
    maps = [torch.randn(1, 2, 8, 4, 4, device='cuda', requires_grad=True)
            for _ in range(2)]
    metas = dict(projection_mat=torch.eye(4, device='cuda')[None, None].expand(1, 2, 4, 4),
                 image_wh=torch.full((1, 2, 2), 4., device='cuda'))
    targets = (features, embedding, anchor) + tuple(maps) + tuple(module.parameters())
    torch.manual_seed(91)
    expected = module(features, anchor, embedding, maps, metas)
    probe = torch.randn_like(expected)
    expected_grad = torch.autograd.grad((expected * probe).sum(), targets, retain_graph=True)
    prepared = DAF.feature_maps_format(maps)
    prepared = (prepared[0].contiguous().float(), prepared[1].int(), prepared[2].int())
    module.fuse_point_sum = fuse_point_sum
    torch.manual_seed(91)
    actual = module(features, anchor, embedding, maps, metas, formatted_feature_maps=prepared)
    actual_grad = torch.autograd.grad((actual * probe).sum(), targets)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
