"""Selection repeatability, sparse neighborhood semantics and fusion gradients."""

import copy
import itertools

import pytest
import torch
from torch import nn
from mmengine import Config

from model.encoder.opus_voxel_fusion_encoder import OPUSVoxelFusionEncoder
from model.encoder.opus_encoder import _official_sampling_4d
from model.lifter.fixed_voxel_opus_lifter import FixedVoxelOPUSLifter, _fallback_points
from model.ops.sparse_voxel_sampling import (
    build_voxel_memory, sample_voxel_memory, voxel_fps_indices,
)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    return torch.device(request.param)


def _memory(device, batch=2, channels=8, shape=(4, 4, 4)):
    coords = torch.tensor(list(itertools.product(
        range(batch), range(shape[0]), range(shape[1]), range(shape[2]))),
        dtype=torch.int32, device=device)
    features = torch.randn(len(coords), channels, device=device, requires_grad=True)
    return build_voxel_memory(features, coords, shape, batch,
                               voxel_size=(1., 1., 1.), origin=(0., 0., 0.))


def test_fps_is_repeatable_and_invariant_to_sparse_row_order(device):
    coords = torch.tensor(list(itertools.product(range(3), range(5), range(7))),
                          dtype=torch.int32, device=device)
    reference = voxel_fps_indices(coords, (3, 5, 7), (0.5, 0.5, 0.5), 13)
    expected = coords[reference]
    assert reference.unique().numel() == 13
    assert expected[0].tolist() == [0, 0, 0]
    for _ in range(3):
        permutation = torch.randperm(coords.shape[0], device=device)
        selected = voxel_fps_indices(coords[permutation], (3, 5, 7), (0.5,) * 3, 13)
        assert torch.equal(coords[permutation][selected], expected)


def test_fixed_query_initialization_pairs_features_and_handles_empty_samples(device):
    # Exercise seed selection without allocating a full CPU voxelizer/backbone.
    lifter = FixedVoxelOPUSLifter.__new__(FixedVoxelOPUSLifter)
    nn.Module.__init__(lifter)
    lifter.num_queries, lifter.embed_dims = 4, 2
    lifter.pc_range = (10., 20., -2., 14., 24., 2.)
    lifter.voxel_size = (1.,) * 3
    lifter.register_buffer('fallback_points', _fallback_points(4).to(device))
    indices = torch.tensor([[0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 1]],
                           dtype=torch.int32, device=device)
    features = torch.tensor([[20., 21.], [0., 1.], [10., 11.]],
                            device=device, requires_grad=True)
    # The actual Z shape is smaller than the configured physical range. Query
    # coordinates must follow pitch/origin rather than stretch to spatial_shape.
    memory = build_voxel_memory(features, indices, (3, 4, 4), 2,
                                 (1.,) * 3, lifter.pc_range[:3])
    output = lifter.initialize_queries(memory)
    torch.testing.assert_close(output['query_points'][0, :3], torch.tensor(
        [[.125, .125, .125], [.375, .125, .125], [.625, .125, .125]], device=device))
    torch.testing.assert_close(output['query_features'][0, :3], features[[1, 2, 0]])
    assert output['query_points'].shape == (2, 4, 3)
    assert output['query_valid_mask'].all()
    assert output['query_voxel_valid_mask'].sum().item() == 3
    torch.testing.assert_close(output['query_points'][1], lifter.fallback_points)
    assert output['query_features'][1].count_nonzero().item() == 0
    output['query_features'].sum().backward()
    torch.testing.assert_close(features.grad, torch.ones_like(features))


def _brute_local_pool(positions, memory, radius=1):
    features, indices = memory['features'], memory['indices']
    groups = positions.shape[2]
    channels = features.shape[-1] // groups
    result = torch.zeros((*positions.shape[:-1], channels), device=features.device)
    valid = torch.zeros(positions.shape[:-1], dtype=torch.bool, device=features.device)
    xyz = indices[:, 1:].flip(-1).float()
    for b, q, g, k in itertools.product(*(range(n) for n in positions.shape[:-1])):
        point = positions[b, q, g, k]
        cell = point.floor()
        keep = (indices[:, 0] == b) & ((xyz - cell).abs() <= radius).all(-1)
        if keep.any():
            weight = torch.exp(-.5 * (xyz[keep] + .5 - point).square().sum(-1))
            values = features[keep, g * channels:(g + 1) * channels]
            result[b, q, g, k] = (values * weight[:, None]).sum(0) / weight.sum()
            valid[b, q, g, k] = True
    return result, valid


def test_local_voxel_sampling_matches_brute_force_and_has_gradients(device):
    memory = _memory(device)
    # Add holes, retaining one batch with a different feature field. This
    # catches accidental -1 indexing, border wrapping and cross-batch mixing.
    keep = torch.arange(memory['features'].shape[0], device=device) % 5 != 0
    features = memory['features'][keep].detach().requires_grad_()
    memory = build_voxel_memory(features, memory['indices'][keep], (4, 4, 4), 2,
                                 (1.,) * 3, (0.,) * 3)
    positions = torch.tensor([
        [[[[1.2, 1.3, 1.4], [-.2, .3, .4]],
          [[2.2, 2.3, 2.4], [-10., -10., -10.]]]],
        [[[[1.4, 1.5, 1.6], [3.9, 3.9, 3.9]],
          [[2.4, 2.5, 2.6], [10., 10., 10.]]]],
    ], device=device, requires_grad=True)
    actual, valid = sample_voxel_memory(positions, memory, chunk_size=3)
    expected, expected_valid = _brute_local_pool(positions.detach(), memory)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(valid, expected_valid)
    assert actual[~valid].count_nonzero().item() == 0
    actual.square().sum().backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    assert torch.isfinite(positions.grad).all() and positions.grad.abs().sum() > 0


def test_entirely_empty_memory_returns_zero_without_invalid_gathers(device):
    features = torch.empty(0, 8, device=device, requires_grad=True)
    memory = build_voxel_memory(features, torch.empty(0, 4, dtype=torch.int32, device=device),
                                 (4, 4, 4), 1, (1.,) * 3, (0.,) * 3)
    positions = torch.ones(1, 2, 2, 2, 3, device=device, requires_grad=True)
    sampled, valid = sample_voxel_memory(positions, memory)
    assert sampled.count_nonzero().item() == 0 and not valid.any()
    sampled.sum().backward()
    assert torch.isfinite(positions.grad).all()


def test_temporal_image_scale_weights_stay_with_their_channel_group(device):
    frames, groups, views = 3, 2, 2
    features = []
    for level in range(4):
        values = [100. * level + 10. * time + group
                  for time in range(frames) for group in range(groups)]
        feature = torch.tensor(values, device=device).view(frames * groups, 1, 1, 1, 1)
        features.append(feature.expand(-1, views, 2, 2, 1).contiguous())
    positions = torch.tensor([1., 1., 2.], device=device).expand(1, 1, frames, groups, 1, 3)
    weights = torch.zeros(1, 1, groups, frames, 1, 4, device=device)
    weights[:, :, 0, :, :, 0] = 1.
    weights[:, :, 1, :, :, 1] = 1.
    projection = torch.eye(4, device=device)[None, None].expand(1, frames * views, 4, 4)
    sampled, _ = _official_sampling_4d(
        positions, features, weights, projection,
        torch.full((1, frames * views, 2), 2., device=device), views,
        align_scale_weights=True)
    expected = torch.tensor([[[[[0.], [10.], [20.]],
                               [[101.], [111.], [121.]]]]], device=device)
    torch.testing.assert_close(sampled, expected)


def test_multiframe_fusion_refines_fixed_queries_and_trains_both_modalities(device):
    torch.manual_seed(42)
    encoder = OPUSVoxelFusionEncoder(
        embed_dims=8, num_decoder=2, num_frames=2, num_views=2, num_points=2,
        num_levels=4, num_groups=2, num_heads=2, feedforward_channels=16,
        dropout=0., num_classes=3, num_refines=(1, 2),
        pc_range=(0., 0., 0., 4., 4., 4.), decoder_checkpoint=True).to(device).train()
    memory = _memory(device)
    query = torch.randn(2, 3, 8, device=device, requires_grad=True)
    points = torch.tensor([.35, .35, .4], device=device).expand(2, 3, 3).clone()
    images = [torch.randn(2, 4, 8, 4, 4, device=device, requires_grad=True)
              for _ in range(4)]
    metas = dict(projection_mat=torch.eye(4, device=device)[None, None].expand(2, 4, 4, 4),
                 image_wh=torch.full((2, 4, 2), 4., device=device))
    output = encoder(query, points, images, metas, memory)
    assert encoder.layers[0].mixing.in_points == 6  # (T + 1) * K
    assert encoder.layers[0].mixing.out_points == 6
    states = output['representation']
    assert [tuple(state['query_points'].shape) for state in states] == [
        (2, 3, 1, 3), (2, 3, 2, 3)]
    assert all(state['lidar_sample_valid'].all() for state in states)
    loss = sum(state['opus_logits'].square().mean() + state['query_points'].square().mean()
               for state in states)
    loss.backward()
    for gradient in [query.grad, memory['features'].grad] + [image.grad for image in images]:
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    offset = encoder.layers[0].sampling.sampling_offset.weight.grad
    assert offset is not None and torch.isfinite(offset).all() and offset.abs().sum() > 0
    # Checkpointing must not change forward outputs or the selected positions.
    encoder.eval()
    eager = copy.deepcopy(encoder).train()
    eager.decoder_checkpoint = False
    with torch.no_grad():
        expected = eager(query, points, images, metas, memory)['representation']
    for actual, reference in zip(states, expected):
        torch.testing.assert_close(actual['opus_logits'], reference['opus_logits'])


@pytest.mark.parametrize('name,frames,mask', [
    ('1f_occ3d', 1, 'camera'), ('8f_occ3d', 8, 'camera'),
    ('8f_occ3d_nomask', 8, 'none'),
])
def test_voxel_fusion_configs_are_consistent(name, frames, mask):
    config = Config.fromfile(
        f'config/opus/opusv1_voxel_fusion_t_r50_704x256_{name}.py')
    assert config.model.lifter.type == 'FixedVoxelOPUSLifter'
    assert config.model.lifter.num_queries == 600
    assert config.model.encoder.type == 'OPUSVoxelFusionEncoder'
    assert config.model.encoder.num_frames == frames
    assert config.model.head.decoder_outputs_logits
    assert config.occ3d_eval_mask == mask
    for dataset in (config.train_dataset_config, config.val_dataset_config):
        assert dataset.num_lidar_history == 9 and 'lidar_points' in dataset.return_keys
        target = next(op for op in dataset.pipeline if op.type == 'LoadOccupancyOcc3D')
        assert target.train_mask_type == 'none'
    assert config.model.encoder.num_refines == config.model.head.point_multipliers
