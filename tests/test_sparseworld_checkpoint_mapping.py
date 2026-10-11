import torch

from dataset.sparseworld_trajectory import NuScenesSparseWorldTrajectoryDataset
from model.segmentor.sparseworld_strict_segmentor import (
    SparseWorldStrictSegmentor, _EgoAttention)
from model.head.sparseworld_strict_head import _Rasterizer
from tools.convert_sparseworld_checkpoint import _assigned_stamps, _map_key


def test_sparseworld_checkpoint_key_mapping_covers_special_modules():
    assert _map_key('pts_bbox_head.init_points.weight') == 'lifter.reference_points'
    assert _map_key('pts_bbox_head.num_stamps_all') == 'lifter.stamp_statistics'
    assert _map_key(
        'pts_bbox_head.transformer.decoder.decoder_layers.2.self_attn.attention.attn.in_proj_weight'
    ) == 'encoder.layers.2.self_attn.attention.in_proj_weight'
    assert _map_key(
        'pts_bbox_head.transformer.decoder.decoder_layers.4.ffn.layers.1.weight'
    ) == 'encoder.layers.4.ffn.3.weight'
    assert _map_key('ego_cross_attn.pos_embed.weight') == 'ego_attention.position.weight'


def test_assigned_stamps_obeys_exact_capacities():
    statistics = torch.tensor([
        [9, 1, 0], [8, 1, 0], [7, 2, 0],
        [1, 9, 0], [1, 8, 0], [0, 1, 9],
    ])
    assigned = _assigned_stamps(statistics, (3, 2, 1))
    assert torch.bincount(assigned, minlength=3).tolist() == [3, 2, 1]


def test_ego_attention_preserves_upstream_kernel_and_residual():
    module = _EgoAttention(8, 2, 0., (-4., -4., -1., 4., 4., 1.)).eval()
    ego = torch.randn(1, 1, 8)
    features = torch.randn(1, 3, 8)
    points = torch.rand(1, 3, 2, 3)
    with torch.no_grad():
        actual = module(ego, features, points)
        lower = points.new_tensor(module.pc_range[:3])
        extent = points.new_tensor(module.pc_range[3:]) - lower
        world = points.mean(2) * extent + lower
        ego_point = points.new_full((1, 1, 3), .5)
        distance = -torch.cdist(ego_point * extent + lower, world)
        tau = torch.nn.functional.softplus(module.tau(features)).transpose(1, 2)
        mask = (torch.exp(distance / 40.)[:, None] * tau[:, :, None]).flatten(0, 1)
        query = ego + module.position(torch.zeros_like(ego_point))
        key = features + module.position(world)
        expected = ego + module.attention(
            query, key, features, attn_mask=mask, need_weights=False)[0]
    torch.testing.assert_close(actual, expected)


def test_sparseworld_rasterizer_applies_normalized_compactness_prior():
    rasterizer = _Rasterizer(
        (0., 0., 0., 10., 10., 10.), 1., (10, 10, 10),
        [.5] * 17, 20., False)
    points = torch.tensor([[[1., 1., 1.], [9., 9., 9.]]])
    logits = torch.full((1, 2, 17), -10.)
    logits[..., 15] = 1.
    # The two-point group has normalized radius > .1, reducing both class-15
    # scores below .5. Without the source compactness rule both would survive.
    output = rasterizer(points, logits, group_size=2)
    assert (output != 17).sum() == 0


def test_official_protocol_keyframe_boundary_selection():
    keyframes = [('scene-a', i) for i in range(20)]
    keyframes += [('scene-b', i) for i in range(15)]
    selected = NuScenesSparseWorldTrajectoryDataset._official_protocol_keyframes(
        keyframes)
    assert selected == ([('scene-a', i) for i in range(5, 13)] +
                        [('scene-b', i) for i in range(5, 8)])


def test_sparseworld_history_skips_dense_camera_sweeps():
    frames = [
        {'token': 'k0'}, {}, {}, {'token': 'k1'}, {},
        {'token': 'k2'}, {}, {}, {'token': 'current'},
    ]
    indices = NuScenesSparseWorldTrajectoryDataset._history_indices(
        frames, 8, 3, keyframes_only=True)
    assert indices == [5, 3, 0]


def test_forecast_snapshot_is_an_independent_complete_clone():
    model = SparseWorldStrictSegmentor.__new__(SparseWorldStrictSegmentor)
    model.pc_range = (0., 0., 0., 10., 10., 10.)
    tensors = {
        'features': torch.randn(1, 3, 4, requires_grad=True),
        'positions': torch.rand(1, 3, 2, 3, requires_grad=True),
        'timestamp': torch.zeros(1, 3, 2, 1),
        'ego': torch.randn(1, 1, 4),
        'all_features': torch.randn(1, 5, 4),
        'all_points': torch.rand(1, 5, 2, 3),
        'stamps': torch.tensor([0, 0, 1, 2, 3]),
        'slot_ids': torch.tensor([0, 1, 2]),
    }
    snapshot = model._snapshot_forecast_state(tensors, completed_steps=1)
    assert snapshot['completed_steps'] == 1
    assert snapshot['active_mask'].tolist() == [True, True, True, False, False]
    assert snapshot['tensor_metadata']['features']['requires_grad'] is True
    assert snapshot['features'].requires_grad is False
    torch.testing.assert_close(
        snapshot['positions_m'], tensors['positions'].detach() * 10.)
    tensors['features'].data.zero_()
    assert snapshot['features'].abs().sum() > 0
