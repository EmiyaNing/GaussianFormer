"""Second-round execution contracts and independent CUDA gradient checks."""

import pickle

import numpy as np
import pytest
import torch
from mmengine import Config

from dataset import loader_worker_options
from dataset.react_voxel import PrepareReactVoxels
from dataset.transform_3d import LoadMultiViewImageFromFiles
from dataset.utils import custom_collate_fn_temporal, move_batch_to_device
from model.encoder.gaussian_encoder.ops import (
    DeformableAggregationFunction as DAF,
    DeformableAggregationSumFunction as DAFSum,
)
from model.lifter.voxel_gaussian_lifter import GaussianVoxelLearnear


def test_worker_voxels_match_original_lifter_including_caps_and_batch_coordinates():
    # Include >5 points in one voxel, boundary/out-of-range points, and a
    # different-size second sample. Compare with the original native algorithm.
    transform = PrepareReactVoxels(pc_range=[0., 0., 0., 2., 2., 2.])
    points = np.array([[.02, .02, .02, i] for i in range(9)] +
                      [[.051, .02, .02, 3], [1.99, 1.99, 1.99, 2],
                       [2., 0., 0., 1], [-.01, 0., 0., 1]], dtype=np.float32)
    items = [transform(dict(lidar_points=sample))['react_voxels']
             for sample in (points, points[9:])]
    # Native outputs must remain valid across generator reuse.
    assert items[0]['voxel_num_points'][0] == 5
    lifter = GaussianVoxelLearnear.__new__(GaussianVoxelLearnear)
    torch.nn.Module.__init__(lifter)
    lifter.marker = torch.nn.Parameter(torch.zeros(1))
    lifter.voxel_spec = transform.spec
    lifter._lidar_processor = None
    expected = lifter.process_lidar_data(dict(lidar_points=[points, points[9:]]))
    batch = custom_collate_fn_temporal([dict(react_voxels=item) for item in items])
    actual = lifter.process_lidar_data(batch)
    for name in ('voxels', 'voxel_num_points', 'voxel_coords'):
        torch.testing.assert_close(actual[name], expected[name].to(actual[name].dtype),
                                   atol=0, rtol=0)
    # Spawn workers reconstruct a lazy, private native generator.
    restored = pickle.loads(pickle.dumps(transform))
    assert restored._generator is None
    assert restored.spec == transform.spec
    batch['react_voxels']['spec'] = dict(transform.spec, max_num_points_per_voxel=6)
    with pytest.raises(ValueError, match='do not match'):
        lifter.process_lidar_data(batch)


def test_lean_image_loader_preserves_model_pixels(monkeypatch):
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    monkeypatch.setattr('dataset.transform_3d.mmcv.imread', lambda *args: image.copy())
    baseline = LoadMultiViewImageFromFiles(to_float32=True)(dict(img_filename=['a', 'b']))
    fast = LoadMultiViewImageFromFiles(to_float32=True, keep_original=False)(
        dict(img_filename=['a', 'b']))
    assert 'ori_img' in baseline and 'ori_img' not in fast
    for actual, expected in zip(fast['img'], baseline['img']):
        np.testing.assert_array_equal(actual, expected)


def test_v2_config_keeps_objective_geometry_and_budget():
    v1 = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast.py')
    v2 = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast_v2.py')
    for key in ('loss', 'optimizer', 'input_shape', 'load_from', 'amp'):
        assert v1.get(key) == v2.get(key)
    for key in ('img_backbone', 'img_neck', 'lifter', 'head'):
        assert v1.model[key] == v2.model[key]
    assert v1.model.encoder.operation_order == v2.model.encoder.operation_order
    for name in ('train_dataset_config', 'val_dataset_config'):
        assert v1[name].get('num_lidar_history', 0) == v2[name].get('num_lidar_history', 0)
        assert v1[name].imageset == v2[name].imageset
    assert v2.train_pipeline[0].keep_original is False
    assert v1.train_pipeline[0].get('keep_original', True) is True
    assert 'ori_img' not in v2.occ3d_return_keys
    assert 'react_voxels' in v2.occ3d_return_keys


def test_no_worker_loader_omits_multiprocessing_only_options():
    assert loader_worker_options(dict(num_workers=0, persistent_workers=True,
                                     prefetch_factor=1)) == dict(num_workers=0, pin_memory=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='DAF requires CUDA')
@pytest.mark.parametrize('points_per_query', [1, 3, 9])
def test_fused_daf_output_and_all_input_gradients(points_per_query):
    torch.manual_seed(27)
    batches, queries, cameras, channels, groups = 2, 11, 3, 16, 4
    feature = torch.randn(batches, cameras, 23, channels * 2, device='cuda')[..., ::2]
    feature.requires_grad_()
    shape = torch.tensor([[4, 5], [1, 3]], dtype=torch.int32, device='cuda')
    starts = torch.tensor([0, 20], dtype=torch.int32, device='cuda')
    location = torch.rand(batches, queries * points_per_query, cameras, 2, device='cuda')
    # Exact boundaries and invalid views must contribute neither value nor grad.
    location[:, 0] = 0
    location[:, 1] = 1
    location[:, 2] = -1
    location.requires_grad_()
    weights = torch.randn(batches, queries * points_per_query, cameras, 2, groups,
                          device='cuda', requires_grad=True)
    variables = (feature, location, weights)
    expected = DAF.apply(feature, shape, starts, location, weights).reshape(
        batches, queries, points_per_query, channels).sum(2)
    actual = DAFSum.apply(feature, shape, starts, location, weights, points_per_query)
    probe = torch.randn_like(actual).transpose(0, 1).contiguous().transpose(0, 1)
    expected_grad = torch.autograd.grad(expected, variables, probe)
    actual_grad = torch.autograd.grad(actual, variables, probe)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=2e-5)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='DAF requires CUDA')
def test_daf_follows_current_stream_and_nested_voxels_transfer():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        feature = torch.ones(1, 1, 4, 4, device='cuda', requires_grad=True)
        shape = torch.tensor([[2, 2]], dtype=torch.int32, device='cuda')
        starts = torch.zeros(1, dtype=torch.int32, device='cuda')
        location = torch.full((1, 3, 1, 2), .5, device='cuda', requires_grad=True)
        weights = torch.ones(1, 3, 1, 1, 1, device='cuda', requires_grad=True)
        output = DAFSum.apply(feature, shape, starts, location, weights, 3)
        output.sum().backward()
        raw = [torch.zeros(3, 4)]
        batch = dict(img=torch.zeros(1), lidar_points=raw,
                     react_voxels=dict(voxels=torch.zeros(2, 5, 4), spec={'x': 1}))
        moved = move_batch_to_device(batch, 'cuda', non_blocking=True)
    stream.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 3.))
    torch.testing.assert_close(feature.grad, torch.full_like(feature, .75))
    assert moved['lidar_points'] is raw
    assert moved['react_voxels']['voxels'].is_cuda
    assert moved['react_voxels']['spec'] == {'x': 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='DAF requires CUDA')
def test_fused_daf_empty_query_and_invalid_grouping():
    feature = torch.zeros(1, 1, 4, 4, device='cuda', requires_grad=True)
    shape = torch.tensor([[2, 2]], device='cuda', dtype=torch.int32)
    starts = torch.zeros(1, device='cuda', dtype=torch.int32)
    location = torch.empty(1, 0, 1, 2, device='cuda', requires_grad=True)
    weights = torch.empty(1, 0, 1, 1, 1, device='cuda', requires_grad=True)
    output = DAFSum.apply(feature, shape, starts, location, weights, 9)
    assert output.shape == (1, 0, 4)
    output.sum().backward()
    assert feature.grad.count_nonzero() == 0
    with pytest.raises(RuntimeError, match='divisible'):
        DAFSum.apply(feature, shape, starts, location, weights, 0)
