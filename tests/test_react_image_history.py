"""Temporal correctness: alignment, padding, gradients, caching and DAF masks."""

from collections import OrderedDict
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn
from mmengine import Config

from dataset.react_image_history import LoadReactImageHistory
from model.segmentor.react_temporal_segmentor import ReactTemporalSegmentor
from model.encoder.gaussian_encoder.deformable_module import DeformableFeatureAggregation
from model.encoder.gaussian_encoder.temporal_deformable import TemporalDeformableFeatureAggregation


def _history_input(current, cameras=2):
    names = [f'CAM_{index}' for index in range(cameras)]
    frames = []
    for index in range(4):
        sensors = {name: dict(
            filename=f'{index}_{name}.jpg',
            calib=dict(camera_intrinsic=np.eye(3), rotation=[1, 0, 0, 0], translation=[0, 0, 0]),
            pose=dict(rotation=[1, 0, 0, 0], translation=[index, 0, 0])) for name in names}
        sensors['LIDAR_TOP'] = {}
        frames.append(dict(token=f'frame{index}', timestamp=index * 500000, data=sensors))
    pose = np.eye(4)
    pose[0, 3] = current
    return dict(
        history_context=dict(scene_infos={'scene': frames}, scene_token='scene',
                             frame_index=current, data_path='/images', sensor_types=names),
        img=[np.full((4, 5, 3), current, dtype=np.float32) for _ in names],
        img_filename=[f'/images/{current}_{name}.jpg' for name in names],
        lidar2img=[np.eye(4) for _ in names], ego2img=[np.eye(4) for _ in names],
        ego_pose=pose, lidar_pose=pose, aug_configs=(1., (5, 4), (0, 0, 5, 4), False, 0.))


@pytest.mark.parametrize('current,valid', [(0, [True, False, False, False]),
                                          (1, [True, True, False, False]),
                                          (3, [True, True, True, True])])
def test_history_loader_is_causal_aligned_and_masks_padding(monkeypatch, current, valid):
    reads = []

    def imread(path, *args):
        reads.append(path)
        index = int(path.rsplit('/', 1)[-1].split('_')[0])
        return np.full((4, 5, 3), index, dtype=np.float32)

    monkeypatch.setattr('dataset.transform_3d.mmcv.imread', imread)
    loader = LoadReactImageHistory(num_history=3, num_cams=2, test_mode=True)
    result = loader(_history_input(current))
    indices = result['image_frame_indices']
    assert len(result['img']) == len(result['img_filename']) == len(result['ego2img']) == 8
    assert result['image_frame_valid'].tolist() == valid
    assert result['num_real_history_frame'] == current
    assert indices[0] == current and np.all(indices <= current)
    assert len(reads) == current * 2 and len(set(reads)) == len(reads)
    np.testing.assert_allclose(result['image_time_offsets'], (current - indices) * .5)
    for frame, index in enumerate(indices):
        assert result['img'][frame * 2][0, 0, 0] == index
        assert result['ego2img'][frame * 2][0, 3] == current - index
    # Padding is a copy, so per-image transforms cannot mutate another frame.
    if current < 3:
        assert not np.shares_memory(result['img'][-1], result['img'][current * 2 + 1])


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return [self.conv(images)]


class _Neck(nn.Module):
    start_level = 0

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)

    def forward(self, features):
        return [self.conv(features[0])]


def _segmentor(checkpoint=True, detach=True, cache=4):
    network = ReactTemporalSegmentor.__new__(ReactTemporalSegmentor)
    nn.Module.__init__(network)
    network.img_backbone = _Backbone()
    network.img_neck = _Neck()
    network.img_backbone_out_indices = [0]
    network.image_channels_last = False
    network.image_backbone_bf16 = True
    network.freeze_img_backbone = network.freeze_img_neck = False
    network.frozen_modules_eval = False
    network.num_image_frames, network.num_cameras = 4, 2
    network.detach_history, network.image_checkpoint = detach, checkpoint
    network.image_cache_size = cache
    network.pack_temporal_features = False
    network._image_cache = OrderedDict()
    network._image_cache_scene = None
    network._image_cache_hits = network._image_cache_misses = 0
    return network


def _cache_metas(keys=(3, 2, 1, 0), scene='scene', augmentation='crop-a'):
    return dict(image_frame_keys=[[tuple(f'{frame}-cam{cam}' for cam in range(2))
                                   for frame in keys]],
                scene_token=[scene], image_aug_key=[augmentation])


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('detach', [False, True])
def test_history_stop_grad_and_fp32_neck(checkpoint, detach):
    torch.manual_seed(4)
    network = _segmentor(checkpoint, detach)
    network.train()
    images = torch.randn(2, 8, 3, 5, 7, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        features = network.extract_img_feat(images)['ms_img_feats'][0]
    assert features.dtype == torch.float32 and features.shape == (2, 8, 4, 5, 7)
    features.square().mean().backward()
    gradients = images.grad.flatten(2).abs().sum(-1)
    assert torch.all(gradients[:, :2] > 0)
    if detach:
        assert torch.all(gradients[:, 2:] == 0)
    else:
        assert torch.all(gradients[:, 2:] > 0)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in network.parameters())
    assert network.image_cache_stats()['frames'] == 0


def test_cache_matches_uncached_and_invalidates_on_augmentation_scene_weights_and_train():
    torch.manual_seed(4)
    network = _segmentor()
    network.eval()
    frames = [torch.randn(1, 2, 3, 5, 7) for _ in range(5)]
    images = torch.cat([frames[index] for index in (3, 2, 1, 0)], dim=1)
    with torch.no_grad():
        expected = network.extract_img_feat(images)['ms_img_feats'][0]
        actual = network.extract_img_feat(images, _cache_metas())['ms_img_feats'][0]
        again = network.extract_img_feat(images, _cache_metas())['ms_img_feats'][0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(again, expected)
        assert network.image_cache_stats()['hits'] == 4
        network.img_backbone.calls = 0
        next_images = torch.cat([frames[index] for index in (4, 3, 2, 1)], dim=1)
        cached = network.extract_img_feat(next_images, _cache_metas((4, 3, 2, 1)))['ms_img_feats'][0]
        assert network.img_backbone.calls == 1
        assert network.image_cache_stats()['frames'] == 4
        uncached = network.extract_img_feat(next_images)['ms_img_feats'][0]
        torch.testing.assert_close(cached, uncached)
        network.img_backbone.calls = 0
        network.extract_img_feat(images, _cache_metas(augmentation='crop-b'))
        assert network.img_backbone.calls == 4
        network.extract_img_feat(images, _cache_metas(scene='new-scene'))
        assert all(key[0] == 'new-scene' for key in network._image_cache)
    network.load_state_dict(network.state_dict())
    assert network.image_cache_stats()['frames'] == 0
    with torch.no_grad():
        network.extract_img_feat(images, _cache_metas())
    network.train()
    assert network.image_cache_stats()['frames'] == 0
    with pytest.raises(ValueError, match='cameras'):
        network.extract_img_feat(images[:, :2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA sampling required')
@pytest.mark.parametrize('fused', [False, True])
def test_temporal_daf_padding_and_disabled_history_match_single_frame_and_learn_bias(fused):
    torch.manual_seed(17)
    options = dict(embed_dims=8, num_groups=2, num_levels=2,
                   use_deformable_func=True, use_camera_embed=True, fuse_point_sum=True,
                   batched_projection=True, factor_camera_weights=True,
                   compact_weight_mask=True, fused_weight_softmax=True,
                   fuse_normalized_sampling=fused, attn_drop=0., residual_mode='cat',
                   kps_generator=dict(type='SparseGaussian3DKeyPointsGenerator', embed_dims=8,
                                      num_learnable_pts=1, fix_scale=[[0., 0., 0.]],
                                      pc_range=[0., 0., 1., 2., 2., 3.], scale_range=[.08, .64]))
    temporal = TemporalDeformableFeatureAggregation(num_image_frames=4, num_cameras=2,
                                                   **deepcopy(options)).cuda()
    single = DeformableFeatureAggregation(num_cams=2, **deepcopy(options)).cuda()
    single.load_state_dict({key: value for key, value in temporal.state_dict().items()
                            if key != 'history_logit_bias'}, strict=True)
    features = torch.randn(1, 11, 8, device='cuda', requires_grad=True)
    embedding = torch.randn_like(features, requires_grad=True)
    anchor = torch.zeros(1, 11, 23, device='cuda', requires_grad=True)
    images = [torch.randn(1, 8, 8, 7, 9, device='cuda', requires_grad=True),
              torch.randn(1, 8, 8, 4, 5, device='cuda', requires_grad=True)]
    metas = dict(projection_mat=torch.eye(4, device='cuda')[None, None].repeat(1, 8, 1, 1),
                 image_wh=torch.ones(1, 8, 2, device='cuda') * 4,
                 image_frame_valid=torch.tensor([[True, False, False, False]], device='cuda'))
    reference = single(features, anchor, embedding, [image[:, :2] for image in images], metas)
    actual = temporal(features, anchor, embedding, images, metas)
    torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-5)
    metas['image_frame_valid'][:] = True
    temporal.history_enabled = False
    torch.testing.assert_close(temporal(features, anchor, embedding, images, metas),
                               reference, atol=3e-6, rtol=3e-5)
    temporal.history_enabled = True
    output = temporal(features, anchor, embedding, images, metas)
    output.square().mean().backward()
    assert torch.isfinite(temporal.history_logit_bias.grad).all()
    assert temporal.history_logit_bias.grad.abs().sum() > 0
    assert images[0].grad[:, 2:].abs().sum() > 0


def test_img4f_config_uses_trained_crop_complete_history_and_explicit_masks():
    cfg = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f.py')
    assert cfg.input_shape == (704, 256) and cfg.occ3d_eval_mask == 'none'
    assert cfg.model.encoder.deformable_model.num_cams == 24
    for settings in (cfg.train_dataset_config, cfg.val_dataset_config):
        assert settings.num_lidar_history == 9
        assert settings.data_aug_conf.final_dim == (256, 704)
        assert 'image_frame_valid' in settings.return_keys
        assert settings.pipeline[1].type == 'LoadReactImageHistory'
        assert settings.pipeline[1].num_history == 3
    assert not cfg.train_dataset_config.pipeline[1].test_mode
    assert cfg.val_dataset_config.pipeline[1].test_mode


def test_img10f_config_keeps_lidar_budget_crop_and_default_eval_mask():
    cfg = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem_lidar10f_img10f_cuda.py')
    assert cfg.model.num_image_frames == 10 and cfg.model.image_cache_size >= 10
    assert cfg.model.encoder.deformable_model.num_cams == 60
    assert cfg.input_shape == (704, 256) and cfg.occ3d_eval_mask == 'none'
    for settings in (cfg.train_dataset_config, cfg.val_dataset_config):
        assert settings.num_lidar_history == 9
        assert settings.pipeline[1].num_history == 9


class _DistributedFeatures(nn.Module):
    def __init__(self, detach, checkpoint):
        super().__init__()
        self.temporal = ReactTemporalSegmentor(
            num_image_frames=4, num_cameras=2, detach_history=detach,
            image_checkpoint=checkpoint, image_cache_size=0,
            img_backbone_out_indices=[0, 1],
            img_backbone=dict(type='ResNet', depth=18, stem_channels=8, base_channels=8,
                              num_stages=2, strides=(1, 2), dilations=(1, 1),
                              out_indices=(0, 1), frozen_stages=-1,
                              norm_cfg=dict(type='BN2d', requires_grad=False),
                              norm_eval=True, with_cp=True),
            img_neck=dict(type='FPN', in_channels=[8, 16], out_channels=4, num_outs=2))

    def forward(self, images):
        return sum(feature.square().mean()
                   for feature in self.temporal.extract_img_feat(images)['ms_img_feats'])


def _distributed_history_worker(rank, rendezvous, detach, checkpoint):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(42)
        network = _DistributedFeatures(detach, checkpoint)
        network.train()
        ddp = nn.parallel.DistributedDataParallel(network, broadcast_buffers=False,
                                                 find_unused_parameters=False)
        optimizer = torch.optim.SGD(network.parameters(), lr=.01)
        for step in range(2):
            torch.manual_seed(100 + rank + step)
            images = torch.randn(1, 8, 3, 16, 16, requires_grad=True)
            eager = deepcopy(network)
            eager.temporal.image_checkpoint = False
            for module in eager.temporal.img_backbone.modules():
                if hasattr(module, 'with_cp'):
                    module.with_cp = False
            eager_images = images.detach().clone().requires_grad_()
            expected = eager(eager_images)
            expected.backward()
            optimizer.zero_grad(set_to_none=True)
            actual = ddp(images)
            torch.testing.assert_close(actual, expected)
            actual.backward()
            torch.testing.assert_close(images.grad, eager_images.grad)
            if detach:
                assert images.grad[:, 2:].count_nonzero() == 0
            else:
                assert torch.all(images.grad[:, 2:].flatten(2).abs().sum(-1) > 0)
            for (name, parameter), (_, reference) in zip(network.named_parameters(), eager.named_parameters()):
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None, name
                gradient = reference.grad.clone()
                dist.all_reduce(gradient)
                torch.testing.assert_close(parameter.grad, gradient / 2)
            optimizer.step()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo unavailable')
@pytest.mark.parametrize('detach,checkpoint', [(True, True), (False, True), (True, False), (False, False)])
def test_history_backbone_gradients_with_two_distributed_ranks(tmp_path, detach, checkpoint):
    torch.multiprocessing.spawn(_distributed_history_worker,
                                args=((tmp_path / 'history_ddp').as_uri(), detach, checkpoint),
                                nprocs=2, join=True)
