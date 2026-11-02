import torch
import numpy as np
import pytest
from argparse import ArgumentParser
from mmengine import Config

from train import select_occ3d_eval_mask
from misc.occ3d_eval import (
    add_occ3d_eval_batch, add_occ3d_eval_mask_argument,
    configure_occ3d_eval_mask, get_occ3d_eval_mask_name,
)


def test_occ3d_eval_mask_selects_one_dense_batch_item():
    mask = torch.zeros(2, 2, 2, 2, dtype=torch.bool)
    mask[1] = True
    selected = select_occ3d_eval_mask({'occ_cam_mask': mask}, 1,
                                      {'occ3d_eval_mask': 'camera'})
    assert selected.shape == (8,)
    assert selected.all()


def test_occ3d_eval_mask_selects_one_flattened_batch_item():
    mask = torch.tensor([[True, False, True], [False, True, False]])
    selected = select_occ3d_eval_mask({'occ_cam_mask': mask}, 1,
                                      {'occ3d_eval_mask': 'camera'})
    assert selected.tolist() == [False, True, False]


def test_occ3d_eval_mask_validates_per_rank_prediction_length():
    mask = torch.ones(2, 2, 2, 2, dtype=torch.bool)
    selected = select_occ3d_eval_mask({'occ_cam_mask': mask}, 1,
                                      {'occ3d_eval_mask': 'camera'}, expected_numel=8)
    assert selected.shape == (8,)


@pytest.mark.parametrize('cfg,expected', [
    ({}, 'none'), ({'eval_mask_flag': False}, 'none'),
    ({'eval_mask_flag': True}, 'camera'),
    ({'occ3d_eval_mask': 'none', 'eval_mask_flag': True}, 'none'),
    ({'occ3d_eval_mask': 'camera', 'eval_mask_flag': False}, 'camera'),
])
def test_policy_defaults_to_none_and_honors_explicit_settings(cfg, expected):
    import train
    import eval as eval_script
    assert train.get_occ3d_eval_mask_name(cfg) == expected
    assert eval_script.get_occ3d_eval_mask_name(cfg) == expected
    metadata = {'occ_cam_mask': torch.zeros(1, 8, dtype=torch.bool)}
    if expected == 'none':
        assert select_occ3d_eval_mask(metadata, 0, cfg) is None


def test_cli_overrides_config_and_is_recorded_in_resolved_config():
    parser = ArgumentParser()
    add_occ3d_eval_mask_argument(parser)
    assert parser.parse_args([]).occ3d_eval_mask is None
    cfg = {'occ3d_eval_mask': 'camera'}
    args = parser.parse_args(['--occ3d-eval-mask', 'none'])
    assert configure_occ3d_eval_mask(cfg, args.occ3d_eval_mask) == 'none'
    assert cfg['occ3d_eval_mask'] == 'none'
    args = parser.parse_args(['--occ3d-eval-mask', 'camera'])
    assert configure_occ3d_eval_mask(cfg, args.occ3d_eval_mask) == 'camera'


@pytest.mark.parametrize('mask_name,key', [
    ('camera', 'occ_cam_mask'), ('lidar', 'occ_lidar_mask'), ('nonempty', 'occ_nonempty_mask')])
def test_explicit_mask_is_required_and_validated(mask_name, key):
    cfg = {'occ3d_eval_mask': mask_name}
    with pytest.raises(ValueError, match='missing'):
        select_occ3d_eval_mask({}, 0, cfg)
    with pytest.raises(ValueError, match='size mismatch'):
        select_occ3d_eval_mask({key: torch.ones(1, 3)}, 0, cfg, expected_numel=4)
    with pytest.raises(IndexError, match='batch index'):
        select_occ3d_eval_mask({key: torch.ones(1, 3)}, 1, cfg)


def test_standalone_eval_counts_voxels_outside_camera_mask_by_default():
    from misc.occ3d_nus_metrics import Metric_mIoU
    prediction = torch.tensor([0, 17, 0, 17])
    target = torch.tensor([0, 1, 17, 17])
    camera = torch.tensor([[True, False, False, True]])
    full = Metric_mIoU()
    add_occ3d_eval_batch(full, prediction, target, {'occ_cam_mask': camera}, 0, {})
    masked = Metric_mIoU()
    add_occ3d_eval_batch(masked, prediction, target, {'occ_cam_mask': camera}, 0,
                       {'occ3d_eval_mask': 'camera'})
    assert full.hist.sum() == 4 and masked.hist.sum() == 2
    assert full.hist[1, 17] == 1 and full.hist[17, 0] == 1
    reference = Metric_mIoU(use_image_mask=True)
    reference.add_batch(prediction.numpy(), target.numpy(), None, camera[0].numpy())
    np.testing.assert_array_equal(masked.hist, reference.hist)


@pytest.mark.parametrize('path', [
    'config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem.py',
    'config/adaptive_allocation/adaptive_allocationv7_occ3d.py',
    'config/adaptive_allocation/adaptive_allocationv8_occ3d.py',
    'config/gaussianformer_occ3d.py',
    'config/opus/opusv1_voxel_fusion_t_r50_704x256_8f_occ3d.py',
    'config/full_occ3d_probability/_base_8f.py',
])
def test_generic_occ3d_recipes_default_to_full_volume(path):
    assert get_occ3d_eval_mask_name(Config.fromfile(path)) == 'none'


def test_explicit_camera_experiment_remains_opted_in():
    cfg = Config.fromfile('config/full_occ3d_camera_set/opusv1_gaussian_small_8f_camera_set_1ep.py')
    assert get_occ3d_eval_mask_name(cfg) == 'camera'
