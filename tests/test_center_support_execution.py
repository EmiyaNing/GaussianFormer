"""Scientific accounting and interruption controls for the real experiment queue."""
import json
from pathlib import Path

import numpy as np
import pytest
from mmengine import Config

from tools.center_support_common import write_json
from tools.eval_center_support import RayCounts, official_pose_infos
from tools.diagnose_center_support import set_objective_configs


def test_invalid_measurement_cannot_erase_completed_scene_checkpoint(tmp_path):
    path = tmp_path / 'result.json'
    write_json(path, {'status': 'running', 'pairs': [{'scene': 'saved'}]})
    with pytest.raises(ValueError):
        write_json(path, {'measurement': float('nan')})
    assert json.loads(path.read_text())['pairs'] == [{'scene': 'saved'}]
    assert not list(tmp_path.glob('*.tmp'))


def test_official_ray_thresholds_are_strict_and_free_predictions_count():
    gt = np.array([[0, 2], [0, 2], [1, 3], [1, 3], [1, 3]], dtype=float)
    pred = np.array([[0, 3], [0, 2], [1, 7], [17, 3], [1, 6]], dtype=float)
    counts = RayCounts()
    counts.add(pred[:2], gt[:2])
    counts.add(pred[2:], gt[2:])
    assert counts.pred[17] == 1
    assert counts.iou()[0, 0] == pytest.approx(1 / 3)
    assert counts.iou()[1, 0] == 1
    assert counts.iou()[2, 1] == pytest.approx(1 / 4)
    assert counts.iou()[0, 1] == counts.iou()[1, 1] == 0


def test_pose_adapter_uses_full_keyframes_and_skips_camera_only_sweeps():
    def frame(token, x):
        return {'token': token, 'data': {'LIDAR_TOP': {
            'pose': {'translation': [x, 0, 0], 'rotation': [1, 0, 0, 0]},
            'calib': {'translation': [1, 0, 2], 'rotation': [1, 0, 0, 0]}}}}
    infos = {'scene': [frame('a', 0), {'data': {}}, frame('b', 10)]}
    converted = official_pose_infos(infos, [('scene', 2), ('scene', 0)])
    assert [f['token'] for f in converted] == ['a', 'b']
    assert converted[1]['ego2global_translation'] == [10, 0, 0]
    assert converted[1]['lidar2ego_translation'] == [1, 0, 2]


def test_low_budget_changes_only_final_children_and_retains_training_protocol():
    root = Path(__file__).resolve().parents[1]
    high = Config.fromfile(str(root / 'config/center_support_contribution1/a1_align_final.py'))
    low = Config.fromfile(str(root / 'config/center_support_contribution1/low_budget_19200.py'))
    assert low.model.lifter.num_queries * low.model.encoder.num_refines[-1] == 19200
    low.model.encoder.num_refines[-1] = high.model.encoder.num_refines[-1]
    assert low.model == high.model
    for key in ('loss', 'optimizer', 'load_from', 'max_epochs', 'train_dataset_config', 'val_dataset_config'):
        assert low[key] == high[key]


@pytest.mark.parametrize('name', ['a1_align_final', 'low_budget_19200'])
def test_mini_diagnostic_retains_distinct_final_stage_supervision_domains(name):
    import torch
    from loss import OPENOCC_LOSS

    root = Path(__file__).resolve().parents[1]
    config = Config.fromfile(str(root / f'config/center_support_contribution1/{name}.py'))
    original = config.loss.loss_cfgs[0].to_dict()
    settings = set_objective_configs(config)
    inputs = {
        'opus_pred_points': [torch.tensor([[[0., 0., 0.]]], requires_grad=True)],
        'opus_pred_logits': [torch.zeros(1, 1, 17, requires_grad=True)],
        'metas': {
            'occ_xyz': torch.tensor([[[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]]]),
            'occ_label': torch.tensor([[1, 2, 17]]),
            'occ_cam_mask': torch.tensor([[True, False, True]]),
        },
    }
    measured = {}
    for domain, cfg in settings.items():
        assert cfg['gt_domain'] == cfg['final_gt_domain'] == domain
        assert cfg['stage_weights'] == original['stage_weights'][-1:]
        terms, counts = OPENOCC_LOSS.build(cfg).components(inputs)
        measured[domain] = counts['stage_0/gt_count']
        assert torch.isfinite(sum(terms.values()))
    assert measured == {'all': 2, 'camera': 1}
    assert config.loss.loss_cfgs[0].to_dict() == original


def test_historical_diagnostic_preserves_objective_weights():
    root = Path(__file__).resolve().parents[1]
    cfg = Config.fromfile(str(root / 'config/full_occ3d_10e/opusv1_gaussian_softplus_learned_bias_no_benchmark.py'))
    original = cfg.loss.loss_cfgs[0].to_dict()
    for domain, settings in set_objective_configs(cfg).items():
        expected = dict(original, type='CenterSupportDiagnosticSetLoss',
                        stage_weights=original['stage_weights'][-1:],
                        gt_domain=domain, final_gt_domain=domain)
        assert settings == expected
