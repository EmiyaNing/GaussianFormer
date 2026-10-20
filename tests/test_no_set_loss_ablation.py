"""Loss removal must be complete; calibration must not select on validation."""
from pathlib import Path

import pytest
import torch
from mmengine import Config

from tools.diagnose_opus_v1_gaussian import (
    calibrated_predictions, choose_empty_shift, training_selection,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_stage_set_loss_is_removed_and_every_other_setting_matches():
    from loss import OPENOCC_LOSS
    baseline = Config.fromfile(str(ROOT / 'config/center_support_contribution1/a0_original.py')).to_dict()
    ablation = Config.fromfile(str(ROOT / 'config/mini_occ3d_4096_1024/opusv1_gaussian_no_set_loss.py'))
    assert len(baseline['loss']['loss_cfgs'][0]['stage_weights']) == 6
    baseline['loss']['loss_cfgs'] = [x for x in baseline['loss']['loss_cfgs'] if x['type'] == 'OccupancyLoss']
    assert ablation.to_dict() == baseline
    module = OPENOCC_LOSS.build(ablation.loss)
    assert len(module.losses) == 1
    logits = torch.randn(1, 18, 8, requires_grad=True)
    result, detail = module({'pred_occ': [logits], 'sampled_xyz': torch.zeros(1, 8, 3),
        'sampled_label': torch.tensor([[0, 1, 2, 3, 4, 5, 17, 17]]),
        'occ_mask': torch.ones(1, 8, dtype=torch.bool)})
    assert set(detail) == {'OccupancyLoss'}
    result.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize('shift', [-4., 0., 4.])
def test_calibration_matches_argmax_including_exact_ties(shift):
    logits = torch.zeros(18, 4)
    logits[0] = torch.tensor([2., 1., 0., -1.])
    logits[17] = torch.tensor([2., 0., 1., 3.]) + shift
    fg_max, fg_class = logits[:17].max(0)
    actual = calibrated_predictions(fg_max, fg_class, logits[17], shift)
    logits[17] -= shift
    assert torch.equal(actual, logits.argmax(0))


def test_training_manifest_cannot_silently_change_population_or_annotations(tmp_path):
    data = tmp_path / 'train.pkl'
    data.write_bytes(b'fixed training split')
    manifest = tmp_path / 'manifest.json'
    a = training_selection(100, 16, 928, data, manifest)
    assert a == training_selection(100, 16, 928, data, manifest)
    assert len(set(a)) == 16
    data.write_bytes(b'different training split')
    with pytest.raises(ValueError, match='manifest differs'):
        training_selection(100, 16, 928, data, manifest)


def test_shift_selection_uses_training_only_and_prefers_no_correction_on_ties():
    values = {'-2.0': {'mIoU': 10.}, '0.0': {'mIoU': 10.}, '2.0': {'mIoU': 10.}}
    assert choose_empty_shift({'split': 'train', 'empty_logit_shift_metrics': values}) == 0.
    with pytest.raises(ValueError, match='exclusively from training'):
        choose_empty_shift({'split': 'val', 'empty_logit_shift_metrics': values})


def test_report_distinguishes_raw_and_calibrated_performance_and_negative_shift(tmp_path, monkeypatch):
    import copy
    from tools import finish_no_set_loss_ablation as finish

    camera = {'mIoU': 20., 'IoU': 40., 'precision': 70., 'recall': 50.,
              'predicted_occupied_fraction': 20., 'gt_occupied_fraction': 25.,
              'per_class_IoU': {'car': 20.}}
    baseline = {'camera_metrics': camera, 'ray_metrics': {'RayIoU': 18.}}
    altered = copy.deepcopy(baseline)
    altered['camera_metrics']['mIoU'] = 15.
    adjusted = copy.deepcopy(altered)
    adjusted['camera_metrics']['mIoU'] = 17.
    models = {}
    for name, raw, calibrated, shift in [('with_set_loss', baseline, baseline, 0.),
                                         ('no_set_loss', altered, adjusted, -2.)]:
        sweep = {'learned_empty_bias': 1., 'empty_scalar': 9.,
            'foreground_class_accuracy_on_gt_occupied': 60.,
            'render_logit_means': {'foreground_max_gt_occupied': 5., 'empty_logit_gt_occupied': 6.},
            'empty_logit_shift_metrics': {'0.0': raw['camera_metrics'], '4.0': raw['camera_metrics'],
                                          str(shift): calibrated['camera_metrics']}}
        models[name] = {'selected_shift': shift, 'raw_official': raw,
                        'calibrated_official': calibrated, 'validation_sweep': sweep,
                        'training_curve': [{'epoch': 5, 'mIoU': raw['camera_metrics']['mIoU'], 'IoU': 40.}]}
    delta = dict(mIoU=-5., IoU=0., precision=0., recall=0., RayIoU=0.)
    summary = dict(models=models, raw_delta_no_set_minus_baseline=delta,
                   calibrated_delta_no_set_minus_baseline=dict(delta, mIoU=-3.))
    path = tmp_path / 'report.md'
    monkeypatch.setattr(finish, 'REPORT', path)
    finish.write_report(summary)
    report = path.read_text()
    assert '| 差值（百分点） | -5.00 |' in report
    assert '**-3.00 个百分点**' in report
    assert '| 移除 set loss | -2.0 | 15.00 | 17.00 | +2.00 |' in report
