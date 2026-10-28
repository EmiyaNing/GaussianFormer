"""Verify objective identity, gradients, standalone configs and epoch stopping."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from mmengine.config import Config

from loss.occupancy_loss import OccupancyLoss
from .diagnostics import classification_statistics
from .focal_loss import FullspaceSoftmaxFocalOccupancyLoss, weighted_softmax_focal
from .protocol import ARMS, BASE_CONFIG
from .train import EpochBoundaryReached, save_boundary


def test_gamma_zero_matches_weighted_ce_value_and_gradient_with_ignore():
    raw = torch.randn(2, 18, 9, generator=torch.Generator().manual_seed(17), requires_grad=True)
    labels = torch.randint(0, 18, (2, 9), generator=torch.Generator().manual_seed(19))
    labels[0, 0] = 255
    weights = torch.linspace(.1, 1.5, 18)
    focal = weighted_softmax_focal(raw, labels, weights, gamma=0)
    ce = F.cross_entropy(raw, labels, weight=weights, ignore_index=255)
    torch.testing.assert_close(focal, ce, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(focal, raw, retain_graph=True)[0],
                               torch.autograd.grad(ce, raw)[0], rtol=1e-6, atol=1e-7)


def test_gamma_zero_preserves_original_lovasz_and_fullspace_loss():
    options = Config.fromfile(str(BASE_CONFIG)).loss.occupancy_loss.to_dict()
    options.pop('type')
    original = OccupancyLoss(**options)
    focal = FullspaceSoftmaxFocalOccupancyLoss(focal_gamma=0, **options)
    values = torch.randn(2, 18, 12, generator=torch.Generator().manual_seed(17), requires_grad=True)
    target = torch.arange(24).reshape(2, 12) % 18
    expected = original.loss_voxel([values], torch.zeros(2, 12, 3), target)
    actual = focal.loss_voxel([values], torch.zeros(2, 12, 3), target)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(actual, values, retain_graph=True)[0],
                               torch.autograd.grad(expected, values)[0], rtol=1e-6, atol=1e-7)


def test_easy_free_voxel_gradient_is_downweighted_without_removing_it():
    raw = torch.zeros(1, 18, 2)
    raw[:, 17] = 8.
    raw.requires_grad_(True)
    target = torch.tensor([[17, 4]])
    weights = torch.ones(18)
    ce_grad = torch.autograd.grad(weighted_softmax_focal(raw, target, weights, 0), raw, retain_graph=True)[0]
    focal_grad = torch.autograd.grad(weighted_softmax_focal(raw, target, weights, 2), raw)[0]
    assert 0 < focal_grad[0, 17, 0] / ce_grad[0, 17, 0] < .001
    assert focal_grad[0, 17, 1] > 0  # Misclassified occupied target still suppresses empty logit.
    assert focal_grad[0, 4, 1] < 0


@pytest.mark.parametrize('gamma', [0., 2.])
def test_analytic_bias_gradient_matches_autograd(gamma):
    cfg = Config.fromfile(str(ARMS['A_softmax_focal' if gamma else 'B_ce_free_0_1']))
    weights = torch.tensor(cfg.loss.occupancy_loss.manual_class_weight)
    base = torch.randn(1, 18, 8, generator=torch.Generator().manual_seed(17))
    bias = torch.tensor(.3, requires_grad=True)
    logits = torch.cat((base[:, :17], base[:, 17:] - bias), dim=1)
    labels = torch.tensor([[17, 4, 11, 17, 17, 4, 255, 17]])
    camera = torch.tensor([[True, False, True, True, False, False, True, False]])
    actual_loss = 10 * weighted_softmax_focal(logits, labels, weights, gamma)
    expected = float(torch.autograd.grad(actual_loss, bias)[0])
    record = classification_statistics(logits.detach(), labels, camera, cfg, index=0)
    assert record['signed_bias_gradient'] == pytest.approx(expected, abs=1e-5)
    assert sum(b['signed_bias_gradient'] for b in record['buckets'].values()) == pytest.approx(expected, abs=1e-5)
    assert sum(b['classification_loss_contribution'] for b in record['buckets'].values()) == pytest.approx(record['current_classification_loss'], rel=1e-6)


def test_all_ignored_targets_have_finite_zero_loss_and_gradient():
    raw = torch.randn(1, 18, 4, requires_grad=True)
    loss = weighted_softmax_focal(raw, torch.full((1, 4), 255), torch.ones(18), 2)
    assert loss == 0
    loss.backward()
    assert raw.grad.count_nonzero() == 0


def test_configs_are_standalone_and_each_changes_exactly_one_objective_factor():
    baseline = Config.fromfile(str(BASE_CONFIG)).to_dict()
    for arm, path in ARMS.items():
        actual = Config.fromfile(str(path)).to_dict()
        assert '_base_' not in path.read_text()
        assert actual['max_epochs'] == 15
        assert actual['loss']['mask_type'] == 'none'
        assert actual['epoch_diagnostics'] == dict(attribute_samples=16, gradient_samples=2)
        for key in ('experiment_name', 'report_directory', 'custom_imports'):
            actual[key] = baseline[key]
        if arm == 'A_softmax_focal':
            assert actual['loss']['occupancy_loss'].pop('focal_gamma') == 2.
            assert actual['loss']['occupancy_loss']['type'] == 'FullspaceSoftmaxFocalOccupancyLoss'
            actual['loss']['occupancy_loss']['type'] = 'OccupancyLoss'
        else:
            assert actual['loss']['occupancy_loss']['manual_class_weight'][-1] == .1
            actual['loss']['occupancy_loss']['manual_class_weight'][-1] = .5
        assert actual == baseline


def test_epoch_boundary_preserves_resume_state_without_claiming_15_epochs_completed(tmp_path):
    analysis = tmp_path / 'analysis'
    analysis.mkdir()
    cfg = Config(dict(max_epochs=15, report_directory=str(analysis)))
    value = dict(epoch=1, next_iter=0, phase='train', updates=1024,
                 config_sha256='fixture', manifest_sha256=dict(train='t', val='v'),
                 optimizer=dict(state={0: dict(step=1024)}), state_dict=dict(weight=torch.ones(2)))
    with pytest.raises(EpochBoundaryReached):
        save_boundary(tmp_path / 'epoch_01.pth', value,
                      lambda path, state: torch.save(state, path), 1, cfg, tmp_path)
    saved = torch.load(tmp_path / 'latest.pth', weights_only=False)
    assert saved['epoch'] == 1 and saved['next_iter'] == 0 and saved['updates'] == 1024
    assert saved['optimizer']['state'][0]['step'] == 1024
    assert cfg.max_epochs == 15
    assert (tmp_path / 'first_epoch_completed.json').exists()
    assert not (tmp_path / 'completed.json').exists()


def test_runner_keeps_15360_updates_and_pauses_before_epoch_two(tmp_path, monkeypatch):
    import json
    import sys
    from . import train as entry
    from .. import train as original_runner
    cfg = Config.fromfile(str(ARMS['A_softmax_focal']))
    cfg.report_directory = str(tmp_path / 'analysis')
    Path(cfg.report_directory).mkdir()
    config_path = tmp_path / 'standalone.py'
    cfg.dump(str(config_path))
    reached_epoch_two = []
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([dict(params=[parameter], lr=.0002, base_lr=.0002)])

    def fake_main():
        original_runner.set_lr(optimizer, 0, 15360, 500, .001)
        original_runner.atomic_save(tmp_path / 'epoch_01.pth', dict(
            epoch=1, next_iter=0, phase='train', updates=1024, config_sha256='fixture',
            manifest_sha256=dict(train='t', val='v'), optimizer=optimizer.state_dict()))
        reached_epoch_two.append(True)

    monkeypatch.setattr(original_runner, 'main', fake_main)
    monkeypatch.setattr(sys, 'argv', ['entry', '--config', str(config_path), '--work-dir', str(tmp_path), '--stop-after-epoch', '1'])
    entry.main()
    assert reached_epoch_two == []
    assert json.loads((tmp_path / 'optimization_budget.json').read_text())['total_updates'] == 15360
    assert json.loads((tmp_path / 'first_epoch_completed.json').read_text())['planned_epochs'] == 15
    assert Config.fromfile(str(config_path)).max_epochs == 15


def test_first_epoch_report_checks_both_arms_and_writes_126_rows(tmp_path):
    import csv
    import json
    import numpy as np
    from ..metrics import HorizonMetrics
    from ..fullspace_diagnostics import region_metrics
    from ..train import digest
    from .report import create_report, verify_arm
    root, output = tmp_path / 'runs', tmp_path / 'analysis'
    baseline_roots = {}
    names = ['Strict', 'Camera_Gaussian', 'Fullspace_CE'] + list(ARMS)
    for index, name in enumerate(names):
        work = root / name
        work.mkdir(parents=True)
        metric = HorizonMetrics()
        metric.samples = 1024
        metric.confusion[:, 4, 4] = index + 20
        metric.confusion[:, 4, 17] = 1024 - index - 20
        metric.confusion[:, 17, 17] = 1024
        formal = dict(metric.results(), epoch=1)
        (work / 'epoch_01_metrics.json').write_text(json.dumps(formal))
        np.savez_compressed(work / 'epoch_01_confusion.npz', confusion=metric.confusion)
        if name not in ARMS:
            baseline_roots[name] = work
            continue
        (work / 'epoch_01.pth').write_bytes(b'fixture checkpoint')
        boundary = dict(planned_epochs=15, trained_epochs=1, updates=1024, next_iter=0,
                        phase='train', checkpoint_sha256=digest(work / 'epoch_01.pth'),
                        config_sha256='fixture', manifest_sha256=dict(train='t', val='v'))
        (work / 'first_epoch_completed.json').write_text(json.dumps(boundary))
        (work / 'optimization_budget.json').write_text(json.dumps(dict(planned_epochs=15, total_updates=15360)))
        (work / 'initialization.json').write_text(json.dumps(dict(initial_model_sha256='identical_model_fixture')))
        diagnostic_dir = output / name / 'epoch_diagnostics'
        diagnostic_dir.mkdir(parents=True)
        arrays = dict(full=metric.confusion, camera=metric.confusion, noncamera=np.zeros_like(metric.confusion))
        diagnostic = dict(is_smoke=False, num_samples=1024, voxel_supervision_mask='none',
                          config_sha256='fixture', gradient_probes=[{}, {}],
                          attributes=[dict(horizon=h, scale_upper_saturation_percent=0.,
                                           opacity_upper_saturation_percent=0., empty_bias=.1)
                                      for _ in range(16) for h in range(7)],
                          visibility={f'h{h}/{key}': region_metrics(value[h])
                                      for key, value in arrays.items() for h in range(7)})
        (diagnostic_dir / 'epoch_01_diagnostics.json').write_text(json.dumps(diagnostic))
        np.savez_compressed(diagnostic_dir / 'epoch_01_visibility_confusion.npz', **arrays)
        buckets = {bucket: dict(classification_loss_contribution=.1, ce_counterfactual_loss_contribution=.2,
                                signed_bias_gradient=.01, mean_correct_class_probability=.5)
                   for bucket in ('camera/free', 'camera/occupied', 'noncamera/free', 'noncamera/occupied')}
        (diagnostic_dir / 'epoch_01_classification_components.json').write_text(json.dumps(dict(
            is_smoke=False, num_samples=16, formal_sample_count=1024,
            records=[dict(buckets=buckets)] * 16)))
    create_report(root, output, baseline_roots)
    rows = list(csv.DictReader((output / 'per_horizon_per_class_iou.csv').open()))
    assert len(rows) == 126
    assert '第一轮' in (output / 'report.md').read_text()
    assert json.loads((output / 'diagnosis_completed.json').read_text())['compared_epoch'] == 1
    name = next(iter(ARMS))
    wrong = metric.confusion.copy()
    wrong[0, 17, 17] += 1
    np.savez_compressed(output / name / 'epoch_diagnostics' / 'epoch_01_visibility_confusion.npz',
                        full=wrong, camera=wrong, noncamera=np.zeros_like(wrong))
    with pytest.raises(ValueError, match='counts differ'):
        verify_arm(root / name, output / name)
