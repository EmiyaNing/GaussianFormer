"""Verify the new semantic objective and preserved B/15-epoch protocol."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mmengine.config import Config

from loss.occupancy_loss import OccupancyLoss
from .diagnostics import conditional_confusion, conditional_statistics
from .loss import conditional_semantic_ce, FullspaceConditionalSemanticOccupancyLoss
from .protocol import BASE_CONFIG, CONFIG


def options():
    value = Config.fromfile(str(BASE_CONFIG)).loss.occupancy_loss.to_dict()
    value.pop('type')
    return value


def test_conditional_ce_is_weighted_17class_ce_on_gt_nonempty_only():
    raw = torch.randn(2, 18, 7, generator=torch.Generator().manual_seed(17), requires_grad=True)
    labels = torch.tensor([[17, 4, 11, 255, 13, 17, 16], [17, 15, 4, 11, 17, 14, 255]])
    weights = torch.linspace(.3, 1.4, 18)
    valid = (labels != 17) & (labels != 255)
    expected = F.cross_entropy(raw.transpose(1, 2)[valid][:, :17], labels[valid], weight=weights[:17])
    actual = conditional_semantic_ce(raw, labels, weights)
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, raw)[0]
    assert gradient[:, 17].count_nonzero() == 0
    assert gradient.transpose(1, 2)[~valid].count_nonzero() == 0
    torch.testing.assert_close(gradient.sum(1), torch.zeros_like(gradient.sum(1)), atol=1e-7, rtol=0.)


def test_predicted_free_gt_car_still_receives_conditional_supervision():
    raw = torch.zeros(1, 18, 1)
    raw[:, 17] = 20.
    raw.requires_grad_(True)
    loss = conditional_semantic_ce(raw, torch.tensor([[4]]), torch.ones(18))
    gradient = torch.autograd.grad(loss, raw)[0]
    assert raw.argmax(1).item() == 17
    assert gradient[0, 4, 0] < 0 and gradient[0, 11, 0] > 0
    assert gradient[0, 17, 0] == 0


def test_auxiliary_is_invariant_to_free_logit_and_common_foreground_shift():
    raw = torch.randn(1, 18, 9, generator=torch.Generator().manual_seed(17))
    labels = torch.tensor([[4, 11, 13, 16, 15, 17, 255, 14, 4]])
    changed = raw.clone()
    changed[:, 17] += 80.
    changed[:, :17] -= 30.
    weights = torch.ones(18)
    torch.testing.assert_close(conditional_semantic_ce(raw, labels, weights),
                               conditional_semantic_ce(changed, labels, weights), atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize('label', [17, 255])
def test_no_occupied_gt_has_finite_connected_zero_auxiliary(label):
    raw = torch.randn(2, 18, 4, requires_grad=True)
    loss = conditional_semantic_ce(raw, torch.full((2, 4), label), torch.ones(18))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert raw.grad is not None and raw.grad.count_nonzero() == 0


def test_zero_auxiliary_matches_B_full_ce_lovasz_value_and_gradient():
    baseline = OccupancyLoss(**options())
    changed = FullspaceConditionalSemanticOccupancyLoss(conditional_semantic_weight=0., **options())
    raw = torch.randn(2, 18, 12, generator=torch.Generator().manual_seed(17), requires_grad=True)
    labels = torch.arange(24).reshape(2, 12) % 18
    expected = baseline.loss_voxel([raw], torch.zeros(2, 12, 3), labels)
    actual = changed.loss_voxel([raw], torch.zeros(2, 12, 3), labels)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    torch.testing.assert_close(torch.autograd.grad(actual, raw, retain_graph=True)[0],
                               torch.autograd.grad(expected, raw)[0], rtol=0., atol=0.)


def test_addition_averages_stages_and_preserves_free_voxel_supervision():
    baseline = OccupancyLoss(**options())
    changed = FullspaceConditionalSemanticOccupancyLoss(conditional_semantic_weight=1., **options())
    layers = [torch.randn(1, 18, 6, generator=torch.Generator().manual_seed(i), requires_grad=True) for i in (17, 19)]
    labels = torch.tensor([[17, 4, 11, 13, 255, 17]])
    xyz = torch.zeros(1, 6, 3)
    base = baseline.loss_voxel(layers, xyz, labels)
    actual = changed.loss_voxel(layers, xyz, labels)
    expected_aux = sum(conditional_semantic_ce(raw, labels, baseline.class_weights) for raw in layers) / 2
    torch.testing.assert_close(actual, base + expected_aux)
    base_grad = torch.autograd.grad(base, layers, retain_graph=True)
    changed_grad = torch.autograd.grad(actual, layers)
    for left, right in zip(base_grad, changed_grad):
        torch.testing.assert_close(left[:, 17], right[:, 17])
        torch.testing.assert_close(left[:, :, [0, 5]], right[:, :, [0, 5]])
        assert right[0, 17, 0] < 0


def test_auxiliary_analytical_gradient_statistics_match_autograd():
    cfg = Config.fromfile(str(CONFIG))
    raw = torch.randn(1, 18, 9, generator=torch.Generator().manual_seed(17), requires_grad=True)
    labels = torch.tensor([[17, 4, 11, 13, 255, 17, 16, 4, 17]])
    camera = torch.tensor([[True, True, False, True, False, True, False, False, False]])
    weights = raw.new_tensor(cfg.loss.occupancy_loss.manual_class_weight)
    auxiliary = conditional_semantic_ce(raw, labels, weights)
    base = 10 * F.cross_entropy(raw, labels, weight=weights, ignore_index=255)
    ag = torch.autograd.grad(auxiliary, raw, retain_graph=True)[0]
    bg = torch.autograd.grad(base, raw)[0]
    record = conditional_statistics(raw.detach(), labels, camera, cfg, index=0)
    assert record['conditional_semantic_loss'] == pytest.approx(float(auxiliary.detach()), abs=1e-6)
    assert record['auxiliary_logit_gradient_norm'] == pytest.approx(float(ag.norm()), abs=1e-6)
    assert record['base_18class_ce_logit_gradient_norm'] == pytest.approx(float(bg.norm()), abs=1e-6)
    assert record['auxiliary_base_ce_logit_gradient_cosine'] == pytest.approx(float((ag * bg).sum() / (ag.norm() * bg.norm())), abs=1e-6)
    assert sum(v['loss_contribution'] for v in record['buckets'].values()) == pytest.approx(float(auxiliary.detach()), abs=1e-6)


def test_conditional_confusion_includes_gt_occupied_predicted_free():
    logits = torch.zeros(1, 18, 4)
    logits[:, 17] = 20.
    logits[0, 4, 0] = 10.
    logits[0, 11, 1] = 10.
    parts = conditional_confusion(logits, torch.tensor([[4, 13, 17, 255]]), torch.tensor([[True, False, True, False]]))
    assert parts['full'].sum() == 2 and parts['full'][4, 4] == 1 and parts['full'][13, 11] == 1
    np.testing.assert_array_equal(parts['full'], parts['camera'] + parts['noncamera'])


def test_config_is_standalone_and_only_adds_the_auxiliary_to_B():
    baseline, actual = Config.fromfile(str(BASE_CONFIG)).to_dict(), Config.fromfile(str(CONFIG)).to_dict()
    assert '_base_' not in CONFIG.read_text()
    assert actual['max_epochs'] == 15 and actual['loss']['mask_type'] == 'none'
    assert actual['loss']['occupancy_loss']['manual_class_weight'][-1] == .1
    assert actual['loss']['occupancy_loss'].pop('conditional_semantic_weight') == 1.
    assert actual['loss']['occupancy_loss']['type'] == 'FullspaceConditionalSemanticOccupancyLoss'
    actual['loss']['occupancy_loss']['type'] = 'OccupancyLoss'
    for key in ('custom_imports', 'report_directory', 'experiment_name'):
        actual[key] = baseline[key]
    assert actual == baseline


def test_C_entry_retains_15360_updates_and_stops_at_saved_epoch_one(tmp_path, monkeypatch):
    from . import train as entry
    from .. import train as runner
    cfg = Config.fromfile(str(CONFIG))
    cfg.report_directory = str(tmp_path / 'analysis')
    Path(cfg.report_directory).mkdir()
    path = tmp_path / 'standalone.py'
    cfg.dump(str(path))
    advanced = []
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([dict(params=[parameter], lr=.0002, base_lr=.0002)])

    def fake_main():
        runner.set_lr(optimizer, 0, 15360, 500, .001)
        runner.atomic_save(tmp_path / 'epoch_01.pth', dict(
            epoch=1, next_iter=0, phase='train', updates=1024, config_sha256='fixture',
            manifest_sha256=dict(train='t', val='v'), optimizer=optimizer.state_dict()))
        advanced.append(True)

    monkeypatch.setattr(runner, 'main', fake_main)
    monkeypatch.setattr(sys, 'argv', ['entry', '--config', str(path), '--work-dir', str(tmp_path), '--stop-after-epoch', '1'])
    entry.main()
    assert advanced == []
    boundary = json.loads((tmp_path / 'first_epoch_completed.json').read_text())
    assert boundary['planned_epochs'] == 15 and boundary['trained_epochs'] == 1 and boundary['updates'] == 1024
    assert json.loads((tmp_path / 'optimization_budget.json').read_text())['total_updates'] == 15360
    assert not (tmp_path / 'completed.json').exists()


def test_automatic_report_compares_126_rows_and_rejects_missing_conditional_gt(tmp_path):
    import csv
    from .report import create_report
    from .protocol import NAME
    from ..metrics import HorizonMetrics
    from ..fullspace_diagnostics import region_metrics
    from ..train import digest

    baseline_names = ['Strict', 'Camera_Gaussian', 'Fullspace_CE', 'A_softmax_focal', 'B_ce_free_0_1']
    baseline_roots = {}
    output = tmp_path / 'C_analysis'
    baseline_output = tmp_path / 'B_analysis'
    work = tmp_path / 'runs' / NAME
    baseline_work = tmp_path / 'runs' / 'B_ce_free_0_1'
    cfg_hash = digest(CONFIG)
    for index, name in enumerate(baseline_names + [NAME]):
        directory = tmp_path / 'runs' / name
        directory.mkdir(parents=True)
        metric = HorizonMetrics()
        metric.samples = 1024
        metric.confusion[:, 4, 4] = 20 + index
        metric.confusion[:, 4, 17] = 1024 - 20 - index
        metric.confusion[:, 17, 17] = 1024
        (directory / 'epoch_01_metrics.json').write_text(json.dumps(dict(metric.results(), epoch=1)))
        np.savez_compressed(directory / 'epoch_01_confusion.npz', confusion=metric.confusion)
        if name != NAME:
            baseline_roots[name] = directory
        if name not in (NAME, 'B_ce_free_0_1'):
            continue
        (directory / 'epoch_01.pth').write_bytes(b'fixture restartable checkpoint')
        boundary = dict(planned_epochs=15, trained_epochs=1, updates=1024, next_iter=0,
                        phase='train', checkpoint_sha256=digest(directory / 'epoch_01.pth'),
                        config_sha256=cfg_hash, manifest_sha256=dict(train='t', val='v'))
        (directory / 'first_epoch_completed.json').write_text(json.dumps(boundary))
        (directory / 'optimization_budget.json').write_text(json.dumps(dict(planned_epochs=15, total_updates=15360)))
        (directory / 'initialization.json').write_text(json.dumps(dict(initial_model_sha256='same_fixture_initialization')))
        destination = output if name == NAME else baseline_output
        diag = destination / 'epoch_diagnostics'
        diag.mkdir(parents=True)
        full = metric.confusion.copy()
        camera = full // 2
        noncamera = full - camera
        np.savez_compressed(diag / 'epoch_01_visibility_confusion.npz', full=full, camera=camera, noncamera=noncamera)
        payload = dict(is_smoke=False, num_samples=1024, voxel_supervision_mask='none', config_sha256=cfg_hash,
                       attributes=[dict(index=i, horizon=h) for i in range(16) for h in range(7)],
                       gradient_probes=[dict(index=i) for i in range(2)],
                       visibility={f'h{h}/{region}': region_metrics(array[h]) for h in range(7)
                                   for region, array in dict(full=full, camera=camera, noncamera=noncamera).items()})
        (diag / 'epoch_01_diagnostics.json').write_text(json.dumps(payload))
        (diag / 'epoch_01_classification_components.json').write_text(json.dumps(dict(
            is_smoke=False, num_samples=16, formal_sample_count=1024, records=[dict(index=i) for i in range(16)])))
        if name == NAME:
            (destination / 'configuration_audit.json').write_text(json.dumps(dict(config_sha256=cfg_hash)))
            conditional_full = np.diag(full[0].sum(1)[:17])
            conditional_camera = np.diag(camera[0].sum(1)[:17])
            np.savez_compressed(diag / 'epoch_01_conditional_semantic_confusion.npz', full=conditional_full,
                                camera=conditional_camera, noncamera=conditional_full - conditional_camera)
            records = [dict(index=i, conditional_semantic_loss=.5,
                            auxiliary_to_base_ce_logit_gradient_norm_ratio=.2, auxiliary_base_ce_logit_gradient_cosine=.5,
                            buckets={region: dict(loss_contribution=.25, mean_correct_conditional_probability=.5)
                                     for region in ('camera', 'noncamera')}) for i in range(16)]
            (diag / 'epoch_01_conditional_semantics.json').write_text(json.dumps(dict(is_smoke=False,
                formal_sample_count=1024, num_samples=16, records=records, formal_current_conditional_class_accuracy_percent=100.)))
    create_report(work, output, baseline_work, baseline_output, baseline_roots)
    rows = list(csv.DictReader((output / 'per_horizon_per_class_iou.csv').open()))
    assert len(rows) == 126 and len({(r['seconds'], r['class_index']) for r in rows}) == 126
    judgement = json.loads((output / 'diagnosis_completed.json').read_text())
    assert judgement['actual_epochs'] == 1 and judgement['planned_epochs'] == 15
    assert judgement['current_miou_relative_to_B'] == 'higher'
    path = output / 'epoch_diagnostics/epoch_01_conditional_semantic_confusion.npz'
    with np.load(path) as saved:
        corrupted = {key: saved[key] * 2 for key in saved.files}
    np.savez_compressed(path, **corrupted)
    with pytest.raises(ValueError, match='excluded GT-occupied'):
        create_report(work, output, baseline_work, baseline_output, baseline_roots)
