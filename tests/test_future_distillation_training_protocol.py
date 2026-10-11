import torch
from torch import nn
from tools.future_distillation_training_protocol_common import (
    TrainingProtocolStudent, set_training_mode, schedule_factor, bn_snapshot, bn_unchanged)
from model.segmentor.sparseworld_predictability_finetune_segmentor import SparseWorldPredictabilityFinetuneSegmentor


def test_fixed_bn_keeps_affine_trainable_and_dropout_active():
    instance=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.Dropout(.1))
    original=bn_snapshot(instance)
    set_training_mode(instance,True)
    assert instance.training and instance[2].training and not instance[1].training
    assert instance[1].weight.requires_grad and instance[1].bias.requires_grad
    instance(torch.randn(8,4)).sum().backward()
    assert instance[1].weight.grad is not None and bn_unchanged(instance,original)


def test_train_bn_updates_statistics():
    instance=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4))
    original=bn_snapshot(instance)
    set_training_mode(instance,False);instance(torch.randn(8,4))
    assert not bn_unchanged(instance,original)


def test_detach_geometry_preserves_values_and_semantic_gradient(monkeypatch):
    points=torch.randn(1,3,3,requires_grad=True);logits=torch.randn(1,3,17,requires_grad=True)
    monkeypatch.setattr(SparseWorldPredictabilityFinetuneSegmentor,'forward',
                        lambda self,*args,**kwargs:dict(future_pred_points=[points],future_pred_logits=[logits]))
    instance=TrainingProtocolStudent.__new__(TrainingProtocolStudent);nn.Module.__init__(instance)
    instance.detach_future_geometry=True;output=instance(None,None)
    torch.testing.assert_close(output['future_pred_points'][0],points)
    assert not output['future_pred_points'][0].requires_grad
    assert output['future_pred_logits'][0] is logits and logits.requires_grad


def test_schedule_does_not_compress_original_warmup():
    assert schedule_factor(0,200,20480)==.005
    assert schedule_factor(199,200,20480)==1.
    assert schedule_factor(200,200,20480)==1.
    assert schedule_factor(255,200,20480)>.999


def test_summary_bn_scores_are_joined_without_repeating_sentence(tmp_path, monkeypatch):
    from tools import summarize_future_distillation_causal as report
    protocol = dict(status='running', arms={}, bn_only=dict(steps={
        '256': dict(metrics={h: dict(mIoU=v) for h, v in
                            zip(report.HORIZONS, (1.443, 1.413, 1.401))})}))
    monkeypatch.setattr(report, 'read', lambda path:
                        protocol if path.name == 'results.json' and
                        path.parent.name == '20261001_training_protocol' else None)
    report.update_reports(report.REPO / 'out/future_distillation_causal/20260930', tmp_path)
    text = (tmp_path / 'summary.md').read_text()
    assert 'mIoU 为 1.443／1.413／1.401。' in text
    assert text.count('仅进行 256 次训练图像前向') == 1


def test_native_verifier_detects_incorrect_report_score(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pytest
    from tools.analyze_future2current_regions import grouped_metrics
    from tools.verify_future_distillation_training_protocol import verify

    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'experiment'
    root.mkdir()
    hist = np.broadcast_to(np.eye(18) * 15 + np.ones((18, 18)), (2, 3, 18, 18)).copy()
    counts = dict(hist=hist, intersection=np.full((2, 3), 10), union=np.full((2, 3), 20))
    np.savez(root / 'baseline.npz', **counts)
    reference = tmp_path / 'out/future_distillation_causal/20260930'
    reference.mkdir(parents=True)
    np.savez(reference / 's1_baseline.npz', **counts)
    manifest = dict(train=dict(indices=[0], scenes=['train']),
                    val=dict(indices=[1, 2], scenes=['a', 'b']), checkpoint_sha256='fixture')
    (root / 'manifest.json').write_text(json.dumps(manifest))
    teacher = tmp_path / 'out/future2current/predictable_full_finetune_4096x1024'
    teacher.mkdir(parents=True)
    (teacher / 'manifest.json').write_text(json.dumps(manifest))
    grouped = grouped_metrics(hist.sum(0))
    horizons = ('1s', '2s', '3s')
    baseline = dict(metrics={h: dict(mIoU=32., OccIoU=50.) for h in horizons},
                    groups={h: {g: dict(mIoU=float(grouped[g]['macro_iou'][i]),
                                        IoU=float(grouped[g]['group_iou'][i]))
                                for g in ('dynamic', 'static')} for i, h in enumerate(horizons)})
    report = dict(status='complete', baseline=baseline, arms={})
    (root / 'results.json').write_text(json.dumps(report))
    assert verify(root)['status'] == 'passed'
    report['baseline']['metrics']['1s']['mIoU'] += 1.
    (root / 'results.json').write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match='verification failed'):
        verify(root)
    assert json.loads((root / 'verification.json').read_text())['status'] == 'failed'


def test_protocol_reading_guide_distinguishes_statistics_and_supervision():
    from tools.summarize_future_distillation_training_protocol import training_protocol_explanation
    from tools.summarize_future_distillation_causal import s0_metric_reading
    guide = '\n'.join(training_protocol_explanation())
    for token in ('full', 'future', 'bn_train', 'bn_fixed', 'detached_geom',
                  'task', 'teacher', 'shuffled', 'seed20260930'):
        assert f'`{token}`' in guide
    assert 'BN-only' in guide and '完全没有优化器' in guide
    assert '不是教师自身的感知成绩' in guide
    assert '不等于冻结历史编码' in guide
    assert '同样的图像前向次数不等于相同的内部 BN 更新次数' in guide
    assert '相同 2 验证样本' in '\n'.join(s0_metric_reading(2))
