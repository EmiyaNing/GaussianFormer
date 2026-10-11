import torch

from tools.train_future_distillation_causal_screen import controlled_gradients
from tools.experiment_future_distillation_conditions import gaussian_nll, ConditionalFeatureModel


def test_task_priority_projection_removes_conflicting_component():
    task=(torch.tensor([1.,0.]),)
    auxiliary=(torch.tensor([-1.,1.]),)
    merged,stats=controlled_gradients(task,auxiliary,.1,project=True)
    added=merged[0]-task[0]
    assert torch.dot(task[0],added)>=-1e-7
    assert stats['cosine']<0
    assert stats['effective_aux_norm']>0


def test_projection_leaves_aligned_gradient_unchanged():
    task=(torch.tensor([1.,0.]),)
    auxiliary=(torch.tensor([1.,1.]),)
    actual,_=controlled_gradients(task,auxiliary,.1,project=True)
    torch.testing.assert_close(actual[0],task[0]+.1*auxiliary[0])


def test_projection_does_not_amplify_almost_antiparallel_residual():
    task=(torch.tensor([1.,0.]),)
    auxiliary=(torch.tensor([-1.,1e-9]),)
    actual,stats=controlled_gradients(task,auxiliary,.1,project=True)
    assert stats['rescale']<=2.
    assert torch.isfinite(actual[0]).all()


def test_missing_auxiliary_keeps_task_gradient():
    task=(torch.tensor([1.,0.]),torch.tensor([2.]))
    auxiliary=(torch.tensor([-1.,1.]),None)
    actual,_=controlled_gradients(task,auxiliary,.1,project=True)
    torch.testing.assert_close(actual[1],task[1])


def test_gaussian_nll_and_model_shapes():
    model=ConditionalFeatureModel(272)
    mean,logvar=model(torch.zeros(3,272))
    assert mean.shape==logvar.shape==(3,256)
    value=gaussian_nll(mean,logvar,torch.ones_like(mean))
    assert torch.isfinite(value)
    value.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_r1_report_keeps_condition_definitions_when_regenerated(tmp_path):
    from tools.summarize_future_distillation_causal import update_reports
    update_reports(tmp_path / 'missing_results', tmp_path / 'reports')
    report = (tmp_path / 'reports/r1_conditional_information/results.md').read_text()
    for condition in ('position_time', 'history', 'privileged_ego'):
        for method in ('deterministic', 'gaussian'):
            assert f'`{condition}_{method}`' in report
    assert '272 维' in report
    assert '48 个点的算术平均中心' in report
    assert 'SparseWorld 和教师保持冻结' in report
    assert '共 12 个数' in report


def test_r2_report_keeps_mixing_and_stage_definitions_when_regenerated(tmp_path):
    from tools.summarize_future_distillation_causal import update_reports
    update_reports(tmp_path / 'missing_results', tmp_path / 'reports')
    report = (tmp_path / 'reports/r2_task_compatibility/results.md').read_text()
    for prefix in ('fixed', 'native_joint'):
        for alpha in ('0', '0.1', '0.25', '0.5', '1'):
            assert f'`{prefix}_mix{alpha}`' in report
    assert '不表示只替换相应百分比的 query' in report
    assert 'future-born queries 保留' in report
    assert '不是联合训练' in report
    assert 'T2 中的 S2' in report
    assert 'T3 中的 R3' in report


def test_s0_report_preserves_checkpoint_intervention_definitions(tmp_path):
    from tools.summarize_future_distillation_causal import update_reports
    update_reports(tmp_path / 'missing_results', tmp_path / 'reports')
    report = (tmp_path / 'reports/s0_protocol/results.md').read_text()
    for name in ('task_epoch5', 'reset_bn', 'disable_adapter', 'reset_heads',
                 'reset_perception', 'reset_bn_disable_adapter'):
        assert f'`{name}`' in report
    assert '不更新优化器，也不覆盖磁盘权重' in report
    assert '不是只恢复图像 backbone' in report
    assert '不是参数责任占比' not in report  # No numerical recovery claim without results.
