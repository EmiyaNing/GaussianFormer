"""Protect scientific interpretation and manually edited report sections."""
import pytest

from tools import report_center_support_execution as report


@pytest.mark.parametrize('own,field,expected', [
    ([-2., -1.], [.1, .3], '自身变好，场变差'),
    ([-2., -1.], [-.3, -.1], '自身与场同时变好'),
    ([-2., -1.], [-.1, .3], '场变化方向未确定'),
    ([-1., 1.], [.1, .3], '自身目标下降未确认'),
])
def test_local_verdict_requires_own_improvement_and_resolved_field_direction(own, field, expected):
    assert report.local_verdict({
        'self_loss': {'ci95': own}, 'same_batch_voxel': {'ci95': field},
    }) == expected


def test_refresh_preserves_manual_intro_appendix_and_is_repeatable(tmp_path, monkeypatch):
    path = tmp_path / 'report.md'
    prefix = '# 人工撰写的研究背景\n\n'
    suffix = '\n\n## 玩具反例与历史审计\n\n不可丢失的说明。\n'
    path.write_text(prefix + report.AUTO_START + '\n旧生成区\n' + report.AUTO_END + suffix)
    monkeypatch.setattr(report, 'REPORT', path)
    monkeypatch.setattr(report, 'RUN', tmp_path / 'artifacts')
    report.refresh()
    first = path.read_text()
    assert first.startswith(prefix + report.AUTO_START)
    assert first.endswith(report.AUTO_END + suffix)
    assert '等待真实模型证据' in first
    assert '本轮分类降权在两项指标上均高于基线' not in first
    assert '旧生成区' not in first
    report.refresh()
    assert path.read_text() == first


def test_missing_boundaries_cannot_erase_manual_report(tmp_path, monkeypatch):
    path = tmp_path / 'report.md'
    original = '# 尚未迁移的人工报告\n\n所有实验说明都必须保留。\n'
    path.write_text(original)
    monkeypatch.setattr(report, 'REPORT', path)
    monkeypatch.setattr(report, 'RUN', tmp_path / 'artifacts')
    with pytest.raises(ValueError, match='start/end'):
        report.refresh()
    assert path.read_text() == original
