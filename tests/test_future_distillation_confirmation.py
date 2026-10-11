from tools.audit_future_distillation_confirmation import (
    scene_audit, confirmation_gate, combination_gate)


def test_new_frames_in_old_scenes_do_not_create_confirmation_scenes():
    value = scene_audit(['scene-a', 'scene-b'], ['scene-a', 'scene-b'])
    assert value['unused_count'] == 0
    assert not confirmation_gate(value['unused_count'], [1, 2, 3], 5)['eligible']


def test_multiseed_confirmation_requires_actual_unused_scenes():
    assert confirmation_gate(10, [1, 2, 3], 5)['eligible']
    assert not confirmation_gate(10, [1, 1, 2], 5)['eligible']
    assert not confirmation_gate(10, [1, 2, 3], 1)['eligible']


def test_combination_rejects_local_gains_and_duplicate_interventions():
    row = dict(name='A', independent_confirmation=True, paired_seeds=3, useful=True)
    assert not combination_gate([row, row])['eligible']
    unconfirmed = dict(row, name='B', independent_confirmation=False)
    assert not combination_gate([row, unconfirmed])['eligible']
    assert combination_gate([row, dict(row, name='B')])['eligible']


def test_scene_inventory_keeps_external_used_scenes_separate():
    value = scene_audit(['a', 'b', 'c'], ['a', 'external'])
    assert value['development_scenes'] == 2
    assert value['used_available_scenes'] == 1
    assert value['unused_scenes'] == ['b', 'c']


def test_summary_keeps_entry_audit_separate_from_completed_experiments(tmp_path, monkeypatch):
    from tools import summarize_future_distillation_causal as report
    entry = dict(status='prerequisite_audit_complete_not_training_complete',
                 scene_coverage={'val': dict(available_scenes=150, unused_count=0)})
    monkeypatch.setattr(report, 'read', lambda path:
                        entry if path.name == 'entry_audit.json' else None)
    report.update_reports(tmp_path/'prior_stage', tmp_path/'docs')
    value = (tmp_path/'docs/summary.md').read_text()
    assert '没有完成对应正式实验' in value
    assert '未涉及场景 0 个' in value
    assert '未改变 R1–R3 证据等级' in value
    assert '[状态](u2_multiple_futures/results.md)' in value
