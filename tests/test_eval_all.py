from argparse import Namespace
from pathlib import Path

import pytest

import eval as eval_script


def _args(work_dir, **overrides):
    values = dict(
        py_config='config/opus/sparseworld_strict.py',
        work_dir=str(work_dir), resume_from='', sparseworld_python='',
        seed=None, vis_occ=False, eval_all=True, eval_tag='')
    values.update(overrides)
    return Namespace(**values)


def test_discover_eval_checkpoints_is_nonrecursive_and_naturally_sorted(tmp_path):
    for name in ('epoch_10.pth', 'epoch_2.pth', 'best.pth', 'notes.txt'):
        (tmp_path / name).touch()
    nested = tmp_path / 'nested'
    nested.mkdir()
    (nested / 'epoch_1.pth').touch()
    assert [path.name for path in eval_script.discover_eval_checkpoints(tmp_path)] == [
        'best.pth', 'epoch_2.pth', 'epoch_10.pth']


def test_discover_eval_checkpoints_rejects_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match='No .pth checkpoints'):
        eval_script.discover_eval_checkpoints(tmp_path)


def test_run_eval_all_launches_one_isolated_eval_per_checkpoint(tmp_path, monkeypatch):
    (tmp_path / 'epoch_10.pth').touch()
    (tmp_path / 'epoch_2.pth').touch()
    calls = []

    def record(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(eval_script.subprocess, 'run', record)
    eval_script.run_eval_all(_args(
        tmp_path, seed=17, vis_occ=True, sparseworld_python='/opt/sw/python'))
    assert len(calls) == 2
    assert [Path(call[0][call[0].index('--resume-from') + 1]).name
            for call in calls] == ['epoch_2.pth', 'epoch_10.pth']
    assert [call[0][call[0].index('--eval-tag') + 1]
            for call in calls] == ['epoch_2', 'epoch_10']
    for command, kwargs in calls:
        assert '--eval-all' not in command
        assert command[command.index('--eval-log-file') + 1].endswith(
            '_eval_all.log')
        assert command[command.index('--seed') + 1] == '17'
        assert '--vis-occ' in command
        assert command[command.index('--sparseworld-python') + 1] == '/opt/sw/python'
        assert kwargs['check'] is True
    assert calls[0][0][calls[0][0].index('--eval-epoch') + 1] == '2'
    assert calls[1][0][calls[1][0].index('--eval-epoch') + 1] == '10'
    assert '--skip-config-log' not in calls[0][0]
    assert '--skip-config-log' in calls[1][0]


def test_eval_separator_uses_epoch_number_or_checkpoint_name():
    assert eval_script.checkpoint_epoch_label('epoch_12.pth') == '12'
    assert eval_script.make_eval_separator('12') == (
        '-------------------- Epoch 12 --------------------')
    assert eval_script.make_eval_separator('best') == (
        '-------------------- Checkpoint best --------------------')


def test_select_eval_checkpoints_starts_from_epoch_inclusively(tmp_path):
    checkpoints = []
    for name in ('best.pth', 'epoch_2.pth', 'epoch_5.pth', 'epoch_10.pth',
                 'latest.pth'):
        checkpoint = tmp_path / name
        checkpoint.touch()
        checkpoints.append(checkpoint)
    assert [path.name for path in eval_script.select_eval_checkpoints(
            checkpoints, start_from=5)] == ['epoch_5.pth', 'epoch_10.pth']


def test_select_eval_checkpoints_rejects_start_after_last_epoch(tmp_path):
    checkpoint = tmp_path / 'epoch_3.pth'
    checkpoint.touch()
    with pytest.raises(FileNotFoundError, match='at or after --start-from 4'):
        eval_script.select_eval_checkpoints([checkpoint], start_from=4)


def test_select_eval_checkpoints_rejects_negative_epoch():
    with pytest.raises(ValueError, match='zero or greater'):
        eval_script.select_eval_checkpoints([], start_from=-1)


def test_run_eval_all_rejects_explicit_resume(tmp_path):
    (tmp_path / 'epoch_1.pth').touch()
    with pytest.raises(ValueError, match='cannot be used together'):
        eval_script.run_eval_all(_args(tmp_path, resume_from='other.pth'))
