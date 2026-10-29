"""Compare the camera-filtered Set fine-tune with the audited small baseline.

Reuses the existing full-data measurement implementation without modifying
historical scripts or outputs. No training is performed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import diagnose_small_gaussians as census
from tools.stat_boundary_gaussians import read, write, sha

RUN = ROOT / 'work_dirs/small_camera_set_full_1ep/diagnostics'
BASELINE = ROOT / 'work_dirs/small_gaussian_diagnosis_full'
CONFIG = ROOT / 'config/full_occ3d_camera_set/opusv1_gaussian_small_8f_camera_set_1ep.py'
CHECKPOINT = ROOT / 'work_dirs/small_camera_set_full_1ep/epoch_1.pth'


def configure():
    census.RUN, census.CONFIG, census.CHECKPOINT = RUN, CONFIG, CHECKPOINT
    RUN.mkdir(parents=True, exist_ok=True)
    cfg, protocol = census.initialize()
    from mmengine import Config
    before = read(BASELINE / 'protocol.json')
    original = Config.fromfile(ROOT / before['config'])
    assert cfg.model.to_dict() == original.model.to_dict(), 'Model changed'
    for split in ('train', 'val'):
        assert cfg[split + '_dataset_config'].to_dict() == original[split + '_dataset_config'].to_dict()
    assert protocol['annotation_sha256'] == before['annotation_sha256']
    assert before['checkpoint_sha256'] == sha(ROOT / cfg.train_load_from)
    assert read(BASELINE / 'audit.json')['status'] == 'passed'
    pairing = dict(
        status='initialized', baseline=str(BASELINE.relative_to(ROOT)),
        baseline_checkpoint_sha256=before['checkpoint_sha256'],
        checkpoint_sha256=protocol['checkpoint_sha256'],
        baseline_protocol_sha256=sha(BASELINE / 'protocol.json'),
        runner_sha256=sha(Path(__file__)),
        scope=protocol['expected'], model_and_dataset_configs_identical=True,
        measurement='Unmodified diagnose_small_gaussians and stat_query_budget',
        intervention='One full epoch; all six Set stages filter GT by camera mask',
        causal_limit='No matched original-objective continuation checkpoint is available',
    )
    path = RUN / 'comparison_protocol.json'
    if path.exists():
        assert read(path) == pairing, 'Comparison protocol changed'
    else:
        write(path, pairing)
    return protocol


def audit():
    from tools import audit_small_gaussian_diagnosis as saved
    saved.RUN = RUN
    saved.audit()
    for split in ('train', 'val'):
        paths = sorted((RUN / 'scenes' / split).glob('*.json'))
        for path in paths:
            previous = read(BASELINE / 'scenes' / split / path.name)
            current = read(path)
            assert current['indices'] == previous['indices']
            assert [r['sample_token'] for r in current['records']] == [r['sample_token'] for r in previous['records']]
            assert [r['foreground'] for r in current['records']] == [r['foreground'] for r in previous['records']]
    write(RUN / 'pairing_audit.json', dict(
        status='passed', frames=34149, scenes=850,
        baseline_checkpoint_sha256=read(BASELINE / 'protocol.json')['checkpoint_sha256'],
        checkpoint_sha256=read(RUN / 'protocol.json')['checkpoint_sha256'],
        frame_tokens_and_GT_foreground_counts_identical=True,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pilot', 'run', 'report'), default='run', nargs='?')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'diagnosis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = configure()
        if args.action == 'pilot':
            census.pilot()
            return
        if args.action == 'run':
            write(RUN / 'status.json', dict(status='running', pid=os.getpid(),
                phase='full train/val census', checkpoint_sha256=protocol['checkpoint_sha256']))
            try:
                if not (RUN / 'pilot.json').exists():
                    census.pilot()
                assert read(RUN / 'pilot.json')['status'] == 'passed'
                census.run(args.workers)
                audit()
            except BaseException as error:
                write(RUN / 'status.json', dict(status='failed', pid=os.getpid(), error=repr(error)))
                raise
        from tools.report_camera_set_finetune import report
        report()
        write(RUN / 'status.json', dict(status='complete', pid=os.getpid(),
            frames=34149, scenes=850, report='analysis/camera_set_finetune_coverage_results.md'))
        print(json.dumps(read(RUN / 'status.json'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
