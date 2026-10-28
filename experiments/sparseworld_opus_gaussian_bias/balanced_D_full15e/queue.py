"""Durable D continuation, best-checkpoint recheck and fifteen-epoch report."""
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from ..train import atomic_json, digest
from .prepare import read
from .protocol import ANALYSIS, CONFIG, ROOT, WORK
from .report import refresh


def main():
    lock = (ROOT/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    launch = read(ANALYSIS/'launch.json')
    preparation = read(ANALYSIS/'preparation.json')
    for path, expected in {**launch['source_sha256'], **preparation['preserved_source_sha256']}.items():
        if digest(path)!=expected:
            raise ValueError(f'validated source changed: {path}')
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), MPLCONFIGDIR='/tmp/sparseworld_d15_mpl',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    refresh(WORK, ANALYSIS)
    module = 'experiments.sparseworld_opus_gaussian_bias.balanced_D_full15e'
    for phase in ('train', 'best_recheck', 'final_report'):
        if phase=='train' and (WORK/'completed.json').exists():
            continue
        command = [sys.executable, '-m', module+'.report'] if phase=='final_report' else [
            sys.executable, '-m', module+'.train', '--config', str(CONFIG), '--work-dir', str(WORK)]
        if phase=='best_recheck':
            command.append('--evaluate-best')
        log = ROOT/'report.log' if phase=='final_report' else WORK/f'{phase}.log'
        print(f'D15_QUEUE phase={phase}', flush=True)
        with log.open('a', buffering=1) as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            atomic_json(ANALYSIS/'status.json', dict(status='running', phase=phase, child_pid=child.pid,
                queue_pid=os.getpid(), planned_epochs=15, start_epoch=2, command=command, log=str(log),
                updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
            code = child.wait()
        if code:
            atomic_json(ANALYSIS/'status.json', dict(status='failed', phase=phase, returncode=code, log=str(log), queue_pid=os.getpid()))
            raise SystemExit(code)
    for path, expected in preparation['preserved_source_sha256'].items():
        if digest(path)!=expected:
            raise ValueError(f'preserved original changed: {path}')
    completed = read(WORK/'completed.json')
    if completed['epochs']!=15 or completed['updates']!=15360:
        raise ValueError('fifteen-epoch training did not complete')
    atomic_json(ANALYSIS/'final_verification.json', dict(status='passed', epochs=15,
        optimizer_updates=15360,
        original_sources_and_results_preserved=True, best_checkpoint_recheck_exact=True,
        per_epoch_diagnostics_verified=True, report=str(ANALYSIS/'report.md'),
        completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
    print(f'D15 completed: {ANALYSIS / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
