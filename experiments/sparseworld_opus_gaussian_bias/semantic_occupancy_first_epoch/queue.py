"""Durable serial GPU queue; both routes stop at the saved first-epoch boundary."""
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from ..train import atomic_json, digest
from ..ab_first_epoch.report import read
from .protocol import ANALYSIS, ARMS, ROOT


def main():
    lock = (ROOT / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    audit = read(ANALYSIS / 'configuration_audit.json')
    launch = read(ANALYSIS / 'launch.json')
    for path, expected in launch['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'validated source changed: {path}')
    for name, path in ARMS.items():
        if digest(path) != audit['arms'][name]['config_sha256']:
            raise ValueError(f'validated standalone configuration changed: {name}')
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), MPLCONFIGDIR='/tmp/sparseworld_de_mpl',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

    def run(command, log_path, phase, arm=None):
        print(f'DE_QUEUE phase={phase} arm={arm}', flush=True)
        with log_path.open('a', buffering=1) as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            atomic_json(ANALYSIS / 'status.json', dict(status='running', phase=phase, arm=arm,
                command=command, child_pid=child.pid, queue_pid=os.getpid(), planned_epochs_per_arm=15,
                runtime_stop_after_epoch=1, updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
            result = child.wait()
        if result:
            atomic_json(ANALYSIS / 'status.json', dict(status='failed', phase=phase, arm=arm,
                returncode=result, log=str(log_path), queue_pid=os.getpid()))
            raise SystemExit(result)

    if not (ANALYSIS / 'C_reference_eval/reference_verified.json').exists():
        run([sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch.reference'],
            ROOT / 'C_reference_eval.log', 'C_readonly_paired_reference')
    for name, config in ARMS.items():
        work = ROOT / name
        work.mkdir(exist_ok=True)
        if not (work / 'first_epoch_completed.json').exists():
            run([sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch.train',
                 '--config', str(config), '--work-dir', str(work), '--stop-after-epoch', '1'],
                work / 'train.log', 'first_epoch_train_and_diagnose', name)
        run([sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch.report'],
            ROOT / 'report.log', 'first_epoch_comparison', name)
    atomic_json(ANALYSIS / 'status.json', dict(status='paused_after_first_epoch_diagnosis',
        planned_epochs_per_arm=15, completed_training_epochs_per_arm=1, diagnostic_epoch=1,
        completed_arms=list(ARMS), report=str(ANALYSIS / 'report.md'),
        judgement=read(ANALYSIS / 'diagnosis_completed.json'),
        updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
    print('D/E complete: both saved at epoch one of the preserved 15-epoch schedule.', flush=True)


if __name__ == '__main__':
    main()

