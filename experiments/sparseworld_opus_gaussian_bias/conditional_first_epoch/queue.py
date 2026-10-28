"""Train one complete C epoch, compare automatically, then remain paused."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from ..train import atomic_json, digest
from ..ab_first_epoch.report import read
from .protocol import ANALYSIS, CONFIG, ROOT, WORK


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    audit = read(ANALYSIS / 'configuration_audit.json')
    if digest(CONFIG) != audit['config_sha256']:
        raise ValueError('C standalone configuration changed after validation')
    launch = read(ANALYSIS / 'launch.json')
    for path, expected in launch['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'validated C source changed: {path}')
    WORK.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), MPLCONFIGDIR='/tmp/sparseworld_conditional_mpl',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

    def run(command, log_path, phase):
        atomic_json(ANALYSIS / 'status.json', dict(status='running', phase=phase, command=command,
                    planned_epochs=15, runtime_stop_after_epoch=1, conditional_semantic_weight=1.))
        print(f'C_QUEUE phase={phase}', flush=True)
        with log_path.open('a', buffering=1) as stream:
            child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        if child.returncode:
            atomic_json(ANALYSIS / 'status.json', dict(status='failed', phase=phase,
                        returncode=child.returncode, log=str(log_path)))
            raise SystemExit(child.returncode)

    if not (WORK / 'first_epoch_completed.json').exists():
        run([sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.conditional_first_epoch.train',
             '--config', str(CONFIG), '--work-dir', str(WORK), '--stop-after-epoch', '1'],
            WORK / 'train.log', 'first_epoch_train_and_diagnose')
    run([sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.conditional_first_epoch.report'],
        ROOT / 'report.log', 'first_epoch_comparison')
    atomic_json(ANALYSIS / 'status.json', dict(status='paused_after_first_epoch_diagnosis',
                planned_epochs=15, completed_training_epochs=1, diagnostic_epoch=1,
                report=str(ANALYSIS / 'report.md'), judgement=read(ANALYSIS / 'diagnosis_completed.json')))
    print('C first epoch and comparison complete; saved 15-epoch protocol remains paused.', flush=True)


if __name__ == '__main__':
    main()
