"""Resume the mini all-stage set-loss removal and train-selected calibration."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.center_support_common import write_json

RUN = ROOT / 'work_dirs/no_set_loss_ablation_mini'
CONFIG = 'config/mini_occ3d_4096_1024/opusv1_gaussian_no_set_loss.py'
TRAIN = RUN / 'train/no_set_loss_seed928'


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    if args.status:
        state = read(RUN / 'status.json')
        for key in ('pid', 'child_pid'):
            pid = state.get(key)
            state[key + '_alive'] = bool(pid and Path(f'/proc/{pid}').exists())
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if args.detach:
        with (RUN / 'console.log').open('a') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print('Detached no-set-loss experiment PID:', process.pid)
        return
    lock = (ROOT / 'work_dirs/center_support_contribution1/execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from mmengine import Config
    cfg = Config.fromfile(CONFIG)
    reference = Config.fromfile('config/center_support_contribution1/a0_original.py')
    expected = reference.to_dict()
    expected['loss']['loss_cfgs'] = [x for x in expected['loss']['loss_cfgs'] if x['type'] == 'OccupancyLoss']
    if cfg.to_dict() != expected:
        raise ValueError('Ablation changes more than removing the full OPUS set loss')
    fingerprint = hashlib.sha256((cfg.pretty_text + '\nseed=928\naccumulation=4').encode()).hexdigest()
    TRAIN.mkdir(parents=True, exist_ok=True)
    previous = read(TRAIN / 'protocol.json')
    if previous and previous['sha256'] != fingerprint:
        raise ValueError('Training protocol changed')
    if list(TRAIN.glob('*.pth')) and not previous:
        raise ValueError('Refusing untracked training checkpoints')
    write_json(TRAIN / 'protocol.json', dict(sha256=fingerprint, config=CONFIG, seed=928, accumulation=4))
    state = dict(status='running', pid=os.getpid(), started_at=now(), updated_at=now(), phase=None)
    child = None
    def save():
        state['updated_at'] = now()
        write_json(RUN / 'status.json', state)
    def stop(signum, frame):
        if child and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        trained = (read(TRAIN / 'training_runtime.json').get('status') == 'complete' and
                   all((TRAIN / f'epoch_{i}.pth').is_file() and (TRAIN / f'epoch_{i}.pth').stat().st_size > 2**20 for i in range(1, 6)))
        jobs = [] if trained else [('training', [sys.executable, 'tools/train_center_support.py',
            '--py-config', CONFIG, '--work-dir', str(TRAIN.relative_to(ROOT)), '--seed', '928',
            '--gradient-accumulation', '4'], TRAIN / 'console.log')]
        jobs.append(('evaluation_and_calibration', [sys.executable, 'tools/finish_no_set_loss_ablation.py'], RUN / 'evaluation.log'))
        for name, command, logfile in jobs:
            state.update(phase=name, command=command, phase_started_at=now())
            print(now(), 'starting', name, flush=True)
            with logfile.open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save()
                while child.poll() is None:
                    time.sleep(5)
                    if name == 'training':
                        progress = [line for line in logfile.read_text(errors='replace').splitlines()
                                    if '[TRAIN] Epoch' in line or '[EVAL]' in line]
                        if progress:
                            state['latest_training_log'] = progress[-1]
                    save()
            if child.returncode:
                raise RuntimeError(f'{name} exited {child.returncode}; see {logfile}')
            print(now(), 'completed', name, flush=True)
        summary = read(RUN / 'summary.json')
        if summary.get('status') != 'complete':
            raise RuntimeError('Evaluation did not produce a complete experiment summary')
        state.update(status='complete', phase=None, finished_at=now())
        save()
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
