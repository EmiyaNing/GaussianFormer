"""Locked, resumable single-4090 queue: tiny -> small -> middle -> large."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work_dirs/full_occ3d_gaussian_8f_30e'
SIZES = ('tiny', 'small', 'middle', 'large')


def main():
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (RUN / 'launcher.pid').write_text(str(os.getpid()) + '\n')
    env = dict(os.environ, PYTORCH_ALLOC_CONF='expandable_segments:True', CUDA_VISIBLE_DEVICES='0')
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    child = None
    current = None
    completed = []

    def state(name, **extra):
        content = dict(state=name, current_size=current, completed_sizes=completed,
                       queue=list(SIZES), target_epochs=30, pid=os.getpid(),
                       updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'), **extra)
        tmp = RUN / 'status.json.tmp'
        tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(RUN / 'status.json')

    def summarize():
        subprocess.run([sys.executable, 'tools/summarize_opus_gaussian_8f.py'],
                       check=True, env=env)

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
        state('interrupted', signal=signum)
        summarize()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for size in SIZES:
            path = RUN / 'preflight' / f'{size}.json'
            if not path.exists() or not json.loads(path.read_text()).get('passed'):
                raise RuntimeError(f'Full-capacity CUDA preflight missing for {size}')
        # A real CUDA compute probe is required; NVML is not used for this.
        subprocess.run([sys.executable, '-c',
                        'import torch; x=torch.ones(8,device="cuda"); '
                        'assert x.sum().item()==8; torch.cuda.synchronize()'],
                       env=env, check=True)
        for current in SIZES:
            directory = RUN / current
            directory.mkdir(parents=True, exist_ok=True)
            resolved = directory / 'resolved_config.py'
            if not resolved.exists():
                from mmengine import Config
                cfg = Config.fromfile(f'config/full_occ3d_oups_guassian/'
                                      f'opusv1_gaussian_softplus_learned_bias_{current}_8f.py')
                cfg.dump(str(resolved))
            state('training')
            summarize()
            child_env = dict(env, OPUS_TEMPORAL_MEMORY_FILE=str(directory / 'phase_memory.jsonl'))
            command = [sys.executable, '-u', 'tools/train_opus_gaussian_8f.py',
                       '--py-config', str(resolved),
                       '--work-dir', str(directory), '--gradient-accumulation', '8', '--seed', '928']
            with (directory / 'console.log').open('a') as console:
                child = subprocess.Popen(command, env=child_env, stdout=console,
                                         stderr=subprocess.STDOUT)
                state('training', train_pid=child.pid)
                while child.poll() is None:
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        summarize()
                if child.returncode:
                    raise RuntimeError(f'{current} training exited {child.returncode}')
            summarize()
            results = json.loads((RUN / 'summary.json').read_text())
            if not results['models'][current]['complete']:
                raise RuntimeError(f'{current} exited without 30 complete evaluations')
            completed.append(current)
        current = None
        state('complete')
        summarize()
    except Exception as exc:
        if child is not None and child.poll() is None:
            child.terminate()
        state('failed', error=str(exc))
        summarize()
        raise


if __name__ == '__main__':
    main()
