"""Execute and resume every first-round contribution-1 real experiment serially.

GPU jobs share one lock. Completed artifacts are checked before skipping. A
failure stops the queue with its log and command recorded. Reports refresh each
minute and after each task, so launching a job is never reported as completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.center_support_common import write_json
from tools.report_center_support_execution import CONTROLS, refresh

RUN = ROOT / 'work_dirs/center_support_contribution1'
FULL_CONFIG = 'config/full_occ3d_10e/opusv1_gaussian_softplus_learned_bias_no_benchmark.py'
FULL_CHECKPOINTS = 'work_dirs/full_occ3d_10e_no_benchmark/opusv1_gaussian_softplus_learned_bias'


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def complete(task):
    if task['kind'] == 'train':
        directory = (ROOT / task['output']).parent
        cost = read(directory / 'training_runtime.json')
        if not cost or cost.get('status') != 'complete':
            return False
        return all((directory / f'epoch_{epoch}.pth').is_file() and
                   (directory / f'epoch_{epoch}.pth').stat().st_size > 2**20
                   for epoch in range(1, 6))
    data = read(ROOT / task['output'])
    if not data or data.get('status') != 'complete':
        return False
    protocol = data.get('protocol', {})
    for flag in ('--checkpoint', '--py-config', '--pairs', '--samples-per-domain', '--limit'):
        if flag in task['command']:
            value = task['command'][task['command'].index(flag) + 1]
            key = flag[2:].replace('-', '_')
            if str(protocol.get(key)) != value:
                raise ValueError(f'Existing artifact has different {flag}: {task["output"]}')
    if task['kind'] == 'diagnostic':
        if bool(protocol.get('full_occupied', False)) != ('--full-occupied' in task['command']):
            raise ValueError('Complete attribution has a different occupied sampling domain')
        pairs = int(task['command'][task['command'].index('--pairs') + 1])
        if len(data.get('pairs', [])) != pairs:
            raise ValueError('Complete diagnostic has wrong scene count')
        for pair in data['pairs']:
            if not pair['contribution'].get('renderer_parity_passed'):
                raise ValueError('Complete diagnostic lacks renderer parity')
            if '--intervene' in task['command'] and len(pair.get('interventions', [])) != 45:
                raise ValueError('Complete diagnostic lacks all 15 branches x 3 radii')
    elif task['kind'] == 'evaluation' and not data.get('streaming_official_parity_passed'):
        raise ValueError('Complete evaluation lacks official RayIoU parity')
    return True


def diagnostic(name, config, checkpoint, pairs=8, samples=1024, only=False, full=False):
    output = f'work_dirs/center_support_contribution1/real/{name}.json'
    command = [sys.executable, 'tools/diagnose_center_support.py', '--py-config', config,
        '--checkpoint', checkpoint, '--pairs', str(pairs), '--samples-per-domain', str(samples),
        '--out', output, '--scene-pairs', 'work_dirs/center_support_contribution1/scene_pairs.json', '--resume']
    command += ['--contributions-only'] if only else ['--intervene']
    if full:
        command += ['--full-occupied']
    return dict(name=name, kind='diagnostic', command=command, output=output, status='pending')


def evaluation(name, config, checkpoint, output):
    return dict(name=name, kind='evaluation', command=[sys.executable, 'tools/eval_center_support.py',
        '--py-config', config, '--checkpoint', checkpoint, '--out', output], output=output, status='pending')


def tasks():
    result = []
    for epoch in (10, 1, 5):
        checkpoint = f'{FULL_CHECKPOINTS}/epoch_{epoch}.pth'
        result.append(diagnostic(f'epoch{epoch}_intervene', FULL_CONFIG, checkpoint))
    # Fix the 32-pair expansion before inspecting later checkpoint outcomes.
    # Eight-pair screening showed intervals crossing zero; expand all anchors.
    for epoch in (10, 1, 5):
        checkpoint = f'{FULL_CHECKPOINTS}/epoch_{epoch}.pth'
        result.append(diagnostic(f'epoch{epoch}_intervene_32', FULL_CONFIG, checkpoint, pairs=32))
    for epoch in (1, 5, 10):
        checkpoint = f'{FULL_CHECKPOINTS}/epoch_{epoch}.pth'
        for samples in (4096, 16384):
            result.append(diagnostic(f'epoch{epoch}_sampling{samples}', FULL_CONFIG, checkpoint, samples=samples, only=True))
        result.append(diagnostic(f'epoch{epoch}_full_occupied', FULL_CONFIG, checkpoint, only=True, full=True))
    for name in CONTROLS:
        config = f'config/center_support_contribution1/{name}.py'
        directory = f'work_dirs/center_support_contribution1/train/{name}_seed928'
        result.append(dict(name=f'train_{name}_seed928', kind='train',
            command=[sys.executable, 'tools/train_center_support.py', '--py-config', config, '--work-dir', directory,
                     '--gradient-accumulation', '4', '--seed', '928'],
            output=f'{directory}/epoch_5.pth', status='pending'))
        result.append(evaluation(f'eval_{name}_epoch5', config, f'{directory}/epoch_5.pth',
                                 f'{directory}/evaluation_epoch5.json'))
    # Same training schedule and final set-domain alignment for budget comparison.
    for budget, config_name in (('high', 'a1_align_final'), ('low', 'low_budget_19200')):
        config = f'config/center_support_contribution1/{config_name}.py'
        for epoch in (1, 5):
            result.append(diagnostic(f'mini_{budget}_epoch{epoch}_intervene', config,
                f'work_dirs/center_support_contribution1/train/{config_name}_seed928/epoch_{epoch}.pth'))
    for epoch in (1, 5, 10):
        result.append(evaluation(f'full_epoch{epoch}_official_eval', FULL_CONFIG,
            f'{FULL_CHECKPOINTS}/epoch_{epoch}.pth',
            f'work_dirs/center_support_contribution1/real/epoch{epoch}_official_eval.json'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--detach', action='store_true', help='Run independently of the interactive tool session')
    parser.add_argument('--handoff', action='store_true', help='Resume this queue in a detached session')
    parser.add_argument('--status', action='store_true', help='Read progress and check actual host processes')
    args = parser.parse_args()
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    if args.status:
        state = read(RUN / 'execution_status.json') or {}
        def alive(pid):
            if not pid:
                return False
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
        current = next((t for t in state.get('tasks', []) if t['name'] == state.get('current_task')), {})
        print(json.dumps(dict(status=state.get('status'), current_task=state.get('current_task'),
            queue_pid=state.get('pid'), queue_alive=alive(state.get('pid')),
            task_pid=current.get('pid'), task_alive=alive(current.get('pid')),
            completed=sum(t['status'] == 'complete' for t in state.get('tasks', [])),
            total=len(state.get('tasks', [])), updated_at=state.get('updated_at')), indent=2))
        return
    if args.handoff:
        active = read(RUN / 'execution.pid.json')
        if active:
            pid = active['pid']
            proc = Path(f'/proc/{pid}')
            if proc.exists():
                argv = (proc / 'cmdline').read_bytes().split(b'\0')
                if not any(part.endswith(b'tools/run_center_support_queue.py') for part in argv):
                    raise RuntimeError('Saved PID is not this queue; refusing to signal it')
                if (proc / 'cwd').resolve() != ROOT:
                    raise RuntimeError('Saved queue PID belongs to another workspace')
                os.kill(pid, signal.SIGTERM)
                for _ in range(100):
                    if not proc.exists():
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('Queue did not terminate; refusing to launch a second worker')
        args.detach = True
    if args.detach:
        with (RUN / 'execution_console.log').open('a') as log:
            detached = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print('Detached contribution-1 queue PID:', detached.pid)
        return
    lock = (RUN / 'execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = tasks()
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    import torch
    import mmengine
    from mmengine import Config
    if not torch.cuda.is_available():
        raise RuntimeError('Host CUDA required; rerun outside sandbox')
    if shutil.disk_usage(RUN).free < 60 * 2**30:
        raise RuntimeError('At least 60 GiB free space required')
    for job in jobs:
        path = job['command'][job['command'].index('--py-config') + 1]
        cfg = Config.fromfile(path)
        if job['kind'] == 'train':
            directory = ROOT / Path(job['output']).parent
            fingerprint = hashlib.sha256((cfg.pretty_text + '\nseed=928\naccumulation=4').encode()).hexdigest()
            previous = read(directory / 'protocol.json')
            if directory.exists() and list(directory.glob('*.pth')) and not previous:
                raise ValueError(f'Untracked checkpoint directory: {directory}')
            if previous and previous['sha256'] != fingerprint:
                raise ValueError(f'Training protocol changed: {directory}')
            write_json(directory / 'protocol.json', dict(sha256=fingerprint, config=path, seed=928, accumulation=4))
    manifest = RUN / 'scene_pairs.json'
    if not manifest.exists():
        full = mmengine.load(Config.fromfile(FULL_CONFIG).val_dataset_config.imageset)
        scenes = sorted({scene for scene, frame in full['metadata']})
        import numpy as np
        initial = np.linspace(0, len(scenes) - 1, 16, dtype=int)
        fixed = [scenes[i] for i in initial]
        rest = [s for s in scenes if s not in fixed]
        if len(rest) < 48:
            raise ValueError('Need 64 distinct validation scenes')
        extra = [rest[i] for i in np.linspace(0, len(rest) - 1, 48, dtype=int)]
        flat = fixed + extra
        write_json(manifest, [flat[i:i + 2] for i in range(0, 64, 2)])
    previous = read(RUN / 'execution_status.json') or {}
    history = {task['name']: task for task in previous.get('tasks', [])}
    for job in jobs:
        if job['name'] in history:
            job.update({k: v for k, v in history[job['name']].items()
                        if k in ('started_at', 'finished_at', 'elapsed_seconds', 'attempts')})
    state = dict(status='running', pid=os.getpid(), started_at=previous.get('started_at', now()),
        updated_at=now(), current_task=None, tasks=jobs,
        scope='First round: three full-data checkpoints, 8/32 scene pairs, sampling sensitivity, eight 5-epoch mini controls, independent matched 19.2k model, official RayIoU')
    write_json(RUN / 'execution.pid.json', dict(pid=os.getpid(), started_at=now()))
    child = None
    def save():
        state['updated_at'] = now()
        write_json(RUN / 'execution_status.json', state)
        refresh()
    def interrupted(signum, frame):
        if child and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        save()
        for job in jobs:
            state['current_task'] = job['name']
            if complete(job):
                job['status'] = 'complete'
                save()
                print(now(), 'already complete', job['name'], flush=True)
                continue
            job.update(status='running', started_at=now(), attempts=job.get('attempts', 0) + 1)
            save()
            output = ROOT / job['output']
            output.parent.mkdir(parents=True, exist_ok=True)
            log_path = (output.parent / 'console.log' if job['kind'] == 'train'
                        else output.with_suffix('.console.log'))
            job['log'] = str(log_path.relative_to(ROOT))
            print(now(), 'starting', job['name'], flush=True)
            started = time.monotonic()
            with log_path.open('a') as log:
                log.write('\nQueue attempt ' + now() + '\n')
                log.flush()
                child = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job['pid'] = child.pid
                save()
                last = time.monotonic()
                while child.poll() is None:
                    time.sleep(5)
                    if time.monotonic() - last >= 60:
                        save()
                        last = time.monotonic()
            job.update(exit_code=child.returncode, finished_at=now(), elapsed_seconds=time.monotonic() - started)
            if child.returncode or not complete(job):
                job['status'] = 'failed'
                raise RuntimeError(f'{job["name"]} failed; see {log_path}')
            job['status'] = 'complete'
            save()
            print(now(), 'completed', job['name'], flush=True)
        state.update(status='complete', current_task=None, finished_at=now())
        save()
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
