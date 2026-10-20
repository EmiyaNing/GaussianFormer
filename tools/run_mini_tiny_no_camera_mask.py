"""Run and report the isolated temporal tiny experiment without camera masks."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_mini_occ3d import CLASSES, parse_log

CONFIG = ROOT / 'config/mini_occ3d_4096_1024/opusv1_gaussian_softplus_learned_bias_tiny_8f_no_camera_mask.py'
BASE = ROOT / 'config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py'
RUN = ROOT / 'work_dirs/mini_occ3d_tiny_8f_no_camera_mask'
TRAIN = RUN / 'train_seed928'
REPORT = ROOT / 'plans/sparse_render_offical_9_28/mini_tiny_8f_no_camera_mask_results.md'


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def summarize(state, cfg):
    epochs = parse_log(sorted(TRAIN.glob('20*.log')))
    result = dict(status=state['status'], updated_at=now(), train_samples=4096,
                  val_samples=1024, max_epochs=cfg.max_epochs, camera_mask=False,
                  num_frames=8, seed=928, gradient_accumulation=8, epochs=epochs)
    lines = ['# Tiny 8 帧 mini：训练和评估均不使用 camera mask', '',
             f"状态：{state['status']}；完整验证 {len(epochs)}/{cfg.max_epochs} 轮。", '',
             '4096 train / 1024 val；600 queries、76,800 Gaussian；seed 928；batch 1、累积 8。',
             'Occupancy CE/Lovasz 与 mIoU/占用 IoU 均覆盖完整网格。'
             'Camera mask 不传入模型及 loss；OPUS set loss 对所有非空 GT 使用原有距离加权规则。',
             '保留 tiny 的 8 帧输入、初始化、4000 micro-step warmup、优化器和学习率组比例。', '',
             '| epoch | mIoU (%) | 占用 IoU (%) |', '| --- | ---: | ---: |']
    lines += [f"| {r['epoch']} | {r['mIoU']:.2f} | {r['IoU']:.2f} |" for r in epochs]
    if epochs:
        best = max(epochs, key=lambda r: (r['mIoU'], -r['epoch']))
        for metric in ('mIoU', 'IoU'):
            row = max(epochs, key=lambda r: (r[metric], -r['epoch']))
            result[f'best_{metric}'] = row
            target = TRAIN / f"epoch_{row['epoch']}.pth"
            if not target.is_file():
                raise FileNotFoundError(target)
            link = TRAIN / f'best_{metric}.pth'
            if link.is_symlink():
                if os.readlink(link) != target.name:
                    link.unlink()
                else:
                    continue
            if not link.exists():
                link.symlink_to(target.name)
        lines += ['', f"最佳 mIoU：{best['mIoU']:.2f}%（epoch {best['epoch']}），同轮占用 IoU：{best['IoU']:.2f}%。", '',
                  '| 类别（最佳 mIoU checkpoint） | IoU (%) |', '| --- | ---: |']
        lines += [f"| {name} | {best['per_class'][name]:.2f} |" for name in CLASSES]
    else:
        lines += ['', '尚无完整验证结果。']
    lines += ['', '本实验为单种子 mini 实验；完整网格分数与此前 camera-mask 分数的评估区域不同，'
              '二者差值不能单独解释为模型质量变化。当前没有同预算的 tiny 8 帧 camera-mask 对照结果。', '',
              f'配置：`{CONFIG.relative_to(ROOT)}`。',
              f'原始日志及 checkpoint：`{TRAIN.relative_to(ROOT)}`。', '']
    write(RUN / 'summary.json', result)
    write(REPORT, '\n'.join(lines))
    return epochs


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
        with (RUN / 'launcher.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=ROOT)
        print('Detached experiment PID:', process.pid)
        return
    lock = (RUN / 'execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from mmengine import Config
    cfg, baseline = Config.fromfile(str(CONFIG)), Config.fromfile(str(BASE))
    expected = baseline.to_dict()
    expected.update(max_epochs=cfg.max_epochs, eval_mask_flag=False, occ3d_eval_mask='none')
    expected['loss_input_convertion']['occ_mask'] = 'occ_loss_mask'
    expected['occ3d_return_keys'].remove('occ_cam_mask')
    for phase, split in [('train', 'train'), ('val', 'val')]:
        expected[f'{phase}_dataset_config']['imageset'] = f'data/mini_occ3d_4096_1024/{split}.pkl'
        expected[f'{phase}_dataset_config']['return_keys'] = expected['occ3d_return_keys']
    if expected != cfg.to_dict():
        raise ValueError('Unexpected change relative to temporal tiny baseline')
    fingerprint = hashlib.sha256(cfg.pretty_text.encode()).hexdigest()
    protocol = dict(config=str(CONFIG.relative_to(ROOT)), sha256=fingerprint, seed=928, accumulation=8,
                    mini_manifest=read(ROOT / 'data/mini_occ3d_4096_1024/manifest.json'))
    previous = read(RUN / 'protocol.json')
    if previous and previous != protocol:
        raise ValueError('Refusing to mix different experiment protocols')
    TRAIN.mkdir(parents=True, exist_ok=True)
    if list(TRAIN.glob('*.pth')) and not previous:
        raise ValueError('Untracked checkpoints in experiment directory')
    write(RUN / 'protocol.json', protocol)
    resolved = RUN / 'resolved_config.py'
    cfg.dump(str(resolved))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTORCH_ALLOC_CONF='expandable_segments:True',
               OPUS_TEMPORAL_MEMORY_FILE=str(TRAIN / 'phase_memory.jsonl'))
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    state = dict(status='running', pid=os.getpid(), started_at=now())
    child = None

    def save():
        state['updated_at'] = now()
        write(RUN / 'status.json', state)
        return summarize(state, cfg)

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        preflight = read(RUN / 'preflight.json')
        if not (preflight.get('passed') and preflight.get('camera_mask_disabled')
                and preflight.get('train_samples') == 4096 and preflight.get('val_samples') == 1024):
            raise ValueError('Real train/backward/eval preflight must pass before launching')
        command = [sys.executable, '-u', 'tools/train_opus_gaussian_8f.py',
                   '--py-config', str(resolved), '--work-dir', str(TRAIN),
                   '--gradient-accumulation', '8', '--seed', '928']
        state.update(command=command, phase='training_and_validation')
        with (TRAIN / 'console.log').open('a') as log:
            child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save()
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    progress = [line for line in (TRAIN / 'console.log').read_text(errors='replace').splitlines()
                                if '[TRAIN] Epoch' in line or '[EVAL] Epoch' in line]
                    if progress:
                        state['latest_progress'] = progress[-1]
                    save()
        if child.returncode:
            raise RuntimeError(f'Training exited {child.returncode}; see train_seed928/console.log')
        epochs = save()
        if [r['epoch'] for r in epochs] != list(range(1, cfg.max_epochs + 1)):
            raise RuntimeError('Training exited without all complete validation epochs')
        state.update(status='complete', phase=None, finished_at=now())
        save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
