"""Freeze controls, train one mini epoch, validate and report."""
from pathlib import Path
import argparse, fcntl, json, os, shutil, subprocess, sys, threading, time, traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.unweighted_strict_classwise_set_experiment import *


def freeze():
    from tools.strict_classwise_set_experiment import assert_protocol as strict_protocol
    strict_protocol()
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'protocol.json').exists():
        return assert_protocol()
    check = json.loads((RUN / 'numerical_checks.json').read_text())
    assert check['status'] == 'passed'
    c, d = config(), config('strict_classwise')
    assert c.model == d.model and c.optimizer == d.optimizer
    assert c.train_dataset_config == d.train_dataset_config
    assert c.val_dataset_config == d.val_dataset_config
    assert c.loss.loss_cfgs[1] == d.loss.loss_cfgs[1]
    controls = {str(p.relative_to(ROOT)): sha(p) for p in (INITIAL, BASELINE,
        CLASSWISE_RUN/'train/classwise/epoch_1.pth',
        STRICT_RUN/'train/strict_classwise/epoch_1.pth')}
    evals = {
        'baseline': (BASELINE_EVAL, BASELINE),
        'classwise': (CLASSWISE_RUN/'eval/classwise', CLASSWISE_RUN/'train/classwise/epoch_1.pth'),
        'strict_classwise': (STRICT_RUN/'eval/strict_classwise', STRICT_RUN/'train/strict_classwise/epoch_1.pth'),
    }
    for folder, checkpoint in evals.values():
        meta = json.loads((folder/'complete.json').read_text())
        assert meta['frames'] == 6019 and meta['scenes'] == 150
        assert meta['checkpoint_sha256'] == sha(checkpoint)
        paths = [folder/'complete.json', *sorted((folder/'scenes').glob('*.json')),
                 *sorted((folder/'scenes').glob('*.npz'))]
        assert len(paths) == 301
        controls.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    sources = protocol_files()
    protocol = dict(version=1, created=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        sources={str(p.relative_to(ROOT)): sha(p) for p in sources}, controls=controls,
        initial=str(INITIAL.relative_to(ROOT)), seed=928, train_frames=4096,
        train_scenes=698, epochs=1, optimizer_updates=2048, accumulation=2,
        val_frames=6019, val_scenes=150, set_class_weights=[1]*17,
        reduction='uniform mean over active predicted classes for both geometry and classification',
        classification_target='nearest GT of current detached predicted class; unchanged from D',
        controls_preserved='same-class grouping, all six stages, per-class mean, active-class skip rule, distance weights, rare-class geometry weighting, spatial focal weighting, Occupancy CE+Lovasz, model, renderer and optimizer',
        no_validation_tuning=True,
        user_confirmation='2026-10-27: 接下来的实验中回到上一版 loss，去掉类别加权后再微调并看性能。')
    for path in sources:
        target = RUN/'sources'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write(RUN/'protocol.json', protocol)
    return protocol


def run():
    freeze()
    from tools.report_unweighted_strict_classwise_set_mini import REPORT
    stop = threading.Event()
    def progress():
        while not stop.is_set():
            train_path = RUN/'train/unweighted_classwise/progress.json'
            eval_path = RUN/'eval/unweighted_classwise/progress.json'
            train = json.loads(train_path.read_text()) if train_path.exists() else {}
            evaluation = json.loads(eval_path.read_text()) if eval_path.exists() else {}
            t = '4096/4096' if train.get('status') == 'complete' else f"{train.get('microbatches', 0)}/4096"
            e = '6019/6019' if evaluation.get('status') == 'complete' else f"{evaluation.get('scenes', 0)}/150 场景"
            REPORT.write_text('\n'.join(['# 逐类 Set loss 去除类别加权：实验结果', '',
                '**实验进行中，当前不判断效果。**', '',
                '设计和消融范围见[实验设计](unweighted_strict_classwise_set_mini_design.md)。', '',
                '| 阶段 | 进度 |', '| --- | --- |', f'| E训练 | {t}帧 |', f'| E完整验证 | {e} |', '']))
            stop.wait(55)
    watcher = threading.Thread(target=progress, daemon=True); watcher.start()
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    jobs = [
        ('train_unweighted_classwise', ['tools/train_unweighted_strict_classwise_set_mini.py', 'unweighted_classwise']),
        ('eval_unweighted_classwise', ['tools/eval_unweighted_strict_classwise_set_mini.py', 'unweighted_classwise']),
        ('report', ['tools/report_unweighted_strict_classwise_set_mini.py']),
    ]
    try:
        for phase, args in jobs:
            write(RUN/'queue_progress.json', dict(status='running', phase=phase,
                pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            print('Starting '+phase, flush=True)
            with (RUN/(phase+'.log')).open('a', buffering=1) as log:
                result = subprocess.run([sys.executable, *args], env=env, cwd=ROOT,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{phase} returned {result.returncode}; see {phase}.log')
        write(RUN/'queue_progress.json', dict(status='complete', pid=os.getpid(),
            updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
    finally:
        stop.set(); watcher.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args(); os.chdir(ROOT); RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            freeze() if args.freeze_only else run()
        except Exception as error:
            write(RUN/'queue_progress.json', dict(status='failed', error=repr(error),
                traceback=traceback.format_exc(), pid=os.getpid()))
            raise
