"""Preserve the 15-epoch runner; optionally pause at a saved epoch boundary."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

from mmengine.config import Config

from .. import train as runner
from . import focal_loss  # Register A without editing existing package imports.
from .diagnostics import evaluate


class EpochBoundaryReached(Exception):
    pass


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_boundary(path, value, original_save, stop_epoch, cfg, work):
    original_save(path, value)
    path = Path(path)
    if stop_epoch is None or path.name != f'epoch_{stop_epoch:02d}.pth':
        return
    if value['epoch'] != stop_epoch or value['next_iter'] != 0 or value['phase'] != 'train':
        raise ValueError('pause boundary did not preserve the complete epoch state')
    if stop_epoch == 1 and value['updates'] != 1024:
        raise ValueError('first epoch must contain exactly 1024 optimizer updates')
    temporary = work / 'latest.pth.tmp'
    shutil.copyfile(path, temporary)
    os.replace(temporary, work / 'latest.pth')
    record = dict(status='paused_after_epoch', trained_epochs=stop_epoch, planned_epochs=cfg.max_epochs,
                  checkpoint=str(path), checkpoint_sha256=runner.digest(path),
                  config_sha256=value['config_sha256'], manifest_sha256=value['manifest_sha256'],
                  updates=value['updates'], next_iter=value['next_iter'], phase=value['phase'],
                  resume_epoch=stop_epoch + 1)
    runner.atomic_json(work / f'epoch_{stop_epoch:02d}_boundary.json', record)
    if stop_epoch == 1:
        runner.atomic_json(work / 'first_epoch_completed.json', record)
    runner.atomic_json(Path(cfg.report_directory) / 'status.json', record)
    print(f'PAUSED_AFTER_EPOCH epoch={stop_epoch} planned_epochs={cfg.max_epochs} updates={value["updates"]}', flush=True)
    raise EpochBoundaryReached


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stop-after-epoch', type=int)
    options, remaining = parser.parse_known_args()
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + remaining
    config_path = remaining[remaining.index('--config') + 1]
    work = Path(remaining[remaining.index('--work-dir') + 1])
    cfg = Config.fromfile(config_path)
    if cfg.max_epochs != 15 or cfg.loss.mask_type != 'none':
        raise ValueError('A/B must retain max_epochs=15 and full-space supervision')
    if options.stop_after_epoch is not None and not 1 <= options.stop_after_epoch <= cfg.max_epochs:
        raise ValueError('invalid runtime epoch boundary')
    if (options.stop_after_epoch == 1 and (work / 'first_epoch_completed.json').exists() and
            '--smoke' not in remaining and '--evaluate-best' not in remaining):
        print('First epoch already complete; preserved checkpoint will not be advanced.', flush=True)
        return
    original_save, original_evaluate = runner.atomic_save, runner.evaluate
    original_initialize, original_lr = runner.load_image_initialization, runner.set_lr
    runner.evaluate_original = original_evaluate

    def initialize(model, path):
        audit = original_initialize(model, path)
        audit['initial_model_sha256'] = model_fingerprint(model)
        return audit

    def learning_rate(optimizer, step, total, warmup, minimum):
        if total != 15360:
            raise ValueError('runtime stopping must not shorten the 15-epoch LR schedule')
        if step == 0:
            runner.atomic_json(work / 'optimization_budget.json', dict(
                planned_epochs=15, total_updates=total, first_epoch_updates=1024,
                warmup_updates=warmup, minimum_lr_ratio=minimum,
                runtime_stop_after_epoch=options.stop_after_epoch))
        return original_lr(optimizer, step, total, warmup, minimum)

    runner.load_image_initialization = initialize
    runner.set_lr = learning_rate
    runner.evaluate = original_evaluate if '--evaluate-best' in remaining else evaluate
    runner.atomic_save = lambda path, value: save_boundary(
        path, value, original_save, options.stop_after_epoch, cfg, work)
    try:
        runner.main()
    except EpochBoundaryReached:
        pass
    finally:
        runner.atomic_save, runner.evaluate = original_save, original_evaluate
        runner.load_image_initialization, runner.set_lr = original_initialize, original_lr
        del runner.evaluate_original
        sys.argv = original_argv


if __name__ == '__main__':
    main()
