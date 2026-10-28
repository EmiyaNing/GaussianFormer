"""Resumable independent runner; preserves existing train.py and eval.py."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor
from torch.utils.data import DataLoader

from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS
from .metrics import HorizonMetrics


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def atomic_save(path, value):
    temp = Path(str(path) + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def move_batch(data, device):
    result = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
              for k, v in data.items()}
    return result.pop('img'), result


def load_image_initialization(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    source = checkpoint.get('state_dict', checkpoint)
    expected = model.state_dict()
    selected = {}
    for key, value in source.items():
        name = key.removeprefix('module.')
        for before, after in (('backbone.', 'img_backbone.'), ('neck.', 'img_neck.')):
            if name.startswith(before):
                name = after + name[len(before):]
                break
        if not name.startswith(('img_backbone.', 'img_neck.')):
            continue
        if name not in expected or expected[name].shape != value.shape:
            raise ValueError(f'incompatible image initialization tensor: {name}')
        selected[name] = value
    required = {k for k in expected if k.startswith(('img_backbone.', 'img_neck.'))}
    missing = required - selected.keys()
    if missing:
        raise ValueError(f'missing image initialization: {sorted(missing)[:10]}')
    incompatible = model.load_state_dict(selected, strict=False)
    if incompatible.unexpected_keys or any(k in required for k in incompatible.missing_keys):
        raise ValueError('initialization audit differs from load result')
    return dict(checkpoint=path, sha256=digest(path), loaded_tensors=len(selected),
                scope='backbone_and_fpn_only', randomly_initialized_world_tensors=len(incompatible.missing_keys))


def make_optimizer(model, cfg):
    options = cfg.optimizer['optimizer']
    keys = cfg.optimizer.get('paramwise_cfg', {}).get('custom_keys', {})
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matched = next((key for key in sorted(keys, key=len, reverse=True) if key in name), None)
        setting = keys.get(matched, {})
        lr = options['lr'] * setting.get('lr_mult', 1.)
        groups.append(dict(params=[parameter], lr=lr, base_lr=lr,
                           weight_decay=options['weight_decay'] * setting.get('decay_mult', 1.)))
    return torch.optim.AdamW(groups)


def set_lr(optimizer, step, total, warmup, minimum):
    if step < warmup:
        factor = 1 / 3 + (2 / 3) * step / max(warmup, 1)
    else:
        progress = (step - warmup) / max(total - warmup, 1)
        factor = minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * factor


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def loader(dataset, cfg, order=None, seed=17):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                      sampler=order, shuffle=False, collate_fn=custom_collate_fn_temporal,
                      pin_memory=True, generator=generator)


@torch.no_grad()
def evaluate(model, dataset, cfg, directory, epoch, limit=None):
    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    metric = HorizonMetrics()
    started = time.monotonic()
    for index, data in enumerate(loader(dataset, cfg.val_loader)):
        images, metas = move_batch(data, 'cuda')
        result = model(imgs=images, metas=metas)
        metric.add(result, metas)
        if (index + 1) % 128 == 0:
            print(f'EVAL epoch={epoch} samples={metric.samples}/{len(dataset)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        if limit is not None and index + 1 >= limit:
            break
    metrics = metric.results()
    metrics.update(epoch=epoch, evaluation_seconds=time.monotonic()-started)
    if limit is None and metrics['num_samples'] != 1024:
        raise ValueError('validation must contain exactly 1024 samples')
    atomic_json(directory / f'epoch_{epoch:02d}_metrics.json', metrics)
    np.savez_compressed(directory / f'epoch_{epoch:02d}_confusion.npz', confusion=metric.confusion)
    model.memory.clear()
    model.memory_queue.clear()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--smoke', action='store_true', help='full-horizon forward/backward and one validation sample')
    parser.add_argument('--evaluate-best', action='store_true')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if '_base_' in Path(args.config).read_text():
        raise ValueError('experiment configurations must be standalone')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run this command outside the restricted GPU sandbox')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_random_seed(cfg.seed)
    directory = Path(args.work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    run_lock = open(directory / 'run.lock', 'w')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('this model work directory already has an active runner')
    run_lock.write(str(os.getpid()))
    run_lock.flush()
    config_hash = digest(args.config)
    manifest_hashes = {split: digest(cfg[f'{split}_dataset_config']['selection_manifest'])
                       for split in ('train', 'val')}
    source_hashes = {}
    for split in ('train', 'val'):
        manifest = json.loads(Path(cfg[f'{split}_dataset_config']['selection_manifest']).read_text())
        for path, expected_hash in manifest['source_files'].items():
            if path not in source_hashes:
                source_hashes[path] = digest(path)
            if source_hashes[path] != expected_hash:
                raise ValueError(f'original dataset source changed after subset preparation: {path}')
    cfg.dump(str(directory / 'resolved_config.py'))
    model = build_segmentor(cfg.model)
    model.init_weights()
    initialization = load_image_initialization(model, cfg.train_load_from)
    atomic_json(directory / 'initialization.json', initialization)
    model.cuda()
    optimizer = make_optimizer(model, cfg)
    loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
    train_data = OPENOCC_DATASET.build(cfg.train_dataset_config)
    val_data = OPENOCC_DATASET.build(cfg.val_dataset_config)
    if (len(train_data), len(val_data)) != (4096, 1024):
        raise ValueError('experiment dataset sizes differ from 4096/1024')
    batch_size = cfg.train_loader.batch_size
    batches_per_epoch = math.ceil(len(train_data) / batch_size)
    total_updates = math.ceil(batches_per_epoch / cfg.gradient_accumulation) * cfg.max_epochs
    epoch, next_iter, updates, best = 0, 0, 0, -1.
    phase = 'train'
    latest = directory / 'latest.pth'
    resume_rng = None
    resume_stamps = None
    if latest.exists() and not args.smoke:
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['config_sha256'] != config_hash or saved['manifest_sha256'] != manifest_hashes:
            raise ValueError('resume configuration or immutable subset changed')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        epoch, next_iter, updates = saved['epoch'], saved['next_iter'], saved['updates']
        best, phase, resume_rng = saved['best_score'], saved['phase'], saved['rng']
        resume_stamps = saved['query_stamps']
        print(f'RESUME epoch={epoch+1} next_iter={next_iter} updates={updates} phase={phase}', flush=True)
    if args.evaluate_best:
        best_checkpoint = torch.load(directory / 'best_mIoU.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(best_checkpoint['state_dict'], strict=True)
        model.set_epoch(best_checkpoint['epoch_index'])
        model.pts_bbox_head.ind_stamps_all = best_checkpoint['query_stamps'].cuda()
        model.pts_bbox_head.reset_mask()
        recheck_dir = directory / 'best_recheck'
        recheck_dir.mkdir(exist_ok=True)
        evaluate(model, val_data, cfg, recheck_dir, best_checkpoint['epoch_index'] + 1)
        return

    def save(progress_epoch, progress_iter, progress_phase, path=latest):
        atomic_save(path, dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=progress_epoch, next_iter=progress_iter, phase=progress_phase,
            updates=updates, best_score=best, rng=capture_rng(),
            config_sha256=config_hash, manifest_sha256=manifest_hashes,
            query_stamps=model.pts_bbox_head.ind_stamps_all,
            initialization=initialization))

    if args.smoke:
        epoch = cfg.max_epochs - 1
    while epoch < cfg.max_epochs:
        model.train()
        statistics = model.pts_bbox_head.num_stamps_all.clone() if resume_rng is not None else None
        model.set_epoch(epoch)
        if statistics is not None:
            model.pts_bbox_head.num_stamps_all.copy_(statistics)
            model.pts_bbox_head.ind_stamps_all = resume_stamps.cuda()
            if not model.pretrain:
                model.pts_bbox_head.reset_mask()
        train_data.set_epoch(epoch)
        if resume_rng is not None:
            restore_rng(resume_rng)
            resume_rng = None
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(train_data)).tolist()
        data_loader = loader(train_data, cfg.train_loader, order[next_iter:], cfg.seed + epoch)
        started = time.monotonic()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        if phase != 'evaluate':
            for local_index, data in enumerate(data_loader):
                index = next_iter + local_index * batch_size
                processed = min(index + batch_size, len(train_data))
                batch_index = index // batch_size
                images, metas = move_batch(data, 'cuda')
                result = model(imgs=images, metas=metas)
                loss, details = loss_fn(dict(result, metas=metas, global_iter=epoch * batches_per_epoch + batch_index))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite loss at epoch {epoch+1}, sample {index}')
                accumulation = min(cfg.gradient_accumulation,
                                   batches_per_epoch - (batch_index // cfg.gradient_accumulation) * cfg.gradient_accumulation)
                (loss / accumulation).backward()
                losses.append(float(loss.detach()))
                boundary = (batch_index + 1) % cfg.gradient_accumulation == 0 or processed == len(train_data)
                if boundary or args.smoke:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
                    set_lr(optimizer, updates, total_updates, cfg.warmup_iters, cfg.min_lr_ratio)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                if (local_index + 1) % cfg.print_freq == 0 or args.smoke:
                    elapsed = time.monotonic() - started
                    row = dict(epoch=epoch+1, sample=processed, updates=updates,
                        mean_loss=float(np.mean(losses[-cfg.print_freq:])),
                        seconds_per_sample=elapsed/(local_index+1)/batch_size,
                        peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                        empty_bias=float(model.gaussian_head.empty_bias.detach()) if hasattr(model,'gaussian_head') else None)
                    print('TRAIN ' + json.dumps(row), flush=True)
                    atomic_json(directory / 'progress.json', dict(status='training', **row))
                if args.smoke:
                    break
                if boundary and updates % cfg.checkpoint_interval_steps == 0:
                    save(epoch, processed, 'train')
            if not args.smoke:
                save(epoch, len(train_data), 'evaluate')
        metrics = evaluate(model, val_data, cfg, directory, epoch + 1, 1 if args.smoke else None)
        print('EPOCH ' + json.dumps(metrics), flush=True)
        if args.smoke:
            atomic_json(directory / 'smoke_success.json', dict(status='passed', metrics=metrics,
                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20))
            return
        score = metrics['selection_mean_miou']
        if score > best:
            best = score
            atomic_save(directory / 'best_mIoU.pth', dict(state_dict=model.state_dict(),
                epoch_index=epoch, metrics=metrics, config_sha256=config_hash,
                query_stamps=model.pts_bbox_head.ind_stamps_all,
                manifest_sha256=manifest_hashes, initialization=initialization))
            atomic_json(directory / 'best.json', metrics)
        save(epoch + 1, 0, 'train', directory / f'epoch_{epoch+1:02d}.pth')
        shutil.copyfile(directory / f'epoch_{epoch+1:02d}.pth', directory / 'latest.pth.tmp')
        os.replace(directory / 'latest.pth.tmp', latest)
        epoch, next_iter, phase = epoch + 1, 0, 'train'
    atomic_json(directory / 'completed.json', dict(status='completed', epochs=15,
        best_score=best, config_sha256=config_hash, manifest_sha256=manifest_hashes))


if __name__ == '__main__':
    main()
