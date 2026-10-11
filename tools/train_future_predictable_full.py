#!/usr/bin/env python3
"""Paired 4096/1024 native-head full fine tuning, isolated from other trainers."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from torch.utils.data import DataLoader, Subset

import model  # noqa: F401
import loss  # noqa: F401
from dataset.utils import custom_collate_fn_temporal
from eval_teacher_aligned import _metric
from loss import OPENOCC_LOSS
from model.multi_horizon_teacher import file_sha256
from model.segmentor.sparseworld_predictability_finetune_segmentor import SparseWorldPredictabilityFinetuneSegmentor
from tools.diagnose_teacher_alignment import loader_for, tree_to, seed_all, add_metric
from tools.diagnose_future_image_joint import metric_row, rows_metrics
from tools.evaluate_future_predictable_native import summarized_rows, relative_to_baseline, load_saved_rows
from tools.evaluate_future2current_native import save_rows
from tools.experiment_future2current import _future_window, teacher_at_student
from tools.experiment_future_predictable_information import save_json
from tools.experiment_future_response_distillation import build_student, paths_for
from tools.summarize_future_predictable_full import write_report

HORIZONS = (2, 4, 6)


def save_report(root, report, cfg):
    save_json(root / 'results.json', report)
    if cfg.full_tune_train_samples == 4096 and cfg.full_tune_val_samples == 1024:
        write_report(root)


def historical_config(cfg):
    result = copy.deepcopy(cfg)
    pipeline = []
    for entry in cfg.train_pipeline:
        entry = copy.deepcopy(entry)
        if entry['type'] in ('LoadMultiViewImageFuture', 'PhotoMetricDistortionMultiViewImage'):
            continue
        if entry['type'] == 'LoadMultiViewImageHistory':
            entry['num_history'] = 4
        if entry['type'] == 'NuScenesAdaptor':
            entry['num_cams'] = 30
        pipeline.append(entry)
    result.train_pipeline = pipeline
    return result


def image_loader(dataset, workers, order=None):
    if order is not None:
        dataset = Subset(dataset, order)
    options = dict(batch_size=1, num_workers=workers, pin_memory=True,
                   collate_fn=custom_collate_fn_temporal,
                   generator=torch.Generator().manual_seed(20260926))
    if workers:
        options.update(prefetch_factor=1)
    return DataLoader(dataset, **options)


def atomic_torch_save(value, path):
    temp = path.with_suffix('.pth.tmp')
    torch.save(value, temp)
    temp.replace(path)


def cached_teacher_path(cfg, root, split, index):
    # Recompute in this run: checkpoint/sample identity alone does not establish
    # numerical identity of older feature caches and the current image pipeline.
    return root / 'teacher_features' / split / f'{index:06d}.pth'


@torch.no_grad()
def perceive(reference, data, imgs, ego, horizon, cfg, native=False):
    selected_imgs, metas = _future_window(data, imgs, horizon)
    state = dict(imgs=selected_imgs, metas=metas, points=None)
    state.update(reference.extract_img_feat(**state))
    state.update(reference.lifter(**state))
    state['query_points'] = state['query_points'] * (
        (torch.tanh(reference.points_scale(ego)) + 1.) * .35 + .8)
    state.update(reference.encoder(**state))
    encoded, current = state['representation'][-1], state['query_stamps'] == 0
    record = dict(features=encoded['query_features'][:, current].half().cpu(),
                  centers=reference._world(encoded['query_points'][:, current]).mean(2).half().cpu())
    row = None
    if native:
        state.update(reference.head(**state))
        metric = _metric(cfg)
        metric.add_batch(state['final_occ_grid'][0].cpu().numpy(),
                         data['future_occ_labels'][0, horizon - 1].cpu().numpy(),
                         data['future_occ_lidar_masks'][0, horizon - 1].cpu().numpy(),
                         data['future_occ_cam_masks'][0, horizon - 1].cpu().numpy())
        row = metric_row(metric)
    return record, row


@torch.no_grad()
def prepare_teachers(reference, images, manifest, cfg, root, split):
    paths = [cached_teacher_path(cfg, root, split, index) for index in manifest['indices']]
    native_rows = [None] * len(paths)
    for i, path in enumerate(paths):
        if path.exists():
            existing = torch.load(path, map_location='cpu', weights_only=False)
            if (existing.get('index'), existing.get('scene'), existing.get('checkpoint_sha256')) != (
                    manifest['indices'][i], manifest['scenes'][i], cfg.student_checkpoint_sha256):
                raise RuntimeError('Teacher cache sample/checkpoint identity differs')
            if split == 'val':
                native_rows[i] = existing.get('native_rows')
    missing = [i for i, path in enumerate(paths) if not path.exists() or
               (split == 'val' and native_rows[i] is None)]
    for i, data in zip(missing, image_loader(images.dataset, cfg.full_tune_workers, missing)):
        data = tree_to(data, 'cuda')
        imgs = data.pop('img')
        ego = reference.plan_head(data['temporal_ego_states'].float().reshape(1, -1)).unsqueeze(1)
        states, sample_rows = {}, {}
        for h in HORIZONS:
            states[h], row = perceive(reference, data, imgs, ego, h, cfg,
                                      native=(split == 'val' and native_rows[i] is None))
            if row is not None:
                sample_rows[h] = row
        destination = root / 'teacher_features' / split / f'{manifest["indices"][i]:06d}.pth'
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = dict(index=manifest['indices'][i], scene=manifest['scenes'][i], states=states,
                      checkpoint_sha256=cfg.student_checkpoint_sha256)
        if sample_rows:
            record['native_rows'] = sample_rows
            native_rows[i] = sample_rows
        atomic_torch_save(record, destination)
        paths[i] = destination
        if (i + 1) % 32 == 0 or i == missing[-1]:
            print(f'[prepare teacher {split}] {i + 1}/{len(paths)}; remaining {len(missing) - missing.index(i) - 1}', flush=True)
    if split == 'val':
        save_rows(root / 'teacher_native.npz', native_rows)
        return paths, summarized_rows(native_rows, cfg)
    return paths, None


def build_full_student(cfg, strict, weights):
    spec = copy.deepcopy(strict.model).to_dict()
    spec.pop('type')
    spec['img_backbone']['frozen_stages'] = -1
    spec['img_backbone']['norm_eval'] = False
    instance = SparseWorldPredictabilityFinetuneSegmentor(**spec).cuda()
    mismatch = instance.load_state_dict(weights, strict=False)
    expected = {name for name, _ in instance.named_parameters() if name.startswith('future_adapter.')}
    if set(mismatch.missing_keys) != expected or mismatch.unexpected_keys:
        raise RuntimeError(f'Full fine tune initialization mismatch: {mismatch}')
    instance.set_epoch(100)
    instance.requires_grad_(True)
    return instance


def optimizer_for(instance, cfg):
    groups = {}
    for name, parameter in instance.named_parameters():
        rate = (cfg.full_tune_backbone_lr if name.startswith('img_backbone.') else
                cfg.full_tune_adapter_lr if name.startswith('future_adapter.') else cfg.full_tune_lr)
        groups.setdefault(rate, []).append(parameter)
    return torch.optim.AdamW([dict(params=params, lr=rate, initial_lr=rate)
                              for rate, params in groups.items()], weight_decay=.01)


@torch.no_grad()
def verify_initial_parity(instance, cfg, strict, dataset):
    reference = build_student(cfg, strict)
    instance.eval()
    data = tree_to(next(iter(image_loader(dataset, 0))), 'cuda')
    imgs = data.pop('img')
    native = reference(imgs=imgs, metas=data)
    changed = instance(imgs=imgs, metas=data)
    for step in range(6):
        torch.testing.assert_close(changed['future_pred_points'][step],
                                   native['future_pred_points'][step], atol=1e-5, rtol=0)
        torch.testing.assert_close(changed['future_pred_logits'][step],
                                   native['future_pred_logits'][step], atol=1e-5, rtol=0)
        if changed['future_pred_points'][step].shape[1] != (720 + sum(instance.future_queries[:step + 1])) * 48:
            raise RuntimeError('Future-born query count differs at initialization')
    del reference
    torch.cuda.empty_cache()
    return True


def alignment_loss(instance, output, teacher, arm, generator):
    values = []
    for h in HORIZONS:
        state = output['future_query_states'][h - 1]
        target, valid = teacher_at_student(instance, teacher['states'][h], state['positions'], 4, 4.)
        if not valid.any():
            continue
        prediction = state['features'][:, valid].float()
        target = target[:, valid].float()
        if arm == 'teacher_shuffled' and int(valid.sum()) > 1:
            target = target[:, torch.randperm(int(valid.sum()), device='cuda', generator=generator)]
        values.append(F.smooth_l1_loss(F.layer_norm(prediction, (256,)),
                                       F.layer_norm(target, (256,))) +
                      (1 - F.cosine_similarity(prediction, target, dim=-1)).mean())
    return torch.stack(values).mean() if values else output['pred_traj'].sum() * 0.


@torch.no_grad()
def evaluate(instance, dataset, cfg, destination):
    instance.eval()
    rows = []
    for i, data in enumerate(image_loader(dataset, cfg.full_tune_workers)):
        data = tree_to(data, 'cuda')
        output = instance(imgs=data.pop('img'), metas=data)
        row = {}
        for h in HORIZONS:
            metric = _metric(cfg)
            add_metric(metric, instance, output['future_pred_points'][h - 1],
                       output['future_pred_logits'][h - 1], data, h)
            row[h] = metric_row(metric)
        rows.append(row)
        if (i + 1) % 128 == 0 or i + 1 == len(dataset):
            print(f'[native eval {destination.stem}] {i + 1}/{len(dataset)}', flush=True)
    save_rows(destination, rows)
    result = summarized_rows(rows, cfg)
    result['rows'] = str(destination)
    return result


def checkpoint(instance, optimizer, epoch, position, steps, generator, manifest):
    return dict(state_dict=instance.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, position=position, steps=steps, manifest=manifest,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                shuffle_rng=generator.get_state())


def train_arm(cfg, strict, weights, train_images, val_images, train_manifest,
              teacher_paths, root, report, arm, seed, protocol):
    label = f'{arm}_seed{seed}'
    arm_root = root / label
    arm_root.mkdir(exist_ok=True)
    seed_all(seed)
    instance = build_full_student(cfg, strict, weights)
    optimizer = optimizer_for(instance, cfg)
    objective = OPENOCC_LOSS.build(strict.loss).cuda()
    generator = torch.Generator(device='cuda').manual_seed(seed + 917)
    epoch, position, steps = 0, 0, 0
    latest = arm_root / 'latest.pth'
    report['arms'].setdefault(label, dict(epochs={}))
    # Recover an epoch evaluation interrupted after the epoch checkpoint was
    # committed. Restore the latest training state/RNG only after recovery.
    for epoch_file in sorted(arm_root.glob('epoch_*.pth')):
        number = epoch_file.stem.split('_')[-1]
        if number not in report['arms'][label]['epochs']:
            saved_epoch = torch.load(epoch_file, map_location='cpu', weights_only=False)
            instance.load_state_dict(saved_epoch['state_dict'], strict=True)
            recovered = evaluate(instance, val_images.dataset, cfg, arm_root / f'epoch_{number}.npz')
            report['arms'][label]['epochs'][number] = relative_to_baseline(recovered, report['baseline'])
            save_report(root, report, cfg)
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['manifest'] != protocol:
            raise RuntimeError('Full fine tune resume protocol differs')
        instance.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        epoch, position, steps = saved['epoch'], saved['position'], saved['steps']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        generator.set_state(saved['shuffle_rng'])
    total_steps = cfg.full_tune_epochs * len(train_images.dataset)
    torch.cuda.reset_peak_memory_stats()
    for ep in range(epoch, cfg.full_tune_epochs):
        instance.train().requires_grad_(True)
        frozen = [name for name, p in instance.named_parameters() if not p.requires_grad]
        if frozen:
            raise RuntimeError(f'Frozen Student parameters found: {frozen}')
        order = np.random.default_rng(seed + ep * 1009).permutation(len(train_images.dataset)).tolist()
        start = position if ep == epoch else 0
        running, began = [], time.monotonic()
        for offset, data in enumerate(image_loader(train_images.dataset, cfg.full_tune_workers, order[start:]), start):
            teacher = None
            sample_position = order[offset]
            if arm != 'task_only':
                teacher = torch.load(teacher_paths[sample_position], map_location='cpu', weights_only=False)
                if (teacher['index'], teacher['scene']) != (train_manifest['indices'][sample_position], train_manifest['scenes'][sample_position]):
                    raise RuntimeError('Training teacher cache sample mismatch')
            data = tree_to(data, 'cuda')
            factor = (min((steps + 1) / cfg.full_tune_warmup_steps, 1.) if steps < cfg.full_tune_warmup_steps else
                      .05 + .95 * .5 * (1 + math.cos(math.pi * (steps - cfg.full_tune_warmup_steps) /
                                                     max(total_steps - cfg.full_tune_warmup_steps, 1))))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * factor
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = instance(imgs=data.pop('img'), metas=data)
            task, _ = objective(output)
            align = (task.new_zeros(()) if teacher is None else alignment_loss(instance, output, teacher, arm, generator))
            value = task + cfg.full_tune_feature_weight * align
            if not torch.isfinite(value):
                raise FloatingPointError(f'Nonfinite loss {label} epoch={ep + 1} sample={offset}')
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            grad = torch.nn.utils.clip_grad_norm_(instance.parameters(), 5., error_if_nonfinite=True)
            if steps == 0:
                report['arms'][label]['gradient_audit'] = dict(
                    frozen_names=frozen,
                    no_gradient_names=[n for n, p in instance.named_parameters() if p.grad is None],
                    trainable_parameters=sum(p.numel() for p in instance.parameters()))
            optimizer.step()
            steps += 1
            running.append(float(value.detach()))
            if (offset + 1) % cfg.full_tune_log_every == 0 or offset + 1 == len(order):
                print(f'[{label}] epoch {ep + 1}/{cfg.full_tune_epochs} sample {offset + 1}/{len(order)} '
                      f'task={task.item():.4f} align={align.item():.4f} grad={float(grad):.3f} '
                      f'peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB elapsed={time.monotonic()-began:.0f}s', flush=True)
                report['arms'][label]['progress'] = dict(epoch=ep + 1, processed=offset + 1,
                                                        total=len(order), updates=steps)
                save_report(root, report, cfg)
            if (offset + 1) % 512 == 0:
                atomic_torch_save(checkpoint(instance, optimizer, ep, offset + 1, steps, generator, protocol), latest)
            del output, value, task, align, data, teacher
        epoch_file = arm_root / f'epoch_{ep + 1}.pth'
        saved = checkpoint(instance, optimizer, ep + 1, 0, steps, generator, protocol)
        atomic_torch_save(saved, epoch_file)
        atomic_torch_save(saved, latest)
        result = evaluate(instance, val_images.dataset, cfg, arm_root / f'epoch_{ep + 1}.npz')
        report['arms'][label]['epochs'][str(ep + 1)] = relative_to_baseline(result, report['baseline'])
        report['arms'][label]['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        save_report(root, report, cfg)
        position = 0
    report['arms'][label]['status'] = 'complete'
    save_report(root, report, cfg)
    del instance, optimizer, objective
    torch.cuda.empty_cache()


def main(args):
    cfg = Config.fromfile(args.config)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.smoke:
        cfg.full_tune_train_samples = cfg.full_tune_val_samples = 2
        cfg.full_tune_epochs = 1
        cfg.full_tune_seeds = [20260930]
        cfg.full_tune_arms = ['teacher_all']
        cfg.full_tune_workers = 0
    root = Path(args.output_dir or (cfg.full_tune_root + '_smoke2' if args.smoke else cfg.full_tune_root))
    root.mkdir(parents=True, exist_ok=True)
    # Hold the lock for the entire process; two resumes cannot write the same
    # optimizer/checkpoint while using the same GPU.
    run_lock = (root / 'train.lock').open('a')
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if file_sha256(cfg.student_checkpoint) != cfg.student_checkpoint_sha256:
        raise RuntimeError('Student checkpoint SHA256 mismatch')
    strict = Config.fromfile(cfg.student_model_config)
    datasets, manifests = {}, {}
    for split, count in [('train', cfg.full_tune_train_samples), ('val', cfg.full_tune_val_samples)]:
        images, manifest = loader_for(cfg, split, count, cfg.full_tune_workers, cfg.identity_sample_seed)
        if len(manifest['indices']) != count:
            raise RuntimeError(f'{split} contains fewer than requested {count} samples')
        datasets[split], manifests[split] = images, manifest
    if set(manifests['train']['scenes']) & set(manifests['val']['scenes']):
        raise RuntimeError('Train and validation scenes overlap')
    earlier = json.loads((Path(cfg.predictability_identity_root) / 'manifest.json').read_text())
    if earlier['checkpoint_sha256'] != cfg.student_checkpoint_sha256:
        raise RuntimeError('Earlier teacher-feature cache checkpoint differs')
    for split in ('train', 'val'):
        overlap = min(len(earlier[split]['indices']), len(manifests[split]['indices']))
        for key in ('indices', 'scenes'):
            if earlier[split][key][:overlap] != manifests[split][key][:overlap]:
                raise RuntimeError('Expanded dataset is not a prefix-compatible extension')
    protocol = dict(config=args.config, checkpoint_sha256=cfg.student_checkpoint_sha256,
                    train=manifests['train'], val=manifests['val'], epochs=cfg.full_tune_epochs,
                    seeds=cfg.full_tune_seeds, arms=cfg.full_tune_arms,
                    loss='native current/temporal/future occupancy + trajectory; optional feature distillation',
                    all_student_parameters_trainable=True, dtype='BF16 compute + FP32 loss/evaluation',
                    learning_rates=dict(backbone=cfg.full_tune_backbone_lr, existing=cfg.full_tune_lr,
                                        adapter=cfg.full_tune_adapter_lr), feature_weight=cfg.full_tune_feature_weight)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != protocol:
        raise RuntimeError('Existing full fine tune manifest differs')
    save_json(manifest_path, protocol)
    report_path = root / 'results.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(status='preparing', arms={}, manifest=str(manifest_path))
    save_report(root, report, cfg)
    teacher_paths = {}
    reference = build_student(cfg, strict)
    for split in ('train', 'val'):
        teacher_paths[split], native_teacher = prepare_teachers(reference, datasets[split], manifests[split], cfg, root, split)
        if native_teacher is not None:
            report['teacher'] = native_teacher
    save_report(root, report, cfg)
    del reference, datasets
    torch.cuda.empty_cache()
    if args.prepare_only:
        return
    student_cfg = historical_config(cfg)
    train_images, train_manifest = loader_for(student_cfg, 'train', cfg.full_tune_train_samples, cfg.full_tune_workers, cfg.identity_sample_seed)
    val_images, val_manifest = loader_for(student_cfg, 'val', cfg.full_tune_val_samples, cfg.full_tune_workers, cfg.identity_sample_seed)
    if train_manifest != manifests['train'] or val_manifest != manifests['val']:
        raise RuntimeError('History-only dataset selection differs from teacher dataset')
    weights = torch.load(cfg.student_checkpoint, map_location='cpu', weights_only=False)['state_dict']
    if 'baseline' not in report:
        initial = build_full_student(cfg, strict, weights)
        report['identity_initialization_parity'] = verify_initial_parity(initial, cfg, strict, val_images.dataset)
        report['baseline'] = evaluate(initial, val_images.dataset, cfg, root / 'baseline.npz')
        save_report(root, report, cfg)
        del initial
        torch.cuda.empty_cache()
    report['teacher'] = relative_to_baseline(report['teacher'], report['baseline'])
    report['status'] = 'training'
    save_report(root, report, cfg)
    for arm in cfg.full_tune_arms:
        for seed in cfg.full_tune_seeds:
            label = f'{arm}_seed{seed}'
            if report['arms'].get(label, {}).get('status') == 'complete':
                continue
            train_arm(cfg, strict, weights, train_images, val_images, train_manifest,
                      teacher_paths['train'], root, report, arm, seed, protocol)
    report['status'] = 'complete'
    save_report(root, report, cfg)
    print(f'[complete full fine tune] {report_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/opus/future_predictable_full_finetune.py')
    parser.add_argument('--output-dir')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--background', action='store_true')
    options = parser.parse_args()
    if options.background:
        config = Config.fromfile(options.config)
        destination = Path(options.output_dir or (config.full_tune_root + '_smoke2' if options.smoke else config.full_tune_root))
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / 'train.lock').open('a') as run_lock:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [sys.executable, str(Path(__file__).resolve()), '--config', options.config,
                   '--output-dir', str(destination)]
        if options.smoke:
            command.append('--smoke')
        if options.prepare_only:
            command.append('--prepare-only')
        environment = dict(os.environ, MPLCONFIGDIR='/tmp/matplotlib-cache')
        with (destination / 'run.log').open('a', buffering=1) as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                       env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        save_json(destination / 'job.json', dict(pid=process.pid, command=command,
                                                started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())))
        print(f'[background job] pid={process.pid} log={destination / "run.log"}')
    else:
        try:
            main(options)
        except BlockingIOError:
            raise
        except Exception as error:
            config = Config.fromfile(options.config)
            destination = Path(options.output_dir or (config.full_tune_root + '_smoke2' if options.smoke else config.full_tune_root))
            report_path = destination / 'results.json'
            if report_path.exists():
                failed = json.loads(report_path.read_text())
                failed.update(status='failed', error=f'{type(error).__name__}: {error}')
                save_json(report_path, failed)
            raise
