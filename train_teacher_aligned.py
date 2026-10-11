#!/usr/bin/env python3
"""Independent 16-epoch trainer for six-horizon teacher-aligned SparseWorld."""

import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch
from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.runner import set_random_seed
from mmengine.utils import symlink
from mmseg.models import build_segmentor
from timm.scheduler import CosineLRScheduler


FROZEN_PREFIXES = (
    'img_backbone.', 'img_neck.', 'lifter.', 'encoder.', 'head.',
    'plan_head.', 'points_scale.')
NEW_PREFIXES = (
    'future_plan_head.', 'future_feature_mlp.', 'student_projection.')


def _move_batch(data, device):
    for key, value in list(data.items()):
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device, non_blocking=True)
    return data


def _nonfinite_gradients(model):
    return [name for name, parameter in model.named_parameters()
            if parameter.grad is not None and
            not torch.isfinite(parameter.grad).all()]


def _load_student_initial(model, filename, expected_sha256=None):
    from model.multi_horizon_teacher import file_sha256
    if not osp.isfile(filename):
        raise FileNotFoundError(filename)
    actual = file_sha256(filename)
    if expected_sha256 and actual != expected_sha256.lower():
        raise RuntimeError(
            f'student checkpoint SHA256 mismatch: expected '
            f'{expected_sha256}, got {actual}')
    checkpoint = torch.load(filename, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    invalid_missing = [name for name in missing
                       if not name.startswith(NEW_PREFIXES)]
    if invalid_missing or result.unexpected_keys:
        raise RuntimeError(
            'SparseWorld initialization mismatch: '
            f'missing={invalid_missing}, unexpected={result.unexpected_keys}')
    expected_new = {name for name, _ in model.named_parameters()
                    if name.startswith(NEW_PREFIXES)}
    if set(missing) != expected_new:
        raise RuntimeError(
            'only newly introduced alignment parameters may be absent; '
            f'missing={missing}, expected={sorted(expected_new)}')
    return actual, missing


def _audit_freeze(model):
    trainable, frozen = [], []
    for name, parameter in model.named_parameters():
        (trainable if parameter.requires_grad else frozen).append(name)
    leaked = [name for name in trainable if name.startswith(FROZEN_PREFIXES)]
    absent = [prefix for prefix in NEW_PREFIXES
              if not any(name.startswith(prefix) for name in trainable)]
    if leaked or absent:
        raise RuntimeError(
            f'invalid freeze contract: leaked={leaked}, absent={absent}')
    return dict(
        trainable_names=trainable, frozen_names=frozen,
        trainable_parameters=sum(parameter.numel() for parameter in
                                 model.parameters()
                                 if parameter.requires_grad),
        frozen_parameters=sum(parameter.numel() for parameter in
                              model.parameters()
                              if not parameter.requires_grad))


def _build_optimizer(model, cfg):
    base_lr = float(cfg.optimizer.optimizer.lr)
    future_lr = float(cfg.optimizer.existing_future_lr)
    weight_decay = float(cfg.optimizer.optimizer.get('weight_decay', 0.))
    new_parameters, existing_parameters, projection_parameters = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('student_projection.'):
            projection_parameters.append(parameter)
        elif name.startswith('future_feature_mlp.'):
            new_parameters.append(parameter)
        else:
            existing_parameters.append(parameter)
    if not new_parameters or not existing_parameters or not projection_parameters:
        raise RuntimeError('future optimizer groups are incomplete')
    return torch.optim.AdamW([
        dict(params=new_parameters, lr=base_lr,
             weight_decay=weight_decay, group_name='new_feature_mlp'),
        dict(params=projection_parameters, lr=base_lr,
             weight_decay=0., group_name='alignment_projection'),
        dict(params=existing_parameters, lr=future_lr,
             weight_decay=weight_decay, group_name='existing_future'),
    ], lr=base_lr, weight_decay=weight_decay)


def _save_checkpoint(filename, student, optimizer, scheduler, scaler,
                     epoch, global_iter, manifest, overflows):
    checkpoint = dict(
        state_dict=student.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
        epoch=epoch, global_iter=global_iter, model_epoch=max(epoch - 1, 0),
        amp_overflow_count=overflows, alignment_manifest=manifest)
    torch.save(checkpoint, filename)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('teacher-aligned training requires CUDA')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    set_random_seed(args.seed)
    cfg = Config.fromfile(args.py_config)
    os.makedirs(args.work_dir, exist_ok=True)
    cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
    logger = MMLogger(
        'sparseworld_teacher_alignment',
        log_file=osp.join(args.work_dir, time.strftime('%Y%m%d_%H%M%S.log')))
    MMLogger._instance_dict['sparseworld_teacher_alignment'] = logger

    import model  # noqa: F401
    import model.segmentor.sparseworld_teacher_aligned_segmentor  # noqa: F401
    import loss.sparseworld_teacher_alignment_loss  # noqa: F401
    from dataset import get_dataloader
    from eval_teacher_aligned import evaluate_model
    from loss import OPENOCC_LOSS
    from model.multi_horizon_teacher import MultiHorizonFrozenOPUSV1Teacher

    student = build_segmentor(cfg.model)
    student.init_weights()
    student_sha, missing = _load_student_initial(
        student, cfg.student_init_checkpoint,
        cfg.get('student_init_checkpoint_sha256'))
    student.initialize_future_from_pretrained()
    student.set_epoch(0)
    freeze = _audit_freeze(student)
    student.to(device)
    logger.info(
        'SparseWorld init loaded; only new keys missing: %s', missing)
    logger.info('trainable=%d frozen=%d', freeze['trainable_parameters'],
                freeze['frozen_parameters'])

    teacher = MultiHorizonFrozenOPUSV1Teacher(
        config=cfg.teacher_config, checkpoint=cfg.teacher_checkpoint,
        device=device, history_frames=cfg.teacher_history_frames,
        num_horizons=cfg.num_future, num_cams=cfg.num_cams,
        prefer_ema=cfg.get('teacher_prefer_ema', True),
        expected_sha256=cfg.get('teacher_checkpoint_sha256'),
        amp=cfg.get('teacher_amp', True))
    logger.info('teacher manifest: %s', teacher.manifest())

    train_loader, val_loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False)
    optimizer = _build_optimizer(student, cfg)
    loss_function = OPENOCC_LOSS.build(cfg.loss).to(device)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=len(train_loader) * cfg.max_epochs,
        lr_min=cfg.optimizer.optimizer.lr * cfg.get('min_lr_ratio', .05),
        warmup_t=cfg.get('warmup_iters', 500), warmup_lr_init=1e-6,
        t_in_epochs=False)
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg.amp), init_scale=cfg.get('amp_loss_scale', 64.),
        growth_interval=cfg.get('amp_growth_interval', 2000))
    accumulation = int(args.gradient_accumulation or
                       cfg.get('gradient_accumulation', 1))
    epoch, global_iter, amp_overflow_count = 0, 0, 0

    manifest = dict(
        contract_version=int(cfg.alignment_contract_version),
        student_checkpoint=cfg.student_init_checkpoint,
        student_checkpoint_sha256=student_sha,
        teacher=teacher.manifest(), max_epochs=int(cfg.max_epochs),
        coordinate_system='per_horizon_native_future_ego',
        future_born_queries=list(student.future_queries),
        trainable_names=freeze['trainable_names'],
        frozen_prefixes=list(FROZEN_PREFIXES),
        batch_size=int(cfg.train_loader.batch_size),
        gradient_accumulation=accumulation)
    with open(osp.join(args.work_dir, 'alignment_manifest.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    resume = args.resume_from
    if not resume and osp.isfile(osp.join(args.work_dir, 'latest.pth')):
        resume = osp.join(args.work_dir, 'latest.pth')
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        stored = checkpoint.get('alignment_manifest', {})
        if stored.get('contract_version') != manifest['contract_version']:
            raise RuntimeError('checkpoint alignment contract is incompatible')
        if stored.get('teacher', {}).get('checkpoint_sha256') != \
                manifest['teacher']['checkpoint_sha256']:
            raise RuntimeError('checkpoint teacher identity is incompatible')
        student.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        epoch = int(checkpoint['epoch'])
        global_iter = int(checkpoint['global_iter'])
        amp_overflow_count = int(checkpoint.get('amp_overflow_count', 0))
        logger.info('resumed from %s at epoch %d', resume, epoch)

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    consecutive_overflows = 0
    try:
        while epoch < int(cfg.max_epochs):
            student.train()
            student.set_epoch(epoch)
            _audit_freeze(student)
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)
            started = time.perf_counter()
            running = []
            last_grad_norm = 0.
            for iteration, data in enumerate(train_loader):
                data = _move_batch(data, device)
                imgs = data.pop('img')
                teacher_states = teacher.extract_all(imgs, data)
                with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                    output = student(imgs=imgs, metas=data)
                    loss_inputs = dict(output)
                    loss_inputs.update(
                        metas=data, teacher_states=teacher_states,
                        future_query_states=output['future_query_states'],
                        global_iter=global_iter)
                    loss, details = loss_function(loss_inputs)
                    scaled_loss = loss / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f'non-finite loss at iteration {global_iter}')
                scaler.scale(scaled_loss).backward()
                smoke_last = bool(args.max_train_batches and
                                  iteration + 1 >= args.max_train_batches)
                do_step = ((iteration + 1) % accumulation == 0 or
                           iteration + 1 == len(train_loader) or smoke_last)
                step_state = 'pending'
                if do_step:
                    scaler.unscale_(optimizer)
                    bad = _nonfinite_gradients(student)
                    if bad and bool(cfg.amp) and cfg.get(
                            'recover_amp_overflow', True):
                        before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        amp_overflow_count += 1
                        consecutive_overflows += 1
                        step_state = 'overflow-skip'
                        logger.warning(
                            'AMP skipped iter %d, scale %.1f -> %.1f: %s',
                            global_iter, before, scaler.get_scale(), bad[:20])
                        if consecutive_overflows > int(cfg.get(
                                'max_consecutive_amp_overflows', 8)):
                            raise FloatingPointError(
                                'too many consecutive AMP overflows')
                    else:
                        if bad:
                            raise FloatingPointError(
                                f'non-finite gradients: {bad[:20]}')
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in student.parameters()
                             if parameter.requires_grad],
                            cfg.grad_max_norm, error_if_nonfinite=True)
                        last_grad_norm = float(grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        consecutive_overflows = 0
                        step_state = 'ok'
                scheduler.step_update(global_iter)
                global_iter += 1
                running.append(float(loss.detach()))
                del teacher_states, output, loss_inputs
                peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
                if peak_gb > float(cfg.max_cuda_memory_gb):
                    raise RuntimeError(
                        f'peak CUDA memory {peak_gb:.2f} GiB exceeds '
                        f'{cfg.max_cuda_memory_gb:.2f} GiB')
                if iteration % int(cfg.print_freq) == 0:
                    compact = ', '.join(
                        f'{key}={float(value):.4f}'
                        for key, value in details.items()
                        if torch.is_tensor(value) and value.numel() == 1 and
                        torch.isfinite(value))
                    logger.info(
                        '[TRAIN] epoch=%d iter=%d/%d loss=%.4f avg=%.4f '
                        'grad=%.3f step=%s scale=%.1f peak=%.2fGiB %s',
                        epoch + 1, iteration, len(train_loader), float(loss),
                        np.mean(running), last_grad_norm, step_state,
                        scaler.get_scale(), peak_gb, compact)
                if smoke_last:
                    break

            epoch += 1
            filename = osp.abspath(osp.join(
                args.work_dir, f'epoch_{epoch}.pth'))
            _save_checkpoint(
                filename, student, optimizer, scheduler, scaler, epoch,
                global_iter, manifest, amp_overflow_count)
            symlink(filename, osp.join(args.work_dir, 'latest.pth'))
            logger.info('epoch %d completed in %.1fs', epoch,
                        time.perf_counter() - started)
            if epoch % int(cfg.get('eval_every_epochs', 1)) == 0:
                result = evaluate_model(
                    student, val_loader, cfg, device,
                    max_batches=args.max_val_batches)
                logger.info('[EVAL current/1s/2s/3s] %s', result)
            if args.max_train_batches:
                logger.info('CUDA smoke run completed')
                break
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        raise RuntimeError(
            f'CUDA OOM (observed peak before failure: {peak:.2f} GiB). '
            'The supported 24 GiB recipe is batch_size=1 with sequential '
            'teachers and frozen perception.') from exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', default=(
        'config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py'))
    parser.add_argument('--work-dir', default=(
        'out/sparseworld_teacher_alignment/16e'))
    parser.add_argument('--resume-from', default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-accumulation', type=int, default=None)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    main(parser.parse_args())
