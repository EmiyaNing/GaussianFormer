#!/usr/bin/env python3
"""Independent single-GPU trainer for SparseWorld-FFA."""

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


def _move_batch(data, device):
    for key, value in list(data.items()):
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device, non_blocking=True)
    return data


def _find_nonfinite_gradients(model):
    """Return trainable parameter names whose gradients contain NaN/Inf."""
    bad = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            bad.append(name)
    return bad


def _load_student_initial(model, filename, expected_sha256):
    from model.ffa_teacher import file_sha256
    if not osp.isfile(filename):
        raise FileNotFoundError(filename)
    actual_sha256 = file_sha256(filename)
    if expected_sha256 and actual_sha256 != expected_sha256.lower():
        raise RuntimeError(
            f'student checkpoint SHA256 mismatch: {actual_sha256}')
    checkpoint = torch.load(filename, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys
               if not key.startswith('ffa.')]
    if unexpected or missing:
        raise RuntimeError(
            'student initialization mismatch: '
            f'missing={missing}, unexpected={unexpected}')
    return actual_sha256


def _build_optimizer(model, cfg):
    """Build only the trainable FFA groups without frozen-param spam."""
    options = cfg.optimizer.optimizer
    base_lr = float(options.lr)
    weight_decay = float(options.get('weight_decay', 0.0))
    groups = dict(ffa=[], projection=[], semantic=[])
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == 'ffa.student_projection.weight':
            groups['projection'].append(parameter)
        elif name.startswith('semantic.'):
            groups['semantic'].append(parameter)
        else:
            groups['ffa'].append(parameter)
    if not groups['ffa'] or not groups['semantic']:
        raise RuntimeError('expected trainable FFA and semantic parameters')
    return torch.optim.AdamW([
        dict(params=groups['ffa'], lr=base_lr,
             weight_decay=weight_decay),
        dict(params=groups['projection'], lr=base_lr, weight_decay=0.0),
        dict(params=groups['semantic'], lr=base_lr * 0.1,
             weight_decay=weight_decay),
    ], lr=base_lr, weight_decay=weight_decay)


@torch.no_grad()
def evaluate(model, loader, cfg, device, max_batches=None):
    from misc.sparseworld_eval import SparseWorldHorizonMetric
    from model.sparse_latent_dynamics import transform_points
    metric = SparseWorldHorizonMetric(
        num_classes=18, use_lidar_mask=False,
        use_image_mask=cfg.occ3d_eval_mask == 'camera',
        free_label=17, report_occ_iou=True)
    model.eval()
    target = int(cfg.target_horizon_index) - 1
    for batch_index, data in enumerate(loader):
        data = _move_batch(data, device)
        imgs = data.pop('img')
        with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
            output = model(imgs=imgs, metas=data)
        future_to_current = data['future_ego_to_current'][:, target]
        current_to_future = torch.linalg.inv(future_to_current.float())
        state = output['ffa_output']['prediction']
        native_points = transform_points(
            state['points_metric'].float(), current_to_future).flatten(1, 2)
        native_logits = output['ffa_future_logits'].float()
        born_points = output['ffa_output'].get('future_born_points_native')
        born_logits = output['ffa_output'].get('future_born_logits')
        if born_points is not None:
            native_points = torch.cat(
                [native_points, born_points.float()], dim=1)
            native_logits = torch.cat(
                [native_logits, born_logits.float()], dim=1)
        prediction = model.head.rasterizer(
            native_points, native_logits,
            model.num_refines).view(
                native_points.shape[0], *model.head.rasterizer.grid_shape
            ).cpu().numpy()
        labels = data['future_occ_labels'][:, target].cpu().numpy()
        camera = data['future_occ_cam_masks'][:, target].cpu().numpy()
        lidar = data['future_occ_lidar_masks'][:, target].cpu().numpy()
        for item in range(prediction.shape[0]):
            metric.add_batch(
                prediction[item], labels[item], lidar[item], camera[item])
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return metric.results()


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('SparseWorld-FFA training requires a CUDA device')
    if torch.cuda.device_count() != 1:
        print('train_ffa.py uses one CUDA device; select it with CUDA_VISIBLE_DEVICES.')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    set_random_seed(args.seed)

    cfg = Config.fromfile(args.py_config)
    os.makedirs(args.work_dir, exist_ok=True)
    cfg.work_dir = args.work_dir
    cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
    logger = MMLogger(
        'sparseworld_ffa',
        log_file=osp.join(args.work_dir, time.strftime('%Y%m%d_%H%M%S.log')))
    MMLogger._instance_dict['sparseworld_ffa'] = logger

    import model  # noqa: F401
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS
    from model.ffa_teacher import FrozenOPUSV1Teacher

    student = build_segmentor(cfg.model)
    student.init_weights()
    student_sha256 = _load_student_initial(
        student, cfg.student_init_checkpoint,
        cfg.get('student_init_checkpoint_sha256'))
    student.to(device)
    student.set_epoch(0)
    trainable = sum(parameter.numel() for parameter in student.parameters()
                    if parameter.requires_grad)
    logger.info('trainable parameters: %d', trainable)

    teacher = FrozenOPUSV1Teacher(
        config=cfg.teacher_config, checkpoint=cfg.teacher_checkpoint,
        expected_sha256=cfg.get('teacher_checkpoint_sha256'),
        prefer_ema=cfg.get('teacher_prefer_ema', True),
        target_horizon_index=cfg.target_horizon_index,
        num_student_frames=cfg.num_student_frames,
        num_cams=cfg.num_cams, amp=cfg.get('teacher_amp', False),
        device=device)
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
        enabled=bool(cfg.amp), init_scale=cfg.get('amp_loss_scale', 128.),
        growth_interval=cfg.get('amp_growth_interval', 2000))
    accumulation = int(args.gradient_accumulation or
                       cfg.get('gradient_accumulation', 1))
    epoch, global_iter = 0, 0
    amp_overflow_count = 0
    consecutive_amp_overflows = 0

    resume = args.resume_from
    if not resume and osp.isfile(osp.join(args.work_dir, 'latest.pth')):
        resume = osp.join(args.work_dir, 'latest.pth')
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        expected_contract = int(cfg.get('ffa_contract_version', 1))
        checkpoint_contract = int(
            checkpoint.get('ffa_manifest', {}).get('contract_version', 1))
        if checkpoint_contract != expected_contract:
            raise RuntimeError(
                f'cannot resume FFA contract v{checkpoint_contract} with '
                f'v{expected_contract}; teacher coordinates, future-born '
                'queries, and loss weights changed. Start a new work-dir.')
        student.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        epoch = int(checkpoint['epoch'])
        global_iter = int(checkpoint['global_iter'])
        amp_overflow_count = int(checkpoint.get('amp_overflow_count', 0))
        consecutive_amp_overflows = int(
            checkpoint.get('consecutive_amp_overflows', 0))
        logger.info('resumed from %s at epoch %d', resume, epoch)

    manifest = dict(
        contract_version=int(cfg.get('ffa_contract_version', 1)),
        target_horizon_index=int(cfg.target_horizon_index),
        target_horizon_seconds=float(cfg.target_horizon_seconds),
        student_checkpoint=cfg.student_init_checkpoint,
        student_checkpoint_sha256=student_sha256,
        teacher=teacher.manifest(), max_epochs=int(cfg.max_epochs),
        persistent_queries=720,
        future_born_queries=int(sum(student.future_queries)),
        batch_size=int(cfg.train_loader.batch_size),
        gradient_accumulation=accumulation)
    with open(osp.join(args.work_dir, 'ffa_manifest.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        while epoch < cfg.max_epochs:
            student.train()
            student.set_epoch(epoch)
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            running = []
            started = time.perf_counter()
            # Keep the most recent completed optimizer-step norm. Logging is
            # based on micro-batches, so a printed iteration is not
            # necessarily an optimizer-step boundary under accumulation.
            last_grad_norm = 0.0
            for iteration, data in enumerate(train_loader):
                data = _move_batch(data, device)
                imgs = data.pop('img')
                # Teacher and student execute sequentially.  Only the final
                # detached 720-query state survives the teacher call.
                teacher_state = teacher.extract(imgs, data)
                with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                    output = student(
                        imgs=imgs, metas=data, teacher_state=teacher_state)
                    loss, details = loss_function(dict(
                        ffa_output=output['ffa_output'], metas=data,
                        global_iter=global_iter))
                    scaled_loss = loss / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f'non-finite loss at iteration {global_iter}')
                scaler.scale(scaled_loss).backward()
                smoke_last = bool(
                    args.max_train_batches and
                    iteration + 1 >= args.max_train_batches)
                do_step = ((iteration + 1) % accumulation == 0 or
                           iteration + 1 == len(train_loader) or smoke_last)
                optimizer_step_succeeded = False
                if do_step:
                    scaler.unscale_(optimizer)
                    bad_gradients = _find_nonfinite_gradients(student)
                    recover_overflow = bool(
                        cfg.get('recover_amp_overflow', True))
                    if bad_gradients and bool(cfg.amp) and recover_overflow:
                        scale_before = scaler.get_scale()
                        # unscale_ has already recorded found_inf. GradScaler
                        # therefore skips this update and lowers its scale.
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scale_after = scaler.get_scale()
                        amp_overflow_count += 1
                        consecutive_amp_overflows += 1
                        last_grad_norm = float('inf')
                        logger.warning(
                            '[AMP] skipped optimizer step at global_iter=%d; '
                            'scale %.1f -> %.1f; consecutive=%d, total=%d; '
                            'non-finite gradients: %s',
                            global_iter, scale_before, scale_after,
                            consecutive_amp_overflows, amp_overflow_count,
                            ', '.join(bad_gradients[:20]))
                        max_overflows = int(cfg.get(
                            'max_consecutive_amp_overflows', 8))
                        if consecutive_amp_overflows > max_overflows:
                            raise FloatingPointError(
                                'AMP overflow persisted for '
                                f'{consecutive_amp_overflows} consecutive '
                                f'optimizer steps at global_iter={global_iter}: '
                                + ', '.join(bad_gradients[:20]))
                    else:
                        if bad_gradients:
                            raise FloatingPointError(
                                f'non-finite gradients at global_iter='
                                f'{global_iter}: '
                                + ', '.join(bad_gradients[:20]))
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            student.parameters(), cfg.grad_max_norm,
                            error_if_nonfinite=True)
                        last_grad_norm = float(grad_norm)
                        scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step_succeeded = (
                            scaler.get_scale() >= scale_before)
                        consecutive_amp_overflows = 0
                scheduler.step_update(global_iter)
                global_iter += 1
                running.append(float(loss.detach()))
                del teacher_state, output

                peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
                if peak_gb > float(cfg.max_cuda_memory_gb):
                    raise RuntimeError(
                        f'peak CUDA memory {peak_gb:.2f} GiB exceeds configured '
                        f'limit {cfg.max_cuda_memory_gb:.2f} GiB')
                if iteration % cfg.print_freq == 0:
                    detail_text = ', '.join(
                        f'{key}={float(value):.4f}' for key, value in details.items()
                        if torch.is_tensor(value) and value.numel() == 1 and
                        torch.isfinite(value))
                    logger.info(
                        '[TRAIN] epoch=%d iter=%d/%d loss=%.4f avg=%.4f '
                        'grad=%.3f step=%s amp_scale=%.1f overflows=%d '
                        'peak=%.2fGiB %s', epoch + 1, iteration,
                        len(train_loader), float(loss), np.mean(running),
                        last_grad_norm,
                        'ok' if optimizer_step_succeeded else 'pending/skip',
                        scaler.get_scale(), amp_overflow_count, peak_gb,
                        detail_text)
                if args.max_train_batches and iteration + 1 >= args.max_train_batches:
                    break

            epoch += 1
            checkpoint = dict(
                state_dict=student.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                epoch=epoch, global_iter=global_iter,
                model_epoch=epoch - 1,
                amp_overflow_count=amp_overflow_count,
                consecutive_amp_overflows=consecutive_amp_overflows,
                ffa_manifest=manifest)
            filename = osp.abspath(osp.join(
                args.work_dir, f'epoch_{epoch}.pth'))
            torch.save(checkpoint, filename)
            symlink(filename, osp.join(args.work_dir, 'latest.pth'))
            logger.info('epoch %d finished in %.1fs', epoch,
                        time.perf_counter() - started)

            if epoch % cfg.get('eval_every_epochs', 1) == 0:
                metrics = evaluate(
                    student, val_loader, cfg, device,
                    max_batches=args.max_val_batches)
                logger.info('[EVAL h=%d] %s', cfg.target_horizon_index, metrics)
            if args.max_train_batches:
                logger.info('max-train-batches smoke run completed')
                break
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        raise RuntimeError(
            f'SparseWorld-FFA exhausted CUDA memory (peak before failure '
            f'{peak:.2f} GiB). Keep batch_size=1 and increase gradient '
            'accumulation; do not enable student perception gradients.') from exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', default=(
        'config/opus/sparseworld_ffa_r50_704x256_5f_to_6f.py'))
    parser.add_argument('--work-dir', default='out/sparseworld_ffa/h6_16e')
    parser.add_argument('--resume-from', default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-accumulation', type=int, default=None)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    main(parser.parse_args())
