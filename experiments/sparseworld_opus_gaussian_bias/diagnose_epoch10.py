"""Read-only epoch-10 rendering, visibility and loss-gradient diagnostics.

All alternative predictions are diagnostic probes on fixed samples. They are
not selected checkpoints, retrained models, or tuned evaluation thresholds.
The active training files and configuration are never modified.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS
from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from .diagnose_first_epoch import quantiles, summarize
from .metrics import CLASS_NAMES
from .train import atomic_json, digest, move_batch


def add_histogram(histograms, key, prediction, target, mask):
    gt = target.long().reshape(-1)
    pred = prediction.long().reshape(-1)
    valid = (gt >= 0) & (gt < 18)
    if mask is not None:
        valid &= mask.reshape(-1).bool()
    hist = torch.bincount(gt[valid] * 18 + pred[valid], minlength=324)
    histograms.setdefault(key, np.zeros((18, 18), dtype=np.int64))
    histograms[key] += hist.reshape(18, 18).cpu().numpy()


def gradient_relation(left, right):
    left, right = left.reshape(-1), right.reshape(-1)
    return dict(set_norm=float(left.norm()), voxel_norm=float(right.norm()),
                voxel_to_set_norm_ratio=float(right.norm() / left.norm().clamp_min(1e-20)),
                cosine=float((left * right).sum() /
                             (left.norm() * right.norm()).clamp_min(1e-20)))


def gradient_probe(model, cfg, points, classes, gaussian, xyz, metas, horizon):
    """Compare directly supervised set and voxel terms at fixed outputs.

    Gradients are measured on raw class logits and world-coordinate centers,
    not on encoder/backbone parameters. Opacity, covariance and features stay
    fixed. Current uses weight 1; the supervised future voxel terms use .5.
    """
    loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
    target = metas['occ_label'] if horizon == 0 else metas['future_occ_labels'][:, horizon - 1]
    cam = metas['occ_cam_mask'] if horizon == 0 else metas['future_occ_cam_masks'][:, horizon - 1]
    mask = None
    if horizon:
        world = _opus_decode_points(points, model.pc_range)
        ego2lidar = metas['ego2lidar'].to(world)
        lidar = world @ ego2lidar[:, None, :3, :3].transpose(-1, -2)
        lidar += ego2lidar[:, None, None, :3, 3]
        trajectory = metas['temporal_trajs'][:, horizon - 1].to(world)
        lidar += torch.cat((-trajectory, trajectory.new_zeros(1, 1)), dim=-1)[:, None, None]
        lidar2ego = torch.linalg.inv(ego2lidar)
        shifted = lidar @ lidar2ego[:, None, :3, :3].transpose(-1, -2)
        shifted += lidar2ego[:, None, None, :3, 3]
        mask = _opus_encode_points(shifted, model.pc_range)[0, ..., 0] >= 0
    with torch.enable_grad():
        raw = classes.detach().clone().requires_grad_(True)
        world = _opus_decode_points(points, model.pc_range).detach().clone().requires_grad_(True)
        targets = loss_fn.strict._dense_points(target[0], world)
        cls_loss, pts_loss = loss_fn.strict._set_loss(
            world[0].flatten(0, 1), raw[0].flatten(0, 1), *targets,
            prediction_mask=mask)
        set_grad = torch.autograd.grad(cls_loss + pts_loss, (raw, world))
        lower = world.new_tensor(model.pc_range[:3])
        extent = world.new_tensor(model.pc_range[3:]) - lower
        means = torch.minimum(torch.maximum(world, lower + extent * 1e-4),
                              lower + extent * (1 - 1e-4))
        altered = gaussian._replace(means=means, semantics=F.softplus(raw))
        rendered = model.gaussian_head.render_state(altered, xyz)
        weight = 1. if horizon == 0 else .5
        voxel_loss = weight * loss_fn.occupancy.loss_voxel(
            [rendered], xyz, target.flatten(1), cam)
        voxel_grad = torch.autograd.grad(voxel_loss, (raw, world))
    return dict(horizon=horizon, seconds=horizon * .5,
                scope='Raw output class logits and world centers; fixed features/opacity/covariance.',
                set_classification_loss=float(cls_loss.detach()),
                set_point_loss=float(pts_loss.detach()),
                weighted_voxel_loss=float(voxel_loss.detach()),
                voxel_weight=weight,
                semantic=gradient_relation(set_grad[0], voxel_grad[0]),
                geometry=gradient_relation(set_grad[1], voxel_grad[1]))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--gradient-samples', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1024 or not 0 <= args.gradient_samples <= args.samples:
        raise ValueError('invalid diagnostic sample count')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    MMLogger.get_current_instance().setLevel(logging.ERROR)
    output = dict(epoch=args.epoch, samples=args.samples, class_names=CLASS_NAMES,
                  note='Fixed-sample diagnostics; variants are not formal full-set evaluation results.',
                  quantile_probabilities=[0., .1, .5, .9, 1.], models={})
    for name in ('sparseworld_strict', 'sparseworld_opus_gaussian_bias'):
        cfg = Config.fromfile(f'config/sparseworld_opus_gaussian_bias/{name}_4096_1024_15e.py')
        model = build_segmentor(cfg.model)
        model.init_weights()
        model.set_epoch(args.epoch - 1)
        checkpoint = Path(f'out/sparseworld_opus_gaussian_bias_4096_1024_15e/{name}/epoch_{args.epoch:02d}.pth')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'], strict=True)
        model.cuda().eval()
        model.pts_bbox_head.ind_stamps_all = saved['query_stamps'].cuda()
        model.pts_bbox_head.reset_mask()
        del saved
        captured = []
        if name == 'sparseworld_opus_gaussian_bias':
            original_render = model.gaussian_head.render_state

            def record_render(gaussian, xyz):
                logits = original_render(gaussian, xyz)
                captured.append((gaussian, logits))
                return logits

            model.gaussian_head.render_state = record_render
        dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
        indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=int).tolist()
        histograms, records, gradients = {}, [], []
        for sample_number, index in enumerate(indices):
            captured.clear()
            images, metas = move_batch(custom_collate_fn_temporal([dataset[index]]), 'cuda')
            result = model(imgs=images, metas=metas)
            current = model.pts_bbox_head.ind_stamps_all == 0
            points_list = [result['all_refine_pts'][-1][:, current]] + result['forecast_points_list']
            classes_list = [result['all_cls_scores'][-1][:, current]] + result['forecast_semantics_list']
            features_list = [result['query_feat'][:, current]] + result['forecast_features_list']
            grids = [result['final_occ_grid']] + result['future_occ_grids']
            xyz = metas['occ_xyz'].float().flatten(1, 3)
            for horizon, (points, classes, features, default) in enumerate(
                    zip(points_list, classes_list, features_list, grids)):
                target = metas['occ_label'] if horizon == 0 else metas['future_occ_labels'][:, horizon - 1]
                cam = metas['occ_cam_mask'] if horizon == 0 else metas['future_occ_cam_masks'][:, horizon - 1]
                inside = torch.isfinite(points).all(-1) & ((points >= 0) & (points < 1)).all(-1)
                modes = dict(default=default)
                record = dict(index=index, horizon=horizon, seconds=horizon * .5,
                              queries=points.shape[1], children=points.shape[2],
                              in_range_percent=100 * float(inside.float().mean()),
                              feature_norm_quantiles=quantiles(features.norm(dim=-1)),
                              point_score_quantiles=quantiles(classes.sigmoid().amax(-1)[inside]),
                              camera_visible_percent=100 * float(cam.float().mean()))
                if name == 'sparseworld_opus_gaussian_bias':
                    gaussian, logits = captured[horizon]
                    modes['point_rasterizer'] = model.pts_bbox_head.get_occ(
                        dict(cls_scores=classes, refine_pts=points))
                    for delta in (-2., -1., 1., 2.):
                        altered_logits = logits.clone()
                        altered_logits[:, 17] -= delta
                        modes[f'empty_bias_delta_{delta:+g}'] = altered_logits.argmax(1).view_as(default)
                    altered = gaussian._replace(scales=torch.full_like(gaussian.scales, .2))
                    modes['fixed_scale_0_2'] = original_render(altered, xyz).argmax(1).view_as(default)
                    altered = gaussian._replace(opacities=gaussian.opacities * .5)
                    modes['opacity_times_0_5'] = original_render(altered, xyz).argmax(1).view_as(default)
                    gt = target.flatten(1)
                    foreground = logits[:, :17].amax(1)
                    record.update(empty_bias=float(model.gaussian_head.empty_bias),
                                  empty_scalar=float(model.gaussian_head.empty_scalar),
                                  scales_quantiles=quantiles(gaussian.scales[inside]),
                                  opacity_quantiles=quantiles(gaussian.opacities[inside]),
                                  scale_upper_saturation_percent=100 * float((gaussian.scales[inside] > .395).float().mean()),
                                  opacity_upper_saturation_percent=100 * float((gaussian.opacities[inside] > .99).float().mean()),
                                  softplus_semantic_max_quantiles=quantiles(gaussian.semantics.amax(-1)[inside]),
                                  foreground_logits_on_gt_occupied_quantiles=quantiles(foreground[gt != 17]),
                                  empty_logits_on_gt_occupied_quantiles=quantiles(logits[:, 17][gt != 17]))
                    supervised = not model.pretrain and horizon in (0, 2, 4, 6)
                    trained_horizon = max(1, min(args.epoch - 5, 6))
                    if sample_number < args.gradient_samples and supervised and horizon <= trained_horizon:
                        # Avoid including diagnostic renders in the capture list.
                        model.gaussian_head.render_state = original_render
                        gradients.append(dict(index=index, **gradient_probe(
                            model, cfg, points, classes, gaussian, xyz, metas, horizon)))
                        model.gaussian_head.render_state = record_render
                for mode, prediction in modes.items():
                    for region, mask in [('full', None), ('camera', cam), ('noncamera', ~cam.bool())]:
                        add_histogram(histograms, f'h{horizon}/{mode}/{region}', prediction, target, mask)
                records.append(record)
                del modes
            print(name, index, 'done', flush=True)
            model.memory.clear()
            model.memory_queue.clear()
        output['models'][name] = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                                     indices=indices, records=records, gradient_probes=gradients,
                                     metrics={k: summarize(v) for k, v in histograms.items()})
        captured.clear()
        del model, dataset, result, images, metas, points_list, classes_list, features_list
        if name == 'sparseworld_opus_gaussian_bias':
            del original_render, record_render, gaussian, logits
        torch.cuda.empty_cache()
    atomic_json(args.output, output)
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
