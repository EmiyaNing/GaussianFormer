"""Reusable diagnostics for each epoch of the full-space experiment."""
import numpy as np
import torch
import torch.nn.functional as F

from loss import OPENOCC_LOSS
from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from .diagnose_first_epoch import quantiles, summarize
from .metrics import CLASS_NAMES


def partition_histograms(prediction, target, camera):
    gt = target.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    pred = prediction.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    visible = camera.detach().cpu().numpy().reshape(-1).astype(bool, copy=False)
    if pred.shape != gt.shape or visible.shape != gt.shape:
        raise ValueError('visibility grid shapes differ')
    if ((pred < 0) | (pred >= 18)).any():
        raise ValueError('invalid predicted class')
    valid = (gt >= 0) & (gt < 18)
    ids = gt[valid] * 18 + pred[valid]
    full = np.bincount(ids, minlength=324).reshape(18, 18)
    camera_hist = np.bincount(ids[visible[valid]], minlength=324).reshape(18, 18)
    return dict(full=full, camera=camera_hist, noncamera=full - camera_hist)


def region_metrics(hist):
    result = summarize(hist) if hist.sum() else dict(
        miou=0., class_iou=[0.] * 18, occupied_iou=0., occupied_precision=0.,
        occupied_recall=0., predicted_occupied_percent=0.)
    tp = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - tp
    result.update(voxel_count=int(hist.sum()), gt_occupied=int(hist[:17].sum()),
                  occupied_tp=int(hist[:17, :17].sum()),
                  occupied_fp=int(hist[17, :17].sum()),
                  occupied_fn=int(hist[:17, 17].sum()),
                  class_gt_count=hist.sum(1).tolist(), class_union=union.tolist(),
                  absent_union_classes=[CLASS_NAMES[i] for i in range(18) if union[i] == 0])
    return result


def add_evaluation_batch(metric, visibility, result, metas):
    grids = [result['final_occ_grid']] + result['future_occ_grids']
    if len(grids) != 7:
        raise ValueError('all seven horizons are required')
    for h, prediction in enumerate(grids):
        target = metas['occ_label'] if h == 0 else metas['future_occ_labels'][:, h - 1]
        cam = metas['occ_cam_mask'] if h == 0 else metas['future_occ_cam_masks'][:, h - 1]
        partitions = partition_histograms(prediction, target, cam)
        metric.confusion[h] += partitions['full']
        for region, hist in partitions.items():
            visibility[region][h] += hist
    trajectories = result['pred_traj']
    errors = (trajectories - metas['temporal_trajs'].to(trajectories)).square().sum(-1)
    count = errors.shape[1]
    metric.trajectory_squared[:count] += errors.sum(0).cpu().numpy()
    metric.trajectory_l2[:count] += errors.sqrt().sum(0).cpu().numpy()
    metric.trajectory_count[:count] += errors.shape[0]
    metric.samples += grids[0].shape[0]


def attribute_record(index, horizon, points, classes, features, gaussian, renderer):
    inside = torch.isfinite(points).all(-1) & ((points >= 0) & (points < 1)).all(-1)
    scales, opacity = gaussian.scales[inside], gaussian.opacities[inside]
    return dict(index=index, horizon=horizon, seconds=horizon * .5,
                queries=points.shape[1], children=points.shape[2],
                valid_children=int(inside.sum()),
                in_range_percent=100 * float(inside.float().mean()),
                feature_norm_quantiles=quantiles(features.norm(dim=-1)),
                scales_quantiles=quantiles(scales), opacity_quantiles=quantiles(opacity),
                scale_upper_saturation_percent=100 * float((scales > .395).float().mean()) if scales.numel() else 0.,
                opacity_upper_saturation_percent=100 * float((opacity > .99).float().mean()) if opacity.numel() else 0.,
                softplus_semantic_max_quantiles=quantiles(gaussian.semantics.amax(-1)[inside]),
                point_score_quantiles=quantiles(classes.sigmoid().amax(-1)[inside]),
                empty_bias=float(renderer.empty_bias.detach()),
                empty_scalar=float(renderer.empty_scalar.detach()))


def gradient_relation(left, right):
    left, right = left.reshape(-1), right.reshape(-1)
    return dict(set_norm=float(left.norm()), voxel_norm=float(right.norm()),
                voxel_to_set_norm_ratio=float(right.norm() / left.norm().clamp_min(1e-20)),
                cosine=float((left * right).sum() /
                             (left.norm() * right.norm()).clamp_min(1e-20)))


def gradient_probe(model, cfg, result, gaussian, xyz, metas, horizon, loss_fn):
    current = model.pts_bbox_head.ind_stamps_all == 0
    temporal = horizon == 0 and model.pretrain
    if horizon == 0:
        points, classes = result['all_refine_pts'][-1], result['all_cls_scores'][-1]
        if not temporal:
            points, classes = points[:, current], classes[:, current]
    else:
        points, classes = result['forecast_points_list'][horizon - 1], result['forecast_semantics_list'][horizon - 1]
    target = metas['occ_label'] if horizon == 0 else metas['future_occ_labels'][:, horizon - 1]
    prediction_mask = None
    if horizon:
        original_world = _opus_decode_points(points, model.pc_range)
        ego2lidar = metas['ego2lidar'].to(original_world)
        lidar = original_world @ ego2lidar[:, None, :3, :3].transpose(-1, -2)
        lidar += ego2lidar[:, None, None, :3, 3]
        trajectory = metas['temporal_trajs'][:, horizon - 1].to(original_world)
        lidar += torch.cat((-trajectory, trajectory.new_zeros(trajectory.shape[0], 1)), dim=-1)[:, None, None]
        inverse = torch.linalg.inv(ego2lidar)
        shifted = lidar @ inverse[:, None, :3, :3].transpose(-1, -2)
        shifted += inverse[:, None, None, :3, 3]
        prediction_mask = _opus_encode_points(shifted, model.pc_range)[0, ..., 0] >= 0
    with torch.enable_grad():
        raw = classes.detach().clone().requires_grad_(True)
        world = _opus_decode_points(points, model.pc_range).detach().clone().requires_grad_(True)
        targets = (loss_fn.strict._expanded_targets(metas, 0, world) if temporal else
                   loss_fn.strict._dense_points(target[0], world))
        cls, pts = loss_fn.strict._set_loss(
            world[0].flatten(0, 1), raw[0].flatten(0, 1), *targets,
            temporal_reweight=temporal, prediction_mask=prediction_mask)
        set_grad = torch.autograd.grad(cls + pts, (raw, world))
        render_raw, render_world = (raw[:, current], world[:, current]) if temporal else (raw, world)
        lower = model.pc_range[:3].to(world)
        extent = model.pc_range[3:].to(world) - lower
        means = torch.minimum(torch.maximum(render_world, lower + extent * 1e-4),
                              lower + extent * (1 - 1e-4))
        altered = gaussian._replace(means=means, semantics=F.softplus(render_raw))
        rendered = model.gaussian_head.render_state(altered, xyz)
        mask = None
        if cfg.loss.mask_type == 'camera':
            mask = metas['occ_cam_mask'] if horizon == 0 else metas['future_occ_cam_masks'][:, horizon - 1]
        weight = cfg.loss.current_weight if horizon == 0 else cfg.loss.future_weight
        voxel = weight * loss_fn.occupancy.loss_voxel([rendered], xyz, target.flatten(1), mask)
        voxel_grad = torch.autograd.grad(voxel, (raw, world))
    if temporal:
        set_grad = tuple(value[:, current] for value in set_grad)
        voxel_grad = tuple(value[:, current] for value in voxel_grad)
    return dict(horizon=horizon, seconds=horizon * .5, set_target='temporal_expanded' if temporal else 'direct_horizon',
                voxel_mask=cfg.loss.mask_type, voxel_weight=weight,
                scope='Final raw semantic logits and world-coordinate centers; fixed features/covariance/opacity.',
                set_classification_loss=float(cls.detach()), set_point_loss=float(pts.detach()),
                weighted_voxel_loss=float(voxel.detach()),
                semantic=gradient_relation(set_grad[0], voxel_grad[0]),
                geometry=gradient_relation(set_grad[1], voxel_grad[1]))
