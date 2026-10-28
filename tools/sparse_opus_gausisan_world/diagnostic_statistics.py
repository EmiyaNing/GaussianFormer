"""Pure statistics and output-gradient probes used by per-epoch diagnostics.

These probes never update model parameters or select a calibration threshold.
Fixed-sample attributes are distinct from the full-subset confusion matrices.
"""

import numpy as np
import torch
import torch.nn.functional as F
from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from .metrics import CLASS_NAMES


def summarize(hist):
    tp = np.diag(hist).astype(float)
    union = hist.sum(0) + hist.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros(18), where=union > 0) * 100
    occupied_tp = float(hist[:17, :17].sum())
    occupied_gt = float(hist[:17].sum())
    occupied_pred = float(hist[:, :17].sum())
    return dict(
        miou=float(iou[:17].mean()),
        class_iou=iou.tolist(),
        occupied_iou=100
        * occupied_tp
        / max(occupied_gt + occupied_pred - occupied_tp, 1),
        occupied_precision=100 * occupied_tp / max(occupied_pred, 1),
        occupied_recall=100 * occupied_tp / max(occupied_gt, 1),
        predicted_occupied_percent=100 * occupied_pred / hist.sum(),
    )


def quantiles(value):
    value = value.detach().float().reshape(-1)
    if not value.numel():
        return None
    return (
        torch.quantile(value, value.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
        .cpu()
        .tolist()
    )


@torch.no_grad()
def classification_statistics(logits, target, camera, cfg, index):
    """Decompose D's full-space CE and its signed learned-bias gradient.

    A positive dL/db suppresses the empty bias under gradient descent. This
    probe includes the original CE term only; conditional CE is saved separately.
    """
    weights = logits.new_tensor(cfg.loss.occupancy_loss.manual_class_weight)
    labels = target.flatten(1).long()
    visible = camera.flatten(1).bool()
    valid = labels != 255
    safe = labels.masked_fill(~valid, 0)
    log_probability = logits.float().log_softmax(1)
    log_correct = log_probability.gather(1, safe[:, None]).squeeze(1)
    ce = -log_correct
    sample_weights = weights.to(log_correct)[safe] * valid.to(log_correct)
    pt = (-ce).exp()
    empty_probability = logits.float().softmax(1)[:, 17]
    denominator = sample_weights.sum().clamp_min(torch.finfo(sample_weights.dtype).tiny)
    coefficient = (
        cfg.loss.occupancy_loss.multi_loss_weights.loss_voxel_ce_weight
        * cfg.loss.current_weight
    )
    empty_target = labels == 17
    bias_gradient = (
        coefficient
        * sample_weights
        * (empty_target.float() - empty_probability)
        / denominator
    )
    rows = {}
    for region, region_mask in (("camera", visible), ("noncamera", ~visible)):
        for group, occupied_mask in (
            ("free", empty_target),
            ("occupied", ~empty_target),
        ):
            mask = valid & region_mask & occupied_mask
            count = int(mask.sum())
            contribution = float(
                coefficient * (ce * sample_weights)[mask].sum() / denominator
            )
            rows[f"{region}/{group}"] = dict(
                voxels=count,
                label_weight_sum=float(sample_weights[mask].sum()),
                classification_loss_contribution=contribution,
                ce_counterfactual_loss_contribution=contribution,
                signed_bias_gradient=float(bias_gradient[mask].sum()),
                mean_correct_class_probability=float(pt[mask].mean()) if count else 0.0,
                correct_class_probability_over_0_9_percent=(
                    100 * float((pt[mask] > 0.9).float().mean()) if count else 0.0
                ),
            )
    return dict(
        index=index,
        objective="weighted_ce",
        gamma=0.0,
        empty_class_relative_weight=float(weights[17]),
        denominator=float(denominator),
        current_classification_loss=float(
            coefficient * (ce * sample_weights).sum() / denominator
        ),
        signed_bias_gradient=float(bias_gradient.sum()),
        buckets=rows,
        scope="Rendered-logit CE only; excludes conditional CE, Lovasz, set and trajectory losses.",
    )


def partition_histograms(prediction, target, camera):
    gt = target.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    pred = prediction.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    visible = camera.detach().cpu().numpy().reshape(-1).astype(bool, copy=False)
    if pred.shape != gt.shape or visible.shape != gt.shape:
        raise ValueError("visibility grid shapes differ")
    if ((pred < 0) | (pred >= 18)).any():
        raise ValueError("invalid predicted class")
    valid = (gt >= 0) & (gt < 18)
    ids = gt[valid] * 18 + pred[valid]
    full = np.bincount(ids, minlength=324).reshape(18, 18)
    camera_hist = np.bincount(ids[visible[valid]], minlength=324).reshape(18, 18)
    return dict(full=full, camera=camera_hist, noncamera=full - camera_hist)


def region_metrics(hist):
    result = (
        summarize(hist)
        if hist.sum()
        else dict(
            miou=0.0,
            class_iou=[0.0] * 18,
            occupied_iou=0.0,
            occupied_precision=0.0,
            occupied_recall=0.0,
            predicted_occupied_percent=0.0,
        )
    )
    tp = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - tp
    result.update(
        voxel_count=int(hist.sum()),
        gt_occupied=int(hist[:17].sum()),
        occupied_tp=int(hist[:17, :17].sum()),
        occupied_fp=int(hist[17, :17].sum()),
        occupied_fn=int(hist[:17, 17].sum()),
        class_gt_count=hist.sum(1).tolist(),
        class_union=union.tolist(),
        absent_union_classes=[CLASS_NAMES[i] for i in range(18) if union[i] == 0],
    )
    return result


def add_evaluation_batch(metric, visibility, result, metas):
    grids = [result["final_occ_grid"]] + result["future_occ_grids"]
    if len(grids) != 7:
        raise ValueError("all seven horizons are required")
    for h, prediction in enumerate(grids):
        target = metas["occ_label"] if h == 0 else metas["future_occ_labels"][:, h - 1]
        cam = (
            metas["occ_cam_mask"] if h == 0 else metas["future_occ_cam_masks"][:, h - 1]
        )
        partitions = partition_histograms(prediction, target, cam)
        metric.confusion[h] += partitions["full"]
        for region, hist in partitions.items():
            visibility[region][h] += hist
    trajectories = result["pred_traj"]
    errors = (trajectories - metas["temporal_trajs"].to(trajectories)).square().sum(-1)
    count = errors.shape[1]
    metric.trajectory_squared[:count] += errors.sum(0).cpu().numpy()
    metric.trajectory_l2[:count] += errors.sqrt().sum(0).cpu().numpy()
    metric.trajectory_count[:count] += errors.shape[0]
    metric.samples += grids[0].shape[0]


def attribute_record(index, horizon, points, classes, features, gaussian, renderer):
    inside = torch.isfinite(points).all(-1) & ((points >= 0) & (points < 1)).all(-1)
    scales, opacity = gaussian.scales[inside], gaussian.opacities[inside]
    return dict(
        index=index,
        horizon=horizon,
        seconds=horizon * 0.5,
        queries=points.shape[1],
        children=points.shape[2],
        valid_children=int(inside.sum()),
        in_range_percent=100 * float(inside.float().mean()),
        feature_norm_quantiles=quantiles(features.norm(dim=-1)),
        scales_quantiles=quantiles(scales),
        opacity_quantiles=quantiles(opacity),
        scale_upper_saturation_percent=(
            100 * float((scales > 0.395).float().mean()) if scales.numel() else 0.0
        ),
        opacity_upper_saturation_percent=(
            100 * float((opacity > 0.99).float().mean()) if opacity.numel() else 0.0
        ),
        softplus_semantic_max_quantiles=quantiles(gaussian.semantics.amax(-1)[inside]),
        point_score_quantiles=quantiles(classes.sigmoid().amax(-1)[inside]),
        empty_bias=float(renderer.empty_bias.detach()),
        empty_scalar=float(renderer.empty_scalar.detach()),
    )


def gradient_relation(left, right):
    left, right = left.reshape(-1), right.reshape(-1)
    return dict(
        set_norm=float(left.norm()),
        voxel_norm=float(right.norm()),
        voxel_to_set_norm_ratio=float(right.norm() / left.norm().clamp_min(1e-20)),
        cosine=float(
            (left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-20)
        ),
    )


def gradient_probe(model, cfg, result, gaussian, xyz, metas, horizon, loss_fn):
    current = model.pts_bbox_head.ind_stamps_all == 0
    temporal = horizon == 0 and model.pretrain
    if horizon == 0:
        points, classes = result["all_refine_pts"][-1], result["all_cls_scores"][-1]
        if not temporal:
            points, classes = points[:, current], classes[:, current]
    else:
        points, classes = (
            result["forecast_points_list"][horizon - 1],
            result["forecast_semantics_list"][horizon - 1],
        )
    target = (
        metas["occ_label"]
        if horizon == 0
        else metas["future_occ_labels"][:, horizon - 1]
    )
    prediction_mask = None
    if horizon:
        original_world = _opus_decode_points(points, model.pc_range)
        ego2lidar = metas["ego2lidar"].to(original_world)
        lidar = original_world @ ego2lidar[:, None, :3, :3].transpose(-1, -2)
        lidar += ego2lidar[:, None, None, :3, 3]
        trajectory = metas["temporal_trajs"][:, horizon - 1].to(original_world)
        lidar += torch.cat(
            (-trajectory, trajectory.new_zeros(trajectory.shape[0], 1)), dim=-1
        )[:, None, None]
        inverse = torch.linalg.inv(ego2lidar)
        shifted = lidar @ inverse[:, None, :3, :3].transpose(-1, -2)
        shifted += inverse[:, None, None, :3, 3]
        prediction_mask = _opus_encode_points(shifted, model.pc_range)[0, ..., 0] >= 0
    with torch.enable_grad():
        raw = classes.detach().clone().requires_grad_(True)
        world = (
            _opus_decode_points(points, model.pc_range)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        targets = (
            loss_fn.strict._expanded_targets(metas, 0, world)
            if temporal
            else loss_fn.strict._dense_points(target[0], world)
        )
        cls, pts = loss_fn.strict._set_loss(
            world[0].flatten(0, 1),
            raw[0].flatten(0, 1),
            *targets,
            temporal_reweight=temporal,
            prediction_mask=prediction_mask,
        )
        set_grad = torch.autograd.grad(cls + pts, (raw, world))
        render_raw, render_world = (
            (raw[:, current], world[:, current]) if temporal else (raw, world)
        )
        lower = model.pc_range[:3].to(world)
        extent = model.pc_range[3:].to(world) - lower
        means = torch.minimum(
            torch.maximum(render_world, lower + extent * 1e-4),
            lower + extent * (1 - 1e-4),
        )
        altered = gaussian._replace(means=means, semantics=F.softplus(render_raw))
        rendered = model.gaussian_head.render_state(altered, xyz)
        mask = None
        if cfg.loss.mask_type == "camera":
            mask = (
                metas["occ_cam_mask"]
                if horizon == 0
                else metas["future_occ_cam_masks"][:, horizon - 1]
            )
        weight = cfg.loss.current_weight if horizon == 0 else cfg.loss.future_weight
        voxel = weight * loss_fn.occupancy.loss_voxel(
            [rendered], xyz, target.flatten(1), mask
        )
        voxel_grad = torch.autograd.grad(voxel, (raw, world))
    if temporal:
        set_grad = tuple(value[:, current] for value in set_grad)
        voxel_grad = tuple(value[:, current] for value in voxel_grad)
    return dict(
        horizon=horizon,
        seconds=horizon * 0.5,
        set_target="temporal_expanded" if temporal else "direct_horizon",
        voxel_mask=cfg.loss.mask_type,
        voxel_weight=weight,
        scope="Final raw semantic logits and world-coordinate centers; fixed features/covariance/opacity.",
        set_classification_loss=float(cls.detach()),
        set_point_loss=float(pts.detach()),
        weighted_voxel_loss=float(voxel.detach()),
        semantic=gradient_relation(set_grad[0], voxel_grad[0]),
        geometry=gradient_relation(set_grad[1], voxel_grad[1]),
    )


@torch.no_grad()
def conditional_statistics(logits, target, camera, cfg, index):
    labels, visible = target.flatten(1).long(), camera.flatten(1).bool()
    valid = (labels >= 0) & (labels < 18)
    occupied = valid & (labels != 17)
    weights = logits.new_tensor(cfg.loss.occupancy_loss.manual_class_weight).float()
    auxiliary_weights = weights[:17] * logits.new_tensor(
        cfg.loss.occupancy_loss.get("conditional_class_multipliers", [1.0] * 17)
    )
    coefficient = (
        cfg.loss.current_weight * cfg.loss.occupancy_loss.conditional_semantic_weight
    )
    safe = labels.masked_fill(~valid, 0)
    sample_weights = weights[safe] * valid
    base_factor = (
        cfg.loss.current_weight
        * cfg.loss.occupancy_loss.multi_loss_weights.loss_voxel_ce_weight
    )
    probabilities = logits.float().softmax(1)
    correct = probabilities.gather(1, safe[:, None]).squeeze(1)
    factors = base_factor * sample_weights / sample_weights.sum().clamp_min(1e-30)
    base_norm_sq = (
        factors.square()
        * (probabilities.square().sum(1) - 2 * correct + 1).clamp_min(0)
    ).sum()
    selected = logits.transpose(1, 2)[occupied][:, :17].float()
    targets = labels[occupied]
    lp = selected.log_softmax(1)
    p = lp.exp()
    pt = p.gather(1, targets[:, None]).squeeze(1)
    ce = -lp.gather(1, targets[:, None]).squeeze(1)
    selected_weights = auxiliary_weights[targets]
    denominator = selected_weights.sum().clamp_min(1e-30)
    auxiliary_factors = coefficient * selected_weights / denominator
    aux_norm_sq = (
        auxiliary_factors.square() * (p.square().sum(1) - 2 * pt + 1).clamp_min(0)
    ).sum()
    selected_base = probabilities.transpose(1, 2)[occupied][:, :17]
    dot = (
        factors[occupied]
        * auxiliary_factors
        * ((selected_base * p).sum(1) - correct[occupied] - pt + 1)
    ).sum()
    buckets = {}
    for name, region in (("camera", visible), ("noncamera", ~visible)):
        mask = region[occupied]
        buckets[name] = dict(
            gt_occupied_voxels=int(mask.sum()),
            loss_contribution=float(
                (coefficient * ce * selected_weights)[mask].sum() / denominator
            ),
            mean_correct_conditional_probability=(
                float(pt[mask].mean()) if mask.any() else 0.0
            ),
        )
    classes = []
    for c, name in enumerate(CLASS_NAMES[:17]):
        mask = targets == c
        classes.append(
            dict(
                class_name=name,
                gt_voxels=int(mask.sum()),
                auxiliary_weight=float(auxiliary_weights[c]),
                auxiliary_loss_contribution=float((ce * auxiliary_factors)[mask].sum()),
                correct_conditional_probability=(
                    float(pt[mask].mean()) if mask.any() else 0.0
                ),
            )
        )
    return dict(
        index=index,
        auxiliary_weight=float(coefficient),
        gt_occupied_voxels=int(occupied.sum()),
        label_weight_sum=float(selected_weights.sum()),
        conditional_semantic_loss=float((ce * auxiliary_factors).sum()),
        direct_empty_logit_gradient=0.0,
        auxiliary_logit_gradient_norm=float(aux_norm_sq.sqrt()),
        base_18class_ce_logit_gradient_norm=float(base_norm_sq.sqrt()),
        auxiliary_to_base_ce_logit_gradient_norm_ratio=float(
            (aux_norm_sq / base_norm_sq.clamp_min(1e-30)).sqrt()
        ),
        auxiliary_base_ce_logit_gradient_cosine=float(
            dot / (base_norm_sq * aux_norm_sq).sqrt().clamp_min(1e-30)
        ),
        buckets=buckets,
        classes=classes,
        scope="Analytical gradients at the rendered 18-class output; excludes Lovasz, set losses and shared parameter gradients.",
    )


def conditional_confusion(logits, target, camera):
    labels = target.flatten(1).long()
    prediction = logits[:, :17].argmax(1)
    valid = (labels >= 0) & (labels < 17)
    visible = camera.flatten(1).bool()
    full = (
        torch.bincount(labels[valid] * 17 + prediction[valid], minlength=289)
        .reshape(17, 17)
        .cpu()
        .numpy()
    )
    mask = valid & visible
    camera_hist = (
        torch.bincount(labels[mask] * 17 + prediction[mask], minlength=289)
        .reshape(17, 17)
        .cpu()
        .numpy()
    )
    return dict(full=full, camera=camera_hist, noncamera=full - camera_hist)
