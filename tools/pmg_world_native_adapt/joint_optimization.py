"""Explicit parameter ownership and the requested 5+10 epoch LR policy."""

import torch

from tools.sparse_opus_gausisan_world import runtime


PERCEPTION_MODULES = (
    "img_backbone",
    "img_neck",
    "pts_bbox_head",
    "points_scale_branch",
    "gaussian_head",
)
PREDICTION_MODULES = (
    "plan_head",
    "ego_cross_attn",
    "position_encoder",
    "reg_branch",
    "vel_branch",
    "cls_branch",
    "traj_head",
    "query_update_gate",
    "reference_forecast_modules",
)


def parameter_role(name):
    root = name.split(".", 1)[0]
    if root in PERCEPTION_MODULES:
        return "perception"
    if root in PREDICTION_MODULES:
        return "prediction"
    raise ValueError("Parameter has no declared optimization role: " + name)


def make_optimizer(model, cfg):
    """Every parameter is optimized once; no hidden backbone LR multipliers."""
    buckets = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError("Full-network training found frozen weight: " + name)
        role = parameter_role(name)
        decay = 0.0 if name == "gaussian_head.empty_bias" else cfg.weight_decay
        buckets.setdefault((role, decay), []).append(parameter)
    groups = [
        dict(
            params=parameters,
            role=role,
            lr=cfg.base_lr,
            base_lr=cfg.base_lr,
            weight_decay=decay,
        )
        for (role, decay), parameters in buckets.items()
    ]
    return torch.optim.AdamW(groups)


def set_learning_rates(optimizer, update, total, epoch, cfg):
    """epoch is zero-based: ratio 1 through epoch 5, then exactly 0.1."""
    runtime.set_lr(optimizer, update, total, cfg.warmup_iters, cfg.min_lr_ratio)
    ratio = (
        1.0 if epoch < cfg.perception_lr.equal_epochs else cfg.perception_lr.late_ratio
    )
    for group in optimizer.param_groups:
        if group["role"] == "perception":
            group["lr"] *= ratio
    rates = {group["role"]: group["lr"] for group in optimizer.param_groups}
    assert abs(rates["perception"] / rates["prediction"] - ratio) < 1e-12
    return dict(
        perception_lr=rates["perception"],
        prediction_lr=rates["prediction"],
        perception_to_prediction_lr_ratio=ratio,
    )
