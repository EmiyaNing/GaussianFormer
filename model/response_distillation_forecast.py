"""Experiment-only SparseWorld rollout with explicit position-gradient policy.

This has exactly the operations and parameters of the strict SparseWorld
rollout. The optional difference is when the current output is detached.
The next step always receives a detached position, as in the strict model.
"""

import torch


def forecast_with_position_gradient(model, encoded, stamps, ego, position_gradient=True):
    """Roll out with an explicit geometry-gradient policy.

    ``False`` restores the strict model's detach-before-output behavior.
    The default preserves earlier experiments that deliberately enabled it.
    Both policies have identical forward values and recurrent detach points.
    """
    all_features = encoded['query_features']
    all_points = encoded['query_points']
    current = stamps == 0
    features = all_features[:, current]
    positions = all_points[:, current].detach()
    timestamp = positions.new_zeros(*positions.shape[:-1], 1)
    pred_points, pred_logits, trajectories = [], [], []
    for step in range(model.future_steps):
        fused = model.ego_attention(ego, features.detach(), positions.detach())
        trajectories.append(model.traj(fused))
        future = stamps == step + 1
        features = torch.cat([features, all_features[:, future]], 1)
        positions = torch.cat([positions, all_points[:, future]], 1).detach()
        timestamp = torch.cat([
            timestamp,
            timestamp.new_full((timestamp.shape[0], int(future.sum()),
                                model.num_refines, 1), .5),
        ], 1)
        features = features + fused.expand(-1, features.shape[1], -1) + model.position(
            torch.cat([positions, timestamp], -1).flatten(2))
        semantic = model.semantic(features).view(
            features.shape[0], features.shape[1], model.num_refines, 17)
        offset = model.reg(features).view(
            features.shape[0], features.shape[1], model.num_refines, 3) * .5
        velocity = model.velocity(features).view(
            features.shape[0], features.shape[1], model.num_refines, 2)
        moving = ((semantic.argmax(-1) >= 2) &
                  (semantic.argmax(-1) <= 10)).unsqueeze(-1)
        offset = torch.cat([offset[..., :2] + velocity * moving,
                            offset[..., 2:]], -1)
        predicted_positions = model._refine(positions, offset)
        output_positions = predicted_positions if position_gradient else predicted_positions.detach()
        pred_points.append(model._world(output_positions).flatten(1, 2))
        pred_logits.append(semantic.flatten(1, 2))
        positions = predicted_positions.detach()
    return dict(points=pred_points, logits=pred_logits,
                trajectories=torch.cat(trajectories, 1))
