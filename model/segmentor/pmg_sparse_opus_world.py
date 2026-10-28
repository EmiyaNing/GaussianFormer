"""Isolated architecture ablations of Sparse OPUS Gaussian World.

The production D implementation is unchanged. All modes retain its parameter
namespace, point/voxel objectives, semantic motion gate and temporal curriculum.
The first experiment changes geometry propagation only; content persistence is
an independent switch, so its gain cannot be mistaken for a geometry effect.
"""

import contextlib

import torch
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from model.encoder.pmg_shape_projection import centered, structured_update
from .sparse_opus_gausisan_world import SparseOpusGaussianWorld


def transport_renderer_quaternion(rotations, angle):
    """Rotate renderer covariance by world yaw under its Q^T D Q convention."""
    w, x, y, z = rotations.unbind(-1)
    c, s = (angle[..., None] * .5).cos(), -(angle[..., None] * .5).sin()
    return torch.stack((w*c-z*s, x*c+y*s, y*c-x*s, z*c+w*s), -1)


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGSparseOpusWorld(SparseOpusGaussianWorld):
    def __init__(self, geometry_mode="baseline", shape_penalty=2.0,
                 shape_radius_rms=0.10, persistent_content=False, **kwargs):
        super().__init__(**kwargs)
        if geometry_mode not in {"baseline", "translation", "rigid", "free", "prox"}:
            raise ValueError("invalid PMG geometry mode")
        if shape_penalty < 0 or shape_radius_rms < 0:
            raise ValueError("invalid PMG shape constraint")
        self.geometry_mode = geometry_mode
        self.shape_penalty = float(shape_penalty)
        self.shape_radius_rms = float(shape_radius_rms)
        self.persistent_content = bool(persistent_content)

    def _forecast(self, outputs, ego_feature, metas):
        if self.geometry_mode == "baseline":
            return super()._forecast(outputs, ego_feature, metas)
        batch = ego_feature.shape[0]
        stamps = self.pts_bbox_head.ind_stamps_all
        all_features = outputs["query_feat"]
        all_points = outputs["all_refine_pts"][-1]
        current = stamps == 0
        features = all_features[:, current]
        points = all_points[:, current].detach()
        template = centered(_opus_decode_points(points, self.pc_range)).detach()
        timestamp = points.new_zeros(batch, self.num_query, self.num_refines, 1)
        horizon = (self.num_fu_frames if not self.training else
                   max(1, min(self.curr_epoch - self.finetune_epoch + 1,
                              self.num_fu_frames)))
        forecast_points, forecast_classes, forecast_features = [], [], []
        trajectories, masks, angles, state_order = [], [], [], []
        indices = current.nonzero(as_tuple=True)[0]
        for interval in range(horizon):
            fused, _ = self.ego_cross_attn(
                ego_feature.new_ones(batch, 1, 3) * .5, ego_feature,
                points.detach(), features.detach())
            trajectories.append(self.traj_head(fused))
            future = stamps == interval + 1
            features = torch.cat((features, all_features[:, future]), 1)
            points = torch.cat((points, all_points[:, future]), 1).detach()
            indices = torch.cat((indices, future.nonzero(as_tuple=True)[0]))
            timestamp = torch.cat((timestamp, points.new_full(
                (batch, int(future.sum()), self.num_refines, 1), .5)), 1)
            position = self.position_encoder(torch.cat((points, timestamp), -1).flatten(2, 3))
            features = features + fused + position
            offsets = self.reg_branch(features).unflatten(-1, (-1, 3)) * .5
            classes = self.cls_branch(features).unflatten(-1, (-1, 17))
            velocity = self.vel_branch(features).unflatten(-1, (-1, 2))
            labels = classes.argmax(-1)
            moving = ((labels >= 2) & (labels <= 10)).unsqueeze(-1)
            offsets = torch.cat((offsets[..., :2] + velocity * moving, offsets[..., 2:]), -1)
            previous_world = _opus_decode_points(points, self.pc_range)
            current_world, current_angle = structured_update(
                previous_world[:, :self.num_query], template, offsets[:, :self.num_query],
                self.geometry_mode, self.shape_penalty, self.shape_radius_rms)
            # Birth slots have no reliable historical shape. Preserve their
            # native geometry rule in this mechanism experiment, including
            # subsequent rollout steps, rather than freezing an unsupervised
            # perception-time template for an unseen object.
            birth_world = previous_world[:, self.num_query:].mean(-2, keepdim=True) + offsets[:, self.num_query:]
            world = torch.cat((current_world, birth_world), 1)
            angle = torch.cat((current_angle, current_angle.new_zeros(batch, birth_world.shape[1])), 1)
            points = _opus_encode_points(world, self.pc_range)
            forecast_points.append(points)
            forecast_classes.append(classes)
            forecast_features.append(features)
            angles.append(angle)
            state_order.append(indices)
            if self.training:
                ego2lidar = metas["ego2lidar"].to(points)
                lidar = world @ ego2lidar[:, None, :3, :3].transpose(-1, -2)
                lidar = lidar + ego2lidar[:, None, None, :3, 3]
                trajectory = metas["temporal_trajs"][:, interval].to(points)
                lidar = lidar + torch.cat((-trajectory, trajectory.new_zeros(batch, 1)), -1)[:, None, None]
                lidar2ego = torch.linalg.inv(ego2lidar)
                shifted = lidar @ lidar2ego[:, None, :3, :3].transpose(-1, -2)
                shifted = shifted + lidar2ego[:, None, None, :3, 3]
                masks.append(_opus_encode_points(shifted, self.pc_range)[..., 0] >= 0)
        if not self.pretrain and len(trajectories) < self.num_fu_frames:
            fused, _ = self.ego_cross_attn(
                ego_feature.new_zeros(batch, 1, 3), ego_feature, points, features)
            trajectories.append(self.traj_head(fused))
        outputs["pmg_angles"] = angles
        outputs["pmg_state_order"] = state_order
        return forecast_points, forecast_classes, forecast_features, trajectories, masks

    def _render_occupancy(self, outputs, forecast_points, forecast_classes,
                          forecast_features, current_queries, sampled_xyz,
                          batch_size, render_all=True):
        if not self.persistent_content:
            return super()._render_occupancy(
                outputs, forecast_points, forecast_classes, forecast_features,
                current_queries, sampled_xyz, batch_size, render_all)
        source = outputs["representation"][-1]["gaussian"]
        state = type(source)(*(value[:, current_queries] if value is not None else None for value in source))
        logits_by_horizon = {0: self.gaussian_head.render_state(state, sampled_xyz)}
        grids = [logits_by_horizon[0].argmax(1).view(batch_size, 200, 200, 16)]
        # The production perception state already multiplies opacity by the
        # current ROI mask. Recover the SAME branch's intrinsic opacity once,
        # so an initially outside persistent slot can enter the grid later.
        encoder = self.pts_bbox_head.transformer.encoder
        _, _, intrinsic_opacity = encoder.attribute_branches[str(len(encoder.layers) - 1)](
            outputs["query_feat"])
        query_valid = outputs["representation"][-1]["query_valid_mask"]
        source = source._replace(opacities=intrinsic_opacity * query_valid[:, :, None, None])
        for horizon, points in enumerate(forecast_points, 1):
            supervised = self.training and not self.pretrain and horizon in self.gaussian_loss_horizons
            if not render_all and not supervised:
                continue
            with contextlib.nullcontext() if supervised else torch.no_grad():
                order = outputs["pmg_state_order"][horizon - 1]
                state = type(source)(*(value[:, order] if value is not None else None for value in source))
                count = int(current_queries.sum())
                state = type(source)(*(value[:, :count] if value is not None else None for value in state))
                angle = outputs["pmg_angles"][horizon - 1][:, :count]
                # The renderer stores C=Q^T diag(s^2) Q, hence Q_new=Q_old R^T.
                rotations = transport_renderer_quaternion(state.rotations, angle)
                current_points = points[:, :count]
                valid = torch.isfinite(current_points).all(-1) & ((current_points >= 0) & (current_points < 1)).all(-1)
                # localagg requires EVERY center inside the ROI, even at zero
                # opacity. Clamp only the temporary renderer view; raw rollout
                # coordinates remain intact for loss, diagnostics and re-entry.
                epsilon = self.pts_bbox_head.transformer.encoder.render_eps
                means = _opus_decode_points(current_points.clamp(epsilon, 1. - epsilon), self.pc_range)
                state = state._replace(means=means, rotations=rotations,
                                       opacities=state.opacities * valid[..., None])
                if points.shape[1] > count:
                    birth_features = forecast_features[horizon - 1][:, count:]
                    births = encoder._make_gaussian(
                        birth_features, points[:, count:], forecast_classes[horizon - 1][:, count:],
                        torch.ones(birth_features.shape[:2], dtype=torch.bool, device=points.device),
                        len(encoder.layers) - 1)
                    state = type(source)(*(torch.cat((a, b), 1) if a is not None and b is not None else None
                                           for a, b in zip(state, births)))
                logits = self.gaussian_head.render_state(state, sampled_xyz)
                if supervised:
                    logits_by_horizon[horizon] = logits
                if render_all:
                    grids.append(logits.argmax(1).view(batch_size, 200, 200, 16))
        return logits_by_horizon, grids
