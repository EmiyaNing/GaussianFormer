"""The completed D model, registered as ``sparse_opus_gausisan_world``.

    Images -> timestamp-aware OPUS perception -> current Gaussian occupancy
                                             -> autoregressive world rollout

The rollout and trajectory modules are the native SparseWorld implementation.
This module owns image memory controls and current/future Gaussian rendering.
It preserves D's state_dict names and its half-second horizon indexing.
"""

import contextlib

import torch
from torch.utils.checkpoint import checkpoint
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.opus_encoder import _opus_decode_points
from model.head.sparse_opus_gausisan_world_head import (  # registry imports
    SparseOpusGaussianPerceptionHead,
    SparseOpusGaussianRenderer,
)
from .sparseworld_strict import SparseWorldStrict


@MODELS.register_module(name="sparse_opus_gausisan_world")
@SEGMENTORS.register_module(name="sparse_opus_gausisan_world")
class SparseOpusGaussianWorld(SparseWorldStrict):
    """D's Gaussian perception and SparseWorld forecasting without adapters.

    ``gaussian_loss_horizons`` indexes half-second steps, not seconds. D uses
    (0, 2, 4, 6): current plus 1/2/3 seconds. Every half-second output is still
    predicted and evaluated. With ``render_training_grids=False``, training
    renders only loss horizons and returns the current grid; point predictions
    still participate in the original set objective. Evaluation returns all
    grids, as does training with ``return_all_occ=True``.
    """

    def __init__(
        self,
        gaussian_head,
        gaussian_loss_horizons=(0, 2, 4, 6),
        image_chunk_size=6,
        image_checkpoint=True,
        image_amp_dtype="bfloat16",
        render_training_grids=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_chunk_size = int(image_chunk_size)
        self.image_checkpoint = bool(image_checkpoint)
        if self.image_chunk_size < 1 or image_amp_dtype not in ("bfloat16", "float32"):
            raise ValueError("invalid image memory configuration")
        self.image_amp_dtype = image_amp_dtype
        self.render_training_grids = bool(render_training_grids)
        # Chunk checkpointing owns recomputation; disable backbone-local copies.
        for module in self.img_backbone.modules():
            if hasattr(module, "with_cp"):
                module.with_cp = False
        self.gaussian_head = MODELS.build(gaussian_head)
        self.gaussian_loss_horizons = tuple(gaussian_loss_horizons)
        if 0 not in self.gaussian_loss_horizons:
            raise ValueError("current Gaussian supervision is required")
        if any(
            horizon < 0 or horizon > self.num_fu_frames
            for horizon in self.gaussian_loss_horizons
        ):
            raise ValueError("invalid Gaussian horizon")

    def _apply(self, function):
        result = super()._apply(function)
        # CUDA/dtype conversion replaces the buffer referenced by this alias.
        self.pc_range = self.pts_bbox_head.pc_range
        return result

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if self.pretrain:
            # Evaluation installs masks; warm-up reassigns timestamps per batch.
            for layer in self.pts_bbox_head.transformer.decoder.decoder_layers:
                layer.self_attn.ind_mask = None

    def _image_chunk(self, images):
        """Run the backbone in BF16 and the FPN in FP32, as in D."""
        with torch.autocast(
            device_type=images.device.type,
            dtype=torch.bfloat16,
            enabled=images.is_cuda and self.image_amp_dtype == "bfloat16",
        ):
            features = self.img_backbone(images)
        if isinstance(features, dict):
            features = list(features.values())
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = self.img_neck([feature.float() for feature in features])
            if isinstance(features, dict):
                features = features["fpn_out"]
            return tuple(feature.float() for feature in features)

    def _extract_feat_batch(self, images):
        images, batch_size, cameras = self._preprocess_images(images)
        chunks = []
        for image_chunk in images.split(self.image_chunk_size):
            if self.training and torch.is_grad_enabled() and self.image_checkpoint:
                features = checkpoint(
                    self._image_chunk, image_chunk, use_reentrant=False
                )
            else:
                features = self._image_chunk(image_chunk)
            chunks.append(features)
        return [
            torch.cat(level).view(batch_size, cameras, *level[0].shape[1:])
            for level in zip(*chunks)
        ]

    def _render_occupancy(
        self,
        outputs,
        forecast_points,
        forecast_classes,
        forecast_features,
        current_queries,
        sampled_xyz,
        batch_size,
        render_all=True,
    ):
        """Render loss horizons during training; optionally return every grid."""
        gaussian = outputs["representation"][-1]["gaussian"]
        current_gaussian = type(gaussian)(
            *(
                field[:, current_queries] if field is not None else None
                for field in gaussian
            )
        )
        logits_by_horizon = {
            0: self.gaussian_head.render_state(current_gaussian, sampled_xyz)
        }
        grids = [logits_by_horizon[0].argmax(1).view(batch_size, 200, 200, 16)]
        encoder = self.pts_bbox_head.transformer.encoder
        forecast_states = zip(forecast_points, forecast_classes, forecast_features)
        for horizon, (points, classes, features) in enumerate(forecast_states, 1):
            supervised = (
                self.training
                and not self.pretrain
                and horizon in self.gaussian_loss_horizons
            )
            if not render_all and not supervised:
                continue
            context = contextlib.nullcontext() if supervised else torch.no_grad()
            with context:
                valid = torch.ones(
                    features.shape[:2], dtype=torch.bool, device=features.device
                )
                future_gaussian = encoder._make_gaussian(
                    features, points, classes, valid, len(encoder.layers) - 1
                )
                logits = self.gaussian_head.render_state(future_gaussian, sampled_xyz)
                if supervised:
                    logits_by_horizon[horizon] = logits
                if render_all:
                    grids.append(logits.argmax(1).view(batch_size, 200, 200, 16))
        return logits_by_horizon, grids

    def forward(self, imgs=None, metas=None, **kwargs):
        if imgs is None or metas is None:
            raise ValueError("imgs and metas are required")
        batch_size = imgs.shape[0]
        ego_state = metas["temporal_ego_states"].to(imgs).reshape(batch_size, 1, -1)
        if ego_state.shape[-1] != 21:
            raise ValueError("ego state must contain 21 values")
        ego_features = self.plan_head(ego_state)
        point_scale = (
            torch.tanh(self.points_scale_branch(ego_features)) + 1
        ) * 0.35 + 0.8
        outputs = self.pts_bbox_head(
            self.extract_feat(imgs, metas), metas, points_scale=point_scale
        )
        points, classes, features, trajectories, masks = self._forecast(
            outputs, ego_features, metas
        )
        stamps = self.pts_bbox_head.ind_stamps_all
        current = stamps == 0
        xyz = metas["occ_xyz"].float().flatten(1, 3)
        logits_by_horizon, grids = self._render_occupancy(
            outputs, points, classes, features, current, xyz, batch_size,
            # Evaluation always supplies the full temporal metric contract.
            # Training callers can request grids explicitly for diagnostics.
            not self.training or getattr(self, 'render_training_grids', True)
            or kwargs.get('return_all_occ', False),
        )

        result = dict(outputs)
        result.update(
            cls_score=outputs["all_cls_scores"][-1][:, current],
            refine_pts=outputs["all_refine_pts"][-1][:, current],
            forecast_points_list=points,
            forecast_semantics_list=classes,
            forecast_features_list=features,
            pred_trajs_list=trajectories,
            pred_traj=torch.cat(trajectories, 1),
            forecast_points_mask_list=masks,
            sparseworld_pretrain=self.pretrain,
            query_stamps=stamps,
            stamp_statistics=self.pts_bbox_head.num_stamps_all,
            update_stamp_statistics=self.training,
            initial_pred_points=_opus_decode_points(
                outputs["init_points"], self.pc_range
            ).flatten(1, 2),
            current_pred_points=[
                _opus_decode_points(p[:, current], self.pc_range).flatten(1, 2)
                for p in outputs["all_refine_pts"]
            ],
            current_pred_logits=[
                c[:, current].flatten(1, 2) for c in outputs["all_cls_scores"]
            ],
            temporal_pred_points=[
                _opus_decode_points(p, self.pc_range).flatten(1, 2)
                for p in outputs["all_refine_pts"]
            ],
            temporal_pred_logits=[c.flatten(1, 2) for c in outputs["all_cls_scores"]],
            future_pred_points=[
                _opus_decode_points(p, self.pc_range).flatten(1, 2) for p in points
            ],
            future_pred_logits=[c.flatten(1, 2) for c in classes],
            future_pred_masks=[mask.flatten(1) for mask in masks],
            gaussian_logits_by_horizon=logits_by_horizon,
            final_occ=grids[0].flatten(1),
            final_occ_grid=grids[0],
            future_occ_grids=grids[1:],
            # The generic runner pairs sampled_label with flattened final_occ.
            # Without a metric mask, MeanIoU does not flatten either tensor.
            # Keep dense labels in metas for the unchanged training objective.
            sampled_label=metas["occ_label"].flatten(1),
            occ_cam_mask=metas.get("occ_cam_mask"),
            occ_lidar_mask=metas.get("occ_lidar_mask"),
        )
        # Keep the legacy keys: their integer suffix is a half-second index.
        for horizon, grid in enumerate(grids):
            result[f"semantic_occ_{horizon}s"] = grid
            result[f"geo_occ_{horizon}s"] = torch.where(grid != 17, 0, 17)
        return result
