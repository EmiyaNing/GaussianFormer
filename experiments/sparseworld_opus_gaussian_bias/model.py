"""Independent baseline runtime and complete Gaussian perception/world model."""
import contextlib

import torch
from torch.utils.checkpoint import checkpoint
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.opus_encoder import _opus_decode_points
from model.head.gaussian_opus_head import _flatten_grouped
from model.head.opus_v1_gaussian_trainable_bias_head import OPUSV1GaussianTrainableBiasHead
from model.segmentor.sparseworld_strict import SparseWorldStrict
from . import encoder  # Register experiment-only components explicitly.


@MODELS.register_module()
class SparseWorldBiasRenderer(OPUSV1GaussianTrainableBiasHead):
    def render_state(self, gaussian, sampled_xyz):
        flat = _flatten_grouped(gaussian)
        if flat.scales.detach().amax() > self.max_render_scale:
            raise ValueError('Gaussian scale exceeds bounded local aggregation range')
        args = self.prepare_gaussian_args(flat)
        logits = self._render_batched(sampled_xyz, *args)
        return torch.cat((logits[:, :self.empty_label],
                          logits[:, self.empty_label:] - self.empty_bias), dim=1)


@SEGMENTORS.register_module()
class SparseWorldStrictExperiment(SparseWorldStrict):
    """Same SparseWorld graph; shared memory controls for both experiment arms."""
    def __init__(self, image_chunk_size=6, image_checkpoint=True,
                 image_amp_dtype='bfloat16', **kwargs):
        super().__init__(**kwargs)
        self.image_chunk_size = int(image_chunk_size)
        self.image_checkpoint = bool(image_checkpoint)
        if self.image_chunk_size < 1 or image_amp_dtype not in ('bfloat16', 'float32'):
            raise ValueError('invalid image memory configuration')
        self.image_amp_dtype = image_amp_dtype
        for module in self.img_backbone.modules():
            if hasattr(module, 'with_cp'):
                module.with_cp = False

    def _apply(self, function):
        result = super()._apply(function)
        # SparseWorld keeps a plain alias to a head buffer. Module.cuda() replaces
        # that buffer, so refresh the alias for both opt-in experiment arms.
        self.pc_range = self.pts_bbox_head.pc_range
        return result

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if self.pretrain:
            # Validation installs timestamp masks. Clear them during the next
            # temporal warm-up epoch, where assignment changes every batch.
            for layer in self.pts_bbox_head.transformer.decoder.decoder_layers:
                layer.self_attn.ind_mask = None

    def _image_chunk(self, images):
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                            enabled=images.is_cuda and self.image_amp_dtype == 'bfloat16'):
            features = self.img_backbone(images)
        if isinstance(features, dict):
            features = list(features.values())
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = self.img_neck([feature.float() for feature in features])
            if isinstance(features, dict):
                features = features['fpn_out']
            return tuple(feature.float() for feature in features)

    def _extract_feat_batch(self, images):
        images, batch, cameras = self._preprocess_images(images)
        chunks = []
        for chunk in images.split(self.image_chunk_size):
            if self.training and torch.is_grad_enabled() and self.image_checkpoint:
                values = checkpoint(self._image_chunk, chunk, use_reentrant=False)
            else:
                values = self._image_chunk(chunk)
            chunks.append(values)
        return [torch.cat(parts).view(batch, cameras, *parts[0].shape[1:])
                for parts in zip(*chunks)]


@SEGMENTORS.register_module()
class SparseWorldOPUSGaussianBias(SparseWorldStrictExperiment):
    def __init__(self, gaussian_head, gaussian_loss_horizons=(0, 2, 4, 6), **kwargs):
        super().__init__(**kwargs)
        self.gaussian_head = MODELS.build(gaussian_head)
        self.gaussian_loss_horizons = tuple(gaussian_loss_horizons)
        if 0 not in self.gaussian_loss_horizons:
            raise ValueError('current Gaussian supervision is required')
        if any(h < 0 or h > self.num_fu_frames for h in self.gaussian_loss_horizons):
            raise ValueError('invalid Gaussian horizon')

    def forward(self, imgs=None, metas=None, **kwargs):
        if imgs is None or metas is None:
            raise ValueError('imgs and metas are required')
        batch = imgs.shape[0]
        state = metas['temporal_ego_states'].to(imgs).reshape(batch, 1, -1)
        if state.shape[-1] != 21:
            raise ValueError('ego state must contain 21 values')
        ego = self.plan_head(state)
        scale = (torch.tanh(self.points_scale_branch(ego)) + 1) * .35 + .8
        outputs = self.pts_bbox_head(self.extract_feat(imgs, metas), metas, points_scale=scale)
        fp, fc, ff, trajectories, masks = self._forecast(outputs, ego, metas)
        stamps = self.pts_bbox_head.ind_stamps_all
        current = stamps == 0
        gaussian = outputs['representation'][-1]['gaussian']
        current_gaussian = type(gaussian)(*(
            field[:, current] if field is not None else None for field in gaussian))
        xyz = metas['occ_xyz'].float().flatten(1, 3)
        logits_by_horizon = {0: self.gaussian_head.render_state(current_gaussian, xyz)}
        grids = [logits_by_horizon[0].argmax(1).view(batch, 200, 200, 16)]
        encoder_module = self.pts_bbox_head.transformer.encoder
        for horizon, (points, classes, features) in enumerate(zip(fp, fc, ff), 1):
            supervised = self.training and not self.pretrain and horizon in self.gaussian_loss_horizons
            context = contextlib.nullcontext() if supervised else torch.no_grad()
            with context:
                valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
                future_gaussian = encoder_module._make_gaussian(
                    features, points, classes, valid, len(encoder_module.layers) - 1)
                logits = self.gaussian_head.render_state(future_gaussian, xyz)
                if supervised:
                    logits_by_horizon[horizon] = logits
                grids.append(logits.argmax(1).view(batch, 200, 200, 16))
        result = dict(outputs)
        result.update(
            cls_score=outputs['all_cls_scores'][-1][:, current],
            refine_pts=outputs['all_refine_pts'][-1][:, current],
            forecast_points_list=fp, forecast_semantics_list=fc,
            forecast_features_list=ff, forecast_points_mask_list=masks,
            pred_trajs_list=trajectories, pred_traj=torch.cat(trajectories, 1),
            sparseworld_pretrain=self.pretrain, query_stamps=stamps,
            stamp_statistics=self.pts_bbox_head.num_stamps_all,
            update_stamp_statistics=self.training,
            initial_pred_points=_opus_decode_points(outputs['init_points'], self.pc_range).flatten(1, 2),
            current_pred_points=[_opus_decode_points(p[:, current], self.pc_range).flatten(1, 2)
                                 for p in outputs['all_refine_pts']],
            current_pred_logits=[c[:, current].flatten(1, 2) for c in outputs['all_cls_scores']],
            temporal_pred_points=[_opus_decode_points(p, self.pc_range).flatten(1, 2)
                                  for p in outputs['all_refine_pts']],
            temporal_pred_logits=[c.flatten(1, 2) for c in outputs['all_cls_scores']],
            future_pred_points=[_opus_decode_points(p, self.pc_range).flatten(1, 2) for p in fp],
            future_pred_logits=[c.flatten(1, 2) for c in fc],
            future_pred_masks=[m.flatten(1) for m in masks],
            gaussian_logits_by_horizon=logits_by_horizon,
            final_occ=grids[0].flatten(1), final_occ_grid=grids[0], future_occ_grids=grids[1:],
            sampled_label=metas['occ_label'], occ_cam_mask=metas.get('occ_cam_mask'),
            occ_lidar_mask=metas.get('occ_lidar_mask'))
        for h, grid in enumerate(grids):
            result[f'semantic_occ_{h}s'] = grid
            result[f'geo_occ_{h}s'] = torch.where(grid != 17, 0, 17)
        return result

