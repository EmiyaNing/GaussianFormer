import torch
from torch import nn
from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from .opus_rasterizer import OPUSRasterizer


@MODELS.register_module()
class OPUSHead(BaseTaskHead):
    def __init__(self, embed_dims=128, num_classes=17,
                 point_multipliers=(1, 4, 16, 32, 64, 128),
                 pc_range=(-40., -40., -1., 40., 40., 5.4), grid_size=0.4,
                 grid_shape=(200, 200, 16), empty_label=17,
                 score_threshold=0.5, center_distance_threshold=3.0,
                 padding=True, decoder_outputs_logits=False, init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.point_multipliers = tuple(point_multipliers)
        self.pc_range = tuple(pc_range)
        self.score_threshold = score_threshold
        self.center_distance_threshold = center_distance_threshold
        self.decoder_outputs_logits = decoder_outputs_logits
        if decoder_outputs_logits:
            # Official V1 owns both cls/reg branches inside decoder layers.
            # Do not register unused duplicate heads in the strict path.
            self.offset_heads = nn.ModuleList()
            self.class_heads = nn.ModuleList()
        else:
            self.offset_heads = nn.ModuleList([
                nn.Linear(embed_dims, multiplier * 3) for multiplier in self.point_multipliers
            ])
            self.class_heads = nn.ModuleList([
                nn.Linear(embed_dims, multiplier * num_classes)
                for multiplier in self.point_multipliers
            ])
        self.rasterizer = OPUSRasterizer(
            pc_range, grid_size, grid_shape, empty_label, score_threshold,
            center_distance_threshold, padding)

    def _to_world(self, points):
        lower = points.new_tensor(self.pc_range[:3])
        upper = points.new_tensor(self.pc_range[3:])
        return points * (upper - lower) + lower

    @staticmethod
    def _flatten_voxel_tensor(value, keep_last_dim=False):
        if value is None:
            return None
        return value.flatten(1, -2) if keep_last_dim else value.flatten(1)

    def forward(self, representation, metas=None, **kwargs):
        if len(representation) != len(self.point_multipliers):
            raise ValueError('point_multipliers must contain one entry per decoder layer')
        pred_points, pred_logits, pred_valid_masks = [], [], []
        for stage, (state, multiplier) in enumerate(zip(representation, self.point_multipliers)):
            features, base_points = state['query_features'], state['query_points']
            batch_size, queries, _ = features.shape
            if base_points.ndim == 4:
                if base_points.shape[2] != multiplier:
                    raise ValueError('OPUS coarse-to-fine point count does not match point_multipliers')
                points = base_points
            else:
                offset_head = self.offset_heads[stage]
                offsets = torch.tanh(offset_head(features).view(batch_size, queries, multiplier, 3))
                points = (base_points.unsqueeze(2) + offsets * 0.08).clamp(0.0, 1.0)
            if self.decoder_outputs_logits:
                logits = state['opus_logits']
                if logits.shape != (batch_size, queries, multiplier, self.num_classes):
                    raise ValueError('strict OPUS decoder logits have an unexpected shape')
            else:
                class_head = self.class_heads[stage]
                logits = class_head(features).view(batch_size, queries, multiplier, self.num_classes)
            pred_points.append(self._to_world(points.flatten(1, 2)))
            pred_logits.append(logits.flatten(1, 2))
            query_valid_mask = state.get('query_valid_mask')
            if query_valid_mask is None:
                query_valid_mask = torch.ones((batch_size, queries), device=features.device,
                                              dtype=torch.bool)
            pred_valid_masks.append(
                query_valid_mask.unsqueeze(-1).expand(-1, -1, multiplier).flatten(1))

        # Rasterization is an eval/visualization adapter and must not retain a
        # large, unused autograd graph during OPUS set-loss training.
        final_occ = self.rasterizer(
            pred_points[-1].detach(), pred_logits[-1].detach(), self.score_threshold,
            pred_valid_masks[-1], group_size=self.point_multipliers[-1],
            center_distance_threshold=self.center_distance_threshold)
        # Keep the same [B, V] metric contract as GaussianHead. The loss uses
        # raw values from metas directly, so this does not alter supervision.
        sampled_xyz = self._flatten_voxel_tensor(metas.get('occ_xyz'), keep_last_dim=True)
        sampled_label = self._flatten_voxel_tensor(metas.get('occ_label'))
        # Keep occupancy masks in their dataset layout.  The evaluator selects
        # an individual batch item and flattens it beside ``final_occ``; doing
        # it here loses the batch/layout distinction for tail batches.
        occ_mask = metas.get('occ_mask')
        occ_cam_mask = metas.get('occ_cam_mask')
        occ_lidar_mask = metas.get('occ_lidar_mask')
        occ_nonempty_mask = metas.get('occ_nonempty_mask')
        occ_loss_mask = metas.get('occ_loss_mask')
        result = {
            'opus_pred_points': pred_points,
            'opus_pred_logits': pred_logits,
            'opus_point_valid_masks': pred_valid_masks,
            'opus_points': pred_points[-1],
            'opus_point_valid_mask': pred_valid_masks[-1],
            'opus_labels': pred_logits[-1].argmax(dim=-1),
            'opus_scores': pred_logits[-1].sigmoid().amax(dim=-1),
            'final_occ': final_occ,
            'final_occ_grid': final_occ.view(final_occ.shape[0], *self.rasterizer.grid_shape),
            'sampled_xyz': sampled_xyz,
            'sampled_label': sampled_label,
            'occ_mask': occ_mask,
            'occ_cam_mask': occ_cam_mask,
            'occ_lidar_mask': occ_lidar_mask,
            'occ_nonempty_mask': occ_nonempty_mask,
            'occ_loss_mask': occ_loss_mask,
        }
        return result
