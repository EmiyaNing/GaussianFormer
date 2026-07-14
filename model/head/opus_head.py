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
                 score_threshold=0.0, init_cfg=None):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.point_multipliers = tuple(point_multipliers)
        self.pc_range = tuple(pc_range)
        self.score_threshold = score_threshold
        self.offset_heads = nn.ModuleList([
            nn.Linear(embed_dims, multiplier * 3) for multiplier in self.point_multipliers
        ])
        self.class_heads = nn.ModuleList([
            nn.Linear(embed_dims, multiplier * num_classes)
            for multiplier in self.point_multipliers
        ])
        self.rasterizer = OPUSRasterizer(pc_range, grid_size, grid_shape, empty_label)

    def _to_world(self, points):
        lower = points.new_tensor(self.pc_range[:3])
        upper = points.new_tensor(self.pc_range[3:])
        return points * (upper - lower) + lower

    def forward(self, representation, metas=None, **kwargs):
        if len(representation) != len(self.point_multipliers):
            raise ValueError('point_multipliers must contain one entry per decoder layer')
        pred_points, pred_logits = [], []
        for stage, (state, offset_head, class_head, multiplier) in enumerate(zip(
                representation, self.offset_heads, self.class_heads, self.point_multipliers)):
            features, base_points = state['query_features'], state['query_points']
            batch_size, queries, _ = features.shape
            offsets = torch.tanh(offset_head(features).view(batch_size, queries, multiplier, 3))
            points = (base_points.unsqueeze(2) + offsets * 0.08).clamp(0.0, 1.0)
            logits = class_head(features).view(batch_size, queries, multiplier, self.num_classes)
            pred_points.append(self._to_world(points.flatten(1, 2)))
            pred_logits.append(logits.flatten(1, 2))

        final_occ = self.rasterizer(pred_points[-1], pred_logits[-1], self.score_threshold)
        result = {
            'opus_pred_points': pred_points,
            'opus_pred_logits': pred_logits,
            'opus_points': pred_points[-1],
            'opus_labels': pred_logits[-1].argmax(dim=-1),
            'opus_scores': pred_logits[-1].softmax(dim=-1).amax(dim=-1),
            'final_occ': final_occ,
            'final_occ_grid': final_occ.view(final_occ.shape[0], *self.rasterizer.grid_shape),
            'sampled_xyz': metas.get('occ_xyz') if metas is not None else None,
            'sampled_label': metas.get('occ_label') if metas is not None else None,
            'occ_cam_mask': metas.get('occ_cam_mask') if metas is not None else None,
            'occ_lidar_mask': metas.get('occ_lidar_mask') if metas is not None else None,
            'occ_nonempty_mask': metas.get('occ_nonempty_mask') if metas is not None else None,
            'occ_loss_mask': metas.get('occ_loss_mask') if metas is not None else None,
        }
        return result
