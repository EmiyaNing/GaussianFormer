"""Native SparseWorld OPUS head with the released checkpoint namespace."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS

from model.encoder.opus_encoder import _opus_decode_points


def get_matched_inds(maps, counts):
    """Released greedy allocation of 1040 queries to seven timestamps."""
    if maps.shape[0] != sum(counts):
        raise ValueError('timestamp score rows must equal the query count')
    num_points = maps.shape[0]
    result = torch.full((num_points,), -1, device=maps.device, dtype=torch.long)
    work = torch.cat((maps, torch.arange(num_points, device=maps.device,
                                         dtype=maps.dtype)[:, None]), dim=-1)

    def reweight(value):
        order = value[:, :-1].max(-1).values.argsort(descending=True)
        return value[order]

    work = reweight(work)
    for row in range(num_points):
        stamp = int(work[row, :-1].argmax().item())
        result[int(work[row, -1].item())] = stamp
        if int((result == stamp).sum().item()) == counts[stamp] and row + 1 < num_points:
            work[row + 1:, stamp] = 0
            work[row + 1:] = reweight(work[row + 1:])
    if (result < 0).any():
        raise RuntimeError('SparseWorld timestamp allocation did not finish')
    return result


@MODELS.register_module()
class SparseWorldStrictHead(BaseModule):
    """OPUS head ported from SparseWorld without legacy OpenMMLab APIs.

    The four geometry buffers, ``num_stamps_all``, transformer and
    ``init_points`` are deliberately registered in the same order/names as the
    official model. Runtime-only expanded-grid constants remain ordinary
    attributes so strict checkpoint loading has no local-only keys.
    """

    def __init__(self, num_classes=17, in_channels=256, num_query=720,
                 num_fu_query=(60, 60, 60, 60, 40, 40), num_fu_frames=6,
                 transformer=None, pc_range=(-40., -40., -1., 40., 40., 5.4),
                 empty_label=17, voxel_size=(.4, .4, .4), is_pretrain=False,
                 train_cfg=None, test_cfg=None, init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        self.num_query = int(num_query)
        self.num_fu_query = list(num_fu_query)
        self.num_fu_frames = int(num_fu_frames)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.empty_label = int(empty_label)
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}
        self.pretrain = bool(is_pretrain)
        self.transformer = MODELS.build(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims

        pc_range_tensor = torch.tensor(pc_range, dtype=torch.float32)
        scene_size = pc_range_tensor[3:] - pc_range_tensor[:3]
        voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32)
        self.register_buffer('pc_range', pc_range_tensor)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size_tensor)
        self.register_buffer('voxel_num', (scene_size / voxel_size_tensor).long())
        total_queries = self.num_query + sum(self.num_fu_query)
        self.register_buffer('num_stamps_all', torch.ones(
            total_queries, self.num_fu_frames + 1, dtype=torch.long))
        self.init_points = torch.nn.Embedding(total_queries, 3)
        torch.nn.init.uniform_(self.init_points.weight[:, 0], 0., 1.1)
        torch.nn.init.uniform_(self.init_points.weight[:, 1:], 0., 1.)

        # SparseWorld temporal-pretraining target grid. Not checkpoint state.
        self.scene_size_new = (160., 120., 9.6)
        self.pc_range_new = (-80., -60., -3., 80., 60., 6.6)
        self.voxel_num_new = (400, 300, 24)
        self.ind_stamps_all = None
        self.points_scale = None

    def init_weights(self):
        self.transformer.init_weights()

    @torch.no_grad()
    def update_stamp_assignment(self, force=False):
        if self.ind_stamps_all is None or force or self.pretrain:
            frequencies = self.num_stamps_all.float()
            frequencies = frequencies / frequencies.sum(-1, keepdim=True).clamp_min(1.)
            self.ind_stamps_all = get_matched_inds(
                frequencies, [self.num_query] + self.num_fu_query)
            if not self.training:
                self.reset_mask()
        return self.ind_stamps_all

    @torch.no_grad()
    def reset_mask(self):
        stamps = self.update_stamp_assignment() if self.ind_stamps_all is None else self.ind_stamps_all
        mask = self.init_points.weight.new_zeros(len(stamps), len(stamps))
        for stamp in range(self.num_fu_frames):
            rows = (stamps == stamp).nonzero(as_tuple=True)[0]
            cols = (stamps > stamp).nonzero(as_tuple=True)[0]
            if rows.numel() and cols.numel():
                grid_row, grid_col = torch.meshgrid(rows, cols, indexing='ij')
                mask[grid_row, grid_col] = -1e5
        for layer in self.transformer.decoder.decoder_layers:
            layer.self_attn.ind_mask = mask

    def forward(self, mlvl_feats, metas, points_scale=None):
        batch = mlvl_feats[0].shape[0]
        queries = self.num_query + sum(self.num_fu_query)
        initial = self.init_points.weight[None, :, None].repeat(batch, 1, 1, 1)
        scale = self.points_scale if points_scale is None else points_scale
        if scale is not None:
            initial = initial * scale.unsqueeze(1)
        query = initial.new_zeros(batch, queries, self.embed_dims)
        self.update_stamp_assignment(force=self.pretrain)
        query, scores, points = self.transformer(
            initial, query, mlvl_feats,
            metas['projection_mat'].to(initial), metas['image_wh'].to(initial))
        return dict(init_points=initial, all_cls_scores=scores,
                    all_refine_pts=points, query_feat=query)

    def discretize(self, points, clip=True):
        indices = torch.floor((points - self.pc_range[:3]) / self.voxel_size)
        if clip:
            indices = torch.minimum(torch.maximum(indices, indices.new_zeros(3)),
                                    (self.voxel_num - 1).to(indices))
        return indices.long()

    def get_occ(self, pred_dicts, expand_range=False, thre1=.1, thre2=.1):
        """Released max-score point rasterizer and morphological closing."""
        pc_range = (self.pc_range.new_tensor(self.pc_range_new) if expand_range
                    else self.pc_range)
        voxel_num = (self.voxel_num.new_tensor(self.voxel_num_new) if expand_range
                     else self.voxel_num)
        all_scores = pred_dicts['cls_scores'].sigmoid().clone()
        all_points = pred_dicts['refine_pts']
        for class_index, decay in ((15, thre1), (16, thre2)):
            if decay is None:
                continue
            class_mask = all_scores.argmax(-1) == class_index
            distance = torch.norm(all_points - all_points.mean(2, keepdim=True), dim=-1)
            if class_mask.any():
                all_scores[class_mask] *= torch.clamp(
                    decay / distance[class_mask].clamp_min(1e-12), max=1.)[:, None]

        threshold = all_scores.new_tensor(self.test_cfg.get(
            'score_thr', [.35] * 15 + [.25, .3]))
        center_threshold = float(self.test_cfg.get('ctr_dist_thr', 3.))
        outputs = []
        for batch_index in range(all_points.shape[0]):
            points = _opus_decode_points(all_points[batch_index], self.pc_range)
            scores = all_scores[batch_index]
            center_distance = torch.norm(points - points.mean(1, keepdim=True), dim=-1)
            best_score, best_class = scores.max(-1)
            keep = (center_distance < center_threshold) & (best_score > threshold[best_class])
            points, scores = points[keep], scores[keep]
            indices = torch.floor((points - pc_range[:3]) / self.voxel_size).long()
            valid = ((indices >= 0) & (indices < voxel_num)).all(-1)
            indices, scores = indices[valid], scores[valid]

            dense = scores.new_zeros(*voxel_num.tolist(), self.num_classes)
            if indices.numel():
                flat = (indices[:, 0] * voxel_num[1] * voxel_num[2] +
                        indices[:, 1] * voxel_num[2] + indices[:, 2])
                reduced = scores.new_zeros(int(voxel_num.prod()), self.num_classes)
                reduced.scatter_reduce_(0, flat[:, None].expand_as(scores), scores,
                                        reduce='amax', include_self=True)
                dense = reduced.view(*voxel_num.tolist(), self.num_classes)
            volume = dense.permute(3, 0, 1, 2).unsqueeze(0)
            if self.test_cfg.get('padding', True):
                closed = -F.max_pool3d(-F.max_pool3d(volume, 3, 1, 1), 3, 1, 1)
                original_class = volume.argmax(1)
                original = (volume.max(1).values > threshold[original_class]).expand_as(closed)
                closed[original] = volume[original]
                volume = closed
            dense = volume[0].permute(1, 2, 3, 0)
            occupied = (dense > threshold).any(-1)
            prediction = torch.full(voxel_num.tolist(), self.empty_label,
                                    device=dense.device, dtype=torch.long)
            prediction[occupied] = dense[occupied].argmax(-1)
            outputs.append(prediction)
        return torch.stack(outputs)
