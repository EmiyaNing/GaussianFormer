"""Strict OPUSv2 head ported from the official implementation.

Only the outer data/result dictionaries are adapted to GaussianFormer's runner.
Module names and tensor operations intentionally follow the official project so
that its checkpoints require a prefix-only state-dict conversion.
"""
import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule
from mmengine.model import bias_init_with_prob
from mmengine.registry import MODELS
from spconv.pytorch import (SparseConv3d, SparseConvTensor, SparseSequential,
                            SubMConv3d)
from torch_scatter import scatter_max

from model.ops.opus_msmv_sampling import MSMV_CUDA, msmv_sampling


def decode_points(points, pc_range):
    return points * (pc_range[3:] - pc_range[:3]) + pc_range[:3]


def encode_points(points, pc_range):
    return (points - pc_range[:3]) / (pc_range[3:] - pc_range[:3])


def sampling_4d(sample_points, mlvl_feats, scale_weights, projection,
                image_wh, num_views=6, eps=1e-5):
    """Official one-valid-view, multi-frame MSMV sampling tensor flow."""
    batch, queries, frames, groups, points, _ = sample_points.shape
    if projection.shape[1] != frames * num_views:
        raise ValueError('projection count must equal num_frames * num_views')
    sample_points = sample_points.reshape(batch, queries, frames, groups * points, 3)
    projection = projection[:, :, None, None].expand(
        batch, frames * num_views, queries, groups * points, 4, 4
    ).reshape(batch, frames, num_views, queries, groups * points, 4, 4)
    homogeneous = torch.cat([sample_points, torch.ones_like(sample_points[..., :1])], -1)
    homogeneous = homogeneous[:, :, None, ..., None].expand(
        batch, queries, num_views, frames, groups * points, 4, 1).transpose(1, 3)
    # Keep geometry in FP32 under the runner's global AMP context.  In FP16,
    # the denominator's eps**2 underflows in DivBackward and produces NaNs for
    # points behind a camera, even though the subsequent validity mask rejects
    # those samples.  The official MSMV kernel consumes FP32 locations too.
    with torch.cuda.amp.autocast(enabled=False):
        projected = torch.matmul(
            projection.float(), homogeneous.float()).squeeze(-1)
        depth = projected[..., 2:3]
        uv = projected[..., :2] / torch.maximum(
            depth, torch.full_like(depth, eps))
        image_wh = image_wh.float().reshape(batch, frames, num_views, 2)
        uv = uv / image_wh[:, :, :, None, None]
    valid = ((depth > eps) & (uv[..., 0:1] > 0.) & (uv[..., 0:1] < 1.) &
             (uv[..., 1:2] > 0.) & (uv[..., 1:2] < 1.)).squeeze(-1).float()
    valid = valid.permute(0, 1, 3, 4, 2)
    uv = uv.permute(0, 1, 3, 4, 2, 5)
    selected = valid.argmax(-1, keepdim=True)
    ib = torch.arange(batch, device=uv.device).view(batch, 1, 1, 1, 1)
    it = torch.arange(frames, device=uv.device).view(1, frames, 1, 1, 1)
    iq = torch.arange(queries, device=uv.device).view(1, 1, queries, 1, 1)
    ip = torch.arange(groups * points, device=uv.device).view(1, 1, 1, groups * points, 1)
    uv = uv[ib, it, iq, ip, selected, :]
    uv = torch.cat([uv, selected[..., None].to(uv.dtype) / (num_views - 1)], -1)
    locations = uv.reshape(batch, frames, queries, groups, points, 1, 3)
    locations = locations.permute(0, 1, 3, 2, 4, 5, 6).reshape(
        batch * frames * groups, queries, points, 3).contiguous()
    weights = scale_weights.reshape(batch, queries, groups, frames, points, -1)
    weights = weights.permute(0, 2, 3, 1, 4, 5).reshape(
        batch * groups * frames, queries, points, -1).contiguous()
    output = msmv_sampling(mlvl_feats, locations, weights)
    channels = output.shape[2]
    return output.reshape(batch, frames, groups, queries, channels, points).permute(
        0, 3, 2, 1, 5, 4).flatten(3, 4)


class OPUSSelfAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=(), init_cfg=None):
        super().__init__(init_cfg)
        self.register_buffer('_pc_range', torch.tensor(pc_range), persistent=False)
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0., 2.)

    def forward(self, query_points, query_feat):
        points = decode_points(query_points, self._pc_range.to(query_points)).mean(2)
        distance = -torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        tau = self.gen_tau(query_feat).permute(0, 2, 1)
        return self.attention(query_feat, attn_mask=(distance[:, None] * tau[..., None]).flatten(0, 1))


class OPUSSampling(BaseModule):
    def __init__(self, embed_dims=256, num_frames=8, num_views=6, num_groups=4,
                 num_points=4, num_levels=4, pc_range=(), init_cfg=None):
        super().__init__(init_cfg)
        self.num_frames, self.num_views = num_frames, num_views
        self.num_groups, self.num_points, self.num_levels = num_groups, num_points, num_levels
        self.register_buffer('_pc_range', torch.tensor(pc_range), persistent=False)
        self.sampling_prototype = nn.Embedding(num_groups * num_points, 3)
        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        nn.init.normal_(self.sampling_prototype.weight, mean=0, std=1)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(self.sampling_offset.bias.view(-1, 3), -.5, .5)

    def forward(self, query_points, query_feat, mlvl_feats, metas):
        batch, queries = query_points.shape[:2]
        decoded = decode_points(query_points, self._pc_range.to(query_points))
        if decoded.shape[2] == 1:
            center, scale = decoded, torch.ones_like(decoded)
        else:
            center, scale = decoded.mean(2, keepdim=True), decoded.std(2, keepdim=True)
        offsets = self.sampling_offset(query_feat).view(batch, queries, -1, 3)
        prototype = self.sampling_prototype.weight[None, None].expand(batch, queries, -1, -1)
        points = center + prototype * scale + offsets
        points = points.view(batch, queries, 1, self.num_groups, self.num_points, 3).expand(
            batch, queries, self.num_frames, self.num_groups, self.num_points, 3)
        weights = self.scale_weights(query_feat).view(
            batch, queries, self.num_groups, 1, self.num_points, self.num_levels).softmax(-1)
        weights = weights.expand(batch, queries, self.num_groups, self.num_frames,
                                 self.num_points, self.num_levels)
        return sampling_4d(points, mlvl_feats, weights,
                           metas['projection_mat'].to(query_feat),
                           metas['image_wh'].to(query_feat), self.num_views)


class AdaptiveMixing(nn.Module):
    def __init__(self, in_dim, in_points, n_groups=4, out_points=32):
        super().__init__()
        self.in_points, self.n_groups, self.out_points = in_points, n_groups, out_points
        self.eff_in_dim = in_dim // n_groups
        self.m_parameters = self.eff_in_dim * self.eff_in_dim
        self.s_parameters = in_points * out_points
        self.parameter_generator = nn.Linear(
            in_dim, n_groups * (self.m_parameters + self.s_parameters))
        self.out_proj = nn.Linear(self.eff_in_dim * out_points * n_groups, in_dim)
        self.act = nn.ReLU(inplace=True)

    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def forward(self, x, query):
        batch, queries, groups, points, channels = x.shape
        params = self.parameter_generator(query).reshape(batch * queries, groups, -1)
        matrix, spatial = params.split([self.m_parameters, self.s_parameters], 2)
        matrix = matrix.reshape(batch * queries, groups, channels, channels)
        spatial = spatial.reshape(batch * queries, groups, self.out_points, self.in_points)
        output = x.reshape(batch * queries, groups, points, channels)
        output = self.act(F.layer_norm(torch.matmul(output, matrix), [points, channels]))
        output = self.act(F.layer_norm(torch.matmul(spatial, output), [self.out_points, channels]))
        return query + self.out_proj(output.reshape(batch, queries, -1))


class OPUSTransformerDecoderLayer(BaseModule):
    def __init__(self, embed_dims, num_frames, num_views, num_points, num_levels,
                 num_groups, num_pt_channels, num_refines, last_refines,
                 last_layer=False, scale=1., pc_range=(), init_cfg=None):
        super().__init__(init_cfg)
        self.num_refines, self.last_layer, self.scale = num_refines, last_layer, scale
        self.register_buffer('_pc_range', torch.tensor(pc_range), persistent=False)
        self.position_encoder = nn.Sequential(
            nn.Linear(3 * last_refines, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(True),
            nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(True))
        self.self_attn = OPUSSelfAttention(embed_dims, 8, .1, pc_range)
        self.sampling = OPUSSampling(embed_dims, num_frames, num_views, num_groups,
                                     num_points, num_levels, pc_range)
        self.mixing = AdaptiveMixing(embed_dims, num_points * num_frames, num_groups, 32)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=.1)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(embed_dims) for _ in range(3))
        cls = []
        for _ in range(2):
            cls += [nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(True)]
        cls.append(nn.Linear(embed_dims, num_pt_channels * num_refines))
        self.cls_branch = nn.Sequential(*cls)
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(True),
            nn.Linear(embed_dims, 3 * num_refines))

    def init_weights(self):
        self.self_attn.init_weights(); self.sampling.init_weights(); self.mixing.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, metas):
        query_feat = query_feat + self.position_encoder(query_points.flatten(2, 3))
        query_feat = self.norm1(self.mixing(
            self.sampling(query_points, query_feat, mlvl_feats, metas), query_feat))
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))
        batch, queries = query_points.shape[:2]
        proposal = decode_points(query_points, self._pc_range.to(query_points)).mean(2, keepdim=True)
        offsets = self.scale * self.reg_branch(query_feat).view(batch, queries, self.num_refines, 3)
        refined = encode_points(proposal + offsets, self._pc_range.to(query_points))
        point_feat = None
        if self.training or self.last_layer:
            point_feat = self.cls_branch(query_feat).view(batch, queries, self.num_refines, -1)
        return query_feat, point_feat, refined


class OPUSTransformerDecoder(BaseModule):
    def __init__(self, embed_dims, num_frames, num_views, num_points, num_layers,
                 num_levels, num_refines, num_groups, num_pt_channels, scales,
                 pc_range, init_cfg=None):
        super().__init__(init_cfg)
        if len(scales) == 1: scales = list(scales) * num_layers
        before = [1] + list(num_refines)
        self.num_frames, self.num_views, self.num_groups = num_frames, num_views, num_groups
        self.decoder_layers = nn.ModuleList([
            OPUSTransformerDecoderLayer(
                embed_dims, num_frames, num_views, num_points, num_levels, num_groups,
                num_pt_channels, num_refines[i], before[i], i == num_layers - 1,
                scales[i], pc_range) for i in range(num_layers)])

    def init_weights(self):
        for layer in self.decoder_layers: layer.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, metas):
        grouped = []
        batch = query_feat.shape[0]
        for feature in mlvl_feats:
            _, cameras, channels, height, width = feature.shape
            if cameras != self.num_frames * self.num_views:
                raise ValueError('feature camera dimension must equal T * N')
            group_channels = channels // self.num_groups
            feature = feature.reshape(batch, self.num_frames, self.num_views,
                                      self.num_groups, group_channels, height, width)
            if MSMV_CUDA and feature.is_cuda:
                feature = feature.permute(0, 1, 3, 2, 5, 6, 4).reshape(
                    batch * self.num_frames * self.num_groups,
                    self.num_views, height, width, group_channels)
            else:
                # Local wrapper has a channel-last fallback too.
                feature = feature.permute(0, 1, 3, 2, 5, 6, 4).reshape(
                    batch * self.num_frames * self.num_groups,
                    self.num_views, height, width, group_channels)
            grouped.append(feature.contiguous())
        point_feats, refined_points = [], []
        for layer in self.decoder_layers:
            query_points = query_points.detach()
            query_feat, point_feat, query_points = layer(
                query_points, query_feat, grouped, metas)
            point_feats.append(None if point_feat is None else torch.nan_to_num(point_feat))
            refined_points.append(torch.nan_to_num(query_points))
        return point_feats, refined_points


class OPUSV2Transformer(BaseModule):
    def __init__(self, embed_dims=256, num_frames=8, num_views=6, num_points=4,
                 num_layers=5, num_levels=4, num_groups=4,
                 num_refines=(8, 16, 32, 64, 128), num_pt_channels=32,
                 scales=(.5,), pc_range=(), init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims, self.num_refines = embed_dims, tuple(num_refines)
        self.num_pt_channels, self.num_layers = num_pt_channels, num_layers
        self.decoder = OPUSTransformerDecoder(
            embed_dims, num_frames, num_views, num_points, num_layers, num_levels,
            num_refines, num_groups, num_pt_channels, scales, pc_range)

    def init_weights(self): self.decoder.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, metas):
        return self.decoder(query_points, query_feat, mlvl_feats, metas)


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe, self.use_norm = last_layer, use_norm
        if not last_layer: out_channels //= 2
        self.linear = nn.Linear(in_channels, out_channels, bias=not use_norm)
        if use_norm: self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=.01)
        self.relu = nn.ReLU()

    def forward(self, inputs, inverse):
        output = self.relu(self.norm(self.linear(inputs)) if self.use_norm else self.linear(inputs))
        maximum = scatter_max(output, inverse, dim=0)[0]
        return maximum if self.last_vfe else torch.cat([output, maximum[inverse]], dim=1)


class Densifier(BaseModule):
    def __init__(self, in_channels, pfn_channels, num_classes, pc_range, voxel_num):
        super().__init__()
        self.pfn_layers = nn.ModuleList([
            PFNLayer(in_channels + 6 if i == 0 else channel, channel,
                     last_layer=i == len(pfn_channels) - 1)
            for i, channel in enumerate(pfn_channels)])
        self.cls_branch = SparseSequential(
            SparseConv3d(pfn_channels[-1], pfn_channels[-1], 3, 1, 1, indice_key='densifier1'),
            nn.ReLU(inplace=True),
            SubMConv3d(pfn_channels[-1], num_classes, 3, 1, 1, indice_key='densifier2'))
        self.register_buffer('pc_range', pc_range.clone())
        self.register_buffer('voxel_num', voxel_num.clone())
        self.scale_xyz = int((voxel_num[0] * voxel_num[1] * voxel_num[2]).item())
        self.scale_yz = int((voxel_num[1] * voxel_num[2]).item())
        self.scale_z = int(voxel_num[2].item())

    def init_weights(self): nn.init.constant_(self.cls_branch[-1].bias, bias_init_with_prob(.01))

    def forward(self, point_feats, refine_points):
        if point_feats is None: return None, None
        batch, queries, points, _ = refine_points.shape
        refine_points = decode_points(refine_points, self.pc_range)
        encoded = encode_points(refine_points, self.pc_range)
        coords = torch.floor(encoded * self.voxel_num.to(encoded.dtype))
        coords = torch.minimum(torch.maximum(coords, torch.zeros_like(coords)),
                               (self.voxel_num - 1).to(coords)).long()
        centers = decode_points((coords + .5) / self.voxel_num.to(coords), self.pc_range)
        batch_index = torch.arange(batch, device=coords.device).view(batch, 1, 1, 1).expand(
            batch, queries, points, 1)
        coords = torch.cat([batch_index, coords], -1)
        linear = (coords[..., 0] * self.scale_xyz + coords[..., 1] * self.scale_yz +
                  coords[..., 2] * self.scale_z + coords[..., 3]).reshape(-1)
        point_feats = torch.cat([refine_points - centers,
                                 refine_points - refine_points.mean(2, keepdim=True), point_feats], -1)
        unique, inverse = linear.unique(return_inverse=True)
        point_feats = point_feats.reshape(-1, point_feats.shape[-1])
        for layer in self.pfn_layers: point_feats = layer(point_feats, inverse)
        sparse_coords = torch.stack([unique // self.scale_xyz,
            (unique % self.scale_xyz) // self.scale_yz,
            (unique % self.scale_yz) // self.scale_z, unique % self.scale_z], 1)
        result = self.cls_branch(SparseConvTensor(
            point_feats, sparse_coords.int(), self.voxel_num.tolist(), batch))
        return result.features, result.indices.long()


@MODELS.register_module()
class OfficialOPUSV2Head(BaseModule):
    def __init__(self, num_classes=17, in_channels=256, num_query=600,
                 pc_range=(-40., -40., -1., 40., 40., 5.4), voxel_size=(.4, .4, .4),
                 pfn_channels=(64, 64), empty_label=17, score_thr=.25,
                 transformer=None, init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        self.num_query, self.num_classes = num_query, num_classes
        self.empty_label, self.score_thr = empty_label, score_thr
        pc_range = torch.tensor(pc_range)
        voxel_size = torch.tensor(voxel_size)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_num = (scene_size / voxel_size).long()
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)
        transformer = dict(transformer or {})
        transformer.setdefault('pc_range', pc_range.tolist())
        self.transformer = OPUSV2Transformer(**transformer)
        self.densifiers = nn.ModuleList([
            Densifier(self.transformer.num_pt_channels, pfn_channels, num_classes,
                      self.pc_range, self.voxel_num)
            for _ in range(self.transformer.num_layers)])
        self.init_points = nn.Embedding(num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0., 1.)

    def init_weights(self):
        self.transformer.init_weights()
        for densifier in self.densifiers: densifier.init_weights()

    def _dense(self, logits, coords, batch_size):
        total = int(self.voxel_num.prod().item())
        dense = torch.full((batch_size, total), self.empty_label,
                           device=self.pc_range.device, dtype=torch.long)
        if logits is None or coords is None or coords.numel() == 0: return dense
        scores, labels = logits.sigmoid().max(-1)
        keep = scores > self.score_thr
        coords, labels = coords[keep], labels[keep]
        flat = ((coords[:, 1] * self.voxel_num[1] + coords[:, 2]) *
                self.voxel_num[2] + coords[:, 3])
        dense[coords[:, 0], flat] = labels
        return dense

    def forward(self, ms_img_feats, metas, **kwargs):
        batch = ms_img_feats[0].shape[0]
        initial = self.init_points.weight[None, :, None].repeat(batch, 1, 1, 1)
        query = initial.new_zeros(batch, self.num_query, self.transformer.embed_dims)
        point_feats, refined = self.transformer(initial, query, ms_img_feats, metas)
        logits, coords = [], []
        for densifier, feat, points in zip(self.densifiers, point_feats, refined):
            score, coord = densifier(feat, points); logits.append(score); coords.append(coord)
        final_occ = self._dense(logits[-1], coords[-1], batch)
        flatten = lambda value: None if value is None else value.flatten(1)
        return dict(init_points=initial, all_refine_pts=refined,
                    all_cls_scores=logits, all_voxel_coors=coords,
                    final_occ=final_occ,
                    final_occ_grid=final_occ.view(batch, *self.voxel_num.tolist()),
                    sampled_label=flatten(metas.get('occ_label')),
                    sampled_xyz=(None if metas.get('occ_xyz') is None else
                                 metas['occ_xyz'].flatten(1, -2)),
                    occ_cam_mask=metas.get('occ_cam_mask'),
                    occ_lidar_mask=metas.get('occ_lidar_mask'),
                    occ_loss_mask=metas.get('occ_loss_mask'))
