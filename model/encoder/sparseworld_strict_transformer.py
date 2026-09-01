"""Native SparseWorld OPUS transformer for the current MMEngine runtime.

The module names intentionally mirror the released SparseWorld checkpoint.
Only framework glue differs from the MMCV-1 implementation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS

from .opus_encoder import (_official_sampling_4d, _opus_decode_points,
                           _opus_encode_points)


class SparseWorldMultiheadAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1,
                 batch_first=True, init_cfg=None):
        super().__init__(init_cfg)
        self.batch_first = batch_first
        # Keep the official ``attention.attn.*`` state namespace.
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout,
                                          batch_first=batch_first)
        self.proj_drop = nn.Dropout(0.0)
        self.dropout_layer = nn.Dropout(0.0)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_pos=None, attn_mask=None, **kwargs):
        key = query if key is None else key
        value = key if value is None else value
        identity = query if identity is None else identity
        query_input = query if query_pos is None else query + query_pos
        key_input = key if key_pos is None else key + key_pos
        output = self.attn(query_input, key_input, value, attn_mask=attn_mask,
                           need_weights=False)[0]
        return identity + self.dropout_layer(self.proj_drop(output)), None


class SparseWorldFFN(BaseModule):
    """MMCV-1 FFN layout and residual semantics with identical state keys."""
    def __init__(self, embed_dims=256, feedforward_channels=512, dropout=0.1,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, feedforward_channels),
                          nn.ReLU(inplace=True), nn.Dropout(dropout)),
            nn.Linear(feedforward_channels, embed_dims), nn.Dropout(dropout))

    def forward(self, x, identity=None):
        return (x if identity is None else identity) + self.layers(x)


class SparseWorldSelfAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1,
                 pc_range=(), init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = tuple(pc_range)
        self.num_heads = num_heads
        self.attention = SparseWorldMultiheadAttention(
            embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        self.ind_mask = None

    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    @torch.no_grad()
    def calc_points_dists(self, points):
        world = _opus_decode_points(points, self.pc_range).mean(dim=2)
        return -torch.cdist(world, world)

    def forward(self, query_points, query_feat):
        distance = self.calc_points_dists(query_points)
        tau = self.gen_tau(query_feat).permute(0, 2, 1)
        mask = distance[:, None] * tau[..., None]
        if self.ind_mask is not None:
            mask = mask + self.ind_mask.to(mask)[None, None]
        return self.attention(query_feat, query_feat, query_feat,
                              attn_mask=mask.flatten(0, 1))[0]


class SparseWorldCrossAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1,
                 pc_range=(), init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = tuple(pc_range)
        self.attention = SparseWorldMultiheadAttention(
            embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        self.pos_embed = nn.Linear(3, embed_dims)

    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    @torch.no_grad()
    def calc_points_dists(self, query_points, key_points):
        query = _opus_decode_points(query_points, self.pc_range)
        key = _opus_decode_points(key_points, self.pc_range).mean(dim=2)
        return -torch.norm(query.unsqueeze(-2) - key.unsqueeze(-3), dim=-1)

    def forward(self, query_points, query_feat, key_points, key_feat):
        distance = torch.exp(self.calc_points_dists(query_points, key_points) / 40.0)
        tau = F.softplus(self.gen_tau(key_feat)).permute(0, 2, 1)
        mask = (distance[:, None] * tau[:, :, None]).flatten(0, 1)
        query_pos = self.pos_embed(query_points * 0.0)
        key_pos = self.pos_embed(_opus_decode_points(
            key_points, self.pc_range).mean(dim=2))
        return self.attention(query_feat, key_feat, key_feat, attn_mask=mask,
                              query_pos=query_pos, key_pos=key_pos)


class SparseWorldSampling(BaseModule):
    def __init__(self, embed_dims=256, num_frames=5, num_views=6, num_groups=4,
                 num_points=4, num_levels=4, pc_range=(), init_cfg=None):
        super().__init__(init_cfg)
        self.num_frames, self.num_views = num_frames, num_views
        self.num_groups, self.num_points = num_groups, num_points
        self.num_levels, self.pc_range = num_levels, tuple(pc_range)
        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(self.sampling_offset.bias.view(-1, 3), -0.5, 0.5)

    def forward(self, query_points, query_feat, mlvl_feats, projection, image_wh):
        batch, queries = query_points.shape[:2]
        world = _opus_decode_points(query_points, self.pc_range)
        if world.shape[2] == 1:
            center, spread = world, torch.zeros_like(world)
        else:
            center, spread = world.mean(2, keepdim=True), world.std(2, keepdim=True)
        offsets = self.sampling_offset(query_feat).view(batch, queries, -1, 3)
        samples = (center + offsets * spread).view(
            batch, queries, self.num_groups, self.num_points, 3)
        samples = samples[:, :, None].expand(
            batch, queries, self.num_frames, self.num_groups, self.num_points, 3)
        weights = self.scale_weights(query_feat).view(
            batch, queries, self.num_groups, 1, self.num_points, self.num_levels)
        weights = weights.softmax(-1).expand(
            batch, queries, self.num_groups, self.num_frames,
            self.num_points, self.num_levels)
        sampled, _ = _official_sampling_4d(
            samples, mlvl_feats, weights, projection, image_wh, self.num_views)
        return sampled


class SparseWorldAdaptiveMixing(nn.Module):
    def __init__(self, in_dim=256, in_points=20, n_groups=4,
                 query_dim=None, out_dim=None, out_points=32):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        query_dim = in_dim if query_dim is None else query_dim
        self.in_points, self.n_groups, self.out_points = in_points, n_groups, out_points
        self.eff_in_dim, self.eff_out_dim = in_dim // n_groups, out_dim // n_groups
        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = in_points * out_points
        self.total_parameters = self.m_parameters + self.s_parameters
        self.parameter_generator = nn.Linear(
            query_dim, n_groups * self.total_parameters)
        self.out_proj = nn.Linear(
            self.eff_out_dim * out_points * n_groups, query_dim)
        self.act = nn.ReLU(inplace=True)

    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def forward(self, x, query):
        batch, queries, groups, points, channels = x.shape
        params = self.parameter_generator(query).reshape(batch * queries, groups, -1)
        output = x.reshape(batch * queries, groups, points, channels)
        matrix, spatial = params.split([self.m_parameters, self.s_parameters], 2)
        matrix = matrix.reshape(batch * queries, groups,
                                self.eff_in_dim, self.eff_out_dim)
        spatial = spatial.reshape(batch * queries, groups,
                                  self.out_points, self.in_points) 
        output = self.act(F.layer_norm(output @ matrix,
                                       [output.shape[-2], self.eff_out_dim]))
        output = self.act(F.layer_norm(spatial @ output,
                                       [self.out_points, self.eff_out_dim]))
        return query + self.out_proj(output.reshape(batch, queries, -1))


class SparseWorldDecoderLayer(BaseModule):
    def __init__(self, embed_dims, num_frames, num_views, num_points, num_levels,
                 num_classes, num_groups, num_refines, last_refines, scale,
                 pc_range, init_cfg=None):
        super().__init__(init_cfg)
        self.num_refines, self.scale, self.pc_range = num_refines, scale, tuple(pc_range)
        self.position_encoder = nn.Sequential(
            nn.Linear(3 * last_refines, embed_dims), nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True), nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims), nn.ReLU(inplace=True))
        self.self_attn = SparseWorldSelfAttention(
            embed_dims, 8, 0.1, pc_range)
        self.sampling = SparseWorldSampling(
            embed_dims, num_frames, num_views, num_groups, num_points,
            num_levels, pc_range)
        self.mixing = SparseWorldAdaptiveMixing(
            embed_dims, num_points * num_frames, num_groups, out_points=32)
        self.ffn = SparseWorldFFN(embed_dims, 512, 0.1)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(embed_dims),
                                              nn.LayerNorm(embed_dims),
                                              nn.LayerNorm(embed_dims))
        cls = []
        for _ in range(2):
            cls += [nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims),
                    nn.ReLU(inplace=True)]
        cls.append(nn.Linear(embed_dims, num_classes * num_refines))
        self.cls_branch = nn.Sequential(*cls)
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3 * num_refines))

    def init_weights(self):
        self.self_attn.init_weights(); self.sampling.init_weights(); self.mixing.init_weights()
        nn.init.constant_(self.cls_branch[-1].bias, -4.59511985013459)

    def forward(self, points, query, mlvl_feats, projection, image_wh):
        query = query + self.position_encoder(points.flatten(2, 3))
        sampled = self.sampling(points, query, mlvl_feats, projection, image_wh)
        query = self.norm1(self.mixing(sampled, query))
        query = self.norm2(self.self_attn(points, query))
        query = self.norm3(self.ffn(query))
        batch, queries = points.shape[:2]
        cls = self.cls_branch(query).view(batch, queries, self.num_refines, -1)
        delta = self.scale * self.reg_branch(query).view(
            batch, queries, self.num_refines, 3)
        proposal = _opus_decode_points(points, self.pc_range).mean(2, keepdim=True)
        points = _opus_encode_points(proposal + delta, self.pc_range)
        return query, cls, points


class SparseWorldDecoder(BaseModule):
    def __init__(self, embed_dims=256, num_frames=5, num_views=6, num_points=4,
                 num_layers=6, num_levels=4, num_classes=17, num_refines=(),
                 num_groups=4, scales=(0.5,), pc_range=(), init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        if len(scales) == 1: scales = tuple(scales) * num_layers
        previous = (1,) + tuple(num_refines[:-1])
        self.decoder_layers = nn.ModuleList([
            SparseWorldDecoderLayer(embed_dims, num_frames, num_views, num_points,
                                    num_levels, num_classes, num_groups, current,
                                    last, scale, pc_range)
            for last, current, scale in zip(previous, num_refines, scales)])

    def init_weights(self):
        for layer in self.decoder_layers: layer.init_weights()

    def forward(self, points, query, mlvl_feats, projection, image_wh):
        # Match the released channel-last MSMV feature layout exactly:
        # [B,TN,GC,H,W] -> [B*T*G,N,H,W,C].
        reshaped = []
        for feature in mlvl_feats:
            batch, _, grouped_channels, height, width = feature.shape
            channels = grouped_channels // self.decoder_layers[0].sampling.num_groups
            frames = self.decoder_layers[0].sampling.num_frames
            views = self.decoder_layers[0].sampling.num_views
            groups = self.decoder_layers[0].sampling.num_groups
            feature = feature.reshape(batch, frames, views, groups,
                                      channels, height, width)
            feature = feature.permute(0, 1, 3, 2, 5, 6, 4)
            reshaped.append(feature.reshape(batch * frames * groups, views,
                                            height, width, channels).contiguous())
        scores, refined = [], []
        for layer in self.decoder_layers:
            points = points.detach()
            query, cls, points = layer(points, query, reshaped, projection, image_wh)
            scores.append(torch.nan_to_num(cls)); refined.append(torch.nan_to_num(points))
        return query, scores, refined


@MODELS.register_module()
class SparseWorldTransformer(BaseModule):
    def __init__(self, embed_dims=256, num_frames=5, num_views=6, num_points=4,
                 num_layers=6, num_levels=4, num_classes=17, num_groups=4,
                 num_refines=(1,4,16,24,32,48), scales=(0.5,), pc_range=(),
                 init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        self.embed_dims, self.num_refines = embed_dims, tuple(num_refines)
        self.decoder = SparseWorldDecoder(
            embed_dims, num_frames, num_views, num_points, num_layers, num_levels,
            num_classes, num_refines, num_groups, scales, pc_range)

    def init_weights(self): self.decoder.init_weights()

    def forward(self, points, query, mlvl_feats, projection, image_wh):
        return self.decoder(points, query, mlvl_feats, projection, image_wh)
