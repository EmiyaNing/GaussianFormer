"""Causal all-query OPUS decoder used exclusively by SparseWorldStrict."""
import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from .opus_encoder import (_StrictOPUSSampling, _StrictAdaptiveMixing,
                           _opus_decode_points, _opus_encode_points)


class _CausalGeometrySelfAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, dropout, pc_range):
        super().__init__(); self.pc_range = tuple(pc_range)
        self.attention = nn.MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.tau = nn.Linear(embed_dims, num_heads)
        nn.init.zeros_(self.tau.weight); nn.init.uniform_(self.tau.bias, 0., 2.)

    def forward(self, points, features, stamps):
        world = _opus_decode_points(points, self.pc_range).mean(2)
        geometry = -torch.cdist(world, world)[:, None] * self.tau(features).permute(0, 2, 1)[..., None]
        # Query at timestamp t cannot observe any future timestamp > t.
        causal = (stamps[None, :] > stamps[:, None]).to(features.dtype) * -1.e5
        mask = (geometry + causal[None, None]).flatten(0, 1)
        qkv = features.transpose(0, 1)
        output, _ = F.multi_head_attention_forward(
            qkv, qkv, qkv, self.attention.embed_dim, self.attention.num_heads,
            self.attention.in_proj_weight, self.attention.in_proj_bias,
            self.attention.bias_k, self.attention.bias_v, self.attention.add_zero_attn,
            self.attention.dropout, self.attention.out_proj.weight, self.attention.out_proj.bias,
            training=self.training, key_padding_mask=None, need_weights=False, attn_mask=mask,
            use_separate_proj_weight=False, average_attn_weights=True, is_causal=False)
        return features + output.transpose(0, 1)


class _Layer(BaseModule):
    def __init__(self, embed_dims, num_frames, num_views, num_points, num_levels, num_groups,
                 num_classes, last_refine, num_refine, num_heads, feedforward_channels, dropout, scale, pc_range):
        super().__init__(); self.num_refine, self.scale, self.pc_range = num_refine, scale, tuple(pc_range)
        self.position = nn.Sequential(nn.Linear(3 * last_refine, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(True), nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(True))
        self.sampling = _StrictOPUSSampling(embed_dims, num_frames, num_views, num_groups, num_points, num_levels, pc_range)
        self.mixing = _StrictAdaptiveMixing(embed_dims, num_frames * num_points, num_groups, 32)
        self.self_attn = _CausalGeometrySelfAttention(embed_dims, num_heads, dropout, pc_range)
        self.ffn = nn.Sequential(nn.Linear(embed_dims, feedforward_channels), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(feedforward_channels, embed_dims), nn.Dropout(dropout))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(embed_dims), nn.LayerNorm(embed_dims), nn.LayerNorm(embed_dims)
        cls=[]
        for _ in range(2): cls += [nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(True)]
        self.cls = nn.Sequential(*cls, nn.Linear(embed_dims, num_classes * num_refine))
        self.reg = nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.ReLU(True), nn.Linear(embed_dims, embed_dims), nn.ReLU(True), nn.Linear(embed_dims, 3 * num_refine))
        nn.init.constant_(self.cls[-1].bias, -4.59511985013459)

    def forward(self, points, features, image_features, metas, stamps):
        features = features + self.position(points.flatten(2))
        sampled, visible = self.sampling(points, features, image_features, metas)
        features = self.norm1(self.mixing(sampled, features))
        features = self.norm2(self.self_attn(points, features, stamps))
        features = self.norm3(features + self.ffn(features))
        b, q = features.shape[:2]
        logits = self.cls(features).view(b, q, self.num_refine, -1)
        delta = self.scale * self.reg(features).view(b, q, self.num_refine, 3)
        refined = _opus_encode_points(_opus_decode_points(points, self.pc_range).mean(2, keepdim=True) + delta, self.pc_range)
        return features, logits, refined, visible


@MODELS.register_module()
class SparseWorldStrictEncoder(BaseModule):
    """Original full 1040-query OPUS decoding plus timestamp causal masking."""
    def __init__(self, embed_dims=256, num_decoder=6, num_frames=5, num_views=6, num_points=4,
                 num_levels=4, num_groups=4, num_heads=8, feedforward_channels=512, dropout=.1,
                 num_classes=17, num_refines=(1,4,16,24,32,48), scales=(.5,), pc_range=(-40.,-40.,-1.,40.,40.,5.4), init_cfg=None):
        super().__init__(init_cfg)
        if len(num_refines) != num_decoder: raise ValueError('num_refines must match num_decoder')
        scales = tuple(scales) * num_decoder if len(scales) == 1 else tuple(scales)
        prior = (1,) + tuple(num_refines[:-1]); self.num_frames, self.num_views, self.num_groups = num_frames, num_views, num_groups
        self.layers = nn.ModuleList([_Layer(embed_dims,num_frames,num_views,num_points,num_levels,num_groups,num_classes,a,b,num_heads,feedforward_channels,dropout,c,pc_range) for a,b,c in zip(prior,num_refines,scales)])

    def forward(self, query_features, query_points, query_stamps, ms_img_feats, metas, **kwargs):
        if ms_img_feats[0].shape[1] != self.num_frames * self.num_views: raise ValueError('expected T*N camera features')
        b = query_features.shape[0]; grouped=[]
        for feature in ms_img_feats:
            _, cameras, channels, h, w = feature.shape
            if channels % self.num_groups or cameras != self.num_frames * self.num_views: raise ValueError('invalid strict OPUS feature shape')
            grouped.append(feature.reshape(b,self.num_frames,self.num_views,self.num_groups,channels//self.num_groups,h,w).permute(0,1,3,2,5,6,4).reshape(b*self.num_frames*self.num_groups,self.num_views,h,w,channels//self.num_groups).contiguous())
        points=query_points.unsqueeze(2); out=[]
        for layer in self.layers:
            query_features, logits, points, visible = layer(points,query_features,grouped,metas,query_stamps)
            out.append(dict(query_features=query_features, query_points=points, opus_logits=logits, visible=visible))
            points=points.detach()
        return dict(representation=out)
