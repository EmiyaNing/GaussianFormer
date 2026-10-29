from typing import List, Optional
import torch, torch.nn as nn

from mmseg.registry import MODELS
from mmengine import build_from_cfg
from ..base_encoder import BaseEncoder
from .deformable_module import DeformableFeatureAggregation, DAF
from .topk_module.topk_densify_v2 import DensifyOnly


@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        mid_refine_layer: dict = None,
        densify_layer: dict = None,
        spconv_layer: dict = None,
        ball_query_attn: dict = None,
        voxel_query_layer: dict = None,
        history_attn: dict = None,
        num_decoder: int = 6,
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
        cache_image_features=False,
        skip_unused_anchor_embed=False,
        terminal_densify=False,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder
        self.cache_image_features = cache_image_features
        self.skip_unused_anchor_embed = skip_unused_anchor_embed
        self.terminal_densify = terminal_densify

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, MODELS)
        self.op_config_map = {
            "norm": [norm_layer, MODELS],
            "ffn": [ffn, MODELS],
            "deformable": [deformable_model, MODELS],
            "refine": [refine_layer, MODELS],
            "densify" : [densify_layer, MODELS],
            "mid_refine":[mid_refine_layer, MODELS],
            "spconv": [spconv_layer, MODELS],
            "bqattn": [ball_query_attn, MODELS],
            "query": [voxel_query_layer, MODELS],
            "history": [history_attn, MODELS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        representation,
        rep_features,
        ms_img_feats=None,
        metas=None,
        multi_stride_features=None,
        **kwargs
    ):
        feature_maps = ms_img_feats
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        instance_feature = rep_features
        anchor = representation

        anchor_embed = self.anchor_encoder(anchor)

        prediction = []
        formatted_features = {}
        for i, op in enumerate(self.operation_order):
            if anchor_embed is None and (op in ('deformable', 'history') or 'refine' in op):
                anchor_embed = self.anchor_encoder(anchor)
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable" or op == "history":
                extra = {}
                layer = self.layers[i]
                if self.cache_image_features and isinstance(layer, DeformableFeatureAggregation):
                    cameras = layer.num_cams
                    if cameras not in formatted_features:
                        prepared = DAF.feature_maps_format([fm[:, :cameras] for fm in feature_maps])
                        formatted_features[cameras] = (
                            prepared[0].contiguous().float(), prepared[1].int(), prepared[2].int())
                    extra['formatted_feature_maps'] = formatted_features[cameras]
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                    **extra,
                )
            elif "refine" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                )
          
                prediction.append({'gaussian': gaussian})
                if self.skip_unused_anchor_embed:
                    anchor_embed = None
                elif i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            elif "densify" in op:
                terminal = (self.terminal_densify and i == len(self.operation_order) - 1
                            and type(self.layers[i]) is DensifyOnly)
                if terminal:
                    gaussian, delta = self.layers[i](instance_feature, anchor, gaussian, terminal=True)
                    densify_prediction = dict(gaussian=gaussian, delta_gaussian=delta,
                                              densify_parent_index=len(prediction) - 1)
                else:
                    anchor, gaussian, instance_feature = self.layers[i](
                        instance_feature, anchor, gaussian)
                    densify_prediction = {'gaussian': gaussian}
                if hasattr(self.layers[i], 'get_allocation_aux'):
                    allocation_aux = self.layers[i].get_allocation_aux()
                    if allocation_aux is not None:
                        densify_prediction['allocation_aux'] = allocation_aux
                prediction.append(densify_prediction)
                if self.skip_unused_anchor_embed:
                    anchor_embed = None
                elif i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            elif "query" in op:
                instance_feature = self.layers[i](
                    gaussian,
                    instance_feature,
                    multi_stride_features
                )
            elif "bqattn" in op:
                instance_feature = self.layers[i](
                    anchor,
                    instance_feature
                )
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return {"representation": prediction}
