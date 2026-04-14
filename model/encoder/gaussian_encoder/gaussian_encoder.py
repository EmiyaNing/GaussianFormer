from typing import List, Optional
import torch, torch.nn as nn

from mmseg.registry import MODELS
from mmengine import build_from_cfg
from ..base_encoder import BaseEncoder


@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        sem_optimize: dict = None, # experiment module for decompose the refine module
        geo_optimize: dict = None, # experiment module for decompose the refine module
        mid_refine_layer: dict = None,
        densify_layer: dict = None,
        spconv_layer: dict = None,
        voxel_query_layer: dict = None,
        gaussian_photo: dict = None, # experiment module for photo supervision, color value get dynamically from input images
        gaussian_paint: dict = None, # experiment module for photo supervision, color value get only once.
        photo2render: dict = None, # experiment module for photo supervision, render the gaussian into image plane and calculate photometric loss.
        num_decoder: int = 6,
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder

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
            "sem_optimize": [sem_optimize, MODELS],
            "geo_optimize": [geo_optimize, MODELS],
            "densify" : [densify_layer, MODELS],
            "mid_refine":[mid_refine_layer, MODELS],
            "spconv": [spconv_layer, MODELS],
            "query": [voxel_query_layer, MODELS],
            "photo": [gaussian_photo, MODELS],
            "paint": [gaussian_paint, MODELS],
            "render": [photo2render, MODELS],
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
        render_imgs= []
        for i, op in enumerate(self.operation_order):
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
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif "refine" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                )
          
                prediction.append({'gaussian': gaussian})
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            elif "geo_optimize" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    gaussian
                )
                prediction.append({'gaussian': gaussian})
            elif "sem_optimize" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    gaussian
                )
                prediction.append({'gaussian': gaussian})
            elif "densify" in op:
                anchor, gaussian, instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    gaussian
                )

                prediction.append({'gaussian': gaussian})
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            elif "query" in op:
                instance_feature = self.layers[i](
                    gaussian,
                    instance_feature,
                    multi_stride_features
                )
            elif "photo" in op:
                rendered_img = self.layers[i](
                    kwargs['imgs'],
                    metas['mask_img'],
                    gaussian,
                    metas
                )
                render_imgs.append(rendered_img)
            elif "paint" in op:
                gaussian = self.layers[i](
                    kwargs['imgs'],
                    metas['mask_img'],
                    gaussian,
                    metas
                )
            elif "render" in op:
                gaussian, rendered_img = self.layers[i](
                    instance_feature,
                    gaussian,
                    metas
                )
                render_imgs.append(rendered_img)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return {"representation": prediction, "render_imgs": render_imgs}