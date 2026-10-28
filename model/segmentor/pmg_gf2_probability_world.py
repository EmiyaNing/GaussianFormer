"""PMG A0 with GF2 foreground/background probability rendering at every horizon."""

import contextlib

import torch
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.sparse_opus_gausisan_world_encoder import SparseOpusGaussianEncoder
from model.head.pmg_gf2_probability_head import PMGGF2ProbabilityRenderer  # registry
from .pmg_sparse_opus_world import PMGSparseOpusWorld


@MODELS.register_module()
class PMGGF2ProbabilityEncoder(SparseOpusGaussianEncoder):
    def _make_gaussian(self, features, points, logits, valid, stage):
        gaussian = super()._make_gaussian(features, points, logits, valid, stage)
        # The inherited encoder stores softplus scores. GF2 needs raw logits
        # before its per-Gaussian softmax; point/world losses retain raw logits.
        return gaussian._replace(semantics=logits)


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGGF2ProbabilityWorld(PMGSparseOpusWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.geometry_mode != "baseline" or self.persistent_content:
            raise ValueError("This matched experiment uses PMG A0 geometry/content")

    def _render_occupancy(self, outputs, forecast_points, forecast_classes,
                          forecast_features, current_queries, sampled_xyz,
                          batch_size, render_all=True):
        gaussian = outputs["representation"][-1]["gaussian"]
        current = type(gaussian)(*(
            value[:, current_queries] if value is not None else None for value in gaussian
        ))
        joint, conditional = self.gaussian_head.render_distribution(current, sampled_xyz)
        probabilities, conditionals = {0: joint}, {0: conditional}
        grids = [joint.argmax(1).view(batch_size, 200, 200, 16)]
        encoder = self.pts_bbox_head.transformer.encoder
        for horizon, (points, classes, features) in enumerate(
                zip(forecast_points, forecast_classes, forecast_features), 1):
            supervised = self.training and not self.pretrain and horizon in self.gaussian_loss_horizons
            if not render_all and not supervised:
                continue
            with contextlib.nullcontext() if supervised else torch.no_grad():
                valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
                state = encoder._make_gaussian(features, points, classes, valid, len(encoder.layers) - 1)
                joint, conditional = self.gaussian_head.render_distribution(state, sampled_xyz)
                if supervised:
                    probabilities[horizon], conditionals[horizon] = joint, conditional
                if render_all:
                    grids.append(joint.argmax(1).view(batch_size, 200, 200, 16))
        # SparseOpusGaussianWorld.forward copies outputs into the final result.
        # gaussian_logits_by_horizon is a legacy key carrying probabilities here.
        outputs["gaussian_conditionals_by_horizon"] = conditionals
        return probabilities, grids
