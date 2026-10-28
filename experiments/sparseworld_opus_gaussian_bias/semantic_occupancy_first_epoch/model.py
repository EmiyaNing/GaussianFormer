"""An independent child occupancy branch using the existing Gaussian geometry."""
import math
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.head.gaussian_opus_head import _flatten_grouped
from ..encoder import SparseWorldBiasGaussianEncoder
from ..model import SparseWorldBiasRenderer, SparseWorldOPUSGaussianBias


class OccupancyGaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    semantics: Tensor
    original_means: Optional[Tensor] = None
    delta_means: Optional[Tensor] = None
    occupancy_evidence: Optional[Tensor] = None


def joint_log_probabilities(foreground_logits, occupancy_logit):
    if foreground_logits.shape[1] != 17 or occupancy_logit.shape != foreground_logits[:, 0].shape:
        raise ValueError('expected seventeen conditional logits and one occupancy logit per voxel')
    semantic = foreground_logits.float().log_softmax(1)
    occupied = F.logsigmoid(occupancy_logit.float())[:, None]
    free = F.logsigmoid(-occupancy_logit.float())[:, None]
    return torch.cat((semantic + occupied, free), dim=1)


def predict_factorized(log_probabilities, threshold=.5):
    if not 0 < threshold < 1:
        raise ValueError('occupancy threshold must lie in (0,1)')
    # Free probability depends only on the independent gate, even at floating-point precision.
    occupied = log_probabilities[:, 17] < math.log1p(-threshold)
    return torch.where(occupied, log_probabilities[:, :17].argmax(1), 17)


@MODELS.register_module()
class FactorizedOccupancyGaussianEncoder(SparseWorldBiasGaussianEncoder):
    def __init__(self, initial_occupancy_logit=-4.59511985013459, **kwargs):
        super().__init__(**kwargs)
        self.initial_occupancy_logit = float(initial_occupancy_logit)
        # Extra layers must not consume the RNG used to initialize the unchanged world model.
        with torch.random.fork_rng(devices=[]):
            self.occupancy_branches = nn.ModuleDict()
            for stage in self.attribute_stages:
                head = nn.Linear(self.embed_dims, self.num_refines[stage])
                nn.init.zeros_(head.weight)
                nn.init.constant_(head.bias, self.initial_occupancy_logit)
                self.occupancy_branches[str(stage)] = head

    def _make_gaussian(self, features, points, logits, valid, stage):
        gaussian = super()._make_gaussian(features, points, logits, valid, stage)
        if str(stage) in self.occupancy_branches:
            raw = self.occupancy_branches[str(stage)](features)[..., None]
        else:
            raw = logits.new_full((*points.shape[:3], 1), self.initial_occupancy_logit)
        return OccupancyGaussianPrediction(**gaussian._asdict(), occupancy_evidence=F.softplus(raw))


@MODELS.register_module()
class FactorizedOccupancyGaussianRenderer(SparseWorldBiasRenderer):
    def __init__(self, occupancy_threshold=.5, occupancy_prior_offset=math.log(17.), **kwargs):
        super().__init__(**kwargs)
        self.occupancy_threshold = float(occupancy_threshold)
        self.occupancy_prior_offset = float(occupancy_prior_offset)
        if not 0 < self.occupancy_threshold < 1:
            raise ValueError('invalid fixed occupancy threshold')
        self.record_decisions = False
        self.decision_cache = []

    def render_state(self, gaussian, sampled_xyz):
        flat = _flatten_grouped(gaussian)
        if flat.scales.detach().amax() > self.max_render_scale:
            raise ValueError('Gaussian scale exceeds the unchanged local aggregation bound')
        if gaussian.occupancy_evidence is None:
            raise ValueError('factorized rendering requires an independent occupancy branch')
        args = list(self.prepare_gaussian_args(flat))
        evidence = gaussian.occupancy_evidence.flatten(1, 2)
        # Reuse channel 17 to accumulate foreground evidence minus the existing
        # broad background field. The first 17 channels retain C's exact renderer.
        background = -self.empty_scalar.to(evidence).reshape(1, 1, 1).expand(evidence.shape[0], 1, 1)
        args[2] = torch.cat((args[2][..., :17], torch.cat((evidence, background), dim=1)), dim=-1)
        raw = self._render_batched(sampled_xyz, *args)
        gate = raw[:, 17] + self.empty_bias + self.occupancy_prior_offset
        joint = joint_log_probabilities(raw[:, :17], gate)
        if self.record_decisions:
            self.decision_cache.append(predict_factorized(joint.detach(), self.occupancy_threshold))
        return joint


@SEGMENTORS.register_module()
class SparseWorldFactorizedOccupancy(SparseWorldOPUSGaussianBias):
    def forward(self, *args, **kwargs):
        renderer = self.gaussian_head
        renderer.decision_cache.clear()
        renderer.record_decisions = True
        try:
            result = super().forward(*args, **kwargs)
            decisions = renderer.decision_cache[:]
        finally:
            renderer.record_decisions = False
            renderer.decision_cache.clear()
        if len(decisions) != 1 + len(result['future_occ_grids']):
            raise ValueError('independent occupancy decisions did not cover every horizon')
        result['joint_argmax_occ_grids'] = [result['final_occ_grid']] + result['future_occ_grids']
        grids = [value.view_as(result['final_occ_grid']) for value in decisions]
        result.update(final_occ=grids[0].flatten(1), final_occ_grid=grids[0], future_occ_grids=grids[1:])
        for h, grid in enumerate(grids):
            result[f'semantic_occ_{h}s'] = grid
            result[f'geo_occ_{h}s'] = torch.where(grid != 17, 0, 17)
        return result

