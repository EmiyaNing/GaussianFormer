"""Matched-capacity current-state and historical-image position correction heads."""

import torch
from torch import nn
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.opus_encoder import _sample_image_features
from .pmg_motion_residual_world import PMGMotionResidualWorld


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGEvidenceResidualWorld(PMGMotionResidualWorld):
    """Reuse native rollout and its output-only correction/range-mask handling.

    Both arms have identical trainable tensors. Their 256-dimensional evidence
    is either the current query state or pooled projected historical image
    features at that query's current center. No explicit motion alignment is
    claimed for this first historical-image control.
    """

    def __init__(self, correction_evidence="current", **kwargs):
        super().__init__(**kwargs)
        if correction_evidence not in ("current", "history"):
            raise ValueError("unsupported correction evidence")
        self.correction_evidence = correction_evidence
        with torch.random.fork_rng(devices=[]):
            self.motion_correction = nn.Sequential(nn.Linear(513, 256), nn.ReLU(),
                nn.Linear(256, self.num_refines * 4))
            nn.init.zeros_(self.motion_correction[-1].weight)
            nn.init.zeros_(self.motion_correction[-1].bias)
        self._evidence_maps = None
        self._local_evidence = None
        self._correction_step = 0

    def extract_feat(self, images, metas=None):
        maps = super().extract_feat(images, metas)
        self._evidence_maps = maps
        return maps

    def correction(self, features):
        self._correction_step += 1
        time = features.new_full((*features.shape[:2], 1), self._correction_step * .5)
        inputs = torch.cat((features.detach(), self._local_evidence.detach(), time), -1)
        raw = self.motion_correction(inputs).unflatten(-1, (self.num_refines, 4))
        radius = self.correction_radius_m
        xy = radius * torch.tanh(raw[..., :2] / radius)
        xy = xy / (xy.norm(dim=-1, keepdim=True) / radius).clamp_min(1.)
        xy = xy * raw[..., 3:].sigmoid()
        return torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)

    def _forecast(self, outputs, ego_feature, metas):
        current = self.pts_bbox_head.ind_stamps_all == 0
        if self.correction_evidence == "current":
            self._local_evidence = outputs["query_feat"][:, current]
        else:
            centers = outputs["all_refine_pts"][-1][:, current].mean(-2)
            self._local_evidence, _ = _sample_image_features(
                self._evidence_maps, centers, metas, self.pc_range)
        self._correction_step = 0
        epoch = self.curr_epoch
        if self.training:
            self.curr_epoch = max(epoch, self.finetune_epoch + self.num_fu_frames - 1)
        try:
            return super()._forecast(outputs, ego_feature, metas)
        finally:
            self.curr_epoch = epoch
            self._evidence_maps = None
            self._local_evidence = None
