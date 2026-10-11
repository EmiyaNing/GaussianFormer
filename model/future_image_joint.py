"""Isolated image-conditioned joint feature/geometry refinement experiment."""
import copy

import torch
from torch import nn

from .future_image_probe import AdaptiveOPUSImageReader, FutureImageFusionProbe


class FutureImageJointRefiner(nn.Module):
    """Refine every existing future query twice without removing born slots.

    Frozen forecasting provides initial queries and points. A shared OPUS
    sampling/mixing + spatial attention + FFN block reads the target camera
    features at the updated geometry on each round. Trainable semantic,
    regression and velocity heads decode a residual relative to frozen source
    heads. Zero-output fusion and identical heads reproduce the forecast at
    initialization. Target horizons are refined independently, not rolled into
    another forecast horizon. Future RGB is privileged information.
    """

    def __init__(self, student, rounds=2, max_step_m=4.):
        super().__init__()
        layer = student.encoder.layers[-1]
        self.reader = AdaptiveOPUSImageReader(layer)
        self.position_encoder = copy.deepcopy(layer.position)
        self.self_attn = copy.deepcopy(layer.self_attn)
        self.ffn = copy.deepcopy(layer.ffn)
        self.norm1 = copy.deepcopy(layer.norm1)
        self.norm2 = copy.deepcopy(layer.norm2)
        self.norm3 = copy.deepcopy(layer.norm3)
        self.fusion = FutureImageFusionProbe()
        self.semantic = copy.deepcopy(student.semantic)
        self.reg = copy.deepcopy(student.reg)
        self.velocity = copy.deepcopy(student.velocity)
        self.reference_semantic = copy.deepcopy(student.semantic)
        self.reference_reg = copy.deepcopy(student.reg)
        self.reference_velocity = copy.deepcopy(student.velocity)
        self.pc_range = tuple(student.pc_range)
        self.num_refines = student.num_refines
        self.rounds = int(rounds)
        self.max_step_m = float(max_step_m)
        if self.rounds < 1 or self.max_step_m <= 0:
            raise ValueError('rounds and max_step_m must be positive')
        # The last pretrained position encoder expects the preceding layer's
        # 32-point set; select uniformly from the forecast's 48-point set.
        source_points = self.position_encoder[0].in_features // 3
        self.register_buffer('position_indices', torch.linspace(
            0, self.num_refines - 1, source_points).round().long())
        self.requires_grad_(True)
        for module in (self.reference_semantic, self.reference_reg,
                       self.reference_velocity):
            module.requires_grad_(False)

    def world(self, points):
        lower = points.new_tensor(self.pc_range[:3])
        extent = points.new_tensor(self.pc_range[3:]) - lower
        return points * extent + lower

    def _offset(self, query, logits, reg, velocity):
        shape = (*query.shape[:2], self.num_refines)
        offset = reg(query).view(*shape, 3) * .5
        motion = velocity(query).view(*shape, 2)
        moving = ((logits.argmax(-1) >= 2) & (logits.argmax(-1) <= 10))
        return torch.cat((offset[..., :2] + motion * moving.unsqueeze(-1),
                          offset[..., 2:]), dim=-1)

    def forward(self, query, normalized_points, image_maps=None, metas=None,
                shuffle=False):
        points = normalized_points
        outputs = []
        extent = points.new_tensor(self.pc_range[3:]) - points.new_tensor(self.pc_range[:3])
        for _ in range(self.rounds):
            with torch.no_grad():
                reference_logits = self.reference_semantic(query).view(
                    *query.shape[:2], self.num_refines, 17)
                reference_offset = self._offset(
                    query, reference_logits, self.reference_reg,
                    self.reference_velocity)
            positional = self.position_encoder(
                points.index_select(2, self.position_indices).flatten(2))
            sampling_query = query + positional
            if image_maps is None:
                extra = torch.zeros_like(query)
                visible = torch.zeros(query.shape[:2], device=query.device,
                                      dtype=torch.bool)
            else:
                extra, visible = self.reader(
                    points, sampling_query, image_maps, metas)
                if shuffle:
                    indices = visible[0].nonzero(as_tuple=False).flatten()
                    permutation = torch.arange(query.shape[1], device=query.device)
                    permutation[indices] = indices.roll(1)
                    extra = extra.index_select(1, permutation)
            decoded = self.norm1(sampling_query + extra)
            # All active slots now represent the same target timestamp,
            # including retained future-born slots, so mutual attention is valid.
            stamps = torch.zeros(query.shape[1], device=query.device, dtype=torch.long)
            decoded = self.norm2(self.self_attn(points, decoded, stamps))
            decoded = self.norm3(decoded + self.ffn(decoded))
            query, delta = self.fusion(query, decoded)
            logits = self.semantic(query).view(
                *query.shape[:2], self.num_refines, 17)
            correction = self._offset(query, logits, self.reg, self.velocity) - reference_offset
            correction = self.max_step_m * torch.tanh(correction / self.max_step_m)
            # Point movement remains differentiable, including between reads.
            points = points + correction / extent
            outputs.append(dict(points=self.world(points).flatten(1, 2),
                                logits=logits.flatten(1, 2),
                                query=query, normalized_points=points,
                                feature_delta=delta, correction=correction,
                                visible=visible))
        return outputs
