"""Batched PMG adapter for the upstream, unchanged GF2 probability operator."""

import torch
from mmengine.registry import MODELS

from .gaussian_opus_head import _flatten_grouped
from .opus_gf2_probability_head import OPUSGF2ProbabilityHead


@MODELS.register_module()
class PMGGF2ProbabilityRenderer(OPUSGF2ProbabilityHead):
    """Return joint class probabilities and conditional occupied semantics.

    GF2 geometry uses 1-prod(1-kernel), independently of opacity. PMG's zero
    opacities mark invalid timestamps/ROI points, so exclude those children
    before calling the official single-sample kernel. Positive opacities still
    weight the conditional semantic mixture exactly as upstream.
    """

    def render_distribution(self, gaussian, sampled_xyz):
        gaussian = _flatten_grouped(gaussian)
        if gaussian.scales.detach().amax() > self.max_render_scale + 1e-6:
            raise ValueError("Gaussian scale exceeds registered renderer support")
        joints, conditionals = [], []
        for index in range(sampled_xyz.shape[0]):
            valid = gaussian.opacities[index, :, 0].detach() > 0
            sample = type(gaussian)(*(
                value[index:index + 1, valid] if value is not None else None
                for value in gaussian
            ))
            if not valid.any():
                zero = gaussian.means[index].sum() * 0
                conditional = sampled_xyz.new_full((sampled_xyz.shape[1], 17), 1 / 17) + zero
                occupied = sampled_xyz.new_zeros(sampled_xyz.shape[1]) + zero
            else:
                means, opacity, semantic, scales, inverse = self.prepare_gaussian_args(sample)
                conditional, occupied, _ = self.aggregator(
                    sampled_xyz[index:index + 1].contiguous().float(),
                    means.contiguous(), opacity.reshape(1, -1).contiguous(),
                    semantic.contiguous(), scales.contiguous(), inverse.contiguous(),
                )
                conditional = conditional[:, :17]
            joint = torch.cat((conditional * occupied[:, None], (1 - occupied)[:, None]), -1)
            joints.append(joint.T)
            conditionals.append(conditional.T)
        return torch.stack(joints), torch.stack(conditionals)

    def render_state(self, gaussian, sampled_xyz):
        return self.render_distribution(gaussian, sampled_xyz)[0]
