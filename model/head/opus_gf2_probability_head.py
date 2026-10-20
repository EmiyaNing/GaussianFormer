"""OPUS outputs with official GF2 geometry/semantic probability aggregation."""
import torch
from mmengine.registry import MODELS
from .gaussian_head import GaussianHead
from .gaussian_opus_head import _flatten_grouped
from .gf2_probability_backend import probability_backend
from model.utils.utils import get_rotation_matrix


@MODELS.register_module()
class OPUSGF2ProbabilityHead(GaussianHead):
    def __init__(self, max_render_scale=.4, **kwargs):
        # Construct the shared head without importing an installed probability
        # package; then use the project-local, unmodified official extension.
        kwargs.pop('with_empty', None)
        kwargs.pop('use_localaggprob', None)
        kwargs.pop('combine_geosem', None)
        kwargs.pop('empty_logit_offset', None)
        kwargs.pop('initial_empty_bias', None)
        kwargs.pop('use_localagg_react', None)
        super().__init__(with_empty=False, use_localaggprob=False, **kwargs)
        self.use_localaggprob = True
        self.combine_geosem = True
        self.max_render_scale = max_render_scale
        self.aggregator = probability_backend().LocalAggregator(**kwargs['cuda_kwargs'])

    def prepare_gaussian_args(self, gaussian):
        means, scales = gaussian.means, gaussian.scales
        rotation = get_rotation_matrix(gaussian.rotations)
        transform = torch.diag_embed(scales) @ rotation
        inverse = torch.linalg.inv(transform.transpose(-1, -2) @ transform)
        semantic = gaussian.semantics.softmax(-1)
        semantic = torch.cat([semantic, torch.zeros_like(semantic[..., :1])], -1)
        opacity = gaussian.opacities
        if opacity.numel() == 0:
            opacity = torch.ones_like(semantic[..., :1])
        return means, opacity, semantic, scales, inverse

    def forward(self, representation, metas=None, **kwargs):
        flat, points, logits, valid = [], [], [], []
        for state in representation:
            gaussian = _flatten_grouped(state['gaussian'])
            raw = state['opus_logits'].reshape(gaussian.means.shape[0], -1, 17)
            # The encoder's softplus is retained for baseline compatibility;
            # GF2 rendering consumes the ORIGINAL raw logits through softmax.
            gaussian = gaussian._replace(semantics=raw)
            flat.append(dict(gaussian=gaussian))
            points.append(gaussian.means)
            logits.append(raw)
            batch, queries, children = state['query_points'].shape[:3]
            mask = state.get('query_valid_mask')
            if mask is None:
                mask = torch.ones(batch, queries, device=raw.device, dtype=torch.bool)
            valid.append(mask[..., None].expand(-1, -1, children).reshape(batch, -1))
        indices = [len(flat)-1] if not self.training else self.fixed_apply_loss_layers
        for stage in indices:
            if flat[stage]['gaussian'].scales.detach().max() > self.max_render_scale + 1e-6:
                raise RuntimeError('Gaussian exceeds registered renderer support bound')
        xyz, labels = self._sampling(metas['occ_xyz'], metas['occ_label'], None)
        if xyz.shape[0] != 1:
            raise ValueError('This registered mini experiment uses batch size1')
        predictions, conditional, occupancies, densities = [], [], [], []
        for stage in indices:
            args = self.prepare_gaussian_args(flat[stage]['gaussian'])
            means, opacity, semantic, scales, inverse = args
            q, o, density = self.aggregator(xyz.clone().float(), means,
                opacity.reshape(1,-1), semantic, scales, inverse)
            probabilities = torch.cat([q[:,:17]*o[:,None],(1-o)[:,None]],-1)
            predictions.append(probabilities.T[None])
            conditional.append(q[:,:17].T[None])
            occupancies.append(o[None]);densities.append(density[None])
        result = dict(pred_occ=predictions, conditional_semantics=conditional,
            bin_logits=occupancies,density=densities,final_occ=predictions[-1].argmax(1),
            sampled_xyz=xyz,sampled_label=labels,gaussian=flat[-1]['gaussian'],
            gaussians=[state['gaussian'] for state in flat],
            occ_mask=metas.get('occ_mask'),occ_loss_mask=metas.get('occ_loss_mask',metas.get('occ_mask')),
            occ_cam_mask=metas.get('occ_cam_mask'))
        result.update(opus_pred_points=points, opus_pred_logits=logits,
            opus_point_valid_masks=valid, opus_points=points[-1],
            opus_labels=logits[-1].argmax(-1), opus_scores=logits[-1].sigmoid().amax(-1),
            supervised_stage_indices=indices)
        return result
