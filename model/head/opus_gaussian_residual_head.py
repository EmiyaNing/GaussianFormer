"""Final-point Semantic Gaussian residual head for the strict OPUS decoder.

The Gaussian centres are exactly the final OPUS points.  This module therefore
implements a sparse, normalized Gaussian message-passing layer on the final
point set rather than a dense voxel renderer.
"""
from mmengine.registry import MODELS

from .opus_head import OPUSHead
from .point_anchored_semantic_gaussian import PointAnchoredSemanticGaussian


# Backward-compatible private alias: existing imports and, more importantly,
# the ``gaussian.*`` state namespace of OPUSGaussianResidualHead stay intact.
_PointAnchoredGaussianResidual = PointAnchoredSemanticGaussian


@MODELS.register_module()
class OPUSGaussianResidualHead(OPUSHead):
    """OPUS head with final-point Gaussian semantic residual aggregation.

    ``opus`` mode uses original point logits; ``gaussian_refined`` uses a
    detached OPUS logit baseline plus the Gaussian residual; ``ensemble`` uses
    trainable OPUS logits plus a configurable residual multiplier. All modes
    use the original OPUS hard voxelizer.
    """
    def __init__(self, gaussian_scale_range=((.1, .1, .1), (.6, .6, .6)),
                 gaussian_initial_scale=.35, gaussian_initial_opacity=.02,
                 gaussian_num_neighbors=8, gaussian_include_self=True,
                 gaussian_query_chunk_size=256, ensemble_gamma=1.,
                 eval_mode='ensemble', **kwargs):
        super().__init__(**kwargs)
        if not self.decoder_outputs_logits:
            raise ValueError('OPUSGaussianResidualHead requires decoder_outputs_logits=True')
        if eval_mode not in ('opus', 'gaussian_refined', 'ensemble'):
            raise ValueError('eval_mode must be opus, gaussian_refined, or ensemble')
        self.eval_mode = eval_mode
        self.ensemble_gamma = ensemble_gamma
        self.gaussian = _PointAnchoredGaussianResidual(
            self.embed_dims, self.point_multipliers[-1], self.num_classes,
            self.pc_range, gaussian_scale_range, gaussian_initial_scale,
            gaussian_initial_opacity, gaussian_num_neighbors, gaussian_include_self,
            gaussian_query_chunk_size)

    def forward(self, representation, metas=None, **kwargs):
        output = super().forward(representation, metas, **kwargs)
        state = representation[-1]
        token = state['query_features']
        points = self._to_world(state['query_points'])
        base_logits = state['opus_logits'].flatten(1, 2)
        query_mask = state.get('query_valid_mask')
        if query_mask is None:
            query_mask = torch.ones(token.shape[:2], dtype=torch.bool, device=token.device)
        gaussian = self.gaussian(token, points, query_mask)
        refined_logits = base_logits.detach() + gaussian['residual']
        ensemble_logits = base_logits + self.ensemble_gamma * gaussian['residual']
        output.update({
            'gaussian_point_logits': refined_logits,
            'gaussian_point_points': gaussian['points'],
            'gaussian_point_valid_mask': gaussian['valid_mask'],
            'gaussian_scales': gaussian['scales'],
            'gaussian_opacities': gaussian['opacities'],
            'gaussian_residual': gaussian['residual'],
            'opus_final_occ': output['final_occ'],
        })
        # Rasterization is used only for validation/inference metrics. During
        # training the Gaussian loss consumes point-aligned logits directly;
        # avoiding two extra hard voxelizations preserves OPUS throughput.
        if self.training:
            return output
        gaussian_occ = self.rasterizer(
            gaussian['points'].detach(), refined_logits.detach(), self.score_threshold,
            gaussian['valid_mask'], group_size=self.point_multipliers[-1],
            center_distance_threshold=self.center_distance_threshold)
        ensemble_occ = self.rasterizer(
            gaussian['points'].detach(), ensemble_logits.detach(), self.score_threshold,
            gaussian['valid_mask'], group_size=self.point_multipliers[-1],
            center_distance_threshold=self.center_distance_threshold)
        output.update({'gaussian_final_occ': gaussian_occ, 'ensemble_final_occ': ensemble_occ})
        if self.eval_mode == 'gaussian_refined':
            output['final_occ'] = gaussian_occ
        elif self.eval_mode == 'ensemble':
            output['final_occ'] = ensemble_occ
        return output
