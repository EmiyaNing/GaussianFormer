"""SparseWorld with point-anchored Semantic Gaussian residual refinement."""
from __future__ import annotations

import torch
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.opus_encoder import _opus_decode_points
from .sparseworld_strict import SparseWorldStrict


@SEGMENTORS.register_module()
class SparseGaussianWorld(SparseWorldStrict):
    """Add shared current/future Semantic Gaussians to SparseWorldStrict.

    The inherited SparseWorld parameter namespace and all official set-loss
    outputs are untouched.  This adapter adds only ``semantic_gaussian.*``
    state and diagnostic/refined occupancy outputs.
    """

    def __init__(self, semantic_gaussian, gaussian_horizons=(0, 2, 4, 6),
                 eval_mode='ensemble', ensemble_gamma=1.,
                 freeze_sparseworld=False, **kwargs):
        super().__init__(**kwargs)
        if eval_mode not in ('base', 'gaussian_refined', 'ensemble'):
            raise ValueError(
                'eval_mode must be base, gaussian_refined, or ensemble')
        horizons = tuple(sorted(set(int(value) for value in gaussian_horizons)))
        if not horizons or horizons[0] < 0 or horizons[-1] > self.num_fu_frames:
            raise ValueError('gaussian_horizons must be within [0, num_fu_frames]')
        if 0 not in horizons:
            raise ValueError('SparseGaussianWorld training requires current horizon 0')
        gaussian_cfg = dict(semantic_gaussian)
        gaussian_cfg.setdefault('type', 'PointAnchoredSemanticGaussianCUDA')
        gaussian_cfg.setdefault('embed_dims', self.out_dim)
        gaussian_cfg.setdefault('num_children', self.num_refines)
        gaussian_cfg.setdefault('num_classes', self.pts_bbox_head.num_classes)
        gaussian_cfg.setdefault('pc_range', self.pc_range.tolist())
        self.freeze_sparseworld = bool(freeze_sparseworld)
        if self.freeze_sparseworld:
            # This is evaluated before the new adapter is registered, so only
            # the released SparseWorld graph is frozen.
            self.requires_grad_(False)
        self.semantic_gaussian = MODELS.build(gaussian_cfg)
        if self.semantic_gaussian.embed_dims != self.out_dim:
            raise ValueError('Semantic Gaussian embed_dims must match SparseWorld out_dim')
        if self.semantic_gaussian.num_children != self.num_refines:
            raise ValueError('Semantic Gaussian num_children must match final SparseWorld refines')
        if self.semantic_gaussian.num_classes != self.pts_bbox_head.num_classes:
            raise ValueError('Semantic Gaussian num_classes must match SparseWorld classes')
        self.gaussian_horizons = horizons
        self.eval_mode = eval_mode
        self.ensemble_gamma = float(ensemble_gamma)
        self.gaussian_pc_range = tuple(float(value) for value in self.pc_range)

    def _refine_horizon(self, features, normalized_points, base_logits):
        world_points = _opus_decode_points(
            normalized_points, self.gaussian_pc_range)
        lower = world_points.new_tensor(self.gaussian_pc_range[:3])
        upper = world_points.new_tensor(self.gaussian_pc_range[3:])
        point_valid = torch.isfinite(world_points).all(dim=-1)
        point_valid &= ((world_points >= lower) & (world_points < upper)).all(dim=-1)
        query_valid = torch.ones(
            features.shape[:2], dtype=torch.bool, device=features.device)
        gaussian = self.semantic_gaussian(
            features, world_points, query_valid, point_valid)
        base_flat = base_logits.flatten(1, 2)
        refined = base_flat.detach() + gaussian['residual']
        ensemble = base_flat + self.ensemble_gamma * gaussian['residual']
        gaussian.update(
            refined_logits=refined,
            ensemble_logits=ensemble,
            refined_grouped=refined.view_as(base_logits),
            ensemble_grouped=ensemble.view_as(base_logits))
        return gaussian

    def _rasterize(self, normalized_points, grouped_logits):
        return self.pts_bbox_head.get_occ(dict(
            cls_scores=grouped_logits.detach(),
            refine_pts=normalized_points.detach()))

    @staticmethod
    def _geometry_from_semantics(occupancy):
        geometry = torch.full_like(occupancy, 17)
        geometry[occupancy != 17] = 0
        return geometry

    def forward(self, imgs=None, metas=None, **kwargs):
        result = super().forward(imgs=imgs, metas=metas, **kwargs)
        base_current = result['final_occ_grid']
        base_future = list(result['future_occ_grids'])
        gaussian_current = base_current
        ensemble_current = base_current
        gaussian_future = list(base_future)
        ensemble_future = list(base_future)

        stamps = result['query_stamps']
        current_state = self._refine_horizon(
            result['query_feat'][:, stamps == 0], result['refine_pts'],
            result['cls_score'])
        gaussian_current = self._rasterize(
            result['refine_pts'], current_state['refined_grouped'])
        ensemble_current = self._rasterize(
            result['refine_pts'], current_state['ensemble_grouped'])

        future_states, future_horizons = [], []
        for horizon in self.gaussian_horizons:
            if horizon == 0:
                continue
            future_index = horizon - 1
            if future_index >= len(result['forecast_points_list']):
                continue
            state = self._refine_horizon(
                result['forecast_features_list'][future_index],
                result['forecast_points_list'][future_index],
                result['forecast_semantics_list'][future_index])
            future_states.append(state)
            future_horizons.append(horizon)
            gaussian_future[future_index] = self._rasterize(
                result['forecast_points_list'][future_index],
                state['refined_grouped'])
            ensemble_future[future_index] = self._rasterize(
                result['forecast_points_list'][future_index],
                state['ensemble_grouped'])

        result.update(
            base_final_occ_grid=base_current,
            base_future_occ_grids=base_future,
            gaussian_final_occ_grid=gaussian_current,
            gaussian_future_occ_grids=gaussian_future,
            ensemble_final_occ_grid=ensemble_current,
            ensemble_future_occ_grids=ensemble_future,
            current_gaussian_point_points=current_state['points'],
            current_gaussian_point_logits=current_state['refined_logits'],
            current_gaussian_point_valid_mask=current_state['valid_mask'],
            gaussian_scales=current_state['scales'],
            gaussian_opacities=current_state['opacities'],
            gaussian_residual=current_state['residual'],
            future_gaussian_horizons=future_horizons,
            future_gaussian_point_points=[value['points'] for value in future_states],
            future_gaussian_point_logits=[value['refined_logits'] for value in future_states],
            future_gaussian_point_valid_masks=[value['valid_mask'] for value in future_states],
            future_gaussian_scales=[value['scales'] for value in future_states],
            future_gaussian_opacities=[value['opacities'] for value in future_states],
            future_gaussian_residuals=[value['residual'] for value in future_states])

        if self.eval_mode == 'base':
            selected_current, selected_future = base_current, base_future
        elif self.eval_mode == 'gaussian_refined':
            selected_current, selected_future = gaussian_current, gaussian_future
        else:
            selected_current, selected_future = ensemble_current, ensemble_future
        result['final_occ_grid'] = selected_current
        result['final_occ'] = selected_current.flatten(1)
        result['future_occ_grids'] = selected_future
        for horizon, occupancy in enumerate([selected_current] + selected_future):
            result[f'semantic_occ_{horizon}s'] = occupancy
            result[f'geo_occ_{horizon}s'] = self._geometry_from_semantics(occupancy)
        return result
