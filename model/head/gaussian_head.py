import numpy as np
import torch, torch.nn as nn

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from ..utils.utils import get_rotation_matrix
from .gaussian_execution import inverse_gaussian_covariance, fixed_gaussian_weights


@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self, 
        init_cfg=None,
        apply_loss_type=None,
        num_classes=18,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='nusc',
        empty_label=17,
        use_localaggprob=False,
        use_localaggprob_fast=False,
        use_localagg_react=False,
        combine_geosem=False,
        analytic_covariance=False,
        separate_empty=False,
        incremental_densify=False,
        cache_render_grid=False,
        react_renderer='legacy',
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        self.num_classes = num_classes
        self.use_localaggprob = use_localaggprob
        self.use_localagg_react = use_localagg_react
        self.analytic_covariance = analytic_covariance
        self.separate_empty = separate_empty
        self.incremental_densify = incremental_densify
        self.cache_render_grid = cache_render_grid
        if react_renderer not in ('legacy', 'tiled'):
            raise ValueError('react_renderer must be legacy or tiled')
        if react_renderer != 'legacy' and not use_localagg_react:
            raise ValueError('react_renderer requires use_localagg_react=True')
        if (separate_empty or incremental_densify or cache_render_grid) and not use_localagg_react:
            raise ValueError('React execution optimizations require the additive React renderer')
        if separate_empty and not with_empty:
            raise ValueError('separate_empty requires with_empty=True')
        self.register_buffer('_cached_points_int', torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer('_cached_empty_weights', torch.empty(0), persistent=False)
        if use_localagg_react:
            if react_renderer == 'tiled':
                from .react_tiled_aggregate import TiledReactAggregator
                self.aggregator = TiledReactAggregator(**cuda_kwargs)
            elif analytic_covariance or separate_empty or incremental_densify or cache_render_grid:
                from .react_local_aggregate import PreparedReactAggregator
                self.aggregator = PreparedReactAggregator(**cuda_kwargs)
            else:
                import local_aggregate_react
                self.aggregator = local_aggregate_react.LocalAggregator(**cuda_kwargs)
        elif use_localaggprob:
            if use_localaggprob_fast:
                import local_aggregate_prob_fast
                self.aggregator = local_aggregate_prob_fast.LocalAggregator(**cuda_kwargs)
            else:
                import local_aggregate_prob
                self.aggregator = local_aggregate_prob.LocalAggregator(**cuda_kwargs)
        else:
            import local_aggregate
            self.aggregator = local_aggregate.LocalAggregator(**cuda_kwargs)
        
        self.combine_geosem = combine_geosem
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz, gt_label

    def prepare_gaussian_args(self, gaussians, append_empty=True):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)
            if append_empty:
                means = torch.cat([means, self.empty_mean], dim=1)
                scales = torch.cat([scales, self.empty_scale], dim=1)
                rotations = torch.cat([rotations, self.empty_rot], dim=1)
                empty_sem = self.empty_sem.clone()
                empty_sem[..., self.empty_label] += self.empty_scalar
                opacities = torch.cat([opacities, empty_sem], dim=1)
                origi_opa = torch.cat([origi_opa, self.empty_opa], dim=1)
        elif self.use_localaggprob:
            assert opacities.shape[-1] == self.num_classes - 1
            opacities = opacities.softmax(dim=-1)
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)

        bs, g, _ = means.shape
        if self.analytic_covariance:
            CovInv = inverse_gaussian_covariance(scales, rotations)
        else:
            S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
            S[..., 0, 0] = scales[..., 0]
            S[..., 1, 1] = scales[..., 1]
            S[..., 2, 2] = scales[..., 2]
            R = get_rotation_matrix(rotations) # b, g, 3, 3
            M = torch.matmul(S, R)
            Cov = torch.matmul(M.transpose(-1, -2), M)
            CovInv = Cov.cpu().inverse().cuda() # b, g, 3, 3
        return means, origi_opa, opacities, scales, CovInv

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Cached fields are not checkpoint state and must follow newly loaded
        # fixed background geometry, even when tensor shapes did not change.
        self._cached_points_int = self._cached_points_int.new_empty(0)
        self._cached_empty_weights = self._cached_empty_weights.new_empty(0)

    def _react_context(self, sampled_xyz):
        if not self.use_localagg_react or not (
                self.cache_render_grid or self.separate_empty or self.incremental_densify):
            return {}
        points = sampled_xyz[0].float()
        if self.cache_render_grid and self._cached_points_int.shape == points.shape:
            points_int = self._cached_points_int
        else:
            points_int = ((points - self.aggregator.pc_min) /
                          self.aggregator.grid_size).to(torch.int32)
            if self.cache_render_grid:
                self._cached_points_int = points_int.detach()
        context = dict(points_int=points_int)
        if self.separate_empty:
            if self.cache_render_grid and self._cached_empty_weights.shape == points.shape[:1]:
                weights = self._cached_empty_weights
            else:
                with torch.no_grad():
                    weights = fixed_gaussian_weights(
                        points, self.empty_mean, self.empty_scale, self.empty_rot,
                        self.aggregator.scale_multiplier, self.aggregator.pc_min,
                        self.aggregator.grid_size,
                        (self.aggregator.H, self.aggregator.W, self.aggregator.D))
                if self.cache_render_grid:
                    self._cached_empty_weights = weights
            background_sem = self.empty_sem[0, 0].clone()
            background_sem[self.empty_label] += self.empty_scalar[0]
            context.update(empty_weights=weights, background_sem=background_sem)
        return context

    def _render(self, gaussians, sampled_xyz, context, include_empty=True):
        append_empty = include_empty and not self.separate_empty
        arguments = (self.prepare_gaussian_args(gaussians) if append_empty else
                     self.prepare_gaussian_args(gaussians, append_empty=False))
        means, opacity, semantics, scales, inverse = arguments
        batch, count = means.shape[:2]
        extra = dict(points_int=context['points_int']) if 'points_int' in context else {}
        optimized = (self.analytic_covariance or self.separate_empty or
                     self.incremental_densify or self.cache_render_grid)
        points = sampled_xyz if optimized else sampled_xyz.clone()
        logits = self.aggregator(points.float(), means, opacity.reshape(batch, count),
                                 semantics, scales, inverse, **extra)
        if self.separate_empty and include_empty:
            # Additive renderer outputs are not saved for backward; this
            # in-place epilogue avoids an additional dense logits allocation.
            logits.addcmul_(context['empty_weights'][:, None], context['background_sem'][None])
        return logits

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        prediction = []
        bin_logits = []
        density = []
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device) if 'occ_cam_mask' in metas.keys() else None
        occ_mask     = metas['occ_mask'].to(self.zero_tensor.device) if 'occ_mask' in metas.keys() else None
        occ_lidar_mask = metas['occ_lidar_mask'].to(self.zero_tensor.device) if 'occ_lidar_mask' in metas.keys() else None
        occ_nonempty_mask = metas['occ_nonempty_mask'].to(self.zero_tensor.device) if 'occ_nonempty_mask' in metas.keys() else None
        occ_loss_mask = metas['occ_loss_mask'].to(self.zero_tensor.device) if 'occ_loss_mask' in metas.keys() else occ_mask
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        context = self._react_context(sampled_xyz)
        rendered = {}
        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']
            parent = representation[idx].get('densify_parent_index')
            if self.incremental_densify and parent in rendered:
                # Provenance is emitted only by the append-only terminal
                # DensifyOnly path. Reuse parent logits including background.
                delta = representation[idx]['delta_gaussian']
                semantics = rendered[parent] + self._render(
                    delta, sampled_xyz, context, include_empty=False)
            else:
                semantics = self._render(gaussians, sampled_xyz, context)
            if self.incremental_densify:
                rendered[idx] = semantics
            if self.use_localaggprob:
                if self.combine_geosem:
                    sem = semantics[0][:, :-1] * semantics[1].unsqueeze(-1)
                    geo = 1 - semantics[1].unsqueeze(-1)
                    geosem = torch.cat([sem, geo], dim=-1)
                else:
                    geosem = semantics[0]
                    
                prediction.append(geosem[None].transpose(1, 2))
                bin_logits.append(semantics[1][None])
                density.append(semantics[2][None])
            else:
                prediction.append(semantics[None].transpose(1, 2))
        
        if self.use_localaggprob and not self.combine_geosem:
            threshold = kwargs.get("sigmoid_thresh", 0.5)
            final_semantics = prediction[-1].argmax(dim=1)
            final_occupancy = bin_logits[-1] > threshold
            final_prediction = torch.ones_like(final_semantics) * self.empty_label
            final_prediction[final_occupancy] = final_semantics[final_occupancy]
        else:
            final_prediction = prediction[-1].argmax(dim=1)

        allocation_aux = [
            item['allocation_aux']
            for item in representation
            if 'allocation_aux' in item
        ]
        
        return {
            'pred_occ': prediction,
            'bin_logits': bin_logits,
            'density': density,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_mask,
            'occ_loss_mask': occ_loss_mask,
            'occ_cam_mask': occ_cam_mask,
            'occ_lidar_mask': occ_lidar_mask,
            'occ_nonempty_mask': occ_nonempty_mask,
            'final_occ': final_prediction,
            'gaussian': representation[-1]['gaussian'],
            'gaussians': [r['gaussian'] for r in representation],
            'allocation_aux': allocation_aux,
        }
