"""Final-point Semantic Gaussian residual head for the strict OPUS decoder.

The Gaussian centres are exactly the final OPUS points.  This module therefore
implements a sparse, normalized Gaussian message-passing layer on the final
point set rather than a dense voxel renderer.
"""
import torch
import torch.nn.functional as F
from torch import nn
from mmengine.registry import MODELS

from .opus_head import OPUSHead

try:
    from mmcv.ops import knn as mmcv_knn
except (ImportError, ModuleNotFoundError):
    mmcv_knn = None


class _PointAnchoredGaussianResidual(nn.Module):
    """Predict Gaussian attributes and aggregate residuals on a point KNN graph."""
    def __init__(self, embed_dims, num_children, num_classes, pc_range, scale_range,
                 initial_scale=.35, initial_opacity=.02, num_neighbors=8,
                 include_self=True, query_chunk_size=256):
        super().__init__()
        self.num_children = num_children
        self.num_neighbors = num_neighbors
        self.include_self = include_self
        self.query_chunk_size = query_chunk_size
        self.register_buffer('pc_lower', torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer('pc_upper', torch.tensor(pc_range[3:], dtype=torch.float32))
        self.register_buffer('scale_min', torch.tensor(scale_range[0], dtype=torch.float32))
        self.register_buffer('scale_max', torch.tensor(scale_range[1], dtype=torch.float32))
        self.slot_embedding = nn.Embedding(num_children, embed_dims)
        self.position_encoder = nn.Sequential(nn.Linear(3, embed_dims), nn.ReLU(inplace=True),
                                              nn.Linear(embed_dims, embed_dims))
        self.trunk = nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
                                   nn.Linear(embed_dims, embed_dims // 2), nn.ReLU(inplace=True))
        hidden_dims = embed_dims // 2
        self.scale_head = nn.Linear(hidden_dims, 3)
        self.opacity_head = nn.Linear(hidden_dims, 1)
        self.residual_head = nn.Linear(hidden_dims, num_classes)
        scale_prob = ((torch.tensor(initial_scale) - self.scale_min) /
                      (self.scale_max - self.scale_min)).clamp(.001, .999)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, torch.logit(scale_prob).mean().item())
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.constant_(self.opacity_head.bias, torch.logit(torch.tensor(initial_opacity)).item())
        # Zero residual makes the Gaussian-refined branch initially identical
        # to the detached OPUS classifier, preserving a stable warm-up point.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _knn(self, points, neighbours):
        """Return [N,K] neighbour indices without materialising an N×N matrix."""
        count = points.shape[0]
        neighbours = min(neighbours, count)
        if points.is_cuda and mmcv_knn is not None:
            # mmcv.knn returns [B,K,N_query], indices into its xyz argument.
            return mmcv_knn(neighbours, points[None].contiguous(), points[None].contiguous())[0].transpose(0, 1)
        indices = []
        for start in range(0, count, self.query_chunk_size):
            distance = torch.cdist(points[start:start + self.query_chunk_size].float(), points.float())
            indices.append(distance.topk(neighbours, dim=-1, largest=False).indices)
        return torch.cat(indices, dim=0)

    def forward(self, token, points, valid_mask):
        """Return Gaussian attributes and normalized local semantic residuals.

        Args:
            token: Final OPUS query tokens [B,Q,D].
            points: Final OPUS child points in metres [B,Q,R,3].
            valid_mask: Query validity [B,Q].

        Returns:
            dict with flattened point-aligned tensors: ``points [B,G,3]``,
            ``scales [B,G,3]``, ``opacities [B,G,1]``, ``residual [B,G,C]``
            and ``valid_mask [B,G]`` where G=Q*R.
        """
        batch, queries, children, _ = points.shape
        if children != self.num_children:
            raise ValueError('final OPUS child count does not match Gaussian slot count')
        slot = self.slot_embedding.weight[None, None]
        normalized = ((points - self.pc_lower.to(points)) /
                      (self.pc_upper.to(points) - self.pc_lower.to(points))).clamp(0., 1.)
        child_feature = token[:, :, None] + slot + self.position_encoder(normalized)
        hidden = self.trunk(child_feature)
        scales = self.scale_min.to(points) + (self.scale_max.to(points) - self.scale_min.to(points)) * \
            torch.sigmoid(self.scale_head(hidden))
        opacities = torch.sigmoid(self.opacity_head(hidden))
        seed = self.residual_head(hidden)
        flat_points = points.flatten(1, 2)
        flat_scales = scales.flatten(1, 2)
        flat_opacities = opacities.flatten(1, 2)
        flat_seed = seed.flatten(1, 2)
        flat_valid = valid_mask[:, :, None].expand(-1, -1, children).flatten(1)
        residual = torch.zeros_like(flat_seed)
        for batch_index in range(batch):
            active = flat_valid[batch_index]
            if not active.any():
                continue
            active_points = flat_points[batch_index, active]
            active_scales = flat_scales[batch_index, active]
            active_opacity = flat_opacities[batch_index, active]
            active_seed = flat_seed[batch_index, active]
            neighbour_count = self.num_neighbors if self.include_self else self.num_neighbors + 1
            neighbour_index = self._knn(active_points, neighbour_count)
            if not self.include_self and neighbour_index.shape[1] > 1:
                # KNN includes the query itself at index zero for identical
                # point/query sets; remove it to measure pure neighbourhood use.
                neighbour_index = neighbour_index[:, 1:]
            neighbour_points = active_points[neighbour_index]
            neighbour_scales = active_scales[neighbour_index].clamp_min(1e-4)
            neighbour_opacity = active_opacity[neighbour_index]
            neighbour_seed = active_seed[neighbour_index]
            delta = (active_points[:, None] - neighbour_points) / neighbour_scales
            kernel = torch.exp(-.5 * delta.square().sum(dim=-1))
            weights = neighbour_opacity.squeeze(-1) * kernel
            # Normalization makes semantic magnitude invariant to local
            # Gaussian count, unlike the existing unnormalized localagg head.
            message = (weights[..., None] * neighbour_seed).sum(dim=1) / \
                weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            residual[batch_index, active] = message
        return dict(points=flat_points, scales=flat_scales, opacities=flat_opacities,
                    residual=residual, valid_mask=flat_valid)


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
