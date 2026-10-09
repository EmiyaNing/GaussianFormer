"""OPUS-V2 continuous-point discretization via PFN and sparse 3D convolution."""
import torch
from torch import nn
import torch_scatter
from spconv.pytorch import SparseConvTensor, SparseConv3d, SubMConv3d, SparseSequential
from mmengine.registry import MODELS

from .base_head import BaseTaskHead


class _PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        linear_channels = out_channels if last_layer else out_channels // 2
        self.linear = nn.Linear(in_channels, linear_channels, bias=False)
        self.norm = nn.BatchNorm1d(linear_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, values, inverse):
        values = self.relu(self.norm(self.linear(values)))
        pooled = torch_scatter.scatter_max(values, inverse, dim=0)[0]
        if self.last_layer:
            return pooled
        return torch.cat([values, pooled[inverse]], dim=-1)


class _SparseDensifier(nn.Module):
    def __init__(self, point_channels, num_classes, pc_range, grid_shape,
                 pfn_channels=(64, 64)):
        super().__init__()
        self.register_buffer('lower', torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer('extent', torch.tensor(pc_range[3:], dtype=torch.float32)
                             - torch.tensor(pc_range[:3], dtype=torch.float32))
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.pfn = nn.ModuleList()
        channels = point_channels + 6
        for index, target in enumerate(pfn_channels):
            self.pfn.append(_PFNLayer(channels, target,
                                      last_layer=index == len(pfn_channels) - 1))
            channels = target
        self.classifier = SparseSequential(
            SparseConv3d(channels, channels, kernel_size=3, stride=1, padding=1,
                         indice_key='opusv2_densifier'),
            nn.ReLU(inplace=True),
            SubMConv3d(channels, num_classes, kernel_size=3, stride=1, padding=1,
                       indice_key='opusv2_classifier'))
        nn.init.constant_(self.classifier[-1].bias, -4.59511985013459)

    def forward(self, normalized_points, point_features):
        batch, queries, children = normalized_points.shape[:3]
        lower = self.lower.to(normalized_points)
        extent = self.extent.to(normalized_points)
        world = normalized_points * extent + lower
        grid = world.new_tensor(self.grid_shape)
        cell = torch.floor((world.detach() - lower) / extent * grid).long()
        for axis, bound in enumerate(self.grid_shape):
            cell[..., axis].clamp_(0, bound - 1)
        center = lower + (cell.to(world.dtype) + 0.5) / grid * extent
        query_center = world.mean(dim=2, keepdim=True)
        features = torch.cat([world - center, world - query_center, point_features], dim=-1)
        features = features.reshape(batch * queries * children, -1)
        batch_index = torch.arange(batch, device=world.device).view(batch, 1, 1)
        linear = (((batch_index * self.grid_shape[0] + cell[..., 0])
                   * self.grid_shape[1] + cell[..., 1])
                  * self.grid_shape[2] + cell[..., 2]).reshape(-1)
        unique, inverse = torch.unique(linear, sorted=True, return_inverse=True)
        for layer in self.pfn:
            features = layer(features, inverse)
        z = unique % self.grid_shape[2]
        y = unique // self.grid_shape[2] % self.grid_shape[1]
        x = unique // (self.grid_shape[1] * self.grid_shape[2]) % self.grid_shape[0]
        b = unique // (self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2])
        coords = torch.stack([b, x, y, z], dim=-1).int()
        sparse = SparseConvTensor(features, coords, self.grid_shape, batch)
        output = self.classifier(sparse)
        return output.features, output.indices.long()


@MODELS.register_module()
class OPUSV2SparseHead(BaseTaskHead):
    """Per-stage V2 densifiers and a sparse-to-dense evaluator adapter."""

    def __init__(self, num_stages=5, point_feature_dims=32, num_classes=17,
                 pc_range=(-40, -40, -1, 40, 40, 5.4),
                 grid_shape=(200, 200, 16), empty_label=17,
                 score_threshold=0.25, init_cfg=None):
        super().__init__(init_cfg)
        self.grid_shape = tuple(grid_shape)
        self.empty_label = empty_label
        self.score_threshold = score_threshold
        self.pc_range = tuple(pc_range)
        self.densifiers = nn.ModuleList([
            _SparseDensifier(point_feature_dims, num_classes, pc_range,
                             grid_shape) for _ in range(num_stages)])

    def forward(self, representation, metas=None, **kwargs):
        if len(representation) != len(self.densifiers):
            raise ValueError('V2 stage count mismatch')
        points, sparse_logits, sparse_coords = [], [], []
        lower = representation[0]['query_points'].new_tensor(self.pc_range[:3])
        extent = representation[0]['query_points'].new_tensor(self.pc_range[3:]) - lower
        for stage, state in enumerate(representation):
            normalized = state['query_points']
            batch, queries, children = normalized.shape[:3]
            points.append((normalized * extent + lower).reshape(batch, queries * children, 3))
            if self.training or stage == len(representation) - 1:
                score, coord = self.densifiers[stage](normalized, state['point_features'])
            else:
                score, coord = None, None
            sparse_logits.append(score)
            sparse_coords.append(coord)
        batch = points[-1].shape[0]
        voxels = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        final = torch.full((batch, voxels), self.empty_label,
                           device=points[-1].device, dtype=torch.long)
        logits = sparse_logits[-1]
        coords = sparse_coords[-1]
        confidence, category = logits.detach().sigmoid().max(dim=-1)
        selected = confidence > self.score_threshold
        coords = coords[selected]
        category = category[selected]
        flat = ((coords[:, 1] * self.grid_shape[1] + coords[:, 2])
                * self.grid_shape[2] + coords[:, 3])
        final[coords[:, 0], flat] = category
        label = metas['occ_label'].flatten(1).to(final.device)
        return {
            'v2_pred_points': points,
            'v2_sparse_logits': sparse_logits,
            'v2_sparse_coords': sparse_coords,
            'v2_init_points': kwargs.get('v2_init_points'),
            'final_occ': final,
            'sampled_label': label,
            'occ_cam_mask': metas.get('occ_cam_mask'),
            'occ_lidar_mask': metas.get('occ_lidar_mask'),
        }
