"""Initialize fixed-count OPUS queries directly from sparse voxel features."""

import math

import numpy as np
import torch
from torch import nn
from mmseg.registry import MODELS

from .base_lifter import BaseLifter
from .spconv_backbone import MeanVFE, VoxelResBackBone8x
from .spconv_voxelize import VoxelGeneratorWrapper
from ..ops.sparse_voxel_sampling import build_voxel_memory, voxel_fps_indices


def _fallback_points(count):
    """Deterministic Halton positions for missing/empty LiDAR seed slots only."""
    result = []
    for index in range(1, count + 1):
        row = []
        for base in (2, 3, 5):
            value, fraction, remaining = 0.0, 1.0, index
            while remaining:
                fraction /= base
                value += fraction * (remaining % base)
                remaining //= base
            row.append(value)
        result.append(row)
    return torch.tensor(result, dtype=torch.float32)


@MODELS.register_module()
class FixedVoxelOPUSLifter(BaseLifter):
    """FPS on all active 8x voxels; retain all voxels as decoder memory.

``voxel_size`` is the feature-grid pitch in metres. Raw points are voxelized
at voxel_size/8. Coordinates use actual pitch and scene origin, never a
division by the sparse output shape (which can shrink along Z due to padding).
"""

    def __init__(self, num_queries=600, embed_dims=256, voxel_size=0.4,
                 pc_range=(-40., -40., -1., 40., 40., 5.4),
                 max_num_points_per_voxel=5, max_num_voxels=1600000,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if num_queries < 1 or voxel_size <= 0:
            raise ValueError('num_queries and voxel_size must be positive')
        if len(pc_range) != 6 or any(pc_range[i + 3] <= pc_range[i] for i in range(3)):
            raise ValueError('pc_range must contain increasing XYZ bounds')
        self.num_queries = int(num_queries)
        self.embed_dims = int(embed_dims)
        self.pc_range = tuple(pc_range)
        self.voxel_size = (float(voxel_size),) * 3
        input_pitch = voxel_size / 8.0
        shape_xyz = [int(math.ceil((pc_range[i + 3] - pc_range[i]) / input_pitch))
                     for i in range(3)]
        self.input_shape = tuple(shape_xyz[::-1])
        z, y, x = self.input_shape
        z = (((z + 1) // 2 + 1) // 2 - 1) // 2
        self.output_shape = (z, (y + 7) // 8, (x + 7) // 8)
        if min(self.output_shape) < 1:
            raise ValueError('point-cloud grid is too small for the 8x sparse backbone')
        self.lidar_processor = VoxelGeneratorWrapper(
            vsize_xyz=[input_pitch] * 3, coors_range_xyz=pc_range,
            num_point_features=4, max_num_points_per_voxel=max_num_points_per_voxel,
            max_num_voxels=max_num_voxels)
        self.lidar_vfe = MeanVFE(num_point_features=4)
        self.lidar_backbone = VoxelResBackBone8x(4, embed_dims, self.input_shape)
        self.memory_projection = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                               nn.LayerNorm(embed_dims))
        self.register_buffer('fallback_points', _fallback_points(self.num_queries))

    def _voxelize(self, lidar_points, device):
        voxels, coordinates, counts = [], [], []
        for batch_index, points in enumerate(lidar_points):
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            points = np.asarray(points, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 4:
                raise ValueError('lidar_points must contain [N,>=4] XYZ/intensity arrays')
            voxel, coordinate, count = self.lidar_processor.generate(
                np.ascontiguousarray(points[:, :4]))
            if len(coordinate) == 0:
                continue
            voxels.append(torch.from_numpy(voxel).to(device))
            counts.append(torch.from_numpy(count).to(device))
            coordinate = torch.from_numpy(coordinate).to(device=device, dtype=torch.int32)
            batch_column = coordinate.new_full((coordinate.shape[0], 1), batch_index)
            coordinates.append(torch.cat([batch_column, coordinate], dim=-1))
        if not voxels:
            return None
        return dict(voxels=torch.cat(voxels), voxel_num_points=torch.cat(counts),
                    voxel_coords=torch.cat(coordinates), batch_size=len(lidar_points))

    def initialize_queries(self, memory):
        """Pair each FPS coordinate with exactly the same selected feature row."""
        features, indices = memory['features'], memory['indices']
        batch_size = memory['index_map'].shape[0]
        points = self.fallback_points.to(features)[None].repeat(batch_size, 1, 1)
        query_features = features.new_zeros((batch_size, self.num_queries, self.embed_dims))
        voxel_valid = torch.zeros((batch_size, self.num_queries), dtype=torch.bool,
                                  device=features.device)
        lower = features.new_tensor(self.pc_range[:3])
        extent = features.new_tensor(self.pc_range[3:]) - lower
        pitch = features.new_tensor(self.voxel_size)
        for batch_index in range(batch_size):
            rows = torch.nonzero(indices[:, 0] == batch_index, as_tuple=False).flatten()
            selected = voxel_fps_indices(indices[rows, 1:], memory['spatial_shape'],
                                          self.voxel_size, self.num_queries)
            rows = rows[selected]
            count = rows.numel()
            xyz = (indices[rows, 1:].flip(-1).to(features.dtype) + 0.5) * pitch + lower
            points[batch_index, :count] = (xyz - lower) / extent
            query_features[batch_index, :count] = features[rows]
            voxel_valid[batch_index, :count] = True
        return dict(query_points=points, query_features=query_features,
                    query_valid_mask=torch.ones_like(voxel_valid),
                    query_voxel_valid_mask=voxel_valid)

    def forward(self, imgs, metas, **kwargs):
        lidar_points = metas.get('lidar_points')
        if lidar_points is None or len(lidar_points) != imgs.shape[0]:
            raise ValueError('FixedVoxelOPUSLifter requires one lidar_points array per sample')
        # spconv and the decoder memory stay FP32 even under runner autocast.
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            voxel_dict = self._voxelize(lidar_points, imgs.device)
            if voxel_dict is None:
                features = self.fallback_points.new_zeros((0, self.embed_dims))
                indices = torch.empty((0, 4), dtype=torch.int32, device=imgs.device)
                spatial_shape = self.output_shape
            else:
                sparse = self.lidar_backbone(self.lidar_vfe(voxel_dict))['out_tensor']
                features = self.memory_projection(sparse.features.float())
                indices, spatial_shape = sparse.indices, sparse.spatial_shape
            memory = build_voxel_memory(features, indices, spatial_shape, imgs.shape[0],
                                         self.voxel_size, self.pc_range[:3])
            output = self.initialize_queries(memory)
        output['voxel_memory'] = memory
        return output
