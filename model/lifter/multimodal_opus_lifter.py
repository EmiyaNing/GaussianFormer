"""Construct variable-length OPUS queries from sparse LiDAR voxels.

This module is the LiDAR branch of the multi-modal OPUS model. It voxelizes
each sample independently, extracts 8x sparse voxel features, and uses them
in two complementary roles:

1. The 8x sparse tensor is returned as LiDAR memory for decoder ball-query
   fusion.
2. All active 16x voxels become OPUS queries. Dense batches are structurally
   reduced once more to 32x, rather than using score-based Top-K selection.

Samples naturally contain different numbers of active voxels. The module pads
only at the final batch boundary and returns ``query_valid_mask`` so that
padding is ignored by attention, loss calculation, rasterization, and
visualization.
"""

import math

import numpy as np
import torch
from mmseg.registry import MODELS
from torch import nn

from .base_lifter import BaseLifter
from .spconv_backbone import MeanVFE, VoxelResBackBone8x
from .spconv_utils import spconv
from .spconv_voxelize import VoxelGeneratorWrapper


def _build_xy_sparse_downsampler(input_channels, output_channels, indice_key):
    """Create one sparse 2x XY reduction while retaining the Z resolution.

    spconv stores spatial dimensions as ``[z, y, x]``. Consequently the
    stride ``(1, 2, 2)`` doubles the LiDAR feature stride in the ground plane
    without making the comparatively short vertical axis unnecessarily coarse.
    """
    return spconv.SparseSequential(
        spconv.SparseConv3d(
            input_channels,
            output_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            indice_key=indice_key,
        ),
        nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


@MODELS.register_module()
class MultiModalOPUSLifter(BaseLifter):
    """Generate LiDAR-initialized OPUS queries and local sparse memory.

    Args:
        embed_dims: Shared OPUS feature dimension.
        pc_range: Ego-frame point-cloud bounds in ``[xmin, ymin, zmin, xmax,
            ymax, zmax]`` order.
        voxel_size: Target 8x voxel width in metres. Input voxelization uses
            ``voxel_size / 8`` so that ``VoxelResBackBone8x`` emits this size.
        query_stride: Preferred query resolution, either 16 or 32.
        query_cap: If any sample has more active 16x voxels than this value,
            use the 32x structural reduction for the whole batch. This is a
            resolution switch, not a score-based hard selection.
        max_num_points_per_voxel: Maximum raw LiDAR points aggregated by VFE.
        max_num_voxels: Voxelization capacity for one sample.
        num_fallback_queries: Learned queries appended to every sample. They
            keep the decoder valid for empty LiDAR frames and provide image-only
            coverage outside occupied LiDAR voxels.
    """

    def __init__(self, embed_dims=128, pc_range=(-40., -40., -1., 40., 40., 5.4),
                 voxel_size=0.4, query_stride=16, query_cap=4096,
                 max_num_points_per_voxel=5, max_num_voxels=1600000,
                 num_fallback_queries=32, **kwargs):
        super().__init__(**kwargs)
        if query_stride not in (16, 32):
            raise ValueError('query_stride must be 16 or 32')

        self.embed_dims = embed_dims
        self.pc_range = tuple(pc_range)
        self.query_stride = query_stride
        self.query_cap = query_cap
        self.num_fallback_queries = num_fallback_queries

        input_voxel_size = voxel_size / 8.0
        input_grid_shape_zyx = self._compute_input_grid_shape(input_voxel_size)
        self.lidar_processor = VoxelGeneratorWrapper(
            vsize_xyz=[input_voxel_size] * 3,
            coors_range_xyz=pc_range,
            num_point_features=4,
            max_num_points_per_voxel=max_num_points_per_voxel,
            max_num_voxels=max_num_voxels,
        )
        self.lidar_vfe = MeanVFE(num_point_features=4)
        self.lidar_backbone = VoxelResBackBone8x(
            4, embed_dims, input_grid_shape_zyx)

        # x_conv4 has 128 channels at 8x stride; OPUS uses embed_dims channels.
        self.memory_feature_projection = nn.Sequential(
            nn.Linear(128, embed_dims),
            nn.LayerNorm(embed_dims),
        )
        self.query_downsample_16x = _build_xy_sparse_downsampler(
            128, embed_dims, 'opus_query16')
        self.query_downsample_32x = _build_xy_sparse_downsampler(
            embed_dims, embed_dims, 'opus_query32')

        self.lidar_source_embedding = nn.Parameter(torch.zeros(embed_dims))
        self.fallback_query_features = nn.Parameter(
            torch.empty(num_fallback_queries, embed_dims))
        # Stored as logits so the final sigmoid keeps fallback points in [0, 1].
        self.fallback_query_point_logits = nn.Parameter(
            torch.empty(num_fallback_queries, 3))
        self.init_weights()

    def _compute_input_grid_shape(self, input_voxel_size):
        """Return the initial sparse grid shape in spconv's ``[z, y, x]`` order."""
        lower_bound = self.pc_range[:3]
        upper_bound = self.pc_range[3:]
        # ``ceil`` retains the full configured range when its extent is not an
        # integer multiple of the input voxel width.
        grid_shape_xyz = [
            int(math.ceil((upper - lower) / input_voxel_size))
            for lower, upper in zip(lower_bound, upper_bound)
        ]
        return grid_shape_xyz[::-1]

    def init_weights(self):
        """Initialize learned fallback queries and the LiDAR feature adapter."""
        nn.init.normal_(self.fallback_query_features, std=0.02)
        normalized_points = torch.empty_like(
            self.fallback_query_point_logits).uniform_(0.05, 0.95)
        self.fallback_query_point_logits.data.copy_(torch.logit(normalized_points))
        for module in self.memory_feature_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _ensure_four_point_features(points):
        """Return an ``[N, 4]`` NumPy array expected by the sparse voxelizer."""
        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points)
        point_array = points.detach().cpu().numpy()
        if point_array.ndim != 2:
            raise ValueError('Each lidar_points entry must have shape [N, C]')
        if point_array.shape[1] < 4:
            missing_features = 4 - point_array.shape[1]
            padding = np.zeros((point_array.shape[0], missing_features),
                               dtype=point_array.dtype)
            point_array = np.concatenate((point_array, padding), axis=1)
        return point_array[:, :4]

    def _voxelize_lidar_batch(self, lidar_points, device):
        """Voxelize a ragged LiDAR batch into the backbone input dictionary.

        ``VoxelGeneratorWrapper`` operates on one CPU NumPy array at a time.
        The resulting coordinates are prepended with the sample index to form
        spconv's required ``[batch, z, y, x]`` coordinate layout.
        """
        batched_voxels = []
        batched_point_counts = []
        batched_coordinates = []
        for batch_index, sample_points in enumerate(lidar_points):
            point_array = self._ensure_four_point_features(sample_points)
            voxels, coordinates_zyx, point_counts = self.lidar_processor.generate(point_array)
            if coordinates_zyx.shape[0] == 0:
                continue

            batched_voxels.append(
                torch.from_numpy(voxels).to(device=device, dtype=torch.float32))
            batched_point_counts.append(torch.from_numpy(point_counts).to(device=device))
            batch_column = torch.full(
                (coordinates_zyx.shape[0], 1), batch_index,
                device=device, dtype=torch.int32)
            coordinates = torch.from_numpy(coordinates_zyx).to(device=device)
            batched_coordinates.append(torch.cat((batch_column, coordinates), dim=1))

        if not batched_voxels:
            return None
        return {
            'voxels': torch.cat(batched_voxels),
            'voxel_num_points': torch.cat(batched_point_counts),
            'voxel_coords': torch.cat(batched_coordinates).int(),
        }

    @staticmethod
    def _unpack_sparse_tensor(sparse_tensor, batch_size, feature_projection,
                              source_embedding=None):
        """Split a sparse tensor into per-sample OPUS features and coordinates.

        Returns normalized coordinates in OPUS ``[x, y, z]`` order. Sparse
        indices use ``[batch, z, y, x]``; reversing the final axes aligns their
        order with the point-cloud range used by the decoder and OPUS head.
        """
        per_sample_features = []
        per_sample_points = []
        sparse_indices = sparse_tensor.indices
        spatial_shape_xyz = sparse_indices.new_tensor(
            sparse_tensor.spatial_shape[::-1]).clamp_min(1)

        for batch_index in range(batch_size):
            is_current_sample = sparse_indices[:, 0] == batch_index
            sparse_features = feature_projection(
                sparse_tensor.features[is_current_sample])
            if source_embedding is not None:
                sparse_features = sparse_features + source_embedding

            coordinates_xyz = sparse_indices[is_current_sample, 1:].flip(-1)
            # ``+ 0.5`` maps the integer cell index to its centre. Dividing by
            # the sparse-grid shape produces the normalized OPUS reference point.
            normalized_points = (
                coordinates_xyz.to(sparse_features.dtype) + 0.5
            ) / spatial_shape_xyz.to(sparse_features.dtype)
            per_sample_features.append(sparse_features)
            per_sample_points.append(normalized_points.clamp(0.0, 1.0))

        return per_sample_features, per_sample_points

    def _pack_variable_queries(self, per_sample_features, per_sample_points,
                                batch_size, device):
        """Pad variable LiDAR queries and append learned fallback queries.

        The padding values are zero. ``query_valid_mask`` marks LiDAR-derived
        and fallback queries, allowing downstream code to exclude only padding.
        """
        fallback_features = self.fallback_query_features
        fallback_points = self.fallback_query_point_logits.sigmoid()
        query_counts = [
            features.shape[0] + self.num_fallback_queries
            for features in per_sample_features
        ]
        max_query_count = max(query_counts)
        packed_features = fallback_features.new_zeros(
            (batch_size, max_query_count, self.embed_dims))
        packed_points = fallback_points.new_zeros(
            (batch_size, max_query_count, 3))
        query_valid_mask = torch.zeros(
            (batch_size, max_query_count), device=device, dtype=torch.bool)

        for batch_index, (features, points) in enumerate(
                zip(per_sample_features, per_sample_points)):
            lidar_query_count = features.shape[0]
            if lidar_query_count:
                packed_features[batch_index, :lidar_query_count] = features
                packed_points[batch_index, :lidar_query_count] = points

            fallback_start = lidar_query_count
            fallback_end = fallback_start + self.num_fallback_queries
            packed_features[batch_index, fallback_start:fallback_end] = fallback_features
            packed_points[batch_index, fallback_start:fallback_end] = fallback_points
            query_valid_mask[batch_index, :fallback_end] = True

        return packed_features, packed_points, query_valid_mask

    def _empty_lidar_outputs(self, batch_size, device):
        """Create valid fallback-only queries for a batch with no active voxels."""
        empty_features = [
            self.fallback_query_features.new_zeros((0, self.embed_dims))
            for _ in range(batch_size)
        ]
        empty_points = [
            self.fallback_query_point_logits.new_zeros((0, 3))
            for _ in range(batch_size)
        ]
        query_features, query_points, query_valid_mask = self._pack_variable_queries(
            empty_features, empty_points, batch_size, device)
        return dict(
            query_features=query_features,
            query_points=query_points,
            query_valid_mask=query_valid_mask,
            lidar_memory_features=empty_features,
            lidar_memory_points=empty_points,
        )

    def _select_query_tensor(self, memory_tensor, batch_size):
        """Produce 16x queries, optionally reducing the entire batch to 32x."""
        query_tensor_16x = self.query_downsample_16x(memory_tensor)
        use_32x_queries = self.query_stride == 32
        if self.query_cap is not None:
            active_queries_per_sample = torch.bincount(
                query_tensor_16x.indices[:, 0], minlength=batch_size)
            use_32x_queries = use_32x_queries or bool(
                (active_queries_per_sample > self.query_cap).any())
        if use_32x_queries:
            return self.query_downsample_32x(query_tensor_16x)
        return query_tensor_16x

    def forward(self, imgs, metas, **kwargs):
        """Build OPUS inputs from a batch of images and ragged LiDAR points.

        Returns:
            A dictionary containing padded ``query_features`` and
            ``query_points``, their boolean ``query_valid_mask``, and per-sample
            8x LiDAR memory features/points for decoder ball-query fusion.
        """
        lidar_points = metas.get('lidar_points')
        if lidar_points is None:
            raise KeyError(
                'MultiModalOPUSLifter requires lidar_points in dataset return_keys')

        batch_size = imgs.shape[0]
        device = imgs.device
        voxel_batch = self._voxelize_lidar_batch(lidar_points, device)
        if voxel_batch is None:
            return self._empty_lidar_outputs(batch_size, device)

        voxel_batch['batch_size'] = batch_size
        voxel_batch = self.lidar_vfe(voxel_batch)
        backbone_outputs = self.lidar_backbone(voxel_batch)
        # x_conv4 is the final 8x stage: local enough for ball-query memory and
        # sufficiently compact to avoid retaining pre-backbone sparse tensors.
        memory_tensor_8x = backbone_outputs['multi_scale_3d_features']['x_conv4']
        query_tensor = self._select_query_tensor(memory_tensor_8x, batch_size)

        memory_features, memory_points = self._unpack_sparse_tensor(
            memory_tensor_8x,
            batch_size,
            self.memory_feature_projection,
            self.lidar_source_embedding,
        )
        query_features, query_points = self._unpack_sparse_tensor(
            query_tensor,
            batch_size,
            nn.Identity(),
            self.lidar_source_embedding,
        )
        query_features, query_points, query_valid_mask = self._pack_variable_queries(
            query_features, query_points, batch_size, device)
        return dict(
            query_features=query_features,
            query_points=query_points,
            query_valid_mask=query_valid_mask,
            lidar_memory_features=memory_features,
            lidar_memory_points=memory_points,
        )
