"""Exact-support React rendering with coarse CSR bins and warp backward."""

import math

import torch
from torch.autograd.function import once_differentiable

from .react_local_aggregate import PreparedReactAggregator


def extension():
    try:
        from .react_render_ops import _C
    except ImportError as error:
        raise RuntimeError(
            'Build the round-three renderer: cd model/head/react_render_ops && '
            'python3.10 setup.py build_ext --inplace') from error
    return _C


class _TiledAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pts, cells, means, centers, opacity, semantics, radii,
                inverse, heads, next_query, h, w, d, tile, capacity):
        pts, means, opacity, semantics, inverse = [value.contiguous().float()
            for value in (pts, means, opacity, semantics, inverse)]
        cells, centers, radii = [value.contiguous().int()
                                 for value in (cells, centers, radii)]
        native = extension()
        offsets, ids = native.tile_bins(centers, radii, h, w, d, tile, capacity)
        output = native.render(pts, cells, means, centers, radii, opacity,
                               semantics, inverse, offsets, ids, h, w, d, tile)
        # Bins are forward-only. Backward enumerates each exact AABB directly,
        # so no duplicated neighbor list is retained by any decoder's graph.
        ctx.save_for_backward(pts, means, centers, radii, opacity, semantics,
                              inverse, heads, next_query)
        ctx.shape = (h, w, d)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        values = ctx.saved_tensors
        dm, dop, ds, dc = extension().backward(
            *values, grad.contiguous().float(), *ctx.shape)
        return None, None, dm, None, dop, ds, None, dc, None, None, None, None, None, None, None


class TiledReactAggregator(PreparedReactAggregator):
    def __init__(self, *args, tile_size=4, radius_hint=5, cache_query_map=True, **kwargs):
        super().__init__(*args, **kwargs)
        if tile_size < 1 or radius_hint < 0:
            raise ValueError('tile_size must be positive and radius_hint nonnegative')
        self.tile_size = int(tile_size)
        self.radius_hint = int(radius_hint)
        self.cache_query_map = cache_query_map
        self.register_buffer('_query_heads', torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer('_query_next', torch.empty(0, dtype=torch.int32), persistent=False)
        self._query_ref = None
        self._query_version = None

    def _clear_query_cache(self):
        self._query_ref = None
        self._query_version = None
        self._query_heads = self._query_heads.new_empty(0)
        self._query_next = self._query_next.new_empty(0)

    def _apply(self, fn, recurse=True):
        self._clear_query_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._clear_query_cache()

    def prepare_query_map(self, cells):
        try:
            version = cells._version
        except RuntimeError:
            # inference_mode tensors have no version counter. Build a fresh
            # map rather than caching a mutable tensor without invalidation.
            version = None
        if (self.cache_query_map and self._query_ref is cells and version is not None
                and version == self._query_version):
            return self._query_heads, self._query_next
        heads, next_query = extension().query_map(cells.contiguous(), self.H, self.W, self.D)
        if self.cache_query_map and version is not None:
            self._query_ref = cells
            self._query_version = version
            self._query_heads, self._query_next = heads, next_query
        return heads, next_query

    def forward(self, pts, means3D, opacities, semantics, scales, cov3D, points_int=None):
        arguments = self._prepare_inputs(pts, means3D, opacities, semantics,
                                        scales, cov3D, points_int)
        cells = arguments[1]
        if torch.is_grad_enabled() and any(value.requires_grad for value in arguments):
            heads, next_query = self.prepare_query_map(cells)
        else:
            heads = next_query = cells.new_empty(0)
        tile = self.tile_size
        capacity = self.capacity_for(means3D.shape[1])
        return _TiledAggregate.apply(*arguments, heads, next_query,
                                      self.H, self.W, self.D, tile, capacity)

    def capacity_for(self, gaussian_count):
        tile = self.tile_size
        grid = [math.ceil(size / tile) for size in (self.H, self.W, self.D)]
        overlap = (2 * self.radius_hint + tile - 1) // tile + 1
        capacity = gaussian_count * math.prod(min(count, overlap) for count in grid)
        if capacity >= 2**31 or gaussian_count >= 2**31:
            raise ValueError('React workspace exceeds int32 Gaussian/index capacity')
        return capacity
