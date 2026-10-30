"""Run React's unchanged CPU voxelizer inside a data-loading worker."""

import numpy as np
import torch

from . import OPENOCC_TRANSFORMS


@OPENOCC_TRANSFORMS.register_module()
class CastReactImage:
    """Cast after FP32 augmentation/normalization, before collation/pinning."""
    def __call__(self, results):
        results['img'] = results['img'].to(torch.bfloat16)
        return results


@OPENOCC_TRANSFORMS.register_module()
class PrepareReactVoxels:
    def __init__(self, pc_range, voxel_size=.4):
        # Lists also survive older DataLoader pin_memory implementations,
        # which convert ordinary tuples to lists during recursive traversal.
        self.spec = dict(vsize_xyz=[voxel_size / 8] * 3,
                         coors_range_xyz=list(pc_range), num_point_features=4,
                         max_num_points_per_voxel=5, max_num_voxels=1600000)
        self._generator = None

    def __getstate__(self):
        # The native generator is neither pickled nor shared between workers.
        return dict(spec=self.spec, _generator=None)

    def __call__(self, results):
        if self._generator is None:
            from model.lifter.spconv_voxelize import VoxelGeneratorWrapper
            self._generator = VoxelGeneratorWrapper(**self.spec)
        points = np.ascontiguousarray(results['lidar_points'], dtype=np.float32)
        voxels, coordinates, counts = self._generator.generate(points)
        results['react_voxels'] = dict(voxels=voxels, voxel_coords=coordinates,
                                       voxel_num_points=counts, spec=self.spec)
        return results
