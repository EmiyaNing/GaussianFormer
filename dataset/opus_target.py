import numpy as np
from . import OPENOCC_TRANSFORMS


@OPENOCC_TRANSFORMS.register_module()
class PrepareOPUSTarget:
    """Creates a fixed-budget, deterministic occupied point set for OPUS loss."""

    def __init__(self, max_gt_points=76800, empty_label=17, mask_key='occ_loss_mask'):
        self.max_gt_points = max_gt_points
        self.empty_label = empty_label
        self.mask_key = mask_key

    def __call__(self, results):
        labels = results['occ_label'].reshape(-1)
        points = results['occ_xyz'].reshape(-1, 3)
        valid = labels != self.empty_label
        if self.mask_key in results:
            valid &= results[self.mask_key].reshape(-1).astype(bool)
        indices = np.flatnonzero(valid)
        selected = np.zeros(self.max_gt_points, dtype=np.int64)
        selected_valid = np.zeros(self.max_gt_points, dtype=bool)
        if len(indices):
            if len(indices) > self.max_gt_points:
                take = np.linspace(0, len(indices) - 1, self.max_gt_points, dtype=np.int64)
                indices = indices[take]
            selected[:len(indices)] = indices
            selected_valid[:len(indices)] = True
        results['opus_gt_points'] = points[selected].astype(np.float32, copy=False)
        results['opus_gt_labels'] = labels[selected].astype(np.int64, copy=False)
        results['opus_gt_valid'] = selected_valid
        # OPUS-V1 keeps every occupied voxel as a geometric target and uses
        # camera visibility only to adapt the GT-to-prediction penalty.  Keep
        # this bit alongside the compact target instead of losing it during
        # the fixed-budget packing step.
        camera_mask = results.get('occ_cam_mask')
        if camera_mask is None:
            selected_camera_valid = np.ones(self.max_gt_points, dtype=bool)
        else:
            selected_camera_valid = camera_mask.reshape(-1)[selected].astype(
                bool, copy=False)
            selected_camera_valid &= selected_valid
        results['opus_gt_camera_valid'] = selected_camera_valid
        return results
