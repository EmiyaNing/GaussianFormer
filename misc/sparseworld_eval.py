"""Metrics and coordinate utilities used only by SparseWorld future evaluation.

The utilities are deliberately independent from the generic evaluator: they
are instantiated only when a config explicitly enables ``sparseworld_future_eval``.
Consequently, standard OPUS/GaussianFormer evaluation contracts are unchanged.
"""
import numpy as np
import torch
import torch.distributed as dist

from .occ3d_nus_metrics import Metric_mIoU


def points_current_to_future(points, future_to_current):
    """Map row-vector points from current ego coordinates into a future ego frame.

    Args:
        points: Current-frame world points, float tensor ``[N, 3]``.
        future_to_current: Homogeneous transform ``[4, 4]`` that maps a
            future-ego point to the current ego frame.

    Returns:
        The same points represented in the future ego frame, ``[N, 3]``.

    Notes:
        The training loss maps future GT as ``p_current = p_future @ R.T + t``.
        This function applies the exact inverse convention to predictions so
        that they can be rasterized and compared with native future Occ3D GT.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError('points must have shape [N, 3]')
    if future_to_current.shape != (4, 4):
        raise ValueError('future_to_current must have shape [4, 4]')
    current_to_future = torch.linalg.inv(future_to_current.to(points))
    rotation = current_to_future[:3, :3]
    translation = current_to_future[:3, 3]
    return points @ rotation.T + translation


class SparseWorldHorizonMetric:
    """Accumulate semantic mIoU and binary occupancy IoU for one horizon.

    ``Metric_mIoU`` provides the established Occ3D semantic IoU protocol.  A
    separate intersection/union accumulator reports binary ``OccIoU`` where
    every class except ``free_label`` is occupied.  Metrics are intentionally
    per horizon; no prediction or confusion matrix is averaged across time.
    """
    def __init__(self, num_classes=18, use_lidar_mask=False, use_image_mask=True,
                 free_label=17, report_occ_iou=True):
        self.semantic = Metric_mIoU(
            num_classes=num_classes, use_lidar_mask=use_lidar_mask,
            use_image_mask=use_image_mask)
        self.num_classes = int(num_classes)
        self.free_label = int(free_label)
        self.report_occ_iou = bool(report_occ_iou)
        self.occupied_intersection = np.int64(0)
        self.occupied_union = np.int64(0)

    def _evaluation_mask(self, label, lidar_mask, camera_mask):
        """Select exactly the validity mask used by the semantic metric."""
        if self.semantic.use_image_mask:
            if camera_mask is None:
                raise ValueError('camera mask is required by the selected future evaluation protocol')
            mask = camera_mask.astype(bool)
        elif self.semantic.use_lidar_mask:
            if lidar_mask is None:
                raise ValueError('lidar mask is required by the selected future evaluation protocol')
            mask = lidar_mask.astype(bool)
        else:
            mask = np.ones_like(label, dtype=bool)
        # Match Metric_mIoU.hist_info: ignore invalid/void labels consistently.
        return mask & (label >= 0) & (label < self.num_classes)

    def add_batch(self, prediction, label, lidar_mask=None, camera_mask=None):
        """Add one future-frame prediction in that future frame's native grid."""
        if prediction.shape != label.shape:
            raise ValueError('future prediction and label grid shapes must match')
        self.semantic.add_batch(prediction, label, lidar_mask, camera_mask)
        if not self.report_occ_iou:
            return
        mask = self._evaluation_mask(label, lidar_mask, camera_mask)
        prediction_occupied = prediction[mask] != self.free_label
        label_occupied = label[mask] != self.free_label
        self.occupied_intersection += np.logical_and(prediction_occupied, label_occupied).sum()
        self.occupied_union += np.logical_or(prediction_occupied, label_occupied).sum()

    def synchronize(self, device):
        """All-reduce one horizon's confusion matrix and occupancy counts."""
        if not dist.is_available() or not dist.is_initialized():
            return
        semantic_hist = torch.as_tensor(self.semantic.hist, dtype=torch.float64, device=device)
        semantic_count = torch.tensor(self.semantic.cnt, dtype=torch.int64, device=device)
        occupancy = torch.tensor([self.occupied_intersection, self.occupied_union],
                                 dtype=torch.int64, device=device)
        dist.all_reduce(semantic_hist, op=dist.ReduceOp.SUM)
        dist.all_reduce(semantic_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(occupancy, op=dist.ReduceOp.SUM)
        self.semantic.hist = semantic_hist.cpu().numpy()
        self.semantic.cnt = int(semantic_count.item())
        self.occupied_intersection = np.int64(occupancy[0].item())
        self.occupied_union = np.int64(occupancy[1].item())

    def results(self):
        """Return serializable per-class IoU, mIoU, and optional OccIoU."""
        iou = self.semantic.per_class_iu(self.semantic.hist) * 100.0
        output = {
            'num_samples': self.semantic.cnt,
            'mIoU': float(np.nanmean(iou[:self.num_classes - 1])),
        }
        for class_index, class_name in enumerate(self.semantic.class_names[:self.num_classes - 1]):
            output[f'iou/{class_name}'] = float(iou[class_index])
        if self.report_occ_iou:
            output['OccIoU'] = float(
                100.0 * self.occupied_intersection / max(int(self.occupied_union), 1))
        return output
