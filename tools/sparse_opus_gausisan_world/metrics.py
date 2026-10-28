"""Seven-horizon full-space metrics; mIoU always averages 17 occupied classes."""

import numpy as np
import torch

CLASS_NAMES = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
]


class HorizonMetrics:
    def __init__(self):
        self.confusion = np.zeros((7, 18, 18), dtype=np.int64)
        self.samples = 0
        self.trajectory_squared = np.zeros(6, dtype=np.float64)
        self.trajectory_l2 = np.zeros(6, dtype=np.float64)
        self.trajectory_count = np.zeros(6, dtype=np.int64)

    def add(self, result, metas):
        grids = [result["final_occ_grid"]] + result["future_occ_grids"]
        if len(grids) != 7:
            raise ValueError("evaluation must predict all six future horizons")
        for horizon, prediction in enumerate(grids):
            target = (
                metas["occ_label"]
                if horizon == 0
                else metas["future_occ_labels"][:, horizon - 1]
            )
            pred, gt = prediction.long().reshape(-1), target.long().reshape(-1)
            if pred.shape != gt.shape:
                raise ValueError("prediction and target grid sizes differ")
            if ((pred < 0) | (pred >= 18)).any():
                raise ValueError("invalid predicted class")
            valid = (gt >= 0) & (gt < 18)
            histogram = torch.bincount(gt[valid] * 18 + pred[valid], minlength=324)
            self.confusion[horizon] += histogram.reshape(18, 18).cpu().numpy()
        trajectories = result["pred_traj"]
        errors = (
            (trajectories - metas["temporal_trajs"].to(trajectories)).square().sum(-1)
        )
        count = errors.shape[1]
        self.trajectory_squared[:count] += errors.sum(0).cpu().numpy()
        self.trajectory_l2[:count] += errors.sqrt().sum(0).cpu().numpy()
        self.trajectory_count[:count] += errors.shape[0]
        self.samples += grids[0].shape[0]

    def results(self):
        horizons = []
        for index, hist in enumerate(self.confusion):
            tp = hist.diagonal().astype(np.float64)
            gt, pred = hist.sum(1), hist.sum(0)
            union = gt + pred - tp
            iou = np.divide(tp, union, out=np.zeros(18), where=union > 0) * 100
            occupied_tp = float(hist[:17, :17].sum())
            occupied_pred, occupied_gt = float(hist[:, :17].sum()), float(
                hist[:17].sum()
            )
            occupied_union = occupied_pred + occupied_gt - occupied_tp
            horizons.append(
                dict(
                    index=index,
                    seconds=index * 0.5,
                    miou=float(iou[:17].sum() / 17),
                    class_iou=iou.tolist(),
                    class_gt_count=gt.tolist(),
                    class_union=union.tolist(),
                    absent_union_classes=[
                        CLASS_NAMES[c] for c in range(18) if union[c] == 0
                    ],
                    occupied_iou=100 * occupied_tp / max(occupied_union, 1),
                    occupied_precision=100 * occupied_tp / max(occupied_pred, 1),
                    occupied_recall=100 * occupied_tp / max(occupied_gt, 1),
                )
            )
        return dict(
            num_samples=self.samples,
            horizons=horizons,
            selection_mean_miou=float(
                np.mean([horizons[h]["miou"] for h in (0, 2, 4, 6)])
            ),
            all_horizons_mean_miou=float(np.mean([h["miou"] for h in horizons])),
            trajectory_l2_m=(
                self.trajectory_l2 / np.maximum(self.trajectory_count, 1)
            ).tolist(),
            trajectory_mse_m2=(
                self.trajectory_squared / np.maximum(self.trajectory_count, 1)
            ).tolist(),
            miou_divisor=17,
            evaluation_mask="none",
        )
