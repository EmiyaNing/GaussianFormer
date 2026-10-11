"""Exact four-horizon SparseWorld occupancy evaluation protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


class SparseWorldStrictMetric:
    class_names = [
        'others', 'barrier', 'bicycle', 'bus', 'car',
        'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
        'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
        'terrain', 'manmade', 'vegetation', 'free']

    def __init__(self, boundary_stride=10, per_sample_path=None,
                 spatial_samples=0):
        self.horizons = (0, 2, 4, 6)
        self.semantic = {h: np.zeros((18, 18), dtype=np.float64) for h in self.horizons}
        self.binary = {h: np.zeros((2, 2), dtype=np.float64) for h in self.horizons}
        self.count = 0
        # Boundary IoU is a diagnostic, not part of the published protocol, and
        # it costs a per-class dilation per scene.  ``boundary_stride`` samples
        # every N-th scene (0 disables it) so it rides along with the normal
        # evaluation instead of needing a second pass over the data.
        self.boundary_stride = int(boundary_stride)
        self.boundary = {h: [] for h in self.horizons}
        # Per-sample confusion matrices.  Without these the report set could only
        # ever quote one aggregate per variant, which is why three reports
        # (P1-5, P1-6, P4-7) had to say "no confidence interval is possible".
        # Recording them costs 18x18 ints per (sample, horizon) and makes a
        # bootstrap CI and exact paired tests available offline.
        self.per_sample_path = Path(per_sample_path) if per_sample_path else None
        self.per_sample = []
        # P4.2 item 5 asks for visualisation of where the model misses boundaries
        # or hallucinates occupied space.  The confusion matrices above cannot
        # show *where*; capturing the flat voxel indices of FP/FN/TP for the first
        # few scenes makes a BEV (and, with a renderer, a 3D) view possible.  Flat
        # indices are lossless and compact -- only the non-TN voxels are kept.
        self.spatial_samples = int(spatial_samples)
        self.spatial = []

    @staticmethod
    def _hist(prediction, target, classes):
        prediction, target = prediction.reshape(-1), target.reshape(-1)
        valid = (target >= 0) & (target < classes)
        return np.bincount(classes * target[valid].astype(np.int64) +
                           prediction[valid].astype(np.int64),
                           minlength=classes ** 2).reshape(classes, classes)

    @staticmethod
    def _scene_labels(metas, batch_size):
        """Per-batch scene identifier, or ``None`` when unavailable.

        The per-sample dump needs the *scene*, not the sample index: the reduced
        split has 113 samples drawn from only 4 scenes, so samples within a scene
        are correlated and a bootstrap that treats them as independent units is
        anti-conservative (pseudo-replication).  With a scene label the caller can
        resample whole scenes instead.

        ``scene_token`` is preferred (the dataset carries it), and the scene name
        embedded in the camera filename -- ``<scene>__<camera>__<timestamp>.jpg``
        -- is the fallback, because ``img_filename`` is in the collator's
        ``return_keys`` while ``scene_token`` is not.
        """
        if not isinstance(metas, dict):
            return [None] * batch_size
        tokens = metas.get('scene_token')
        names = metas.get('img_filename')
        labels = []
        for index in range(batch_size):
            label = None
            if tokens is not None and index < len(tokens):
                candidate = tokens[index]
                if isinstance(candidate, str) and candidate:
                    label = candidate
            if label is None and names is not None and index < len(names):
                row = names[index]
                first = row[0] if isinstance(row, (list, tuple)) and row else row
                if isinstance(first, str) and first:
                    base = first.replace('\\', '/').rsplit('/', 1)[-1]
                    label = base.split('__')[0] or None
            labels.append(label)
        return labels

    def add_batch(self, result, metas):
        current = result['final_occ_grid']
        future = list(result['future_occ_grids'])
        indices = (None, 1, 3, 5)
        take_boundary = (self.boundary_stride > 0
                         and self.count % self.boundary_stride == 0)
        scenes = self._scene_labels(metas, current.shape[0])
        for batch in range(current.shape[0]):
            for horizon, index in zip(self.horizons, indices):
                prediction_tensor = current if index is None else future[index]
                prediction = prediction_tensor[batch].detach().cpu().numpy()
                target = (metas['occ_label'][batch] if index is None else
                          metas['future_occ_labels'][batch, index]).detach().cpu().numpy()
                semantic_hist = self._hist(prediction, target, 18)
                pred_occ = (prediction != 17).astype(np.int64)
                target_occ = (target != 17).astype(np.int64)
                binary_hist = self._hist(pred_occ, target_occ, 2)
                self.semantic[horizon] += semantic_hist
                self.binary[horizon] += binary_hist
                if (self.per_sample_path is not None
                        and self.count + batch < self.spatial_samples):
                    flat_target = target.reshape(-1)
                    flat_prediction = prediction.reshape(-1)
                    target_occ = flat_target != 17
                    predicted_occ = flat_prediction != 17
                    self.spatial.append({
                        'sample': self.count + batch,
                        'horizon': horizon,
                        'grid_shape': list(target.shape),
                        'tp': np.flatnonzero(target_occ & predicted_occ).astype(np.int32).tolist(),
                        'fp': np.flatnonzero(~target_occ & predicted_occ).astype(np.int32).tolist(),
                        'fn': np.flatnonzero(target_occ & ~predicted_occ).astype(np.int32).tolist(),
                    })
                if self.per_sample_path is not None:
                    self.per_sample.append({
                        'sample': self.count + batch,
                        'scene': scenes[batch],
                        'horizon': horizon,
                        'semantic': semantic_hist.astype(np.int64).tolist(),
                        'binary': binary_hist.astype(np.int64).tolist(),
                    })
                if take_boundary:
                    from misc.diagnostic_metrics import boundary_iou
                    score, _ = boundary_iou(
                        torch.as_tensor(prediction).unsqueeze(0),
                        torch.as_tensor(target).unsqueeze(0),
                        num_classes=18, radius=1, ignore_label=17)
                    if score == score:          # skip NaN scenes
                        self.boundary[horizon].append(score)
        self.count += current.shape[0]

    def synchronize(self, device):
        if not dist.is_available() or not dist.is_initialized():
            return
        for horizon in self.horizons:
            for table in (self.semantic, self.binary):
                tensor = torch.as_tensor(table[horizon], device=device, dtype=torch.float64)
                dist.all_reduce(tensor)
                table[horizon] = tensor.cpu().numpy()
        count = torch.tensor(self.count, device=device, dtype=torch.long)
        dist.all_reduce(count)
        self.count = int(count.item())

    @staticmethod
    def _iou(histogram):
        denominator = histogram.sum(0) + histogram.sum(1) - np.diag(histogram)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.diag(histogram) / denominator

    def results(self):
        result, semantic_mean, occupied_mean = {'num_samples': self.count}, [], []
        precision_mean, recall_mean = [], []
        # Pooled counts for the average precision/recall.  Averaging the
        # per-horizon *ratios* with nanmean would divide by the number of
        # horizons whose prediction (or target) was non-empty, which can differ
        # between variants -- the same variable-divisor failure as the mIoU
        # artifact in P1-5, one level down.  Pooling keeps the denominator fixed.
        pooled_tp = pooled_predicted = pooled_target = 0.0
        for horizon in self.horizons:
            semantic = self._iou(self.semantic[horizon])
            binary = self._iou(self.binary[horizon])
            miou = float(np.nanmean(semantic[:17]) * 100.)
            occupied = float(binary[1] * 100.)
            result[f'{horizon}s/mIoU'] = round(miou, 2)
            result[f'{horizon}s/IoU'] = round(occupied, 2)
            # The divisor ``nanmean`` just used.  P1-5 showed that this number can
            # differ between two runs of the *same* configuration (a class that is
            # NaN in one run and 0.00 in another), which moves the reported mIoU by
            # up to 0.75 without any model change.  Emitting it makes the artifact
            # visible at its source instead of having to be reconstructed from
            # per-class IoUs afterwards.
            result[f'{horizon}s/mIoU_finite_classes'] = int(
                np.count_nonzero(~np.isnan(semantic[:17])))
            semantic_mean.append(miou)
            occupied_mean.append(occupied)
            # Occupied precision / recall straight from the binary confusion
            # matrix that is already accumulated.  ``_hist(prediction, target)``
            # fills ``hist[target, prediction]``, so row 1 is the target-occupied
            # set (recall denominator) and column 1 the predicted-occupied set
            # (precision denominator).  The plan's P4.2 asks for five quality
            # diagnostics; IoU alone cannot distinguish "misses occupied space"
            # from "hallucinates occupied space", and the pair can -- and unlike
            # ECE it needs no probability tensor, so no model change is required.
            tp = float(self.binary[horizon][1, 1])
            predicted_occupied = float(self.binary[horizon][:, 1].sum())
            target_occupied = float(self.binary[horizon][1, :].sum())
            with np.errstate(divide='ignore', invalid='ignore'):
                precision = 100. * tp / predicted_occupied if predicted_occupied else np.nan
                recall = 100. * tp / target_occupied if target_occupied else np.nan
            result[f'{horizon}s/occupiedPrecision'] = round(float(precision), 2)
            result[f'{horizon}s/occupiedRecall'] = round(float(recall), 2)
            precision_mean.append(precision)
            recall_mean.append(recall)
            pooled_tp += tp
            pooled_predicted += predicted_occupied
            pooled_target += target_occupied
            for index, name in enumerate(self.class_names):
                result[f'{horizon}s/{name}'] = round(float(semantic[index] * 100.), 2)
        result['avg/mIoU'] = round(float(np.mean(semantic_mean)), 2)
        result['avg/IoU'] = round(float(np.mean(occupied_mean)), 2)
        # Emitted unconditionally: a missing key cannot be told apart from an
        # undefined ratio, whereas NaN can.
        result['avg/occupiedPrecision'] = round(
            100.0 * pooled_tp / pooled_predicted if pooled_predicted else float('nan'), 2)
        result['avg/occupiedRecall'] = round(
            100.0 * pooled_tp / pooled_target if pooled_target else float('nan'), 2)
        result['avg/occupiedPrecision_pooled'] = True
        # A single boolean so a reader (or a cross-run comparison script) can see
        # at a glance whether all horizons used the same mIoU divisor.  When it is
        # false, the per-horizon ``*_finite_classes`` values say where the
        # difference is -- and comparing this run's mIoU against another run's is
        # only valid on the intersection class set (P1-5).
        divisors = {result[f'{horizon}s/mIoU_finite_classes']
                    for horizon in self.horizons}
        result['avg/mIoU_divisor_uniform_across_horizons'] = len(divisors) == 1
        result['avg/mIoU_divisor_min'] = min(divisors)
        result['avg/mIoU_divisor_max'] = max(divisors)
        boundary_mean = []
        for horizon in self.horizons:
            scores = self.boundary[horizon]
            if not scores:
                continue
            per_horizon = float(np.mean(scores))
            result[f'{horizon}s/boundaryIoU'] = round(per_horizon, 2)
            boundary_mean.append(per_horizon)
        if boundary_mean:
            result['avg/boundaryIoU'] = round(float(np.mean(boundary_mean)), 2)
            result['boundary_samples_per_horizon'] = len(self.boundary[0])
        if self.per_sample_path is not None and self.per_sample:
            self.per_sample_path.parent.mkdir(parents=True, exist_ok=True)
            self.per_sample_path.write_text(json.dumps({
                'samples': self.count,
                'horizons': list(self.horizons),
                'class_names': self.class_names,
                'per_sample': self.per_sample,
                'spatial': self.spatial,
            }), encoding='utf-8')
        return result
