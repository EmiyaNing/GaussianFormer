"""Future occupancy, planning L2 and open-loop collision metrics for React World."""
import numpy as np
import torch
import torch.distributed as dist

from .sparseworld_strict_eval import SparseWorldStrictMetric


class ReactWorldMetric(SparseWorldStrictMetric):
    # Current frame key and the stacked per-future-step keys returned by
    # ``NuScenesReactWorldDataset`` for the same Occ3D grid as the labels.
    MASK_KEYS = {
        'camera': ('occ_cam_mask', 'future_occ_cam_masks'),
        'lidar': ('occ_lidar_mask', 'future_occ_lidar_masks'),
    }

    # ST-P3 / UniAD open-loop collision protocol.  Only vehicles and humans
    # count as obstacles: driveable surface, sidewalk, terrain, manmade,
    # vegetation and free are not collisions.  Barrier/traffic cone are static
    # objects and are intentionally absent, matching ST-P3's vehicle+human mask.
    AGENT_CLASSES = (2, 3, 4, 5, 6, 7, 9, 10)
    COLLISION_MODES = ('none', 'agent')
    EGO_WIDTH_M = 1.85
    EGO_LENGTH_M = 4.084
    EGO_X_OFFSET_M = 0.5
    BEV_ORIGIN = (-40.0, -40.0)
    BEV_RESOLUTION = 0.4

    def __init__(self, mask='none', collision='none'):
        super().__init__(boundary_stride=0)
        if mask not in ('none', 'camera', 'lidar'):
            raise ValueError(
                f'Unsupported React World evaluation mask {mask!r}; '
                "choose from 'none', 'camera' or 'lidar'")
        if collision not in self.COLLISION_MODES:
            raise ValueError(
                f'Unsupported React World collision mode {collision!r}; '
                f'choose from {self.COLLISION_MODES}')
        self.mask = mask
        self.collision_mode = collision
        self.plan_distance_sum = np.zeros(6, dtype=np.float64)
        self.plan_count = 0
        # Per future step (0.5s ... 3.0s): predicted hits, GT hits and the
        # number of samples whose GT did not collide (the valid denominator).
        self.collision_hits = np.zeros(6, dtype=np.int64)
        self.collision_gt_hits = np.zeros(6, dtype=np.int64)
        self.collision_valid = np.zeros(6, dtype=np.int64)

    def horizon_mask(self, metas, batch, index):
        """Occ3D observation mask of the current or one future frame.

        The future masks are loaded from each target frame's own ``labels.npz``
        (``future_occ_cam_masks``/``future_occ_lidar_masks``), so they live in
        the same native future-ego grid as the rendered prediction.  Masking
        therefore only removes unobserved voxels and never needs a coordinate
        transform.
        """
        if self.mask == 'none':
            return None
        current_key, future_key = self.MASK_KEYS[self.mask]
        key = current_key if index is None else future_key
        masks = metas.get(key)
        if masks is None:
            raise ValueError(
                f'Occ3D {self.mask}-mask evaluation requires {key!r}; add it '
                'to the dataset return_keys (world_return_keys).')
        mask = masks[batch] if index is None else masks[batch, index]
        return mask

    def _agent_bev(self, labels):
        """[H, W, D] semantic labels -> [H, W] BEV agent occupancy."""
        lookup = torch.zeros(18, dtype=torch.bool, device=labels.device)
        lookup[list(self.AGENT_CLASSES)] = True
        return lookup[labels].any(dim=-1)

    def _bev_box_hit(self, bev, x, y):
        """Whether the ST-P3 ego rectangle at (x, y) touches an occupied cell.

        ``bev`` is the agent occupancy in the future ego frame (x forward,
        y left).  Cells are included when their centre lies inside the
        rectangle; cells outside the grid are ignored, so a box that is fully
        outside the map cannot collide.
        """
        origin_x, origin_y = self.BEV_ORIGIN
        resolution = self.BEV_RESOLUTION
        x_min = x + self.EGO_X_OFFSET_M - self.EGO_LENGTH_M / 2.0
        x_max = x + self.EGO_X_OFFSET_M + self.EGO_LENGTH_M / 2.0
        y_min = y - self.EGO_WIDTH_M / 2.0
        y_max = y + self.EGO_WIDTH_M / 2.0
        i0 = max(int(np.floor((x_min - origin_x) / resolution)), 0)
        i1 = min(int(np.floor((x_max - origin_x) / resolution)), bev.shape[0] - 1)
        j0 = max(int(np.floor((y_min - origin_y) / resolution)), 0)
        j1 = min(int(np.floor((y_max - origin_y) / resolution)), bev.shape[1] - 1)
        if i1 < i0 or j1 < j0:
            return False
        index_i = np.arange(i0, i1 + 1)
        index_j = np.arange(j0, j1 + 1)
        centre_x = origin_x + (index_i + 0.5) * resolution
        centre_y = origin_y + (index_j + 0.5) * resolution
        inside_x = (centre_x >= x_min) & (centre_x <= x_max)
        inside_y = (centre_y >= y_min) & (centre_y <= y_max)
        if not inside_x.any() or not inside_y.any():
            return False
        occupied = bev[np.ix_(index_i[inside_x], index_j[inside_y])]
        return bool(occupied.any())

    def _update_collision(self, result, metas):
        """ST-P3 open-loop collision against the GT future agent occupancy."""
        prediction = result['pred_traj'].detach()
        target = metas.get('temporal_trajs')
        poses = metas.get('future_ego_to_current')
        labels = metas.get('future_occ_labels')
        if target is None or poses is None or labels is None:
            raise ValueError(
                'Collision evaluation requires temporal_trajs, '
                'future_ego_to_current and future_occ_labels in the batch')
        target = target.to(prediction)
        poses = poses.to(device=prediction.device, dtype=prediction.dtype)
        if labels.shape[1] != 6 or poses.shape[1] != 6:
            raise ValueError('Collision evaluation expects six future steps')
        for batch in range(prediction.shape[0]):
            for step in range(6):
                transform = poses[batch, step].detach().cpu().numpy()
                rotation = transform[:3, :3]
                translation = transform[:3, 3]
                # ``future_ego_to_current`` maps future -> current; the plan and
                # the GT waypoint are both in the current ego frame.
                predicted = prediction[batch, step].detach().cpu().numpy()
                truth = target[batch, step].detach().cpu().numpy()
                predicted = rotation.T @ (np.array(
                    [predicted[0], predicted[1], 0.0], dtype=np.float64) - translation)
                truth = rotation.T @ (np.array(
                    [truth[0], truth[1], 0.0], dtype=np.float64) - translation)
                bev = self._agent_bev(labels[batch, step]).detach().cpu().numpy()
                gt_hit = self._bev_box_hit(bev, truth[0], truth[1])
                predicted_hit = self._bev_box_hit(bev, predicted[0], predicted[1])
                self.collision_gt_hits[step] += int(gt_hit)
                self.collision_valid[step] += int(not gt_hit)
                self.collision_hits[step] += int(predicted_hit and not gt_hit)

    def add_batch(self, result, metas):
        if result['future_occ_frame'] != 'future_ego':
            raise ValueError('Future occupancy metrics require native future-frame readout')
        super().add_batch(result, metas)
        prediction = result['pred_traj'].detach()
        target = metas['temporal_trajs'].to(prediction)
        if prediction.shape != target.shape or prediction.shape[1:] != (6, 2):
            raise ValueError('Planning metrics require six aligned xy waypoints')
        distance = (prediction - target).norm(dim=-1)
        self.plan_distance_sum += distance.double().sum(0).cpu().numpy()
        self.plan_count += prediction.shape[0]
        if self.collision_mode != 'none':
            self._update_collision(result, metas)

    def synchronize(self, device):
        super().synchronize(device)
        if dist.is_available() and dist.is_initialized():
            values = np.concatenate([
                self.plan_distance_sum,
                np.asarray([self.plan_count], dtype=np.float64),
                self.collision_hits.astype(np.float64),
                self.collision_gt_hits.astype(np.float64),
                self.collision_valid.astype(np.float64),
            ])
            tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
            dist.all_reduce(tensor)
            values = tensor.cpu().numpy()
            self.plan_distance_sum = values[:6].copy()
            self.plan_count = int(values[6])
            self.collision_hits = values[7:13].astype(np.int64)
            self.collision_gt_hits = values[13:19].astype(np.int64)
            self.collision_valid = values[19:25].astype(np.int64)

    def results(self):
        # The inherited metric uses half-second indices in its legacy keys.
        # New React outputs name actual seconds, consistently with plan metrics.
        legacy = super().results()
        seconds = {'0s': '0s', '2s': '1s', '4s': '2s', '6s': '3s'}
        result = {}
        for key, value in legacy.items():
            prefix, separator, suffix = key.partition('/')
            result[seconds.get(prefix, prefix) + separator + suffix] = value
        distance = self.plan_distance_sum / max(self.plan_count, 1)
        for seconds, step in ((1, 2), (2, 4), (3, 6)):
            result[f'plan/{seconds}s_endpoint_L2_m'] = float(distance[step - 1])
            result[f'plan/{seconds}s_mean_L2_m'] = float(distance[:step].mean())
        if self.collision_mode != 'none':
            total = max(self.plan_count, 1)
            rates = []
            for seconds, step in ((1, 2), (2, 4), (3, 6)):
                index = step - 1
                rate = 100.0 * self.collision_hits[index] / total
                valid = int(self.collision_valid[index])
                result[f'collision/{seconds}s'] = round(float(rate), 3)
                result[f'collision_valid/{seconds}s'] = (
                    round(float(100.0 * self.collision_hits[index] / valid), 3)
                    if valid else float('nan'))
                result[f'collision_gt/{seconds}s'] = round(
                    float(100.0 * self.collision_gt_hits[index] / total), 3)
                result[f'collision_valid_count/{seconds}s'] = valid
                rates.append(rate)
            result['collision/avg'] = round(float(np.mean(rates)), 3)
        return result
