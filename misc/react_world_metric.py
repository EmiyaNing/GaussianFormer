"""Future occupancy and current-ego-frame planning metrics for React World."""
import numpy as np
import torch
import torch.distributed as dist

from .sparseworld_strict_eval import SparseWorldStrictMetric


class ReactWorldMetric(SparseWorldStrictMetric):
    def __init__(self):
        super().__init__(boundary_stride=0)
        self.plan_distance_sum = np.zeros(6, dtype=np.float64)
        self.plan_count = 0

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

    def synchronize(self, device):
        super().synchronize(device)
        if dist.is_available() and dist.is_initialized():
            values = torch.as_tensor(
                np.append(self.plan_distance_sum, self.plan_count),
                dtype=torch.float64, device=device)
            dist.all_reduce(values)
            values = values.cpu().numpy()
            self.plan_distance_sum = values[:6]
            self.plan_count = int(values[6])

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
        return result
