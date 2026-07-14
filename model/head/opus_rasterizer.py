import torch


class OPUSRasterizer:
    """Vectorized sparse-point to semantic-voxel conversion used by eval and vis."""

    def __init__(self, pc_range, grid_size, grid_shape, empty_label=17):
        self.pc_range = pc_range
        self.grid_size = grid_size
        self.grid_shape = tuple(grid_shape)
        self.empty_label = empty_label

    def __call__(self, points, logits, score_threshold=0.0):
        batch_size = points.shape[0]
        total_voxels = int(torch.tensor(self.grid_shape).prod().item())
        result = torch.full((batch_size, total_voxels), self.empty_label,
                            device=points.device, dtype=torch.long)
        labels = logits.argmax(dim=-1)
        scores = logits.softmax(dim=-1).amax(dim=-1)
        lower = points.new_tensor(self.pc_range[:3])
        voxel = torch.floor((points - lower) / self.grid_size).long()
        bounds = voxel.new_tensor(self.grid_shape)
        valid = ((voxel >= 0) & (voxel < bounds)).all(dim=-1)
        valid = valid & torch.isfinite(points).all(dim=-1) & (scores >= score_threshold)
        for batch_idx in range(batch_size):
            keep = valid[batch_idx]
            if not keep.any():
                continue
            coords = voxel[batch_idx, keep]
            flat = ((coords[:, 0] * self.grid_shape[1] + coords[:, 1]) *
                    self.grid_shape[2] + coords[:, 2])
            # Unique perturbation makes ties deterministic without a Python voxel loop.
            priority = scores[batch_idx, keep] + torch.arange(
                flat.numel(), device=flat.device, dtype=scores.dtype) * 1e-8
            best = torch.full((total_voxels,), -torch.inf, device=points.device,
                              dtype=priority.dtype)
            best.scatter_reduce_(0, flat, priority, reduce='amax', include_self=True)
            winners = priority == best[flat]
            result[batch_idx].scatter_(0, flat[winners], labels[batch_idx, keep][winners])
        return result
