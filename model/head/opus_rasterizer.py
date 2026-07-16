import torch
import torch.nn.functional as F


class OPUSRasterizer:
    """OPUS-V1 point-set discretisation used for evaluation and visualisation."""

    def __init__(self, pc_range, grid_size, grid_shape, empty_label=17,
                 score_threshold=0.5, center_distance_threshold=3.0,
                 padding=True):
        self.pc_range = pc_range
        self.grid_size = grid_size
        self.grid_shape = tuple(grid_shape)
        self.empty_label = empty_label
        self.score_threshold = score_threshold
        self.center_distance_threshold = center_distance_threshold
        self.padding = padding

    def _closing(self, scores, original):
        # Same max-pool dilation/erosion sequence as the official V1 head.
        dilated = F.max_pool3d(scores, 3, stride=1, padding=1)
        closed = -F.max_pool3d(-dilated, 3, stride=1, padding=1)
        return torch.where(original.expand_as(closed), scores, closed)

    def __call__(self, points, logits, score_threshold=None, point_valid_mask=None,
                 group_size=None, center_distance_threshold=None, padding=None):
        score_threshold = self.score_threshold if score_threshold is None else score_threshold
        center_distance_threshold = (self.center_distance_threshold if center_distance_threshold is None
                                     else center_distance_threshold)
        padding = self.padding if padding is None else padding
        batch_size, num_points = points.shape[:2]
        total_voxels = int(torch.tensor(self.grid_shape).prod().item())
        result = torch.full((batch_size, total_voxels), self.empty_label,
                            device=points.device, dtype=torch.long)
        scores = logits.sigmoid()
        valid = torch.isfinite(points).all(dim=-1) & (scores > score_threshold).any(dim=-1)
        lower = points.new_tensor(self.pc_range[:3])
        voxel = torch.floor((points - lower) / self.grid_size).long()
        bounds = voxel.new_tensor(self.grid_shape)
        valid &= ((voxel >= 0) & (voxel < bounds)).all(dim=-1)
        if point_valid_mask is not None:
            valid &= point_valid_mask.bool()
        if group_size is not None:
            if num_points % group_size:
                raise ValueError('group_size must divide the number of OPUS points')
            grouped = points.view(batch_size, -1, group_size, 3)
            distance = torch.linalg.vector_norm(grouped - grouped.mean(dim=2, keepdim=True), dim=-1)
            valid &= distance.flatten(1) < center_distance_threshold

        for batch_index in range(batch_size):
            keep = valid[batch_index]
            if not keep.any():
                continue
            coords = voxel[batch_index, keep]
            flat = ((coords[:, 0] * self.grid_shape[1] + coords[:, 1]) *
                    self.grid_shape[2] + coords[:, 2])
            point_scores = scores[batch_index, keep]
            # Voxelization aggregates all points in a voxel, rather than
            # selecting a single softmax winner.
            score_sum = point_scores.new_zeros((total_voxels, point_scores.shape[-1]))
            score_sum.index_add_(0, flat, point_scores)
            count = point_scores.new_zeros(total_voxels)
            count.index_add_(0, flat, torch.ones_like(flat, dtype=point_scores.dtype))
            occupied = count > 0
            dense_scores = score_sum / count.clamp_min(1).unsqueeze(-1)
            if padding:
                score_grid = dense_scores.view(*self.grid_shape, -1).permute(3, 0, 1, 2).unsqueeze(0)
                original = occupied.view(1, 1, *self.grid_shape)
                dense_scores = self._closing(score_grid, original).squeeze(0).permute(1, 2, 3, 0).reshape(
                    total_voxels, -1)
            semantic = dense_scores.argmax(dim=-1)
            occupied = (dense_scores > score_threshold).any(dim=-1)
            result[batch_index, occupied] = semantic[occupied]
        return result
