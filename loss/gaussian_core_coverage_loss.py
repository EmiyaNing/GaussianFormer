"""Foreground core-coverage deficits supervise Gaussian centers only."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices


@torch.no_grad()
def clipped_core_distance(means, scales, rotations, xyz, foreground, grid_shape,
                          pc_min, grid_size=.4, max_scale=.4, chunk_size=2048):
    """Exact min(Mahalanobis distance, 2) on a regular occupancy grid.

    A distance below 2 requires Euclidean distance <= 2 * largest scale.
    Enumerate that bounded voxel neighborhood for every valid Gaussian and
    scatter the minimum distance. No opacity/semantic values, KNN truncation,
    SDF, or additional compiled operators enter the coverage definition.
    """
    if not (scales > 0).all() or not torch.isfinite(scales).all():
        raise ValueError('Coverage requires finite positive scales')
    if scales.numel() and float(scales.max()) > max_scale + 1e-6:
        raise ValueError('max_scale must bound every physical Gaussian scale')
    if not torch.isfinite(means).all() or not torch.isfinite(rotations).all():
        raise ValueError('Nonfinite Gaussian geometry')
    means, scales = means.float(), scales.float()
    q = F.normalize(rotations.float(), dim=-1)
    lower = means.new_tensor(pc_min)
    shape = torch.tensor(grid_shape, device=means.device)
    nvox = math.prod(grid_shape)
    if xyz.shape != (nvox, 3) or foreground.numel() != nvox:
        raise ValueError('Coverage grid/GT shape mismatch')
    radius = math.ceil(2 * max_scale / grid_size)
    axis = torch.arange(-radius, radius+1, device=means.device)
    offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    minimum2 = means.new_full((nvox,), 4.)
    for start in range(0, len(means), chunk_size):
        m, s, rotation = means[start:start+chunk_size], scales[start:start+chunk_size], q[start:start+chunk_size]
        cells = torch.floor((m-lower)/grid_size).long()[:, None] + offsets[None]
        inside = ((cells >= 0) & (cells < shape)).all(-1)
        ids = (cells[..., 0]*grid_shape[1]+cells[..., 1])*grid_shape[2]+cells[..., 2]
        selected = inside & foreground[ids.clamp(0, nvox-1)]
        gaussian_id, offset_id = selected.nonzero(as_tuple=True)
        ids = ids[gaussian_id, offset_id]
        delta = xyz[ids].float()-m[gaussian_id]
        qq = rotation[gaussian_id]
        uv = 2*torch.cross(qq[:, 1:], delta, dim=-1)
        # Same convention as renderer: Sigma=R.T @ diag(scale**2) @ R.
        local = delta + qq[:, :1]*uv + torch.cross(qq[:, 1:], uv, dim=-1)
        distance2 = (local/s[gaussian_id]).square().sum(-1)
        minimum2.scatter_reduce_(0, ids, distance2, reduce='amin', include_self=True)
    return minimum2.clamp_min(0).sqrt()


@OPENOCC_LOSS.register_module()
class GaussianCoreCoverageLoss(nn.Module):
    def __init__(self, weight=.1, empty_label=17, grid_shape=(200, 200, 16),
                 pc_range=(-40., -40., -1., 40., 40., 5.4), grid_size=.4,
                 max_scale=.4, chunk_size=2048, knn_chunk_size=1024):
        super().__init__()
        self.weight = weight
        self.empty_label = empty_label
        self.grid_shape = tuple(grid_shape)
        self.pc_range = tuple(pc_range)
        self.grid_size = grid_size
        self.max_scale = max_scale
        self.chunk_size = chunk_size
        self.knn_chunk_size = knn_chunk_size
        expected = [(pc_range[i+3]-pc_range[i])/grid_size for i in range(3)]
        if any(abs(x-n)>1e-5 for x,n in zip(expected, grid_shape)):
            raise ValueError('pc_range, grid_shape and grid_size must agree')

    def forward(self, inputs):
        g, metas = inputs['gaussian'], inputs['metas']
        labels = metas['occ_label'].reshape(g.means.shape[0], -1).long()
        camera = metas['occ_cam_mask'].reshape_as(labels).bool()
        xyz = metas['occ_xyz'].reshape(g.means.shape[0], -1, 3)
        masks = inputs.get('opus_point_valid_masks')
        valid = (torch.ones(g.means.shape[:2], dtype=torch.bool, device=g.means.device)
                 if masks is None else masks[-1].bool())
        zero = g.means.sum()*0
        numerator = zero
        count = 0
        active_count = zero.detach()
        weight_sum = zero.detach()
        for b in range(g.means.shape[0]):
            fg = camera[b] & (labels[b]>=0) & (labels[b]<self.empty_label)
            n = int(fg.sum())
            count += n
            if not n:
                continue
            if not valid[b].any():
                raise ValueError('Foreground exists but no valid predicted Gaussians')
            means = g.means[b, valid[b]]
            radius = clipped_core_distance(means.detach(), g.scales[b, valid[b]].detach(),
                g.rotations[b, valid[b]].detach(), xyz[b], fg, self.grid_shape,
                self.pc_range[:3], self.grid_size, self.max_scale, self.chunk_size)
            weights = (radius-1).clamp(0, 1).detach()
            active = fg & (weights>0)
            active_count = active_count + active.sum().detach()
            weight_sum = weight_sum + weights[fg].sum()
            if active.any():
                target = xyz[b, active].detach()
                nearest = _nearest_indices(target, means.detach(), self.knn_chunk_size)
                distance = (means[nearest]-target).norm(dim=-1)/self.grid_size
                penalty = F.smooth_l1_loss(distance, torch.zeros_like(distance), reduction='none', beta=1.)
                numerator = numerator + (weights[active]*penalty).sum()
        value = numerator/max(count, 1)
        return self.weight*value, dict(raw_loss=value.detach(), gt_foreground=zero.detach()+count,
            active_gt=active_count, no_core_fraction=active_count/max(count, 1),
            mean_gap_weight=weight_sum/max(count, 1))
