"""Paired state interventions and raw statistics for SparseWorld T02--T04."""
from collections import defaultdict

import numpy as np
import torch


def clone_snapshot(snapshot):
    """Clone tensor state while retaining immutable diagnostic metadata."""
    return {key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
            for key, value in snapshot.items()}


def query_selection(snapshot, subset, seed, block=(-20., 0., -10., 10.)):
    """Return a deterministic [B,Q] selection mask decided before evaluation."""
    centers = snapshot['positions_m'].mean(2)
    batch, queries = centers.shape[:2]
    if subset == 'all':
        return torch.ones(batch, queries, dtype=torch.bool, device=centers.device)
    if subset == 'random10':
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        count = max(1, int(round(queries * .1)))
        mask = torch.zeros(batch, queries, dtype=torch.bool)
        for batch_index in range(batch):
            mask[batch_index, torch.randperm(queries, generator=generator)[:count]] = True
        return mask.to(centers.device)
    if subset in ('block', 'block_shared'):
        xmin, xmax, ymin, ymax = block
        return ((centers[..., 0] >= xmin) & (centers[..., 0] < xmax) &
                (centers[..., 1] >= ymin) & (centers[..., 1] < ymax))
    raise ValueError(f'unknown query subset: {subset}')


def position_intervention(snapshot, sigma_m, subset, seed,
                          pc_range=(-40., -40., -1., 40., 40., 5.4),
                          block=(-20., 0., -10., 10.)):
    """Translate selected query point groups in XY without clipping to the ROI."""
    output = clone_snapshot(snapshot)
    selected = query_selection(snapshot, subset, seed, block)
    return position_intervention_mask(
        snapshot, sigma_m, selected, seed,
        shared=subset == 'block_shared', pc_range=pc_range,
        metadata=dict(subset=subset, block_m=list(block)))


def position_intervention_mask(snapshot, sigma_m, selected, seed,
                               shared=False,
                               pc_range=(-40., -40., -1., 40., 40., 5.4),
                               metadata=None):
    """Apply an XY query translation to an explicit deterministic mask."""
    output = clone_snapshot(snapshot)
    batch, queries = selected.shape
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    if shared:
        delta_xy = torch.randn(batch, 1, 2, generator=generator) * float(sigma_m)
        delta_xy = delta_xy.expand(batch, queries, 2).clone()
    else:
        delta_xy = torch.randn(batch, queries, 2, generator=generator) * float(sigma_m)
    delta_xy *= selected.cpu().unsqueeze(-1)
    delta_m = torch.zeros(batch, queries, 1, 3, dtype=torch.float32)
    delta_m[..., :2] = delta_xy.unsqueeze(2)
    delta_m = delta_m.to(device=snapshot['positions'].device,
                         dtype=snapshot['positions'].dtype)
    extent = delta_m.new_tensor(pc_range[3:]) - delta_m.new_tensor(pc_range[:3])
    output['positions'] = snapshot['positions'] + delta_m / extent
    output['positions_m'] = snapshot['positions_m'] + delta_m
    outside_before = ((snapshot['positions'] < 0.) | (snapshot['positions'] > 1.)).any(-1)
    outside_after = ((output['positions'] < 0.) | (output['positions'] > 1.)).any(-1)
    selected_delta = delta_m[..., 0, :2][selected]
    actual_rms = (selected_delta.square().mean().sqrt().item()
                  if selected_delta.numel() else 0.)
    record = dict(
        requested_sigma_m=float(sigma_m), actual_coordinate_rms_m=float(actual_rms),
        selected_queries=int(selected.sum().item()), total_queries=int(selected.numel()),
        out_of_bounds_before=float(outside_before.float().mean().item()),
        out_of_bounds_after=float(outside_after.float().mean().item()),
        seed=int(seed), shared_translation=bool(shared))
    if metadata:
        record.update(metadata)
    return output, record, delta_m


def feature_intervention(snapshot, scale, channel_std, subset, seed,
                         block=(-20., 0., -10., 10.)):
    """Add Gaussian feature noise scaled by a fixed per-channel std vector."""
    output = clone_snapshot(snapshot)
    selected = query_selection(snapshot, subset, seed, block)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    noise = torch.randn(snapshot['features'].shape, generator=generator)
    std = torch.as_tensor(channel_std, dtype=noise.dtype).flatten()
    if std.numel() != noise.shape[-1]:
        raise ValueError('channel_std must contain one value per feature channel')
    noise = noise * std * float(scale) * selected.cpu().unsqueeze(-1)
    output['features'] = snapshot['features'] + noise.to(snapshot['features'])
    return output, dict(
        feature_scale=float(scale), selected_queries=int(selected.sum().item()),
        total_queries=int(selected.numel()), seed=int(seed), subset=subset,
        feature_std_source='fixed_external_vector')


def confusion_matrix(prediction, label, num_classes=18, mask=None):
    prediction = np.asarray(prediction).reshape(-1)
    label = np.asarray(label).reshape(-1)
    valid = (label >= 0) & (label < num_classes)
    if mask is not None:
        valid &= np.asarray(mask).reshape(-1).astype(bool)
    index = num_classes * label[valid].astype(np.int64) + prediction[valid]
    return np.bincount(index, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_confusion(hist, free_label=17):
    hist = np.asarray(hist, dtype=np.float64)
    union = hist.sum(1) + hist.sum(0) - np.diag(hist)
    iou = np.divide(np.diag(hist), union, out=np.full_like(union, np.nan), where=union > 0)
    occupied_intersection = hist[:free_label, :free_label].sum()
    occupied_union = (hist[:free_label, :].sum() + hist[:, :free_label].sum() -
                      occupied_intersection)
    return dict(
        mIoU=float(np.nanmean(iou[:free_label]) * 100.),
        OccIoU=float(occupied_intersection / max(occupied_union, 1.) * 100.),
        per_class_IoU=(iou * 100.).tolist())


class InterventionAccumulator:
    """Accumulate paired matrices before calculating any IoU summaries."""
    def __init__(self, num_classes=18):
        self.num_classes = num_classes
        self.hist = defaultdict(lambda: np.zeros((num_classes, num_classes), dtype=np.int64))
        self.geometry_sum = defaultdict(float)
        self.geometry_count = defaultdict(int)
        self.sample_tokens = defaultdict(list)

    def add(self, key, prediction, label, mask, points, clean_points, sample_token):
        self.hist[key] += confusion_matrix(prediction, label, self.num_classes, mask)
        difference = (points - clean_points).norm(dim=-1)
        self.geometry_sum[key] += float(difference.sum().item())
        self.geometry_count[key] += int(difference.numel())
        self.sample_tokens[key].append(str(sample_token))

    def results(self):
        output = {}
        for key in sorted(self.hist, key=str):
            name = '|'.join(map(str, key)) if isinstance(key, tuple) else str(key)
            output[name] = dict(
                **metrics_from_confusion(self.hist[key]),
                mean_paired_point_error_m=(self.geometry_sum[key] /
                                           max(self.geometry_count[key], 1)),
                confusion_matrix=self.hist[key].tolist(),
                sample_tokens=self.sample_tokens[key])
        return output


def occupied_voxel_centers(label, pc_range=(-40., -40., -1., 40., 40., 5.4),
                           grid_size=.4, free_label=17, mask=None, max_points=8192):
    """Convert a native Occ3D label volume to deterministic occupied centers."""
    valid = label != free_label
    if mask is not None:
        valid &= mask.bool()
    indices = valid.nonzero(as_tuple=False)
    if indices.shape[0] > max_points:
        stride = int(np.ceil(indices.shape[0] / max_points))
        indices = indices[::stride]
    lower = label.new_tensor(pc_range[:3], dtype=torch.float32)
    return indices.to(torch.float32) * float(grid_size) + lower + float(grid_size) / 2.


def bidirectional_set_distance(points, targets, chunk=2048):
    """Symmetric nearest-neighbour distance with bounded temporary memory."""
    if points.numel() == 0 or targets.numel() == 0:
        return points.new_tensor(float('nan'))
    forward = []
    for start in range(0, points.shape[0], chunk):
        forward.append(torch.cdist(points[start:start + chunk], targets).amin(1))
    backward = []
    for start in range(0, targets.shape[0], chunk):
        backward.append(torch.cdist(targets[start:start + chunk], points).amin(1))
    return .5 * (torch.cat(forward).mean() + torch.cat(backward).mean())


def privileged_query_translation(points, targets, num_refines=48, steps=20,
                                 learning_rate=.05, max_translation_m=.8,
                                 chunk=2048):
    """Fit capped query translations using clean-state fixed NN assignments."""
    grouped = points.detach().view(-1, num_refines, 3)
    # Establish matching once at the uncorrected state. This avoids changing
    # the supervision target while optimizing and keeps memory independent of
    # the requested number of optimization steps.
    matched = []
    with torch.no_grad():
        flattened = grouped.flatten(0, 1)
        for start in range(0, flattened.shape[0], chunk):
            distance = torch.cdist(flattened[start:start + chunk], targets)
            matched.append(targets[distance.argmin(1)])
        matched = torch.cat(matched).view_as(grouped)
    raw = torch.zeros(grouped.shape[0], 2, device=points.device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=float(learning_rate))
    losses = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        raw_norm_sq = raw.square().sum(dim=-1, keepdim=True)
        delta_xy = (raw / torch.sqrt(1. + raw_norm_sq) *
                    float(max_translation_m))
        delta = torch.cat([delta_xy, torch.zeros_like(delta_xy[:, :1])], -1)
        moved = (grouped + delta[:, None]).flatten(0, 1)
        # Point offsets and clean-state correspondence remain fixed. The
        # untouched symmetric set distance is separately reported afterward.
        loss = torch.nn.functional.smooth_l1_loss(
            moved, matched.flatten(0, 1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    raw = raw.detach()
    raw_norm_sq = raw.square().sum(dim=-1, keepdim=True)
    delta_xy = (raw / torch.sqrt(1. + raw_norm_sq) *
                float(max_translation_m))
    delta = torch.cat([delta_xy, torch.zeros_like(delta_xy[:, :1])], -1)
    return delta, losses
