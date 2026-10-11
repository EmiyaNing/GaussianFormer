"""Diagnostic metrics the P1 plan asks for but the repository never had.

Plan reference: ``plans/sparse_render_plans/README.md`` section 5.1.  These are
deliberately decoupled from any model: each takes plain tensors so it can be
validated against hand-computed values and reused by P1-P4.

**ECE** (expected calibration error) uses equal-width confidence bins over the
occupied/empty decision.

**Boundary IoU** follows the boundary-band formulation of Cheng et al.,
"Boundary IoU: Improving Object-Centric Image Segmentation Evaluation"
(CVPR 2021): build a band around the mask boundary by dilating and eroding, then
restrict the ordinary IoU to voxels inside either band.  The band is taken on
*both* sides of the boundary, i.e. ``dilate(M) \\ erode(M)``.  An outer-only
band (``dilate(M) \\ M``) is degenerate: for identical masks it scores NaN,
because the intersection term is empty by construction.

**RayIoU is deliberately NOT implemented here.**  It must follow the exact
Occ3D/S2GO ray-casting protocol, and no reference implementation exists in this
repository to validate against.  A plausible-looking but unverified RayIoU
would be worse than none, because it would silently enter comparison tables.
It is needed for the P3 comparison against S2GO's reported numbers, where the
protocol version can be pinned against a published result.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def expected_calibration_error(probabilities, correct, num_bins=15):
    """ECE for a binary decision.

    Args:
        probabilities: ``[...]`` predicted probability of the positive class.
        correct: ``[...]`` bool/0-1 tensor, 1 where the prediction is right.
        num_bins: number of equal-width confidence bins.

    Returns:
        ``(ece, bin_counts)`` where ``ece`` is a Python float.
    """
    probabilities = probabilities.reshape(-1).float()
    correct = correct.reshape(-1).float()
    if probabilities.numel() == 0:
        return 0.0, torch.zeros(num_bins, dtype=torch.long)
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=probabilities.device)
    # Interior edges only, so p == 0 and p == 1 both land in a valid bin.
    index = torch.bucketize(probabilities, edges[1:-1], right=False)
    ece = torch.zeros((), device=probabilities.device)
    counts = torch.zeros(num_bins, dtype=torch.long, device=probabilities.device)
    total = probabilities.numel()
    for bin_index in range(num_bins):
        mask = index == bin_index
        count = int(mask.sum().item())
        counts[bin_index] = count
        if count == 0:
            continue
        confidence = probabilities[mask].mean()
        accuracy = correct[mask].mean()
        ece = ece + (count / total) * (accuracy - confidence).abs()
    return float(ece.item()), counts


def _dilate(mask, radius):
    """Binary 3D dilation by a ``(2r+1)^3`` box."""
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool3d(mask.float(), kernel, stride=1, padding=radius) > 0


def _erode(mask, radius):
    """Binary 3D erosion, defined as the complement of a dilated complement."""
    if radius <= 0:
        return mask
    return ~_dilate(~mask, radius)


def boundary_band(mask, radius=1):
    """Band of voxels within ``radius`` of the mask boundary, both sides.

    Args:
        mask: bool tensor with leading singleton dims, e.g. ``[1, 1, D, H, W]``.
        radius: band half-width in voxels.

    Returns:
        Bool tensor of the same shape.
    """
    return _dilate(mask, radius) & ~_erode(mask, radius)


def boundary_iou(prediction, target, num_classes, radius=1,
                 ignore_label=None, per_class=False):
    """Mean boundary IoU over classes present in ``target``, in percent.

    Args:
        prediction: ``[D, H, W]`` or ``[1, D, H, W]`` integer labels.
        target: same shape, integer labels.
        num_classes: number of semantic classes to score.
        radius: boundary band half-width in voxels.
        ignore_label: label to exclude from the class loop (e.g. the empty
            class).
        per_class: also return the per-class list.
    """
    prediction = prediction.reshape(-1, *prediction.shape[-3:]).long()
    target = target.reshape(-1, *target.shape[-3:]).long()
    if prediction.shape != target.shape:
        raise ValueError('prediction and target must share D/H/W')
    if prediction.shape[0] != 1:
        raise ValueError('evaluate one scene at a time (leading dim must be 1)')
    prediction = prediction[0]
    target = target[0]

    scores = []
    for label in range(num_classes):
        if ignore_label is not None and label == ignore_label:
            continue
        target_mask = (target == label).unsqueeze(0).unsqueeze(0)
        if not bool(target_mask.any()):
            continue
        prediction_mask = (prediction == label).unsqueeze(0).unsqueeze(0)
        band = boundary_band(target_mask, radius) | boundary_band(prediction_mask, radius)
        intersection = (prediction_mask & target_mask & band).sum().item()
        union = ((prediction_mask | target_mask) & band).sum().item()
        scores.append(100.0 * intersection / union if union > 0 else float('nan'))

    if not scores:
        return float('nan'), ([] if per_class else [])
    mean = sum(scores) / len(scores)
    return mean, (scores if per_class else [])


def occupancy_precision_recall(prediction, target, ignore_label,
                               eval_mask=None):
    """Voxel-wise occupied/empty precision and recall, in percent.

    A voxel is "occupied" when its label differs from ``ignore_label`` (the
    empty class).  Unlike an earlier version of this function, the prediction
    is *not* intersected with the target's occupied set -- doing so makes false
    positives impossible by construction and forces precision to 100%.

    Args:
        prediction: integer label tensor.
        target: integer label tensor.
        ignore_label: the empty-class label.
        eval_mask: optional bool tensor restricting the evaluation region.
    """
    if ignore_label is None:
        raise ValueError('ignore_label is required to define "occupied"')
    pred_occupied = prediction != ignore_label
    target_occupied = target != ignore_label
    if eval_mask is not None:
        region = eval_mask.bool()
        pred_occupied = pred_occupied & region
        target_occupied = target_occupied & region
    true_positive = int((pred_occupied & target_occupied).sum().item())
    false_positive = int((pred_occupied & ~target_occupied).sum().item())
    false_negative = int((~pred_occupied & target_occupied).sum().item())
    precision = 100.0 * true_positive / max(true_positive + false_positive, 1)
    recall = 100.0 * true_positive / max(true_positive + false_negative, 1)
    return precision, recall
