"""Reproduce the strict focal distance-weight gradient on synthetic CPU points.

This calls the existing loss unchanged. It establishes a local gradient path,
not its prevalence or net effect in the trained model or dataset.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from loss.sparseworld_strict_loss import SparseWorldStrictLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    objective = SparseWorldStrictLoss()
    gt = torch.zeros(1, 3, dtype=torch.float64)
    labels = torch.tensor([4])
    logits = torch.zeros(1, 17, dtype=torch.float64)
    rows = []
    for distance in (0.5, 1.5, 2.0, 3.0):
        points = torch.tensor([[distance, 0., 0.]], dtype=torch.float64,
                              requires_grad=True)
        cls, point = objective._set_loss(points, logits, gt, labels)
        cls_grad = torch.autograd.grad(cls, points, retain_graph=True)[0]
        point_grad = torch.autograd.grad(point, points)[0]
        # The GT is at the XY scene center: its scene-distance multiplier is 1.
        # This control retains exactly the same forward weighting and removes
        # only its position derivative. The actual source file is never edited.
        radius = torch.linalg.vector_norm(points - gt, dim=-1, keepdim=True)
        weight = radius.clamp_min(1e-12).reciprocal().clamp(.4, 1.)
        class_weight = logits.new_tensor(objective.class_weights)[None]
        detached_cls = objective._focal(logits, labels, class_weight * weight.detach())
        detached_cls = detached_cls + points.sum() * 0.
        detached_grad = torch.autograd.grad(detached_cls, points)[0]
        torch.testing.assert_close(detached_cls.detach(), cls.detach())
        torch.testing.assert_close(detached_grad, torch.zeros_like(points))
        epsilon = 1e-5
        plus = points.detach().clone()
        minus = points.detach().clone()
        plus[0, 0] += epsilon
        minus[0, 0] -= epsilon
        plus_cls, _ = objective._set_loss(plus, logits, gt, labels)
        minus_cls, _ = objective._set_loss(minus, logits, gt, labels)
        finite_difference = (plus_cls - minus_cls) / (2 * epsilon)
        torch.testing.assert_close(finite_difference, cls_grad[0, 0], atol=1e-8, rtol=1e-7)
        assert (cls_grad[0, 0] < 0) if 1. < distance < 2.5 else (cls_grad[0, 0] == 0)
        rows.append(dict(distance_m=distance, classification_loss=float(cls),
                         point_loss=float(point), classification_gradient_x=float(cls_grad[0, 0]),
                         point_gradient_x=float(point_grad[0, 0]),
                         summed_gradient_x=float((cls_grad + point_grad)[0, 0]),
                         detached_weight_classification_gradient_x=float(detached_grad[0, 0]),
                         finite_difference_classification_gradient_x=float(finite_difference)))
    source = Path(__file__).resolve().parents[2] / "loss/sparseworld_strict_loss.py"
    result = dict(status="passed", device="cpu", actual_loss_called=True,
                  source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  setup="One prediction on positive X; one GT at origin, class 4; fixed zero logits; default Strict set-loss weights.",
                  boundary="Synthetic local derivative only; no dataset prevalence, voxel gradient, neural parameter update or performance effect measured.",
                  rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
