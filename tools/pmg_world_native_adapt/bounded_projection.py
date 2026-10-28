"""Project one jointly trained parameter vector onto its initial L2 ball."""

import math

import torch


class JointDisplacementBudget:
    """A cumulative parameter constraint, independent of gradient clipping.

    All parameters share one projection factor. Optimizer moments are never
    read or modified. Origins must be captured before loading a resumed state.
    """

    def __init__(self, named_parameters, radius):
        self.parameters = dict(named_parameters)
        self.radius = float(radius)
        if not self.parameters or not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("A positive finite radius and nonempty parameters are required")
        self.origins = {name: value.detach().clone()
                        for name, value in self.parameters.items()}

    @torch.no_grad()
    def norm(self):
        squared = [((value - self.origins[name]).double().square().sum())
                   for name, value in self.parameters.items()]
        return float(torch.stack(squared).sum().sqrt())

    @torch.no_grad()
    def project(self):
        before = self.norm()
        if not math.isfinite(before):
            raise FloatingPointError("Nonfinite parameter displacement")
        factor = min(1.0, self.radius / before) if before else 1.0
        if factor < 1.0:
            for name, value in self.parameters.items():
                origin = self.origins[name]
                value.copy_(origin + factor * (value - origin))
        after = self.norm()
        correction = 1.0
        # FP32 parameter storage can round just outside the mathematical ball.
        # One common inward correction handles this without module-wise scales.
        if after > self.radius:
            epsilon = max(torch.finfo(p.dtype).eps for p in self.parameters.values())
            correction = self.radius / after * (1.0 - 8.0 * epsilon)
            for name, value in self.parameters.items():
                origin = self.origins[name]
                value.copy_(origin + correction * (value - origin))
            after = self.norm()
        if after > self.radius + 1e-6:
            raise AssertionError("Projection exceeded the declared displacement budget")
        return dict(pre_delta_l2=before, post_delta_l2=after,
                    projection_scale=factor, roundoff_correction_scale=correction,
                    projected=bool(factor < 1.0))


def projection_self_check():
    """Check known geometry, a strict interior point, and optimizer isolation."""
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    budget = JointDisplacementBudget([("a", first), ("b", second)], 5.0)
    optimizer = torch.optim.AdamW([first, second], lr=0.01)
    (first.square().sum() + second.square().sum()).backward()
    optimizer.step()
    moments = {id(p): {k: v.clone() for k, v in optimizer.state[p].items()}
               for p in (first, second)}
    with torch.no_grad():
        first.copy_(budget.origins["a"] + first.new_tensor([6., 8.]))
        second.copy_(budget.origins["b"])
    record = budget.project()
    torch.testing.assert_close(first, first.new_tensor([4., 2.]))
    torch.testing.assert_close(second, second.new_tensor([3.]))
    assert record["pre_delta_l2"] == 10. and record["post_delta_l2"] == 5.
    assert record["projection_scale"] == .5
    for p in (first, second):
        assert all(torch.equal(v, moments[id(p)][k]) for k, v in optimizer.state[p].items())
    with torch.no_grad():
        first.copy_(budget.origins["a"] + first.new_tensor([.3, .4]))
    before = first.detach().clone()
    assert not budget.project()["projected"] and torch.equal(first, before)
    with torch.no_grad():
        first.copy_(budget.origins["a"])
    assert budget.project()["post_delta_l2"] == 0.
    return dict(status="passed", known_projection=True, interior_unchanged=True,
                zero_displacement=True, optimizer_moments_unchanged=True)
