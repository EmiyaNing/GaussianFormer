"""A fixed-budget sigmoid relaxation with the threshold's implicit gradient."""
import torch


class _BudgetSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, budget, temperature):
        if scores.ndim != 2 or not 0 < budget < scores.shape[-1]:
            raise ValueError('Expected [batch, candidates] and 0 < budget < candidates')
        if temperature <= 0 or not torch.is_floating_point(scores):
            raise ValueError('A positive temperature and floating scores are required')
        valid = torch.isfinite(scores)
        if torch.isnan(scores).any() or torch.isposinf(scores).any() or (valid.sum(-1) <= budget).any():
            raise ValueError('Each sample needs more finite candidate scores than its budget')
        lo = scores.masked_fill(~valid, torch.inf).amin(-1, keepdim=True) - 30 * temperature
        hi = scores.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True) + 30 * temperature
        # The root is monotone. Keep the solve off the graph and supply the
        # full derivative of the constraint below, including cross-candidate terms.
        for _ in range(40):
            mid = (lo + hi) * .5
            mass = torch.sigmoid((scores - mid) / temperature).sum(-1, keepdim=True)
            lo = torch.where(mass > budget, mid, lo)
            hi = torch.where(mass > budget, hi, mid)
        p = torch.sigmoid((scores - (lo + hi) * .5) / temperature)
        ctx.save_for_backward(p)
        ctx.temperature = temperature
        return p

    @staticmethod
    def backward(ctx, grad_output):
        (p,) = ctx.saved_tensors
        sensitivity = p * (1 - p) / ctx.temperature
        denominator = sensitivity.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
        common = (grad_output * sensitivity).sum(-1, keepdim=True) / denominator
        return sensitivity * (grad_output - common), None, None


def budget_sigmoid(scores, budget, temperature):
    """Return soft inclusion weights whose per-sample sum is ``budget``."""
    return _BudgetSigmoid.apply(scores, int(budget), float(temperature))


def gather_candidates(value, indices):
    """Gather grouped [B,Q,K,D] attributes into [B,1,N,D]."""
    flat = value.flatten(1, 2)
    return flat.gather(1, indices[..., None].expand(-1, -1, flat.shape[-1])).unsqueeze(1)


def gather_gaussian(gaussian, indices):
    """Preserve optional unset GaussianPrediction fields when selecting GS."""
    return type(gaussian)(*(None if v is None else gather_candidates(v, indices)
                            for v in gaussian))
