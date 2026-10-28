"""Extend the frozen diagnostic runner with score-rank-preserving semantic controls.

The 64-sample primary experiment concentrates semantic mass in one class. This
follow-up preserves each Gaussian's entire score multiset, including peak and
entropy, by swapping its largest score into the associated GT class.
"""

import torch

from . import diagnose, report
from .interventions import VARIANTS, make_variants


CONFIRMATION_VARIANTS = VARIANTS + ("D01_rank_swap", "D11_r08_rank_swap", "D11_r20_rank_swap")


def make_with_rank_controls(state, mapping, current_queries, seed):
    states, checks = make_variants(state, mapping, current_queries, seed)
    semantics = state.semantics.clone()
    flat = semantics.reshape(-1, 17)
    ids = torch.as_tensor(mapping["indices"], device=flat.device, dtype=torch.long)
    labels = torch.as_tensor(mapping["labels"], device=flat.device, dtype=torch.long)
    if ids.numel():
        selected = flat[ids].clone()
        largest, predicted = selected.max(-1)
        previous = selected.gather(1, labels[:, None]).squeeze(1)
        flat[ids, predicted] = previous
        flat[ids, labels] = largest
        assert torch.equal(flat[ids].sort(-1).values, selected.sort(-1).values)
    states["D01_rank_swap"] = state._replace(semantics=semantics)
    states["D11_r08_rank_swap"] = states["D10_r08"]._replace(semantics=semantics)
    states["D11_r20_rank_swap"] = states["D10_r20"]._replace(semantics=semantics)
    checks["semantic_score_multiset_preserved"] = True
    return states, checks


def main():
    # Reuse the identical runner and counter serializer; only register extra groups.
    diagnose.VARIANTS = CONFIRMATION_VARIANTS
    diagnose.make_variants = make_with_rank_controls
    report.VARIANTS = CONFIRMATION_VARIANTS
    diagnose.main()


if __name__ == "__main__":
    main()
