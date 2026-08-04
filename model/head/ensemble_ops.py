"""
Multi-Stage Gaussian Ensemble Operations.

This module provides functions to render occupancy from every decoder stage's
gaussian predictions and fuse them via average weighting.  It reuses the
existing GaussianHead (or GaussianHeadSemantic) public interfaces —
prepare_gaussian_args() and aggregator — so no existing code needs to change.

Sub-task 1:  render_all_stage_occupancies  — per-stage occupancy rendering
Sub-task 2:  fuse_stage_occupancies       — average fusion across stages
"""

from typing import List, Tuple, Union
import torch

from model.encoder.gaussian_encoder.utils import GaussianPrediction


# ---------------------------------------------------------------------------
#  Sub-task 2 helpers:  normalise aggregator output to a uniform shape
# ---------------------------------------------------------------------------

def _normalize_aggregator_output(
    result: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    head,
) -> torch.Tensor:
    """Convert an aggregator call result into a uniform ``(B, C, N)`` logit tensor.

    Different aggregator backends return different formats:

    * **Standard** (``local_aggregate`` / ``local_aggregate_react``):
      single ``Tensor (N, C)`` — **no batch dimension** (the aggregator
      internally squeezes B=1).

    * **Probabilistic** (``local_aggregate_prob*``):
      ``(sem_logits, bin_logits, density)`` where
        ``sem_logits``  shape ``(N, C+1)`` — semantic class logits **including**
                        the empty class,
        ``bin_logits``  shape ``(N,)``     — per-voxel geometry/occupancy logit.

      When ``head.combine_geosem`` is ``True``, the two are multiplicatively
      combined; otherwise only ``sem_logits`` is used.

    All outputs are normalised to ``(1, C, N)`` (batch-first, class-second)
    for consistent downstream fusion.

    Args:
        result:  Raw return value of ``head.aggregator(...)``.
        head:    A ``GaussianHead`` / ``GaussianHeadSemantic`` instance whose
                 attributes (``use_localaggprob``, ``combine_geosem``) determine
                 the interpretation.

    Returns:
        occupancy:  ``Tensor (1, C, N)`` — occupancy logits ready for fusion.
    """
    # ----- standard / react aggregator: single tensor (N, C) -----
    if isinstance(result, torch.Tensor):
        # result: (N, C)  — no batch dim, no class-first ordering
        occupancy = result  # (N, C)

    # ----- probabilistic aggregator: 3-tuple (N, C+1), (N,), (N,) -----
    elif isinstance(result, (tuple, list)) and len(result) == 3:
        sem_logits, bin_logits, _density = result
        # sem_logits: (N, C+1)   — includes empty-class channel
        # bin_logits: (N,)       — occupancy probability (sigmoid logit)

        combine_geosem = getattr(head, 'combine_geosem', False)

        if combine_geosem:
            # Multiplicative combination: sem * σ(bin) + (1-σ(bin)) for empty
            sem = sem_logits[:, :-1]                  # (N, C)
            geo_weight = bin_logits.unsqueeze(1)       # (N, 1)
            weighted_sem = sem * geo_weight             # (N, C)
            geo_empty = 1.0 - geo_weight                # (N, 1)
            occupancy = torch.cat([weighted_sem, geo_empty], dim=1)  # (N, C+1)
        else:
            # Use raw semantic logits directly
            occupancy = sem_logits  # (N, C+1)
    else:
        raise TypeError(
            f"Unexpected aggregator return type: {type(result)}. "
            f"Expected Tensor or 3-tuple."
        )

    # ----- normalise to (B, C, N) = (1, C, N) -----
    if occupancy.dim() == 2:
        # (N, C) → (1, C, N)
        occupancy = occupancy.transpose(0, 1).unsqueeze(0)  # (1, C, N)
    elif occupancy.dim() == 3:
        # Already 3-D; assume (B, C, N) — verify C is on dim 1
        pass
    else:
        raise ValueError(f"Unexpected occupancy dims: {occupancy.shape}")

    return occupancy


# ---------------------------------------------------------------------------
#  Sub-task 1:  render occupancy for every decoder stage
# ---------------------------------------------------------------------------

def render_all_stage_occupancies(
    head,
    gaussians: List[GaussianPrediction],
    sampled_xyz: torch.Tensor,
) -> List[torch.Tensor]:
    """Render occupancy logits from **every** decoder stage's gaussian.

    Unlike the standard ``GaussianHead.forward()`` that only renders the last
    stage during evaluation, this function iterates over **all** stages and
    calls ``head.aggregator()`` for each one independently.

    Args:
        head:         A ``GaussianHead`` or ``GaussianHeadSemantic`` instance.
                      Must expose ``prepare_gaussian_args()`` and ``aggregator``.
        gaussians:    List of ``GaussianPrediction``, one per decoder stage.
                      Length = ``num_stages``.  Each element holds the means,
                      scales, rotations, opacities and semantics for that stage.
        sampled_xyz:  ``Tensor (B, N, 3)`` — voxel-centre coordinates (the
                      query points passed to the aggregator).

    Returns:
        all_occs:  ``List[Tensor (B, C, N)]`` — occupancy logits for every
                   stage (same length as *gaussians*).  Stages whose gaussian
                   contains zero primitives are skipped.

    Example::

        result_dict = model(imgs=input_imgs, metas=data)
        all_occs = render_all_stage_occupancies(
            head=raw_model.head,
            gaussians=result_dict['gaussians'],
            sampled_xyz=result_dict['sampled_xyz'],
        )
        fused_logits, hard_labels = fuse_stage_occupancies(all_occs)
    """
    all_occs: List[torch.Tensor] = []

    for stage_idx, gaussian in enumerate(gaussians):
        # --- Skip stages that produce no gaussians (e.g. before first refine) ---
        if gaussian.means.shape[1] == 0:
            continue

        # --- Step 1: prepare gaussian rendering arguments ---
        # This handles with_empty / use_localaggprob logic internally.
        means, origi_opa, opacities, scales, CovInv = \
            head.prepare_gaussian_args(gaussian)
        # means:     (B, G', 3)       G' = G + (1 if with_empty else 0)
        # origi_opa: (B, G')
        # opacities: (B, G', C')      C' = num_classes
        # scales:    (B, G', 3)
        # CovInv:    (B, G', 3, 3)

        bs, g = means.shape[:2]

        # --- Step 2: CUDA rasterisation via the head's aggregator ---
        result = head.aggregator(
            sampled_xyz.clone().float(),   # (B, N, 3)
            means,                          # (B, G', 3)
            origi_opa.reshape(bs, g),       # (B, G')
            opacities,                      # (B, G', C')
            scales,                         # (B, G', 3)
            CovInv,                         # (B, G', 3, 3)
        )

        # --- Step 3: unify output format → (B, C, N) ---
        occupancy = _normalize_aggregator_output(result, head)
        all_occs.append(occupancy)

    return all_occs


# ---------------------------------------------------------------------------
#  Sub-task 2:  fuse stage occupancies via average weighting
# ---------------------------------------------------------------------------

def fuse_stage_occupancies(
    all_occs: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse multiple stage occupancies through equal-weight logit averaging.

    Given a list of ``(B, C, N)`` logit tensors (one per decoder stage),
    this function:

    1. Stacks them along a new leading dimension: ``(S, B, C, N)``.
    2. Averages over the stage dimension: ``mean(dim=0) → (B, C, N)``.
    3. Takes ``argmax`` over the class dimension to produce hard labels.

    Args:
        all_occs:  ``List[Tensor (B, C, N)]`` — per-stage occupancy logits.

    Returns:
        fused_logits:  ``Tensor (B, C, N)`` — averaged soft logits.
        hard_labels:   ``Tensor (B, N)``    — argmax class indices.
    """
    if not all_occs:
        raise ValueError("all_occs is empty — no stage produced valid occupancy.")

    # --- Stack all stages and compute the mean ---
    stacked = torch.stack(all_occs, dim=0)   # (S, B, C, N)
    fused_logits = stacked.mean(dim=0)        # (B, C, N)

    # --- Hard labels via argmax ---
    hard_labels = fused_logits.argmax(dim=1)  # (B, N)

    return fused_logits, hard_labels
