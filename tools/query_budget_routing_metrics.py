"""Frozen query-routing census using the historical CUDA support definition.

``routing_metrics(network, out, batch, variants=...)`` returns a JSON-compatible
frame summary and a dictionary of one-dimensional NumPy arrays, one row per
query. ``out`` must retain the encoder's unmodified ``representation`` and
``allocation`` in addition to its usual rendered predictions. Batch size is one.

Ownership follows ``stat_query_budget.frame_statistics`` exactly: each valid
camera foreground voxel chooses its nearest stage-5 center, without a validity
filter on that nearest-neighbor search. Invalid query counts and the GT assigned
to them are reported explicitly. Invalid final candidates never enter a census.
The arrays preserve all query IDs, including invalid and zero-load queries.

Core coverage is geometric, class-agnostic 1-sigma coverage under the renderer's
native candidate bounds and positive FP32 opacity times kernel. It is not a
threshold on rendered foreground evidence and is not an occupancy prediction.
All variants retain the original Q x K ordering, so setting excluded opacity to
zero preserves the parent-query mapping inside the historical census kernel.

No gradients, optimizer steps, image preprocessing, or model-mode changes occur.
"""
from __future__ import annotations

import numpy as np
import torch


VARIANTS = ('current_hard', 'uniform64_per_query', 'all_candidates')


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


def _ranks(x):
    """Average ranks for ties; stable across platforms and no SciPy dependency."""
    x = np.asarray(x)
    order = np.argsort(x, kind='stable')
    result = np.empty(len(x), dtype=np.float64)
    if not len(x):
        return result
    starts = np.r_[0, np.flatnonzero(x[order][1:] != x[order][:-1]) + 1]
    stops = np.r_[starts[1:], len(x)]
    for start, stop in zip(starts, stops):
        result[order[start:stop]] = (start + stop - 1) / 2
    return result


def _correlations(load, allocation, query_valid):
    groups = {'all_queries': np.ones(len(load), dtype=bool),
              'valid_queries': query_valid,
              'positive_load_queries': load > 0}
    result = {}
    for name, selected in groups.items():
        x, y = load[selected], allocation[selected]
        result[name] = dict(count=int(selected.sum()), pearson=_pearson(x, y),
                            spearman=_pearson(_ranks(x), _ranks(y)))
    return result


def _field(batch, names):
    for name in names:
        if name in batch:
            return batch[name]
    raise KeyError(f'Expected one of {names} in the evaluation batch')


@torch.inference_mode()
def routing_metrics(network, out, batch, variants=VARIANTS):
    """Return ``(frame_summary, per_query_arrays)`` without modifying inputs.

    Ratios are fractions, not percentages. Summary totals are additive except
    ratios/correlations; aggregate raw counts or the per-query arrays across
    frames instead of averaging ratios with different denominators. ``top20``
    uses exactly floor(Q/5) query IDs sorted by GT load (stable lower-ID ties),
    preserving the historical all-query denominator. Arrays also retain validity
    flags to permit an explicitly labeled sensitivity analysis if needed.
    """
    from loss.opus_set_loss import _nearest_indices
    from tools.probe_gaussian_scale_supervision import layout
    from tools.query_budget_stats import extension

    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError(f'Unique nonempty variants from {VARIANTS} required')
    states = out['representation']
    final, coarse_state = states[-1], states[-2]
    grouped = final['gaussian']
    if grouped.means.ndim != 4 or grouped.means.shape[0] != 1:
        raise ValueError('Expected unmodified [1,Q,K,3] final Gaussian candidates')
    _, queries, children, _ = grouped.means.shape
    allocation = out['allocation']
    if (int(allocation['candidate_count']) != queries * children or
            int(allocation['children']) != children):
        raise ValueError('Allocation dimensions differ from original candidates')
    # Only flatten grouped tensor fields; preserve optional unset fields.
    gaussian = type(grouped)(*(None if value is None else value.flatten(1, 2)
                              for value in grouped))
    # layout calls prepare_gaussian_args, then removes exactly its appended empty
    # Gaussian. Consequently all arrays below contain decoder candidates only.
    _, values = layout(network.head, gaussian)
    means, scales, inverse, alpha, semantics, centers, radii = values
    if len(means) != queries * children:
        raise AssertionError('Global empty Gaussian was not removed correctly')
    device = means.device
    if device.type != 'cuda':
        raise ValueError('The audited historical census requires CUDA tensors')
    final_valid = final['query_valid_mask'][0].bool().reshape(queries)
    coarse_valid = coarse_state['query_valid_mask'][0].bool().reshape(queries)
    candidate_valid = final_valid[:, None].expand(queries, children).reshape(-1)
    original_alpha = alpha.masked_fill(~candidate_valid, 0).contiguous()
    coarse_grouped = coarse_state['gaussian'].means
    coarse_children = coarse_grouped.shape[2]
    coarse = coarse_grouped[0].flatten(0, 1).contiguous()
    if coarse_grouped.shape[:2] != (1, queries):
        raise ValueError('Stage-5 and final query IDs must align')
    # Historical frame_statistics uses out['gaussians'][-2], without filtering.
    if 'gaussians' in out:
        torch.testing.assert_close(coarse, out['gaussians'][-2].means[0], rtol=0, atol=0)
    xyz = out['sampled_xyz'][0].float().contiguous()
    labels = _field(batch, ('occ_label', 'voxel_label')).to(device).flatten().long()
    camera = _field(batch, ('occ_cam_mask', 'camera_mask')).to(device).flatten().bool().contiguous()
    valid_voxel = camera & (labels >= 0) & (labels < 18)
    targets = torch.where(valid_voxel & (labels < 17))[0]
    gt = xyz[targets]
    nearest = (_nearest_indices(gt, coarse, 1024) if len(gt) else
               torch.empty(0, device=device, dtype=torch.long))
    assigned = torch.div(nearest, coarse_children, rounding_mode='floor').long()
    distance = (gt - coarse[nearest]).norm(dim=1)
    load = torch.bincount(assigned, minlength=queries)
    lookup = torch.full((len(labels),), -1, device=device, dtype=torch.int32)
    lookup[targets] = torch.arange(len(targets), device=device, dtype=torch.int32)
    # Winner only affects the unused dominance-owner output, not core coverage.
    winner = out['pred_occ'][-1][0, :17].argmax(0).int().contiguous()
    shape = (network.head.aggregator.H, network.head.aggregator.W, network.head.aggregator.D)
    if int(np.prod(shape)) != len(labels) or len(xyz) != len(labels):
        raise ValueError('Full voxel-grid predictions are required by census')

    indices = allocation['selected_indices'][0].long()
    if (len(indices) != int(allocation['keep_budget']) or
            len(torch.unique(indices)) != len(indices) or
            bool((indices < 0).any()) or bool((indices >= len(means)).any())):
        raise AssertionError('Hard selection must contain unique in-range candidates')
    if not bool(candidate_valid[indices].all()):
        raise AssertionError('Current hard routing selected an invalid query')
    hard_mask = torch.zeros(len(means), device=device, dtype=torch.bool)
    hard_mask[indices] = True
    hard_counts = hard_mask.reshape(queries, children).sum(1)
    torch.testing.assert_close(hard_counts, allocation['selected_per_query'][0], rtol=0, atol=0)
    gates = allocation['gates'][0].reshape(queries, children)
    score = allocation['query_scores'][0].reshape(queries)
    pred = out['final_occ'].flatten().long()
    load_np = load.cpu().numpy()
    final_valid_np = final_valid.cpu().numpy()
    top_n = max(queries // 5, 1)
    top_indices = np.argsort(-load_np, kind='stable')[:top_n]
    top = np.zeros(queries, dtype=bool)
    top[top_indices] = True

    def grouped_sum(value):
        result = torch.zeros(queries, device=device, dtype=torch.float64)
        result.index_add_(0, assigned, value.double())
        return result.cpu().numpy()

    arrays = dict(query_id=np.arange(queries, dtype=np.int32), gt_load=load_np,
                  final_query_valid=final_valid_np, coarse_query_valid=coarse_valid.cpu().numpy(),
                  gt_high_load_top20=top, query_score=score.cpu().numpy(),
                  selected_count=hard_counts.cpu().numpy(), soft_mass=gates.sum(1).cpu().numpy(),
                  mean_opacity=alpha.reshape(queries, children).mean(1).cpu().numpy(),
                  gt_coarse_distance_sum=grouped_sum(distance),
                  hard_gt_miss=grouped_sum(pred[targets] == 17),
                  hard_gt_wrong_class=grouped_sum((pred[targets] != labels[targets]) & (pred[targets] != 17)))
    summary = dict(queries=int(queries), children=int(children), foreground_voxels=int(len(gt)),
                   coarse_valid_queries=int(coarse_valid.sum()), final_valid_queries=int(final_valid.sum()),
                   gt_assigned_to_invalid_coarse_query=int(load[~coarse_valid].sum()),
                   gt_assigned_to_invalid_final_query=int(load[~final_valid].sum()),
                   positive_load_queries=int((load > 0).sum()), zero_load_queries=int((load == 0).sum()),
                   high_load_query_count=int(top_n), high_load_foreground=int(load_np[top].sum()),
                   high_load_foreground_share=_ratio(load_np[top].sum(), len(gt)),
                   soft_mass_sum=float(gates.sum()),
                   high_load_soft_mass=float(arrays['soft_mass'][top].sum()),
                   high_load_soft_mass_share=_ratio(arrays['soft_mass'][top].sum(), gates.sum().item()),
                   gt_load_query_score_correlations=_correlations(load_np, arrays['query_score'], final_valid_np),
                   gt_load_soft_mass_correlations=_correlations(load_np, arrays['soft_mass'], final_valid_np),
                   assignment='nearest stage-5 center, historical unfiltered ownership',
                   core_definition='native candidate bounds; positive opacity*kernel; Mahalanobis distance <= 1',
                   variants={})

    for name in variants:
        if name == 'current_hard':
            selected = hard_mask
        elif name == 'uniform64_per_query':
            if children < 64:
                raise ValueError('uniform64_per_query requires at least 64 candidates per query')
            local = torch.argsort(alpha.reshape(queries, children), dim=1, descending=True, stable=True)[:, :64]
            selected = torch.zeros((queries, children), device=device, dtype=torch.bool)
            selected.scatter_(1, local, True)
            selected = selected.flatten() & candidate_valid
        else:
            selected = candidate_valid
        masked_alpha = original_alpha.masked_fill(~selected, 0).contiguous()
        census = extension().census(means, inverse, masked_alpha, semantics, centers, radii, xyz,
            labels.int().contiguous(), camera, lookup, winner, assigned.int().contiguous(),
            queries, children, *shape)
        counts, bits, packed, gs, own, query_counts, query_coverage = census
        no_core, own = counts[:, 1] == 0, own.bool()
        if bool((own & no_core).any()) or not torch.equal(query_counts > 0, counts[:, 1] > 0):
            raise AssertionError('Historical core-count and query-bitset reductions disagree')
        if not torch.equal(query_coverage[:, 0], query_coverage[:, 1] + query_coverage[:, 2]):
            raise AssertionError('Exclusive and shared query coverage fail conservation')
        per_variant = dict(selected_count=selected.reshape(queries, children).sum(1).cpu().numpy(),
                           gt_no_core=grouped_sum(no_core), gt_own_core=grouped_sum(own),
                           gt_neighbor_only_core=grouped_sum(~no_core & ~own),
                           gt_core_count_sum=grouped_sum(counts[:, 1]),
                           core_coverage_all_fg=query_coverage[:, 0].cpu().numpy(),
                           core_exclusive=query_coverage[:, 2].cpu().numpy(),
                           core_gs_touches_fg=(gs[:, 2] > 0).reshape(queries, children).sum(1).cpu().numpy())
        np.testing.assert_array_equal(per_variant['gt_no_core'] + per_variant['gt_own_core'] +
                                      per_variant['gt_neighbor_only_core'], load_np)
        arrays.update({f'{name}_{key}': value for key, value in per_variant.items()})
        n = per_variant['selected_count']
        holes = per_variant['gt_no_core']
        active = load_np > 0
        zero_retained_active = active & (n == 0)
        total_selected = int(n.sum())
        summary['variants'][name] = dict(
            selected_count=total_selected, zero_selected_queries=int((n == 0).sum()),
            zero_selected_query_fraction=float((n == 0).mean()),
            positive_load_zero_selected_queries=int(zero_retained_active.sum()),
            positive_load_zero_selected_query_fraction=_ratio(zero_retained_active.sum(), active.sum()),
            foreground_assigned_to_zero_selected_queries=int(load_np[n == 0].sum()),
            high_load_selected_count=int(n[top].sum()), high_load_budget_share=_ratio(n[top].sum(), total_selected),
            zero_load_selected_count=int(n[~active].sum()), zero_load_budget_share=_ratio(n[~active].sum(), total_selected),
            no_core_voxels=int(holes.sum()), no_core_rate=_ratio(holes.sum(), len(gt)),
            own_core_voxels=int(per_variant['gt_own_core'].sum()),
            neighbor_only_core_voxels=int(per_variant['gt_neighbor_only_core'].sum()),
            core_count_sum=int(per_variant['gt_core_count_sum'].sum()),
            high_load_no_core_voxels=int(holes[top].sum()),
            high_load_no_core_rate=_ratio(holes[top].sum(), load_np[top].sum()),
            high_load_share_of_all_holes=_ratio(holes[top].sum(), holes.sum()),
            gt_load_selected_count_correlations=_correlations(load_np, n, final_valid_np))
        del census, counts, bits, packed, gs, own, query_counts, query_coverage
    return summary, arrays
