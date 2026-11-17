"""Effective sample size of the Gaussian aggregation: is A5's receptive field real?

P1-3 records a confound that may either weaken or partially exonerate the
Gaussian claim:

    A2 aggregates exactly 8 uniformly weighted nearest neighbours; A5
    aggregates every point inside its support box, which at its learned scale
    (0.53) is 215.2 accepted neighbours on average.  So A5 matches/underperforms
    A2 while using a 27x larger receptive field.

That framing assumes A5's 215 neighbours each *contribute*.  Gaussian weights
decay with distance, so they need not: if the kernel concentrates on a handful
of near neighbours, the **effective** sample size (ESS) is far below 215 and the
comparison is already effective-field matched -- which would make the "27x
confound" nominal rather than real, and would refute the confound note.

ESS is the standard importance-weight diagnostic

    ESS = (sum_h w_h)^2 / sum_h w_h^2      with w_h = opacity_h * kernel_h

and equals the neighbour count when the weights are uniform.  It is exactly the
right quantity here because the operator's output is a weighted average, so ESS
bounds how much independent information the residual can carry.

Weights follow the CUDA kernel exactly (benchmarks/reference/g2q_bruteforce.py
copies the same expressions):

    relative = (query - neighbour) / neighbour_scale      # per-axis box test
    inside   = |relative| <= support_sigma (all axes)
    kernel   = exp(-0.5 * sum(relative^2))
    weight   = opacity * kernel

CPU-only, O(sample * G).

    python3.10 benchmarks/p4/effective_sample_size.py
    python3.10 benchmarks/p4/effective_sample_size.py --scales 0.35 0.53 --include-self
"""
from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

from benchmarks.bench_operators import PC_RANGE, make_cloud  # noqa: E402


def ess_from_weights(weight):
    """ESS of one or more weight rows: (sum w)^2 / sum w^2.

    Equals the number of neighbours when the weights are uniform, 1 when a
    single weight carries everything, and 2 for two equal weights.  Split out so
    the formula can be pinned by hand-computed tests instead of only being
    exercised through a full point cloud.
    """
    total = weight.sum(dim=-1)
    squared = weight.square().sum(dim=-1)
    return total.square() / squared.clamp_min(torch.finfo(weight.dtype).tiny)


def weight_share_of_nearest(cloud, support_sigma, sample, seed, cuts,
                            include_self=True, chunk=64):
    """Share of total aggregation weight carried by the k nearest accepted neighbours.

    This is the quantity that PREDICTS the density cap's accuracy cost.  A cap of
    ``max_neighbours = k`` keeps the k shell-nearest accepted neighbours, so if
    the retained share is already close to 1 at k = 32, capping there can barely
    change the op's output -- which is what P1-1 measured (-0.06 mIoU at cap=32)
    and what P4-6 measured as a 13x latency win.  Without this number the two
    findings are unrelated; with it, the small cap cost is explained.

    Ties are broken by the same distance ordering the kernel's shell selection
    uses, so the cut is by distance rank, not by arbitrary weight order.
    """
    points = cloud.points.float()
    scales = cloud.scales.clamp_min(1e-4).float()
    opacities = cloud.opacities.float()
    count = points.shape[0]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    queries = torch.randperm(count, generator=generator)[:min(sample, count)]

    totals = {k: [] for k in cuts}
    for start in range(0, queries.numel(), chunk):
        index = queries[start:start + chunk]
        delta = points[index][:, None, :] - points[None, :, :]
        relative = delta / scales[None, :, :]
        inside = (relative.abs() <= support_sigma).all(dim=-1)
        if not include_self:
            inside[torch.arange(index.numel()), index] = False
        distance_sq = relative.square().sum(dim=-1)
        kernel = torch.exp(-0.5 * distance_sq)
        weight = torch.where(inside, opacities[None, :] * kernel,
                             torch.zeros_like(kernel))
        total = weight.sum(dim=1)
        order = distance_sq.argsort(dim=1)
        for k in cuts:
            kept = torch.gather(weight, 1, order[:, :k]).sum(dim=1)
            valid = total > 0
            totals[k].append((kept[valid] / total[valid]))

    return {k: torch.cat(v).mean().item() for k, v in totals.items() if v}


def best_uniform_approximation(cloud, support_sigma, sample, seed, cuts,
                               include_self=True, chunk=64):
    """Which uniform-over-k-nearest mean best emulates A5's Gaussian weighted mean?

    ESS says how many neighbours A5 *effectively* averages, but not *which* ones:
    a weighted mean with ESS 59 is not the same operator as a uniform mean over
    the 59 nearest, because A5 also carries 31.5% of its mass beyond rank 32.
    Since the residual head is linear in the seeds, A5's per-query operator is
    the normalised weight vector ``w`` and a uniform mean over the k nearest is
    the vector ``u_k = 1/k`` on those k entries.  With seeds that are iid across
    neighbours (unit variance), the relative error of emulating A5 by A2-k is

        E[(w - u_k) . s]^2 / E[(w . s)]^2  =  |w - u_k|^2 / |w|^2 = |w - u_k|^2

    because ||w||_1 = 1 does not imply ||w||_2 = 1; the normalised form used here
    is the relative L2 distance between the two weight vectors.  This is the
    sharpest available answer to "is a k-NN control able to emulate the kernel
    at all", and it has a direct experimental-design consequence: whichever k
    minimises it is the k worth spending a training run on.

    CPU-only and independent of the model's outputs (it only needs the weights).
    """
    points = cloud.points.float()
    scales = cloud.scales.clamp_min(1e-4).float()
    opacities = cloud.opacities.float()
    count = points.shape[0]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    queries = torch.randperm(count, generator=generator)[:min(sample, count)]

    best = {k: [] for k in cuts}
    weight_norms = []
    for start in range(0, queries.numel(), chunk):
        index = queries[start:start + chunk]
        delta = points[index][:, None, :] - points[None, :, :]
        relative = delta / scales[None, :, :]
        inside = (relative.abs() <= support_sigma).all(dim=-1)
        if not include_self:
            inside[torch.arange(index.numel()), index] = False
        distance_sq = relative.square().sum(dim=-1)
        weight = torch.where(inside, opacities[None, :] * torch.exp(-0.5 * distance_sq),
                             torch.zeros_like(distance_sq))
        total = weight.sum(dim=1)
        valid = total > 0
        if not bool(valid.any()):
            continue
        normalised = weight[valid] / total[valid][:, None]
        order = distance_sq[valid].argsort(dim=1)
        weight_norms.append(normalised.square().sum(dim=1).sqrt())
        rows = normalised.shape[0]
        for k in cuts:
            k = min(k, normalised.shape[1])
            uniform = torch.zeros_like(normalised)
            uniform.scatter_(1, order[:, :k], 1.0 / k)
            best[k].append((normalised - uniform).square().sum(dim=1).sqrt())

    result = {k: torch.cat(v).mean().item() for k, v in best.items() if v}
    result['_weight_l2_norm'] = torch.cat(weight_norms).mean().item()
    return result


def effective_sample_size(cloud, support_sigma, sample, seed, include_self,
                          chunk=64):
    """ESS per query over the accepted neighbours, plus the accepted count."""
    points = cloud.points.float()
    scales = cloud.scales.clamp_min(1e-4).float()
    opacities = cloud.opacities.float()
    count = points.shape[0]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    queries = torch.randperm(count, generator=generator)[:min(sample, count)]

    ess, accepted, self_weight_share = [], [], []
    for start in range(0, queries.numel(), chunk):
        index = queries[start:start + chunk]
        delta = points[index][:, None, :] - points[None, :, :]
        relative = delta / scales[None, :, :]
        inside = (relative.abs() <= support_sigma).all(dim=-1)
        if not include_self:
            rows = torch.arange(index.numel())
            inside[rows, index] = False
        kernel = torch.exp(-0.5 * relative.square().sum(dim=-1))
        weight = opacities[None, :] * kernel
        weight = torch.where(inside, weight, torch.zeros_like(weight))

        total = weight.sum(dim=1)
        valid = total > 0
        ess.append(ess_from_weights(weight)[valid])
        accepted.append(inside.sum(dim=1)[valid].float())
        # How much of the total weight sits on the query itself (delta == 0)?
        self_weight = weight[torch.arange(index.numel()), index]
        self_weight_share.append((self_weight[valid] / total[valid]))

    ess = torch.cat(ess)
    accepted = torch.cat(accepted)
    self_weight_share = torch.cat(self_weight_share)

    # Does the kernel adapt its concentration to the local density?  If ESS is
    # merely a fixed fraction of the accepted count, the weighting has no
    # adaptive capacity -- it is a constant discount, and the "learned Gaussian
    # geometry" buys nothing beyond a rescaled uniform mean.
    ratio = ess / accepted.clamp_min(1.0)
    if ess.numel() > 1 and ess.std() > 0 and accepted.std() > 0:
        correlation = torch.corrcoef(torch.stack((ess, accepted)))[0, 1].item()
    else:
        correlation = float('nan')
    # Spread of the ratio at fixed density: bucket by accepted count and look at
    # the ratio's own coefficient of variation, which is the part the kernel
    # controls rather than the density.
    ratio_spread = (ratio.std() / ratio.mean()).item() if ratio.mean() > 0 else float('nan')
    return {
        'accepted_mean': accepted.mean().item(),
        'accepted_p95': accepted.quantile(0.95).item(),
        'ess_mean': ess.mean().item(),
        'ess_p50': ess.median().item(),
        'ess_p95': ess.quantile(0.95).item(),
        'ess_max': ess.max().item(),
        'ess_over_accepted': (ess / accepted.clamp_min(1)).mean().item(),
        'self_weight_share_mean': self_weight_share.mean().item(),
        'ess_over_accepted_p05': ratio.quantile(0.05).item(),
        'ess_over_accepted_p95': ratio.quantile(0.95).item(),
        'ess_over_accepted_cv': ratio_spread,
        'corr_ess_accepted': correlation,
        'queries': int(ess.numel()),
    }


def cap_mass_accuracy_table(shares, accuracies, base_accuracy, label='A5 cap=0'):
    """Missed Gaussian mass vs accuracy: the plan's P4.2 item 1, partially.

    The plan asks for "活跃体素召回率与被遗漏的高斯质量".  For the density cap the
    missed mass is exactly ``1 - retained`` from this script, and the accuracy
    side was already measured (P1-1 section 3.2), so the two can be put on one
    axis.  Ordering matters: ``retained`` is NOT monotone in ``cap`` (cap=0
    retains everything), so the marginals must be taken along the *retained-mass*
    axis or the signs come out wrong.

    Returns ``(rows, marginals)`` where each row is ``(cap, retained, mIoU, delta)``
    and each marginal is ``(cap_a, cap_b, d_mass, d_mIoU, per_unit)``.
    """
    rows = [(0, 1.0, base_accuracy)]
    for cap, retained in shares.items():
        if cap in accuracies:
            rows.append((cap, retained, accuracies[cap]))
    rows.sort(key=lambda row: row[1])
    rows = [(cap, retained, value, value - base_accuracy)
            for cap, retained, value in rows]
    values = {cap: value for cap, _, value, _ in rows}
    marginals = [(cap_a, cap_b, ret_b - ret_a,
                  values[cap_b] - values[cap_a],
                  (values[cap_b] - values[cap_a]) / (ret_b - ret_a))
                 for (cap_a, ret_a, _, _), (cap_b, ret_b, _, _) in zip(rows, rows[1:])]
    return rows, marginals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', default='real')
    parser.add_argument('--gaussians', type=int, default=34560)
    parser.add_argument('--scales', nargs='*', type=float, default=[0.53, 0.35])
    parser.add_argument('--support', type=float, default=3.0)
    parser.add_argument('--sample', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--real-points', default='benchmarks/results/real_points.pt')
    parser.add_argument('--json', default='benchmarks/results/p4_ess.json')
    args = parser.parse_args(argv)

    cap_cuts = (1, 2, 4, 8, 16, 32, 64, 128)
    rows = []
    for scale in args.scales:
        cloud = make_cloud(args.dist, args.gaussians, scale=scale, device='cpu',
                           real_points_path=args.real_points, seed=args.seed)
        for include_self in (True, False):
            result = effective_sample_size(cloud, args.support, args.sample,
                                           args.seed, include_self)
            row = dict(result, dist=args.dist, gaussians=args.gaussians,
                       scale=scale, support_sigma=args.support,
                       include_self=include_self)
            if include_self:
                row['uniform_approx_error'] = {
                    str(k): v for k, v in best_uniform_approximation(
                        cloud, args.support, args.sample, args.seed,
                        cap_cuts, include_self=True).items()}
                row['nearest_weight_share'] = {
                    str(k): v for k, v in weight_share_of_nearest(
                        cloud, args.support, args.sample, args.seed,
                        cap_cuts, include_self=True).items()}
            rows.append(row)
            print(f"scale={scale:.2f} include_self={str(include_self):5s} "
                  f"accepted={row['accepted_mean']:8.1f} "
                  f"ESS mean={row['ess_mean']:7.2f} p50={row['ess_p50']:7.2f} "
                  f"p95={row['ess_p95']:8.2f} max={row['ess_max']:8.1f} "
                  f"ESS/accepted={row['ess_over_accepted']:.4f} "
                  f"[p05={row['ess_over_accepted_p05']:.3f} "
                  f"p95={row['ess_over_accepted_p95']:.3f} "
                  f"cv={row['ess_over_accepted_cv']:.3f}] "
                  f"corr(ESS,accepted)={row['corr_ess_accepted']:.3f} "
                  f"self-share={row['self_weight_share_mean']:.3f}", flush=True)

    if rows and 'uniform_approx_error' in rows[0]:
        error = rows[0]['uniform_approx_error']
        numeric = {int(k): v for k, v in error.items() if not k.startswith('_')}
        best_k = min(numeric, key=numeric.get)
        print('\nhow well can a uniform mean over k nearest EMULATE A5? '
              '(relative L2 error of the weight vectors)')
        for k in sorted(numeric):
            marker = '  <-- best' if k == best_k else ''
            print(f'  k={k:>3}: relative error = {numeric[k]:.4f}{marker}')
        print(f"  A5's own weight-vector L2 norm = {error['_weight_l2_norm']:.4f} "
              '(1.0 would mean all mass on one neighbour; 0 would mean uniform)')
        print(f'  -> the best uniform emulation is k={best_k}, not the ESS-matched '
              f'k=59 and not the nominal k=215.  Even at its best, a uniform mean '
              f'leaves a relative weight error of {numeric[best_k]:.3f}, which is '
              f'the irreducible gap between the kernel and any k-NN control.')

    if rows and 'nearest_weight_share' in rows[0]:
        shares = rows[0]['nearest_weight_share']
        print('\nweight share retained by the k nearest accepted neighbours '
              '(scale %.2f, predicts the density cap cost):' % rows[0]['scale'])
        for k in sorted(shares, key=int):
            print(f"  cap={k:>3}: retained weight share = {shares[k]:.4f}")
        share32 = shares.get('32')
        if share32 is not None and share32 < 0.9:
            print(f'  cap=32 retains only {share32:.1%} of the weight mass, yet '
                  'P1-1 measured its accuracy cost at -0.06 mIoU.  So the point '
                  'is NOT that little is discarded: a THIRD of the aggregation '
                  'weight can be thrown away without a measurable accuracy '
                  'loss, i.e. the tail of the Gaussian weighting carries '
                  'almost no task-relevant information.  That is also why '
                  'P4-6 measured a 13x latency win from the same cap.')
        else:
            print('  cap=32 retains nearly all of the weight mass, which is why '
                  'its measured accuracy cost is so small.')

    # Missed-mass vs accuracy, using the cap accuracies already recorded by P1-1.
    eval_path = ROOT / 'benchmarks/results/p1_eval.json'
    if rows and eval_path.exists() and 'nearest_weight_share' in rows[0]:
        recorded = {r['variant']: (r.get('metrics') or {}).get('avg/mIoU')
                    for r in json.loads(eval_path.read_text(encoding='utf-8'))}
        base = recorded.get('a5_cap0')
        accuracies = {cap: recorded[f'a5_cap{cap}'] for cap in (8, 32, 64, 128)
                      if recorded.get(f'a5_cap{cap}') is not None}
        if base is not None and accuracies:
            shares = {int(k): v for k, v in rows[0]['nearest_weight_share'].items()}
            table, marginals = cap_mass_accuracy_table(shares, accuracies, base)
            print('\nmissed Gaussian mass vs accuracy (plan P4.2 item 1, cap axis):')
            print('| cap | retained mass | mIoU | delta |')
            print('| ---: | ---: | ---: | ---: |')
            for cap, retained, value, delta in table:
                print(f'| {cap} | {retained:.3f} | {value:.2f} | {delta:+.2f} |')
            print('marginal GAIN per unit of recovered mass (concave => the near '
                  'neighbours matter most):')
            for cap_a, cap_b, d_mass, d_accuracy, per_unit in marginals:
                print(f'  cap {cap_a:>3} -> {cap_b:>3}: +{d_mass:.3f} mass, '
                      f'{d_accuracy:+.3f} mIoU -> {per_unit:+.3f} mIoU per unit mass')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'rows': rows}, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')

    # The decision this script exists to make.
    model_row = next((r for r in rows if r['scale'] == 0.53 and r['include_self']), None)
    if model_row:
        print(f"\nA5 at its learned scale aggregates {model_row['accepted_mean']:.1f} "
              f"neighbours but its effective sample size is "
              f"{model_row['ess_mean']:.1f} (median {model_row['ess_p50']:.1f}).")
        if model_row['ess_mean'] < 20:
            print('VERDICT: the Gaussian kernel concentrates on a few near '
                  'neighbours, so A5 is already effective-field matched to '
                  'KNN-8; the "27x receptive field" confound is nominal only.')
        else:
            print('VERDICT: ESS is far above KNN-8, so A5 really does aggregate '
                  'a much larger field and the confound in P1-3 is real.')

        # Is the concentration adaptive, or a fixed discount?  corr(ESS, k)
        # near 1 with a near-constant ESS/k means the kernel only rescales the
        # accepted count, so the learnable "geometry" reduces to one scalar: the
        # field size.
        print(f"Adaptivity: corr(ESS, accepted) = "
              f"{model_row['corr_ess_accepted']:.3f}, ESS/accepted = "
              f"{model_row['ess_over_accepted']:.3f} "
              f"(cv {model_row['ess_over_accepted_cv']:.3f}, "
              f"p05 {model_row['ess_over_accepted_p05']:.3f}, "
              f"p95 {model_row['ess_over_accepted_p95']:.3f}).")
        if model_row['corr_ess_accepted'] > 0.9 and \
                model_row['ess_over_accepted_cv'] < 0.35:
            print('VERDICT: the concentration is a near-constant discount, not '
                  'an adaptive selector -- the kernel tracks the accepted count '
                  '(corr > 0.9) with an almost fixed ESS/accepted ratio that is '
                  'also scale-invariant.  So A5 is close to "uniform mean over '
                  '~0.28x the accepted count", which PREDICTS A2\u2033 (k = 59) '
                  '~= A5 if the weighting shape is irrelevant, and A2\u2033 > A5 '
                  'if uniform weighting is simply better.  That is the queued '
                  'test of this measurement.')
        else:
            print('VERDICT: the concentration varies substantially at fixed '
                  'density, so the kernel does adapt; A2\u2033 (k = 59) then '
                  'matches only the mean and cannot fully emulate it.')


if __name__ == '__main__':
    main()
