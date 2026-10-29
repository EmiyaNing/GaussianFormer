"""Fixed-A0, fixed-geometry mechanism probes; these are not validation scores."""
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config

from tools.gaussian_responsibility_experiment import RUN, initialize, write
from tools.eval_gaussian_responsibility import build, confusion, plain, summarize_counts
from tools.empty_decision_optimization import build_loader, to_cuda
from tools.empty_mechanism_stage1 import network_output
from loss.opus_set_loss import OPUSSetLoss, _nearest_indices
from loss.gaussian_responsibility_loss import sparse_support, allocation_targets, scatter_sum


def measure(q, o, y, mask):
    q, o, y = q[mask], o[mask], y[mask]
    foreground = y < 17
    pred = torch.cat([q * o[:, None], 1 - o[:, None]], -1).argmax(-1)
    cm = confusion(y.cpu().numpy(), pred.cpu().numpy())
    return dict(counts=cm.tolist(), foreground=int(foreground.sum()),
        rank_correct=int(((q.argmax(-1) == y) & foreground).sum()),
        conditional_NLL_sum=float(-q[foreground].gather(1, y[foreground, None]).clamp_min(1e-6).log().sum()),
        metrics=plain(summarize_counts(cm)))


def probe_frame(net, batch, settings):
    with torch.no_grad():
        out = network_output(net, batch)
        gaussian = out['gaussian']
        flat = gaussian._replace(**{k: getattr(gaussian, k)[0] for k in gaussian._fields
                                    if getattr(gaussian, k) is not None})
        xyz = batch['occ_xyz'][0].reshape(-1, 3)
        y = batch['occ_label'][0].flatten().long()
        selected = (y >= 0) & (y < 17)
        gt_xyz, gt_y = xyz[selected], y[selected]
        nn = _nearest_indices(flat.means, gt_xyz, 1024)
        nearest = gt_y[nn]
        gt_nearest = _nearest_indices(gt_xyz, flat.means, 1024)
        edges = sparse_support(flat, gt_xyz, gt_y)
        t2, diagnostic2, state2 = allocation_targets(*edges, gt_y, nearest, gt_nearest)
        t3, diagnostic3, state3 = allocation_targets(*edges, gt_y, nearest, gt_nearest, transport=True)
        q0 = out['conditional_semantics'][-1][0].T
        o0 = out['bin_logits'][-1][0]
        mask = batch['occ_cam_mask'][0].flatten().bool() & (y >= 0) & (y < 18)
        base = measure(q0, o0, y, mask)
        # Check the sparse reference against actual CUDA rendering, including
        # the covariance normalizer and opacity. Guard edges are excluded.
        row, col, weight = edges
        density = scatter_sum(weight, col, len(gt_y))
        mixed = weight.new_zeros(len(gt_y), 17)
        semantic = out['opus_pred_logits'][-1][0].softmax(-1)
        for start in range(0, len(row), 131072):
            stop = start + 131072
            mixed.index_add_(0, col[start:stop], weight[start:stop, None] * semantic[row[start:stop]])
        mixed /= density[:, None].clamp_min(1e-30)
        mixed[density <= 1e-9] = 1 / 17
        error = float((mixed - q0[selected]).abs().max())
        assert error < 5e-4, error
        # Pure renderer evidence, separate from the training guard edges.
        actual_edge = density[col] > 1e-9
        ar, av = row[actual_edge], col[actual_edge]
        responsibility = weight[actual_edge] / density[av]
        actual_mass = scatter_sum(responsibility, ar, len(nearest))
        hist = scatter_sum(responsibility, ar * 17 + gt_y[av], len(nearest) * 17).reshape(-1, 17)
        actual_target = hist / actual_mass[:, None].clamp_min(1e-30)
        supported = actual_mass >= 1e-3
        actual_mismatch = (actual_target.argmax(-1) != nearest) & supported
        actual_share = actual_target.gather(1, nearest[:, None]).squeeze(-1)
        render_only = dict(supported_gaussians=int(supported.sum()),
            mismatched_gaussians=int(actual_mismatch.sum()),
            responsibility_mass=float(actual_mass.sum()),
            mismatch_mass=float(actual_mass[actual_mismatch].sum()),
            nearest_share_mass_sum=float((actual_share * actual_mass).sum()),
            supported_nearest_share_sum=float(actual_share[supported].sum()),
            guard_voxels=int((density <= 1e-9).sum()), gt_voxels=len(gt_y),
            minimum_gaussian_mass=1e-3)
        # Does responsibility disagreement concentrate in difficult voxels?
        disagree_mass = scatter_sum(weight * actual_mismatch[row], col, len(gt_y))
        fraction = disagree_mass / density.clamp_min(1e-30)
        fg_mask = mask[selected]
        exposure = []
        edges_fraction = [0., .25, .5, .75, 1.00001]
        winner = q0[selected].argmax(-1)
        native = out['final_occ'][0][selected]
        for low, high in zip(edges_fraction[:-1], edges_fraction[1:]):
            group = fg_mask & (density > 1e-9) & (fraction >= low) & (fraction < high)
            exposure.append(dict(low=low, high=high, voxels=int(group.sum()),
                rank_correct=int(((winner == gt_y) & group).sum()),
                final_correct=int(((native == gt_y) & group).sum()),
                missed=int(((native == 17) & group).sum())))
    loss_module = OPUSSetLoss(**settings)
    sample_weight = loss_module.class_weights.to(q0)[nearest] * loss_module._distance_weight(gt_xyz[nn])
    raw = out['opus_pred_logits'][-1][0].detach()
    z = raw.clone().requires_grad_(True)
    losses = dict(A0=loss_module._sigmoid_focal(z, nearest, sample_weight),
        A1=(-(F.one_hot(nearest, 17) * z.log_softmax(-1)).sum(-1) * sample_weight).mean(),
        A2=(-(t2 * z.log_softmax(-1)).sum(-1) * sample_weight).mean(),
        A3=(-(t3 * z.log_softmax(-1)).sum(-1) * sample_weight).mean())
    gradients = {key: torch.autograd.grad(loss, z, retain_graph=True)[0].detach() for key, loss in losses.items()}
    args = tuple(t.detach() for t in net.head.prepare_gaussian_args(gaussian))
    baseline_pred = out['final_occ'][0]
    records = []
    with torch.no_grad():
        for method, gradient in gradients.items():
            for step in (.01, .03, .1):
                delta = step * gradient / gradient.square().mean().sqrt().clamp_min(1e-30)
                updated = (raw - delta).softmax(-1)
                payload = torch.cat([updated, torch.zeros_like(updated[:, :1])], -1)[None]
                means, opacity, _, scales, inverse = args
                q, o, _ = net.head.aggregator(out['sampled_xyz'].clone().float(), means,
                    opacity.reshape(1, -1), payload, scales, inverse)
                assert torch.equal(o, o0), 'A semantic-only update changed occupancy evidence'
                measured = measure(q[:, :17], o, y, mask)
                pred = torch.cat([q[:, :17] * o[:, None], 1 - o[:, None]], -1).argmax(-1)
                correct = (y < 17) & (baseline_pred == y) & mask
                missed = (y < 17) & (baseline_pred == 17) & mask
                measured.update(correct_to_empty=int((correct & (pred == 17)).sum()),
                    missed_to_correct=int((missed & (pred == y)).sum()),
                    empty_to_foreground=int(((y == 17) & (baseline_pred == 17) & (pred < 17) & mask).sum()),
                    foreground_to_empty=int(((y == 17) & (baseline_pred < 17) & (pred == 17) & mask).sum()))
                records.append(dict(method=method, logit_rms_step=step,
                    actual_rms=float(delta.square().mean().sqrt()), occupancy_unchanged=True,
                    gradient_norm=float(gradient.norm()), outcome=measured))
    return dict(baseline=base, A2=diagnostic2, A3=diagnostic3, render_only=render_only,
        sparse_renderer_max_error=error, exposure=exposure, interventions=records)


def main():
    protocol = initialize()
    folder = RUN / 'probe'
    folder.mkdir(parents=True, exist_ok=True)
    net = build('A0')
    cfg = Config.fromfile('config/center_support_contribution1/a0_original.py')
    settings = cfg.loss.loss_cfgs[0].to_dict()
    for k in ('type', 'gt_domain', 'final_gt_domain', 'final_geometry_weight', 'final_classification_weight'):
        settings.pop(k)
    for split, frames in protocol['probe_samples'].items():
        indices = [r['index'] for r in frames]
        _, loader, _ = build_loader(split, indices)
        for frame, batch in zip(frames, loader):
            path = folder / f'{split}_{frame["index"]:04d}.json'
            if path.exists():
                continue
            result = probe_frame(net, to_cuda(batch), settings)
            write(path, dict(split=split, frame=frame, **result))
            print('probe', split, frame['index'], result['A2']['mismatched_gaussians'], flush=True)
    files = list(folder.glob('train_*.json')) + list(folder.glob('val_*.json'))
    assert len(files) == 64
    write(folder / 'complete.json', dict(status='complete', frames=64,
        scope='32 train +32 validation fixed-A0, same-frame oracle-gradient probes; not held-out performance',
        semantic_only=True, occupancy_unchanged=True, steps=[.01, .03, .1]))


if __name__ == '__main__':
    main()
