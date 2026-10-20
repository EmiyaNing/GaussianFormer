"""CPU falsifiable toy experiments for contribution 1, geometry and semantics."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/center_support_mpl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from loss.opus_set_loss import OPUSSetLoss
from tools.center_support_common import (contribution_diagnostic, field_metrics,
                                         kernel_weights, write_json)

LOWER = (-2.2, -2.2, -0.2)


def scene(name):
    x = torch.arange(-2., 2.01, .4, dtype=torch.float64)
    if name == 'diagonal_thin':
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        xyz = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(xx.numel())), -1)
        occupied = (xyz[:, 0].abs() <= .81) & ((xyz[:, 0] - xyz[:, 1]).abs() < .01)
        direction = torch.tensor([1., -1., 0.], dtype=xyz.dtype) / 2**.5
    else:
        xyz = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), -1)
        if name == 'adjacent':
            occupied = (x >= -.01) & (x <= .41)
        elif name == 'gap':
            occupied = (x.abs() - .4).abs() < .01
        else:
            occupied = x.abs() < .01
        direction = torch.tensor([1., 0., 0.], dtype=x.dtype)
    return xyz, occupied, direction


def inverse_cov(scales, angle=0.):
    r = scales.new_tensor([[np.cos(angle), -np.sin(angle), 0.],
                          [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    # Matches production covariance = (diag(scale) @ R).T @ (diag(scale) @ R).
    return r.T @ torch.diag_embed(scales.reciprocal().square()) @ r


def rendered(xyz, means, scales, opacity, semantics, angle=0.):
    w = kernel_weights(xyz, means, opacity, scales, inverse_cov(scales, angle), LOWER)
    # Fixed, spatially constant empty score is an explicit toy simplification.
    return torch.cat((w @ semantics, xyz.new_full((len(xyz), 1), 1.5)), -1)


def set_loss():
    return OPUSSetLoss(stage_weights=[1.], lambda_pts=.5, lambda_cls=2.,
                       rare_classes=(), pc_range=(*LOWER, 2.2, 2.2, .2)).double()


def scalar(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def geometry_terms(loss, means, xyz, occupied):
    gt = xyz[occupied]
    _, p, g, _, _ = loss._single(means, means.new_zeros(len(means), 1), gt,
                                torch.zeros(len(gt), dtype=torch.long),
                                torch.ones(len(gt), dtype=torch.bool))
    return .5 * p, .5 * g


def geometry_scan(out, steps):
    records, trajectories = [], []
    loss = set_loss()
    for name in ('adjacent', 'gap', 'isolated', 'diagonal_thin'):
        xyz, occupied, direction = scene(name)
        labels = (~occupied).long()
        anchor = xyz.new_tensor([.2, 0., 0.]) if name == 'adjacent' else xyz.new_zeros(3)
        centers = np.linspace(-.6, .6, 161)
        angle = -np.pi / 4 if name == 'diagonal_thin' else 0.
        figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
        for si, sigma in enumerate((.08, .16, .24, .40)):
            for amplitude in (1., 4., 8., 16.):
                values, grads = [], []
                scales = xyz.new_tensor([[sigma, sigma, .2]])
                if name == 'diagonal_thin':
                    scales[0, 0] = .4
                for t in centers:
                    offset = xyz.new_tensor(t).requires_grad_()
                    means = (anchor + direction * offset)[None]
                    p, g = geometry_terms(loss, means, xyz, occupied)
                    ce = F.cross_entropy(rendered(xyz, means, scales, xyz.new_ones(1),
                                                  xyz.new_tensor([[amplitude]]), angle), labels)
                    geometry = p + g
                    dg = torch.autograd.grad(geometry, offset, retain_graph=True)[0]
                    dv = torch.autograd.grad(ce, offset)[0]
                    values.append([float(p.detach()), float(g.detach()), float(ce.detach()), float((geometry + 10 * ce).detach())])
                    grads.append([float(dg), float(dv)])
                arr, grad = np.array(values), np.array(grads)
                idx = int(arr[:, 2].argmin())
                row = dict(scene=name, scale=sigma, amplitude=amplitude,
                           field_min_offset=float(centers[idx]),
                           field_min_at_scan_edge=idx in (0, len(centers) - 1),
                           geometry_min_offset=float(centers[(arr[:, 0] + arr[:, 1]).argmin()]),
                           joint_min_offset=float(centers[arr[:, 3].argmin()]),
                           field_ce_at_field_min=float(arr[idx, 2]),
                           field_gradient_at_field_min=float(grad[idx, 1]),
                           field_grid_curvature=(float((arr[idx + 1, 2] - 2 * arr[idx, 2] + arr[idx - 1, 2]) /
                                                       (centers[1] - centers[0])**2)
                                                 if 0 < idx < len(centers) - 1 else None),
                           geometry_gradient_at_field_min=float(grad[idx, 0]),
                           interventions=[])
                for displacement in (.002, .01, .04):
                    # Re-evaluate both objectives after a bounded geometry-only step.
                    t0 = float(centers[idx])
                    t1 = t0 - displacement * np.sign(grad[idx, 0])
                    mu0, mu1 = (anchor + direction * t0)[None], (anchor + direction * t1)[None]
                    gp0, gg0 = geometry_terms(loss, mu0, xyz, occupied)
                    gp1, gg1 = geometry_terms(loss, mu1, xyz, occupied)
                    ce1 = F.cross_entropy(rendered(xyz, mu1, scales, xyz.new_ones(1),
                                                   xyz.new_tensor([[amplitude]]), angle), labels)
                    row['interventions'].append(dict(displacement=displacement,
                        delta_geometry=float(gp1 + gg1 - gp0 - gg0),
                        delta_ce=float(ce1) - row['field_ce_at_field_min']))
                row['local_harm_observed'] = any(
                    s['delta_geometry'] < -1e-8 and s['delta_ce'] > 1e-6
                    for s in row['interventions']) and not row['field_min_at_scan_edge']
                records.append(row)
                np.savez_compressed(out / f'landscape_{name}_s{sigma}_a{amplitude}.npz',
                                    offset=centers, losses=arr, gradients=grad)
                if amplitude == 8.:
                    axes[0, si].plot(centers, arr[:, 0], label='p2g')
                    axes[0, si].plot(centers, arr[:, 1], label='g2p')
                    axes[0, si].plot(centers, arr[:, 2] * 10, label='10 * voxel CE')
                    axes[0, si].plot(centers, arr[:, 3], label='joint')
                    axes[0, si].axvline(centers[idx], color='gray', linestyle='--')
                    axes[0, si].set_title(f'{name}: scale={sigma}, amplitude=8')
                    axes[1, si].plot(centers, grad[:, 0], label='geometry gradient')
                    axes[1, si].plot(centers, grad[:, 1] * 10, label='10 * CE gradient')
                    axes[1, si].axhline(0, color='gray', linewidth=.5)
                    axes[1, si].set_xlabel('center offset (m)')
            axes[0, si].legend(fontsize=7)
            axes[1, si].legend(fontsize=7)
        figure.savefig(out / f'geometry_{name}.png', dpi=160)
        plt.close(figure)
        print(f'geometry scans: {name}', flush=True)
        # Multiple starts and bounded attribute freedom prevent a single-picked example.
        for free_attributes in (False, True):
            for initial in (-.16, .013, .16):
                for objective in ('geometry', 'voxel', 'joint'):
                    t = torch.nn.Parameter(xyz.new_tensor(initial))
                    raw_scale = torch.nn.Parameter(xyz.new_zeros(()), requires_grad=free_attributes)
                    raw_opacity = torch.nn.Parameter(xyz.new_tensor(2.), requires_grad=free_attributes)
                    parameters = [t] + ([raw_scale, raw_opacity] if free_attributes else [])
                    optimizer = torch.optim.Adam(parameters, lr=.01)
                    path = []
                    for iteration in range(steps):
                        mu = (anchor + direction * t)[None]
                        sigma = .08 + .32 * raw_scale.sigmoid() if free_attributes else xyz.new_tensor(.24)
                        scales = torch.stack([sigma, sigma, xyz.new_tensor(.2)])[None]
                        opacity = raw_opacity.sigmoid()[None] if free_attributes else xyz.new_ones(1)
                        p, g = geometry_terms(loss, mu, xyz, occupied)
                        logits = rendered(xyz, mu, scales, opacity, xyz.new_tensor([[8.]]), angle)
                        ce = F.cross_entropy(logits, labels)
                        value = {'geometry': p + g, 'voxel': 10 * ce, 'joint': p + g + 10 * ce}[objective]
                        optimizer.zero_grad()
                        value.backward()
                        optimizer.step()
                        if iteration % 10 == 0:
                            path.append([iteration, scalar(t), scalar(sigma), scalar(opacity[0]), scalar(ce)])
                    with torch.no_grad():
                        mu = (anchor + direction * t)[None]
                        sigma = .08 + .32 * raw_scale.sigmoid() if free_attributes else xyz.new_tensor(.24)
                        scales = torch.stack([sigma, sigma, xyz.new_tensor(.2)])[None]
                        opacity = raw_opacity.sigmoid()[None] if free_attributes else xyz.new_ones(1)
                        logits = rendered(xyz, mu, scales, opacity, xyz.new_tensor([[8.]]), angle)
                        metrics = field_metrics(logits, labels, 1)
                    trajectories.append(dict(scene=name, free_attributes=free_attributes,
                        initial_offset=initial, objective=objective, final_offset=scalar(t),
                        final_scale=float(sigma), final_opacity=float(opacity[0]), metrics=metrics,
                        path_columns=['iteration', 'offset', 'scale', 'opacity', 'ce'], path=path))
    return records, trajectories


def two_dimensional_scan(out):
    xyz, occupied, _ = scene('diagonal_thin')
    labels, loss = (~occupied).long(), set_loss()
    coordinates = np.linspace(-.4, .4, 41)
    values = np.zeros((41, 41, 3))
    gradients = np.zeros((41, 41, 2, 2))
    scales = xyz.new_tensor([[.4, .16, .2]])
    for i, x in enumerate(coordinates):
        for j, y in enumerate(coordinates):
            mu = xyz.new_tensor([[x, y, 0.]]).requires_grad_()
            p, g = geometry_terms(loss, mu, xyz, occupied)
            ce = F.cross_entropy(rendered(xyz, mu, scales, xyz.new_ones(1),
                                          xyz.new_tensor([[8.]]), -np.pi / 4), labels)
            dg = torch.autograd.grad(p + g, mu, retain_graph=True)[0][0, :2]
            dv = torch.autograd.grad(10 * ce, mu)[0][0, :2]
            values[i, j] = [float((p + g).detach()), float(10 * ce.detach()), float((p + g + 10 * ce).detach())]
            gradients[i, j] = np.stack([dg.numpy(), dv.numpy()])
    angles = np.linspace(-np.pi / 2, np.pi / 2, 121)
    rotation_losses, rotation_gradients = [], []
    for sigma in (.08, .16, .24, .4):
        losses, derivs = [], []
        for angle in angles:
            # Central finite difference includes the renderer's piecewise binning.
            s = xyz.new_tensor([[.4, sigma, .2]])
            def objective(theta):
                return float(F.cross_entropy(rendered(xyz, xyz.new_zeros(1, 3), s,
                    xyz.new_ones(1), xyz.new_tensor([[8.]]), theta), labels))
            losses.append(objective(angle))
            derivs.append((objective(angle + 1e-4) - objective(angle - 1e-4)) / 2e-4)
        rotation_losses.append(losses)
        rotation_gradients.append(derivs)
    np.savez_compressed(out / 'diagonal_2d_center_rotation.npz', coordinates=coordinates,
        center_losses=values, center_gradients=gradients, angle=angles,
        rotation_ce=rotation_losses, rotation_ce_gradient=rotation_gradients,
        scales=[.08, .16, .24, .4])
    figure, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing='ij')
    for i, title in enumerate(('center geometry', '10 * voxel CE', 'joint')):
        im = axes[i].pcolormesh(xx, yy, values[:, :, i], shading='auto')
        figure.colorbar(im, ax=axes[i])
        axes[i].set(title=title, xlabel='center x (m)', ylabel='center y (m)')
        if i < 2:
            axes[i].quiver(xx[::5, ::5], yy[::5, ::5],
                -gradients[::5, ::5, i, 0], -gradients[::5, ::5, i, 1], color='white')
    for sigma, losses in zip((.08, .16, .24, .4), rotation_losses):
        axes[3].plot(angles * 180 / np.pi, losses, label=f'scale={sigma}')
    axes[3].set(title='rotation: fixed center', xlabel='angle (degrees)', ylabel='voxel CE')
    axes[3].legend(fontsize=7)
    figure.savefig(out / 'diagonal_2d_center_rotation.png', dpi=160)
    plt.close(figure)
    print('2D center and rotation scan complete', flush=True)


def semantic_experiment(out, steps):
    records = []
    xyz, occupied, _ = scene('isolated')
    x = xyz[:, 0]
    occupied = x.abs() <= .81
    for name in ('minority_nearest', 'separated', 'overlap_specialization'):
        labels = torch.full((len(x),), 2, dtype=torch.long)
        labels[occupied] = 0
        if name == 'minority_nearest':
            labels[(x - .4).abs() < .01] = 1
            means = xyz.new_tensor([[.36, 0., 0.]])
            scales = xyz.new_tensor([[.4, .2, .2]])
        else:
            labels[occupied & (x >= -.01)] = 1
            means = xyz.new_tensor([[-.45, 0., 0.], [.45, 0., 0.]])
            scales = xyz.new_tensor([[.24, .2, .2], [.24, .2, .2]])
            if name == 'overlap_specialization':
                means = xyz.new_tensor([[-.12, 0., 0.], [.12, 0., 0.]])
                scales[:, 0] = .4
        opacity = xyz.new_ones(len(means))
        nearest = labels[occupied][torch.cdist(means, xyz[occupied]).argmin(-1)]
        initial = xyz.new_zeros(len(means), 2)
        if len(means) == 2:
            initial = xyz.new_tensor([[3., -3.], [-3., 3.]])
        diag, target_info = contribution_diagnostic(xyz, labels, means, opacity, scales,
            inverse_cov(scales), F.softplus(initial), nearest, 2, LOWER)
        targets = dict(center_distribution=F.one_hot(nearest, 2).to(xyz),
                       support_majority=F.one_hot(target_info['target'].argmax(-1), 2).to(xyz),
                       support_soft=target_info['target'],
                       center_smoothing=.9 * F.one_hot(nearest, 2).to(xyz) + .05)
        neighborhood = (torch.cdist(means, xyz[occupied]) <= .8).to(xyz)
        targets['ordinary_neighborhood'] = (neighborhood @ F.one_hot(labels[occupied], 2).to(xyz)) / neighborhood.sum(-1, keepdim=True).clamp_min(1)
        for objective in ('center_focal', *targets, 'voxel'):
            z = torch.nn.Parameter(initial.clone())
            optimizer = torch.optim.Adam([z], lr=.03)
            path = []
            for iteration in range(steps):
                logits = rendered(xyz, means, scales, opacity, F.softplus(z))
                ce = F.cross_entropy(logits, labels)
                if objective == 'center_focal':
                    value = 2 * set_loss()._sigmoid_focal(z, nearest, xyz.new_ones(len(means)))
                elif objective == 'voxel':
                    value = 10 * ce
                else:
                    composition = F.softplus(z) / F.softplus(z).sum(-1, keepdim=True)
                    value = 2 * -(targets[objective] * composition.log()).sum(-1).mean()
                optimizer.zero_grad()
                value.backward()
                optimizer.step()
                if iteration % 10 == 0:
                    path.append([iteration, scalar(value), scalar(ce)])
            with torch.no_grad():
                logits = rendered(xyz, means, scales, opacity, F.softplus(z))
            records.append(dict(scene=name, objective=objective, diagnostic=diag,
                nearest_labels=nearest.tolist(), support_target=target_info['target'].tolist(),
                initial_metrics=field_metrics(rendered(xyz, means, scales, opacity, F.softplus(initial)), labels, 2),
                final_metrics=field_metrics(logits, labels, 2), final_amplitude=F.softplus(z).detach().tolist(), path=path))
        figure, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for row in records:
            if row['scene'] == name:
                a = np.array(row['path'])
                ax.plot(a[:, 0], a[:, 2], label=row['objective'])
        ax.set(xlabel='optimization steps (fixed geometry)', ylabel='voxel CE', title=name)
        ax.legend(fontsize=8)
        figure.savefig(out / f'semantics_{name}.png', dpi=160)
        plt.close(figure)
        print(f'semantic fits: {name}', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='work_dirs/center_support_contribution1/toy')
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(928)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scans, paths = geometry_scan(out, args.steps)
    two_dimensional_scan(out)
    semantic = semantic_experiment(out, args.steps)
    write_json(out / 'results.json', dict(
        protocol=dict(seed=928, dtype='float64', grid_size=.4, steps=args.steps,
            geometry='inherited OPUS bidirectional Smooth-L1; actual detached assignment/weights',
            renderer='localagg box convention and additive Gaussian kernel; constant empty=1.5',
            limits='1/2 foreground classes, unweighted toy CE; not full Occ3D or a CUDA parity test'),
        geometry_scans=scans, geometry_trajectories=paths, semantic_fits=semantic))
    print(out / 'results.json')


if __name__ == '__main__':
    main()
