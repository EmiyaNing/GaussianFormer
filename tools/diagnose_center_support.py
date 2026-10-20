"""Real-checkpoint contribution-1 diagnostics and reversible local interventions.

No checkpoint is written. Uses deterministic validation preprocessing, full GT
for losses, and uniformly sampled voxels only for contribution attribution.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools.center_support_common import (contribution_diagnostic, gradient_summary,
                                         kernel_weights, write_json)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=928)
    parser.add_argument('--samples-per-domain', type=int, default=1024)
    parser.add_argument('--minimum-effective-voxels', type=float, default=3.)
    parser.add_argument('--step-radii', type=float, nargs='+', default=[1e-5, 1e-4, 1e-3])
    parser.add_argument('--intervene', action='store_true')
    parser.add_argument('--contributions-only', action='store_true',
                        help='Forward and renderer attribution only, for sampling sensitivity')
    parser.add_argument('--full-occupied', action='store_true',
                        help='Accumulate exact responsibility targets over every visible occupied voxel; empty remains sampled')
    parser.add_argument('--scene-pairs', help='JSON list of fixed [source, heldout] scene pairs')
    parser.add_argument('--resume', action='store_true', help='Resume completed scene pairs in this exact protocol')
    args = parser.parse_args()
    if args.contributions_only and args.intervene:
        parser.error('--contributions-only cannot be combined with --intervene')
    if (min(args.pairs, args.samples_per_domain, args.minimum_effective_voxels) <= 0 or
            any(r <= 0 for r in args.step_radii)):
        parser.error('Counts and step radii must be positive')
    return args


def cpu_number(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else value


def make_loss_inputs(output, metas, config):
    inputs = {'metas': metas}
    inputs.update({name: output[key] for name, key in config.loss_input_convertion.items()})
    for name in ('opus_pred_points', 'opus_pred_logits', 'opus_point_valid_masks'):
        inputs[name] = inputs[name][-1:]
    # This experiment measures final-stage loss only.
    inputs['pred_occ'] = output['pred_occ'][-1:]
    return inputs


def set_objective_configs(config):
    candidates = [x for x in config.loss.loss_cfgs
                  if x['type'] in ('OPUSSetLoss', 'CenterSupportDiagnosticSetLoss')]
    if len(candidates) != 1:
        raise ValueError('Diagnostic requires exactly one OPUS-compatible set loss')
    set_config = copy.deepcopy(candidates[0])
    set_config['type'] = 'CenterSupportDiagnosticSetLoss'
    set_config['stage_weights'] = set_config['stage_weights'][-1:]
    # A single retained stage is also the final stage. Override both selectors
    # so inherited mini-training alignment cannot collapse the all/camera control.
    return {domain: dict(set_config, gt_domain=domain, final_gt_domain=domain)
            for domain in ('all', 'camera')}


def build_objectives(config):
    from loss import OPENOCC_LOSS
    sets = {domain: OPENOCC_LOSS.build(settings).cuda()
            for domain, settings in set_objective_configs(config).items()}
    voxel_config = next(copy.deepcopy(x) for x in config.loss.loss_cfgs if x['type'] == 'OccupancyLoss')
    if voxel_config.get('use_focal_loss', False) or voxel_config.get('use_dice_loss', False) or voxel_config.get('use_sem_geo_scal_loss', True):
        raise ValueError('This diagnostic supports the CE+Lovasz reference configuration')
    ce_cfg, lovasz_cfg = copy.deepcopy(voxel_config), copy.deepcopy(voxel_config)
    ce_cfg['use_lovasz_loss'] = False
    lovasz_cfg['multi_loss_weights']['loss_voxel_ce_weight'] = 0.
    return sets, OPENOCC_LOSS.build(ce_cfg).cuda(), OPENOCC_LOSS.build(lovasz_cfg).cuda()


def losses(output, batch, config, modules):
    inputs = make_loss_inputs(output, batch, config)
    # Explicit camera mask keeps the two set-domain comparisons anchored.
    camera = batch['occ_cam_mask'].bool()
    valid = (batch['occ_label'] >= 0) & (batch['occ_label'] < config.model.head.num_classes)
    inputs['occ_mask'] = camera & valid
    sets, ce_module, lovasz_module = modules
    objectives, counts = {}, {}
    for domain, module in sets.items():
        terms, stats = module.components(inputs)
        objectives.update({domain + '/' + name.split('/')[-1]: value for name, value in terms.items()})
        counts[domain] = {key: cpu_number(value) for key, value in stats.items()}
    objectives['voxel_ce'], objectives['voxel_lovasz'] = ce_module(inputs), lovasz_module(inputs)
    return objectives, counts


def field_scores(output, batch, empty_label):
    labels = output['sampled_label'].long()
    mask = batch['occ_cam_mask'].bool().flatten(1) & (labels >= 0) & (labels <= empty_label)
    logits = output['pred_occ'][-1].transpose(1, 2)[mask]
    target = labels[mask]
    pred = logits.argmax(-1)
    tp = int(((pred != empty_label) & (target != empty_label)).sum())
    union = int(((pred != empty_label) | (target != empty_label)).sum())
    classes = []
    for k in range(empty_label):
        intersection = int(((pred == k) & (target == k)).sum())
        total = int(((pred == k) | (target == k)).sum())
        classes.append(dict(intersection=intersection, union=total))
    return dict(error=float((pred != target).float().mean()),
                binary_intersection=tp, binary_union=union, per_class=classes)


def parameter_groups(network):
    s = len(network.encoder.layers) - 1
    prefix = f'encoder.layers.{s}.'
    names, parameters, groups = [], [], {}
    for name, p in network.named_parameters():
        if not p.requires_grad:
            continue
        group = None
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            group = ('classification_parameters' if suffix.startswith('cls_branch.') else
                     'geometry_parameters' if suffix.startswith('reg_branch.') else 'shared_parameters')
        elif name.startswith(f'encoder.attribute_branches.{s}.'):
            group = 'attribute_parameters'
        elif name.startswith('head.'):
            group = 'empty_parameters'
        if group:
            groups.setdefault(group, []).append(len(parameters))
            names.append(name)
            parameters.append(p)
    return names, parameters, groups


def combine_gradients(grads, coefficients):
    size = len(next(iter(grads.values())))
    return [sum(grads[name][i] * weight for name, weight in coefficients.items()) for i in range(size)]


def combine_optional_gradients(grads, names):
    combined = []
    for i in range(len(next(iter(grads.values())))):
        available = [grads[name][i] for name in names if grads[name][i] is not None]
        combined.append(sum(available) if available else None)
    return combined


@torch.no_grad()
def contributions(network, output, batch, args, pair):
    from loss.opus_set_loss import _nearest_indices
    head, gaussian = network.head, output['gaussian']
    all_means, all_opacity, all_semantics, all_scales, all_inv = head.prepare_gaussian_args(gaussian)
    means, opacity, semantics, scales, inv = [t[0, :-1].detach() for t in
        (all_means, all_opacity, all_semantics, all_scales, all_inv)]
    xyz, y = output['sampled_xyz'][0], output['sampled_label'][0].long()
    camera = batch['occ_cam_mask'][0].bool().flatten()
    valid = camera & (y >= 0) & (y <= head.empty_label)
    rng = np.random.default_rng(args.seed + pair)
    sampled = []
    strata = {}
    for name, m in (('occupied', valid & (y < head.empty_label)), ('empty', valid & (y == head.empty_label))):
        ids = torch.nonzero(m).flatten().cpu().numpy()
        chosen = (ids if name == 'occupied' and args.full_occupied else
                  np.sort(rng.choice(ids, min(len(ids), args.samples_per_domain), replace=False)))
        sampled.extend(chosen.tolist())
        strata[name] = dict(population=len(ids), sampled=len(chosen))
    ids = torch.tensor(sorted(sampled), device=xyz.device, dtype=torch.long)
    if not (valid & (y < head.empty_label)).any():
        return {'status': 'no visible occupied GT'}, {}
    nearest = y[valid & (y < head.empty_label)][_nearest_indices(
        means, xyz[valid & (y < head.empty_label)], 1024)]
    aggregator = head.aggregator
    lower = aggregator.pc_min.flatten().tolist()
    report, info = contribution_diagnostic(xyz[ids], y[ids], means, opacity, scales, inv,
        semantics, nearest, head.empty_label, lower, aggregator.grid_size, aggregator.scale_multiplier,
        minimum_effective_voxels=args.minimum_effective_voxels)
    # Compare all 18 channels, including actual empty Gaussian and learned bias.
    reference = xyz.new_zeros(len(ids), head.num_classes)
    for v in range(0, len(ids), 32):
        for g in range(0, all_means.shape[1], 2048):
            w = kernel_weights(xyz[ids[v:v + 32]], all_means[0, g:g + 2048],
                all_opacity[0, g:g + 2048], all_scales[0, g:g + 2048], all_inv[0, g:g + 2048],
                lower, aggregator.grid_size, aggregator.scale_multiplier)
            reference[v:v + 32] += w @ all_semantics[0, g:g + 2048]
    if hasattr(head, 'empty_bias'):
        reference[:, head.empty_label] -= head.empty_bias
    actual = output['pred_occ'][-1][0, :, ids].T
    maximum = float((actual - reference).abs().max())
    parity = torch.allclose(actual, reference, atol=2e-4, rtol=2e-4)
    if not parity:
        raise RuntimeError(f'Contribution reference failed CUDA parity: max error {maximum}')
    # Relationship with render errors, using sampled responsibility support.
    pred = actual.argmax(-1)
    bad_voxels = (pred != y[ids]).to(means)
    supported_error = means.new_zeros(len(means))
    for v in range(0, len(ids), 32):
        x = xyz[ids[v:v + 32]]
        total = means.new_zeros(len(x))
        for g in range(0, len(means), 2048):
            total += kernel_weights(x, means[g:g + 2048], opacity[g:g + 2048], scales[g:g + 2048],
                                    inv[g:g + 2048], lower, aggregator.grid_size, aggregator.scale_multiplier).sum(-1)
        occ = y[ids[v:v + 32]] < head.empty_label
        for g in range(0, len(means), 2048):
            w = kernel_weights(x, means[g:g + 2048], opacity[g:g + 2048], scales[g:g + 2048],
                               inv[g:g + 2048], lower, aggregator.grid_size, aggregator.scale_multiplier)
            supported_error[g:g + 2048] += (w / total[:, None].clamp_min(1e-12)).T @ (bad_voxels[v:v + 32] * occ)
    for name, mask in (('mismatch', info['supported'] & info['mismatch']),
                       ('agreement', info['supported'] & ~info['mismatch'])):
        report[name + '_responsibility_weighted_error'] = (
            float(supported_error[mask].sum() / info['mass'][mask].sum()) if mask.any() else None)
    report.update(sampling=strata, renderer_max_abs_error=maximum, renderer_parity_passed=bool(parity))
    entropy = -(info['target'] * info['target'].clamp_min(1e-12).log()).sum(-1)
    distance = means[:, :2].norm(dim=-1)
    primitive_groups = {
        'supported_mismatch': info['supported'] & info['mismatch'],
        'supported_agreement': info['supported'] & ~info['mismatch'],
        'mixed_label_proxy': info['supported'] & (entropy > .3),
        'low_entropy_support': info['supported'] & (entropy <= .3),
        'weak_or_no_support': ~info['supported'],
        'thin_class_proxy': torch.isin(nearest, nearest.new_tensor([2, 6, 7, 8])),
        'distance_0_10m': distance < 10,
        'distance_10_30m': (distance >= 10) & (distance < 30),
        'distance_30m_plus': distance >= 30,
    }
    artifact = Path(args.out).with_name(Path(args.out).stem + f'_pair{pair}_primitives.npz')
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact, means=means.cpu().numpy(), scales=scales.cpu().numpy(),
        opacity=opacity.cpu().numpy(), nearest_labels=nearest.cpu().numpy(),
        sampled_responsibility_mass=info['mass'].cpu().numpy(), support_target=info['target'].cpu().numpy(),
        effective_sampled_voxels=info['effective_voxels'].cpu().numpy(),
        supported_error_mass=supported_error.cpu().numpy(), sampled_voxel_indices=ids.cpu().numpy())
    report['primitive_artifact'] = str(artifact)
    report['scale_ge_0p396_fraction'] = float((scales >= .396).float().mean())
    return report, {k: v.cpu() for k, v in primitive_groups.items()}


def main():
    args = arguments()
    if not torch.cuda.is_available():
        write_json(args.out, dict(status='blocked_no_cuda', checkpoint=args.checkpoint,
            py_config=args.py_config, reason='torch.cuda.is_available() is false; no real-model results'))
        raise SystemExit('CUDA unavailable; wrote blocked status without fabricated measurements.')
    from mmengine import Config
    from mmseg.models import build_segmentor
    import model  # noqa: F401
    from dataset import get_dataloader
    from dataset.utils import custom_collate_fn_temporal
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(args.py_config)
    if cfg.model.head.get('empty_logit_offset', 0):
        raise ValueError('Post-hoc empty offsets are excluded from this protocol')
    cfg.val_loader.batch_size, cfg.val_loader.num_workers = 1, 0
    _, loader = get_dataloader(cfg.train_dataset_config, cfg.val_dataset_config,
                               cfg.train_loader, cfg.val_loader, val_only=True, seed=args.seed)
    dataset = loader.dataset
    # Pick first keyframe from evenly spaced scenes, with distinct holdout scenes.
    by_scene = {}
    for index, (scene, frame) in enumerate(dataset.keyframes):
        by_scene.setdefault(scene, index)
    scenes = sorted(by_scene)
    if len(scenes) < 2 * args.pairs:
        raise ValueError(f'Need {2 * args.pairs} distinct scenes; available {len(scenes)}')
    if args.scene_pairs:
        scene_pairs = json.loads(Path(args.scene_pairs).read_text())[:args.pairs]
        flat = [scene for pair in scene_pairs for scene in pair]
        if (len(scene_pairs) != args.pairs or any(len(pair) != 2 for pair in scene_pairs)
                or len(set(flat)) != len(flat) or not set(flat).issubset(by_scene)):
            raise ValueError('Invalid fixed scene-pair manifest')
    else:
        selected = np.linspace(0, len(scenes) - 1, 2 * args.pairs, dtype=int)
        scene_pairs = [(scenes[selected[i]], scenes[selected[i + 1]]) for i in range(0, len(selected), 2)]
    network = build_segmentor(cfg.model).cuda().eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    network.load_state_dict(state, strict=True)
    del state
    modules = build_objectives(cfg)
    parameter_names, parameters, groups = parameter_groups(network)
    baseline = [p.detach().clone() for p in parameters]
    scale = sum(float(p.detach().double().square().sum()) for p in parameters)**.5
    result = dict(status='running', protocol=vars(args), parameter_names=parameter_names,
        parameter_scope='final decoder layer, final attribute branch, empty head; eval mode; normalized SGD',
        pairs=[])
    if args.resume and Path(args.out).exists():
        previous = json.loads(Path(args.out).read_text())
        protocol = {k: v for k, v in vars(args).items() if k != 'resume'}
        saved = {k: v for k, v in previous.get('protocol', {}).items() if k != 'resume'}
        saved.setdefault('full_occupied', False)
        if saved != protocol:
            raise ValueError('Refusing to resume a different diagnostic protocol')
        if previous.get('status') == 'complete':
            print('Already complete:', args.out)
            return
        result['pairs'] = previous.get('pairs', [])
    write_json(args.out, result)
    for pair_index, (source_scene, heldout_scene) in enumerate(scene_pairs):
        if pair_index < len(result['pairs']):
            saved_pair = result['pairs'][pair_index]
            if (saved_pair['source_scene'], saved_pair['heldout_scene']) != (source_scene, heldout_scene):
                raise ValueError('Saved scene pairs differ from manifest')
            continue
        pair_started = time.monotonic()
        def load_batch(scene):
            batch = custom_collate_fn_temporal([dataset[by_scene[scene]]])
            return {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        batch, holdout = load_batch(source_scene), load_batch(heldout_scene)
        def forward(data):
            return network(imgs=data['img'], metas={k: v for k, v in data.items() if k != 'img'})
        with torch.set_grad_enabled(not args.contributions_only):
            output = forward(batch)
        if args.contributions_only:
            contribution, _ = contributions(network, output, batch, args, pair_index)
            result['pairs'].append(dict(source_scene=source_scene, heldout_scene=heldout_scene,
                contribution=contribution, gradients={}, interventions=[],
                elapsed_seconds=time.monotonic() - pair_started))
            del output, batch, holdout
            write_json(args.out, result)
            print(f'completed attribution pair {pair_index + 1}/{args.pairs}', flush=True)
            continue
        objectives, counts = losses(output, batch, cfg, modules)
        contribution, primitive_groups = contributions(network, output, batch, args, pair_index)
        row = dict(source_scene=source_scene, heldout_scene=heldout_scene,
                   losses={k: cpu_number(v) for k, v in objectives.items()}, counts=counts,
                   contribution=contribution, gradients={})
        stage = output['representation'][-1]
        tensors = [stage['gaussian'].means, stage['opus_logits'], stage['query_features']]
        tensor_names = ['center_output', 'raw_classification_output', 'shared_query_features']
        gradients, output_gradients, parameter_missing = {}, {}, {}
        for name, value in objectives.items():
            grad = torch.autograd.grad(value, tensors + parameters, allow_unused=True, retain_graph=True)
            output_gradients[name] = [g.detach().cpu() if g is not None else None for g in grad[:3]]
            gradients[name] = [g.detach().cpu() if g is not None else torch.zeros_like(p, device='cpu')
                               for g, p in zip(grad[3:], parameters)]
            parameter_missing[name] = [g is None for g in grad[3:]]
        gradients['voxel'] = combine_gradients(gradients, {'voxel_ce': 1, 'voxel_lovasz': 1})
        parameter_missing['voxel'] = [all(parameter_missing[k][i] for k in ('voxel_ce', 'voxel_lovasz'))
                                      for i in range(len(parameters))]
        output_gradients['voxel'] = combine_optional_gradients(output_gradients, ['voxel_ce', 'voxel_lovasz'])
        for domain in ('all', 'camera'):
            keys = [domain + '/pred_to_gt', domain + '/gt_to_pred']
            gradients[domain + '/geometry'] = combine_gradients(gradients, dict.fromkeys(keys, 1))
            parameter_missing[domain + '/geometry'] = [all(parameter_missing[k][i] for k in keys)
                                                       for i in range(len(parameters))]
            output_gradients[domain + '/geometry'] = combine_optional_gradients(output_gradients, keys)
        for a, b in itertools.combinations(output_gradients, 2):
            comparison = {}
            for i, name in enumerate(tensor_names):
                comparison[name] = gradient_summary(output_gradients[a][i], output_gradients[b][i])
            for name, indices in groups.items():
                ga = torch.cat([gradients[a][i].flatten() for i in indices])
                gb = torch.cat([gradients[b][i].flatten() for i in indices])
                comparison[name] = gradient_summary(
                    None if all(parameter_missing[a][i] for i in indices) else ga,
                    None if all(parameter_missing[b][i] for i in indices) else gb)
                comparison[name]['missing_parameter_count_a'] = sum(parameter_missing[a][i] for i in indices)
                comparison[name]['missing_parameter_count_b'] = sum(parameter_missing[b][i] for i in indices)
            row['gradients'][a + ' vs ' + b] = comparison
            if ((a.endswith(('geometry', 'classification')) and b == 'voxel') or
                    (b.endswith(('geometry', 'classification')) and a == 'voxel')):
                comparison['primitive_strata'] = {}
                for stratum, mask in primitive_groups.items():
                    summary = dict(primitive_count=int(mask.sum()))
                    for i, name in enumerate(tensor_names):
                        if i == 2:
                            children = stage['opus_logits'].shape[2]
                            select = mask.reshape(-1, children).any(-1)
                            shape = (-1, stage['query_features'].shape[-1])
                        else:
                            select = mask
                            shape = (-1, 3 if i == 0 else stage['opus_logits'].shape[-1])
                        ga, gb = output_gradients[a][i], output_gradients[b][i]
                        summary[name] = gradient_summary(
                            ga.reshape(shape)[select] if ga is not None else None,
                            gb.reshape(shape)[select] if gb is not None else None)
                    comparison['primitive_strata'][stratum] = summary
        with torch.no_grad():
            original_field = field_scores(output, batch, network.head.empty_label)
            holdout_output = forward(holdout)
            holdout_losses, _ = losses(holdout_output, holdout, cfg, modules)
            heldout_values = {k: cpu_number(v) for k, v in holdout_losses.items()}
            heldout_field = field_scores(holdout_output, holdout, network.head.empty_label)
        del output, objectives, holdout_output, holdout_losses, stage, tensors, value, grad
        row['field_before'], row['heldout_field_before'] = original_field, heldout_field
        row['heldout_losses_before'] = heldout_values
        row['interventions'] = []
        if args.intervene:
            branches = {}
            for domain in ('all', 'camera'):
                p, g, c = (domain + '/' + k for k in ('pred_to_gt', 'gt_to_pred', 'classification'))
                branches.update({domain + '/geometry_only': {p: 1, g: 1},
                    domain + '/classification_only': {c: 1},
                    domain + '/joint': {p: 1, g: 1, c: 1, 'voxel': 1},
                    domain + '/geometry_quarter': {p: .25, g: .25, c: 1, 'voxel': 1},
                    domain + '/classification_quarter': {p: 1, g: 1, c: .25, 'voxel': 1},
                    domain + '/no_geometry': {c: 1, 'voxel': 1},
                    domain + '/no_classification': {p: 1, g: 1, 'voxel': 1}})
            branches['voxel_only'] = {'voxel': 1}
            with torch.no_grad():
                for branch, coefficients in branches.items():
                    direction = combine_gradients(gradients, coefficients)
                    norm = sum(float(g.double().square().sum()) for g in direction)**.5
                    for radius in args.step_radii:
                        alpha = radius * scale / norm if norm > 1e-12 else 0.
                        try:
                            for p, original, d in zip(parameters, baseline, direction):
                                p.copy_(original - alpha * d.to(p))
                            actual_step = sum(float((p.detach() - original).double().square().sum())
                                              for p, original in zip(parameters, baseline))**.5
                            updated, independent = forward(batch), forward(holdout)
                            updated_losses, _ = losses(updated, batch, cfg, modules)
                            independent_losses, _ = losses(independent, holdout, cfg, modules)
                            row['interventions'].append(dict(branch=branch, radius=radius,
                                gradient_norm=norm, effective_sgd_lr=alpha,
                                parameter_step_l2=actual_step,
                                requested_parameter_step_l2=radius * scale if alpha else 0.,
                                delta_losses={k: cpu_number(v) - row['losses'][k] for k, v in updated_losses.items()},
                                delta_heldout_losses={k: cpu_number(v) - heldout_values[k] for k, v in independent_losses.items()},
                                field_after=field_scores(updated, batch, network.head.empty_label),
                                heldout_field_after=field_scores(independent, holdout, network.head.empty_label)))
                            del updated, independent, updated_losses, independent_losses
                        finally:
                            for p, original in zip(parameters, baseline):
                                p.copy_(original)
        row['elapsed_seconds'] = time.monotonic() - pair_started
        result['pairs'].append(row)
        write_json(args.out, result)
        print(f'completed pair {pair_index + 1}/{args.pairs}: {source_scene} / {heldout_scene}', flush=True)
    digest = hashlib.sha256()
    with open(args.checkpoint, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    result['checkpoint_sha256'] = digest.hexdigest()
    result['status'] = 'complete'
    result['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    write_json(args.out, result)


if __name__ == '__main__':
    main()
