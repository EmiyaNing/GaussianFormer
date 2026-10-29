"""GT-guided, bounded center-only refinement of frozen tiny/small predictions.

This is a per-frame diagnostic using the evaluation frame's own GT. It is not
ordinary inference, a trained checkpoint, or a deployment performance result.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import fcntl
import gc
import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mmengine import Config
from tools.stat_boundary_gaussians import read, write, sha, forward, to_cuda
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.eval_gaussian_sizes import paths, build, CLASS_NAMES, CLASS_COLUMNS
from tools.probe_gaussian_scale_supervision import layout
from tools.semantic_field_support import extension

RUN = ROOT / 'work_dirs/gaussian_center_oracle_full_val_v1'
HISTORY = ROOT / 'work_dirs/gaussian_sizes_full_val'
MODELS = ('tiny', 'small')
STEPS = (0, 5, 10, 20)
RADIUS = .4
LR = .02
TRANSITIONS = ('rescued_FN', 'new_FN', 'removed_binary_FP', 'new_binary_FP',
               'corrected_semantic_prediction', 'damaged_semantic_prediction',
               'corrected_foreground_class', 'damaged_foreground_class')
MOVEMENT = ('count', 'sum_m', 'sum_squared_m', 'max_m', 'moving_count', 'at_radius_count')


def initialize(size):
    config, checkpoint = paths(size)
    cfg = Config.fromfile(config)
    previous = read(HISTORY / size / 'protocol.json')
    assert sha(checkpoint) == previous['checkpoint_sha256']
    for name, digest in previous['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    files = list(dict.fromkeys([Path(__file__), ROOT / 'loss/occupancy_loss.py',
        ROOT / 'loss/utils/lovasz_softmax.py',
        ROOT / 'model/head/localagg/local_aggregate/__init__.py',
        ROOT / 'model/head/localagg/src/backward.cu',
        ROOT / 'tools/eval_gaussian_sizes.py',
        *[ROOT / p for p in previous['source_sha256']]]))
    files = [p for p in files if p.is_file()]
    protocol = dict(version=1, model=size, checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=sha(checkpoint), config=str(config.relative_to(ROOT)),
        resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        annotation_sha256=sha(cfg.val_dataset_config.imageset),
        frames=6019, scenes=150, seed=928, steps=list(STEPS), radius_m=RADIUS,
        optimizer=dict(type='Adam', lr_m=LR, betas=[.9, .999], eps=1e-8,
                       weight_decay=0, reset='Every frame; zero initial displacement'),
        projection='After each update: Euclidean displacement <=0.4m; clamp inside native ROI',
        active='Only final-stage predicted Gaussian centers, independently per Gaussian',
        fixed=['network', 'semantics', 'opacity', 'scales', 'rotations',
               'empty Gaussian', 'empty scalar', 'empty bias', 'GS count', 'parent query'],
        loss='Original configured final-stage Occupancy CE+Lovasz; no Set or coverage loss',
        domain='All camera-valid GT voxels, labels 0..17; CE includes empty; Lovasz ignores 17',
        interpretation='Same-frame GT-assisted local geometry diagnostic, not normal validation or a global upper bound',
        queries=previous['queries'], children=previous['children'],
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        class_columns=CLASS_COLUMNS, transitions=list(TRANSITIONS), movement_columns=list(MOVEMENT))
    folder = RUN / size
    if (folder / 'protocol.json').exists():
        assert read(folder / 'protocol.json') == protocol, 'Protocol changed: preserve this run and choose a new directory'
    else:
        write(folder / 'protocol.json', protocol)
        cfg.dump(folder / 'resolved_config.py')
        for p in files:
            dest = folder / 'sources' / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    return cfg, protocol


def objective(cfg):
    from loss import OPENOCC_LOSS
    options = next(deepcopy(c) for c in cfg.loss.loss_cfgs if c.type == 'OccupancyLoss')
    occ = OPENOCC_LOSS.build(options).cuda()
    assert not occ.use_focal_loss and not occ.use_dice_loss and not occ.use_sem_geo_scal_loss
    assert occ.use_lovasz_loss and occ.lovasz_use_softmax and occ.lovasz_ignore == 17
    return occ


def loss_terms(z, target, occ):
    from loss.utils.lovasz_softmax import lovasz_softmax
    ce = F.cross_entropy(z.unsqueeze(0), target.unsqueeze(0),
        weight=occ.class_weights.to(z)) * occ.loss_voxel_ce_weight
    lov = lovasz_softmax(z.softmax(0).T, target, ignore=occ.lovasz_ignore) * occ.loss_voxel_lovasz_weight
    return torch.stack((ce + lov, ce, lov)) * occ.weight


class FrozenRenderer:
    def __init__(self, head, gaussian, xyz, empty_logits=None):
        self.head = head
        self.gaussian = type(gaussian)(*(v.detach().clone() if torch.is_tensor(v) else v for v in gaussian))
        args = head.prepare_gaussian_args(self.gaussian)
        self.args = tuple(v.detach() for v in args)
        self.initial = self.gaussian.means
        self.xyz = xyz.contiguous()
        self.empty_bias = head.empty_bias.detach().clone()
        # The appended global empty Gaussian has zero foreground semantics and
        # is entirely fixed. Its backward kernel would otherwise spend time on
        # every grid cell although none of its gradients is used. Cache exactly
        # its calibrated output and render only the predicted foreground GS.
        if empty_logits is None:
            with torch.no_grad():
                m, a, s, scale, inverse = self.args
                full = head.aggregator(self.xyz[None], m, a.reshape(1, -1), s, scale, inverse).T
                empty_logits = full[17] - self.empty_bias
        self.empty_logits = empty_logits.detach().clone()
        agg = head.aggregator
        self.lower = agg.pc_min[:, None, :] + 1e-4
        self.upper = agg.pc_min[:, None, :] + self.initial.new_tensor(
            [agg.H, agg.W, agg.D]) * agg.grid_size - 1e-4

    def render(self, delta):
        _, opacity, semantic, scales, inverse = self.args
        means = self.initial + delta
        z = self.head.aggregator(self.xyz[None], means, opacity[:, :-1].reshape(1, -1),
            semantic[:, :-1], scales[:, :-1], inverse[:, :-1]).T
        return torch.cat((z[:17], self.empty_logits[None]), 0)

    @torch.no_grad()
    def project(self, delta):
        norm = delta.norm(dim=-1, keepdim=True)
        delta.mul_((RADIUS / norm.clamp_min(RADIUS)).clamp(max=1))
        delta.copy_((self.initial + delta).maximum(self.lower).minimum(self.upper) - self.initial)


@torch.no_grad()
def coverage(renderer, delta, xyz_full, labels, valid, queries, children):
    # Reuse precomputed covariance and attributes. Only center positions change.
    means = (renderer.initial + delta)[0].contiguous()
    _, alpha, semantic, scales, inverse = renderer.args
    alpha = alpha[0, :-1].flatten().contiguous()
    semantic = semantic[0, :-1, :17].contiguous()
    inverse = inverse[0, :-1].flatten(1)[:, [0, 4, 8, 1, 5, 2]].contiguous()
    scales = scales[0, :-1]
    agg = renderer.head.aggregator
    centers = ((means - agg.pc_min) / agg.grid_size).int().contiguous()
    radii = torch.ceil(scales.max(-1).values * agg.scale_multiplier / agg.grid_size).int().contiguous()
    ids = torch.where(valid & (labels < 17))[0]
    lookup = torch.full_like(labels, -1)
    lookup[ids] = torch.arange(len(ids), device=labels.device, dtype=torch.int32)
    cls = semantic.argmax(-1).int().contiguous()
    _, counts, _ = extension().support(means, inverse, alpha, centers, radii, xyz_full,
        lookup, labels, cls, cls, len(ids), queries, children, agg.H, agg.W, agg.D)
    hit = counts[:, [0, 1, 2, 3, 4, 5, 7]] > 0
    cc = torch.zeros((17, 8), device=labels.device, dtype=torch.int64)
    y = labels[ids].long()
    cc[:, 0] = torch.bincount(y, minlength=17)
    for j in range(7):
        cc[:, j + 1] = torch.bincount(y[hit[:, j]], minlength=17)
    return cc


@torch.no_grad()
def observation(renderer, delta, z, target, baseline_pred, xyz_full, labels, valid, protocol):
    pred = z.argmax(0)
    cm = torch.bincount(target * 18 + pred, minlength=324).reshape(18, 18)
    fg, oldfg, newfg = target < 17, baseline_pred < 17, pred < 17
    oldok, newok = baseline_pred == target, pred == target
    values = [fg & ~oldfg & newfg, fg & oldfg & ~newfg,
        ~fg & oldfg & ~newfg, ~fg & ~oldfg & newfg,
        ~oldok & newok, oldok & ~newok,
        fg & oldfg & newfg & ~oldok & newok,
        fg & oldfg & newfg & oldok & ~newok]
    transition = torch.stack([v.sum() for v in values])
    d = delta.norm(dim=-1).double()
    move = torch.stack((d.new_tensor(d.numel()), d.sum(), d.square().sum(), d.max(),
                        (d > 1e-5).sum(), (d >= RADIUS - 1e-5).sum()))
    cc = coverage(renderer, delta, xyz_full, labels, valid, protocol['queries'], protocol['children'])
    return dict(confusion=cm.cpu().numpy(), class_coverage=cc.cpu().numpy(),
        transitions=transition.cpu().numpy(), movement=move.cpu().numpy())


def refine(net, out, batch, occ, protocol, audit=False):
    labels = batch['occ_label'].flatten().int().contiguous()
    camera = batch['occ_cam_mask'].flatten().bool()
    valid = camera & (labels >= 0) & (labels < 18)
    target = labels[valid].long()
    xyz_full = out['sampled_xyz'][0].detach().contiguous()
    xyz = xyz_full[valid].contiguous()
    baseline = out['pred_occ'][-1][0, :, valid].detach().clone()
    renderer = FrozenRenderer(net.head, out['gaussian'], xyz, baseline[17])
    delta = torch.nn.Parameter(torch.zeros_like(renderer.initial))
    opt = torch.optim.Adam([delta], lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0)
    baseline_pred = baseline.argmax(0)
    rows, trajectory = {}, []
    initial_attrs = {name:getattr(renderer.gaussian, name).clone() for name in
                     ('scales', 'rotations', 'opacities', 'semantics')}
    max_native_diff = 0.
    max_grad = 0.
    reference_gradient = None
    if audit:
        reference_delta = torch.zeros_like(renderer.initial, requires_grad=True)
        m, a, s, scale, inverse = renderer.args
        reference_means = torch.cat((renderer.initial+reference_delta, m[:, -1:]), dim=1)
        reference_z = net.head.aggregator(xyz[None], reference_means,
            a.reshape(1, -1), s, scale, inverse).T
        reference_z = torch.cat((reference_z[:17], reference_z[17:]-renderer.empty_bias),0)
        reference_gradient, = torch.autograd.grad(loss_terms(reference_z, target, occ)[0], reference_delta)
    for step in range(max(STEPS) + 1):
        opt.zero_grad(set_to_none=True)
        z = renderer.render(delta)
        if step == 0:
            max_native_diff = float((z.detach() - baseline).abs().max())
            torch.testing.assert_close(z, baseline, rtol=0, atol=0)
        values = loss_terms(z, target, occ)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Nonfinite objective')
        trajectory.append(values.detach().cpu().numpy())
        if step in STEPS:
            row = observation(renderer, delta, z.detach(), target, baseline_pred,
                              xyz_full, labels, valid, protocol)
            for key, value in row.items():
                rows.setdefault(key, []).append(value)
        if step == max(STEPS):
            break
        values[0].backward()
        if delta.grad is None or not bool(torch.isfinite(delta.grad).all()):
            raise FloatingPointError('Nonfinite center gradient')
        max_grad = max(max_grad, float(delta.grad.abs().max()))
        if step == 0 and reference_gradient is not None:
            torch.testing.assert_close(delta.grad, reference_gradient, rtol=0, atol=0)
        opt.step()
        renderer.project(delta)
    for name, value in initial_attrs.items():
        assert torch.equal(getattr(renderer.gaussian, name), value), name
    assert all(p.grad is None for p in net.parameters()), 'A network parameter received a gradient'
    assert float(delta.detach().norm(dim=-1).max()) <= RADIUS + 1e-5
    result = {key:np.stack(value) for key, value in rows.items()}
    result['loss_trajectory'] = np.stack(trajectory)
    result['audit'] = np.asarray([max_native_diff, max_grad], np.float64)
    if audit:
        with torch.no_grad():
            old = occ.loss_voxel([baseline[None]], xyz[None], target[None], None) * occ.weight
            first = loss_terms(baseline, target, occ)[0]
            torch.testing.assert_close(old, first, rtol=1e-6, atol=1e-6)
        result['_pilot'] = dict(status='passed', native_masked_logits_bitwise_exact=True,
            occupancy_loss_matches_original=True, frozen_attributes_exact=True,
            no_network_gradients=True, max_abs_center_gradient=max_grad,
            native_full_renderer_center_gradient_bitwise_exact=True,
            step0_loss=float(result['loss_trajectory'][0, 0]),
            step20_loss=float(result['loss_trajectory'][-1, 0]))
    return result


def numerical_audit(net):
    """Central finite differences of real CUDA mean gradients on a small scene."""
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
    device = next(net.parameters()).device
    means = torch.tensor([[-.13, -.17, .31], [.43, .29, .71], [-.53, .41, .51]], device=device)[None]
    scales = means.new_tensor([[.23, .29, .31], [.28, .25, .33], [.27, .32, .24]])[None]
    rotation = means.new_tensor([[1., 0, 0, 0]]).repeat(1, 3, 1)
    alpha = means.new_tensor([.8, .9, .7]).reshape(1, 3, 1)
    semantic = means.new_full((1, 3, 17), .1)
    semantic[0, 0, 4] = 3.
    semantic[0, 1, 11] = 3.
    semantic[0, 2, 16] = 3.
    gaussian = GaussianPrediction(means, scales, rotation, alpha, semantic)
    axes = [torch.arange(-.6, .61, .4, device=device),
            torch.arange(-.6, .61, .4, device=device),
            torch.arange(-.4, 1.21, .4, device=device)]
    xyz = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    renderer = FrozenRenderer(net.head, gaussian, xyz)
    delta = torch.zeros_like(means, requires_grad=True)
    target = torch.full((len(xyz),), 17, device=device, dtype=torch.long)
    target[:len(xyz)//2] = 4
    # Reduce in float64 so subtracting two similar losses does not hide the
    # small change from one center coordinate. The production renderer is FP32.
    value = F.cross_entropy(renderer.render(delta)[None].double(), target[None])
    grad, = torch.autograd.grad(value, delta)
    errors = []
    for idx in grad.abs().flatten().argsort(descending=True)[:6].tolist():
        plus, minus = delta.detach().clone(), delta.detach().clone()
        plus.flatten()[idx] += .005
        minus.flatten()[idx] -= .005
        with torch.no_grad():
            fd = (F.cross_entropy(renderer.render(plus)[None].double(), target[None]) -
                  F.cross_entropy(renderer.render(minus)[None].double(), target[None])) / .01
        exact = grad.flatten()[idx]
        relative = float((fd - exact).abs() / exact.abs().clamp_min(1e-3))
        assert relative < .04, (idx, float(fd), float(exact), relative)
        errors.append(relative)
    return dict(status='passed', real_CUDA_center_finite_differences=6,
                max_relative_error=max(errors), finite_difference_step_m=.005)


def pilot(size):
    from dataset.utils import custom_collate_fn_temporal
    cfg, protocol = initialize(size)
    net = build(cfg, size)
    net.requires_grad_(False)
    occ = objective(cfg)
    numeric = numerical_audit(net)
    dataset = EvaluationFrames(cfg, 'val')
    cache = FeatureCache(net, dataset)
    try:
        batch = to_cuda(custom_collate_fn_temporal([dataset[0]]))
        with torch.no_grad():
            out = forward(net, batch)
        row = refine(net, out, batch, occ, protocol, True)
        scene = dataset.keyframes[0][0]
        with np.load(HISTORY / size / 'scenes' / f'{scene}.npz') as z:
            np.testing.assert_array_equal(row['confusion'][0], z['confusion'][0])
            np.testing.assert_array_equal(row['class_coverage'][0], z['class_coverage'][0])
        info = dict(**row.pop('_pilot'), numerical_gradient=numeric,
                    historical_first_frame_confusion_and_coverage_exact=True,
                    seconds=None, optimizer_hyperparameters_fixed_before_results=True)
        write(RUN / size / 'pilot.json', info)
        print(json.dumps(dict(model=size, pilot=info)), flush=True)
    finally:
        cache.close()
        net.flush_memory_phase()
        del net, cache, dataset
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(size, workers):
    from dataset.utils import custom_collate_fn_temporal
    cfg, protocol = initialize(size)
    folder = RUN / size
    assert read(folder / 'pilot.json')['status'] == 'passed'
    if (folder / 'complete.json').exists():
        assert read(folder / 'complete.json')['checkpoint_sha256'] == protocol['checkpoint_sha256']
        return
    net = build(cfg, size)
    net.requires_grad_(False)
    occ = objective(cfg)
    dataset = EvaluationFrames(cfg, 'val')
    assert len(dataset) == 6019
    scenes = {}
    for i, (scene, _) in enumerate(dataset.keyframes):
        scenes.setdefault(scene, []).append(i)
    assert len(scenes) == 150
    pending = []
    for scene, ids in scenes.items():
        path = folder / 'scenes' / f'{scene}.json'
        if path.exists():
            meta = read(path)
            assert meta['indices'] == ids and meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
            assert path.with_suffix('.npz').exists()
        else:
            pending += ids
    loader = DataLoader(Subset(dataset, pending), batch_size=1, num_workers=workers,
        shuffle=False, collate_fn=custom_collate_fn_temporal, pin_memory=True,
        persistent_workers=workers > 0, **(dict(prefetch_factor=2) if workers else {}))
    cache = FeatureCache(net, dataset)
    started = time.monotonic()
    done = 0
    active = None
    try:
        for index, batch in zip(pending, loader):
            scene, f = dataset.keyframes[index]
            if active is None:
                active = scene
                arrays, indices, tokens = {}, [], []
                old = np.load(HISTORY / size / 'scenes' / f'{scene}.npz')
            assert active == scene
            batch = to_cuda(batch)
            with torch.no_grad():
                out = forward(net, batch)
            result = refine(net, out, batch, occ, protocol)
            j = len(indices)
            np.testing.assert_array_equal(result['confusion'][0], old['confusion'][j])
            np.testing.assert_array_equal(result['class_coverage'][0], old['class_coverage'][j])
            for key, value in result.items():
                arrays.setdefault(key, []).append(value)
            indices.append(index)
            tokens.append(dataset.base.scene_infos[scene][f]['token'])
            done += 1
            if index == scenes[scene][-1]:
                old.close()
                result = {key:np.stack(values) for key, values in arrays.items()}
                path = folder / 'scenes' / f'{scene}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream, **result)
                path.with_suffix('.tmp').replace(path)
                write(path.with_suffix('.json'), dict(status='complete', scene=scene,
                    frames=len(indices), indices=indices, tokens=tokens,
                    checkpoint_sha256=protocol['checkpoint_sha256']))
                completed = len(list((folder / 'scenes').glob('*.json')))
                status = dict(status='running', model=size, frames=6019-len(pending)+done,
                    scenes=completed, expected_frames=6019, expected_scenes=150,
                    elapsed_seconds=time.monotonic()-started,
                    diagnostic_frames_per_second=done/(time.monotonic()-started))
                write(folder / 'progress.json', status)
                print(json.dumps(status), flush=True)
                active = None
            del out, batch, result
        all_paths = list((folder / 'scenes').glob('*.json'))
        indices = [i for p in all_paths for i in read(p)['indices']]
        assert len(all_paths) == 150 and sorted(indices) == list(range(6019))
        for name, digest in protocol['source_sha256'].items():
            assert sha(ROOT / name) == digest, name
        status = dict(status='complete', model=size, frames=6019, scenes=150,
            seconds=time.monotonic()-started, checkpoint_sha256=protocol['checkpoint_sha256'])
        write(folder / 'complete.json', status)
        write(folder / 'progress.json', status)
    finally:
        cache.close()
        net.flush_memory_phase()
        del net, cache, dataset, loader
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('pilot', 'run'), default='run')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for size in args.models:
                write(RUN / 'status.json', dict(status='running', action=args.action, model=size, pid=os.getpid()))
                if args.action == 'pilot':
                    pilot(size)
                else:
                    evaluate(size, args.workers)
            write(RUN / 'status.json', dict(status=args.action+'_complete', models=args.models))
        except BaseException as exc:
            write(RUN / 'status.json', dict(status='failed', error=repr(exc), pid=os.getpid()))
            raise


if __name__ == '__main__':
    main()
