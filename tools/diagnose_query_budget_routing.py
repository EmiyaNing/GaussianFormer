"""Frozen full-data audit of soft gates, hard routing and query demand.

All validation frames: five paired renders, plus query/core census.
All training frames: deployed hard render and query/core census.
Historical features are cached using the previously audited exact pipeline.
No optimizer or model parameter update is performed.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools.stat_boundary_gaussians import forward, to_cuda, read, write, sha
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from model.head.gaussian_opus_head import _flatten_grouped
from model.encoder.gaussian_budget_gate import gather_gaussian

RUN = ROOT / 'work_dirs/small_query_budget_camera_full_1ep/routing_diagnosis'
CONFIG = ROOT / 'config/full_occ3d_query_budget/opusv1_gaussian_small_8f_query_budget_1ep.py'
CHECKPOINT = ROOT / 'work_dirs/small_query_budget_camera_full_1ep/epoch_1.pth'
MODES = ('hard_raw', 'soft_all', 'hard_gated', 'uniform64', 'opacity_only')


def build():
    from mmengine.runner import set_random_seed
    from visualize import load_model
    cfg = Config.fromfile(CONFIG)
    set_random_seed(928)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    return cfg, load_model(cfg, CHECKPOINT, 'cuda:0')


def initialize(cfg):
    files = [Path(__file__), ROOT / 'tools/query_budget_routing_metrics.py',
        ROOT / 'model/encoder/gaussian_budget_gate.py',
        ROOT / 'model/encoder/opus_v1_gaussian_budget_encoder.py',
        ROOT / 'model/head/opus_v1_gaussian_budget_head.py',
        ROOT / 'model/head/gaussian_opus_head.py',
        ROOT / 'model/head/opus_v1_gaussian_trainable_bias_head.py',
        ROOT / 'tools/query_budget_stats/kernels.cu',
        ROOT / 'tools/boundary_gaussian_stats/data.py', CONFIG]
    expected = read(ROOT / 'work_dirs/boundary_gaussian_full/availability.json')
    protocol = dict(version=1, checkpoint=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=sha(CHECKPOINT), config=str(CONFIG.relative_to(ROOT)),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        annotation_sha256={s: sha(cfg[s + '_dataset_config'].imageset) for s in ('val', 'train')},
        expected={s: {k: expected[s][k] for k in ('frames', 'scenes')} for s in ('val', 'train')},
        mode_names=list(MODES), eval_mode=True, seed=928, camera_mask=True,
        scope='Full val: paired render and three geometry variants; full train: hard inference and hard geometry',
        assignment='Visible foreground GT -> nearest stage-5 center -> parent query; unchanged within paired modes',
        core='Native candidate cube, positive opacity*kernel and Mahalanobis distance <= 1',
        ratios='Counts summed across frames before ratios; per-frame correlations reported separately',
        causal_limit='Frozen-weight routing interventions; not retraining ablations')
    if (RUN / 'protocol.json').exists():
        assert read(RUN / 'protocol.json') == protocol, 'Changed protocol; use a new run directory'
    else:
        write(RUN / 'protocol.json', protocol)
        cfg.dump(RUN / 'resolved_config.py')
        for p in files:
            target = RUN / 'sources' / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    return protocol


def selections(out):
    state = out['representation'][-1]
    alpha = state['gaussian'].opacities.squeeze(-1)
    b, q, k = alpha.shape
    assert (b, q, k) == (1, 1200, 96)
    valid = state['query_valid_mask'][:, :, None].expand_as(alpha)
    assert bool(valid.all()), 'Uniform 64 counterfactual needs all queries valid'
    uniform = (torch.argsort(alpha, dim=-1, descending=True, stable=True)[..., :64]
               + torch.arange(q, device=alpha.device)[None, :, None] * k).flatten(1)
    opacity = torch.argsort(alpha.flatten(1), dim=-1, descending=True, stable=True)[:, :76800]
    return uniform, opacity


def render(head, gaussian, xyz):
    logits = head._render_batched(xyz, *head.prepare_gaussian_args(gaussian))
    logits[:, head.empty_label:] -= head.empty_bias.detach()
    assert bool(torch.isfinite(logits).all())
    return logits


def gate_statistics(out):
    a = out['allocation']
    p = a['gates'][0]
    selected = torch.zeros_like(p, dtype=torch.bool)
    selected[a['selected_indices'][0]] = True
    intermediate = (p > .1) & (p < .9)
    data = {}
    for name, mask in (('all', torch.ones_like(selected)), ('kept', selected), ('dropped', ~selected)):
        v = p[mask]
        data[name] = dict(count=len(v), gate_sum=float(v.double().sum()),
            intermediate_count=int(intermediate[mask].sum()),
            binary_sum=float((v.double() * (1 - v.double())).sum()),
            histogram=torch.histc(v, bins=20, min=0, max=1).long().cpu().tolist(),
            quantiles=torch.quantile(v, torch.tensor([0., .1, .25, .5, .75, .9, 1.], device=p.device)).cpu().tolist())
    # Fixed sum implies the removed soft gate mass equals opacity-gate
    # restoration mass on kept candidates; this is not rendered field mass.
    data['gate_mass_balance_error'] = abs(data['dropped']['gate_sum'] -
        (data['kept']['count'] - data['kept']['gate_sum']))
    score = a['query_scores'][0]
    data['query_score_mean'] = float(score.mean())
    data['query_score_std'] = float(score.std(unbiased=False))
    data['temperature'] = float(a['temperature'])
    priority = a['log_priority'][0]
    finite = torch.isfinite(priority)
    p_safe = p.clamp(1e-6, 1 - 1e-6)
    tau = (priority - a['temperature'] * torch.logit(p_safe))[intermediate & finite].median()
    data['soft_threshold'] = float(tau)
    sorted_priority = priority.sort(descending=True).values
    data['hard_boundary_log_priority'] = float(sorted_priority[76799])
    data['hard_boundary_gate'] = float(p[a['selected_indices'][0, -1]])
    data['boundary_priority_gap'] = float(sorted_priority[76799] - sorted_priority[76800])
    return data


def frame(network, out, batch, paired=True):
    from tools.query_budget_routing_metrics import routing_metrics
    labels = batch['occ_label'].flatten().long()
    camera = batch['occ_cam_mask'].flatten().bool()
    valid = camera & (labels >= 0) & (labels < 18)
    y = labels[valid]
    raw = out['representation'][-1]['gaussian']
    allocation = out['allocation']
    gate = allocation['gates'].reshape_as(raw.opacities)
    soft = raw._replace(opacities=raw.opacities * gate)
    base_pred = out['final_occ'].flatten()[valid]
    cms, changes = {}, {}
    modes = MODES if paired else MODES[:1]
    if paired:
        uniform, opacity = selections(out)
    for mode in modes:
        if mode == 'hard_raw':
            pred = base_pred
        else:
            if mode == 'soft_all':
                g = soft
            elif mode == 'hard_gated':
                g = gather_gaussian(soft, allocation['selected_indices'])
            elif mode == 'uniform64':
                g = gather_gaussian(raw, uniform)
            else:
                g = gather_gaussian(raw, opacity)
            z = render(network.head, _flatten_grouped(g), out['sampled_xyz'])
            pred = z.argmax(1).flatten()[valid]
            del z
        cms[mode] = torch.bincount(y * 18 + pred, minlength=18 * 18).reshape(18, 18).cpu().numpy()
        changes[mode] = dict(changed=int((pred != base_pred).sum()),
            hard_fn_rescued=int(((y < 17) & (base_pred == 17) & (pred < 17)).sum()),
            new_fn=int(((y < 17) & (base_pred < 17) & (pred == 17)).sum()),
            hard_fp_removed=int(((y == 17) & (base_pred < 17) & (pred == 17)).sum()),
            new_fp=int(((y == 17) & (base_pred == 17) & (pred < 17)).sum()))
    variants = ('current_hard', 'uniform64_per_query', 'all_candidates') if paired else ('current_hard',)
    summary, arrays = routing_metrics(network, out, batch, variants=variants)
    return dict(gates=gate_statistics(out), query=summary, changes=changes), arrays, cms


def pilot(cfg, network):
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    rows = []
    for split in ('val', 'train'):
        from copy import deepcopy
        options = deepcopy(cfg.val_dataset_config)
        options.imageset = cfg[split + '_dataset_config'].imageset
        native = OPENOCC_DATASET.build(options)
        batch = to_cuda(custom_collate_fn_temporal([native[0]]))
        with torch.inference_mode():
            reference = forward(network, batch)
            ref_logits = reference['pred_occ'][-1].clone()
            raw = _flatten_grouped(gather_gaussian(reference['representation'][-1]['gaussian'],
                reference['allocation']['selected_indices']))
            repeated = render(network.head, raw, reference['sampled_xyz'])
            torch.testing.assert_close(repeated, ref_logits, atol=0, rtol=0)
        del reference, repeated, batch, native
        dataset = EvaluationFrames(cfg, split)
        cache = FeatureCache(network, dataset)
        batch = to_cuda(custom_collate_fn_temporal([dataset[0]]))
        with torch.inference_mode():
            output = forward(network, batch)
            torch.testing.assert_close(output['pred_occ'][-1], ref_logits, atol=0, rtol=0)
            row, arrays, cms = frame(network, output, batch, paired=True)
        assert int(cms['hard_raw'].sum()) == int(cms['soft_all'].sum())
        assert row['gates']['gate_mass_balance_error'] < .1
        rows.append(dict(split=split, native_cache_logits_exact=True, native_repeat_logits_exact=True,
            valid_voxels=int(cms['hard_raw'].sum()), query_keys=list(arrays), gates=row['gates'], query=row['query']))
        cache.close()
        del output, batch, dataset, cache, ref_logits
    write(RUN / 'pilot.json', dict(status='passed', records=rows))


def run(cfg, network, protocol, workers, splits):
    from torch.utils.data import DataLoader, Subset
    from dataset.utils import custom_collate_fn_temporal
    start, processed = time.monotonic(), 0
    for split in splits:
        dataset = EvaluationFrames(cfg, split)
        by_scene = {}
        for i, (scene, _) in enumerate(dataset.keyframes):
            by_scene.setdefault(scene, []).append(i)
        assert len(dataset) == protocol['expected'][split]['frames']
        assert len(by_scene) == protocol['expected'][split]['scenes']
        pending = []
        for scene, ids in by_scene.items():
            path = RUN / 'scenes' / split / (scene + '.json')
            if path.exists():
                saved = read(path)
                assert saved['checkpoint_sha256'] == protocol['checkpoint_sha256'] and saved['indices'] == ids
                assert path.with_suffix('.npz').exists()
            else:
                pending.extend(ids)
        loader = DataLoader(Subset(dataset, pending), batch_size=1, shuffle=False, num_workers=workers,
            collate_fn=custom_collate_fn_temporal, pin_memory=True,
            **(dict(prefetch_factor=2, persistent_workers=True) if workers else {}))
        cache = FeatureCache(network, dataset)
        active = None
        for index, batch in zip(pending, loader):
            scene, scene_frame = dataset.keyframes[index]
            if active is None:
                active, rows, arrays, cms = scene, [], {}, {}
            assert scene == active
            batch = to_cuda(batch)
            with torch.inference_mode():
                output = forward(network, batch)
                row, arr, cm = frame(network, output, batch, paired=(split == 'val'))
            token = dataset.base.scene_infos[scene][scene_frame]['token']
            rows.append(dict(index=index, scene_frame=int(scene_frame), sample_token=token, **row))
            for key, value in arr.items():
                arrays.setdefault(key, []).append(value)
            for key, value in cm.items():
                cms[key] = cms.get(key, np.zeros((18, 18), np.int64)) + value
            processed += 1
            if len(rows) == len(by_scene[scene]):
                path = RUN / 'scenes' / split / (scene + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix('.npz.tmp').open('wb') as stream:
                    np.savez_compressed(stream, **{k: np.stack(v) for k, v in arrays.items()},
                        **{'confusion_' + k: v for k, v in cms.items()})
                path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
                write(path, dict(status='complete', split=split, scene=scene, indices=by_scene[scene],
                    checkpoint_sha256=protocol['checkpoint_sha256'], records=rows))
                completed = sum(1 for _ in (RUN / 'scenes' / split).glob('*.json'))
                elapsed = time.monotonic() - start
                write(RUN / 'progress.json', dict(status='running', pid=os.getpid(), split=split,
                    completed_scenes_in_split=completed, expected=protocol['expected'],
                    processed_this_run=processed, fps=processed / elapsed,
                    elapsed_seconds=elapsed, updated_at=datetime.now().astimezone().isoformat()))
                print(f'{split}: {completed}/{len(by_scene)} scenes, {processed} frames this run, '
                      f'{processed/elapsed:.3f} frames/s', flush=True)
                active = None
            del output, batch, row, arr, cm
        assert active is None
        cache.close()
        del loader, cache, dataset
        write(RUN / (split + '_complete.json'), dict(status='complete', **protocol['expected'][split]))
    write(RUN / 'progress.json', dict(status='complete', splits=splits, seconds=time.monotonic() - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('pilot', 'run'), default='run')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--splits', nargs='+', choices=('val', 'train'), default=['val', 'train'])
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, network = build()
        protocol = initialize(cfg)
        try:
            if not (RUN / 'pilot.json').exists():
                pilot(cfg, network)
            assert read(RUN / 'pilot.json')['status'] == 'passed'
            if args.action == 'run':
                run(cfg, network, protocol, args.workers, args.splits)
        except BaseException as error:
            write(RUN / 'failure.json', dict(error=repr(error), pid=os.getpid(),
                time=datetime.now().astimezone().isoformat()))
            raise


if __name__ == '__main__':
    main()
