#!/usr/bin/env python3
"""Joint future-image feature/geometry refinement, with disk-backed FPN cache."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from mmengine import Config
from mmseg.models import build_segmentor

from dataset.utils import custom_collate_fn_temporal
from tools.diagnose_teacher_alignment import seed_all, tree_to, loader_for, add_metric
from tools.diagnose_future_image_features_adaptive import cpu_maps
from eval_teacher_aligned import _metric, _json_safe
from model.future_image_joint import FutureImageJointRefiner
from model.multi_horizon_teacher import file_sha256
from loss import OPENOCC_LOSS

HORIZONS = (2, 4, 6)
ARMS = ('query_only', 'current_image', 'future_image', 'future_shuffled')


def save_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_json_safe(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def metric_row(metric):
    return dict(hist=metric.semantic.hist.copy(),
                intersection=int(metric.occupied_intersection),
                union=int(metric.occupied_union))


def rows_metrics(rows, cfg, selected=None):
    indices = range(len(rows)) if selected is None else selected
    indices = list(indices)
    result = {}
    for h in HORIZONS:
        metric = _metric(cfg)
        for i in indices:
            row = rows[i][h]
            metric.semantic.hist += row['hist']
            metric.semantic.cnt += 1
            metric.occupied_intersection += row['intersection']
            metric.occupied_union += row['union']
        result[f'{h * .5:g}s'] = metric.results()
    return result


@torch.no_grad()
def build_cache(student, loader, manifest, cfg, directory, label):
    paths = [directory / label / f'{index:06d}' for index in manifest['indices']]
    missing = [i for i, path in enumerate(paths) if not (path / 'complete.json').exists()]
    if missing:
        subset_loader = DataLoader(
            Subset(loader.dataset, missing), batch_size=1, num_workers=loader.num_workers,
            collate_fn=custom_collate_fn_temporal, pin_memory=True)
        for position, data in zip(missing, subset_loader):
            data = tree_to(data, 'cuda')
            imgs = data.pop('img')
            metas = dict(data, projection_mat=data['projection_mat'][:, :30],
                         image_wh=data['image_wh'][:, :30])
            output = student(imgs=imgs[:, :30], metas=metas,
                             forecast_snapshot_step=HORIZONS)
            path = paths[position]
            path.mkdir(parents=True, exist_ok=True)
            torch.save(cpu_maps([f[:, :6] for f in output['ms_img_feats']]),
                       path / 'current.pth')
            states, baseline = {}, {}
            for h in HORIZONS:
                snap = output['forecast_snapshots'][h]
                begin = (cfg.teacher_history_frames + h) * 6
                maps = student.extract_img_feat(imgs[:, begin:begin + 6])['ms_img_feats']
                torch.save(cpu_maps(maps), path / f'future_{h}.pth')
                born = snap['stamps'][snap['slot_ids']] > 0
                assert snap['features'].shape[1] == 720 + sum(student.future_queries[:h])
                states[h] = tree_to(dict(
                    query=snap['features'], normalized_points=snap['positions'],
                    logits=snap['semantic'], born_mask=born,
                    future_metas=dict(projection_mat=data['projection_mat'][:, begin:begin + 6],
                                      image_wh=data['image_wh'][:, begin:begin + 6]),
                    current_metas=dict(projection_mat=data['projection_mat'][:, :6] @
                        data['future_ego_to_current'][:, h - 1, None],
                                      image_wh=data['image_wh'][:, :6])), 'cpu')
                metric = _metric(cfg)
                add_metric(metric, student, student._world(snap['positions']).flatten(1, 2),
                           snap['semantic'].flatten(1, 2), data, h)
                baseline[h] = metric_row(metric)
                del maps
            keep = ('future_occ_labels', 'future_occ_cam_masks', 'future_occ_lidar_masks',
                    'sample_idx', 'scene_token')
            retained = tree_to({k: data[k] for k in keep}, 'cpu')
            retained['future_occ_labels'] = retained['future_occ_labels'].to(torch.uint8)
            record = dict(states=states, metas=retained, baseline=baseline,
                          scene=manifest['scenes'][position], index=manifest['indices'][position])
            torch.save(record, path / 'meta.pth')
            save_json(dict(index=record['index'], scene=record['scene']), path / 'complete.json')
            if position == missing[0]:
                sample_bytes = sum(p.stat().st_size for p in path.iterdir())
                need = sample_bytes * (len(missing) - 1)
                if shutil.disk_usage(directory).free < need * 1.15:
                    raise RuntimeError(f'Insufficient cache storage: estimated remaining {need / 1e9:.1f} GB')
                print(f'[cache budget] {label}: {sample_bytes / 1e6:.1f} MB/sample, '
                      f'{need / 1e9:.1f} GB remaining', flush=True)
            if (position + 1) % 16 == 0 or position + 1 == len(paths):
                print(f'[cache {label}] {position + 1}/{len(paths)}', flush=True)
            del output, record, states, data, imgs, metas
    return paths


class DiskSamples(Dataset):
    def __init__(self, paths, arm):
        self.paths, self.arm = paths, arm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        item = torch.load(path / 'meta.pth', map_location='cpu')
        if self.arm == 'current_image':
            item['current_maps'] = torch.load(path / 'current.pth', map_location='cpu')
        elif self.arm != 'query_only':
            for h in HORIZONS:
                item['states'][h]['maps'] = torch.load(path / f'future_{h}.pth', map_location='cpu')
        return item


def one_item(items):
    return items[0]


def cache_loader(paths, arm, workers, schedule=None):
    dataset = DiskSamples(paths, arm)
    if schedule is not None:
        dataset = Subset(dataset, schedule)
    options = dict(batch_size=1, num_workers=workers, collate_fn=one_item,
                   pin_memory=False)
    if workers:
        options.update(prefetch_factor=1)
    return DataLoader(dataset, **options)


def forward_horizon(refiner, sample, horizon, arm):
    state = sample['states'][horizon]
    maps = None
    metas = None
    if arm == 'current_image':
        maps = sample['current_maps']
        metas = state['current_metas']
    elif arm != 'query_only':
        maps = state['maps']
        metas = state['future_metas']
    return refiner(state['query'].cuda(), state['normalized_points'].cuda(),
                   None if maps is None else [v.cuda().float() for v in maps],
                   None if metas is None else tree_to(metas, 'cuda'),
                   shuffle=arm == 'future_shuffled')


def gradient_norms(refiner):
    groups = ('reader.sampling.sampling_offset', 'reader.sampling.scale_weights',
              'reader.mixing', 'fusion', 'reg', 'velocity', 'semantic', 'self_attn')
    return {prefix: float(sum(p.grad.float().square().sum()
                             for n, p in refiner.named_parameters()
                             if n.startswith(prefix) and p.grad is not None) ** .5)
            for prefix in groups}


def sample_loss(refiner, sample, objective, arm, residual_weight):
    history = []
    for h in HORIZONS:
        predictions = forward_horizon(refiner, sample, h, arm)
        loss = 0.
        for stage, prediction in enumerate(predictions):
            cls, pts = objective._set_loss(
                prediction['points'][0], prediction['logits'][0],
                sample['metas']['future_occ_labels'][0, h - 1].cuda().long())
            weight = 1. if stage == len(predictions) - 1 else .25
            loss = loss + weight * (objective.lambda_future_cls * cls +
                objective.lambda_future_pts * pts +
                residual_weight * prediction['feature_delta'].square().mean())
        loss = loss / ((1. + .25 * (len(predictions) - 1)) * len(HORIZONS))
        loss.backward()
        history.append(float(loss.detach()))
        del predictions, loss
    return sum(history)


def preflight(student, factory, first_path, objective, args):
    refiner = factory().cuda().eval()
    sample = DiskSamples([first_path], 'future_image')[0]
    with torch.no_grad():
        predictions = forward_horizon(refiner, sample, 2, 'future_image')
        for prediction in predictions:
            torch.testing.assert_close(prediction['points'],
                student._world(sample['states'][2]['normalized_points'].cuda()).flatten(1, 2),
                rtol=0, atol=1e-5)
            torch.testing.assert_close(prediction['logits'],
                sample['states'][2]['logits'].cuda().flatten(1, 2), rtol=0, atol=0)
    assert all(p.requires_grad for p in refiner.reader.sampling.parameters())
    trainable = [p for p in refiner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
    refiner.train()
    checks = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        value = sample_loss(refiner, sample, objective, 'future_image', args.residual_weight)
        checks.append(dict(step=step + 1, loss=value, gradients=gradient_norms(refiner)))
        torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        optimizer.step()
    for name in ('reader.sampling.sampling_offset', 'reader.sampling.scale_weights', 'reg'):
        if max(check['gradients'][name] for check in checks) <= 0:
            raise RuntimeError(f'No actual task gradient reaches {name}')
    refiner.eval()
    with torch.no_grad():
        prediction = forward_horizon(refiner, sample, 2, 'future_image')[-1]
        movement = float((prediction['points'] - student._world(
            sample['states'][2]['normalized_points'].cuda()).flatten(1, 2)).norm(dim=-1).mean())
    if movement <= 0:
        raise RuntimeError('Joint refiner cannot change point positions')
    del refiner, sample, predictions, prediction
    return dict(initial_baseline_parity=True, sampling_requires_grad=True,
                gradient_steps=checks, mean_point_movement_after_three_updates_m=movement)


@torch.no_grad()
def evaluate(student, refiner, paths, cfg, arm, args, destination, confirmation_indices,
             intervention=None):
    refiner.eval()
    rows, diagnostics = [], {h: [] for h in HORIZONS}
    for index, sample in enumerate(cache_loader(paths, arm, args.workers)):
        sample_rows = {}
        for h in HORIZONS:
            use_arm = 'query_only' if intervention == 'zero' else (
                'future_shuffled' if intervention == 'shuffle' else arm)
            prediction = forward_horizon(refiner, sample, h, use_arm)[-1]
            metric = _metric(cfg)
            add_metric(metric, student, prediction['points'], prediction['logits'],
                       sample['metas'], h)
            sample_rows[h] = metric_row(metric)
            original = refiner.world(sample['states'][h]['normalized_points'].cuda()).flatten(1, 2)
            diagnostics[h].append(dict(movement_m=float((prediction['points'] - original).norm(dim=-1).mean()),
                                       visible=float(prediction['visible'].float().mean())))
        rows.append(sample_rows)
        if (index + 1) % 64 == 0 or index + 1 == len(paths):
            print(f'[eval] {arm}/{intervention or "normal"}: {index + 1}/{len(paths)}', flush=True)
    # Compact per-sample confusion matrices allow scene-level paired statistics.
    np.savez_compressed(destination,
                        hist=np.stack([[row[h]['hist'] for h in HORIZONS] for row in rows]),
                        intersection=np.asarray([[row[h]['intersection'] for h in HORIZONS] for row in rows]),
                        union=np.asarray([[row[h]['union'] for h in HORIZONS] for row in rows]))
    return dict(metrics=rows_metrics(rows, cfg),
                confirmation=rows_metrics(rows, cfg, confirmation_indices) if confirmation_indices else None,
                diagnostics={f'{h*.5:g}s': {key: float(np.mean([r[key] for r in values]))
                                           for key in values[0]} for h, values in diagnostics.items()},
                confusion_file=destination.name)


def main(args):
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'report.json'
    if report_path.exists() and not args.resume:
        raise FileExistsError(f'Use a new output directory: {output}')
    seed_all(args.sample_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(args.data_config)
    model_cfg = Config.fromfile('config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    student = build_segmentor(model_cfg.model)
    digest = file_sha256(cfg.student_init_checkpoint)
    assert digest == cfg.student_init_checkpoint_sha256
    student.load_state_dict(torch.load(cfg.student_init_checkpoint, map_location='cpu')['state_dict'], strict=True)
    student.set_epoch(100)
    student.requires_grad_(False).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    train_loader, training = loader_for(cfg, 'train', args.train_samples, args.workers, args.sample_seed)
    val_loader, validation = loader_for(cfg, 'val', args.val_samples, args.workers, args.sample_seed)
    assert not set(training['scenes']) & set(validation['scenes'])
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    contract = dict(version=1, checkpoint_sha256=digest, sample_seed=args.sample_seed,
                    config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest())
    if (cache / 'contract.json').exists():
        if json.loads((cache / 'contract.json').read_text()) != contract:
            raise RuntimeError('Incompatible existing FPN cache contract')
    else:
        save_json(contract, cache / 'contract.json')
    train_paths = build_cache(student, train_loader, training, cfg, cache, 'train')
    val_paths = build_cache(student, val_loader, validation, cfg, cache, 'val')
    baseline_rows = [torch.load(path / 'meta.pth', map_location='cpu')['baseline'] for path in val_paths]
    old = json.loads(Path(args.previous_report).read_text())
    old_scenes = set(old['validation']['scenes'])
    confirmation_indices = [i for i, scene in enumerate(validation['scenes']) if scene not in old_scenes]
    factory = lambda: FutureImageJointRefiner(student, args.rounds, args.max_step_m)
    objective = OPENOCC_LOSS.build(copy.deepcopy(cfg.base_sparseworld_loss)).cuda()
    audit = preflight(student, factory, train_paths[0], objective, args)
    print('[preflight]', json.dumps(audit), flush=True)
    report = dict(args=vars(args), checkpoint_sha256=digest, training=training, validation=validation,
                  unique_train_scenes=len(set(training['scenes'])), unique_val_scenes=len(set(validation['scenes'])),
                  confirmation_indices=confirmation_indices,
                  confirmation_scene_count=len({validation['scenes'][i] for i in confirmation_indices}),
                  baseline=rows_metrics(baseline_rows, cfg),
                  baseline_confirmation=rows_metrics(baseline_rows, cfg, confirmation_indices) if confirmation_indices else None,
                  preflight=audit, protocol=dict(
                      scope='privileged single-timestamp RGB; independent target horizons',
                      trainable='sampling offsets/scales, mixing, position encoding, spatial attention, FFN, fusion, semantic/regression/velocity heads',
                      rounds=args.rounds, geometry='differentiable point residual, resampled after every round',
                      query_counts=[840, 960, 1040], future_born='retained and GT supervised',
                      precision='FP32 computation; frozen FPN disk cache stored in FP16; TF32 disabled',
                      current_image='same future query locations projected with GT ego pose',
                      shuffle='training-time cyclic permutation among visible query slots after image sampling',
                      augmentation='fixed validation-style augmentation for all arms'), runs={})
    if args.resume and report_path.exists():
        previous = json.loads(report_path.read_text())
        for key in ('train_samples', 'val_samples', 'updates', 'rounds', 'max_step_m',
                    'lr', 'residual_weight', 'sample_seed', 'training_seeds', 'arms'):
            if previous['args'][key] != vars(args)[key]:
                raise RuntimeError(f'Resume protocol mismatch: {key}')
        report = previous
    save_json(report, report_path)
    for seed in args.training_seeds:
        report['runs'].setdefault(str(seed), {})
        for arm in args.arms:
            if arm in report['runs'][str(seed)]:
                print('[resume completed arm]', seed, arm, flush=True)
                continue
            seed_all(seed)
            refiner = factory().cuda().train()
            initial_sampling = {k: v.detach().cpu().clone() for k, v in refiner.reader.sampling.state_dict().items()}
            parameters = [p for p in refiner.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
            rng = np.random.RandomState(seed)
            schedule = []
            while len(schedule) < args.updates:
                schedule.extend(rng.permutation(len(train_paths)).tolist())
            history, start_step = [], 0
            checkpoint_path = output / f'{seed}_{arm}_latest.pth'
            if args.resume and checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                refiner.load_state_dict(checkpoint['state_dict'], strict=True)
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_step = checkpoint['step']
                history = checkpoint.get('history', [])
                print('[resume step]', seed, arm, start_step, flush=True)
            for step, sample in enumerate(cache_loader(train_paths, arm, args.workers,
                                                       schedule[start_step:args.updates]), start=start_step):
                optimizer.zero_grad(set_to_none=True)
                value = sample_loss(refiner, sample, objective, arm, args.residual_weight)
                norms = gradient_norms(refiner) if step < 3 or (step + 1) % 128 == 0 else None
                norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                if norms is not None:
                    history.append(dict(step=step+1, loss=value, grad_norm=float(norm), gradients=norms))
                    print('[train]', seed, arm, step+1, round(value, 6), flush=True)
                if (step + 1) % 512 == 0 or step + 1 == args.updates:
                    torch.save(dict(state_dict=refiner.state_dict(), optimizer=optimizer.state_dict(),
                                    step=step+1, seed=seed, arm=arm, args=vars(args), history=history),
                               checkpoint_path)
            result = evaluate(student, refiner, val_paths, cfg, arm, args,
                              output / f'{seed}_{arm}_confusions.npz', confirmation_indices)
            result.update(training=history,
                          trainable_parameters=sum(p.numel() for p in parameters),
                          sampling_weight_changes={k: float((v.detach().cpu()-initial_sampling[k]).abs().max())
                              for k, v in refiner.reader.sampling.state_dict().items()})
            if arm == 'future_image':
                for intervention in ('zero', 'shuffle'):
                    result[intervention] = evaluate(student, refiner, val_paths, cfg, arm, args,
                        output / f'{seed}_{arm}_{intervention}_confusions.npz', confirmation_indices,
                        intervention=intervention)
            report['runs'][str(seed)][arm] = result
            report['elapsed_seconds'] = time.perf_counter() - started
            report['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
            save_json(report, report_path)
            print('[arm complete]', seed, arm, {h: round(v['mIoU'], 4) for h, v in result['metrics'].items()}, flush=True)
            del refiner, optimizer, parameters
    print('[complete]', report_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', default='config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py')
    parser.add_argument('--output-dir', default='out/future_alignment_hypotheses/H4_joint_512x256')
    parser.add_argument('--cache-dir', default='out/future_alignment_hypotheses/H4_joint_cache')
    parser.add_argument('--previous-report', default='out/future_alignment_hypotheses/H4_R2_decoder128/report.json')
    parser.add_argument('--train-samples', type=int, default=512)
    parser.add_argument('--val-samples', type=int, default=256)
    parser.add_argument('--updates', type=int, default=2048)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--max-step-m', type=float, default=4.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--residual-weight', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample-seed', type=int, default=20260926)
    parser.add_argument('--training-seeds', type=int, nargs='+', default=[20260926, 20260927, 20260928])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
