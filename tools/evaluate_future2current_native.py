#!/usr/bin/env python3
"""Evaluate Future2Current teacher and original-architecture students natively.

The teacher consumes real future images and uses the pretrained SparseWorld
perception head and its own points. Students use the original SparseWorld
future semantic/geometry heads and rasterizer. No shared probe is involved.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config

from eval_teacher_aligned import _metric
from model.multi_horizon_teacher import file_sha256
from tools.analyze_future2current_regions import grouped_metrics
from tools.diagnose_future_image_joint import metric_row, rows_metrics
from tools.diagnose_teacher_alignment import add_metric, loader_for, tree_to
from tools.experiment_future2current import _future_window, baseline
from tools.experiment_future_response_distillation import build_student, paths_for


HORIZONS = (2, 4, 6)
ARMS = ('task_only', 'raw_align', 'semantic_kd_3s')
SEEDS = (20260926, 20260927, 20260928)


def save_rows(path, rows):
    np.savez_compressed(
        path,
        hist=np.stack([[row[h]['hist'] for h in HORIZONS] for row in rows]),
        intersection=np.asarray([[row[h]['intersection'] for h in HORIZONS] for row in rows]),
        union=np.asarray([[row[h]['union'] for h in HORIZONS] for row in rows]),
    )


def group_results(path):
    with np.load(path) as data:
        hist = data['hist'].sum(axis=0)
    grouped = grouped_metrics(hist)
    return {
        f'{h / 2:g}s': {
            name: {
                'mIoU': float(grouped[name]['macro_iou'][i]),
                'IoU': float(grouped[name]['group_iou'][i]),
            }
            for name in ('dynamic', 'static')
        }
        for i, h in enumerate(HORIZONS)
    }


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding='utf8')
    tmp.replace(path)


def native_baseline(response_root, manifest):
    earlier_manifest = json.loads((response_root / 'manifest.json').read_text())
    if earlier_manifest['validation']['indices'] != manifest['indices'] or \
            earlier_manifest['validation']['scenes'] != manifest['scenes']:
        raise RuntimeError('Native baseline and experiment validation samples differ')
    metrics = json.loads((response_root / 'report.json').read_text())['pretrained_student']
    path = response_root / 'pretrained_student.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(metrics=metrics, groups=group_results(path), rows=str(path))


@torch.no_grad()
def eval_teacher(model, images, manifest, sources, cfg, destination):
    rows = []
    model.eval()
    for index, batch in enumerate(images):
        source = torch.load(sources[index], map_location='cpu', weights_only=False)
        if source['index'] != manifest['indices'][index]:
            raise RuntimeError('Future images/source cache sample identity mismatch')
        data = tree_to(batch, 'cuda')
        imgs = data.pop('img')
        row = {}
        for h in HORIZONS:
            selected_imgs, metas = _future_window(data, imgs, h)
            state = dict(imgs=selected_imgs, metas=metas, points=None)
            state.update(model.extract_img_feat(**state))
            state.update(model.lifter(**state))
            # Match the feature-identity experiment: do not leak a future ego
            # state through plan_head. Spatial coordinates are target ego.
            ego = source['ego'].cuda()
            scale = (torch.tanh(model.points_scale(ego)) + 1.) * .35 + .8
            state['query_points'] = state['query_points'] * scale
            state.update(model.encoder(**state))
            state.update(model.head(**state))
            prediction = state['final_occ_grid'][0].cpu().numpy()
            gt = source['metas']['future_occ_labels'][0, h - 1].numpy()
            camera = source['metas']['future_occ_cam_masks'][0, h - 1].numpy()
            lidar = source['metas']['future_occ_lidar_masks'][0, h - 1].numpy()
            metric = _metric(cfg)
            metric.add_batch(prediction, gt, lidar, camera)
            row[h] = metric_row(metric)
            del state, selected_imgs, metas
        rows.append(row)
        if (index + 1) % 32 == 0 or index + 1 == len(sources):
            print(f'[native teacher] {index + 1}/{len(sources)}', flush=True)
    save_rows(destination, rows)
    return dict(metrics=rows_metrics(rows, cfg), groups=group_results(destination),
                rows=str(destination))


@torch.no_grad()
def eval_student(model, sources, cfg, destination):
    rows = []
    model.eval()
    for index, path in enumerate(sources):
        source = torch.load(path, map_location='cpu', weights_only=False)
        prediction = baseline(model, source)
        row = {}
        for h in HORIZONS:
            metric = _metric(cfg)
            add_metric(metric, model, prediction['points'][h - 1],
                       prediction['logits'][h - 1], source['metas'], h)
            row[h] = metric_row(metric)
        rows.append(row)
        if (index + 1) % 64 == 0 or index + 1 == len(sources):
            print(f'[native {destination.stem}] {index + 1}/{len(sources)}', flush=True)
    save_rows(destination, rows)
    return dict(metrics=rows_metrics(rows, cfg), groups=group_results(destination),
                rows=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/opus/future2current_feature_identity.py')
    parser.add_argument('--experiment-root', type=Path,
                        default=Path('out/future2current/feature_identity_2048x512'))
    parser.add_argument('--response-root', type=Path,
                        default=Path('out/future_distillation_response_based/2048x512'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('out/future2current/native_2048x512'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--teacher-only', action='store_true')
    parser.add_argument('--students-only', action='store_true')
    parser.add_argument('--verify-baseline', action='store_true',
                        help='Replay the pretrained model through this evaluator and compare with saved native results')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = parser.parse_args()
    if args.teacher_only and args.students_only:
        parser.error('--teacher-only and --students-only are mutually exclusive')
    cfg = Config.fromfile(args.config)
    manifest = json.loads((args.experiment_root / 'manifest.json').read_text())
    if file_sha256(cfg.student_checkpoint) != cfg.student_checkpoint_sha256 or \
            manifest['checkpoint_sha256'] != cfg.student_checkpoint_sha256:
        raise RuntimeError('Pretrained checkpoint checksum mismatch')
    validation = manifest['val']
    sources = paths_for(Path(manifest['source_cache']), 'val', validation)
    if not all(path.exists() for path in sources):
        raise FileNotFoundError('Missing cached student encodings')
    earlier = json.loads((args.response_root / 'manifest.json').read_text())
    if earlier['student_checkpoint_sha256'] != manifest['checkpoint_sha256']:
        raise RuntimeError('Native baseline checkpoint differs from experiment')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / 'results.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf8'))
        if report.get('source_manifest') != str(args.experiment_root / 'manifest.json'):
            raise RuntimeError('Existing output belongs to a different experiment')
    else:
        report = dict(status='in_progress', source_manifest=str(args.experiment_root / 'manifest.json'),
                      checkpoint_sha256=manifest['checkpoint_sha256'],
                      protocol='original SparseWorld head and rasterizer; no shared readout',
                      samples=len(sources), validation_scenes=len(set(validation['scenes'])),
                      baseline=native_baseline(args.response_root, validation),
                      teacher=None, arms={})
        save_json(report_path, report)
    strict = Config.fromfile(cfg.student_model_config)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_student(cfg, strict)

    if args.verify_baseline:
        replay_path = args.output_dir / 'baseline_replay.npz'
        replay = eval_student(model, sources, cfg, replay_path)
        for h in ('1s', '2s', '3s'):
            for measure in ('mIoU', 'OccIoU'):
                expected = report['baseline']['metrics'][h][measure]
                actual = replay['metrics'][h][measure]
                if abs(actual - expected) > 0.02:
                    raise RuntimeError(f'Native baseline replay mismatch: {h} {measure} '
                                       f'{actual:.5f} vs {expected:.5f}')
        report['baseline_replay'] = replay
        save_json(report_path, report)

    if not args.students_only and report['teacher'] is None:
        images, image_manifest = loader_for(cfg, 'val', len(sources), args.workers,
                                            cfg.identity_sample_seed)
        if image_manifest['indices'] != validation['indices']:
            raise RuntimeError('Future image loader selects different samples')
        teacher_path = args.output_dir / 'teacher.npz'
        report['teacher'] = eval_teacher(model, images, validation, sources,
                                         cfg, teacher_path)
        save_json(report_path, report)

    if not args.teacher_only:
        original_weights = torch.load(cfg.student_checkpoint, map_location='cpu',
                                      weights_only=False)['state_dict']
        for arm in args.arms:
            report['arms'].setdefault(arm, {})
            for seed in args.seeds:
                if str(seed) in report['arms'][arm]:
                    continue
                checkpoint_path = args.experiment_root / f'{arm}_seed{seed}.pth'
                saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                model.load_state_dict(original_weights, strict=True)
                for module_name in ('ego_attention', 'position'):
                    getattr(model, module_name).load_state_dict(saved[module_name], strict=True)
                output_path = args.output_dir / f'{arm}_seed{seed}.npz'
                result = eval_student(model, sources, cfg, output_path)
                result['checkpoint'] = str(checkpoint_path)
                report['arms'][arm][str(seed)] = result
                save_json(report_path, report)
    report['status'] = ('complete' if report['teacher'] is not None and
                        all(len(report['arms'].get(arm, {})) == len(SEEDS) for arm in ARMS)
                        else 'in_progress')
    save_json(report_path, report)
    print(f"[native evaluation] {report['status']}: {report_path}", flush=True)


if __name__ == '__main__':
    main()
