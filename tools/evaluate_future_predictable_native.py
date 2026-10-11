#!/usr/bin/env python3
"""Evaluate H3 adapters with original SparseWorld perception/forecast heads.

The adapter was trained independently at each frozen rollout horizon. Thus its
output is decoded by the native forecast branches at that horizon, without
feeding the adapted feature into later steps (which would change the experiment).
Previously evaluated native teacher rows are reused only after sample/checkpoint
identity checks; every baseline row is replayed and compared with saved rows.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from mmengine import Config

from eval_teacher_aligned import _metric
from model.multi_horizon_teacher import file_sha256
from tools.analyze_future2current_regions import grouped_metrics
from tools.diagnose_future_image_joint import metric_row, rows_metrics
from tools.diagnose_teacher_alignment import add_metric
from tools.evaluate_future2current_native import save_rows
from tools.experiment_future2current import baseline
from tools.experiment_future2current_identity import ResidualAdapter
from tools.experiment_future_predictable_information import save_json
from tools.experiment_future_response_distillation import build_student, paths_for

HORIZONS = (2, 4, 6)


def decode_with_native_head(model, original, adapter, horizon):
    """Same native semantic/reg/velocity branches and point refinement as baseline."""
    step = original['trace'][horizon - 1]
    features = adapter(step['output_features'])
    batch, count = features.shape[:2]
    semantic = model.semantic(features).view(batch, count, model.num_refines, 17)
    offset = model.reg(features).view(batch, count, model.num_refines, 3) * .5
    velocity = model.velocity(features).view(batch, count, model.num_refines, 2)
    moving = ((semantic.argmax(-1) >= 2) & (semantic.argmax(-1) <= 10)).unsqueeze(-1)
    offset = torch.cat([offset[..., :2] + velocity * moving, offset[..., 2:]], -1)
    positions = model._refine(step['interaction_positions'], offset)
    return model._world(positions).flatten(1, 2), semantic.flatten(1, 2)


def load_saved_rows(path, selected, expected_count):
    with np.load(path) as saved:
        if saved['hist'].shape != (expected_count, 3, 18, 18):
            raise RuntimeError(f'Saved native row shape differs: {path}')
        return [{h: dict(hist=saved['hist'][position, j].copy(),
                         intersection=int(saved['intersection'][position, j]),
                         union=int(saved['union'][position, j]))
                 for j, h in enumerate(HORIZONS)} for position in selected]


def summarized_rows(rows, cfg):
    hist = np.stack([[row[h]['hist'] for h in HORIZONS] for row in rows]).sum(0)
    grouped = grouped_metrics(hist)
    return dict(metrics=rows_metrics(rows, cfg),
                groups={f'{h / 2:g}s': {
                    name: dict(mIoU=float(grouped[name]['macro_iou'][i]),
                               IoU=float(grouped[name]['group_iou'][i]))
                    for name in ('dynamic', 'static')}
                    for i, h in enumerate(HORIZONS)})


def relative_to_baseline(result, reference):
    result['delta_vs_baseline_pp'] = {
        key: {measure: result['metrics'][key][measure] - reference['metrics'][key][measure]
              for measure in ('mIoU', 'OccIoU')}
        for key in reference['metrics']}
    result['group_delta_vs_baseline_pp'] = {
        key: {name: {measure: result['groups'][key][name][measure] -
                               reference['groups'][key][name][measure]
                     for measure in ('mIoU', 'IoU')} for name in ('dynamic', 'static')}
        for key in reference['groups']}
    return result


@torch.no_grad()
def main(args):
    cfg = Config.fromfile(args.config)
    settings = json.loads((args.experiment_root / 'settings.json').read_text())
    prior = json.loads((args.experiment_root / 'results.json').read_text())
    manifest = json.loads(Path(settings['source_manifest']).read_text())
    native = json.loads((args.native_reference_root / 'results.json').read_text())
    native_manifest = json.loads(Path(native['source_manifest']).read_text())
    response = json.loads((args.response_root / 'manifest.json').read_text())
    sha = cfg.student_checkpoint_sha256
    for identity in (settings['checkpoint_sha256'], manifest['checkpoint_sha256'],
                     native['checkpoint_sha256'], response['student_checkpoint_sha256'],
                     file_sha256(cfg.student_checkpoint)):
        if identity != sha:
            raise RuntimeError('Native evaluation checkpoint identities differ')
    if manifest['val'] != native_manifest['val']:
        raise RuntimeError('Teacher native evaluation manifest differs from source cache')
    for key in ('indices', 'scenes'):
        if manifest['val'][key] != response['validation'][key]:
            raise RuntimeError('Saved native baseline sample identities differ')
    locations = {index: i for i, index in enumerate(manifest['val']['indices'])}
    selected = [locations[index] for index in settings['val_indices']]
    val_manifest = dict(indices=settings['val_indices'],
                        scenes=[manifest['val']['scenes'][i] for i in selected])
    expected_baseline = load_saved_rows(native['baseline']['rows'], selected,
                                        len(manifest['val']['indices']))
    teacher_rows = load_saved_rows(native['teacher']['rows'], selected,
                                   len(manifest['val']['indices']))
    sources = paths_for(Path(manifest['source_cache']), 'val', val_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(source_experiment=str(args.experiment_root), checkpoint_sha256=sha,
                    validation=val_manifest,
                    teacher_rows_source=native['teacher']['rows'],
                    baseline_rows_source=native['baseline']['rows'],
                    student_head='pretrained SparseWorld semantic/reg/velocity + native refinement/rasterizer',
                    teacher_head='pretrained SparseWorld perception head on real target-time five-frame window',
                    adapter_insertion='each horizon independently, before native forecast branches; no recurrent feedback',
                    training_loss='historical experiment trained adapters with shared readout, native heads were frozen')
    settings_path = args.output_dir / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != protocol:
        raise RuntimeError('Output directory belongs to a different native evaluation')
    save_json(settings_path, protocol)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_student(cfg, Config.fromfile(cfg.student_model_config))
    adapters = {}
    for arm, runs in prior['arms'].items():
        for seed, record in runs.items():
            adapter = ResidualAdapter().cuda().eval().requires_grad_(False)
            adapter.load_state_dict(torch.load(record['checkpoint'], map_location='cpu',
                                               weights_only=False), strict=True)
            adapters[(arm, seed)] = adapter
    rows = {key: [] for key in adapters}
    baseline_rows = []
    for i, source_path in enumerate(sources):
        source = torch.load(source_path, map_location='cpu', weights_only=False)
        if (source['index'], source['scene']) != (val_manifest['indices'][i], val_manifest['scenes'][i]):
            raise RuntimeError('Cached source sample identity differs')
        original = baseline(model, source)
        base_row = {}
        for h in HORIZONS:
            if i == 0:
                points, logits = decode_with_native_head(model, original, nn.Identity(), h)
                torch.testing.assert_close(points, original['points'][h - 1], atol=1e-5, rtol=0)
                torch.testing.assert_close(logits, original['logits'][h - 1], atol=1e-5, rtol=0)
            metric = _metric(cfg)
            add_metric(metric, model, original['points'][h - 1], original['logits'][h - 1], source['metas'], h)
            base_row[h] = metric_row(metric)
            expected = expected_baseline[i][h]
            if not np.array_equal(base_row[h]['hist'], expected['hist']) or any(
                    base_row[h][key] != expected[key] for key in ('intersection', 'union')):
                raise RuntimeError(f'Original baseline replay differs: sample {i}, horizon {h}')
        baseline_rows.append(base_row)
        for key, adapter in adapters.items():
            row = {}
            for h in HORIZONS:
                points, logits = decode_with_native_head(model, original, adapter, h)
                expected_queries = 720 + sum(model.future_queries[:h])
                if points.shape[1] != expected_queries * 48:
                    raise RuntimeError('Future-born query count changed')
                metric = _metric(cfg)
                add_metric(metric, model, points, logits, source['metas'], h)
                row[h] = metric_row(metric)
            rows[key].append(row)
        if (i + 1) % 32 == 0 or i + 1 == len(sources):
            print(f'[native H3, all {len(adapters)} checkpoints] {i + 1}/{len(sources)}', flush=True)
    save_rows(args.output_dir / 'baseline.npz', baseline_rows)
    save_rows(args.output_dir / 'teacher.npz', teacher_rows)
    base_result = summarized_rows(baseline_rows, cfg)
    report = dict(status='complete', protocol=protocol, baseline_replay_exact=True,
                  identity_adapter_parity=True, baseline=base_result,
                  teacher=relative_to_baseline(summarized_rows(teacher_rows, cfg), base_result), arms={})
    for (arm, seed), values in rows.items():
        destination = args.output_dir / f'{arm}_seed{seed}.npz'
        save_rows(destination, values)
        result = relative_to_baseline(summarized_rows(values, cfg), base_result)
        result['rows'] = str(destination)
        report['arms'].setdefault(arm, {})[seed] = result
    report['mean_arms'] = {}
    for arm, runs in report['arms'].items():
        values = list(runs.values())
        mean_result = dict(metrics={key: {
            measure: float(np.mean([r['metrics'][key][measure] for r in values]))
            for measure in ('mIoU', 'OccIoU')}
            for key in base_result['metrics']},
            groups={key: {group: {
                measure: float(np.mean([r['groups'][key][group][measure] for r in values]))
                for measure in ('mIoU', 'IoU')} for group in ('dynamic', 'static')}
                for key in base_result['groups']})
        report['mean_arms'][arm] = relative_to_baseline(mean_result, base_result)
    save_json(args.output_dir / 'results.json', report)
    print(f'[complete native H3] {args.output_dir / "results.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/opus/future_predictable_information.py')
    parser.add_argument('--experiment-root', type=Path,
                        default=Path('out/future2current/predictable_information_1024x256'))
    parser.add_argument('--native-reference-root', type=Path,
                        default=Path('out/future2current/native_2048x512'))
    parser.add_argument('--response-root', type=Path,
                        default=Path('out/future_distillation_response_based/2048x512'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('out/future2current/predictable_information_native256'))
    main(parser.parse_args())
