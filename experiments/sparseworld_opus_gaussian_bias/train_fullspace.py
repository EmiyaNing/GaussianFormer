"""Opt-in full-space runner with one integrated diagnostic per epoch."""
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from loss import OPENOCC_LOSS
from . import train as base_runner
from .fullspace_diagnostics import (
    add_evaluation_batch, attribute_record, gradient_probe,
    partition_histograms, region_metrics)
from .metrics import CLASS_NAMES, HorizonMetrics


def save_diagnostic(output, epoch, payload, formal):
    output.mkdir(parents=True, exist_ok=True)
    arrays = payload.pop('_visibility_arrays')
    np.savez_compressed(output / f'epoch_{epoch:02d}_visibility_confusion.npz',
                        **arrays)
    # Keep the JSON free of NumPy arrays; the NPZ carries exact counts.
    base_runner.atomic_json(output / f'epoch_{epoch:02d}_diagnostics.json', payload)
    baseline_path = Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e/sparseworld_strict') / f'epoch_{epoch:02d}_metrics.json'
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    rows = []
    for h, value in enumerate(formal['horizons']):
        for c, name in enumerate(CLASS_NAMES):
            reference = baseline['horizons'][h]['class_iou'][c] if baseline else None
            rows.append(dict(seconds=h * .5, class_index=c, class_name=name,
                             fullspace_iou=value['class_iou'][c], strict_iou=reference,
                             delta_percentage_points=value['class_iou'][c] - reference if reference is not None else None,
                             gt_voxels=value['class_gt_count'][c]))
    with (output / f'epoch_{epoch:02d}_class_iou.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'# 全空间监督：第{epoch}轮诊断', '',
             f"评估样本数：{formal['num_samples']}；mIoU固定平均17个非空类。"
             f"当前/1/2/3秒平均mIoU：{formal['selection_mean_miou']:.4f}%。", '',
             '## 本轮观察', '']
    full = payload['visibility']['h0/full']
    hidden = payload['visibility']['h0/noncamera']
    hidden_fp_share = 100 * hidden['occupied_fp'] / max(full['occupied_fp'], 1)
    hidden_gt_percent = 100 * hidden['gt_occupied'] / max(hidden['voxel_count'], 1)
    lines += [f"当前误报中不可见区域占{hidden_fp_share:.3f}%；不可见区域GT占据比例为{hidden_gt_percent:.3f}%，"
              f"模型预测占据比例为{hidden['predicted_occupied_percent']:.3f}%。", '',
              f"当前训练阶段：{'时间扩展预训练' if epoch <= 5 else '当前与未来逐步直接监督'}。"
              f"本轮实际Gaussian体素监督时刻：{[h * .5 for h in payload.get('supervised_gaussian_horizons', [0])]}秒。", '']
    if baseline is not None:
        for h in (0, 2, 4, 6):
            delta = np.asarray(formal['horizons'][h]['class_iou'][:17]) - np.asarray(baseline['horizons'][h]['class_iou'][:17])
            smallest = np.argsort(delta)[:3]
            details = '、'.join(f'{CLASS_NAMES[c]} {delta[c]:+.3f} pp' for c in smallest)
            lines += [f'{h * .5:g}秒与Strict同轮相比差值最小的三个类别：{details}。', '']
    lines += [
             '## 监督范围与可见性', '',
             'Gaussian体素监督mask=none，包含不可见区域及空体素。未来直接监督仍按原训练时域逐步启用。', '',
             '| 时刻 | 区域 | mIoU (%) | 占据IoU (%) | 精度 (%) | 召回 (%) | FP | FN |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for h in range(7):
        for region in ('full', 'camera', 'noncamera'):
            m = payload['visibility'][f'h{h}/{region}']
            lines.append(f"| {h * .5:g}s | {region} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {m['occupied_fp']} | {m['occupied_fn']} |")
    if baseline is not None:
        lines += ['', '## 与同轮Strict的完整子集对比', '',
                  '| 时刻 | Strict mIoU (%) | 全空间Gaussian mIoU (%) | Δ pp | Strict占据IoU (%) | 全空间Gaussian占据IoU (%) |',
                  '|---|---:|---:|---:|---:|---:|']
        for h in range(7):
            a, b = baseline['horizons'][h], formal['horizons'][h]
            lines.append(f"| {h * .5:g}s | {a['miou']:.4f} | {b['miou']:.4f} | {b['miou']-a['miou']:+.4f} | {a['occupied_iou']:.4f} | {b['occupied_iou']:.4f} |")
    lines += ['', '## Gaussian属性与未来特征', '',
              f"固定属性/渲染样本：{payload['attribute_sample_indices']}；梯度探针只覆盖其中前2个样本的实际监督时刻。", '',
              '| 时刻 | 尺度>0.395m (%) | opacity>0.99 (%) | ROI内child (%) | feature范数中位统计 |',
              '|---|---:|---:|---:|---:|']
    for h in range(7):
        records = [r for r in payload['attributes'] if r['horizon'] == h]
        if records:
            lines.append(f"| {h * .5:g}s | {np.mean([r['scale_upper_saturation_percent'] for r in records]):.3f} | {np.mean([r['opacity_upper_saturation_percent'] for r in records]):.3f} | {np.mean([r['in_range_percent'] for r in records]):.3f} | {np.median([r['feature_norm_quantiles'][2] for r in records]):.3f} |")
    lines += ['', '## 渲染探针', '',
              '以下固定小样本变体只用于诊断，不是正式模型成绩，不用于选择checkpoint或阈值。', '',
              '| 时刻 | 方式 | 区域 | mIoU (%) | 占据IoU (%) |', '|---|---|---|---:|---:|']
    for key, value in payload['render_probes'].items():
        h, mode, region = key.split('/')
        lines.append(f"| {int(h[1:]) * .5:g}s | {mode} | {region} | {value['miou']:.4f} | {value['occupied_iou']:.4f} |")
    lines += ['', '## 输出层梯度探针', '',
              '只比较实际启用的集合/体素损失对最后层raw logits与world centers的梯度；features、opacity、covariance固定，不代表整个模型参数梯度。', '',
              '| 样本 | 时刻 | 体素/集合语义范数比 | 语义余弦 | 中心范数比 | 中心余弦 |',
              '|---|---|---:|---:|---:|---:|']
    for p in payload['gradient_probes']:
        lines.append(f"| {p['index']} | {p['seconds']:g}s | {p['semantic']['voxel_to_set_norm_ratio']:.3f} | {p['semantic']['cosine']:.3f} | {p['geometry']['voxel_to_set_norm_ratio']:.3f} | {p['geometry']['cosine']:.3f} |")
    lines += ['', f'- [每时刻每类别IoU与同轮Strict对比](epoch_{epoch:02d}_class_iou.csv)',
              f'- [完整诊断JSON](epoch_{epoch:02d}_diagnostics.json)',
              f'- [可见性混淆矩阵](epoch_{epoch:02d}_visibility_confusion.npz)', '',
              '渲染/梯度探针没有更新参数；诊断前后保留随机数状态。']
    (output / f'epoch_{epoch:02d}_diagnostics.md').write_text('\n'.join(lines) + '\n')


@torch.no_grad()
def evaluate_with_diagnostics(model, dataset, cfg, directory, epoch, limit=None):
    if cfg.loss.mask_type != 'none' or cfg.val_loader.batch_size != 1:
        raise ValueError('full-space diagnostics require mask=none and validation batch size 1')
    options = cfg.epoch_diagnostics
    indices = np.linspace(0, len(dataset) - 1, options.attribute_samples, dtype=int).tolist()
    gradient_indices = set(indices[:options.gradient_samples])
    metric = HorizonMetrics()
    visibility = {region: np.zeros((7, 18, 18), dtype=np.int64)
                  for region in ('full', 'camera', 'noncamera')}
    attributes, gradients, probe_histograms, actual_indices = [], [], {}, []
    captured, capture_active = [], False
    original_render = model.gaussian_head.render_state
    rng = base_runner.capture_rng()

    def record_render(gaussian, xyz):
        logits = original_render(gaussian, xyz)
        if capture_active:
            captured.append((gaussian, logits))
        return logits

    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    model.gaussian_head.render_state = record_render
    loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
    started = time.monotonic()
    try:
        for index, data in enumerate(base_runner.loader(dataset, cfg.val_loader)):
            capture_active = index in indices
            captured.clear()
            images, metas = base_runner.move_batch(data, 'cuda')
            result = model(imgs=images, metas=metas)
            add_evaluation_batch(metric, visibility, result, metas)
            if capture_active:
                if len(captured) != 7:
                    raise ValueError('renderer did not expose all seven horizons')
                actual_indices.append(index)
                current = model.pts_bbox_head.ind_stamps_all == 0
                points_list = [result['all_refine_pts'][-1][:, current]] + result['forecast_points_list']
                classes_list = [result['all_cls_scores'][-1][:, current]] + result['forecast_semantics_list']
                features_list = [result['query_feat'][:, current]] + result['forecast_features_list']
                xyz = metas['occ_xyz'].float().flatten(1, 3)
                trained_horizon = max(1, min(model.curr_epoch - model.finetune_epoch + 1, 6))
                for h, (gaussian, logits) in enumerate(tuple(captured)):
                    points, classes, features = points_list[h], classes_list[h], features_list[h]
                    attributes.append(attribute_record(index, h, points, classes, features, gaussian, model.gaussian_head))
                    target = metas['occ_label'] if h == 0 else metas['future_occ_labels'][:, h - 1]
                    cam = metas['occ_cam_mask'] if h == 0 else metas['future_occ_cam_masks'][:, h - 1]
                    modes = dict(default=logits.argmax(1).view_as(target),
                                 point_rasterizer=model.pts_bbox_head.get_occ(dict(cls_scores=classes, refine_pts=points)))
                    for delta in (-2., -1., 1., 2.):
                        changed = logits.clone()
                        changed[:, 17] -= delta
                        modes[f'empty_bias_delta_{delta:+g}'] = changed.argmax(1).view_as(target)
                    modes['fixed_scale_0_2'] = original_render(
                        gaussian._replace(scales=torch.full_like(gaussian.scales, .2)), xyz).argmax(1).view_as(target)
                    modes['opacity_times_0_5'] = original_render(
                        gaussian._replace(opacities=gaussian.opacities * .5), xyz).argmax(1).view_as(target)
                    for mode, prediction in modes.items():
                        for region, hist in partition_histograms(prediction, target, cam).items():
                            key = f'h{h}/{mode}/{region}'
                            probe_histograms.setdefault(key, np.zeros((18, 18), dtype=np.int64))
                            probe_histograms[key] += hist
                    supervised = h == 0 or (not model.pretrain and h <= trained_horizon and h in model.gaussian_loss_horizons)
                    if index in gradient_indices and supervised:
                        model.gaussian_head.render_state = original_render
                        gradients.append(dict(index=index, **gradient_probe(
                            model, cfg, result, gaussian, xyz, metas, h, loss_fn)))
                        model.gaussian_head.render_state = record_render
            if (index + 1) % 128 == 0:
                print(f'EVAL+DIAG epoch={epoch} samples={metric.samples}/{len(dataset)} elapsed={time.monotonic()-started:.1f}s', flush=True)
            if limit is not None and index + 1 >= limit:
                break
    finally:
        model.gaussian_head.render_state = original_render
        captured.clear()
        base_runner.restore_rng(rng)
        model.memory.clear()
        model.memory_queue.clear()
    if limit is None and metric.samples != 1024:
        raise ValueError('epoch diagnostics require all 1024 held-out samples')
    for h in range(7):
        if not np.array_equal(visibility['full'][h], visibility['camera'][h] + visibility['noncamera'][h]):
            raise ValueError('visibility partitions are inconsistent')
    formal = dict(metric.results(), epoch=epoch, evaluation_seconds=time.monotonic() - started)
    base_runner.atomic_json(directory / f'epoch_{epoch:02d}_metrics.json', formal)
    np.savez_compressed(directory / f'epoch_{epoch:02d}_confusion.npz', confusion=metric.confusion)
    diagnostic_dir = (Path(cfg.report_directory) / 'epoch_diagnostics' if limit is None else
                      directory / 'epoch_diagnostics_smoke')
    payload = dict(epoch=epoch, num_samples=metric.samples, is_smoke=limit is not None,
                   voxel_supervision_mask=cfg.loss.mask_type, config_sha256=base_runner.digest(cfg.filename),
                   class_names=CLASS_NAMES, attribute_sample_indices=actual_indices,
                   supervised_gaussian_horizons=[0] + ([] if model.pretrain else
                       [h for h in model.gaussian_loss_horizons if 0 < h <=
                        max(1, min(model.curr_epoch - model.finetune_epoch + 1, 6))]),
                   visibility={f'h{h}/{region}': region_metrics(visibility[region][h])
                               for h in range(7) for region in visibility},
                   attributes=attributes, gradient_probes=gradients,
                   render_probes={key: region_metrics(hist) for key, hist in probe_histograms.items()},
                   selection_mean_miou=formal['selection_mean_miou'])
    # NPZ arrays are passed separately because JSON accepts only plain values.
    arrays = dict(visibility)
    save_diagnostic(diagnostic_dir, epoch, dict(payload, _visibility_arrays=arrays), formal)
    if limit is None:
        base_runner.atomic_json(Path(cfg.report_directory) / 'status.json',
            dict(status='running', model=cfg.experiment_name, phase='train',
                 last_completed_epoch=epoch, last_completed_epoch_diagnostic=epoch,
                 diagnostic=str(diagnostic_dir / f'epoch_{epoch:02d}_diagnostics.md')))
    print(f'DIAGNOSTIC epoch={epoch} samples={metric.samples} report={diagnostic_dir}', flush=True)
    return formal


def main():
    import sys
    if '--evaluate-best' not in sys.argv:
        base_runner.evaluate = evaluate_with_diagnostics
    base_runner.main()


if __name__ == '__main__':
    main()
