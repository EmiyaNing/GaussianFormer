"""Streaming, paired before/after report for camera-filtered Set fine-tuning."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools import report_query_budget as query_report
from tools.diagnose_small_gaussians import GROUPS, ATTRS, FLAGS, Q, CHILDREN
from tools.stat_query_budget import FIELDS
from tools.stat_boundary_gaussians import read, write, sha

RUN = ROOT / 'work_dirs/small_camera_set_full_1ep/diagnostics'
BASELINE = ROOT / 'work_dirs/small_gaussian_diagnosis_full'
REPORT = ROOT / 'analysis/camera_set_finetune_coverage_results.md'
AGG = RUN / 'aggregate'
CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')
QUERY_GROUPS = ('all_queries', 'all_query_top20', 'active_low20', 'active_high20', 'no_assigned_foreground')


def div(a, b):
    return float(a / b) if b else 0.0


def cm_metrics(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)[..., :17]
    rows, cols = cm.sum(-1), cm.sum(-2)
    union = rows[..., :17] + cols[..., :17] - tp
    fg, empty = rows[..., :17].sum(-1), rows[..., 17]
    detected = cm[..., :17, :17].sum((-2, -1))
    fn, fp = cm[..., :17, 17].sum(-1), cm[..., 17, :17].sum(-1)
    ratio = lambda a, b: np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=b > 0)
    return dict(mIoU=100 * ratio(tp, union).mean(-1), IoU=100 * ratio(detected, fg + fp),
        FN_rate=100 * ratio(fn, fg), FP_over_empty=100 * ratio(fp, empty),
        FP_over_GTfg=100 * ratio(fp, fg), FP_over_predfg=100 * ratio(fp, detected + fp),
        wrong_fg_rate=100 * ratio(detected - tp.sum(-1), fg),
        detected_semantic_accuracy=100 * ratio(tp.sum(-1), detected),
        foreground=fg, empty=empty, FN=fn, FP=fp)


def vector_metrics(sums, queries, groups, attrs, flags, cm):
    f = lambda name: float(sums[FIELDS.index(name)])
    fg, holes = f('gt_foreground'), f('gt_no_core')
    ng = int(groups[0])
    result = {k: float(v) for k, v in cm_metrics(cm).items()}
    result.update(
        gaussians=ng, queries=queries, holes=int(holes),
        core_coverage=100 * div(fg - holes, fg), hole_rate=100 * div(holes, fg),
        mean_core_count=div(f('gt_core_count_sum'), fg),
        own_core=100 * div(f('gt_own_core'), fg),
        neighbor_only_core=100 * div(f('gt_neighbor_only_core'), fg),
        core_gs_contact=100 * div(f('core_gs_touches_fg'), ng),
        native_gs_contact=100 * div(f('native_gs_touches_fg'), ng),
        camera_native_gs=100 * div(f('gs_camera_native'), ng),
        mean_coarse_distance=div(f('gt_coarse_distance_sum'), fg),
        mean_scale=div(f('scale_mean'), queries),
        mean_max_scale=div(f('scale_max_mean'), queries),
        mean_opacity=div(f('opacity_mean'), queries),
        mean_semantic_amplitude=div(f('amplitude_mean'), queries),
        mean_effective_amplitude=div(f('effective_amplitude_mean'), queries),
        no_core_correct_rate=100 * div(f('gt_no_core_correct'), holes),
        center_visible_foreground=100 * div(groups[GROUPS.index('center_visible_foreground')], ng),
        center_visible_empty=100 * div(groups[GROUPS.index('center_visible_empty')], ng),
        center_other=100 * div(groups[GROUPS.index('center_other')], ng),
        pure_empty_camera_native=100 * div(groups[GROUPS.index('pure_empty_camera_native')], ng),
        pure_empty_camera_3sigma=100 * div(groups[GROUPS.index('pure_empty_camera_3sigma')], ng),
        scale_any_ge_039=100 * div(flags[0, FLAGS.index('scale_any_ge_039')], ng),
        scale_all_ge_039=100 * div(flags[0, FLAGS.index('scale_all_ge_039')], ng),
        visible_empty_high_score_opacity=100 * div(
            flags[GROUPS.index('center_visible_empty'), FLAGS.index('score_and_opacity_gt_09')], ng),
    )
    return result


def load_scene(root, split, name):
    path = root / 'scenes' / split / (name + '.json')
    meta = read(path)
    with np.load(path.with_suffix('.npz')) as z:
        arrays = {k: z[k] for k in ('metrics', 'strata', 'confusion', 'group_counts', 'attribute_sums', 'flag_counts')}
    a = arrays['metrics']
    assert a.shape == (meta['frames'], Q, len(FIELDS))
    assert np.isfinite(a).all()
    sums = a.sum((0, 1), dtype=np.float64)
    metric = vector_metrics(sums, a.shape[0] * Q, arrays['group_counts'],
        arrays['attribute_sums'], arrays['flag_counts'], arrays['confusion'])
    assert int(metric['foreground']) == int(arrays['strata'][..., 0].sum())
    assert int(metric['holes']) == sum(r['no_core'] for r in meta['records'])
    load = a[..., FIELDS.index('gt_foreground')]
    order = np.argsort(load, axis=1, kind='stable')
    top = np.zeros_like(load, dtype=bool)
    np.put_along_axis(top, order[:, -Q // 5:], True, axis=1)
    bins = np.stack([query_report.deciles(row) for row in load])
    masks = (np.ones_like(top), top, (bins >= 1) & (bins <= 2), bins >= 9, load == 0)
    arrays['query_sums'] = np.stack([a[mask].sum(0, dtype=np.float64) for mask in masks])
    arrays['query_numbers'] = np.array([mask.sum() for mask in masks], dtype=np.int64)
    arrays['field_sums'] = sums
    arrays['class_stats'] = arrays['strata'].sum((0, 2, 3))
    arrays.pop('metrics')
    return meta, arrays, metric


def add(target, row):
    if target is None:
        return {k: v.copy() for k, v in row.items()}
    assert target.keys() == row.keys()
    for key, value in row.items():
        target[key] += value
    return target


def query_metrics(sums, number, full):
    f = lambda name: float(sums[FIELDS.index(name)])
    fg, holes = f('gt_foreground'), f('gt_no_core')
    return dict(queries=int(number), foreground=int(fg), holes=int(holes),
        foreground_share=100 * div(fg, full['foreground']),
        hole_share=100 * div(holes, full['holes']),
        hole_rate=100 * div(holes, fg), FN_rate=100 * div(f('gt_miss'), fg),
        own_core=100 * div(f('gt_own_core'), fg),
        neighbor_only_core=100 * div(f('gt_neighbor_only_core'), fg),
        core_gs_contact=100 * div(f('core_gs_touches_fg'), number * CHILDREN))


def confidence_intervals(before, after):
    assert len(before) == len(after) and len(before) > 0
    n = len(before)
    rng = np.random.default_rng(928)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=1000)
    def sampled(rows):
        cm = (weights @ np.stack([a['confusion'].reshape(-1) for a in rows])).reshape(-1, 18, 18)
        result = cm_metrics(cm)
        fs = weights @ np.stack([a['field_sums'] for a in rows])
        fg, holes = fs[:, FIELDS.index('gt_foreground')], fs[:, FIELDS.index('gt_no_core')]
        result['core_coverage'] = 100 * (1 - holes / fg)
        result['mean_core_count'] = fs[:, FIELDS.index('gt_core_count_sum')] / fg
        cs = (weights @ np.stack([a['class_stats'].reshape(-1) for a in rows])).reshape(-1, 17, 5)
        gt, no_core = cs[..., 0], cs[..., 1]
        rates = np.divide(no_core, gt, out=np.zeros_like(no_core, dtype=float), where=gt > 0)
        result['macro_core_coverage'] = 100 * (1 - rates.mean(-1))
        other = np.arange(17) != 11
        result['core_coverage_without_driveable'] = 100 * (1 - no_core[:, other].sum(-1) / gt[:, other].sum(-1))
        return result
    b, a = sampled(before), sampled(after)
    keys = ('mIoU', 'IoU', 'FN_rate', 'FP_over_empty', 'core_coverage', 'mean_core_count',
            'macro_core_coverage', 'core_coverage_without_driveable')
    return {k: np.quantile(a[k] - b[k], [.025, .975]).tolist() for k in keys}


def csv_write(path, rows):
    if rows:
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def format_table(before, after, items):
    rows = ['| 指标 | 微调前 | 微调后 | 变化 |', '| --- | ---: | ---: | ---: |']
    for key, label, digits in items:
        b, a = before[key], after[key]
        rows.append(f'| {label} | {b:.{digits}f} | {a:.{digits}f} | {a-b:+.{digits}f} |')
    return rows


def plot_comparison(results):
    if 'val' not in results or results['val']['status'] != 'complete':
        return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False})
    folder = ROOT / 'analysis/figures/camera_set_finetune'
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    splits = [s for s in ('train', 'val') if s in results and results[s]['status'] == 'complete']
    settings = (
        ('hole_rate', 'GT foreground without a 1-sigma core', 'GT foreground (%)'),
        ('core_gs_contact', 'GS cores touching visible foreground', 'All predicted GS (%)'),
        ('center_visible_foreground', 'GS centers in visible foreground', 'All predicted GS (%)'),
    )
    x = np.arange(len(splits))
    for axis, (key, title, ylabel) in zip(axes, settings):
        highest = 0
        for j, (tag, label, color) in enumerate((('before', 'Before', '#4477aa'),
                                                ('after', 'Camera-filtered Set', '#cc6677'))):
            values = [results[s]['models'][tag][key] for s in splits]
            bars = axis.bar(x + (j - .5) * .34, values, width=.34, label=label, color=color)
            axis.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
            highest = max(highest, max(values))
        axis.set_xticks(x, ['Train' if s == 'train' else 'Validation' for s in splits])
        axis.set_ylim(0, highest * (1.40 if key == 'hole_rate' else 1.23))
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Same architecture and final budget: 76,800 Gaussians', fontsize=12)
    fig.savefig(folder / 'coverage_comparison.png', dpi=180)
    fig.savefig(folder / 'coverage_comparison.pdf')
    plt.close(fig)
    return True


def report(partial=False):
    AGG.mkdir(parents=True, exist_ok=True)
    protocol = read(RUN / 'protocol.json')
    baseline_protocol = read(BASELINE / 'protocol.json')
    results, scene_rows, query_rows, class_rows = {}, [], [], []
    for split in ('val', 'train'):
        names = sorted(p.stem for p in (RUN / 'scenes' / split).glob('*.json'))
        if not names:
            continue
        before_total = after_total = None
        per_model = {'before': [], 'after': []}
        metadata, frame_count = [], 0
        for name in names:
            mb, b, bm = load_scene(BASELINE, split, name)
            ma, a, am = load_scene(RUN, split, name)
            assert mb['indices'] == ma['indices']
            assert mb['checkpoint_sha256'] == baseline_protocol['checkpoint_sha256']
            assert ma['checkpoint_sha256'] == protocol['checkpoint_sha256']
            assert [r['sample_token'] for r in mb['records']] == [r['sample_token'] for r in ma['records']]
            np.testing.assert_array_equal(b['confusion'].sum(1), a['confusion'].sum(1))
            np.testing.assert_array_equal(b['strata'][..., 0].sum((0, 2, 3)), a['strata'][..., 0].sum((0, 2, 3)))
            before_total, after_total = add(before_total, b), add(after_total, a)
            # Keep only small arrays needed for the paired bootstrap.
            for tag, values, metrics in (('before', b, bm), ('after', a, am)):
                per_model[tag].append({k: values[k] for k in ('confusion', 'field_sums', 'class_stats')})
                scene_rows.append(dict(split=split, scene=ma['scene_name'], model=tag, frames=ma['frames'], **metrics))
            metadata.append((bm, am))
            frame_count += ma['frames']
        expected = protocol['expected'][split]
        complete = len(names) == expected['scenes'] and frame_count == expected['frames']
        models, groups, standardized = {}, {}, {}
        for tag, total in (('before', before_total), ('after', after_total)):
            n_queries = frame_count * Q
            models[tag] = vector_metrics(total['field_sums'], n_queries, total['group_counts'],
                total['attribute_sums'], total['flag_counts'], total['confusion'])
            cs = total['class_stats']
            other = np.arange(17) != 11
            models[tag].update(
                macro_core_coverage=100 * (1 - np.mean(cs[:, 1] / cs[:, 0])),
                core_coverage_without_driveable=100 * (1 - div(cs[other, 1].sum(), cs[other, 0].sum())),
                driveable_core_coverage=100 * (1 - div(cs[11, 1], cs[11, 0])),
                FN_without_driveable=100 * div(cs[other, 2].sum(), cs[other, 0].sum()))
            groups[tag] = {}
            for j, group in enumerate(QUERY_GROUPS):
                gm = query_metrics(total['query_sums'][j], total['query_numbers'][j], models[tag])
                groups[tag][group] = gm
                query_rows.append(dict(split=split, model=tag, group=group, **gm))
            std = query_report.standardized(total['strata'])
            standardized[tag] = dict(
                low_hole_rate=100 * std['low']['hole_rate'],
                high_hole_rate=100 * std['high']['hole_rate'],
                high_minus_low=100 * (std['high']['hole_rate'] - std['low']['hole_rate']),
                retained_foreground_percent=100 * std['retained_foreground_fraction'])
            np.savez_compressed(AGG / f'{tag}_{split}.npz', **total)
        for c, label in enumerate(CLASSES):
            row = dict(split=split, class_id=c, class_name=label)
            for tag, total in (('before', before_total), ('after', after_total)):
                cm = total['confusion']
                tp = int(cm[c, c]); gt = int(cm[c].sum()); pred = int(cm[:, c].sum())
                s = total['strata'][:, c].sum((0, 1, 2))
                assert gt == int(s[0])
                row.update({tag + '_GT': gt, tag + '_IoU': 100 * div(tp, gt + pred - tp),
                    tag + '_core_coverage': 100 * (1 - div(s[1], s[0])),
                    tag + '_FN_rate': 100 * div(s[2], s[0])})
            for key in ('IoU', 'core_coverage', 'FN_rate'):
                row['delta_' + key] = row['after_' + key] - row['before_' + key]
            class_rows.append(row)
        delta = {k: models['after'][k] - models['before'][k] for k in models['before']}
        if complete:
            historical = read(BASELINE / 'aggregate/diagnosis_summary.json')['statistics'][split]['metrics']
            historical_query = read(BASELINE / 'aggregate/summary.json')['summary'][split]
            for key, old_key in (('mIoU', 'miou'), ('IoU', 'iou'), ('FN_rate', 'fn_rate'),
                    ('FP_over_empty', 'fp_rate_empty')):
                np.testing.assert_allclose(models['before'][key], 100 * historical[old_key], rtol=0, atol=1e-10)
            np.testing.assert_allclose(models['before']['hole_rate'],
                100 * historical_query['hole_rate'], rtol=0, atol=1e-10)
            np.testing.assert_allclose(models['before']['mean_scale'],
                historical_query['scale_mean'], rtol=0, atol=1e-10)
        signs = {k: dict(increased=sum(a[k] > b[k] for b, a in metadata),
            decreased=sum(a[k] < b[k] for b, a in metadata),
            equal=sum(a[k] == b[k] for b, a in metadata))
            for k in ('mIoU', 'IoU', 'core_coverage', 'FN_rate', 'center_visible_foreground')}
        results[split] = dict(status='complete' if complete else 'partial', frames=frame_count,
            scenes=len(names), models=models, delta=delta, query_groups=groups,
            standardized_low_high=standardized,
            paired_scene_bootstrap_95=confidence_intervals(per_model['before'], per_model['after']),
            scene_sign_counts=signs)
        road_fraction = div(before_total['class_stats'][11, 0], models['before']['foreground'])
        results[split]['coverage_change_decomposition_pp'] = dict(
            driveable_GT_fraction=road_fraction,
            driveable=road_fraction * delta['driveable_core_coverage'],
            other_16_classes=(1 - road_fraction) * delta['core_coverage_without_driveable'])
        np.testing.assert_allclose(
            sum(results[split]['coverage_change_decomposition_pp'][k] for k in ('driveable', 'other_16_classes')),
            delta['core_coverage'], rtol=0, atol=1e-10)
        print(json.dumps(dict(split=split, frames=frame_count, models=models), ensure_ascii=False), flush=True)
    complete = len(results) == 2 and all(r['status'] == 'complete' for r in results.values())
    if not partial:
        assert complete, {s: (r['scenes'], r['frames']) for s, r in results.items()}
    native_text = (RUN.parent / '20261024_215128.log').read_text()
    native_match = re.findall(r'mIoU:\s*([\d.]+), iou2:\s*([\d.]+)', native_text)
    assert native_match
    native_miou, native_iou = map(float, native_match[-1])
    if 'val' in results and results['val']['status'] == 'complete':
        actual = results['val']['models']['after']
        assert abs(actual['mIoU'] - native_miou) < .0051, (actual['mIoU'], native_miou)
        assert abs(actual['IoU'] - native_iou) < .0051, (actual['IoU'], native_iou)
    payload = dict(status='complete' if complete else 'partial', splits=results,
        baseline_checkpoint_sha256=baseline_protocol['checkpoint_sha256'],
        checkpoint_sha256=protocol['checkpoint_sha256'], native_eval_mIoU=native_miou,
        native_eval_IoU=native_iou, report_source_sha256=sha(Path(__file__)),
        sampling='None; reports compare identical completed scenes and frames',
        causal_limit='Fine-tune vs initialization; no matched original-loss continuation')
    write(AGG / 'comparison_summary.json', payload)
    csv_write(AGG / 'scene_comparison.csv', scene_rows)
    csv_write(AGG / 'query_groups_comparison.csv', query_rows)
    csv_write(AGG / 'class_comparison.csv', class_rows)
    figure = plot_comparison(results)
    validation_complete = 'val' in results and results['val']['status'] == 'complete'
    if validation_complete and not complete:
        results = {s: r for s, r in results.items() if r['status'] == 'complete'}
        status_line = '**完整验证集统计已完成；完整训练集统计仍在运行。以下只展示已经完整统计的数据划分，训练集结果将在完成后自动补充。**'
    elif complete:
        status_line = '**状态：完整统计已完成。**'
    else:
        status_line = '**状态：统计进行中。以下只比较两份权重已经完成的相同场景，不代表完整数据结果。**'
    lines = ['# Set loss 使用 camera mask 后，GS 覆盖的变化', '',
        status_line, '']
    if 'val' in results:
        v = results['val']; b, a, d = v['models']['before'], v['models']['after'], v['delta']
        coverage_word = '提高' if d['core_coverage'] > 0 else '下降' if d['core_coverage'] < 0 else '不变'
        lines += [f"验证集已完成 **{v['scenes']} 个场景、{v['frames']:,} 帧**。前景 1σ 核心覆盖率从 "
            f"**{b['core_coverage']:.3f}%** 变为 **{a['core_coverage']:.3f}%**，{coverage_word} "
            f"**{abs(d['core_coverage']):.3f} 个百分点**。mIoU 从 {b['mIoU']:.5f}% 变为 "
            f"{a['mIoU']:.5f}%，变化 {d['mIoU']:+.5f} 个百分点。", '']
    lines += ['本轮比较原始 small 权重与完成 1 epoch camera-filtered Set 微调的权重。两者均为 '
        '1,200 个 query、76,800 个最终 GS；模型结构和渲染方式相同。默认统计 camera mask 内有效 GT。', '']
    if figure:
        lines += ['![相同高斯数量下，覆盖缺口、核心接触与中心位置的变化](figures/camera_set_finetune/coverage_comparison.png)', '']
    lines += [
        '## 1. 前景覆盖：有没有覆盖到更多地方', '',
        '核心覆盖率的分母是全部有效 GT 前景；每个体素只计一次，不要求预测类别正确。'
        '平均覆盖数则允许同一个体素被多个 GS 重复覆盖。', '']
    for split, r in results.items():
        label = '验证集' if split == 'val' else '训练集'
        lines += [f"### {label}：{r['scenes']} 场景、{r['frames']:,} 帧", '']
        lines += format_table(r['models']['before'], r['models']['after'], [
            ('core_coverage', '前景至少有一个 1σ 核心覆盖（%）', 4),
            ('macro_core_coverage', '17 类等权平均核心覆盖（%）', 4),
            ('core_coverage_without_driveable', '去除 driveable_surface 后的核心覆盖（%）', 4),
            ('driveable_core_coverage', 'driveable_surface 核心覆盖（%）', 4),
            ('hole_rate', '前景没有 1σ 核心覆盖（%）', 4),
            ('mean_core_count', '每个前景体素的平均 1σ 覆盖数', 4),
            ('core_gs_contact', '核心接触可见前景的 GS／全部 GS（%）', 4),
            ('native_gs_contact', '原生范围接触可见前景的 GS／全部 GS（%）', 4),
            ('mean_coarse_distance', '前景到第五级最近中心的平均距离（m）', 4)])
        lines += ['']
    lines += ['1σ 使用旋转各向异性高斯的 Mahalanobis 距离；沿用历史实现的原生候选范围与正 FP32 核贡献条件。'
        '原生渲染范围是按 3 倍最大轴尺度构造的候选盒，和严格 3σ 椭球不同。', '',
        '## 2. 高斯位置和尺度：改善来自哪里', '']
    for split, r in results.items():
        lines += [f"### {'验证集' if split == 'val' else '训练集'}", '']
        lines += format_table(r['models']['before'], r['models']['after'], [
            ('center_visible_foreground', '中心位于可见前景／全部 GS（%）', 4),
            ('center_visible_empty', '中心位于可见 empty／全部 GS（%）', 4),
            ('center_other', '中心没有有效 camera 标签／全部 GS（%）', 4),
            ('pure_empty_camera_3sigma', 'camera 3σ 支持纯空／全部 GS（%）', 4),
            ('mean_scale', '三轴尺度均值（m）', 5),
            ('scale_any_ge_039', '至少一轴尺度 ≥0.39 m 的 GS（%）', 4),
            ('mean_opacity', '不透明度均值', 5),
            ('mean_semantic_amplitude', '17 类 softplus 语义值之和的均值', 5),
            ('visible_empty_high_score_opacity', '可见 empty 中心且分数、opacity 均 >0.9／全部 GS（%）', 4)])
        lines += ['']
    lines += ['“中心没有有效 camera 标签”包括 mask 外、无效 GT 或网格外等情况，不能全部视为无效 GS。'
        '“camera 3σ 支持纯空”只对可见且有有效标签的支持作判断，不代表不可见区域也被确认是空。'
        '高分使用最大原始前景 logit 的 sigmoid，是诊断分数，不是校准后的占有概率。', '',
        '## 3. query 负载压力是否缓解', '',
        '最高 20% 指每帧全部 1,200 个 query 中负载最大的 240 个。每个 query 仍生成 64 个 GS，'
        '所以这一组始终只占 20% 的 GS 数量。', '',
        '| 集合／权重 | 高负载 20% 的前景份额 | 高负载 20% 的缺口份额 | 该组前景缺口率 |',
        '| --- | ---: | ---: | ---: |']
    for split, r in results.items():
        for tag, name in (('before', '微调前'), ('after', '微调后')):
            q = r['query_groups'][tag]['all_query_top20']
            lines.append(f"| {'验证集' if split == 'val' else '训练集'}／{name} | {q['foreground_share']:.3f}% | {q['hole_share']:.3f}% | {q['hole_rate']:.3f}% |")
    lines += ['', '### 对齐类别、距离和粗层定位误差后的高低负载差', '',
        '| 集合／权重 | 低负载缺口率 | 高负载缺口率 | 高−低（百分点） |',
        '| --- | ---: | ---: | ---: |']
    for split, r in results.items():
        for tag, name in (('before', '微调前'), ('after', '微调后')):
            s = r['standardized_low_high'][tag]
            lines.append(f"| {'验证集' if split == 'val' else '训练集'}／{name} | {s['low_hole_rate']:.3f}% | {s['high_hole_rate']:.3f}% | {s['high_minus_low']:+.3f} |")
    lines += ['', '负载按第五级中心最近邻划分，不是 GS 的实际渲染贡献。每份权重都重新划分、重新排名，'
        '高低组不保证包含相同 query 或空间区域；对齐也使用各自的定位误差分组。'
        '因此这些表用于检查负载与缺口的关系是否仍在，不能单独证明固定数量是缺口的原因。', '',
        '## 4. 最终预测有没有改善', '']
    for split, r in results.items():
        lines += [f"### {'验证集' if split == 'val' else '训练集'}", '']
        lines += format_table(r['models']['before'], r['models']['after'], [
            ('mIoU', '17 类 mIoU（%）', 5), ('IoU', '二值占有 IoU（%）', 5),
            ('FN_rate', '漏检：FN／GT 前景（%）', 4),
            ('FN_without_driveable', '去除 driveable_surface 后的漏检率（%）', 4),
            ('FP_over_empty', '误占用：FP／GT empty（%）', 4),
            ('FP_over_GTfg', 'FP／GT 前景（%）', 4),
            ('FP_over_predfg', 'FP／预测前景（%）', 4),
            ('wrong_fg_rate', 'GT 前景被分为其他前景类别（%）', 4),
            ('detected_semantic_accuracy', '已检出前景的语义正确率（%）', 4)])
        lines += ['']
    lines += [f'训练日志原生完整验证记录为 mIoU {native_miou:.5f}%、二值 IoU {native_iou:.5f}%。'
        '本报告使用 int64 混淆矩阵独立累加；与原生日志的低精度累计可能有极小差异。', '',
        '## 5. 场景差异和结论边界', '',
        '| 集合 | 指标变化 | 配对场景 bootstrap 95% 区间 |', '| --- | --- | --- |']
    for split, r in results.items():
        for key, label in (('core_coverage', '核心覆盖率，百分点'), ('mIoU', 'mIoU，百分点'),
                ('macro_core_coverage', '17 类等权核心覆盖，百分点'),
                ('core_coverage_without_driveable', '去除道路后的核心覆盖，百分点'),
                ('IoU', '二值 IoU，百分点'), ('FN_rate', '漏检率，百分点')):
            lo, hi = r['paired_scene_bootstrap_95'][key]
            lines.append(f"| {'验证集' if split == 'val' else '训练集'} | {label} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['']
    for split, r in results.items():
        s = r['scene_sign_counts']
        lines.append(f"{'验证集' if split == 'val' else '训练集'}中，核心覆盖率提高／下降的场景为 "
            f"{s['core_coverage']['increased']}／{s['core_coverage']['decreased']}；mIoU 提高／下降的场景为 "
            f"{s['mIoU']['increased']}／{s['mIoU']['decreased']}。")
    lines += ['', '区间对完整场景重采样，并重新累计混淆矩阵和覆盖分母；它不包含训练随机性。'
        '本轮比较的是原始权重与这次微调后的权重，没有同预算的原始目标继续训练对照，'
        '因此不能把全部变化单独归因于 camera 过滤。', '']
    if 'val' in results and results['val']['status'] == 'complete':
        r = results['val']; d = r['delta']; s = r['standardized_low_high']['after']
        if d['core_coverage'] > 0:
            lines += [f"**就这份微调权重而言，可见前景的核心覆盖确实扩大了**：无核心覆盖体素净减少 "
                f"{int(-d['holes']):,} 个。这是两次总计数的净差，不是逐体素追踪的救回数量。", '']
        else:
            lines += ['**就这份微调权重而言，可见前景的核心覆盖没有扩大。**'
                '需要区分性能变化、局部重复覆盖和全局覆盖并集。', '']
        valid_classes = [x for x in class_rows if x['split'] == 'val']
        improved = sum(x['delta_core_coverage'] > 0 for x in valid_classes)
        lines += [f"逐类别看，只有 **{improved}/17 类**的核心覆盖率提高；17 类等权平均变化为 "
            f"{d['macro_core_coverage']:+.4f} 个百分点，去除 driveable_surface 后变化为 "
            f"{d['core_coverage_without_driveable']:+.4f} 个百分点。"
            '全局按体素加权的覆盖收益与各类覆盖普遍改善，需要分开判断。', '']
        decomp = r['coverage_change_decomposition_pp']
        lines += [f"driveable_surface 占验证集有效前景的 {100*decomp['driveable_GT_fraction']:.3f}%。"
            f"它单独贡献全局核心覆盖变化 {decomp['driveable']:+.4f} 个百分点，"
            f"其他 16 类合计贡献 {decomp['other_16_classes']:+.4f} 个百分点；二者相加为全局的 "
            f"{d['core_coverage']:+.4f} 个百分点。", '']
        if s['high_minus_low'] > 0:
            lines += [f"微调后，对齐高负载组仍比低负载组多 {s['high_minus_low']:.3f} 个百分点的核心缺口。"
                '预算方向仍有可研究的残余现象，但应以这份新权重重新建立基线，'
                '不能继续沿用微调前的缺口幅度估计方法空间。', '']
        if d['core_coverage'] > 0 and d['mean_scale'] <= 0:
            lines += ['平均尺度没有增大，而核心覆盖和前景接触仍有改善，这不支持用“整体放大 GS”解释全部覆盖收益。'
                '尺度均值不能排除局部尺度、旋转变化的作用，中心位置和局部几何变化应结合表中各项判断。', '']
    lines += ['camera 过滤同时改变了 Set 几何匹配、最近 GT 分类标签和相应归一化。'
        '本轮没有单独拆开这些路径，不能仅凭覆盖变化确定是哪一项提供了主要训练作用。', '']
    lines += ['## 6. 逐类别变化（验证集）', '',
        '| 类别 | 原 IoU | 新 IoU | IoU 变化 | 核心覆盖变化 | 漏检率变化 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in class_rows:
        if row['split'] == 'val':
            lines.append(f"| {row['class_name']} | {row['before_IoU']:.3f}% | {row['after_IoU']:.3f}% | "
                f"{row['delta_IoU']:+.3f} | {row['delta_core_coverage']:+.3f} | {row['delta_FN_rate']:+.3f} |")
    lines += ['', '变化列单位为百分点。逐类别覆盖同样以该类全部有效 GT 为分母。', '',
        '## 7. 复核文件', '',
        '- [统计设计](camera_set_finetune_coverage_design.md)',
        '- [完整数值汇总](../work_dirs/small_camera_set_full_1ep/diagnostics/aggregate/comparison_summary.json)',
        '- [逐场景比较](../work_dirs/small_camera_set_full_1ep/diagnostics/aggregate/scene_comparison.csv)',
        '- [query 分组比较](../work_dirs/small_camera_set_full_1ep/diagnostics/aggregate/query_groups_comparison.csv)',
        '- [逐类别比较](../work_dirs/small_camera_set_full_1ep/diagnostics/aggregate/class_comparison.csv)',
        '- [原生路径与统计校验](../work_dirs/small_camera_set_full_1ep/diagnostics/pilot.json)',
        '- [权重与配置协议](../work_dirs/small_camera_set_full_1ep/diagnostics/comparison_protocol.json)', '']
    if complete:
        lines += ['- [全量数据审计](../work_dirs/small_camera_set_full_1ep/diagnostics/audit.json)',
            '- [微调前后配对校验](../work_dirs/small_camera_set_full_1ep/diagnostics/pairing_audit.json)', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Comparison report written: {REPORT}; complete={complete}', flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    report(args.partial)
