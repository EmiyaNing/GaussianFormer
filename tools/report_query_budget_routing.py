"""CPU-only, count-audited aggregation of frozen query-routing diagnostics.

python3.10 tools/report_query_budget_routing.py --partial
python3.10 tools/report_query_budget_routing.py

The default requires completed train and validation splits. ``--partial`` reads
only atomically completed scene JSON/NPZ pairs and labels every incomplete split
as partial. A full result additionally requires its split-completion marker,
expected scene/frame counts, and an exact partition of frame indices.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.report_camera_set_finetune import cm_metrics
from tools.report_query_budget import deciles

DEFAULT_RUN = ROOT / 'work_dirs/small_query_budget_camera_full_1ep/routing_diagnosis'
DEFAULT_REPORT = ROOT / 'analysis/query_budget_routing_results.md'
BASELINE = ROOT / 'work_dirs/small_camera_set_full_1ep/diagnostics/aggregate/comparison_summary.json'
MODES = ('hard_raw', 'soft_all', 'hard_gated', 'uniform64', 'opacity_only')
MODE_LABELS = dict(hard_raw='当前硬路由，原 opacity', soft_all='全部候选，软门控 opacity',
                   hard_gated='同一硬路由，保留软门控 opacity', uniform64='每 query 保留 64 个',
                   opacity_only='仅按 opacity 全局保留 N 个')
VARIANT_LABELS = dict(current_hard='当前硬路由', uniform64_per_query='每 query 保留 64 个',
                      all_candidates='全部候选（原 opacity）')
CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
           'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')


def class_rows(results):
    rows = []
    for split, values in results['splits'].items():
        for mode, metrics in values['render_modes'].items():
            cm = np.asarray(metrics['confusion'], dtype=np.int64)
            for i, name in enumerate(CLASSES):
                tp, gt, pred = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
                rows.append(dict(split=split, mode=mode, class_id=i, category=name,
                    GT=gt, TP=tp, predicted=pred, FN_to_empty=int(cm[i, 17]),
                    FN_to_other_foreground=gt - tp - int(cm[i, 17]),
                    FP_from_empty=int(cm[17, i]),
                    FP_from_other_foreground=pred - tp - int(cm[17, i]),
                    IoU_pct=100 * tp / (gt + pred - tp) if gt + pred - tp else None,
                    semantic_recall_pct=100 * tp / gt if gt else None,
                    miss_rate_pct=100 * int(cm[i, 17]) / gt if gt else None))
    return rows


def read(path):
    return json.loads(Path(path).read_text())


def ratio(a, b):
    return float(a / b) if b else None


def num(value, digits=3):
    return '—' if value is None else f'{value:,.{digits}f}'


def percent(value):
    return '—' if value is None else f'{100 * value:.3f}%'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['split', 'scene'])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def add_number(row, key, value):
    row[key] = row.get(key, 0) + value


def moments_add(target, key, value):
    if value is None:
        return
    value = float(value)
    if not np.isfinite(value):
        raise AssertionError(f'Nonfinite scalar: {key}')
    entry = target.setdefault(key, dict(sum=0., count=0, minimum=value, maximum=value))
    entry['sum'] += value
    entry['count'] += 1
    entry['minimum'] = min(entry['minimum'], value)
    entry['maximum'] = max(entry['maximum'], value)


def correlations_add(target, prefix, values):
    for group, row in values.items():
        for metric in ('pearson', 'spearman'):
            moments_add(target, f'{prefix}/{group}/{metric}', row[metric])


def empty_totals():
    return dict(frames=0, scenes=0, counts={}, gates={}, frame_statistics={},
                variants={}, confusions={}, changes={}, deciles={}, hard_boundary_gates=[])


def merge_numeric_dict(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            merge_numeric_dict(target.setdefault(key, {}), value)
        elif isinstance(value, np.ndarray):
            target[key] = target.get(key, np.zeros_like(value)) + value
        else:
            add_number(target, key, value)


def merge_totals(target, source):
    for key in ('frames', 'scenes'):
        target[key] += source[key]
    for key in ('counts', 'gates', 'variants', 'confusions', 'changes', 'deciles'):
        merge_numeric_dict(target[key], source[key])
    target['hard_boundary_gates'].extend(source['hard_boundary_gates'])
    for name, value in source['frame_statistics'].items():
        if name not in target['frame_statistics']:
            target['frame_statistics'][name] = value.copy()
        else:
            dest = target['frame_statistics'][name]
            dest['sum'] += value['sum']
            dest['count'] += value['count']
            dest['minimum'] = min(dest['minimum'], value['minimum'])
            dest['maximum'] = max(dest['maximum'], value['maximum'])


def load_scene(path, protocol):
    meta = read(path)
    if meta['status'] != 'complete' or meta['checkpoint_sha256'] != protocol['checkpoint_sha256']:
        raise AssertionError(f'Invalid scene record {path}')
    indices = meta['indices']
    records = meta['records']
    if [row['index'] for row in records] != indices or len(set(indices)) != len(indices):
        raise AssertionError(f'Frame indices are not a unique ordered partition in {path}')
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    load = arrays['gt_load']
    frames, queries = load.shape
    if frames != len(records) or frames == 0:
        raise AssertionError('Scene frame count mismatch')
    for key, value in arrays.items():
        if key.startswith('confusion_'):
            if value.shape != (18, 18) or np.any(value < 0):
                raise AssertionError(f'Invalid confusion matrix {key}')
        elif value.shape != load.shape or not np.isfinite(value).all():
            raise AssertionError(f'Invalid query array {key}')
    np.testing.assert_array_equal(arrays['query_id'], np.broadcast_to(np.arange(queries), load.shape))
    np.testing.assert_array_equal(arrays['selected_count'], arrays['current_hard_selected_count'])
    top = arrays['gt_high_load_top20'].astype(bool)
    expected_top = np.zeros_like(top)
    np.put_along_axis(expected_top, np.argsort(-load, axis=1, kind='stable')[:, :max(queries // 5, 1)],
                      True, axis=1)
    np.testing.assert_array_equal(top, expected_top)
    positive = load > 0
    total = empty_totals()
    total['frames'], total['scenes'] = frames, 1
    counts = total['counts']
    counts.update(queries=frames * queries, foreground=int(load.sum()), positive_load_queries=int(positive.sum()),
                  zero_load_queries=int((~positive).sum()), high_load_queries=int(top.sum()),
                  high_load_foreground=int(load[top].sum()), soft_mass=float(arrays['soft_mass'].sum(dtype=np.float64)),
                  high_load_soft_mass=float(arrays['soft_mass'][top].sum(dtype=np.float64)),
                  coarse_valid_queries=int(arrays['coarse_query_valid'].sum()),
                  final_valid_queries=int(arrays['final_query_valid'].sum()),
                  gt_assigned_to_invalid_coarse_query=int(load[~arrays['coarse_query_valid']].sum()),
                  gt_assigned_to_invalid_final_query=int(load[~arrays['final_query_valid']].sum()),
                  coarse_distance_sum=float(arrays['gt_coarse_distance_sum'].sum()),
                  hard_gt_miss=int(arrays['hard_gt_miss'].sum()),
                  hard_gt_wrong_class=int(arrays['hard_gt_wrong_class'].sum()))
    bins = np.stack([deciles(row) for row in load])
    variants = tuple(records[0]['query']['variants'])
    modes = tuple(records[0]['changes'])
    if set(modes) != (set(MODES) if meta['split'] == 'val' else {'hard_raw'}):
        raise AssertionError('Unexpected render-mode coverage')
    for mode in modes:
        cm = arrays['confusion_' + mode].astype(np.int64)
        if int(cm[:17].sum()) != counts['foreground']:
            raise AssertionError('GT-to-query counts do not conserve confusion-matrix GT foreground')
        np.testing.assert_array_equal(cm.sum(1), arrays['confusion_hard_raw'].sum(1))
        total['confusions'][mode] = cm
    hard = total['confusions']['hard_raw']
    if int(hard[:17, 17].sum()) != counts['hard_gt_miss']:
        raise AssertionError('Query FN counts disagree with confusion matrix')
    if int(hard[:17, :17].sum() - np.trace(hard[:17, :17])) != counts['hard_gt_wrong_class']:
        raise AssertionError('Query semantic-error counts disagree with confusion matrix')

    for f, row in enumerate(records):
        q, gates = row['query'], row['gates']
        if set(q['variants']) != set(variants) or set(row['changes']) != set(modes):
            raise AssertionError('Modes changed within a scene')
        if q['foreground_voxels'] != int(load[f].sum()) or q['queries'] != queries:
            raise AssertionError('JSON and NPZ GT counts disagree')
        all_count = gates['all']['count']
        keep_count, drop_count = gates['kept']['count'], gates['dropped']['count']
        if all_count != keep_count + drop_count or all_count != queries * q['children']:
            raise AssertionError('Candidate partition fails conservation')
        if keep_count != int(arrays['selected_count'][f].sum()):
            raise AssertionError('Gate and per-query hard budgets differ')
        for key in ('gate_sum', 'binary_sum', 'intermediate_count'):
            if not np.isclose(gates['all'][key], gates['kept'][key] + gates['dropped'][key], rtol=1e-10, atol=1e-6):
                raise AssertionError(f'Gate partition fails {key} conservation')
        for group in ('all', 'kept', 'dropped'):
            g = gates[group]
            if sum(g['histogram']) != g['count']:
                raise AssertionError('Gate histogram counts do not sum to candidates')
            dest = total['gates'].setdefault(group, dict(count=0, gate_sum=0., binary_sum=0.,
                intermediate_count=0, histogram=np.zeros(20, np.int64)))
            for key in ('count', 'gate_sum', 'binary_sum', 'intermediate_count'):
                dest[key] += g[key]
            dest['histogram'] += np.asarray(g['histogram'], dtype=np.int64)
        for name in ('gate_mass_balance_error', 'query_score_mean', 'query_score_std', 'temperature',
                     'soft_threshold', 'hard_boundary_log_priority', 'hard_boundary_gate', 'boundary_priority_gap'):
            moments_add(total['frame_statistics'], name, gates[name])
        total['hard_boundary_gates'].append(float(gates['hard_boundary_gate']))
        correlations_add(total['frame_statistics'], 'load_score', q['gt_load_query_score_correlations'])
        correlations_add(total['frame_statistics'], 'load_soft_mass', q['gt_load_soft_mass_correlations'])
        for variant in variants:
            correlations_add(total['frame_statistics'], 'load_selected/' + variant,
                             q['variants'][variant]['gt_load_selected_count_correlations'])
        for mode, change in row['changes'].items():
            merge_numeric_dict(total['changes'].setdefault(mode, {}), change)

    for variant in variants:
        get = lambda field: arrays[variant + '_' + field]
        selected = get('selected_count')
        holes, own, neighbor = get('gt_no_core'), get('gt_own_core'), get('gt_neighbor_only_core')
        np.testing.assert_array_equal(holes + own + neighbor, load)
        for f, row in enumerate(records):
            q = row['query']['variants'][variant]
            for key, value in (('selected_count', selected[f].sum()), ('no_core_voxels', holes[f].sum()),
                               ('own_core_voxels', own[f].sum()), ('high_load_selected_count', selected[f, top[f]].sum())):
                if q[key] != int(value):
                    raise AssertionError(f'JSON/NPZ disagreement: {variant}/{key}')
        zero_selected = selected == 0
        v = dict(selected_count=int(selected.sum()), high_load_selected_count=int(selected[top].sum()),
                 zero_load_selected_count=int(selected[~positive].sum()),
                 zero_selected_queries=int(zero_selected.sum()),
                 positive_load_zero_selected_queries=int((positive & zero_selected).sum()),
                 foreground_assigned_to_zero_selected_queries=int(load[zero_selected].sum()),
                 no_core_voxels=int(holes.sum()), own_core_voxels=int(own.sum()),
                 neighbor_only_core_voxels=int(neighbor.sum()), high_load_no_core_voxels=int(holes[top].sum()),
                 core_count_sum=int(get('gt_core_count_sum').sum()),
                 core_gs_touches_fg=int(get('core_gs_touches_fg').sum()))
        total['variants'][variant] = v
        dec = {}
        for d in range(11):
            mask = bins == d
            dec[str(d)] = dict(queries=int(mask.sum()), foreground=int(load[mask].sum()),
                selected_count=int(selected[mask].sum()), soft_mass=float(arrays['soft_mass'][mask].sum(dtype=np.float64)),
                score_sum=float(arrays['query_score'][mask].sum(dtype=np.float64)),
                no_core_voxels=int(holes[mask].sum()), own_core_voxels=int(own[mask].sum()),
                neighbor_only_core_voxels=int(neighbor[mask].sum()),
                zero_selected_queries=int((zero_selected & mask).sum()), hard_gt_miss=int(arrays['hard_gt_miss'][mask].sum()))
        total['deciles'][variant] = dec
    return meta, total


def finalize(total):
    c = total['counts']
    if not total['frames']:
        return dict(frames=0, scenes=0, counts={}, gates={}, variants={}, render_modes={}, frame_statistics={})
    result = dict(frames=total['frames'], scenes=total['scenes'], counts=c.copy(), gates={},
                  variants={}, render_modes={}, changes=total['changes'], frame_statistics={})
    result['query'] = dict(high_load_foreground_share=ratio(c['high_load_foreground'], c['foreground']),
        high_load_soft_mass_share=ratio(c['high_load_soft_mass'], c['soft_mass']),
        zero_load_query_fraction=ratio(c['zero_load_queries'], c['queries']),
        mean_coarse_distance=ratio(c['coarse_distance_sum'], c['foreground']))
    for group, row in total['gates'].items():
        result['gates'][group] = {**row, 'histogram': row['histogram'].tolist(),
            'gate_mean': ratio(row['gate_sum'], row['count']),
            'intermediate_fraction': ratio(row['intermediate_count'], row['count']),
            'binary_mean': ratio(row['binary_sum'], row['count']),
            'gate_ge_05_count': int(row['histogram'][10:].sum()),
            'gate_ge_05_fraction': ratio(row['histogram'][10:].sum(), row['count'])}
    g = result['gates']
    result['gate_summary'] = dict(histogram_bin_edges=np.linspace(0, 1, 21).tolist(),
        dropped_candidate_fraction=ratio(g['dropped']['count'], g['all']['count']),
        dropped_soft_mass_over_budget=ratio(g['dropped']['gate_sum'], g['kept']['count']),
        kept_restored_gate_mass_over_budget=ratio(g['kept']['count'] - g['kept']['gate_sum'], g['kept']['count']),
        all_soft_mass_over_budget=ratio(g['all']['gate_sum'], g['kept']['count']),
        hard_boundary_gate_quantile_levels=[0., .05, .25, .5, .75, .95, 1.],
        hard_boundary_gate_quantiles=np.quantile(total['hard_boundary_gates'], [0., .05, .25, .5, .75, .95, 1.]).tolist())
    for name, row in total['variants'].items():
        result['variants'][name] = {**row,
            'high_load_budget_share': ratio(row['high_load_selected_count'], row['selected_count']),
            'zero_load_budget_share': ratio(row['zero_load_selected_count'], row['selected_count']),
            'zero_selected_query_fraction': ratio(row['zero_selected_queries'], c['queries']),
            'positive_load_zero_selected_query_fraction': ratio(row['positive_load_zero_selected_queries'], c['positive_load_queries']),
            'foreground_assigned_to_zero_selected_query_share': ratio(row['foreground_assigned_to_zero_selected_queries'], c['foreground']),
            'no_core_rate': ratio(row['no_core_voxels'], c['foreground']),
            'own_core_rate': ratio(row['own_core_voxels'], c['foreground']),
            'neighbor_only_core_rate': ratio(row['neighbor_only_core_voxels'], c['foreground']),
            'mean_core_count': ratio(row['core_count_sum'], c['foreground']),
            'high_load_no_core_rate': ratio(row['high_load_no_core_voxels'], c['high_load_foreground']),
            'high_load_share_of_all_holes': ratio(row['high_load_no_core_voxels'], row['no_core_voxels'])}
    for mode, cm in total['confusions'].items():
        result['render_modes'][mode] = {key: float(value) for key, value in cm_metrics(cm).items()}
        result['render_modes'][mode]['confusion'] = cm.tolist()
        changes = result['changes'][mode]
        hard = total['confusions']['hard_raw']
        fn_delta = int(cm[:17, 17].sum() - hard[:17, 17].sum())
        fp_delta = int(cm[17, :17].sum() - hard[17, :17].sum())
        if fn_delta != changes['new_fn'] - changes['hard_fn_rescued'] or fp_delta != changes['new_fp'] - changes['hard_fp_removed']:
            raise AssertionError('Counterfactual transition counts disagree with confusion deltas')
    for name, values in total['frame_statistics'].items():
        result['frame_statistics'][name] = {**values, 'mean': values['sum'] / values['count']}
    return result


def scene_rows(split, scene, value):
    rows = []
    mapping = dict(hard_raw='current_hard', uniform64='uniform64_per_query')
    for mode, metrics in value['render_modes'].items():
        row = dict(split=split, scene=scene, frames=value['frames'], mode=mode,
                   **{k: v for k, v in metrics.items() if k != 'confusion'})
        row.update(value['query'])
        row.update(gate_middle_fraction=value['gates']['all']['intermediate_fraction'],
                   dropped_middle_fraction=value['gates']['dropped']['intermediate_fraction'],
                   dropped_soft_mass_over_budget=value['gate_summary']['dropped_soft_mass_over_budget'])
        if mode in mapping:
            row.update(value['variants'][mapping[mode]])
        rows.append(row)
    if 'all_candidates' in value['variants']:
        # This is raw-opacity geometric support, not the gated soft_all render.
        rows.append(dict(split=split, scene=scene, frames=value['frames'], mode='geometry_all_candidates',
                         **value['query'], **value['variants']['all_candidates']))
    return rows


def decile_rows(split, scene, total):
    rows = []
    for variant, bins in total['deciles'].items():
        for decile, c in bins.items():
            rows.append(dict(split=split, scene=scene, variant=variant, load_decile=int(decile), **c,
                foreground_share=ratio(c['foreground'], total['counts']['foreground']),
                budget_share=ratio(c['selected_count'], total['variants'][variant]['selected_count']),
                mean_selected_count=ratio(c['selected_count'], c['queries']),
                mean_query_score=ratio(c['score_sum'], c['queries']),
                mean_soft_mass=ratio(c['soft_mass'], c['queries']),
                no_core_rate=ratio(c['no_core_voxels'], c['foreground']),
                own_core_rate=ratio(c['own_core_voxels'], c['foreground'])))
    return rows


def baseline_comparison(split, value, baseline):
    if value['status'] != 'complete' or baseline.get('status') != 'complete':
        return None
    old = baseline.get('splits', {}).get(split, {})
    if old.get('status') != 'complete' or old['frames'] != value['frames'] or old['scenes'] != value['scenes']:
        return None
    previous = old['models']['after']
    now = value['render_modes']['hard_raw']
    if int(previous['foreground']) != value['counts']['foreground'] or int(previous['empty']) != int(now['empty']):
        raise AssertionError('Completed camera baseline and routing evaluation GT counts differ')
    current = {key: now[key] for key in ('mIoU', 'IoU', 'FN_rate', 'FP_over_empty')}
    current['hole_rate'] = 100 * value['variants']['current_hard']['no_core_rate']
    return dict(camera_set=previous, routing=current,
                delta={key: current[key] - previous[key] for key in current},
                camera_set_top20=old['query_groups']['after']['all_query_top20'],
                note='Both complete splits; query ownership is recomputed separately for each checkpoint, not fixed across checkpoints.')


def markdown(results, run):
    lines = ['# Query 软门控与硬路由：冻结权重诊断结果', '',
        f"**统计状态：{'完整训练集和验证集均完成' if results['status'] == 'complete' else '部分结果，完整统计仍在进行'}。**", '',
        f"生成时间：{results['generated_at']}。以下只汇总已经完整落盘的场景；不包含正在处理的半个场景。", '',
        '这份报告回答两个问题：训练结束后有多少门控仍处于中间值，以及最终 GS 配额是否流向 GT 负载较高的 query。'
        '所有比较冻结同一个权重，默认使用 camera mask。验证集比较五种渲染方式；训练集仅统计实际使用的硬路由。', '',
        '| 数据划分 | 已统计场景 / 全部场景 | 已统计帧 / 全部帧 | 状态 |',
        '| --- | ---: | ---: | --- |']
    for split in ('val', 'train'):
        r = results['splits'][split]
        lines.append(f"| {'验证集' if split == 'val' else '训练集'} | {r['scenes']} / {r['expected']['scenes']} | "
                     f"{r['frames']} / {r['expected']['frames']} | {'完整' if r['status'] == 'complete' else '未完成'} |")
    val = results['splits']['val']
    if val['status'] == 'complete':
        hard, soft = val['render_modes']['hard_raw'], val['render_modes']['soft_all']
        vh, vu = val['variants']['current_hard'], val['variants']['uniform64_per_query']
        lines += ['', '**检查结论。**', '',
            f"- 软硬转换是当前验证掉点的主要直接环节：同一权重恢复全部软渲染，mIoU 从 {hard['mIoU']:.3f} 提高到 {soft['mIoU']:.3f}，二值 IoU 从 {hard['IoU']:.3f} 提高到 {soft['IoU']:.3f}。这不增加训练，也不改几何和语义；代价是渲染全部 1.5 倍候选。",
            f"- 配额向高负载组有小幅倾斜：最高负载 20% query 获得 {percent(vh['high_load_budget_share'])} 配额；原固定配额为 20%。但保留数与负载的逐帧平均相关性接近零，尚未形成稳定的按负载分配。",
            f"- 同一批候选中，当前选择的无核心率为 {percent(vh['no_core_rate'])}，低于每 query 固定 64 个的 {percent(vu['no_core_rate'])}，说明选择有局部收益。相对独立 camera-set 基线，整体覆盖和最终指标仍未改善，不能宣布负载问题已解决。", '']
    lines += ['', '**读数约定。** 门控记为 p，原始 opacity 记为 α。模型生成 115,200 个候选，'
        '硬路由保留 76,800 个，固定删除三分之一。“约 59% 的门控处于中间值”不表示删除了 59% 的 GS。'
        '高负载 top 20% 按每帧 GT→第五级中心最近邻归属得到的负载排序，不能与“保留数量最多的 top 20%”混用。', '',
        '所有比例先汇总计数再相除；相关性单独按帧计算后取平均。query 在不同帧之间不按 ID 对齐。', '']
    for split in ('val', 'train'):
        r = results['splits'][split]
        lines += [f"## {'验证集' if split == 'val' else '训练集'}", '']
        if not r['frames']:
            lines += ['尚无完整场景落盘，暂无统计数值。', '']
            continue
        if r['status'] != 'complete':
            lines += [f"**下表仅代表已完成的 {r['scenes']} 个场景、{r['frames']} 帧，不能作为全数据集结论，也不与完整基线直接比较。**", '']
        lines += ['### 门控是否接近硬选择', '',
            '| 候选分组 | 候选数量（跨帧累计） | 平均 p | 中间值比例（0.1 < p < 0.9） | 平均 p(1−p) |',
            '| --- | ---: | ---: | ---: | ---: |']
        for name, label in (('all', '全部'), ('kept', '最终保留'), ('dropped', '最终删除')):
            g = r['gates'][name]
            lines.append(f"| {label} | {g['count']:,} | {num(g['gate_mean'])} | {percent(g['intermediate_fraction'])} | {num(g['binary_mean'])} |")
        g = r['gate_summary']
        lines += ['', f"删除的候选占 {percent(g['dropped_candidate_fraction'])}；这些候选在软渲染中携带的门控总和，"
            f"相当于保留预算 N 的 **{percent(g['dropped_soft_mass_over_budget'])}**。保留候选从 pα 恢复到 α 时增加的门控量为 "
            f"**{percent(g['kept_restored_gate_mass_over_budget'])} × N**。两者近似相等源于 ∑p = N；"
            '它们分布在不同位置，且 opacity、语义和空间核不同，因此不能据此认为渲染贡献相互抵消。', '',
            f"被删除候选中，**{percent(r['gates']['dropped']['gate_ge_05_fraction'])}** 的 p 仍不小于 0.5。"
            f"每帧最后一个保留候选的 p 平均为 **{num(r['frame_statistics']['hard_boundary_gate']['mean'])}**，"
            f"中位数为 **{num(g['hard_boundary_gate_quantiles'][3])}**，5%–95% 分位为 "
            f"**{num(g['hard_boundary_gate_quantiles'][1])}–{num(g['hard_boundary_gate_quantiles'][5])}**。"
            '推理依据排名保留固定数量，硬选择边界并不是 p = 0.5；某候选的 p 超过 0.5，也可能被删除。', '',
            '上述“门控量”是 p 的求和，不是物理占有概率，也不是渲染场总质量。全量直方图保存在 summary.json；'
            '单帧分位数保留在原始场景 JSON，本报告没有把分位数的平均值冒充全局分位数。', '',
            '### 配额是否流向高负载 query', '',
            f"最高负载 20% query 对应 **{percent(r['query']['high_load_foreground_share'])}** 的 GT 前景，"
            f"获得 **{percent(r['query']['high_load_soft_mass_share'])}** 的软门控量。", '',
            '| 几何集合 | 高负载 top 20% 的 GS 配额 | 零负载 query 的配额 | 有负载但零保留的 query / 有负载 query | 前景中分给零保留 query 的比例 |',
            '| --- | ---: | ---: | ---: | ---: |']
        for name, v in r['variants'].items():
            lines.append(f"| {VARIANT_LABELS[name]} | {percent(v['high_load_budget_share'])} | {percent(v['zero_load_budget_share'])} | "
                         f"{percent(v['positive_load_zero_selected_query_fraction'])} | {percent(v['foreground_assigned_to_zero_selected_query_share'])} |")
        lines += ['', '零负载只表示没有 GT 通过第五级最近邻归属分给该 query；它仍可能以最终 GS 覆盖其他 query 负责的前景。'
            '同样，某 query 零保留也不直接意味着其 GT 全部漏检。前景负载占比描述需求集中程度，'
            '不代表最优预算必须与体素数量成比例。', '',
            '| 每帧相关性，取平均 | Pearson | Spearman |', '| --- | ---: | ---: |']
        for prefix, label in (('load_score', 'GT 负载与 query score'), ('load_soft_mass', 'GT 负载与软门控量'),
                              ('load_selected/current_hard', 'GT 负载与硬保留数量')):
            stat = r['frame_statistics']
            p = stat.get(prefix + '/all_queries/pearson', {}).get('mean')
            s = stat.get(prefix + '/all_queries/spearman', {}).get('mean')
            lines.append(f'| {label} | {num(p)} | {num(s)} |')
        lines += ['', '### 前景是否得到高斯核心覆盖', '',
            ('1σ 核心覆盖按空间几何统计，不要求类别一致。三种集合使用同一权重、同一 GT→query 归属；'
             '“全部候选”采用原 opacity，是几何支持的上界检查，不能直接当作软渲染的预测结果。'
             if split == 'val' else '训练集仅统计当前硬选择的 1σ 几何核心覆盖，不要求类别一致。'), '',
            '| 几何集合 | 无任何核心覆盖 / GT 前景 | 被所属 query 核心覆盖 / GT 前景 | 每前景体素平均核心数 | 高负载区域无核心率 | 高负载区域占全部缺口的比例 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, v in r['variants'].items():
            lines.append(f"| {VARIANT_LABELS[name]} | {percent(v['no_core_rate'])} | {percent(v['own_core_rate'])} | "
                         f"{num(v['mean_core_count'])} | {percent(v['high_load_no_core_rate'])} | {percent(v['high_load_share_of_all_holes'])} |")
        lines += ['', '### 实际 Occupancy 指标', '',
            '| 渲染方式 | mIoU（%） | 二值 IoU（%） | FN / GT 前景（%） | FP / GT empty（%） | FP / GT 前景（%） |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for mode, value in r['render_modes'].items():
            lines.append(f"| {MODE_LABELS[mode]} | {num(value['mIoU'])} | {num(value['IoU'])} | "
                         f"{num(value['FN_rate'])} | {num(value['FP_over_empty'])} | {num(value['FP_over_GTfg'])} |")
        if split == 'val':
            lines += ['', 'hard_gated 与 hard_raw 保留完全相同的 GS，仅改变 opacity；soft_all 与 hard_gated 都使用 pα，'
                '区别是是否删除落选 GS。uniform64 在每个 query 内按 opacity 取前 64 个；opacity_only 则全局排序。'
                '后两者使用原 α。这些是同权重推理干预，不能替代重新训练消融。', '']
            if r['status'] == 'complete':
                per_class = class_rows(results)
                get = lambda mode: {x['class_id']: x for x in per_class if x['split'] == 'val' and x['mode'] == mode}
                h, s = get('hard_raw'), get('soft_all')
                order = sorted(h, key=lambda i: s[i]['IoU_pct'] - h[i]['IoU_pct'], reverse=True)[:5]
                lines += ['### 恢复软场后 IoU 提高最多的五类', '',
                    '| 类别 | 硬 / 软 IoU（%） | 硬 / 软漏检率（%） |', '| --- | ---: | ---: |']
                for i in order:
                    lines.append(f"| {h[i]['category']} | {h[i]['IoU_pct']:.3f} / {s[i]['IoU_pct']:.3f} | {h[i]['miss_rate_pct']:.3f} / {s[i]['miss_rate_pct']:.3f} |")
                lines += ['', '道路类的漏检变化尤其明显；各类数量不同，因此全体前景的二值 IoU 与 17 类等权 mIoU 的变化幅度也不同。完整逐类计数见 CSV。', '']
        comparison = r.get('camera_set_comparison')
        if comparison:
            lines += ['### 与完整 camera-filtered Set 微调基线比较', '',
                '| 指标 | Camera Set 基线 | 当前硬路由 | 变化（百分点） |', '| --- | ---: | ---: | ---: |']
            for key, label in (('mIoU', 'mIoU'), ('IoU', '二值 IoU'), ('FN_rate', '漏检率'),
                               ('FP_over_empty', 'FP / GT empty'), ('hole_rate', '无核心覆盖比例')):
                lines.append(f"| {label} | {num(comparison['camera_set'][key])} | {num(comparison['routing'][key])} | "
                             f"{comparison['delta'][key]:+.3f} |")
            lines += ['', '两组均使用完整相同划分；不同权重的第五级中心会变化，因此各自重算的 query 归属不能视为逐 query 因果配对。', '']
    lines += ['## 统计审计与复算', '',
        '完成状态同时检查 split_complete.json、场景数、帧数和帧索引完整性。每个已纳入场景均检查：'
        'query GT 总量等于 confusion matrix 的 GT 前景总量；自己覆盖、邻居覆盖与无核心三项相加等于全部前景；'
        '保留与删除门控计数、质量和直方图守恒；各渲染方式的 GT 行和一致；FN/FP 转移计数与混淆矩阵差值一致。', '',
        f"- [汇总 JSON]({os.path.relpath(run / 'aggregate/summary.json', DEFAULT_REPORT.parent)})", 
        f"- [逐场景结果 CSV]({os.path.relpath(run / 'aggregate/scene_metrics.csv', DEFAULT_REPORT.parent)})",
        f"- [query 负载分位组 CSV]({os.path.relpath(run / 'aggregate/query_load_deciles.csv', DEFAULT_REPORT.parent)})",
        f"- [逐类别指标 CSV]({os.path.relpath(run / 'aggregate/per_class_metrics.csv', DEFAULT_REPORT.parent)})",
        f"- [数据与源码协议]({os.path.relpath(run / 'protocol.json', DEFAULT_REPORT.parent)})", '',
        '负载分位组在每帧的正负载 query 中分别排序，1 为最低、10 为最高；0 单独表示零负载。'
        'CSV 中 scene=__all__ 是当前已完成场景的合计，绝不能与逐场景行再次相加。', '',
        '复算命令（严格检查训练、验证均完整）：', '', '```bash', 'python3.10 tools/report_query_budget_routing.py', '```', '',
        '门控目标与推理差异的解释见[机制诊断](query_budget_routing_diagnosis.md)。']
    if (DEFAULT_REPORT.parent / 'query_budget_1p5x_failure_analysis.md').exists():
        lines += ['', '## 后续分析', '',
            '另外保存的[扩展复核](query_budget_1p5x_failure_analysis.md)讨论了零保留 query、score 分组与场景相关。'
            '跨组平均漏检率替代得到的差额属于情景估算；本轮未保存按 query 划分的软渲染漏检，不能将这种估算作为零保留 query 的因果贡献比例。', '']
    return '\n'.join(lines)


def report(run=DEFAULT_RUN, report_path=DEFAULT_REPORT, partial=False):
    run, report_path = Path(run), Path(report_path)
    protocol = read(run / 'protocol.json')
    baseline = read(BASELINE) if BASELINE.exists() else {}
    result = dict(version=1, generated_at=datetime.now().astimezone().isoformat(),
                  checkpoint_sha256=protocol['checkpoint_sha256'], status='partial',
                  camera_mask=True, fraction_unit='Fractions except confusion-derived metrics in percent',
                  correlation_unit='Unweighted mean of finite per-frame correlations; valid frame counts stored', splits={})
    scenes_csv, deciles_csv = [], []
    for split in ('val', 'train'):
        total, seen, names = empty_totals(), set(), set()
        paths = sorted((run / 'scenes' / split).glob('*.json'))
        for path in paths:
            meta, value = load_scene(path, protocol)
            if meta['split'] != split or meta['scene'] in names or seen.intersection(meta['indices']):
                raise AssertionError('Scene/frame duplication or split mismatch')
            seen.update(meta['indices'])
            names.add(meta['scene'])
            merge_totals(total, value)
            one = finalize(value)
            scenes_csv.extend(scene_rows(split, meta['scene'], one))
            deciles_csv.extend(decile_rows(split, meta['scene'], value))
        expected = protocol['expected'][split]
        if total['frames'] > expected['frames'] or total['scenes'] > expected['scenes'] or any(i < 0 or i >= expected['frames'] for i in seen):
            raise AssertionError('Collected counts exceed protocol or frame indices are out of range')
        marker_path = run / (split + '_complete.json')
        marker = read(marker_path) if marker_path.exists() else None
        counts_match = total['frames'] == expected['frames'] and total['scenes'] == expected['scenes']
        complete = marker is not None and marker.get('status') == 'complete' and counts_match
        if marker is not None:
            if marker.get('status') != 'complete' or any(marker.get(k) != expected[k] for k in ('frames', 'scenes')):
                raise AssertionError('Invalid completion marker')
            if not counts_match:
                raise AssertionError('Completion marker exists but scene files are incomplete')
        if complete and seen != set(range(expected['frames'])):
            raise AssertionError('Completed frame partition does not cover protocol exactly')
        if not partial and not complete:
            raise RuntimeError(f'{split} is incomplete: {total["frames"]}/{expected["frames"]} frames; use --partial')
        value = finalize(total)
        value.update(status='complete' if complete else 'partial', expected=expected, completion_marker_present=marker is not None,
                     audit=dict(scene_count=total['scenes'], unique_frame_count=len(seen),
                                per_scene_conservation_checks='passed', exact_full_partition=complete))
        if total['frames']:
            value['camera_set_comparison'] = baseline_comparison(split, value, baseline)
            scenes_csv.extend(scene_rows(split, '__all__', value))
            deciles_csv.extend(decile_rows(split, '__all__', total))
        result['splits'][split] = value
    if all(value['status'] == 'complete' for value in result['splits'].values()):
        result['status'] = 'complete'
    aggregate = run / 'aggregate'
    aggregate.mkdir(parents=True, exist_ok=True)
    write_json(aggregate / 'summary.json', result)
    write_csv(aggregate / 'scene_metrics.csv', scenes_csv)
    write_csv(aggregate / 'query_load_deciles.csv', deciles_csv)
    write_csv(aggregate / 'per_class_metrics.csv', class_rows(result))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix('.md.tmp')
    temporary.write_text(markdown(result, run))
    temporary.replace(report_path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    result = report(args.run_dir, args.report, args.partial)
    print(json.dumps(dict(status=result['status'], splits={s: dict(status=r['status'], frames=r['frames'], scenes=r['scenes'])
                     for s, r in result['splits'].items()}), ensure_ascii=False))
