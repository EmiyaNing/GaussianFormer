"""Audit and summarize full tiny/small query radii without running inference."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work_dirs/query_influence_radius_full'
REPORT = ROOT / 'analysis/query_influence_radius_results.md'
BINS = ((0, 0), (1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, None))
BIN_NAMES = ('0', '1–5', '6–10', '11–20', '21–50', '51–100', '>100')
VERSION = 1


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def flatten(data, prefix=''):
    result = {}
    for key, value in data.items():
        key = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, key + '_'))
        else:
            result[key] = value
    return result


def stats(values):
    quantiles = np.quantile(values, (.1, .5, .9, .99))
    return dict(mean=float(values.mean(dtype=np.float64)), minimum=float(values.min()),
        p10=float(quantiles[0]), p50=float(quantiles[1]), p90=float(quantiles[2]),
        p99=float(quantiles[3]), maximum=float(values.max()))


def masks(core, camera):
    for label, (lo, hi) in zip(BIN_NAMES, BINS):
        yield label, (core >= lo) if hi is None else ((core >= lo) & (core <= hi))
    for t in (5, 10, 20):
        yield f'≤{t}', core <= t
    yield '零覆盖且核心未接触camera', (core == 0) & ~camera
    yield '零覆盖且核心仅接触camera空区', (core == 0) & camera


def aggregate(model, split, protocol):
    folder = RUN / model / 'scenes' / split
    marker = RUN / model / (split + '_complete.json')
    assert marker.exists()
    expected = protocol['expected'][split]
    assert read(marker) == dict(status='complete', **expected)
    options = protocol['models'][model]
    fields = protocol['fields']
    nf, nq = expected['frames'], options['q']
    source_key = digest(RUN / 'protocol.json') + ':' + digest(Path(__file__))
    destination = RUN / 'aggregate' / model / split
    cached = destination / 'summary.json'
    # Cache is used only after a complete, audited phase has been summarized.
    if cached.exists():
        old = read(cached)
        if old['source_key'] == source_key:
            for relative, expected_hash in old['scene_file_sha256'].items():
                assert digest(RUN / relative) == expected_hash, f'Scene changed: {relative}'
            return old
    paths = sorted(folder.glob('*.json'))
    assert len(paths) == expected['scenes']
    data = np.empty((nf, nq, len(fields)), dtype=np.float32)
    core = np.empty((nf, nq), dtype=np.int32)
    camera = np.empty((nf, nq), dtype=bool)
    kinds = np.empty((nf, nq), dtype=np.int8)
    contacts = np.empty((nf, nq), dtype=np.int16)
    seen = np.zeros(nf, dtype=bool)
    tokens, token_set, scene_rows, file_hashes = {}, set(), [], {}
    for path in paths:
        meta = read(path)
        assert meta['status'] == 'complete' and meta['model'] == model and meta['split'] == split
        assert meta['checkpoint_sha256'] == options['checkpoint_sha256']
        ids = np.array(meta['indices'], dtype=np.int64)
        assert len(ids) == len(meta['records']) and not seen[ids].any()
        assert np.array_equal(ids, [row['index'] for row in meta['records']])
        seen[ids] = True
        for row in meta['records']:
            assert row['token'] not in token_set
            token_set.add(row['token'])
            tokens[row['index']] = row['token']
            assert row['zero_opacity_gs'] == 0
        with np.load(path.with_suffix('.npz')) as z:
            assert z['metrics'].shape == (len(ids), nq, len(fields))
            v = z['metrics']
            assert np.isfinite(v).all() and (v >= 0).all()
            # Euclidean parallel-axis identity separates cloud spread and shift.
            rms2 = v[..., fields.index('center_radius_rms')].astype(np.float64) ** 2
            internal2 = v[..., fields.index('internal_radius_rms')].astype(np.float64) ** 2
            shift2 = v[..., fields.index('centroid_shift')].astype(np.float64) ** 2
            np.testing.assert_allclose(rms2, internal2 + shift2, rtol=2e-5, atol=2e-4)
            data[ids] = v
            core[ids] = z['core_foreground']
            camera[ids] = z['camera_core_contact']
            kinds[ids] = z['reference_kind']
            contacts[ids] = z['core_contact_gs']
            assert z['reference_center'].shape == (len(ids), nq, 3)
            assert z['gs_centroid'].shape == (len(ids), nq, 3)
            assert (z['core_contact_gs'] <= options['children']).all()
            assert np.array_equal(z['core_foreground'] == 0, z['core_contact_gs'] == 0)
        for lower, upper in (('core_outer_radius_lower', 'core_outer_radius_upper'),
                             ('three_sigma_outer_radius_lower', 'three_sigma_outer_radius_upper')):
            assert (v[..., fields.index(lower)] <= v[..., fields.index(upper)] + 1e-5).all()
        row = dict(model=model, split=split, scene=meta['scene'], scene_name=meta['scene_name'],
            frames=len(ids), query_frames=len(ids) * nq, zero_fraction=float((core[ids] == 0).mean()),
            le20_fraction=float((core[ids] <= 20).mean()))
        for j, field in enumerate(fields):
            row[field + '_mean'] = float(v[..., j].mean(dtype=np.float64))
        low = core[ids] <= 20
        if low.any():
            for field in ('center_radius_mean', 'center_radius_max', 'centroid_shift',
                          'internal_radius_mean', 'reference_to_fg_volume'):
                row['le20_' + field + '_mean'] = float(v[..., fields.index(field)][low].mean(dtype=np.float64))
        scene_rows.append(row)
        file_hashes[str(path.relative_to(RUN))] = digest(path)
        file_hashes[str(path.with_suffix('.npz').relative_to(RUN))] = digest(path.with_suffix('.npz'))
    assert seen.all() and len(set(tokens.values())) == nf
    f = data.reshape(-1, len(fields))
    c, cam, kind, contact = (a.reshape(-1) for a in (core, camera, kinds, contacts))
    rows = {}
    selections = [('全部', None), *list(masks(c, cam))]
    bin_total = 0
    for label, selected in selections:
        n = len(f) if selected is None else int(selected.sum())
        row = dict(query_frames=n, fraction=n / len(f), queries_per_frame=n / nf)
        if label in BIN_NAMES:
            bin_total += n
        if n:
            row['metrics'] = {field: stats(f[:, j] if selected is None else f[selected, j])
                              for j, field in enumerate(fields)}
            selected_kind = kind if selected is None else kind[selected]
            row['reference_kind_fraction'] = {str(k): float((selected_kind == k).mean()) for k in (1, 2, 3, 4)}
            row['core_foreground_mean'] = float((c if selected is None else c[selected]).mean(dtype=np.float64))
            row['foreground_touching_child_gs_mean'] = float((contact if selected is None else contact[selected]).mean(dtype=np.float64))
            for k, prefix in ((1, 'core'), (3, 'three_sigma')):
                lo = f[:, fields.index(prefix + '_outer_radius_lower')]
                hi = f[:, fields.index(prefix + '_outer_radius_upper')]
                width = hi - lo
                row[prefix + '_bound_width'] = stats(width if selected is None else width[selected])
                # Per-GS containing sphere bound, averaged across the child GS.
                a = f[:, fields.index('center_radius_mean')]
                b = f[:, fields.index('gs_major_sigma_mean')]
                radius = a + k * b
                row[f'child_mean_{k}sigma_outer_upper'] = stats(radius if selected is None else radius[selected])
        rows[label] = row
        print(f'Summarized {model}/{split}: {label}, n={n:,}', flush=True)
    assert bin_total == len(f)
    assert rows['0']['query_frames'] == rows['零覆盖且核心未接触camera']['query_frames'] + rows['零覆盖且核心仅接触camera空区']['query_frames']
    slots = []
    sums = data.mean(axis=0, dtype=np.float64)
    low_counts = (core <= 20).sum(axis=0)
    low_means = {}
    for field in ('center_radius_mean', 'center_radius_max', 'centroid_shift',
                  'internal_radius_mean', 'gs_major_sigma_mean', 'reference_to_fg_volume'):
        values = data[..., fields.index(field)]
        low_means[field] = np.divide(np.where(core <= 20, values, 0).sum(axis=0, dtype=np.float64),
            low_counts, out=np.zeros(nq), where=low_counts > 0)
    for q in range(nq):
        row = dict(model=model, split=split, query_slot=q, frames=nf,
            core_foreground_mean=float(core[:, q].mean()), zero_fraction=float((core[:, q] == 0).mean()),
            le5_fraction=float((core[:, q] <= 5).mean()), le10_fraction=float((core[:, q] <= 10).mean()),
            le20_fraction=float((core[:, q] <= 20).mean()), le20_frames=int(low_counts[q]))
        row.update({field + '_mean': float(sums[q, j]) for j, field in enumerate(fields)})
        for field, values in low_means.items():
            row['le20_' + field + '_mean'] = float(values[q]) if low_counts[q] else None
        slots.append(row)
    result = dict(status='complete', version=VERSION, source_key=source_key, model=model, split=split,
        frames=nf, scenes=len(paths), queries=nq, children=options['children'], query_frames=nf * nq,
        checkpoint_sha256=options['checkpoint_sha256'], groups=rows,
        index_token_sha256=hashlib.sha256(json.dumps(sorted(tokens.items())).encode()).hexdigest(),
        scene_file_sha256=file_hashes)
    csv_write(destination / 'coverage_groups.csv', [dict(model=model, split=split, group=k, **flatten(v)) for k, v in rows.items()])
    csv_write(destination / 'per_scene.csv', scene_rows)
    csv_write(destination / 'per_query_slot.csv', slots)
    write(cached, result)
    del data, core, camera, kinds, contacts, f, c, cam, kind, contact
    gc.collect()
    return result


def metric(row, field, statistic='mean'):
    return row['metrics'][field][statistic]


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = ROOT / 'analysis/assets'
    destination.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    lookup = {(r['model'], r['split']): r for r in results}
    for i, model in enumerate(('tiny', 'small')):
        for j, split in enumerate(('train', 'val')):
            ax = axes[i, j]
            ax.set_title(model.capitalize() + ' / ' + ('Training split' if split == 'train' else 'Validation split'))
            if (model, split) not in lookup:
                ax.text(.5, .5, 'Not complete yet', ha='center', transform=ax.transAxes)
                continue
            r = lookup[model, split]
            x = np.arange(len(BIN_NAMES))
            group = [r['groups'][name] for name in BIN_NAMES]
            p10 = [metric(a, 'center_radius_mean', 'p10') for a in group]
            p90 = [metric(a, 'center_radius_mean', 'p90') for a in group]
            ax.fill_between(x, p10, p90, color='#2479b8', alpha=.12, label='Center-radius P10–P90')
            for field, label, color, linestyle in (
                ('center_radius_mean', 'GS center radius (mean)', '#2479b8', '-'),
                ('internal_radius_mean', 'Internal spread', '#2a914f', '-'),
                ('centroid_shift', 'Centroid shift', '#cb6a1b', '-'),
                ('gs_major_sigma_mean', 'GS own 1-sigma major radius', '#9b51b6', '--')):
                ax.plot(x, [metric(a, field) for a in group], marker='o', ms=4,
                    lw=1.7, color=color, linestyle=linestyle, label=label)
            ax.axvspan(-.5, 3.5, color='#999999', alpha=.05)
            ax.set_xticks(x, ['0', '1–5', '6–10', '11–20', '21–50', '51–100', '>100'])
            ax.grid(axis='y', alpha=.2)
            ax.set_ylim(bottom=0)
            if j == 0:
                ax.set_ylabel('Radius / distance (voxel grid units)')
            if i == 1:
                ax.set_xlabel('Distinct foreground voxels covered by query 1-sigma union')
    # Shared axes were frozen by set_ylim(bottom=0) before later panels were
    # drawn. Set their final range from every displayed curve and interval.
    displayed = []
    for r in results:
        for name in BIN_NAMES:
            a = r['groups'][name]
            displayed.append(metric(a, 'center_radius_mean', 'p90'))
            displayed.extend(metric(a, field) for field in ('center_radius_mean',
                'internal_radius_mean', 'centroid_shift', 'gs_major_sigma_mean'))
    for ax in axes.flat:
        ax.set_ylim(0, max(displayed) * 1.08)
    handles, labels = next(ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0])
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Query radii by foreground core coverage  |  1 grid unit = 0.4 m', fontsize=14)
    fig.tight_layout(rect=(0, .1, 1, .95))
    fig.savefig(destination / 'query_influence_radius_ranges.png', dpi=180)
    fig.savefig(destination / 'query_influence_radius_ranges.svg')
    plt.close(fig)


def markdown(results, complete):
    lines = ['# Tiny与Small：每个Query把GS铺在多大范围内', '',
        f"**状态：{'完整统计已完成' if complete else '统计进行中；以下仅展示已完成的模型与集合'}。** "
        '使用原始tiny和small预训练权重，分别统计完整训练集与验证集。全量目标是每个模型700个训练场景、'
        '28,130帧，以及150个验证场景、6,019帧。', '',
        '**全文距离单位为体素网格：1格=0.4米，半径使用三维欧氏距离，允许小数。** '
        '本报告的query中心是第六级decoder取图像特征时实际使用的参考中心。'
        'GS是它在第六级生成的最终子GS：tiny每query为128个，small为64个。', '',
        '## 先看低覆盖组与全集的差别', '',
        '| 模型 / 集合 | 全集GS中心平均半径 | ≤20前景体素组的平均半径 | 低覆盖组 / 全集 | 低覆盖Query比例 |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        all_row, low = r['groups']['全部'], r['groups']['≤20']
        a, b = metric(all_row, 'center_radius_mean'), metric(low, 'center_radius_mean')
        lines.append(f"| {r['model']} / {'训练' if r['split']=='train' else '验证'} | {a:.3f} | {b:.3f} | {b/a:.3f}倍 | {100*low['fraction']:.3f}% |")
    ratios = [metric(r['groups']['≤20'], 'center_radius_mean') / metric(r['groups']['全部'], 'center_radius_mean') for r in results]
    if all(ratio > 1 for ratio in ratios):
        lines += ['', f'在已完成的{len(results)}个模型–集合组合中，低覆盖组的GS中心平均半径都更大，'
            f'比各自全集均值高{100*(min(ratios)-1):.1f}%–{100*(max(ratios)-1):.1f}%。'
            '因此，这些query覆盖少并不能简单归结为“把GS铺得太窄”。']
    lines += ['', '下面进一步拆开GS自身尺寸、内部展开和整体位移，判断这种差异主要体现在哪里。', '',
        '![不同前景核心覆盖组的Query半径与位移](assets/query_influence_radius_ranges.png)', '',
        '蓝线为GS中心平均半径，浅蓝带为该指标在组内的P10–P90；绿线为内部展开，橙线为整体位移，紫色虚线为GS自身最长轴1σ半径。'
        '灰底覆盖0至20个前景体素的低覆盖分档。', '',
        '## 先区分三种“半径”', '',
        '| 指标 | 怎么计算 | 回答什么问题 |',
        '|---|---|---|',
        '| GS中心平均半径 | 每个子GS中心到query参考中心的距离，先在query内平均 | 这个query通常把GS中心铺到多远 |',
        '| GS自身1σ半径 | 每个GS三条主轴尺度的最大值，先在query内平均 | GS本身有多大，与它离query多远无关 |',
        '| Query的场包围半径 | 以query参考中心为球心，包住其全部子GS的1σ或3σ椭球 | 整组GS的几何场最远延伸到哪里 |', '',
        '前两项按真实坐标与尺度直接计算。第三项对旋转椭球给出上下界；表中的“下界–上界”'
        '表示精确包围半径被夹在其中。上下界分别在组内平均，并不是半径的最小值与最大值。'
        '这些半径不按opacity或语义得分加权。包围半径描述最远伸展，球内可能有大片没有GS覆盖的空隙，不能据此推算有效覆盖体积。', '',
        '## 全集：平均半径有多大', '',
        '| 模型 / 集合 | Query–帧数 | GS中心平均半径 | GS中心最远半径 | GS自身1σ半径 | Query的1σ场包围半径：下界–上界 | 3σ场包围半径：下界–上界 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        a = r['groups']['全部']
        vals = [metric(a, k) for k in ('center_radius_mean', 'center_radius_max', 'gs_major_sigma_mean',
            'core_outer_radius_lower', 'core_outer_radius_upper', 'three_sigma_outer_radius_lower', 'three_sigma_outer_radius_upper')]
        lines.append(f"| {r['model']} / {'训练' if r['split']=='train' else '验证'} | {r['query_frames']:,} | "
            f'{vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f}–{vals[4]:.3f} | {vals[5]:.3f}–{vals[6]:.3f} |')
    lines += ['', '各列都先得到每个query–帧实例的指标，再对实例平均。“GS中心最远半径”因此是各query最大半径的平均，'
        '不是整个数据集的极值。两个模型最终均生成76,800个GS，但query数量与每query子GS数不同，'
        '模型容量也不同，不能把差异归因为query数量这一个因素。', '',
        '## 低覆盖Query：半径是变大还是变小', '',
        '分档沿用此前统计：数出每个query全部子GS的1σ核心并集内，**不同camera有效GT前景体素**的数量。'
        '这是前景覆盖量，不是GS数，也不是最近邻Set匹配量。几何半径本身在完整场景空间计算；camera mask仅用于前景覆盖分组。', '']
    for r in results:
        label = f"{r['model']} / {'训练集' if r['split']=='train' else '验证集'}"
        lines += [f'### {label}', '',
            '| 核心覆盖前景体素数 | Query比例 | 平均GS中心半径：均值 / P50 / P90 | 最远GS中心半径均值 | GS自身1σ半径均值 | 内部展开半径均值 | GS整体位移均值 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name in BIN_NAMES:
            a = r['groups'][name]
            if not a['query_frames']:
                continue
            x = a['metrics']['center_radius_mean']
            lines.append(f"| {name} | {100*a['fraction']:.3f}% | {x['mean']:.3f} / {x['p50']:.3f} / {x['p90']:.3f} | "
                f"{metric(a,'center_radius_max'):.3f} | {metric(a,'gs_major_sigma_mean'):.3f} | "
                f"{metric(a,'internal_radius_mean'):.3f} | {metric(a,'centroid_shift'):.3f} |")
        lines += ['']
    lines += ['“内部展开”以最终子GS中心的均值为圆心；“整体位移”是这个均值到query采样参考中心的距离。'
        '前者反映这组GS是否分散，后者反映它们是否整体偏离输入参考位置。二者不能直接相加来还原平均半径。', '',
        '### 与此前表格对照：累计低覆盖组', '',
        '| 模型 / 集合 | 覆盖上限 | Query比例 | 平均GS中心半径 | GS内部展开 | GS整体位移 | Query的3σ场半径：下界–上界 | 参考中心到可见前景实体距离 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        for name in ('≤5', '≤10', '≤20'):
            a = r['groups'][name]
            if a['query_frames']:
                lines.append(f"| {r['model']} / {'训练' if r['split']=='train' else '验证'} | {name} | {100*a['fraction']:.3f}% | "
                    f"{metric(a,'center_radius_mean'):.3f} | {metric(a,'internal_radius_mean'):.3f} | {metric(a,'centroid_shift'):.3f} | "
                    f"{metric(a,'three_sigma_outer_radius_lower'):.3f}–{metric(a,'three_sigma_outer_radius_upper'):.3f} | "
                    f"{metric(a,'reference_to_fg_volume'):.3f} |")
    lines += ['', '累计组包含零覆盖，≤5、≤10、≤20互相嵌套。最后一列是query实际参考中心到camera有效前景立方体并集的距离，'
        '不是子GS中心距离，也不是到query“核心”的距离。', '',
        '## 零覆盖Query中，哪些进入了Camera区域', '',
        '| 模型 / 集合 | 零覆盖分组 | 占全部Query比例 | GS中心平均半径 | GS内部展开 | GS整体位移 | 参考中心到可见前景实体距离 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        for name in ('零覆盖且核心未接触camera', '零覆盖且核心仅接触camera空区'):
            a = r['groups'][name]
            if a['query_frames']:
                lines.append(f"| {r['model']} / {'训练' if r['split']=='train' else '验证'} | {name} | {100*a['fraction']:.3f}% | "
                    f"{metric(a,'center_radius_mean'):.3f} | {metric(a,'internal_radius_mean'):.3f} | {metric(a,'centroid_shift'):.3f} | "
                    f"{metric(a,'reference_to_fg_volume'):.3f} |")
    lines += ['', 'camera分组按子GS的1σ核心是否触及有效GT体素中心确定，不按query参考中心是否位于mask内确定。'
        '没有进入camera有效区域的query不能据此判断为真实空区域中的浪费。'
        '核心未进入有效camera区域的query，与核心已进入camera空区域的query，虽然前景覆盖都是零，'
        '它们的半径和前景距离仍可能不同。', '',
        '## 特征采样范围与最终GS范围', '',
        '| 模型 / 集合 | 图像特征采样点：平均 / 最远半径 | 最终GS中心：平均 / 最远半径 | 原生渲染候选体素最远半径 |',
        '|---|---:|---:|---:|']
    for r in results:
        a = r['groups']['全部']
        lines.append(f"| {r['model']} / {'训练' if r['split']=='train' else '验证'} | "
            f"{metric(a,'feature_sampling_radius_mean'):.3f} / {metric(a,'feature_sampling_radius_max'):.3f} | "
            f"{metric(a,'center_radius_mean'):.3f} / {metric(a,'center_radius_max'):.3f} | {metric(a,'native_candidate_radius_max'):.3f} |")
    lines += ['', '采样半径统计投影前的3D采样点，包含未投影到有效图像的点，不等于实际有效图像感受野。'
        '原生渲染一列使用实际整数候选盒，并在场景边界截断；量的是候选体素中心到query参考中心的最大距离。'
        '候选盒包含弱高斯尾部，这个数不等于有效前景证据的作用半径。理想3σ椭球与原生候选盒也不是同一个集合。', '',
        '## 三个方向的空间跨度', '',
        '| 模型 / 集合 | 1σ场包围盒跨度：X / Y / Z | 3σ场包围盒跨度：X / Y / Z | 原生候选包围盒：X / Y / Z |',
        '|---|---:|---:|---:|']
    for r in results:
        a = r['groups']['全部']
        columns = [' / '.join(f'{metric(a,prefix + "_bbox_span_" + axis):.3f}' for axis in 'xyz')
                   for prefix in ('core', 'three_sigma', 'native')]
        lines.append(f"| {r['model']} / {'训练' if r['split']=='train' else '验证'} | " + ' | '.join(columns) + ' |')
    lines += ['', '这张表是包围盒的完整宽度，不是半径；每个query先取其子GS并集的包围盒，再跨query平均。'
        '理想场的跨度是连续几何长度除以0.4米，原生候选跨度是离散体素格数。两者都不表示包围盒内部被全部填满。', '',
        '## 这组统计能说明什么', '',
        '它直接回答低覆盖query的子GS分布范围、自身尺寸，以及是否整体偏离采样参考中心。'
        '分组同时受到camera可见范围、前景体积、参考位置与各query之间覆盖重叠的影响，'
        '因此半径与低覆盖的相关性不能直接证明某个半径导致了漏检，也不能把“覆盖少”直接等同于query无用。', '',
        '训练集与验证集都使用同一份冻结权重、确定性测试预处理和eval模式，比较的是两个数据集合上的推理分布，'
        '不代表模型训练过程中每一步的半径分布。', '',
        '## 数据与复核', '',
        '- [统计设计与数学定义](query_influence_radius_design.md)',
        '- [全部汇总JSON](../work_dirs/query_influence_radius_full/aggregate/summary.json)',
        '- [冻结权重与代码协议](../work_dirs/query_influence_radius_full/protocol.json)',
        '- [统计实现](../tools/query_radius_stats.py)',
        '- [报告与审计实现](../tools/report_query_influence_radius.py)', '']
    for r in results:
        base = f"../work_dirs/query_influence_radius_full/aggregate/{r['model']}/{r['split']}"
        label = f"{r['model']} / {r['split']}"
        lines.append(f'- {label}：[全部分档与分位数CSV]({base}/coverage_groups.csv)；'
            f'[逐场景CSV]({base}/per_scene.csv)；[逐query槽位CSV]({base}/per_query_slot.csv)。')
    lines += ['', '每个场景的NPZ保存逐帧逐query的29个指标、实际参考中心、最终GS质心和前景覆盖分组，'
        '保存在`work_dirs/query_influence_radius_full/{tiny,small}/scenes/{train,val}/`。'
        '逐query槽位CSV按固定网络索引跨帧汇总，这个索引并不代表跨场景追踪的同一物体。', '',
        '两模型均要求：增加采样观察钩子后logit逐元素完全一致；时间特征缓存与原生输入输出完全一致；'
        '中心半径对照独立FP64计算；每帧每query的1σ前景覆盖与历史全量统计完全一致。'
        '汇总检查全部场景、索引、token、权重哈希与分档守恒。均值用FP64累加，分位数从原始数值计算。', '',
        '```bash', 'python3.10 tools/stat_query_influence_radius.py',
        'python3.10 tools/report_query_influence_radius.py', '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true', help='Only completed phases; mark incomplete')
    args = parser.parse_args()
    protocol = read(RUN / 'protocol.json')
    # Every report audits the frozen inference sources, including cached phases.
    for path, expected in protocol['source_sha256'].items():
        assert digest(ROOT / path) == expected, f'Source changed: {path}'
    results = []
    for model in ('tiny', 'small'):
        for split in ('train', 'val'):
            marker = RUN / model / (split + '_complete.json')
            if marker.exists():
                assert read(RUN / model / 'pilot.json')['status'] == 'passed'
                results.append(aggregate(model, split, protocol))
            else:
                assert args.partial, f'Phase not complete: {model}/{split}'
    assert results, 'No completed phases yet'
    complete = len(results) == 4
    if complete:
        assert read(RUN / 'complete.json')['status'] == 'complete'
        for model, options in protocol['models'].items():
            assert digest(ROOT / options['checkpoint']) == options['checkpoint_sha256']
        for split in ('train', 'val'):
            pair = [r for r in results if r['split'] == split]
            assert pair[0]['index_token_sha256'] == pair[1]['index_token_sha256']
        assert sum(r['frames'] for r in results) == 68298
        assert sum(r['scenes'] for r in results) == 1700
    summary = dict(status='complete' if complete else 'partial',
        generated_at=datetime.now().astimezone().isoformat(), reporter_sha256=digest(Path(__file__)),
        protocol_sha256=digest(RUN / 'protocol.json'), results=results)
    write(RUN / 'aggregate/summary.json', summary)
    plot(results)
    REPORT.write_text(markdown(results, complete))
    print(f'Wrote {REPORT}; status={summary["status"]}', flush=True)


if __name__ == '__main__':
    main()
