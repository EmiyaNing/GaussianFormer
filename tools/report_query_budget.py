"""Readable full-query census report and exact-denominator aggregate tables."""
import argparse
import csv
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.stat_query_budget import RUN, OLD, FIELDS, Q, CHILDREN
from tools.stat_boundary_gaussians import read, write

REPORT = ROOT / 'analysis/query_budget_full_results.md'
FIGURES = ROOT / 'analysis/figures/query_budget_full'
AGG = RUN / 'aggregate'
LOAD_EDGES = np.array([1, 8, 16, 32, 64, 128, 256, 512, 1024, np.inf])
LOAD_NAMES = ['0', '1–7', '8–15', '16–31', '32–63', '64–127', '128–255', '256–511', '512–1023', '≥1024']


def csv_write(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def div(a, b):
    return float(a / b) if b else None


def pct(x):
    return '—' if x is None else f'{x*100:.2f}%'


def num(x, decimals=2):
    return '—' if x is None else f'{x:,.{decimals}f}'


def field(array, name):
    return array[..., FIELDS.index(name)]


def summarize(rows):
    sums = rows.sum(axis=0, dtype=np.float64)
    s = lambda k: float(sums[FIELDS.index(k)])
    n, g, covered = len(rows), s('gt_foreground'), s('core_coverage_all_fg')
    return dict(queries=n, foreground=int(g), holes=int(s('gt_no_core')),
        miss=int(s('gt_miss')), correct=int(s('gt_correct')),
        mean_load_per_gaussian=div(g, n*CHILDREN),
        hole_rate=div(s('gt_no_core'), g), miss_rate=div(s('gt_miss'), g),
        own_core_rate=div(s('gt_own_core'), g), neighbor_only_rate=div(s('gt_neighbor_only_core'), g),
        core_fg_coverage_per_gaussian=div(covered, n*CHILDREN),
        shared_coverage_fraction=div(s('core_shared_other_query'), covered),
        exclusive_core_per_gaussian=div(s('core_exclusive'), n*CHILDREN),
        core_gs_touch_foreground_fraction=div(s('core_gs_touches_fg'), n*CHILDREN),
        native_gs_touch_foreground_fraction=div(s('native_gs_touches_fg'), n*CHILDREN),
        zero_exclusive_core_fraction=float((rows[:,FIELDS.index('core_exclusive')]==0).mean()) if n else None,
        no_core_correct_fraction=div(s('gt_no_core_correct'), g),
        no_core_correct_rate=div(s('gt_no_core_correct'), s('gt_no_core')),
        semantic_boundary_fraction=div(s('gt_semantic_boundary'), g),
        empty_boundary_fraction=div(s('gt_empty_boundary'), g),
        mean_coarse_distance=div(s('gt_coarse_distance_sum'), g),
        mean_range=div(s('gt_horizontal_range_sum'), g),
        mean_components=div(s('gt_components'), n),
        mean_classes=div(s('gt_classes'), n),
        mean_span_diagonal=float(np.linalg.norm(rows[:, [FIELDS.index('gt_span_x'),FIELDS.index('gt_span_y'),FIELDS.index('gt_span_z')]],axis=1).mean()) if n else None,
        amplitude_mean=div(s('amplitude_mean'), n), opacity_mean=div(s('opacity_mean'), n),
        effective_amplitude_mean=div(s('effective_amplitude_mean'), n),
        scale_mean=div(s('scale_mean'), n),
        no_core_correct_dom_n=int(s('no_core_correct_dom_n')),
        no_core_correct_dom_amplitude=div(s('no_core_correct_dom_amplitude_sum'), s('no_core_correct_dom_n')),
        no_core_correct_dom_opacity=div(s('no_core_correct_dom_opacity_sum'), s('no_core_correct_dom_n')),
        no_core_correct_dom_mahalanobis=div(s('no_core_correct_dom_mahalanobis_sum'), s('no_core_correct_dom_n')),
        core_correct_dom_amplitude=div(s('core_correct_dom_amplitude_sum'), s('core_correct_dom_n')),
        core_correct_dom_opacity=div(s('core_correct_dom_opacity_sum'), s('core_correct_dom_n')),
        core_correct_dom_mahalanobis=div(s('core_correct_dom_mahalanobis_sum'), s('core_correct_dom_n')),
        dom_no_core_correct_from_same_query_fraction=div(s('dom_no_core_correct_from_same_query'), s('no_core_correct_dom_n')),
        dominant_fp=int(s('dominant_fp_voxels')),
        dominant_fp_amplitude=div(s('dominant_fp_amplitude_sum'), s('dominant_fp_voxels')),
        dominant_fp_opacity=div(s('dominant_fp_opacity_sum'), s('dominant_fp_voxels')))


def deciles(load):
    active = np.flatnonzero(load > 0)
    order = active[np.argsort(load[active], kind='stable')]
    bins = np.zeros(len(load), dtype=np.int8)
    bins[order] = np.arange(len(order)) * 10 // max(len(order), 1) + 1
    return bins


def bootstrap_mean(values, seed=928, repeats=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    if not len(values):
        return dict(mean=None, interval=None, scenes=0)
    samples = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
    return dict(mean=float(values.mean()), interval=np.quantile(samples, [.025, .975]).tolist(), scenes=len(values))


def standardized(strata):
    # Same GT-class x range x coarse-position-error mix for low/high load queries.
    low, high = strata[1:3].sum(0), strata[9:11].sum(0)
    all_fg = strata[..., 0].sum(0)
    common = (low[..., 0] > 0) & (high[..., 0] > 0)
    weight = all_fg[common].astype(float)
    retained = div(weight.sum(), all_fg.sum())
    weight /= max(weight.sum(), 1)
    result = dict(retained_foreground_fraction=retained, common_cells=int(common.sum()),
        total_cells=int(common.size), low={}, high={})
    for name, a in (('low', low), ('high', high)):
        for j, metric in ((1, 'hole_rate'), (2, 'miss_rate'), (3, 'no_core_correct_fraction'), (4, 'own_core_rate')):
            result[name][metric] = float((weight * a[..., j][common] / a[..., 0][common]).sum())
    return result


def matched_complexity(rows, class_counts):
    """Single vs multiple disconnected FG patches, matched on query-level cues."""
    load = field(rows, 'gt_foreground')
    active = load > 0
    a, g = rows[active], load[active]
    cls = class_counts[active].argmax(1)
    lb = np.searchsorted(LOAD_EDGES, g, side='right')
    rb = np.searchsorted([10.,20.,30.], field(a,'gt_horizontal_range_sum')/g, side='right')
    eb = np.searchsorted([.4,.8,1.6], field(a,'gt_coarse_distance_sum')/g, side='right')
    key = ((lb*17+cls)*4+rb)*4+eb
    multi = field(a,'gt_components')>=2
    cells = len(LOAD_NAMES)*17*4*4
    table = np.zeros((2,cells,3),dtype=np.float64)
    for j,mask in enumerate((~multi,multi)):
        for k,name in enumerate(('gt_foreground','gt_no_core','gt_miss')):
            table[j,:,k]=np.bincount(key[mask],weights=field(a,name)[mask],minlength=cells)
    common=(table[0,:,0]>0)&(table[1,:,0]>0)
    weights=table[:,:,0].sum(0)
    retained=div(weights[common].sum(),weights.sum())
    w=weights[common]/max(weights[common].sum(),1)
    result=dict(retained_foreground_fraction=retained,common_cells=int(common.sum()),
        single={},multiple={},unadjusted_single={},unadjusted_multiple={})
    for j,label in enumerate(('single','multiple')):
        for k,name in ((1,'hole_rate'),(2,'miss_rate')):
            result[label][name]=float((w*table[j,common,k]/table[j,common,0]).sum())
            result['unadjusted_'+label][name]=div(table[j,:,k].sum(),table[j,:,0].sum())
        result['unadjusted_'+label]['foreground']=int(table[j,:,0].sum())
    return result


def bootstrap_standardized(scene_pairs, scene_all, repeats=1000):
    """Resample whole scenes, then redo common-cell composition adjustment."""
    pairs=np.asarray(scene_pairs,dtype=np.float64)
    all_fg=np.asarray(scene_all,dtype=np.float64)
    n=len(pairs)
    rng=np.random.default_rng(928)
    w=rng.multinomial(n,np.full(n,1/n),size=repeats).astype(np.float64)
    totals=(w @ pairs.reshape(n,-1)).reshape(repeats,2,272,5)
    fg=w @ all_fg
    common=(totals[:,0,:,0]>0)&(totals[:,1,:,0]>0)
    weights=fg*common
    weights/=np.maximum(weights.sum(1,keepdims=True),1)
    result={}
    for j,name in ((1,'hole_gap'),(2,'miss_gap')):
        rates=np.divide(totals[:,:,:,j],totals[:,:,:,0],out=np.zeros_like(totals[:,:,:,j]),where=totals[:,:,:,0]>0)
        diff=((rates[:,1]-rates[:,0])*weights).sum(1)
        result[name+'_interval']=np.quantile(diff,[.025,.975]).tolist()
    return result


def aggregate(partial=False):
    from scipy.stats import spearmanr
    AGG.mkdir(parents=True, exist_ok=True)
    protocol = read(RUN / 'protocol.json')
    summary, data, class_data, strata_totals = {}, {}, {}, {}
    scene_rows, correlations, identities = [], [], set()
    bucket_rows, decile_rows, group_rows = [], [], []
    scene_index = {}
    for split in ('train', 'val'):
        chunks, classes, scenes, frame_count = [], [], [], 0
        scene_pairs,scene_all=[],[]
        strata = np.zeros((11, 17, 4, 4, 5), dtype=np.int64)
        for path in sorted((RUN / 'scenes' / split).glob('*.json')):
            meta = read(path)
            with np.load(path.with_suffix('.npz')) as z:
                a, cls, s = z['metrics'], z['class_counts'], z['strata']
            assert a.shape == (meta['frames'], Q, len(FIELDS))
            assert cls.shape == (meta['frames'], Q, 17)
            assert meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
            for record in meta['records']:
                key = (split, record['sample_token'])
                assert key not in identities
                identities.add(key)
            np.testing.assert_array_equal(field(a, 'gt_foreground'), cls.sum(2))
            assert int(field(a, 'gt_no_core').sum(dtype=np.float64)) == sum(r['no_core'] for r in meta['records'])
            assert int(field(a, 'gt_miss').sum(dtype=np.float64)) == sum(r['voxels'][0] for r in meta['records'])
            np.testing.assert_array_equal(field(a, 'core_coverage_all_fg'),
                field(a, 'core_shared_other_query') + field(a, 'core_exclusive'))
            flat = a.reshape(-1, len(FIELDS))
            load = field(flat, 'gt_foreground')
            active = load > 0
            m = summarize(flat)
            scenes.append(dict(name=meta['scene_name'], start=frame_count, frames=meta['frames']))
            scene_rows.append(dict(split=split, scene=meta['scene_name'], frames=meta['frames'], **m))
            for predictor, values in (
                ('load', load),
                ('semantic_boundary_fraction', field(flat, 'gt_semantic_boundary') / np.maximum(load, 1)),
                ('components', field(flat, 'gt_components')),
                ('coarse_distance', field(flat, 'gt_coarse_distance_sum') / np.maximum(load, 1))):
                x, y = values[active], field(flat, 'gt_no_core')[active] / load[active]
                rho = float(spearmanr(x, y).statistic) if len(np.unique(x)) > 1 and len(np.unique(y)) > 1 else float('nan')
                correlations.append(dict(split=split, scene=meta['scene_name'], predictor=predictor,
                    spearman_rho=rho, active_queries=int(active.sum())))
            chunks.append(a)
            classes.append(cls)
            strata += s
            scene_pairs.append(np.stack((s[1:3].sum(0),s[9:11].sum(0))).reshape(2,272,5))
            scene_all.append(s[...,0].sum(0).reshape(272))
            frame_count += meta['frames']
        if not chunks:
            continue
        if not partial:
            assert len(chunks) == protocol['expected'][split]['scenes']
            assert frame_count == protocol['expected'][split]['frames']
        a, cls = np.concatenate(chunks), np.concatenate(classes)
        data[split], class_data[split], scene_index[split] = a, cls, scenes
        strata_totals[split] = strata
        flat = a.reshape(-1, len(FIELDS))
        load = field(flat, 'gt_foreground')
        active = load > 0
        m = summarize(flat)
        m.update(frames=frame_count, scenes=len(chunks), active_queries=int(active.sum()),
            active_fraction=float(active.mean()),
            load_quantiles_all=dict(zip(('p10', 'p50', 'p90', 'p99'), np.quantile(load, [.1, .5, .9, .99]).tolist())),
            load_quantiles_active=dict(zip(('p10', 'p50', 'p90', 'p99'), np.quantile(load[active], [.1, .5, .9, .99]).tolist())),
            active_load_cv=float(load[active].std(dtype=np.float64)/load[active].mean(dtype=np.float64)),
            zero_assignment_with_core_coverage=int(((load == 0) & (field(flat, 'core_coverage_all_fg') > 0)).sum()),
            standardized_low_high=standardized(strata))
        m['matched_complexity']=matched_complexity(flat,cls.reshape(-1,17))
        m['standardized_low_high'].update(bootstrap_standardized(scene_pairs,scene_all))
        rank_bins = np.stack([deciles(g) for g in field(a, 'gt_foreground')]).reshape(-1)
        bucket = np.searchsorted(LOAD_EDGES, load, side='right')
        for b, name in enumerate(LOAD_NAMES):
            row = summarize(flat[bucket == b])
            bucket_rows.append(dict(split=split, load_bin=name, **row))
        for b in range(11):
            row = summarize(flat[rank_bins == b])
            row.update(budget_share=div(row['queries'], len(flat)),
                foreground_share=div(row['foreground'], m['foreground']),
                hole_share=div(row['holes'], m['holes']))
            decile_rows.append(dict(split=split, decile=b, **row))
        top = np.zeros(field(a, 'gt_foreground').shape, dtype=bool)
        order = np.argsort(field(a, 'gt_foreground'), axis=1, kind='stable')
        np.put_along_axis(top, order[:, -Q//5:], True, axis=1)
        masks = dict(no_assigned_foreground=~active, active_low20=(rank_bins >= 1)&(rank_bins <= 2),
            active_high20=rank_bins >= 9, all_query_top20=top.flatten(), all_queries=np.ones(len(flat), dtype=bool))
        for name, mask in masks.items():
            row = summarize(flat[mask])
            row.update(budget_share=div(row['queries'], len(flat)), foreground_share=div(row['foreground'], m['foreground']),
                hole_share=div(row['holes'], m['holes']), fp_share=div(row['dominant_fp'], m['dominant_fp']))
            group_rows.append(dict(split=split, group=name, **row))
        m['correlations'] = {p:bootstrap_mean([r['spearman_rho'] for r in correlations if r['split']==split and r['predictor']==p])
            for p in ('load', 'semantic_boundary_fraction', 'components', 'coarse_distance')}
        summary[split] = m
    complete = all(s in summary and {k:summary[s][k] for k in ('frames','scenes')} == protocol['expected'][s] for s in ('train','val'))
    if not partial:
        assert complete
        assert len(identities) == 34149
    csv_write(AGG / 'per_scene.csv', scene_rows)
    csv_write(AGG / 'load_bins.csv', bucket_rows)
    csv_write(AGG / 'active_load_deciles.csv', decile_rows)
    csv_write(AGG / 'query_groups.csv', group_rows)
    csv_write(AGG / 'scene_correlations.csv', correlations)
    np.savez_compressed(AGG / 'voxel_strata.npz', **strata_totals)
    combined={k:sum(m[k] for m in summary.values()) for k in ('frames','scenes','queries','foreground','holes','miss','correct','dominant_fp')}
    combined['hole_rate']=div(combined['holes'],combined['foreground'])
    combined['miss_rate']=div(combined['miss'],combined['foreground'])
    write(AGG / 'summary.json', dict(status='complete' if complete else 'partial',
        summary=summary, combined=combined, fields=FIELDS, completed_frame_identities=len(identities)))
    return summary, data, class_data, bucket_rows, decile_rows, group_rows, complete


def plots(data, decile_rows, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, split in zip(axes, ('train', 'val')):
        rows = [r for r in decile_rows if r['split']==split and r['decile']>0]
        if not rows:
            continue
        x = np.arange(1, 11)
        ax.plot(x, [r['foreground_share']*100 for r in rows], 'o-', label='GT foreground share')
        ax.plot(x, [r['hole_share']*100 for r in rows], 's-', label='Core-hole share')
        ax.plot(x, [r['budget_share']*100 for r in rows], '--', label='GS budget share')
        ax.set(title=split.title(), xlabel='Active-query load decile (low to high)', ylabel='Share of split total (%)', xticks=x)
        ax.legend(fontsize=9)
    fig.savefig(FIGURES / 'load_and_budget.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, split in zip(axes, ('train', 'val')):
        rows = [r for r in decile_rows if r['split']==split and r['decile']>0]
        if not rows:
            continue
        x = np.arange(1, 11)
        ax.stackplot(x, *[[r[k]*100 for r in rows] for k in ('own_core_rate','neighbor_only_rate','hole_rate')],
            labels=['Own-query core', 'Other queries only', 'No core'], colors=['#3b8c6e','#e2b54d','#c96262'], alpha=.9)
        ax.set(title=split.title(), xlabel='Active-query load decile (low to high)', ylabel='Assigned GT foreground (%)', ylim=(0,100), xticks=x)
        ax.legend(fontsize=9, loc='lower left')
    fig.savefig(FIGURES / 'coverage_by_load.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    from matplotlib.colors import LogNorm
    for ax, split in zip(axes, ('train', 'val')):
        if split not in data:
            continue
        a = data[split].reshape(-1, len(FIELDS))
        g = field(a, 'gt_foreground')
        mask = g > 0
        x = np.log2(g[mask])
        y = field(a, 'gt_no_core')[mask] / g[mask]
        hist, xe, ye = np.histogram2d(x, y, bins=(np.linspace(0,16,65), np.linspace(0,1,41)))
        image = ax.pcolormesh(xe, ye, np.ma.masked_equal(hist.T,0), norm=LogNorm(), cmap='viridis')
        ax.set(title=split.title(), xlabel='log2(assigned GT foreground voxels)', ylabel='Core-hole fraction')
        fig.colorbar(image, ax=ax, label='Query-frame count')
    fig.savefig(FIGURES / 'load_hole_distribution.png', dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True,sharey=True)
    lookup={(r['split'],r['group']):r for r in groups}
    upper=max(r[k] for r in groups if r['group'] in ('active_low20','active_high20')
              for k in ('hole_rate','miss_rate'))*100
    upper=np.ceil(upper*1.08/5)*5
    for ax,split in zip(axes,('train','val')):
        if (split,'active_low20') not in lookup:
            continue
        rows=[lookup[(split,k)] for k in ('active_low20','active_high20')]
        x=np.arange(2)
        ax.bar(x-.18,[r['hole_rate']*100 for r in rows],width=.36,label='No 1-sigma core',color='#c96262')
        ax.bar(x+.18,[r['miss_rate']*100 for r in rows],width=.36,label='Missed foreground',color='#527aa5')
        ax.set(title=split.title(),xticks=x,xticklabels=['Low-load 20%','High-load 20%'],ylabel='Assigned GT foreground (%)')
        ax.set_ylim(0,upper)
        ax.legend(fontsize=9)
    fig.savefig(FIGURES/'core_holes_and_misses.png',dpi=180)
    plt.close(fig)


def report(partial=False):
    summary, data, classes, buckets, decs, groups, complete = aggregate(partial)
    plots(data, decs, groups)
    lookup = {(r['split'],r['group']):r for r in groups}
    lines = ['# 固定 query 高斯预算与几何覆盖：全量统计结果', '',
        '**状态：完整统计已完成。**' if complete else '**状态：统计进行中；以下仅含已经完成的场景。**', '',
        '本轮冻结原始 8 帧 tiny 权重，统计每个 query 的前景负载、1σ 核心覆盖、邻居补足和重复覆盖。'
        '每帧仍为 600 个 query、每个 query 128 个 GS。没有训练模型，也没有改变预算分配。', '',
        '## 1. 数据与最直接的结果', '',
        '| 指标 | 训练集 | 验证集 |', '|---|---:|---:|']
    if 'val' in summary and summary['val']['frames']==6019:
        top=lookup[('val','all_query_top20')]
        a=summary['val']['standardized_low_high']
        delta=(a['high']['hole_rate']-a['low']['hole_rate'])*100
        lead=[f"**验证集负载最高的 20% query 只获得 20% 的 GS，却承担 {pct(top['foreground_share'])} 的前景、"
            f"集中了 {pct(top['hole_share'])} 的核心覆盖缺口。** 对齐类别、距离和粗层定位误差后，"
            f"高负载组的核心缺口率仍比低负载组高 {delta:.2f} 个百分点。", '',
            '这支持把预算与覆盖需求不匹配列为后续干预对象；本轮没有改变 GS 数量，尚不能确定固定配额的因果贡献。', '']
        lines[6:6]=lead
    def both(key, formatter=pct):
        return [formatter(summary[s].get(key)) if s in summary else '—' for s in ('train','val')]
    for name, key, fmt in (
        ('场景数','scenes',lambda x:f'{x:,}'), ('帧数','frames',lambda x:f'{x:,}'),
        ('query—帧总数','queries',lambda x:f'{x:,}'), ('分到 GT 前景的 query 比例','active_fraction',pct),
        ('GT 前景无任何 1σ 覆盖','hole_rate',pct), ('GT 前景漏检率','miss_rate',pct),
        ('由归属 query 自身提供核心覆盖','own_core_rate',pct),
        ('仅由其他 query 提供核心覆盖','neighbor_only_rate',pct)):
        a,b=both(key,fmt)
        lines.append(f'| {name} | {a} | {b} |')
    total_fg=sum(m['foreground'] for m in summary.values())
    total_holes=sum(m['holes'] for m in summary.values())
    total_miss=sum(m['miss'] for m in summary.values())
    lines += ['',f"合并总体共 {total_fg:,} 个有效 GT 前景体素；核心缺口率 {pct(div(total_holes,total_fg))}，"
        f"漏检率 {pct(div(total_miss,total_fg))}。合并比例按体素总数加权，训练／验证的比较仍分别报告。"]
    lines += ['', '“分到前景”由倒数第二级中心的最近邻划分决定；它不是模型显式的物体分工。'
        '“自身覆盖、邻居补足、无核心覆盖”三项互斥且穷尽，分母都是 camera mask 内全部 GT 前景。'
        '1σ 是统一的几何统计口径；没有核心覆盖时仍可能收到外围贡献，也可能预测正确。', '',
        '## 2. 相同预算承担的前景量有多大差异', '',
        '| 集合 | 有前景 query 的负载 P10 / P50 / P90 / P99 | 负载最高 20% query 承担的前景 | 集中的核心缺口 | 获得的预算 |',
        '|---|---:|---:|---:|---:|']
    for s,m in summary.items():
        q=m['load_quantiles_active'];top=lookup[(s,'all_query_top20')]
        lines.append(f"| {s} | {' / '.join(num(q[k],0) for k in ('p10','p50','p90','p99'))} | {pct(top['foreground_share'])} | {pct(top['hole_share'])} | {pct(top['budget_share'])} |")
    lines += ['', '上表按每帧全部 600 个 query 排序，最高 20% 始终为 120 个。负载单位是 GT 前景体素数。'
        '负载越高并不自动意味着表示更困难，因此继续比较覆盖率，并控制类别、距离和粗层定位误差。', '',
        '![各负载组的前景、缺口与预算份额](figures/query_budget_full/load_and_budget.png)', '',
        '图中十分位只在分到前景的 query 内划分；未分到前景的 query 仍保留预算，所以预算曲线不必达到 10%。', '',
        '## 3. 高负载 query 是否真的更缺覆盖', '',
        '| 集合与活跃 query 组别 | 核心缺口率 | 漏检率 | 自身核心覆盖 | 仅靠邻居补足 |',
        '|---|---:|---:|---:|---:|']
    for s in summary:
        for group,label in (('active_low20','负载最低 20%'),('active_high20','负载最高 20%')):
            r=lookup[(s,group)]
            lines.append(f"| {s}：{label} | {pct(r['hole_rate'])} | {pct(r['miss_rate'])} | {pct(r['own_core_rate'])} | {pct(r['neighbor_only_rate'])} |")
    lines += ['', '![核心覆盖缺口与漏检分开比较](figures/query_budget_full/core_holes_and_misses.png)']
    lines += ['', '![不同负载组的覆盖来源](figures/query_budget_full/coverage_by_load.png)', '',
        '十分位曲线还用于检查关系是否单调；不能只比较两端，就声称负载每增加一点都会使覆盖变差。', '',
        '### 对齐 GT 类别、水平距离与粗层定位误差后', '',
        '在 GT 类别 × 水平距离 × 倒数第二级最近中心距离的共同格子内，给低／高负载组使用同一套前景权重。'
        '该比较减少三类组成差异，不控制遮挡、物体实例大小或全部形状因素。', '',
        '| 集合 | 共同格子保留前景 | 低负载核心缺口 | 高负载核心缺口 | 高−低 | 低负载漏检 | 高负载漏检 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s,m in summary.items():
        a=m['standardized_low_high'];lo,hi=a['low'],a['high']
        lines.append(f"| {s} | {pct(a['retained_foreground_fraction'])} | {pct(lo['hole_rate'])} | {pct(hi['hole_rate'])} | {(hi['hole_rate']-lo['hole_rate'])*100:+.2f} pp | {pct(lo['miss_rate'])} | {pct(hi['miss_rate'])} |")
    lines += ['', '对齐后的高−低核心缺口差，按完整场景进行 1,000 次重采样：', '']
    for s,m in summary.items():
        lo,hi=m['standardized_low_high']['hole_gap_interval']
        lines.append(f'- {s}：95% 区间 **[{lo*100:+.2f}, {hi*100:+.2f}] pp**。')
    lines += ['', '### 场景内相关性', '',
        '每个场景先在有前景的 query—帧上计算 Spearman 相关系数，再对场景取等权平均。'
        '区间为 1,000 次场景重采样的 95% 区间，避免把同场景的许多 query 当成独立场景。', '',
        '| 集合 | 负载与核心缺口比例 | 类别边界比例与核心缺口 | 连通分量数与核心缺口 | 粗层距离与核心缺口 |',
        '|---|---:|---:|---:|---:|']
    for s,m in summary.items():
        values=[]
        for k in ('load','semantic_boundary_fraction','components','coarse_distance'):
            r=m['correlations'][k]
            values.append('—' if r['mean'] is None else f"{r['mean']:+.3f} [{r['interval'][0]:+.3f}, {r['interval'][1]:+.3f}]")
        lines.append(f"| {s} | {' | '.join(values)} |")
    lines += ['', '![负载与核心缺口的完整联合分布](figures/query_budget_full/load_hole_distribution.png)', '',
        '“连通分量”是有效前景网格的六连通区域，不是物体实例；接触路面的物体可能属于同一连通区域。'
        '空间跨度、类别数、前景／空区边界比例也已保存在逐 query 原始数据和分组表中。', '',
        '### 同样负载下，分散的前景区域是否更难覆盖', '',
        '将 query 分为只涉及一个前景连通区域、涉及至少两个区域。按 query 的绝对负载组、主要 GT 类别、'
        '平均水平距离组、平均粗层定位误差组对齐，再比较核心缺口。该口径的定位误差是 query 均值分组，'
        '与上一张体素级对齐表不同。', '',
        '| 集合 | 原始：单区域 / 多区域 | 对齐后：单区域 / 多区域 | 共同格子保留前景 |',
        '|---|---:|---:|---:|---:|']
    for s,m in summary.items():
        c=m['matched_complexity']
        lines.append(f"| {s} | {pct(c['unadjusted_single']['hole_rate'])} / {pct(c['unadjusted_multiple']['hole_rate'])} | {pct(c['single']['hole_rate'])} / {pct(c['multiple']['hole_rate'])} | {pct(c['retained_foreground_fraction'])} |")
    lines += ['', '## 4. 哪些 query 的覆盖产出较低，哪些覆盖发生重复', '',
        '| 集合与组别 | 预算占比 | 前景核心覆盖 / GS | 覆盖中与其他 query 共享 | 有前景核心接触的 GS | FP 主要贡献归属 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        for g,label in (('no_assigned_foreground','未分到 GT 前景'),('active_low20','活跃负载低 20%'),('active_high20','活跃负载高 20%'),('all_queries','全部')):
            r=lookup[(s,g)]
            lines.append(f"| {s}：{label} | {pct(r['budget_share'])} | {num(r['core_fg_coverage_per_gaussian'])} | {pct(r['shared_coverage_fraction'])} | {pct(r['core_gs_touch_foreground_fraction'])} | {pct(r['fp_share'])} |")
    lines += ['', '覆盖产出按该 query 在**整个有效前景**上的核心覆盖去重计数，而不是只看其归属区域。'
        '同一体素可以计入多个 query 的产出，不能把各 query 产出相加当成全局独有覆盖。'
        '“有前景核心接触的 GS”以该组全部 GS 为分母；FP 归属比例以该集合全部 FP 为分母。'
        '共享覆盖可能提高证据强度，不能直接视为浪费；FP 归属是最大单 GS 贡献者的父 query，不是删除后的因果收益。', '']
    for s,m in summary.items():
        lines.append(f"{s} 中，**{pct(m['zero_exclusive_core_fraction'])}** 的 query 没有独有的前景核心覆盖。"
            '单独移除其中任意一个 query，都不会新增 1σ 覆盖空洞；但不能据此同时移除这些 query，'
            '也不能保证其余 GS 的分数足以维持原预测。')
    lines += ['',
        '## 5. 没有核心覆盖，却仍正确检出的前景', '',
        '| 集合 | 无核心前景中类别也预测正确 | 主要 GS 平均幅度：无核心 / 有核心正确组 | 平均 opacity：无核心 / 有核心 | 主要 GS 距离：无核心 / 有核心 |',
        '|---|---:|---:|---:|---:|']
    for s,m in summary.items():
        lines.append(f"| {s} | {pct(m['no_core_correct_rate'])} | {num(m['no_core_correct_dom_amplitude'])} / {num(m['core_correct_dom_amplitude'])} | {num(m['no_core_correct_dom_opacity'],3)} / {num(m['core_correct_dom_opacity'],3)} | {num(m['no_core_correct_dom_mahalanobis'])}σ / {num(m['core_correct_dom_mahalanobis'])}σ |")
    lines += ['', '幅度为一个 GS 的 17 个 softplus 语义值之和；这里按体素统计主要贡献者。'
        '“正确”要求前景类别也正确；主要贡献者按最高前景分数选择，不使用 GT 类别。'
        '这张表是未做类别／距离对齐的描述性比较，不能把幅度均值差直接解释为训练偏好。'
        '逐负载组的同类指标、query 平均幅度和 FP 贡献见分组 CSV。', '',
        '### 高负载区域在没有核心时，是否仍能正确预测', '',
        '| 集合与组别 | 无核心前景中类别也正确 | 对齐后对应比例 | 无核心正确组主要 GS 幅度 |',
        '|---|---:|---:|---:|']
    for s,m in summary.items():
        for g,label,which in (('active_low20','低负载 20%','low'),('active_high20','高负载 20%','high')):
            r=lookup[(s,g)];std=m['standardized_low_high'][which]
            adjusted_correct=div(std['no_core_correct_fraction'],std['hole_rate'])
            lines.append(f"| {s}：{label} | {pct(r['no_core_correct_rate'])} | {pct(adjusted_correct)} | {num(r['no_core_correct_dom_amplitude'])} |")
    lines += ['', '对齐后比例由第3节相同 GT 组成下的“无核心且正确”比例除以“无核心”比例得到；'
        '幅度列仍为原始分组均值。覆盖缺口与最终漏检不是同一个指标，应保留两者可能方向不同的结果。', '',
        '## 6. 对固定预算假设的判断', '']
    if 'val' in summary:
        v=summary['val'];lo=lookup[('val','active_low20')];hi=lookup[('val','active_high20')]
        adjusted=v['standardized_low_high'];gap=hi['hole_rate']-lo['hole_rate']
        adjgap=adjusted['high']['hole_rate']-adjusted['low']['hole_rate']
        lines += [f"验证集高负载组相对低负载组的原始核心缺口差为 **{gap*100:+.2f} pp**；"
            f"对齐类别、距离和粗层定位误差后为 **{adjgap*100:+.2f} pp**。", '']
        if gap>0 and adjgap>0:
            lines += ['这轮数据支持“高负载与更大的核心覆盖缺口相关”。结合需求与预算份额差异，'
                '可以把固定展开数量列为值得进行数量干预的候选限制。', '']
        elif gap<=0 and adjgap<=0:
            lines += ['高负载组没有表现出更大的核心缺口，因此本轮数据不支持直接把“承担前景多、每组仍只有 128 个 GS”'
                '作为覆盖不足的重要解释。负载差异确实存在，但其空间分布、粗层定位或其他因素需要分开分析。', '']
        else:
            lines += ['原始比较与分层对齐后的方向不同，说明组成差异影响明显。当前统计不支持仅凭负载差异就认定固定展开数量是重要成因。', '']
        rho=v['correlations']['coarse_distance']['mean']
        lines += [f"还需要保留位置因素：验证集中，粗层最近中心距离与核心缺口比例的场景平均相关系数为 {rho:+.3f}。"
            '这说明数量分配研究应同时检查 GS 是否有能力到达目标区域；不能把定位问题全部改写成数量问题。'
            '不同相关系数的大小也不能解释为因果贡献份额。', '',
            '本轮同时区分了前景数量与前景分散程度。若一个 query 承担多个断开的区域，其数量需求可能不同于同样体素数的一整片前景；'
            '因此后续需求预测不宜只使用 GT 体素数作为唯一依据。', '']
    lines += ['本轮是冻结模型的统计，不包含 GS 数量干预，**尚不能证明固定数量分配是覆盖不足的重要原因**。'
        '要作因果判断，仍需在相同总 GS 数下重新分配，并检查接收区域新增的核心覆盖是否大于提供区域损失的覆盖。', '',
        '## 7. 复核与数据入口', '',
        '- [统计设计与指标定义](query_budget_full_design.md)',
        '- [全局汇总 JSON](../work_dirs/query_budget_full/aggregate/summary.json)',
        '- [逐场景汇总](../work_dirs/query_budget_full/aggregate/per_scene.csv)',
        '- [绝对负载分组](../work_dirs/query_budget_full/aggregate/load_bins.csv)',
        '- [活跃 query 负载十分位](../work_dirs/query_budget_full/aggregate/active_load_deciles.csv)',
        '- [主要 query 组别](../work_dirs/query_budget_full/aggregate/query_groups.csv)',
        '- [逐场景相关性](../work_dirs/query_budget_full/aggregate/scene_correlations.csv)',
        '- [保存数据审计](../work_dirs/query_budget_full/audit.json)',
        '- [统计脚本](../tools/stat_query_budget.py)', '',
        '逐帧、逐 query 的原始数据位于 `work_dirs/query_budget_full/scenes/{train,val}/*.npz`：'
        '`metrics[帧,600,指标]`、`class_counts[帧,600,17]`，指标顺序记录在 protocol.json。'
        '同名 JSON 保存场景名称、帧身份、样本 token 和逐帧校验。', '',
        '校验包含独立小场景穷举、四个真实帧的生产覆盖算子交叉检查，以及所有场景对旧统计的逐帧混淆矩阵、'
        '逐场景原生／1σ／3σ 数量总和和零核心覆盖计数复现。报告比例先累加分子、分母再计算。']
    content='\n'.join(lines)+'\n'
    content=content.replace('| train','| 训练集').replace('| val','| 验证集')
    content=content.replace('- train：','- 训练集：').replace('- val：','- 验证集：')
    content=content.replace('\ntrain 中','\n训练集中').replace('\nval 中','\n验证集中')
    REPORT.write_text(content)
    if complete:
        write(RUN / 'complete.json', dict(status='complete', scope={s:{k:summary[s][k] for k in ('frames','scenes')} for s in summary},
            frame_identities=34149, checkpoint_sha256=read(RUN/'protocol.json')['checkpoint_sha256'],
            report=str(REPORT.relative_to(ROOT)), design='analysis/query_budget_full_design.md'))
    print(f'Query report written; complete={complete}; frames={sum(m["frames"] for m in summary.values())}', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial', action='store_true')
    args=p.parse_args()
    report(args.partial)
