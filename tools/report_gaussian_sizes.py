"""Aggregate the matched frozen tiny/small/large full-validation census."""
from pathlib import Path
import csv
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.eval_gaussian_sizes import RUN, CLASS_NAMES, CLASS_COLUMNS, paths
from tools.stat_boundary_gaussians import read, write, sha

MODELS = ('tiny', 'small', 'large')
AGG = RUN / 'aggregate'
REPORT = ROOT / 'analysis/gaussian_sizes_full_val_results.md'
FIGURES = ROOT / 'analysis/figures/gaussian_sizes_full_val'


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def performance(cm):
    tp = np.diag(cm)[:17]
    gt = cm[:17].sum(1)
    class_fp = cm.sum(0)[:17] - tp
    detected = cm[:17, :17].sum()
    fn, fp = cm[:17, 17].sum(), cm[17, :17].sum()
    iou = np.divide(100 * tp, gt + class_fp, out=np.zeros(17, float), where=(gt + class_fp) > 0)
    safe = lambda n, d: float(100 * n / d) if d else 0.
    return dict(mIoU=float(iou.mean()),
        IoU=safe(detected, detected + fn + fp),
        GTfg=int(gt.sum()), FN=int(fn), FP=int(fp),
        FN_rate=safe(fn, gt.sum()), FP_over_GTfg=safe(fp, gt.sum()),
        FP_over_predfg=safe(fp, detected + fp),
        detected_semantic_accuracy=safe(tp.sum(), detected))


def summarize_queries(load, core, k):
    q = load.shape[1]
    totals = load.sum(1)
    ordered = np.sort(load, axis=1)
    good = totals > 0
    gini = np.zeros(len(load), float)
    gini[good] = (2 * (ordered[good] * np.arange(1, q + 1)).sum(1) /
                  (q * totals[good]) - (q + 1) / q)
    cv = np.divide(load.std(1), load.mean(1), out=np.zeros(len(load)), where=good)
    result = dict(queries=q, children=k, GS=q*k,
        mean_load=float(load.mean()), median_load=float(np.median(load)),
        p90_load=float(np.quantile(load, .9)), p99_load=float(np.quantile(load, .99)),
        gini_frame_mean=float(gini.mean()), cv_frame_mean=float(cv.mean()),
        zero_load=float(100 * (load == 0).mean()),
        zero_core=float(100 * (core == 0).mean()), core_le20=float(100 * (core <= 20).mean()),
        mean_core=float(core.mean()), median_core=float(np.median(core)),
        p90_core=float(np.quantile(core, .9)),
        median_load_per_GS=float(np.median(load) / k),
        p90_load_per_GS=float(np.quantile(load, .9) / k),
        p99_load_per_GS=float(np.quantile(load, .99) / k),
        median_core_per_GS=float(np.median(core) / k),
        p90_core_per_GS=float(np.quantile(core, .9) / k))
    descending = ordered[:, ::-1]
    for i in range(5):
        result[f'share_{i*20}_{(i+1)*20}'] = float(
            100 * descending[:, i*q//5:(i+1)*q//5].sum() / totals.sum())
    return result


def aggregate(size):
    folder = RUN / size
    protocol = read(folder / 'protocol.json')
    complete = read(folder / 'complete.json')
    assert complete['status'] == 'complete' and complete['frames'] == 6019 and complete['scenes'] == 150
    assert protocol['checkpoint_sha256'] == complete['checkpoint_sha256'] == sha(paths(size)[1])
    assert read(folder / 'pilot.json')['status'] == 'passed'
    for rel, digest in protocol['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    chunks, indices, tokens, scenes = {}, [], [], []
    totals = dict(confusion=np.zeros((18, 18), np.int64), class_coverage=np.zeros((17, 8), np.int64))
    files = sorted((folder / 'scenes').glob('*.json'))
    assert len(files) == 150
    for path in files:
        meta = read(path)
        assert meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
        indices += meta['indices']
        tokens += meta['tokens']
        with np.load(path.with_suffix('.npz')) as z:
            cm, cc = z['confusion'].sum(0), z['class_coverage'].sum(0)
            np.testing.assert_array_equal(cm[:17].sum(1), cc[:, 0])
            np.testing.assert_array_equal(z['load_sixth'].sum(1), z['class_coverage'][:, :, 0].sum(1))
            np.testing.assert_array_equal(z['load_fifth'].sum(1), z['load_sixth'].sum(1))
            assert int(z['query_core'].sum()) >= int(cc[:, 1].sum())
            assert int(z['core_top20_union'].sum()) <= int(cc[:, 1].sum())
            totals['confusion'] += cm
            totals['class_coverage'] += cc
            for key in ('load_sixth', 'load_fifth', 'query_core', 'core_top20_union'):
                chunks.setdefault(key, []).append(z[key])
            scenes.append(dict(scene=meta['scene'], frames=meta['frames'], confusion=cm,
                class_coverage=cc, **performance(cm)))
    assert sorted(indices) == list(range(6019)) and len(set(tokens)) == 6019
    order = np.argsort(indices)
    arrays = {k: np.concatenate(v)[order] for k, v in chunks.items()}
    arrays.update(totals, frame_indices=np.asarray(indices)[order])
    assert arrays['load_sixth'].shape == arrays['query_core'].shape == (6019, protocol['queries'])
    cc = totals['class_coverage']
    assert np.all(cc[:, 1] <= cc[:, 2]) and np.all(cc[:, 2] <= cc[:, 3])
    assert np.all(cc[:, 4:7] <= cc[:, 1:4])
    metrics = performance(totals['confusion'])
    for j, name in enumerate(CLASS_COLUMNS[1:], 1):
        metrics[name] = float(100 * cc[:, j].sum() / cc[:, 0].sum())
    metrics.update(summarize_queries(arrays['load_sixth'], arrays['query_core'], protocol['children']))
    metrics['fifth_top20'] = float(100 * np.sort(arrays['load_fifth'], axis=1)[:, -protocol['queries']//5:].sum() / cc[:, 0].sum())
    metrics['core_top20_union'] = float(100 * arrays['core_top20_union'].sum() / cc[:, 0].sum())
    if size in ('tiny', 'small'):
        historic = read(ROOT / 'work_dirs/tiny_small_baseline_classwise_comparison/summary.json')['models'][size]
        for key in ('mIoU', 'IoU'):
            assert metrics[key] == historic[key], (size, key)
        assert abs(metrics['geo1'] - historic['overall_coverage']) < 1e-12
        with (ROOT / 'work_dirs/tiny_small_baseline_classwise_comparison/per_class.csv').open(encoding='utf-8-sig') as stream:
            old = list(csv.DictReader(stream))
        np.testing.assert_array_equal(cc[:, 1], [int(r[f'{size}_covered']) for r in old])
    return protocol, arrays, metrics, scenes, np.asarray(tokens)[order].tolist()


def per_class(size, arrays):
    cm, cc = arrays['confusion'], arrays['class_coverage']
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        gt = int(cm[i].sum())
        tp = int(cm[i, i])
        fp_empty = int(cm[17, i])
        fp_other = int(cm[:17, i].sum()) - tp
        rows.append(dict(model=size, class_id=i, class_name=name, GT=gt, TP=tp,
            FP_empty=fp_empty, FP_other_foreground=fp_other, FN_to_empty=int(cm[i, 17]),
            IoU=100 * tp / (gt + fp_empty + fp_other),
            **{k:100 * int(cc[i, j]) / gt for j, k in enumerate(CLASS_COLUMNS[1:], 1)}))
    return rows


def per_query(size, arrays):
    load, core = arrays['load_sixth'], arrays['query_core']
    k = read(RUN / size / 'protocol.json')['children']
    result = []
    load_quantiles = np.quantile(load, [.5, .9], axis=0)
    core_quantiles = np.quantile(core, [.5, .9], axis=0)
    for i in range(load.shape[1]):
        result.append(dict(model=size, query_id=i, GS_per_query=k,
            mean_assigned=float(load[:, i].mean()), median_assigned=float(load_quantiles[0, i]),
            p90_assigned=float(load_quantiles[1, i]), zero_assigned=100 * float((load[:, i] == 0).mean()),
            mean_core=float(core[:, i].mean()), median_core=float(core_quantiles[0, i]),
            p90_core=float(core_quantiles[1, i]), zero_core=100 * float((core[:, i] == 0).mean()),
            core_le20=100 * float((core[:, i] <= 20).mean())))
    return result


def intervals(scene_data):
    rng = np.random.default_rng(928)
    weights = rng.multinomial(150, np.full(150, 1/150), size=1500)
    sampled = {}
    for name, records in scene_data.items():
        records = sorted(records, key=lambda r:r['scene'])
        cm = (weights @ np.stack([r['confusion'].reshape(-1) for r in records])).reshape(-1, 18, 18)
        gt = cm[:, :17].sum(2)
        tp = np.diagonal(cm, axis1=1, axis2=2)[:, :17]
        union = gt + cm.sum(1)[:, :17] - tp
        cover = weights @ np.stack([r['class_coverage'][:, [0, 1]].sum(0) for r in records])
        binary_tp = cm[:, :17, :17].sum((1, 2))
        iou = np.divide(100 * tp, union, out=np.zeros_like(tp, dtype=float), where=union > 0)
        sampled[name] = dict(mIoU=iou.mean(1),
            IoU=100 * binary_tp / (binary_tp + cm[:, :17, 17].sum(1) + cm[:, 17, :17].sum(1)),
            geo1=100 * cover[:, 1] / cover[:, 0])
    return {a+'-'+b:{key:np.quantile(sampled[a][key]-sampled[b][key], [.025, .975]).tolist()
                      for key in ('mIoU', 'IoU', 'geo1')}
            for a, b in [('large', 'small'), ('large', 'tiny'), ('small', 'tiny')]}


def figures(metrics, arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = ['#4477aa', '#dd9955', '#449977']
    plt.rcParams.update({'font.size':10, 'font.family':'DejaVu Sans'})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout='constrained')
    for ax, key, title in zip(axes, ['mIoU', 'IoU', 'geo1', 'same1'],
            ['mIoU', 'Binary occupancy IoU', '1-sigma coverage', 'Same-class 1-sigma']):
        values = [metrics[s][key] for s in MODELS]
        bars = ax.bar(MODELS, values, color=colors)
        ax.bar_label(bars, fmt='%.3f', padding=3)
        ax.set(title=title, ylabel='%', ylim=(0, max(values)*1.15))
    fig.suptitle('Original pretrained checkpoints | 6,019 frames | camera-valid domain')
    for suffix in ('png', 'pdf'):
        fig.savefig(FIGURES / f'overall.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
    thresholds = np.array([0, 1, 5, 10, 20, 50, 100, 200, 500, 1000])
    normalized = np.array([0, .05, .1, .25, .5, 1, 2, 4, 8])
    for name, color in zip(MODELS, colors):
        load, core = arrays[name]['load_sixth'], arrays[name]['query_core']
        rank_sum = np.sort(load, axis=1)[:, ::-1].sum(0)
        axes[0].plot(np.arange(len(rank_sum)+1)/len(rank_sum), np.r_[0, np.cumsum(rank_sum)/rank_sum.sum()], label=name, color=color)
        axes[1].plot(thresholds, [100*(core <= t).mean() for t in thresholds], label=name, color=color)
        k = metrics[name]['children']
        axes[2].plot(normalized, [100*(core <= t*k).mean() for t in normalized], label=name, color=color)
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(title='Final-stage nearest ownership', xlabel='Fraction of queries, ranked per frame', ylabel='Cumulative GT share')
    axes[1].set(title='Actual 1-sigma contacts per query', xlabel='Distinct GT foreground contacts', ylabel='Queries at or below threshold (%)', xscale='symlog')
    axes[2].set(title='Contacts normalized by children count', xlabel='Distinct GT contacts / GS per query', ylabel='Queries at or below threshold (%)')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    for suffix in ('png', 'pdf'):
        fig.savefig(FIGURES / f'queries.{suffix}', dpi=180)
    plt.close(fig)


def markdown(metrics, classes, ci):
    t, s, l = (metrics[n] for n in MODELS)
    lines = ['# Tiny、Small、Large：完整验证对比结果', '',
        '**三份原始预训练权重均已完成完整验证：150个场景、6,019帧。没有重新训练或调整empty bias。**', '',
        f"Large的mIoU为 **{l['mIoU']:.3f}%**，相对Small为 **{l['mIoU']-s['mIoU']:+.3f}**、相对Tiny为 **{l['mIoU']-t['mIoU']:+.3f}** 个百分点。"
        f"1σ几何覆盖分别变化 **{l['geo1']-s['geo1']:+.3f}**、**{l['geo1']-t['geo1']:+.3f}** 个百分点。", '',
        f"最终第六级Top20% query的最近归属份额为：Tiny **{t['share_0_20']:.3f}%**、Small **{s['share_0_20']:.3f}%**、Large **{l['share_0_20']:.3f}%**。"
        '这是前景的空间归属集中度，不是有效GS数量占比；需与实际覆盖和最终性能分别阅读。', '',
        '三者使用256×704、8帧输入及默认camera有效区域，最终均预测76,800个GS。这里的Small是原始权重，与此前额外微调的B不同。定义和核验见[统计设计](gaussian_sizes_full_val_design.md)。', '',
        '## 1. 整体预测与覆盖', '',
        '| 模型 | Query×GS/query | mIoU | 二值IoU | 1σ覆盖 | 2σ覆盖 | 3σ覆盖 | 同类1σ覆盖 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        m = metrics[name]
        lines.append(f"| {name} | {m['queries']}×{m['children']} | {m['mIoU']:.3f}% | {m['IoU']:.3f}% | {m['geo1']:.3f}% | {m['geo2']:.3f}% | {m['geo3']:.3f}% | {m['same1']:.3f}% |")
    lines += ['', '| 比较 | ΔmIoU | Δ二值IoU | Δ1σ覆盖 | Δ同类1σ | Δ漏检率 | Δ二值FP/GT前景 |', '|---|---:|---:|---:|---:|---:|---:|']
    for a, b in [('large', 'small'), ('large', 'tiny'), ('small', 'tiny')]:
        lines.append('| '+a+'−'+b+' | '+' | '.join(f"{metrics[a][k]-metrics[b][k]:+.3f}" for k in ['mIoU','IoU','geo1','same1','FN_rate','FP_over_GTfg'])+' |')
    lines += ['', '差值单位为百分点。几何覆盖不要求类别正确；同类覆盖还要求至少一个GS的预测类别与GT相同。两者均不要求该GS主导最终渲染。', '',
        '![整体性能和覆盖](figures/gaussian_sizes_full_val/overall.png)', '',
        '| 模型 | 漏检/GT前景 | 二值FP/GT前景 | 二值FP/预测前景 | 已检出前景语义准确率 |', '|---|---:|---:|---:|---:|']
    for name in MODELS:
        lines.append('| '+name+' | '+' | '.join(f"{metrics[name][k]:.3f}%" for k in ['FN_rate','FP_over_GTfg','FP_over_predfg','detected_semantic_accuracy'])+' |')
    lines += ['', '## 2. 逐类IoU与1σ几何覆盖', '',
        '每格为“IoU / 1σ覆盖率”，绝对值单位为%；最后两列分别为Large相对Small、Tiny的变化，单位为百分点。差值先用原始数值计算，再取三位小数。', '',
        '| 类别 | Tiny | Small | Large | Large−Small | Large−Tiny |', '|---|---:|---:|---:|---:|---:|']
    for i, name in enumerate(CLASS_NAMES):
        a, b, c = (classes[n][i] for n in MODELS)
        lines.append(f"| {name} | {a['IoU']:.3f} / {a['geo1']:.3f} | {b['IoU']:.3f} / {b['geo1']:.3f} | {c['IoU']:.3f} / {c['geo1']:.3f} | {c['IoU']-b['IoU']:+.3f} / {c['geo1']-b['geo1']:+.3f} | {c['IoU']-a['IoU']:+.3f} / {c['geo1']-a['geo1']:+.3f} |")
    for reference in ['small', 'tiny']:
        ni = sum(a['IoU']>b['IoU'] for a,b in zip(classes['large'],classes[reference]))
        nc = sum(a['geo1']>b['geo1'] for a,b in zip(classes['large'],classes[reference]))
        lines += ['', f'Large相对{reference}有 **{ni}/17** 类IoU提高、**{nc}/17** 类1σ覆盖提高。']
    lines += ['', '总体覆盖按GT前景数量加权，不等于逐类覆盖的简单平均。完整的同类覆盖、2σ、3σ及类别TP/FP计数见[逐类CSV](../work_dirs/gaussian_sizes_full_val/aggregate/per_class.csv)。', '',
        '## 3. 全部query的负载分布', '',
        '本节主口径统一为最终第六级GS的最近中心归属。每个GT前景体素只归给一个父query；归属不要求实际覆盖或类别正确。每帧独立排序后分成五组，各组累计计数再除以全验证GT前景数。', '',
        '| 模型 | 前20% | 第二20% | 第三20% | 第四20% | 最后20% | Gini |', '|---|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        m = metrics[name]
        lines.append('| '+name+' | '+' | '.join(f"{m[f'share_{i*20}_{(i+1)*20}']:.3f}%" for i in range(5))+f" | {m['gini_frame_mean']:.4f} |")
    lines += ['', 'Gini先在每帧全部query上计算，再对帧平均。Top20份额或Gini较小表示最近归属更均衡，本身不等于覆盖或预测更好。', '',
        '| 模型 | 第五级Top20归属 | 最终第六级Top20归属 | 零最近归属query | 零1σ接触query | 1σ接触≤20 | 核心覆盖最高20%的并集/GT前景 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        m = metrics[name]
        lines.append('| '+name+' | '+' | '.join(f"{m[k]:.3f}%" for k in ['fifth_top20','share_0_20','zero_load','zero_core','core_le20','core_top20_union'])+' |')
    lines += ['', '零/低接触比例对全部帧、全部query统计；camera外的GS也可能造成零接触，不能直接称为无效。每query子GS数不同，≤20不能脱离容量直接比较。最后一列按实际核心接触量重新排序，并对前20%的覆盖去重；其余80%与它可以重叠，不能直接用100%相减。', '',
        '| 模型 | 最近归属P50/P90/P99 | 归属数/子GS数P50/P90/P99 | 核心接触P50/P90 | 核心接触/子GS数P50/P90 |', '|---|---:|---:|---:|---:|']
    for name in MODELS:
        m = metrics[name]
        lines.append(f"| {name} | {m['median_load']:.1f} / {m['p90_load']:.1f} / {m['p99_load']:.1f} | {m['median_load_per_GS']:.3f} / {m['p90_load_per_GS']:.3f} / {m['p99_load_per_GS']:.3f} | {m['median_core']:.1f} / {m['p90_core']:.1f} | {m['median_core_per_GS']:.3f} / {m['p90_core_per_GS']:.3f} |")
    lines += ['', '核心接触是一个query全部子GS覆盖的不同GT体素数；不同query之间可以重叠。“接触数/子GS数”是辅助归一化，不是单GS独占贡献或已证明的容量上限。', '',
        '![所有query的归属与实际接触分布](figures/gaussian_sizes_full_val/queries.png)', '',
        '## 4. 结果核验与比较范围', '',
        '三者严格加载已有权重，缓存与原生输入的最终logit和GS逐元素一致。覆盖位集与独立实现一致，并对真实位置枚举全部GS核对计数。Tiny/Small逐帧复现历史第五级负载和实际核心接触，Small还逐帧复现第六级负载；两者整体mIoU、二值IoU和逐类1σ覆盖与历史基线一致。', '',
        '| 比较 | ΔmIoU的95%场景区间 | Δ二值IoU的95%场景区间 | Δ1σ覆盖的95%场景区间 |', '|---|---|---|---|']
    for key, data in ci.items():
        lines.append('| '+key+' | '+' | '.join(f"[{data[k][0]:+.3f}, {data[k][1]:+.3f}]" for k in ['mIoU','IoU','geo1'])+' |')
    lines += ['', '按相同150个场景配对bootstrap重采样1,500次，种子928；区间描述固定权重之间的场景差异，不包含训练随机性。三份权重的差异也不能孤立归因于query配额。统计耗时包含缓存与额外诊断，不能当作部署FPS。', '',
        '## 5. 数据与复现', '',
        '- [整体指标](../work_dirs/gaussian_sizes_full_val/aggregate/metrics.csv)及[两两差值](../work_dirs/gaussian_sizes_full_val/aggregate/metric_differences.csv)。',
        '- [逐类指标](../work_dirs/gaussian_sizes_full_val/aggregate/per_class.csv)及[逐类差值](../work_dirs/gaussian_sizes_full_val/aggregate/per_class_differences.csv)。',
        '- [五等分query负载](../work_dirs/gaussian_sizes_full_val/aggregate/query_quintiles.csv)与[全部6,600个query槽位统计](../work_dirs/gaussian_sizes_full_val/aggregate/per_query.csv)。',
        '- 全部逐帧数组：[Tiny](../work_dirs/gaussian_sizes_full_val/aggregate/tiny.npz)、[Small](../work_dirs/gaussian_sizes_full_val/aggregate/small.npz)、[Large](../work_dirs/gaussian_sizes_full_val/aggregate/large.npz)。`load_sixth`与`query_core`分别保存最近归属和实际核心接触，第一维按原验证帧索引0～6,018排列。',
        '- [汇总与检查结果](../work_dirs/gaussian_sizes_full_val/summary.json)。每个模型目录另存权重、配置、标注和测量代码哈希。', '',
        '不同模型的query编号不能一一对应，原始数组用于比较分布；不将同编号解释为同一物理对象。', '',
        '```bash', 'python3.10 tools/eval_gaussian_sizes.py', 'python3.10 tools/report_gaussian_sizes.py', '```', '']
    return '\n'.join(lines)


def main():
    AGG.mkdir(parents=True, exist_ok=True)
    protocols, arrays, metrics, scenes, tokens, classes = {}, {}, {}, {}, {}, {}
    for name in MODELS:
        protocols[name], arrays[name], metrics[name], scenes[name], tokens[name] = aggregate(name)
        classes[name] = per_class(name, arrays[name])
        np.savez_compressed(AGG / f'{name}.npz', **arrays[name])
    assert tokens['tiny'] == tokens['small'] == tokens['large']
    assert len({p['annotation_sha256'] for p in protocols.values()}) == 1
    for name in MODELS[1:]:
        np.testing.assert_array_equal(arrays[name]['confusion'].sum(1), arrays['tiny']['confusion'].sum(1))
    comparisons = [('large', 'small'), ('large', 'tiny'), ('small', 'tiny')]
    overall_diff = [dict(subject=a, reference=b, **{k:metrics[a][k]-metrics[b][k] for k in metrics[a]}) for a,b in comparisons]
    class_diff = [dict(subject=a, reference=b, class_id=i, class_name=CLASS_NAMES[i],
        **{k:classes[a][i][k]-classes[b][i][k] for k in ['IoU', *CLASS_COLUMNS[1:]]})
        for a,b in comparisons for i in range(17)]
    quintiles = [dict(model=name, group=f'{i*20}-{(i+1)*20}%', GT_share=metrics[name][f'share_{i*20}_{(i+1)*20}'])
                 for name in MODELS for i in range(5)]
    query_rows = [r for name in MODELS for r in per_query(name, arrays[name])]
    assert len(query_rows) == 6600
    for name, rows in [('metrics',[dict(model=n,**metrics[n]) for n in MODELS]),
            ('metric_differences',overall_diff), ('per_class',[r for n in MODELS for r in classes[n]]),
            ('per_class_differences',class_diff), ('per_query',query_rows), ('query_quintiles',quintiles)]:
        csv_write(AGG / f'{name}.csv', rows)
    scene_rows = [dict(model=name, **{k:v for k,v in r.items() if not isinstance(v,np.ndarray)})
                  for name in MODELS for r in scenes[name]]
    csv_write(AGG / 'per_scene.csv', scene_rows)
    ci = intervals(scenes)
    figures(metrics, arrays)
    REPORT.write_text(markdown(metrics, classes, ci))
    result = dict(status='complete', models=metrics, intervals=ci,
        validation_frames=6019, validation_scenes=150,
        all_query_entries={n:6019*protocols[n]['queries'] for n in MODELS},
        checks=dict(original_checkpoints=True, full_frame_token_alignment=True,
            same_per_class_GT=True, historical_tiny_small_metrics_exact=True,
            historical_query_arrays_exact=True, no_training_or_bias_calibration=True),
        protocols={n:sha(RUN/n/'protocol.json') for n in MODELS},
        report_source_sha256=sha(Path(__file__)))
    write(RUN / 'summary.json', result)
    write(RUN / 'status.json', dict(status='complete', models=list(MODELS), report=str(REPORT.relative_to(ROOT))))
    print(json.dumps(dict(status='complete', models=metrics), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
