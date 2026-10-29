"""Full-validation report comparing the uniform classwise Set-loss ablation."""
from pathlib import Path
import csv, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.unweighted_strict_classwise_set_experiment import *
from tools.report_classwise_chamfer_mini import metrics, csv_write
from tools.report_semantic_field_mini import paired_interval
from tools.report_gaussian_geometry_causal import CLASS_NAMES

REPORT = ROOT / 'analysis/unweighted_strict_classwise_set_mini_results.md'
NAMES = ('baseline', 'classwise', 'strict_classwise', 'unweighted_classwise')
LABELS = {
    'baseline': 'B：原几何＋原分类',
    'classwise': 'C：逐类几何＋原分类',
    'strict_classwise': 'D：逐类几何＋逐类分类（类别加权）',
    'unweighted_classwise': 'E：逐类几何＋逐类分类（类间等权）',
}
FOLDERS = {
    'baseline': BASELINE_EVAL,
    'classwise': CLASSWISE_RUN / 'eval/classwise',
    'strict_classwise': STRICT_RUN / 'eval/strict_classwise',
    'unweighted_classwise': RUN / 'eval/unweighted_classwise',
}


def aggregate(arm):
    folder = FOLDERS[arm]
    meta = json.loads((folder / 'complete.json').read_text())
    paths = sorted((folder / 'scenes').glob('*.npz'))
    assert len(paths) == 150
    total, scenes, indices, tokens, loads, cores = {}, [], [], [], [], []
    for path in paths:
        info = json.loads(path.with_suffix('.json').read_text())
        assert info['checkpoint_sha256'] == meta['checkpoint_sha256']
        indices.extend(info['indices']); tokens.extend(info['tokens'])
        with np.load(path) as data:
            row = {key: data[key].sum(0) for key in data.files
                   if key not in ('load', 'query_core', 'aux_geometry')}
            loads.append(data['load']); cores.append(data['query_core'])
        for key, value in row.items():
            total[key] = total.get(key, np.zeros_like(value)) + value
        scenes.append((path.stem, row))
    assert sorted(indices) == list(range(6019)) and len(set(tokens)) == 6019
    total['load'] = np.concatenate(loads)
    total['query_core'] = np.concatenate(cores)
    assert total['load'].shape == total['query_core'].shape == (6019, 1200)
    out = RUN / 'aggregate'; out.mkdir(exist_ok=True)
    np.savez_compressed(out / f'{arm}.npz', **total)
    return total, scenes


def trace_audit():
    trace = RUN / 'train/unweighted_classwise/trace.jsonl'
    old_trace = STRICT_RUN / 'train/strict_classwise/trace.jsonl'
    sums, norms, order = {}, [], []
    first = None
    with trace.open() as stream:
        for line in stream:
            row = json.loads(line); first = first or row; order.append(row['index'])
            for key, value in row['losses'].items():
                sums[key] = sums.get(key, 0.) + value
            if 'grad_norm' in row: norms.append(row['grad_norm'])
    with old_trace.open() as stream:
        oldfirst = json.loads(next(stream))
        oldorder = [oldfirst['index']]
        oldorder.extend(json.loads(line)['index'] for line in stream)
    assert order == oldorder and len(order) == len(set(order)) == 4096
    assert len(norms) == 2048
    occ = {key: abs(value - oldfirst['losses'][key])
           for key, value in first['losses'].items()
           if key == 'OccupancyLoss' or key.startswith('OccupancyLoss/')}
    assert occ and max(occ.values()) == 0.
    means = {key: value / 4096 for key, value in sums.items()}
    return dict(data_order_exact=True, first_occupancy_max_difference=max(occ.values()),
        means=means, gradient_norm_median=float(np.median(norms)),
        gradient_norm_p95=float(np.quantile(norms, .95)),
        gradient_clip_percent=100 * float((np.asarray(norms) > 35).mean()),
        mean_classwise_geometry=sum(v for k, v in means.items()
            if k.endswith(('/pred_to_gt', '/gt_to_pred'))),
        mean_classwise_classification=sum(v for k, v in means.items()
            if k.endswith('/classification')))


def plot(all_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    panels = [('mIoU', 'mIoU'), ('IoU', 'Binary occupancy IoU'),
              ('geo1', '1-sigma coverage'), ('same1', 'Same-class 1-sigma coverage'),
              ('FN_rate', 'Misses / GT foreground'), ('top20_nearest_share', 'Top20 nearest-query share')]
    colors = ['#8a939b', '#cc8c48', '#397aa4', '#4c9b78']
    for ax, (key, title) in zip(axes.flat, panels):
        values = [all_metrics[name][key] for name in NAMES]
        ax.bar(range(4), values, color=colors)
        ax.set_xticks(range(4), ['B', 'C', 'D', 'E'])
        ax.set_title(title); ax.set_ylabel('%'); ax.set_ylim(0, max(values) * 1.2)
        for i, value in enumerate(values):
            ax.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle('Matched 4096-frame fine-tuning; full 6019-frame camera validation')
    folder = ROOT / 'analysis/figures'; folder.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'unweighted_strict_classwise_set_mini_comparison.{ext}', dpi=180)
    plt.close(fig)


def report():
    evaluation = RUN / 'eval/unweighted_classwise/complete.json'
    if not evaluation.exists():
        REPORT.write_text('# 逐类 Set loss 去除类别加权：实验结果\n\n**实验尚未完成。**\n')
        return
    assert_protocol()
    arrays, scenes, all_metrics = {}, {}, {}
    for name in NAMES:
        arrays[name], scenes[name] = aggregate(name)
        all_metrics[name] = metrics(arrays[name])
    for name in NAMES[1:]:
        np.testing.assert_array_equal(arrays[name]['class_coverage'][:, 0],
                                      arrays['baseline']['class_coverage'][:, 0])
    intervals = {name: paired_interval(scenes[name], scenes['unweighted_classwise'])
                 for name in NAMES[:3]}
    csv_write(RUN / 'aggregate/metrics.csv',
              [dict(arm=name, **all_metrics[name]) for name in NAMES])
    class_rows, groups = [], []
    for name in NAMES:
        cm, coverage = arrays[name]['confusion'], arrays[name]['class_coverage']
        for c, label in enumerate(CLASS_NAMES):
            tp = cm[c, c]; union = cm[c].sum() + cm[:, c].sum() - tp; gt = coverage[c, 0]
            class_rows.append(dict(arm=name, class_id=c, class_name=label, GT=int(gt),
                IoU=100 * tp / max(union, 1), core1=100 * coverage[c, 1] / max(gt, 1),
                same_core1=100 * coverage[c, 4] / max(gt, 1),
                FN=100 * cm[c, 17] / max(gt, 1), class_FP=int(cm[:, c].sum() - tp),
                FP_from_empty=int(cm[17, c])))
        sorted_load = np.sort(arrays[name]['load'], axis=1)[:, ::-1]
        for q in range(5):
            block = sorted_load[:, q*240:(q+1)*240]
            groups.append(dict(arm=name, group=f'{q*20}-{(q+1)*20}%',
                GT_count=int(block.sum()), GT_share=100*block.sum()/coverage[:, 0].sum(),
                mean_GT_per_query=float(block.mean())))
    csv_write(RUN / 'aggregate/per_class.csv', class_rows)
    csv_write(RUN / 'aggregate/query_quintiles.csv', groups)
    audit = trace_audit(); write(RUN / 'optimization_audit.json', audit)
    training = json.loads((RUN / 'train/unweighted_classwise/complete.json').read_text())
    summary = dict(status='complete', metrics=all_metrics, paired_comparisons=intervals,
        training=training, optimization_audit=audit, query_quintiles=groups,
        classwise_reduction='uniform mean over active predicted classes')
    write(RUN / 'summary.json', summary)
    plot(all_metrics)

    e = all_metrics['unweighted_classwise']; d = all_metrics['strict_classwise']
    delta = e['mIoU'] - d['mIoU']; lo, hi = intervals['strict_classwise']['mIoU_delta_95CI']
    if lo > 0: verdict = '场景配对区间位于零以上，E 的 mIoU 高于 D。'
    elif hi < 0: verdict = '场景配对区间位于零以下，E 的 mIoU 低于 D。'
    else: verdict = '场景配对区间跨零，当前验证没有清晰的 mIoU 差异。'
    lines = [
        '# 逐类 Set loss 去除类别加权：实验结果', '',
        '**实验已完成。** E 使用相同 4096 帧、698 场景训练 1 epoch，并在完整 6019 帧、150 场景 camera 区域验证。', '',
        f"**E 相对上一组 D：mIoU {delta:+.3f} 个百分点，二值 IoU {e['IoU']-d['IoU']:+.3f}，1σ覆盖 {e['geo1']-d['geo1']:+.3f}。** {verdict}", '',
        '设计、公式与对照设置见[实验设计](unweighted_strict_classwise_set_mini_design.md)。E 从与 B、C、D 相同的预训练权重独立微调。', '',
        '## 1. 整体预测性能', '',
        '| 模型 | mIoU | 二值 IoU | 漏检/GT前景 | FP/GT前景 | 检出前景语义准确率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        cm = arrays[name]['confusion']; accuracy = 100*np.trace(cm[:17, :17])/max(cm[:17, :17].sum(), 1)
        value = all_metrics[name]
        lines.append(f"| {LABELS[name]} | {value['mIoU']:.3f}% | {value['IoU']:.3f}% | {value['FN_rate']:.3f}% | {value['FP_over_GTfg']:.3f}% | {accuracy:.3f}% |")
    lines += ['', '差值区间按 150 个场景配对重采样 1500 次，种子 928；单位为百分点，不含训练种子方差。', '',
        '| 比较 | mIoU差值95%区间 | 1σ覆盖差值95%区间 |', '| --- | --- | --- |']
    for name in NAMES[:3]:
        a, b = intervals[name]['mIoU_delta_95CI']; c, dlt = intervals[name]['geo1_delta_95CI']
        lines.append(f'| E − {name} | [{a:+.3f}, {b:+.3f}] | [{c:+.3f}, {dlt:+.3f}] |')
    lines += ['', '![四组完整验证结果](figures/unweighted_strict_classwise_set_mini_comparison.png)', '',
        '## 2. 几何覆盖与 Query 负载', '',
        '| 模型 | 1σ | 2σ | 3σ | 同类1σ | Top20最近归属 | 零核心接触query |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        m = all_metrics[name]
        lines.append(f"| {LABELS[name]} | {m['geo1']:.3f}% | {m['geo2']:.3f}% | {m['geo3']:.3f}% | {m['same1']:.3f}% | {m['top20_nearest_share']:.3f}% | {m['zero_core']:.3f}% |")
    lines += ['', '最近归属按每帧最近最终 GS 中心映射到父 query，不要求该 GS 实际覆盖 GT；因此负载集中度不能直接解释成容量上限。核心覆盖的存在也不代表渲染分类一定正确。', '',
        '| Query负载组 | B | C | D | E |', '| --- | ---: | ---: | ---: | ---: |']
    for q in range(5):
        values = [groups[i*5+q]['GT_share'] for i in range(4)]
        lines.append(f"| {q*20}–{(q+1)*20}% | " + ' | '.join(f'{x:.3f}%' for x in values) + ' |')
    lines += ['', '每帧独立排序，每组含 240/1200 个 query；各组按全验证前景 GT 数量汇总。', '',
        '## 3. 逐类别指标', '',
        '| 类别 | D IoU | E IoU | E−D IoU | E−D 1σ覆盖 | E−D 同类1σ |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    by = {(row['arm'], row['class_id']): row for row in class_rows}
    for i, label in enumerate(CLASS_NAMES):
        old, new = by[('strict_classwise', i)], by[('unweighted_classwise', i)]
        lines.append(f"| {label} | {old['IoU']:.3f}% | {new['IoU']:.3f}% | {new['IoU']-old['IoU']:+.3f} | {new['core1']-old['core1']:+.3f} | {new['same_core1']-old['same_core1']:+.3f} |")
    lines += ['', '各类的绝对漏检和 FP 计数见[逐类 CSV](../work_dirs/unweighted_strict_classwise_set_mini_v1/aggregate/per_class.csv)。', '',
        '## 4. 训练核验', '',
        f"训练耗时 {training['seconds']/60:.2f} 分钟，峰值已分配 CUDA 显存 {training['peak_cuda_GB']:.3f} GB；裁剪前梯度范数中位数 {audit['gradient_norm_median']:.3f}，95%分位 {audit['gradient_norm_p95']:.3f}，触发范数35裁剪的更新占 {audit['gradient_clip_percent']:.3f}%。", '',
        '预先配置比较通过：除 17 类 Set-loss 权重向量外，模型、优化器、训练数据、验证数据、Occupancy loss 和其他 Set-loss 设置与 D 完全一致。首帧 Occupancy loss 数值与 D 一致；4096 帧训练顺序相同。全验证覆盖 150 个场景，无验证集选模。', '',
        '## 5. 结论范围', '',
        'E−D 是本组最直接的比较：两组都按当前预测类别进行同类匹配，差异是类间汇总从原 17 类权重改为有效类别等权。类别内空间距离权重、几何阈值与稀有类 GT 几何权重保留；Occupancy loss 的类别权重也保留。结果只代表单次 1 epoch 微调，不估计训练随机种子方差。', '',
        '- [汇总指标与场景区间](../work_dirs/unweighted_strict_classwise_set_mini_v1/summary.json)；[优化审计](../work_dirs/unweighted_strict_classwise_set_mini_v1/optimization_audit.json)。', '']
    REPORT.write_text('\n'.join(lines))


if __name__ == '__main__':
    report()
