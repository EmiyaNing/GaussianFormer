"""Explain completed D/E screening using formal confusion and saved probes."""
import json
from pathlib import Path

import numpy as np

from .ab_first_epoch.analyze_a import class_conditionals
from .metrics import CLASS_NAMES
from .semantic_occupancy_first_epoch.protocol import ANALYSIS, C_ANALYSIS, C_WORK, D, E, ROOT


def read(path):
    return json.loads(path.read_text())


def main():
    report = read(ANALYSIS / 'comparison.json')
    if set(report['judgements']) != {D, E}:
        raise ValueError('both routes must have completed first-epoch diagnoses')
    names = {'C': C_WORK, 'D': ROOT / D, 'E': ROOT / E}
    matrices, classes, binary, conditional = {}, {}, {}, {}
    for name, work in names.items():
        with np.load(work / 'epoch_01_confusion.npz', allow_pickle=False) as saved:
            matrices[name] = saved['confusion'].copy()
        hist = matrices[name][0]
        classes[name] = class_conditionals(hist)
        binary[name] = dict(TP=int(hist[:17,:17].sum()), FP=int(hist[17,:17].sum()), FN=int(hist[:17,17].sum()))
        directory = C_ANALYSIS if name == 'C' else ANALYSIS / (D if name == 'D' else E)
        with np.load(directory / 'epoch_diagnostics/epoch_01_conditional_semantic_confusion.npz', allow_pickle=False) as saved:
            cond = saved['full']
        if not np.array_equal(cond.sum(1), hist.sum(1)[:17]):
            raise ValueError('conditional and formal target counts differ')
        recall = np.divide(cond.diagonal(), cond.sum(1), out=np.zeros(17), where=cond.sum(1)>0)*100
        conditional[name] = dict(micro_accuracy_percent=100*float(np.trace(cond))/int(cond.sum()),
                                macro_accuracy_percent=float(recall.mean()), class_recall_percent=recall.tolist())
    attributes, auxiliary = {}, {}
    for label, arm in [('D', D), ('E', E)]:
        diag = report['arms'][arm]['diagnostic']
        rows = [row for row in diag['attributes'] if row['horizon'] == 0]
        attributes[label] = {key: float(np.mean([row[key] for row in rows])) for key in
            ('empty_bias', 'empty_scalar', 'in_range_percent', 'scale_upper_saturation_percent', 'opacity_upper_saturation_percent')}
        records = read(ANALYSIS / arm / 'epoch_diagnostics/epoch_01_conditional_semantics.json')['records']
        auxiliary[label] = {key: float(np.mean([row[key] for row in records])) for key in
            ('conditional_semantic_loss', 'auxiliary_to_base_ce_logit_gradient_norm_ratio', 'auxiliary_base_ce_logit_gradient_cosine')}
    lines = ['# D/E 结果机制分析', '',
             '本文使用完整1024评估样本的正式混淆矩阵。属性与辅助项探针来自固定16样本，不混作正式指标。C/D/E均为第一轮。', '',
             '## 预设判断与主要变化', '']
    for label, arm in [('D', D), ('E', E)]:
        j = report['judgements'][arm]
        interval = j['Bonferroni_two_routes_CI97_5_pp']
        status = '达到预设第一轮筛选条件' if j['meets_prespecified_first_epoch_screen'] else '未达到预设第一轮筛选条件'
        lines += [f"{label}：{status}。当前mIoU变化{j['delta_miou_pp']:+.4f} pp（{j['relative_miou_percent']:+.2f}%），两路线调整后的场景配对区间为[{interval[0]:+.4f},{interval[1]:+.4f}] pp。", '',
            f"相对C：TP {binary[label]['TP']-binary['C']['TP']:+,}、FP {binary[label]['FP']-binary['C']['FP']:+,}、FN {binary[label]['FN']-binary['C']['FN']:+,}；占用precision {j['occupied_precision_delta_pp']:+.4f} pp、recall {j['occupied_recall_delta_pp']:+.4f} pp。", '']
    lines += ['## 分类别区分占用检出与语义识别', '',
              '“命中后类别正确率”只统计GT非空且正式预测非空的体素；条件17类召回覆盖所有该类GT体素，包括正式预测为空的体素。它们与IoU使用不同分母。', '',
              '| 类别 | C占用召回 % | D占用召回 % | E占用召回 % | C命中后正确率 % | D命中后正确率 % | E命中后正确率 % |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for c, name in enumerate(CLASS_NAMES[:17]):
        values = [classes[label][c][key] for key in ('occupied_recall_percent','conditional_semantic_accuracy_percent') for label in ('C','D','E')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '| 条件17类指标 | C % | D % | E % |', '|---|---:|---:|---:|']
    for key in ('micro_accuracy_percent','macro_accuracy_percent'):
        lines.append('| '+key+' | '+' | '.join(f'{conditional[label][key]:.4f}' for label in ('C','D','E'))+' |')
    lines += ['', '## 机制与解释边界', '',
        'D改变辅助CE的类别权重和归一化分母，重点检查car/truck及少数类收益是否超过道路等大类别损失。它不改变原18类CE；权重增加并不保证某类实际梯度或最终IoU同比增加。', '',
        'E在固定几何和共享特征时切断了语义logit对占用概率的直接影响。对GT空体素，联合CE仅监督占用gate，不再直接压低17类条件语义logit；对GT非空体素，联合CE分解成占用CE和条件语义CE。Lovasz及共享特征/几何仍可产生间接耦合。', '',
        'E同时改变了占用证据分支、概率构造与正式决策规则。因此必须结合其同checkpoint联合18类argmax反事实，区分结构训练收益和决策影响；当前两个实验不能单独量化每个子因素的收益。', '',
        '若当前mIoU上升但FP显著增加，应优先验证占用校准和完整预算曲线；若macro条件指标提高而micro下降，应结合逐类IoU判断是否由类别平衡带来预期取舍。任何方案的第一轮胜负不等于第15轮最优性能。', '',
        '## 固定16样本属性与辅助项', '', '| 方案 | scale饱和 % | opacity饱和 % | 合法中心 % | empty bias | 辅助/原CE梯度范数比 |',
        '|---|---:|---:|---:|---:|---:|']
    for label in ('D','E'):
        a = attributes[label]
        lines.append(f"| {label} | {a['scale_upper_saturation_percent']:.3f} | {a['opacity_upper_saturation_percent']:.3f} | {a['in_range_percent']:.3f} | {a['empty_bias']:.5f} | {auxiliary[label]['auxiliary_to_base_ce_logit_gradient_norm_ratio']:.5f} |")
    lines += ['', '没有使用第一轮未来近零/未监督指标选择赢家。本次只包含一个训练seed，场景bootstrap体现评估数据变化，不能证明训练稳定性。', '',
              '- [完整正式对比与预设区间](report.md)',
              '- [逐时刻逐类别IoU](per_horizon_per_class_iou.csv)',
              '- [机制统计JSON](results_analysis.json)',
              '- [类别与性能图](first_epoch_comparison.png)']
    (ANALYSIS / 'results_analysis.md').write_text('\n'.join(lines)+'\n')
    (ANALYSIS / 'results_analysis.json').write_text(json.dumps(dict(binary_counts=binary,
        class_conditionals=classes, conditional_all_gt=conditional, attributes=attributes, auxiliary=auxiliary,
        first_epoch_only=True, training_seeds=1),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios':[2,1]})
    x = np.arange(17)
    for offset, label, key, color in [(-.26,'C','C_conditional_semantic_ce','#596b87'),(0.,'D',D,'#208578'),(.26,'E',E,'#d67a2e')]:
        axes[0].bar(x+offset,report['metrics'][key]['horizons'][0]['class_iou'][:17],width=.25,label=label,color=color)
    axes[0].set_xticks(x, CLASS_NAMES[:17], rotation=35, ha='right', fontsize=9)
    axes[0].set_ylabel('Current per-class IoU (%)')
    axes[0].legend()
    axes[0].grid(axis='y',alpha=.2)
    for i,(label,arm,color) in enumerate([('D',D,'#208578'),('E',E,'#d67a2e')]):
        j=report['judgements'][arm]
        delta=j['delta_miou_pp']
        low,high=j['Bonferroni_two_routes_CI97_5_pp']
        axes[1].scatter(i,delta,color=color,s=60)
        axes[1].vlines(i,low,high,color=color,lw=3)
        axes[1].text(i+.08,delta,f'{delta:+.4f} pp',va='center')
    axes[1].axhline(0,color='#596b87',lw=1)
    axes[1].set_xticks([0,1],['D vs C','E vs C'])
    axes[1].set_xlim(-.5,1.7)
    axes[1].set_ylabel('Current mIoU delta (pp)')
    axes[1].set_title('Paired scene bootstrap; intervals adjusted for two routes; one training seed',fontsize=10)
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('C / D / E: epoch 1 of the same 15-epoch training schedule')
    fig.tight_layout()
    fig.savefig(ANALYSIS/'first_epoch_comparison.png',dpi=180)
    fig.savefig(ANALYSIS/'first_epoch_comparison.pdf')
    plt.close(fig)
    print(f'D/E mechanism analysis: {ANALYSIS / "results_analysis.md"}',flush=True)


if __name__ == '__main__':
    main()
