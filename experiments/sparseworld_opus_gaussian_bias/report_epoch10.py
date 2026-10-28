"""Build the epoch-10 diagnosis from full evaluation and fixed-sample probes."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .metrics import CLASS_NAMES


def main():
    directory = Path('analysis/sparseworld_opus_gaussian_bias_4096_1024_15e')
    work = Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e')
    names = ('sparseworld_strict', 'sparseworld_opus_gaussian_bias')
    formal = {name: json.loads((work / name / 'epoch_10_metrics.json').read_text()) for name in names}
    diagnostic = json.loads((directory / 'epoch10_diagnostics.json').read_text())
    audit = json.loads((directory / 'epoch10_visibility_audit.json').read_text())
    if set(audit['models']) != set(names) or audit['num_samples'] != 1024:
        raise ValueError('complete two-model 1024-sample visibility audit required')
    if any(not audit['models'][name]['exact_formal_confusion_match'] for name in names):
        raise ValueError('visibility audit must reproduce the formal confusion matrices exactly')
    for horizon in range(7):
        for region in ('full', 'camera', 'noncamera'):
            left, right = [audit['models'][name]['metrics'][f'h{horizon}/{region}'] for name in names]
            if (left['voxel_count'], left['gt_occupied']) != (right['voxel_count'], right['gt_occupied']):
                raise ValueError('visibility partitions or ground truth differ between models')
    strict, gaussian = [formal[name] for name in names]
    for left, right in zip(strict['horizons'], gaussian['horizons']):
        if left['class_gt_count'] != right['class_gt_count']:
            raise ValueError('two models evaluated different ground truth')
    with (directory / 'epoch10_per_horizon_per_class_iou.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seconds', 'class_index', 'class_name', 'strict_iou_percent',
                         'gaussian_iou_percent', 'delta_percentage_points', 'gt_voxels'])
        for left, right in zip(strict['horizons'], gaussian['horizons']):
            for index, name in enumerate(CLASS_NAMES):
                writer.writerow([left['seconds'], index, name, left['class_iou'][index],
                                 right['class_iou'][index],
                                 right['class_iou'][index] - left['class_iou'][index],
                                 left['class_gt_count'][index]])
    with (directory / 'epoch10_visibility_metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'seconds', 'region', 'miou_percent', 'occupied_iou_percent',
                         'occupied_precision_percent', 'occupied_recall_percent',
                         'predicted_occupied_percent', 'gt_occupied_percent',
                         'occupied_tp', 'occupied_fp', 'occupied_fn', 'voxel_count'])
        for name in names:
            for horizon in range(7):
                for region in ('full', 'camera', 'noncamera'):
                    value = audit['models'][name]['metrics'][f'h{horizon}/{region}']
                    writer.writerow([name, horizon * .5, region, value['miou'], value['occupied_iou'],
                                     value['occupied_precision'], value['occupied_recall'],
                                     value['predicted_occupied_percent'],
                                     100 * value['gt_occupied'] / value['voxel_count'],
                                     value['occupied_tp'], value['occupied_fp'],
                                     value['occupied_fn'], value['voxel_count']])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, horizon in zip(axes.flat, (0, 2, 4, 6)):
        for name, label, color in zip(names, ('Strict', 'Gaussian'), ('#245ba5', '#cf6533')):
            values = [json.loads((work / name / f'epoch_{e:02d}_metrics.json').read_text())
                      ['horizons'][horizon]['miou'] for e in range(1, 11)]
            ax.plot(range(1, 11), values, marker='o', markersize=3, label=label, color=color)
        ax.set_title(f'{horizon * .5:g} seconds')
        ax.set_ylabel('mIoU (%)')
        ax.set_xlabel('Epoch')
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'epoch10_training_curves.png', dpi=180)
    fig.savefig(directory / 'epoch10_training_curves.pdf')
    plt.close(fig)
    delta = np.array([right['class_iou'][:17] for right in gaussian['horizons']]) - np.array(
        [left['class_iou'][:17] for left in strict['horizons']])
    fig, ax = plt.subplots(figsize=(16, 5))
    bound = max(float(abs(delta).max()), 1)
    heatmap = ax.imshow(delta, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_xticks(range(17), CLASS_NAMES[:17], rotation=45, ha='right')
    ax.set_yticks(range(7), [f'{h * .5:g}s' for h in range(7)])
    for h in range(7):
        for c in range(17):
            ax.text(c, h, f'{delta[h, c]:+.1f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(delta[h, c]) > .65 * bound else 'black')
    fig.colorbar(heatmap, ax=ax, label='Gaussian - Strict IoU (percentage points)')
    ax.set_title('Epoch 10: full 1024-sample evaluation')
    fig.tight_layout()
    fig.savefig(directory / 'epoch10_class_iou_delta.png', dpi=180)
    fig.savefig(directory / 'epoch10_class_iou_delta.pdf')
    plt.close(fig)
    lines = ['# SparseWorld Gaussian 第10轮性能差距分析', '',
             '本报告比较相同第10轮checkpoint。正式结果覆盖同一1024个held-out样本；可见性复评完整覆盖1024样本，并逐元素复现两组正式混淆矩阵。渲染变体仅使用16个固定样本；梯度探针仅使用2个固定样本，不作为新模型成绩。', '',
             '## 同轮结果及问题变化', '',
             '| 时刻 | Strict mIoU (%) | Gaussian mIoU (%) | Strict占据IoU (%) | Gaussian占据IoU (%) |',
             '|---|---:|---:|---:|---:|']
    for h in (0, 2, 4, 6):
        s, g = strict['horizons'][h], gaussian['horizons'][h]
        lines.append(f"| {h * .5:g}s | {s['miou']:.4f} | {g['miou']:.4f} | {s['occupied_iou']:.4f} | {g['occupied_iou']:.4f} |")
    lines += ['', f"四时刻平均mIoU：{strict['selection_mean_miou']:.4f}%与{gaussian['selection_mean_miou']:.4f}%，相对下降17.05%。当前、1s、2s的mIoU相对下降约9–11%，3s下降48.80%。3s贡献约49%的四时刻平均mIoU差距。", '',
              '第1轮的保守预测与道路类别集中解释需要更新：第10轮当前预测占据比例为Strict 7.25%、Gaussian 14.00%；Gaussian当前精度18.92%低于31.13%，召回50.05%高于42.66%。当前Gaussian道路预测占比27.76%，与Strict 28.43%接近；当前主要问题已转为大量几何误报。', '',
              '当前汽车IoU为21.24%与Strict 22.21%接近，Gaussian自行车/摩托车/行人/交通锥部分优于Strict；道路、平地、人行道、地形、建筑和植被六类贡献约96.59%的当前17类IoU总下降。3s则多数物体类别显著下降，汽车从Strict 8.82%降至1.62%，道路仍为22.92%。', '',
              '## 主要证据1：体素监督的可见性范围与最终评估不一致', '',
              '配置 `loss.mask_type=camera` 使新增Gaussian体素CE/Lovasz只覆盖相机可见体素；正式评估 `occ3d_eval_mask=none` 覆盖全空间。完整复评结果如下。分区mIoU仍固定平均17类，不能用分区mIoU加权恢复全空间mIoU。', '',
              '| 时刻 | 区域 | Strict mIoU (%) | Gaussian mIoU (%) | Strict占据IoU (%) | Gaussian占据IoU (%) |',
              '|---|---|---:|---:|---:|---:|']
    for h in (0, 2, 4, 6):
        for region, label in (('camera', '相机可见'), ('noncamera', '相机不可见')):
            s, g = [audit['models'][name]['metrics'][f'h{h}/{region}'] for name in names]
            lines.append(f"| {h * .5:g}s | {label} | {s['miou']:.4f} | {g['miou']:.4f} | {s['occupied_iou']:.4f} | {g['occupied_iou']:.4f} |")
    lines += ['']
    for name, label in zip(names, ('Strict', 'Gaussian')):
        full = audit['models'][name]['metrics']['h0/full']
        hidden = audit['models'][name]['metrics']['h0/noncamera']
        visible = audit['models'][name]['metrics']['h0/camera']
        lines.append(f"{label}当前几何FP={full['occupied_fp']:,}，其中不可见区域FP={hidden['occupied_fp']:,}（{100 * hidden['occupied_fp'] / full['occupied_fp']:.2f}%）。可见区域GT占据比例={100 * visible['gt_occupied']/visible['voxel_count']:.2f}%；不可见区域GT占据比例={100 * hidden['gt_occupied']/hidden['voxel_count']:.2f}%，预测占据比例={hidden['predicted_occupied_percent']:.2f}%。")
    lines += ['',
              '这是监督/评估范围不匹配的重要证据：Gaussian在直接体素监督覆盖的当前/1s/2s区域更强，在未覆盖区域误报更多，3s可见区域也仍有泛化缺口。集合损失仍覆盖全空间，但不能直接约束Gaussian在那些空体素上的累加输出。具体因果贡献仍需保持评估口径不变的训练消融。', '',
              '## 主要证据2：属性持续饱和，求和渲染对点密度敏感', '',
              '16样本中当前有效Gaussian的86.58%尺度分量>0.395m，81.35%不透明度>0.99；中位尺度约0.4m、中位不透明度约0.99999。局部聚合将所有Gaussian正语义贡献相加，不按局部密度归一化；Strict点栅格化对同体素同类别分数取最大值，还执行分数阈值及点中心距离过滤。', '',
              '新模型每query最后128个children，Strict为48个。求和对重复/密集child敏感，max-score不会因相同点重复而同比放大；膨胀到0.4m的Gaussian与较高child密度可能增加相邻空体素的贡献。这里是机制推断，128与48的独立贡献尚未通过训练消融量化。', '',
              '16样本当前正常Gaussian mIoU/占据IoU=10.506/16.829%；将同一checkpoint的空类bias减2，变为9.248/18.205%，几何精度提高但语义mIoU下降。强行把所有尺度设为0.2m则只有0.360/1.452%；将opacity减半为5.107/12.643%。模型已依赖大尺度/高opacity，不能把推理时简单收缩当作修复。', '',
              '## 主要证据3：3秒尚无直接占据监督，统一渲染校准难以跨时域泛化', '',
              '训练预测时域=`min(curr_epoch - finetune_epoch + 1, 6)`，curr_epoch从0开始，finetune_epoch=5。第10轮只展开5个0.5s间隔，即2.5s；评估始终预测到3s。3s的直接集合/体素监督都从第11轮开始，但未来分支参数在短时域共享训练。Gaussian的1s/2s体素监督从第7/9轮分别开始。', '',
              '共享属性分支用于decoder最终query和递归预测features，未增加专门的未来归一化/校准。16样本feature范数中位统计从当前16.60升至3s 62.71；3s opacity>0.99的比例降至36.78%，中位opacity约0.854。3s最大softplus语义中位数也较当前下降。所有时刻共享同一个empty_bias=3.3295、empty_scalar=9.6250，前景幅值与密度却随时域改变。', '',
              '16样本的3s，Gaussian正常渲染mIoU=3.132%，同一预测points/logits改用Strict点栅格化升至5.439%；Strict自身为6.489%。Gaussian 3s汽车IoU在该探针中从0.827%升到5.719%，卡车从0.246%升到10.622%，说明渲染前仍保留了部分物体语义信号。', '',
              '3s将空类bias加2可使mIoU达到4.662%，而当前相同方向会增加误报；当前更严格的空类判定与3s更宽松的空类判定存在相反需求。统一bias难以同时纠正这些时域分布，但不能仅根据held-out探针挑选阈值并宣称新性能。', '',
              '## 主要证据4：新增体素损失的局部梯度影响仍较强', '',
              '2个固定样本比较实际直接监督的current/1s/2s集合与体素损失，对raw class logits和world centers求梯度；几何属性和features保持固定。未来体素权重0.5已计入。这里不测全模型参数梯度，也不包含其它decoder层的集合损失。', '',
              '| 时刻 | 体素/集合语义梯度范数比例范围 | 体素/集合中心梯度范数比例范围 |',
              '|---|---:|---:|']
    probes = diagnostic['models'][names[1]]['gradient_probes']
    for h in (0, 2, 4):
        selected = [value for value in probes if value['horizon'] == h]
        sem = [value['semantic']['voxel_to_set_norm_ratio'] for value in selected]
        geo = [value['geometry']['voxel_to_set_norm_ratio'] for value in selected]
        lines.append(f'| {h * .5:g}s | {min(sem):.2f}–{max(sem):.2f} | {min(geo):.2f}–{max(geo):.2f} |')
    lines += ['',
              '语义和中心梯度整体余弦为正（约0.16–0.33），不支持“全面梯度冲突”的说法。不过新增体素损失仍能明显改变这些输出的优化，并可能优先拟合可见区域。其因果影响要用损失权重/监督范围的独立消融确认。', '',
              '## 暂不支持的解释', '',
              '- 单纯汽车分类完全未学会：当前汽车与Strict接近，部分小类别更好。','- 单纯轨迹预测错误：3s轨迹L2为Strict约0.22924m、Gaussian约0.22958m，差距很小。','- 不同子集或不同mIoU定义：正式样本、GT计数和评估定义一致；可见性复评完整复现原混淆矩阵。','- 仅调一个empty_bias即可修复：当前和3s需要的校准方向不同，几何改善也不保证语义改善。', '',
              '## 建议验证顺序', '',
              '1. 完成原15轮，单独观察第11轮启用3s监督后的变化；不覆盖原checkpoint或改动原配置。','2. 新建独立配置，将Gaussian体素监督对齐既定全空间评估；可以先用小训练段验证可见/不可见区域FP与梯度的变化。若真正任务只要求可见区域，应另报两组在同一mask下的成绩，而非更改原实验成绩。','3. 单独降低/渐增体素损失权重，监测当前/未来语义和中心梯度、opacity/scale饱和比例。','4. 固定48个children，再验证128个children；评估密度归一化或专门的renderer semantic calibration，保持集合分支raw logits接口。','5. 在训练数据上验证未来feature归一化或按时域条件化的opacity/bias，同时保持相同预测训练时域；不能用held-out探针反复选择阈值。', '',
              '## 数据与可复现入口', '',
              '- [1024样本可见性审计](epoch10_visibility_audit.json)','- [可见性指标CSV](epoch10_visibility_metrics.csv)','- [16样本渲染与2样本梯度探针](epoch10_diagnostics.json)','- [7时刻×18类IoU CSV](epoch10_per_horizon_per_class_iou.csv)','- [前10轮训练曲线](epoch10_training_curves.png)','- [每时刻每类IoU差值](epoch10_class_iou_delta.png)','- [诊断代码](../../experiments/sparseworld_opus_gaussian_bias/diagnose_epoch10.py)','- [可见性审计代码](../../experiments/sparseworld_opus_gaussian_bias/audit_visibility_epoch10.py)','- [报告生成代码](../../experiments/sparseworld_opus_gaussian_bias/report_epoch10.py)','',
              '- [体素损失掩码实现](../../experiments/sparseworld_opus_gaussian_bias/loss.py)',
              '- [当前独立实验配置](../../config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_4096_1024_15e.py)',
              '- [预测时域课程实现](../../model/segmentor/sparseworld_strict.py)','',
              '本次分析全部使用新增文件。既有训练实现、实验配置和checkpoint未因诊断而修改。']
    (directory / 'epoch10_analysis.md').write_text('\n'.join(lines) + '\n')
    print('Saved', directory / 'epoch10_analysis.md')


if __name__ == '__main__':
    main()
