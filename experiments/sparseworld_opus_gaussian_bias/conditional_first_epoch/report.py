"""Judge the completed first epoch against B and all matched baselines."""
import csv
from pathlib import Path

import numpy as np

from ..ab_first_epoch.analyze_a import class_conditionals
from ..ab_first_epoch.report import read, verify_arm
from ..metrics import CLASS_NAMES
from ..train import atomic_json, digest
from .protocol import ANALYSIS, BASE_ANALYSIS, BASE_WORK, CONFIG, NAME, WORK


def create_report(work=WORK, output=ANALYSIS, baseline_work=BASE_WORK,
                  baseline_output=BASE_ANALYSIS / 'B_ce_free_0_1', baseline_roots=None):
    c = verify_arm(work, output)
    b = verify_arm(baseline_work, baseline_output)
    if (c['boundary']['manifest_sha256'] != b['boundary']['manifest_sha256'] or
            c['initialization']['initial_model_sha256'] != b['initialization']['initial_model_sha256']):
        raise ValueError('C and B used different subsets or complete initial model weights')
    audit = read(output / 'configuration_audit.json')
    if c['boundary']['config_sha256'] != audit['config_sha256'] or digest(CONFIG) != audit['config_sha256']:
        raise ValueError('C config differs from its validated standalone snapshot')
    conditional = read(output / 'epoch_diagnostics/epoch_01_conditional_semantics.json')
    if (conditional['is_smoke'] or conditional['formal_sample_count'] != 1024 or
            conditional['num_samples'] != 16 or len(conditional['records']) != 16):
        raise ValueError('C conditional semantics diagnosis is incomplete')
    old = Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e')
    roots = (dict(Strict=old / 'sparseworld_strict', Camera_Gaussian=old / 'sparseworld_opus_gaussian_bias',
                  Fullspace_CE=Path('out/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e/sparseworld_opus_gaussian_bias'),
                  A_softmax_focal=baseline_work.parent / 'A_softmax_focal', B_ce_free_0_1=baseline_work)
             if baseline_roots is None else dict(baseline_roots))
    roots[NAME] = work
    metrics, matrices = {}, {}
    for name, path in roots.items():
        metrics[name] = read(path / 'epoch_01_metrics.json')
        if metrics[name]['num_samples'] != 1024 or metrics[name]['epoch'] != 1:
            raise ValueError('all references must be complete epoch-one evaluations')
        with np.load(path / 'epoch_01_confusion.npz') as data:
            matrix = data['confusion'].copy()
        if matrix.shape != (7, 18, 18):
            raise ValueError('all evaluations must contain seven horizons and eighteen classes')
        for h, hist in enumerate(matrix):
            union = hist.sum(0) + hist.sum(1) - hist.diagonal()
            iou = np.divide(hist.diagonal(), union, out=np.zeros(18), where=union > 0) * 100
            if not np.allclose(iou, metrics[name]['horizons'][h]['class_iou'], rtol=1e-12, atol=1e-12):
                raise ValueError('formal class IoUs do not match confusion counts')
        matrices[name] = matrix
    if any(not np.array_equal(m.sum(2), matrices[NAME].sum(2)) for m in matrices.values()):
        raise ValueError('formal target voxel counts differ between experiments')
    with np.load(output / 'epoch_diagnostics/epoch_01_conditional_semantic_confusion.npz') as data:
        conditional_matrices = {key: data[key].copy() for key in data.files}
    if not np.array_equal(conditional_matrices['full'], conditional_matrices['camera'] + conditional_matrices['noncamera']):
        raise ValueError('conditional region partition is inconsistent')
    with np.load(output / 'epoch_diagnostics/epoch_01_visibility_confusion.npz') as data:
        for name, hist in conditional_matrices.items():
            if not np.array_equal(hist.sum(1), data[name][0].sum(1)[:17]):
                raise ValueError('conditional prediction excluded GT-occupied voxels')
    conditions = {name: class_conditionals(hist[0]) for name, hist in matrices.items()}
    current_c, current_b = metrics[NAME]['horizons'][0], metrics['B_ce_free_0_1']['horizons'][0]
    delta_miou = current_c['miou'] - current_b['miou']
    verdict = 'higher' if delta_miou > 0 else ('lower' if delta_miou < 0 else 'equal')
    judgement = dict(scope='single_seed_complete_first_epoch_only', compared_epoch=1,
                     planned_epochs=15, actual_epochs=1,
                     current_miou_relative_to_B=verdict, current_miou_C_minus_B_pp=delta_miou,
                     car_iou_C_minus_B_pp=current_c['class_iou'][4] - current_b['class_iou'][4],
                     sidewalk_iou_C_minus_B_pp=current_c['class_iou'][13] - current_b['class_iou'][13],
                     occupied_recall_C_minus_B_pp=current_c['occupied_recall'] - current_b['occupied_recall'],
                     occupied_precision_C_minus_B_pp=current_c['occupied_precision'] - current_b['occupied_precision'],
                     noncamera_FP_C_minus_B=int(c['diagnostic']['visibility']['h0/noncamera']['occupied_fp']) - int(b['diagnostic']['visibility']['h0/noncamera']['occupied_fp']))
    atomic_json(output / 'comparison.json', dict(metrics=metrics, C=c, B=b, conditional=conditional,
                current_class_conditionals=conditions, judgement=judgement, configuration_audit=audit))
    rows = []
    for h in range(7):
        for index, class_name in enumerate(CLASS_NAMES):
            row = dict(seconds=h * .5, class_index=index, class_name=class_name,
                       gt_voxels=int(matrices[NAME][h, index].sum()))
            row.update({f'{name}_iou': metric['horizons'][h]['class_iou'][index] for name, metric in metrics.items()})
            row.update(C_minus_B_pp=row[f'{NAME}_iou'] - row['B_ce_free_0_1_iou'],
                       C_minus_Strict_pp=row[f'{NAME}_iou'] - row['Strict_iou'])
            rows.append(row)
    with (output / 'per_horizon_per_class_iou.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 条件语义 CE：第一轮判断', '',
             'C 与 B 使用相同完整 scratch 模型初始化、4096 训练样本、1024 验证样本和 15 轮学习率计划。'
             '只增加系数 1.0 的 GT 非空 17 类条件语义 CE；原 18 类全空间 CE=10、free=0.1、Lovasz=1 和时间课程保留。', '',
             '第一轮完整训练/诊断后暂停，保留完整优化器、RNG 与查询分配状态。以下是第一轮判断，不是 15 轮最优成绩。', '',
             '| 模型 | 当前 mIoU % | 占用 IoU % | precision % | recall % |',
             '|---|---:|---:|---:|---:|']
    for name, metric in metrics.items():
        value = metric['horizons'][0]
        lines.append(f"| {name} | {value['miou']:.4f} | {value['occupied_iou']:.4f} | {value['occupied_precision']:.4f} | {value['occupied_recall']:.4f} |")
    lines += ['', '## 相对 B 的主要变化', '',
              f"当前 mIoU {delta_miou:+.4f} pp；car IoU {judgement['car_iou_C_minus_B_pp']:+.4f} pp；"
              f"sidewalk IoU {judgement['sidewalk_iou_C_minus_B_pp']:+.4f} pp。"
              f"占用召回 {judgement['occupied_recall_C_minus_B_pp']:+.4f} pp，precision {judgement['occupied_precision_C_minus_B_pp']:+.4f} pp。"
              f"noncamera 区域 FP 差值为 {judgement['noncamera_FP_C_minus_B']:+,}。", '',
              '语义指标、占用检出和区域误报需一起判断；当前统计不证明多 seed 稳定收益。'
              '近零未来指标属于原时间扩展预训练阶段，不作为第一轮方案的主要胜负依据。', '',
              '| 类别 | B IoU % | C IoU % | B 占用召回 % | C 占用召回 % | B 命中后类别正确率 % | C 命中后类别正确率 % |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for index in (4, 10, 11, 13, 14, 15, 16):
        bc, cc = conditions['B_ce_free_0_1'][index], conditions[NAME][index]
        lines.append(f"| {CLASS_NAMES[index]} | {current_b['class_iou'][index]:.4f} | {current_c['class_iou'][index]:.4f} | {bc['occupied_recall_percent']:.3f} | {cc['occupied_recall_percent']:.3f} | {bc['conditional_semantic_accuracy_percent']:.3f} | {cc['conditional_semantic_accuracy_percent']:.3f} |")
    lines += ['', '上表“命中后类别正确率”只统计 18 类预测为非空的 GT 非空体素。'
              '新增的 17 类条件预测矩阵则覆盖所有 GT 非空体素，包含 18 类预测为空的体素；两个分母不能混用。', '',
              f"C 全部 GT 非空体素的当前条件类别准确率为 {conditional['formal_current_conditional_class_accuracy_percent']:.4f}%。"
              'B 未保存同口径的全验证集条件矩阵，因此不据此构造 B/C 数字对比。', '',
              '## 当前区域误报与漏检', '',
              '| 模型 | 区域 | mIoU % | precision % | recall % | FP | FN |',
              '|---|---|---:|---:|---:|---:|---:|']
    for name, value in (('B', b), ('C', c)):
        for region in ('camera', 'noncamera'):
            r = value['diagnostic']['visibility']['h0/' + region]
            lines.append(f"| {name} | {region} | {r['miou']:.4f} | {r['occupied_precision']:.4f} | {r['occupied_recall']:.4f} | {r['occupied_fp']:,} | {r['occupied_fn']:,} |")
    records = conditional['records']
    lines += ['', '## 固定 16 样本的辅助项诊断', '',
              f"条件语义 loss 均值 {np.mean([r['conditional_semantic_loss'] for r in records]):.6f}；"
              f"辅助/原 18 类 CE 的渲染 logit 梯度范数比均值 {np.mean([r['auxiliary_to_base_ce_logit_gradient_norm_ratio'] for r in records]):.6f}；"
              f"梯度余弦均值 {np.mean([r['auxiliary_base_ce_logit_gradient_cosine'] for r in records]):.6f}。"
              '这是输出 logit 分类梯度，排除 Lovasz、集合/轨迹与共享网络参数梯度。辅助项对 free logit 的直接梯度为零。', '',
              '| 区域 | 条件 CE loss 贡献均值 | 正确条件类别概率均值 |', '|---|---:|---:|']
    for region in ('camera', 'noncamera'):
        lines.append(f"| {region} | {np.mean([r['buckets'][region]['loss_contribution'] for r in records]):.6f} | {np.mean([r['buckets'][region]['mean_correct_conditional_probability'] for r in records]):.6f} |")
    lines += ['', '18 类 CE 分桶另见 classification_components.json，那里只包含原 18 类分类项，未将辅助项混入。'
              '完整属性、渲染和包含辅助项的末层语义/中心梯度探针见 epoch_diagnostics/epoch_01_diagnostics.md。', '',
              '## 各时刻各类别 IoU（%）', '']
    for h in range(7):
        lines += [f'### {h*.5:g} 秒', '',
                  '| 类别 | Strict | camera | 全空间 CE | A | B | C | C−B pp |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows[h * 18:(h + 1) * 18]:
            lines.append(f"| {row['class_name']} | {row['Strict_iou']:.6f} | {row['Camera_Gaussian_iou']:.6f} | {row['Fullspace_CE_iou']:.6f} | {row['A_softmax_focal_iou']:.6f} | {row['B_ce_free_0_1_iou']:.6f} | {row[f'{NAME}_iou']:.6f} | {row['C_minus_B_pp']:+.6f} |")
        lines.append('')
    lines += ['mIoU 固定平均 17 个非空类，free 不进入均值。', '',
              '- [126 行逐时刻逐类别 CSV](per_horizon_per_class_iou.csv)',
              '- [指标、边界、完整诊断与判断 JSON](comparison.json)',
              '- [17 类条件语义诊断](epoch_diagnostics/epoch_01_conditional_semantics.json)',
              '- [属性/渲染/末层梯度诊断](epoch_diagnostics/epoch_01_diagnostics.md)',
              '- [独立配置审计](configuration_audit.json)', '']
    (output / 'report.md').write_text('\n'.join(lines))
    atomic_json(output / 'diagnosis_completed.json', dict(status='completed', **judgement))
    print(f'C first-epoch judgement: mIoU C-B={delta_miou:+.6f} pp; report={output / "report.md"}', flush=True)


if __name__ == '__main__':
    create_report()
