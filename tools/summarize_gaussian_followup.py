"""Write independent follow-up reports and update the original mini comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from summarize_mini_occ3d import CLASSES, ROOT, parse_log


RUN_ROOT = ROOT / 'work_dirs/mini_occ3d_4096_1024'
PLAN_ROOT = ROOT / 'plans/sparse_render_offical_9_28'
GROUPS = {
    'opusv1_gaussian_softplus_camera_mask': ('相机 mask', 'camera_mask'),
    'opusv1_gaussian_softplus_learned_bias': ('可学习空类偏置', 'learned_bias'),
    'opusv1_gaussian_softplus_class_margin': ('类别感知 margin', 'class_margin'),
}
START = '<!-- gaussian_followup_start -->'
END = '<!-- gaussian_followup_end -->'


def read_run(name):
    run_dir = RUN_ROOT / name
    logs = sorted(path for path in run_dir.glob('*.log')
                  if path.name not in ('console.log', 'diagnostic_console.log'))
    if not logs:
        return None
    rows = parse_log(logs)
    if [row['epoch'] for row in rows] != [1, 2, 3, 4, 5]:
        return None
    best_miou = max(rows, key=lambda row: (row['mIoU'], -row['epoch']))
    best_iou = max(rows, key=lambda row: (row['IoU'], -row['epoch']))
    diagnostic_path = run_dir / 'diagnostic_val1024.json'
    if not diagnostic_path.is_file():
        return None
    diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
    bias_by_epoch = None
    if name == 'opusv1_gaussian_softplus_learned_bias':
        bias_by_epoch = {}
        for row in rows:
            checkpoint = torch.load(run_dir / f'epoch_{row["epoch"]}.pth',
                                    map_location='cpu', mmap=True,
                                    weights_only=False)
            bias_by_epoch[row['epoch']] = float(
                checkpoint['state_dict']['head.empty_bias'])
            del checkpoint
    return {'epochs': rows, 'best_mIoU': best_miou, 'best_IoU': best_iou,
            'diagnostic': diagnostic,
            'bias_by_epoch': bias_by_epoch,
            'logs': [str(path.relative_to(ROOT)) for path in logs]}


def write_group_result(name, label, slug, run, softplus, camera=None):
    best = run['best_mIoU']
    best_iou = run['best_IoU']
    base = softplus['best_mIoU']
    diag = run['diagnostic']
    metrics = diag['empty_logit_shift_metrics']['0.0']
    passes_target = best['mIoU'] >= 20.45637 and best['IoU'] >= 38.89461
    lines = [
        f'# 实验 {label}：结果分析', '',
        f'设计见[{slug}_design.md]({slug}_design.md)。训练 4096 帧、验证 1024 帧；seed 928、5 epoch、micro-batch 1、梯度累积 4。空类推理偏移为 0。', '',
        '## 总体结果', '',
        '| 指标 | 本组 | softplus-only 基线 | 差值 |',
        '| --- | ---: | ---: | ---: |',
        f'| 最佳 mIoU (%) | {best["mIoU"]:.2f} (epoch {best["epoch"]}) | {base["mIoU"]:.2f} | {best["mIoU"]-base["mIoU"]:+.2f} |',
        f'| 最佳二值 IoU (%) | {best_iou["IoU"]:.2f} (epoch {best_iou["epoch"]}) | {softplus["best_IoU"]["IoU"]:.2f} | {best_iou["IoU"]-softplus["best_IoU"]["IoU"]:+.2f} |',
        f'| 最佳 mIoU checkpoint 的二值 IoU (%) | {best["IoU"]:.2f} | {base["IoU"]:.2f} | {best["IoU"]-base["IoU"]:+.2f} |', '',
        '## 每轮验证', '',
        '| epoch | mIoU (%) | 二值 IoU (%) |', '| ---: | ---: | ---: |',
    ]
    lines += [f'| {r["epoch"]} | {r["mIoU"]:.2f} | {r["IoU"]:.2f} |'
              for r in run['epochs']]
    if run['bias_by_epoch'] is not None:
        lines += ['', '### 学习到的偏置轨迹', '',
                  '| epoch | empty_bias |', '| ---: | ---: |']
        for epoch, value in run['bias_by_epoch'].items():
            lines.append(f'| {epoch} | {value:.3f} |')
    lines += ['', '## 决策与 Gaussian 属性', '',
              '以下均来自最佳 mIoU checkpoint 的完整 1024 帧相机可见验证集。', '',
              '| 预测占用率 | GT 占用率 | precision | recall | empty scalar | learned bias | opacity > 0.9 | scale > 0.39 m |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
              (f'| {metrics["predicted_occupied_fraction"]:.2f}% | '
               f'{metrics["gt_occupied_fraction"]:.2f}% | {metrics["precision"]:.2f}% | '
               f'{metrics["recall"]:.2f}% | {diag["empty_scalar"]:.3f} | '
               f'{diag["learned_empty_bias"]:.3f}'
               if diag.get('learned_empty_bias') is not None else
               f'| {metrics["predicted_occupied_fraction"]:.2f}% | '
               f'{metrics["gt_occupied_fraction"]:.2f}% | {metrics["precision"]:.2f}% | '
               f'{metrics["recall"]:.2f}% | {diag["empty_scalar"]:.3f} | —')
              + (f' | {diag["gaussian_statistics"]["opacity_over_09"]*100:.2f}%'
                 f' | {diag["gaussian_statistics"]["scale_over_039"]*100:.2f}% |')]
    shifts = diag['empty_logit_shift_metrics']
    lines += ['', '### 额外偏移敏感性（仅诊断）', '',
              '以下偏移是在已训练 checkpoint 上另外减去的数值，只用于观察边界；主实验统一使用 0。', '',
              '| 额外空类偏移 | mIoU (%) | 二值 IoU (%) | 预测占用率 |',
              '| ---: | ---: | ---: | ---: |']
    for shift in ('-2.0', '0.0', '2.0', '4.0'):
        shifted = shifts[shift]
        lines.append(f'| {float(shift):+.0f} | {shifted["mIoU"]:.2f} | '
                     f'{shifted["IoU"]:.2f} | '
                     f'{shifted["predicted_occupied_fraction"]:.2f}% |')
    if camera is not None and slug != 'camera_mask':
        camera_best = camera['best_mIoU']
        lines += ['', '相较仅相机 mask 的实验 A：同一最佳 mIoU checkpoint 的 mIoU '
                  f'{best["mIoU"]-camera_best["mIoU"]:+.2f} 点，二值 IoU '
                  f'{best["IoU"]-camera_best["IoU"]:+.2f} 点。']
    lines += ['', '## 逐类别 IoU：各组最佳 mIoU checkpoint', '',
              '| 类别 | softplus-only (%) | 本组 (%) | 差值 |',
              '| --- | ---: | ---: | ---: |']
    for cls in CLASSES:
        a = base['per_class'][cls]
        b = best['per_class'][cls]
        lines.append(f'| {cls} | {a:.2f} | {b:.2f} | {b-a:+.2f} |')
    lines += ['', '## 判读', '',
              ('本组在同一个最佳 mIoU checkpoint 上达到先前空类减 4 的 20.46% mIoU 与 38.89% 二值 IoU 参照。'
               if passes_target else
               '本组在同一个最佳 mIoU checkpoint 上尚未同时达到先前空类减 4 的 20.46% mIoU 与 38.89% 二值 IoU 参照。'),
              '本结果仅来自单种子 mini 数据集，不能视作完整 Occ3D 或多种子结论。']
    if slug == 'camera_mask':
        lines += ['',
                  '仅把占用损失限制在相机可见体素，预测占用率从 softplus-only 的 7.82% 上升到 17.59%，'
                  '更接近同一区域的 GT 22.54%；召回率从 27.66% 升至 56.04%。'
                  'mIoU 与同一 checkpoint 二值 IoU 同时提高，支持训练和验证监督范围不一致是主要瓶颈之一。'
                  '但 `others` 与 `motorcycle` 退化，说明相机 mask 本身不能保证所有类别获益。']
    elif slug == 'learned_bias':
        lines += ['',
                  '训练学得的 empty_bias 为 3.555，验证时没有再指定空类校准值。'
                  '相比实验 A，预测占用率从 17.59% 略降到 17.15%，precision 从 71.82% 升到 77.45%，'
                  'recall 从 56.04% 升到 58.93%；收益并非只靠扩大预测占用面积。'
                  '它在相同 mini 协议下取得最高总体 mIoU 与二值 IoU，并恢复了 bicycle 和 motorcycle 的一部分性能。'
                  '但 `others`、`barrier`、`bicycle`、`motorcycle`、`pedestrian`、`traffic_cone` 等类别仍不及 GaussianFormer。'
                  '诊断中 opacity > 0.9 和 scale > 0.39 m 的比例均为 100%，属性饱和仍存在。'
                  '该标量本质仍是训练学到的渲染后空类偏移，不能据此声称 Gaussian 表达的校准问题已从结构上消失。']
    elif slug == 'class_margin':
        lines += ['',
                  '与实验 A 相比，同一最佳 mIoU checkpoint 的二值 IoU 提高 1.85 点，mIoU 下降 0.86 点；'
                  '`others` 仅 0.03%、`bicycle` 为 0、`motorcycle` 仅 3.41%。'
                  '真实类别 margin 没有保护最担心的稀有类，因此按设计判据不与实验 B 合并。'
                  '其类别退化表明需要检查 Gaussian 本身的类别赋值与点集监督，而不是继续加大二值决策压力。']
    lines += ['',
              '训练日志：' + '、'.join(f'`{path}`' for path in run['logs']) + '。',
              f'完整诊断：`work_dirs/mini_occ3d_4096_1024/{name}/diagnostic_val1024.json`。', '']
    (PLAN_ROOT / f'{slug}_results.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    runs = {name: run for name in GROUPS if (run := read_run(name)) is not None}
    if len(runs) != len(GROUPS) and not args.allow_partial:
        raise ValueError(f'Only {len(runs)}/{len(GROUPS)} follow-up runs are complete')
    if not runs:
        raise ValueError('No complete follow-up run')
    prior = json.loads((RUN_ROOT / 'gaussian_training_ablation_summary.json').read_text())
    softplus = prior['opusv1_gaussian_softplus']
    for name, run in runs.items():
        label, slug = GROUPS[name]
        write_group_result(name, label, slug, run, softplus,
                           runs.get('opusv1_gaussian_softplus_camera_mask'))

    original = json.loads((PLAN_ROOT / 'mini_occ3d_4096_1024_summary.json').read_text())
    all_models = {}
    for name, data in original['models'].items():
        all_models[name] = {
            'mIoU': data['best_mIoU'], 'mIoU_epoch': data['best_mIoU_epoch'],
            'IoU': data['best_IoU'], 'IoU_epoch': data['best_IoU_epoch'],
            'per_class': data['per_class_at_best_mIoU']}
    for name in ('opusv1_gaussian_softplus', 'opusv1_gaussian_margin_softplus'):
        data = prior[name]
        all_models[name] = {
            'mIoU': data['best_mIoU']['mIoU'], 'mIoU_epoch': data['best_mIoU']['epoch'],
            'IoU': data['best_IoU']['IoU'], 'IoU_epoch': data['best_IoU']['epoch'],
            'per_class': data['best_mIoU']['per_class']}
    for name, run in runs.items():
        all_models[name] = {
            'mIoU': run['best_mIoU']['mIoU'], 'mIoU_epoch': run['best_mIoU']['epoch'],
            'IoU': run['best_IoU']['IoU'], 'IoU_epoch': run['best_IoU']['epoch'],
            'per_class': run['best_mIoU']['per_class']}
    lines = [START, '## Gaussian sparse render 后续实验：统一 mini 协议', '',
             '下表沿用上方四模型的 4096/1024 划分、seed 928、5 epoch、相机可见验证 mask。新模型均不使用额外推理空类偏移；最佳 mIoU 与最佳二值 IoU 可来自不同 epoch。原四模型主表保持历史结果。', '',
             '| 模型 | 最佳 mIoU (%) | epoch | 最佳二值 IoU (%) | epoch |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, data in all_models.items():
        lines.append(f'| {name} | {data["mIoU"]:.2f} | {data["mIoU_epoch"]} | '
                     f'{data["IoU"]:.2f} | {data["IoU_epoch"]} |')
    lines += ['', '原模型训练集选值空类减 4 的验证成绩为 20.46%/38.89%；softplus-only 训练集选值减 4 为 22.28%/41.38%。这两项仍属于推理校准参照，不并入零偏移模型排序。', '',
              '### 逐类对照：每模型最佳 mIoU checkpoint', '',
              '| 类别 | 原四模型最佳 | 相机 mask | 可学习偏置 | 类别 margin | 全部零偏移模型最佳 |',
              '| --- | --- | ---: | ---: | ---: | --- |']
    for cls in CLASSES:
        old = original['per_class_best'][cls]
        old_label = ', '.join(old['winners']) + f' ({old["IoU"]:.2f})'
        scores = {name: data['per_class'][cls] for name, data in all_models.items()}
        top = max(scores.values())
        winners = ', '.join(name for name, value in scores.items() if value == top)
        values = [f'{runs[name]["best_mIoU"]["per_class"][cls]:.2f}' if name in runs else '—'
                  for name in GROUPS]
        lines.append(f'| {cls} | {old_label} | ' + ' | '.join(values)
                     + f' | {winners} ({top:.2f}) |')
    if len(runs) == len(GROUPS):
        lines += ['', '### 实验结论', '',
                  '训练占用损失改用相机可见 mask 后，零额外推理偏移的最佳 mIoU / 同一 checkpoint 二值 IoU '
                  '达到 22.41% / 45.93%；进一步训练学习 empty_bias 后达到 24.77% / 50.30%，'
                  '均超过原模型训练集选值空类减 4 的 20.46% / 38.89%。'
                  '可学习偏置为 3.555，是模型参数，验证时不另选校准值。', '',
                  '类别感知 margin 为 21.55% / 47.78%；相对仅相机 mask，二值 IoU 略升而 mIoU 下降，'
                  '且 others、bicycle、motorcycle 的类别 IoU 明显退化，因此不把它与可学习偏置组合。'
                  '逐类最佳模型并不统一：可学习偏置组领先 8/17 类，GaussianFormer 仍领先 6/17 类。'
                  '偏置组的 opacity 与 scale 仍高度饱和，稀有类问题尚未完全解决。'
                  '这些结论限于固定划分、单种子、5 epoch 的 mini 实验。']
    lines += ['', '各实验的设计与结果：' + '；'.join(
        f'[{label}设计]({slug}_design.md)、[{label}结果]({slug}_results.md)'
        for _, (label, slug) in GROUPS.items()) + '。'
        '几何覆盖诊断见[实验 D](gaussian_coverage_results.md)。', END]
    path = PLAN_ROOT / 'mini_occ3d_4096_1024_results.md'
    existing = path.read_text(encoding='utf-8').replace('ZZZ# Occ3D', '# Occ3D')
    section = '\n'.join(lines)
    if START in existing and END in existing:
        before = existing[:existing.index(START)].rstrip()
        after = existing[existing.index(END) + len(END):].strip()
        updated = before + '\n\n' + section + ('\n\n' + after if after else '\n')
    else:
        updated = existing.rstrip() + '\n\n' + section + '\n'
    path.write_text(updated, encoding='utf-8')
    print(f'{len(runs)} follow-up runs summarized: {path}')


if __name__ == '__main__':
    main()
