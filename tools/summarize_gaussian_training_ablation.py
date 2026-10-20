"""Summarize the five-epoch train-time empty-class calibration ablation."""
from __future__ import annotations

import json
from pathlib import Path

from summarize_mini_occ3d import CLASSES, ROOT, parse_log


RUN_ROOT = ROOT / 'work_dirs/mini_occ3d_4096_1024'
DOC = ROOT / 'plans/sparse_render_offical_9_28/train_time_empty_class_ablation.md'
NAMES = (
    'opusv1_gaussian_sparse_render',
    'opusv1_gaussian_margin',
    'opusv1_gaussian_softplus',
    'opusv1_gaussian_margin_softplus',
)
LABELS = ('原模型', '仅类别平衡 margin loss', '仅渲染语义 softplus', '两者结合')


def main():
    runs = {}
    for name in NAMES:
        logs = sorted((RUN_ROOT / name).glob('*.log'))
        if not logs:
            raise FileNotFoundError(f'{name}: no training log')
        epochs = parse_log(logs)
        if [item['epoch'] for item in epochs] != list(range(1, 6)):
            raise ValueError(f'{name}: expected evaluations at epochs 1..5, got '
                             f'{[item["epoch"] for item in epochs]}')
        best_miou = max(epochs, key=lambda row: (row['mIoU'], -row['epoch']))
        best_iou = max(epochs, key=lambda row: (row['IoU'], -row['epoch']))
        runs[name] = {
            'logs': [str(p.relative_to(ROOT)) for p in logs],
            'epochs': epochs,
            'best_mIoU': best_miou,
            'best_IoU': best_iou,
        }

    baseline = runs[NAMES[0]]
    base_miou = baseline['best_mIoU']['mIoU']
    base_iou = baseline['best_IoU']['IoU']
    lines = [
        '# OPUSv1 Gaussian：训练时空类校准消融', '',
        '## 实验协议', '',
        '- 使用固定 mini Occ3D 划分：训练 4096 帧、验证 1024 帧；随机种子 928。',
        '- 四组均为六相机单帧、704×256、600 query、每 query 128 child、训练 5 epoch、micro-batch 1、梯度累积 4；验证使用相机可见 mask。',
        '- 验证时直接对 18 个原始渲染占用 logit 做 argmax；空类通道偏移固定为 0，没有验证集调参。mIoU 是 17 个非空语义类平均；IoU 是二值占用 IoU。',
        '- 三个新配置都继承原模型的数据、优化器、主干预训练权重和 OPUS set loss；只改表中所述部分。',
        '- margin loss 对最终渲染结果计算 `max(foreground logits) - empty logit`，在全部训练体素上加入正类按当前 batch 负/正比例加权的 BCE（权重上限 50）与 soft Dice，各系数为 1。',
        '- softplus 只作用于送入 GaussianFormer head 的语义值；OPUS set loss 继续使用原始 query 分类 logits。',
        '- 表中的最佳 mIoU 与最佳二值 IoU 各自在同一组的五个验证 epoch 中独立选取。',
        '', '## 最优结果', '',
        '| 变体 | 最佳 mIoU (%) | epoch | 相对原模型 | 最佳 IoU (%) | epoch | 相对原模型 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, label in zip(NAMES, LABELS):
        run = runs[name]
        miou = run['best_mIoU']
        iou = run['best_IoU']
        lines.append(f'| {label} | {miou["mIoU"]:.2f} | {miou["epoch"]} | '
                     f'{miou["mIoU"] - base_miou:+.2f} | {iou["IoU"]:.2f} | '
                     f'{iou["epoch"]} | {iou["IoU"] - base_iou:+.2f} |')
    lines.extend(['', '## 各 epoch 验证指标', '',
                  '| 变体 | epoch | mIoU (%) | 二值 IoU (%) |',
                  '| --- | ---: | ---: | ---: |'])
    for name, label in zip(NAMES, LABELS):
        for row in runs[name]['epochs']:
            lines.append(f'| {label} | {row["epoch"]} | {row["mIoU"]:.2f} | '
                         f'{row["IoU"]:.2f} |')
    diagnostics = {}
    for name in NAMES:
        path = (RUN_ROOT / 'opusv1_gaussian_calibration_val1024.json' if name == NAMES[0]
                else RUN_ROOT / name / 'diagnostic_val1024.json')
        if path.is_file():
            diagnostics[name] = json.loads(path.read_text(encoding='utf-8'))
    if len(diagnostics) == len(NAMES):
        lines.extend(['', '## 空类决策诊断', '',
                      '同一 1024 帧验证集、相机可见 mask；各新变体使用最佳 mIoU checkpoint。以下均取空类偏移为 0 的结果。', '',
                      '| 变体 | 预测占用率 (%) | GT 占用率 (%) | 占用 precision (%) | 占用 recall (%) |',
                      '| --- | ---: | ---: | ---: | ---: |'])
        for name, label in zip(NAMES, LABELS):
            row = diagnostics[name]['empty_logit_shift_metrics']['0.0']
            lines.append(f'| {label} | {row["predicted_occupied_fraction"]:.2f} | '
                         f'{row["gt_occupied_fraction"]:.2f} | {row["precision"]:.2f} | '
                         f'{row["recall"]:.2f} |')
    train_calibration_path = RUN_ROOT / 'opusv1_gaussian_softplus/calibration_train256.json'
    original_calibration_path = RUN_ROOT / 'opusv1_gaussian_calibrated_val1024.json'
    if (train_calibration_path.is_file() and original_calibration_path.is_file()
            and NAMES[2] in diagnostics):
        train_calibration = json.loads(train_calibration_path.read_text(encoding='utf-8'))
        train_scores = train_calibration['empty_logit_shift_metrics']
        chosen = max(train_scores, key=lambda shift: train_scores[shift]['mIoU'])
        original_calibrated = json.loads(
            original_calibration_path.read_text(encoding='utf-8'))
        softplus_val_scores = diagnostics[NAMES[2]]['empty_logit_shift_metrics']
        if train_calibration['samples'] == 256 and chosen in softplus_val_scores:
            lines.extend(['', '## 训练集选值的校准诊断', '',
                          f'仅在训练集随机 256 帧上选择 softplus-only 的空类 logit 偏移：按 mIoU 最大选中 {chosen}；同一值也使该训练子集的二值 IoU 最大。然后固定该值，在完整 1024 帧验证集读取结果。此项仍需要推理时偏移，仅用于判断模型能力与决策校准的差距。', '',
                          '| 模型与决策 | 验证 mIoU (%) | 验证二值 IoU (%) |',
                          '| --- | ---: | ---: |'])
            comparisons = (
                ('原模型，偏移 0', diagnostics[NAMES[0]]['empty_logit_shift_metrics']['0.0']),
                ('原模型，训练集选值偏移 4', original_calibrated['empty_logit_shift_metrics']['4.0']),
                ('softplus-only，偏移 0', softplus_val_scores['0.0']),
                (f'softplus-only，训练集选值偏移 {chosen}', softplus_val_scores[chosen]),
            )
            for label, row in comparisons:
                lines.append(f'| {label} | {row["mIoU"]:.2f} | {row["IoU"]:.2f} |')
    frozen_base_path = RUN_ROOT / 'opusv1_gaussian_calibration_val1024.json'
    frozen_softplus_path = RUN_ROOT / 'opusv1_gaussian_softplus_untrained_val1024.json'
    if frozen_base_path.is_file() and frozen_softplus_path.is_file():
        frozen_base = json.loads(frozen_base_path.read_text(encoding='utf-8'))
        frozen_softplus = json.loads(frozen_softplus_path.read_text(encoding='utf-8'))
        if frozen_base['samples'] == frozen_softplus['samples'] == 1024:
            lines.extend(['', '## 冻结权重的先行对照', '',
                          '使用原模型 epoch 5 权重和完整 mini 验证集 1024 帧，只切换渲染语义 softplus，不更新参数；这是机制检查，不代替上面的重训实验。', '',
                          '| 推理方式 | mIoU (%) | 二值 IoU (%) | 占用召回率 (%) | 预测占用率 (%) |',
                          '| --- | ---: | ---: | ---: | ---: |'])
            for label, report in (('原始渲染', frozen_base),
                                  ('仅渲染语义 softplus', frozen_softplus)):
                row = report['empty_logit_shift_metrics']['0.0']
                lines.append(f'| {label} | {row["mIoU"]:.2f} | {row["IoU"]:.2f} | '
                             f'{row["recall"]:.2f} | '
                             f'{row["predicted_occupied_fraction"]:.2f} |')
    lines.extend(['', '## 逐类别 IoU', '',
                  '每组取该组最佳 mIoU 的 checkpoint，保持与总体指标比较一致。', '',
                  '| 类别 | 原模型 | 仅 margin | 仅 softplus | 两者结合 | 最佳变体 |',
                  '| --- | ---: | ---: | ---: | ---: | --- |'])
    for cls in CLASSES:
        scores = [runs[name]['best_mIoU']['per_class'][cls] for name in NAMES]
        winners = [LABELS[i] for i, value in enumerate(scores) if value == max(scores)]
        lines.append(f'| {cls} | ' + ' | '.join(f'{value:.2f}' for value in scores)
                     + f' | {", ".join(winners)} |')
    improved = [LABELS[i] for i, name in enumerate(NAMES) if i and
                runs[name]['best_mIoU']['mIoU'] > base_miou and
                runs[name]['best_IoU']['IoU'] > base_iou]
    lines.extend(['', '## 结论与限制', ''])
    if improved:
        lines.append('在不使用推理空类偏移的条件下，同时提升 mIoU 与二值 IoU 的变体：'
                     + '、'.join(improved) + '。')
    else:
        lines.append('三种训练改动均未在五轮 mini 实验中同时超过原模型的 mIoU 与二值 IoU。')
    lines.extend([
        '本实验只有一个随机种子和较短训练预算；是否能迁移到完整 Occ3D 训练还需独立验证。',
        '之前空类通道减 4 的推理校准是另一项训练后分析，不参与此表，也未用于选择这三组的 checkpoint。',
        '', '## 配置与数据来源', '',
    ])
    for name in NAMES:
        config = f'config/mini_occ3d_4096_1024/{name}.py'
        lines.append(f'- `{name}`：`{config}`；训练日志：'
                     + '、'.join(f'`{log}`' for log in runs[name]['logs']) + '。')
    lines.extend([
        '- 运行命令：`python3.10 train.py --py-config config/mini_occ3d_4096_1024/<变体>.py --work-dir work_dirs/mini_occ3d_4096_1024/<变体> --gradient-accumulation 4 --seed 928`。',
        '',
    ])
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text('\n'.join(lines), encoding='utf-8')
    main_doc = DOC.parent / 'mini_occ3d_4096_1024_results.md'
    if main_doc.is_file():
        existing = main_doc.read_text(encoding='utf-8')
        relative_link = 'train_time_empty_class_ablation.md'
        if relative_link not in existing:
            main_doc.write_text(
                existing.rstrip() + '\n\n## 训练时空类校准消融\n\n'
                f'详见[margin loss 与渲染语义 softplus 消融]({relative_link})。\n',
                encoding='utf-8')
    json_path = RUN_ROOT / 'gaussian_training_ablation_summary.json'
    json_path.write_text(json.dumps(runs, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    print(DOC)


if __name__ == '__main__':
    main()
