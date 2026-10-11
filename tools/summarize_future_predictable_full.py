#!/usr/bin/env python3
"""Write only observed native-head scores, never substitute probe scores."""
import argparse
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY / 'out/future2current/predictable_full_finetune_4096x1024'
DEFAULT_REPORT = REPOSITORY / 'analysis/hypotheses/h3_predictability/full_finetune4096/results.md'


def score_table(records, baseline, metric, group=None):
    lines = ['| 对象 | 1 秒 | 2 秒 | 3 秒 |', '|---|---:|---:|---:|']
    for label, result in records:
        cells = []
        for horizon in ('1s', '2s', '3s'):
            value = (result['metrics'][horizon][metric] if group is None else
                     result['groups'][horizon][group][metric])
            reference = (baseline['metrics'][horizon][metric] if group is None else
                         baseline['groups'][horizon][group][metric])
            cells.append(f'{value:.3f}（参照）' if label == '原版 SparseWorld' else
                         f'{value:.3f}（{value - reference:+.3f}）')
        lines.append('| ' + ' | '.join([label, *cells]) + ' |')
    return '\n'.join(lines)


def write_report(root=DEFAULT_ROOT, destination=DEFAULT_REPORT):
    root, destination = Path(root), Path(destination)
    report = json.loads((root / 'results.json').read_text())
    protocol = json.loads((root / 'manifest.json').read_text())
    status = dict(preparing='正在准备冻结教师目标', training='训练／逐 epoch 验证中',
                  complete='全部训练和验证完成', failed='运行失败，需要检查错误').get(
                      report['status'], report['status'])
    text = ['# 4096／1024 全参数微调：真实头结果', '', '## 先看状态', '',
            f'**{status}。**本报告只收录已经完成的原版头验证；未完成的 epoch 不填入推测值。', '',
            f'训练样本 {len(protocol["train"]["indices"])}；验证样本 {len(protocol["val"]["indices"])}；'
            f'每组 {protocol["epochs"]} epoch；训练种子 {protocol["seeds"]}。Student 全部参数可训练，Teacher 冻结。', '',
            '教师在目标时刻使用真实未来图像和原版感知头；Student 仅使用当前／历史图像和原版预测头。'
            '所有正负差值只相对本轮同一验证集合上的原版 SparseWorld；不使用共享读取器成绩。', '',
            '**训练协议差异必须保留：**本轮为了全参数微调，将图像 backbone 的 BN 运行统计设为更新，'
            '并开放当前步未来点位置损失的梯度；原版配置固定 BN 统计，原版未来点输出则分离梯度。'
            '原版输出分支和初始化前向相同，不等于训练梯度与统计更新协议完全相同。'
            '这些因素的配对消融见[训练协议续实验](../../future_distillation_causes/s0_training_protocol/results.md)。'
            '因此，此表的共同降点不能直接解释成教师特征不可用。', '',
            '## 1．实际进度', '', '| 组别 | 已完成验证的 epoch | 最近训练进度 |', '|---|---|---|']
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            record = report['arms'].get(f'{arm}_seed{seed}', {})
            completed = ', '.join(sorted(record.get('epochs', {}), key=int)) or '无'
            progress = record.get('progress')
            latest = (f'epoch {progress["epoch"]}，{progress["processed"]}/{progress["total"]} 样本，'
                      f'累计 {progress["updates"]} 次更新' if progress else '尚未开始')
            text.append(f'| `{arm}`，种子 {seed} | {completed} | {latest} |')
    if report.get('error'):
        text.extend(['', f'错误：`{report["error"]}`。'])
    baseline = report.get('baseline')
    if baseline:
        records = [('原版 SparseWorld', baseline)]
        if report.get('teacher'):
            records.append(('真实未来感知教师', report['teacher']))
        for arm in protocol['arms']:
            for seed in protocol['seeds']:
                for epoch, result in sorted(report['arms'].get(f'{arm}_seed{seed}', {}).get('epochs', {}).items(), key=lambda pair: int(pair[0])):
                    records.append((f'`{arm}`，epoch {epoch}，种子 {seed}', result))
        text.extend(['', '## 2．总体原版头性能', '', '单位为 %，括号为相对真实基线的百分点变化。', '',
                     '### mIoU：17 个非空语义类别', '', score_table(records, baseline, 'mIoU'), '',
                     '### OccIoU：空／非空', '', score_table(records, baseline, 'OccIoU')])
        text.extend(['', '## 3．动态／静态类别', '',
                     '组 mIoU 是组内各类别 IoU 的平均；组 IoU 将组内类别合并后比较组内／组外。'
                     '动态组为 bicycle、bus、car、construction vehicle、motorcycle、pedestrian、trailer、truck；'
                     '静态组为 barrier、traffic cone、driveable surface、other flat、sidewalk、terrain、manmade、vegetation。'
                     'others 不属于这两组，但参与总体 mIoU。'])
        for group, label in [('dynamic', '动态'), ('static', '静态')]:
            for metric in ('mIoU', 'IoU'):
                text.extend(['', f'### {label} {metric}', '', score_table(records, baseline, metric, group)])
    else:
        text.extend(['', '## 2．性能数据', '', '本轮 1024 样本的真实基线尚未完成，暂不报告性能或差值。'
                     '此前 256／512 样本成绩不能填入此表。'])
    text.extend(['', '## 4．如何解释结果', '',
                 '先确认正确教师组在原版头上的性能是否超过真实基线；再比较同预算的任务训练组，'
                 '判断提升是否来自教师监督。本文只把正负差值写成相对基线的变化，同预算组之间的比较使用原始分数。', '',
                 '正确目标与打乱目标也需要比较，才能判断目标内容是否有用，而非只来自额外约束。'
                 '一个配对种子不能证明跨随机种子的稳定提升；也不能证明未来信息存在可预测性理论下界。'
                 '本轮同时改变数据规模、时长、可训练参数及任务损失，不能只归因于其中一个变化。', '',
                 '## 5．可追溯数据', '',
                 f'- 实际数值与训练进度：`{root / "results.json"}`',
                 f'- 样本清单和设置：`{root / "manifest.json"}`',
                 f'- 日志：`{root / "run.log"}`',
                 '- 每个 epoch 的 `.npz` 保存逐样本混淆矩阵、占据交集和并集；对应 `.pth` 保存模型和续训状态。', '',
                 '实验设置见[设计文档](design.md)。'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(text) + '\n', encoding='utf8')
    temporary.replace(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    write_report(args.root, args.report)
