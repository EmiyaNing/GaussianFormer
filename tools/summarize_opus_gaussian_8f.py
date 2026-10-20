"""Update temporal size experiment reports and the original comparison."""
import json
from pathlib import Path
import re

from summarize_mini_occ3d import CLASSES, ROOT, parse_log

SIZES = ('tiny', 'small', 'middle', 'large')
RUN_ROOT = ROOT / 'work_dirs/full_occ3d_gaussian_8f_30e'
PLANS = ROOT / 'plans/sparse_render_offical_9_28/full_occ3d_gaussian_8f_30e'


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def main():
    state_file = RUN_ROOT / 'status.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {'state': 'prepared'}
    status_label = state.get('description', state.get('state'))
    summary = dict(train_samples=28130, val_samples=6019, num_frames=8,
                   max_epochs=30, status=state, models={}, per_class_best={})
    table = ['| 尺寸 | 已验证轮次 | 最佳 mIoU (%) / epoch | 该轮 IoU (%) | 独立最佳 IoU (%) / epoch |',
             '| --- | ---: | ---: | ---: | ---: |']
    for size in SIZES:
        run = RUN_ROOT / size
        epochs = parse_log(sorted(run.glob('20*.log')))
        preflight_path = RUN_ROOT / 'preflight' / f'{size}.json'
        preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else None
        memory_path = run / 'phase_memory.jsonl'
        memory_rows = []
        if memory_path.exists():
            for line in memory_path.read_text().splitlines():
                try:
                    memory_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # May be reading the currently appended line.
        memory = {phase: {
            'recorded_phases': len([r for r in memory_rows if r['phase'] == phase]),
            'peak_allocated_mib': max((r['peak_allocated_mib'] for r in memory_rows
                                      if r['phase'] == phase), default=None),
            'peak_reserved_mib': max((r['peak_reserved_mib'] for r in memory_rows
                                     if r['phase'] == phase), default=None)}
            for phase in ('train', 'val')}
        record = dict(epochs=epochs, complete=[e['epoch'] for e in epochs] == list(range(1, 31)),
                      preflight=preflight, phase_memory=memory)
        summary['models'][size] = record
        lines = [f'# {size}：历史 8 帧完整 Occ3D 30 轮结果', '',
                 f'协议见 [{size} 实验设计]({size}_design.md)；'
                 f'已完成完整验证 **{len(epochs)}/30** 轮。', '',
                 f'状态：{status_label}；当前尺寸：`{state.get("current_size") or "—"}`。', '']
        if epochs:
            best = max(epochs, key=lambda r: (r['mIoU'], -r['epoch']))
            best_iou = max(epochs, key=lambda r: (r['IoU'], -r['epoch']))
            record.update(best_mIoU=best, best_IoU=best_iou)
            for metric, row in [('mIoU', best), ('IoU', best_iou)]:
                checkpoint = run / f"epoch_{row['epoch']}.pth"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                link = run / f'best_{metric}.pth'
                if link.is_symlink():
                    link.unlink()
                if not link.exists():
                    link.symlink_to(checkpoint.name)
                record[f'best_{metric}_checkpoint'] = str(checkpoint.relative_to(ROOT))
            table.append(f"| {size} | {len(epochs)}/30 | {best['mIoU']:.2f} / {best['epoch']} | "
                         f"{best['IoU']:.2f} | {best_iou['IoU']:.2f} / {best_iou['epoch']} |")
            lines += [f"截至已完成轮次：最佳 mIoU **{best['mIoU']:.2f}%**（epoch {best['epoch']}），"
                      f"同轮 IoU **{best['IoU']:.2f}%**；独立最佳 IoU **{best_iou['IoU']:.2f}%**"
                      f"（epoch {best_iou['epoch']}）。", '',
                      '| epoch | mIoU (%) | IoU (%) |', '| --- | ---: | ---: |']
            lines += [f"| {r['epoch']} | {r['mIoU']:.2f} | {r['IoU']:.2f} |" for r in epochs]
            lines += ['', '## 最佳 mIoU checkpoint 的逐类结果', '',
                      '| 类别 | IoU (%) |', '| --- | ---: |']
            lines += [f"| {c} | {best['per_class'][c]:.2f} |" for c in CLASSES]
            lines += ['', '## 结果分析', '',
                      '结果来自完整验证集。尺寸之间以各自最佳 mIoU checkpoint 比较；'
                      '短测 loss 和显存不能代表最终精度。总计 30 轮完成前不作最终尺寸优劣结论。', '']
        else:
            table.append(f'| {size} | 0/30 | — | — | — |')
            lines += ['尚无完整验证结果，暂不能判断精度。', '']
        if preflight:
            lines += ['## 真实样本短测', '',
                      f"全 query、48 张相机图像、完整损失及 {len(preflight['loss'])} 次反向/AdamW 更新已通过；"
                      '8 帧图像特征均有非零梯度。',
                      f"参数量 {preflight['num_parameters']:,}；短测训练 allocated/reserved 峰值 "
                      f"{preflight['train_peak']['allocated_mib']/1024:.2f}/"
                      f"{preflight['train_peak']['reserved_mib']/1024:.2f} GiB。",
                      '这是单一样本短测，正式训练峰值会另行累计。', '']
        lines += ['## 正式进程显存', '',
                  '按 train/val 阶段切换重置 allocated 峰值；reserved 包含分配器跨阶段保留的缓存。', '',
                  '| 阶段 | 已结束阶段数 | allocated 峰值 GiB | reserved 峰值 GiB |',
                  '| --- | ---: | ---: | ---: |']
        for phase, values in memory.items():
            a, r = values['peak_allocated_mib'], values['peak_reserved_mib']
            af = f'{a/1024:.2f}' if a is not None else '—'
            rf = f'{r/1024:.2f}' if r is not None else '—'
            lines.append(f"| {phase} | {values['recorded_phases']} | {af} | {rf} |")
        lines += ['', f'原始数据：`work_dirs/full_occ3d_gaussian_8f_30e/{size}/`。', '']
        atomic_write(PLANS / f'{size}_results.md', '\n'.join(lines))
    comparison = ['# 历史 8 帧：四种尺寸完整 Occ3D 对比', '',
                  '28,130 个训练样本 / 6,019 个验证样本；每个尺寸训练 30 轮。', '', *table, '',
                  '## 各类别最佳模型', '',
                  '使用每个尺寸当前最佳 mIoU checkpoint，不能逐类任意换 epoch。', '',
                  '| 类别 | tiny | small | middle | large | 当前最佳 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for cls in CLASSES:
        values = {s: r['best_mIoU']['per_class'][cls] for s, r in summary['models'].items()
                  if 'best_mIoU' in r}
        winner = max(values, key=values.get) if values else '—'
        summary['per_class_best'][cls] = dict(model=winner, IoU=values.get(winner))
        cells = [f'{values[s]:.2f}' if s in values else '—' for s in SIZES]
        comparison.append('| ' + ' | '.join([cls, *cells, winner]) + ' |')
    comparison += ['', '## 分析口径', '',
                   f'当前状态：{status_label}。',
                   '四个尺寸最终都输出 76,800 个 Gaussian；尺寸同时改变高维 query 数、'
                   '每 query 子 Gaussian 数及 sampling 点数，因此属于官方容量设置的整体比较。',
                   '尚未完成的尺寸不参与最终排名。此前单帧、五轮/十轮结果的训练预算、'
                   '初始化和时序输入不同，不能将差值单独归因于历史帧或模型尺寸。', '']
    atomic_write(PLANS / 'comparison.md', '\n'.join(comparison))
    atomic_write(RUN_ROOT / 'summary.json', json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    main_report = ROOT / 'plans/sparse_render_offical_9_28/mini_occ3d_4096_1024_results.md'
    begin, end = '<!-- full_occ3d_gaussian_8f_start -->', '<!-- full_occ3d_gaussian_8f_end -->'
    block = '\n'.join([begin, '## 完整数据：历史 8 帧、四种尺寸、30 轮', '',
                       '[设计与运行说明](full_occ3d_gaussian_8f_30e/design.md)；'
                       '[实时结果与逐类对比](full_occ3d_gaussian_8f_30e/comparison.md)。', '',
                       f"当前状态：{status_label}；当前尺寸：{state.get('current_size') or '—'}。", '',
                       *table, '', '与上文 mini 实验分别报告，尚未完成的训练不作最终精度结论。', end])
    text = main_report.read_text()
    if begin in text and end in text:
        start = text.index(begin)
        finish = text.index(end, start) + len(end)
        text = text[:start] + block + text[finish:]
    else:
        text += '\n' + block + '\n'
    atomic_write(main_report, text)


if __name__ == '__main__':
    main()
