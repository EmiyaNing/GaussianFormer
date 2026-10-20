"""Summarize the independent full Occ3D ten-epoch, no-benchmark experiment."""
import argparse
import json
import re
from pathlib import Path

from summarize_mini_occ3d import CLASSES, ROOT, parse_log


def parse_memory_logs(paths):
    rows = {}
    allocator = None
    for path in paths:
        for line in path.read_text(errors='replace').splitlines():
            if 'CUDA allocator configuration: ' in line:
                allocator = line.split('CUDA allocator configuration: ', 1)[1]
            if '[CUDA_MEMORY]' not in line:
                continue
            values = dict(re.findall(r'(epoch|iter|allocated|reserved|peak_allocated|'
                                     r'peak_reserved|inactive_split|alloc_retries)=([0-9.]+)', line))
            if len(values) != 8:
                continue
            row = {key: int(value) if key in ('epoch', 'iter', 'alloc_retries')
                   else float(value) for key, value in values.items()}
            rows[(row['epoch'], row['iter'])] = row
    ordered = [rows[key] for key in sorted(rows)]
    return {
        'allocator_configuration': allocator,
        'records': ordered,
        'peak_allocated_mib': max((row['peak_allocated'] for row in ordered), default=None),
        'peak_reserved_mib': max((row['peak_reserved'] for row in ordered), default=None),
        'last_record': ordered[-1] if ordered else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'work_dirs/full_occ3d_10e_no_benchmark'
    run = root / 'opusv1_gaussian_softplus_learned_bias'
    logs = sorted(run.glob('*.log'))
    epochs = parse_log(logs)
    memory = parse_memory_logs(logs)
    if not args.allow_partial and [r['epoch'] for r in epochs] != list(range(1, 11)):
        raise RuntimeError(f'Expected ten complete evaluations; found {len(epochs)}')
    summary = {'epochs': epochs, 'complete': len(epochs) == 10,
               'train_samples': 28130, 'val_samples': 6019,
               'cudnn_benchmark': False, 'seed': 928, 'training_memory': memory}
    lines = ['# 完整 Occ3D：关闭 cuDNN benchmark 的十轮实验结果', '',
             '协议见[实验设计](full_occ3d_learned_bias_10e_no_benchmark_design.md)。', '',
             f'已记录 {len(epochs)}/10 轮完整验证。以下结果仅统计已完成的验证轮次。', '']
    if epochs:
        best = max(epochs, key=lambda row: (row['mIoU'], -row['epoch']))
        best_iou = max(epochs, key=lambda row: (row['IoU'], -row['epoch']))
        summary.update(best_mIoU=best, best_IoU=best_iou)
        for metric, row in [('mIoU', best), ('IoU', best_iou)]:
            checkpoint = run / f"epoch_{row['epoch']}.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            summary[f'best_{metric}_checkpoint'] = str(checkpoint.relative_to(ROOT))
        lines += [f"最佳 mIoU：**{best['mIoU']:.2f}%**，epoch {best['epoch']}；"
                  f"同 checkpoint 二值 IoU：**{best['IoU']:.2f}%**。",
                  f"独立最佳二值 IoU：**{best_iou['IoU']:.2f}%**，epoch {best_iou['epoch']}。", '',
                  '| epoch | mIoU (%) | 二值 IoU (%) |', '| --- | ---: | ---: |']
        lines += [f"| {r['epoch']} | {r['mIoU']:.2f} | {r['IoU']:.2f} |" for r in epochs]
        lines += ['', '## 最佳 mIoU checkpoint 的逐类 IoU', '',
                  '| 类别 | IoU (%) |', '| --- | ---: |']
        lines += [f"| {c} | {best['per_class'][c]:.2f} |" for c in CLASSES]
        lines += ['', '## 与原完整数据五轮实验的比较', '',
                  '原五轮最佳 mIoU/IoU 为 33.507772%/58.671242%。',
                  f"本次已完成轮次的最佳 mIoU 差值为 {best['mIoU'] - 33.507771746200675:+.2f} 个百分点，"
                  f"独立最佳 IoU 差值为 {best_iou['IoU'] - 58.671241760253906:+.2f} 个百分点。",
                  '本次总训练预算和学习率调度长度也变为十轮，因此差值不能单独归因于关闭 cuDNN benchmark。', '']
    else:
        lines += ['任务已配置，尚无完整验证结果。', '']
    if memory['records']:
        last = memory['last_record']
        lines += ['## 真实训练进程显存', '',
                  f"分配器配置：`{memory['allocator_configuration']}`。截至 epoch {last['epoch']}、"
                  f"iteration {last['iter']}，记录到的 allocated/reserved 峰值为 "
                  f"**{memory['peak_allocated_mib']/1024:.2f}/{memory['peak_reserved_mib']/1024:.2f} GiB**。",
                  '来源是实际训练进程 CUDA_MEMORY 日志，单位转换为 GiB；当前张量占用、缓存保留与历史峰值分别记录。',
                  '记录仅覆盖已运行到的步骤，不能据此宣称后续十轮的最大值。', '']
    lines += ['原始日志和逐轮 checkpoint 位于 `work_dirs/full_occ3d_10e_no_benchmark/opusv1_gaussian_softplus_learned_bias/`。',
              '完整汇总见 `work_dirs/full_occ3d_10e_no_benchmark/summary.json`。', '']
    root.mkdir(parents=True, exist_ok=True)
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    plans = ROOT / 'plans/sparse_render_offical_9_28'
    (plans / 'full_occ3d_learned_bias_10e_no_benchmark_results.md').write_text('\n'.join(lines))
    main_report = plans / 'mini_occ3d_4096_1024_results.md'
    start, end = '<!-- full_occ3d_10e_start -->', '<!-- full_occ3d_10e_end -->'
    section = [start, '## 完整数据十轮：关闭 cuDNN benchmark', '',
               f'独立十轮任务已记录 {len(epochs)}/10 轮验证；'
               '[设计](full_occ3d_learned_bias_10e_no_benchmark_design.md)与'
               '[结果](full_occ3d_learned_bias_10e_no_benchmark_results.md)。']
    if epochs:
        section += [f"最佳 mIoU {best['mIoU']:.2f}%（epoch {best['epoch']}），"
                    f"独立最佳二值 IoU {best_iou['IoU']:.2f}%（epoch {best_iou['epoch']}）。"]
    section += ['此项采用完整数据集，独立于上文 mini 排名。', end]
    text = main_report.read_text()
    block = '\n'.join(section)
    if start in text and end in text:
        begin = text.index(start)
        finish = text.index(end, begin) + len(end)
        text = text[:begin] + block + text[finish:]
    else:
        text += '\n' + block + '\n'
    main_report.write_text(text)
    print(f'Summarized {len(epochs)} epochs into {root / "summary.json"}')


if __name__ == '__main__':
    main()
