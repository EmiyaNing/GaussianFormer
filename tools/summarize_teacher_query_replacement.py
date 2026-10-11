#!/usr/bin/env python3
"""Generate paired-scene evidence and a readable Teacher-query result note."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.summarize_future_image_joint import bootstrap_pair, score


ARMS = ('task_only', 'teacher_feature', 'teacher_position',
        'teacher_joint', 'teacher_joint_shuffled_feature')
HORIZONS = ('1s', '2s', '3s')
NAMES = dict(task_only='不替换', teacher_feature='只替换 Teacher 特征',
             teacher_position='只替换 Teacher 位置',
             teacher_joint='Teacher 特征＋位置',
             teacher_joint_shuffled_feature='Teacher 位置＋打乱特征')


def arrays(root, filename):
    with np.load(root / filename) as data:
        return {key: data[key].astype(np.float64)
                for key in ('hist', 'intersection', 'union')}


def combined(run, indices):
    return score(**{key: value[indices].sum(0) for key, value in run.items()})


def f(value):
    return f'{value:+.3f}'


def main(args):
    path = Path(args.report)
    report = json.loads(path.read_text())
    seeds = [str(seed) for seed in report['args']['training_seeds']]
    for seed in seeds:
        missing = [arm for arm in ARMS if arm not in report['runs'].get(seed, {})]
        if missing:
            raise RuntimeError(f'Incomplete experiment: seed={seed}, missing={missing}')
    root = path.parent
    per_arm = {arm: [arrays(root, report['runs'][seed][arm]['confusion_file'])
                     for seed in seeds] for arm in ARMS}
    native = arrays(root, report['teacher_native']['confusion_file'])
    scenes = report['val_manifest']['scenes']
    subsets = [('full', list(range(len(scenes))))]
    if report['confirmation_indices']:
        subsets.append(('confirmation', report['confirmation_indices']))
    summary = dict(seeds=seeds, bootstrap_repeats=args.bootstrap_repeats,
                   interval_scope='paired scene resampling, conditional on trained seeds',
                   subsets={})
    for name, indices in subsets:
        result = dict(samples=len(indices), scenes=len({scenes[i] for i in indices}),
                      scores={}, comparisons={})
        for arm, runs in per_arm.items():
            measurements = [combined(run, indices) for run in runs]
            result['scores'][arm] = {
                metric: np.stack([value[metric] for value in measurements]).mean(0).tolist()
                for metric in ('mIoU', 'OccIoU')}
        for arm in ARMS[1:]:
            comparisons = {}
            for control in ('task_only', 'teacher_feature', 'teacher_position',
                            'teacher_joint_shuffled_feature'):
                if control == arm:
                    continue
                differences = []
                for a, b in zip(per_arm[arm], per_arm[control]):
                    left, right = combined(a, indices), combined(b, indices)
                    differences.append({metric: (left[metric]-right[metric]).tolist()
                                        for metric in ('mIoU', 'OccIoU')})
                comparisons[control] = dict(
                    per_seed=differences,
                    mean={metric: np.mean([d[metric] for d in differences], axis=0).tolist()
                          for metric in ('mIoU', 'OccIoU')},
                    scene_bootstrap_95=bootstrap_pair(
                        per_arm[arm], per_arm[control], scenes, indices,
                        args.bootstrap_repeats, 20260928))
            result['comparisons'][arm] = comparisons
        result['teacher_native'] = {metric: value.tolist()
                                    for metric, value in combined(native, indices).items()}
        summary['subsets'][name] = result
    destination = root / 'summary.json'
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf-8')
    full = summary['subsets']['full']
    joint = full['comparisons']['teacher_joint']
    task = joint['task_only']
    gain = float(np.mean(task['mean']['mIoU']))
    ci = task['scene_bootstrap_95']['mIoU']['mean']
    other = [float(np.mean(joint[control]['mean']['mIoU']))
             for control in ('teacher_feature', 'teacher_position',
                             'teacher_joint_shuffled_feature')]
    all_seed_positive = all(np.mean(item['mIoU']) > 0 for item in task['per_seed'])
    confirmation = summary['subsets'].get('confirmation')
    passes = bool(gain >= .2 and ci[0] > 0 and all_seed_positive and all(v > 0 for v in other)
                  and confirmation is not None and
                  np.mean(confirmation['comparisons']['teacher_joint']['task_only']['mean']['mIoU']) > 0)

    lines = ['# Teacher query 替换实验：结果', '',
             f'状态：{len(seeds)} 个种子、五组消融均完成。', '',
             '实验为什么这样设计以及怎样判断结果，见[设计文档](teacher_query_replacement_ablation_design.md)。', '',
             '## 训练条件与空间覆盖', '',
             f"训练 {report['args']['train_samples']} 个样本、{report['unique_train_scenes']} 个场景；验证 {full['samples']} 个样本、{full['scenes']} 个场景。",
             f"每组 {report['args']['updates']} 次更新；只在持续 query 中做位置 Hungarian 匹配，最大中心距离 {report['args']['max_match_distance_m']:g} 米。",
             'future-born queries 均保留，且其输入特征与位置没有被 Teacher 覆盖。',
             (f"Teacher 提取阶段 PyTorch CUDA 分配峰值 {report['cache_peak_cuda_memory_bytes']/2**30:.2f} GiB；" if 'cache_peak_cuda_memory_bytes' in report else 'Teacher 提取阶段显存峰值未记录；') +
             f"后续训练／评估峰值 {report['peak_cuda_memory_bytes']/2**30:.2f} GiB。",
             f"本次累计运行约 {report['elapsed_seconds']/3600:.2f} 小时。", '',
             '| 时距 | 平均匹配数／720 | 平均匹配中心距离 | 平均局部点距离 |',
             '|---|---:|---:|---:|']
    for h in HORIZONS:
        m = report['teacher_native']['match_diagnostics'][h]
        lines.append(f"| {h} | {m['matched']:.1f}／720（{100*m['matched']/720:.1f}%） | {m['distance']:.2f} 米 | {m['local_point_distance']:.2f} 米 |")
    lines += ['', '这里的匹配数只统计 720 个持续 query；未匹配持续 query 和所有 future-born query 仍由 Student 自身预测。', '']
    h4 = json.loads(Path(report['args']['previous_report']).read_text())
    same_h4 = (report['val_manifest']['indices'] == h4['validation']['indices'] and
               all(seed in h4['runs'] and 'future_image' in h4['runs'][seed] for seed in seeds))
    for subset_name, subset in summary['subsets'].items():
        title = '全部验证样本' if subset_name == 'full' else '新增确认场景'
        lines += [f'## {title}', '', f"{subset['samples']} 个样本、{subset['scenes']} 个场景；下表是各种子的平均值。单位：百分数。", '',
                  '| 组别 | 1 秒 mIoU | 2 秒 mIoU | 3 秒 mIoU | 平均 mIoU | 平均 OccIoU |',
                  '|---|---:|---:|---:|---:|---:|']
        baseline = report['baseline' if subset_name == 'full' else 'baseline_confirmation']
        baseline_values = [baseline[h]['mIoU'] for h in HORIZONS]
        lines.append(f"| 原始冻结 SparseWorld | {baseline_values[0]:.3f} | {baseline_values[1]:.3f} | {baseline_values[2]:.3f} | {np.mean(baseline_values):.3f} | {np.mean([baseline[h]['OccIoU'] for h in HORIZONS]):.3f} |")
        for arm in ARMS:
            value = subset['scores'][arm]
            m = value['mIoU']
            lines.append(f"| {NAMES[arm]} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {np.mean(m):.3f} | {np.mean(value['OccIoU']):.3f} |")
        value = subset['teacher_native']; m = value['mIoU']
        lines.append(f"| Teacher 原生预测（特权参考） | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {np.mean(m):.3f} | {np.mean(value['OccIoU']):.3f} |")
        if same_h4:
            key = 'metrics' if subset_name == 'full' else 'confirmation'
            values = [np.mean([h4['runs'][seed]['future_image'][key][h]['mIoU']
                               for seed in seeds]) for h in HORIZONS]
            lines.append(f"| 旧 H4：直接读取未来图像 | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {np.mean(values):.3f} | — |")
        lines += ['', '下面的差值均为“Teacher 特征＋位置”减去对照组；正数表示完整替换更好。', '',
                  '| 对照组 | 1 秒 mIoU 差 | 2 秒 mIoU 差 | 3 秒 mIoU 差 | 平均差 | 场景 bootstrap 95% 区间 | 平均 OccIoU 差 |',
                  '|---|---:|---:|---:|---:|---|---:|']
        joint = subset['comparisons']['teacher_joint']
        for control in ('task_only','teacher_feature','teacher_position',
                        'teacher_joint_shuffled_feature'):
            compare = joint[control]
            d = compare['mean']['mIoU']
            lo, hi = compare['scene_bootstrap_95']['mIoU']['mean']
            lines.append(f"| {NAMES[control]} | {f(d[0])} | {f(d[1])} | {f(d[2])} | {f(np.mean(d))} | [{f(lo)}, {f(hi)}] | {f(np.mean(compare['mean']['OccIoU']))} |")
        per_seed = [float(np.mean(item['mIoU'])) for item in joint['task_only']['per_seed']]
        lines += ['', '完整替换相对 task-only 的各种子平均 mIoU 差：' + '、'.join(f(v) for v in per_seed) + ' 点。', '']
    lines += ['## 如何解释', '',
              ('本轮达到预设的初步正对照门槛：完整替换平均至少提高 0.2 点，超过单独替换和打乱特征组，三个种子及确认场景方向一致。' if passes else
               '本轮没有同时满足“完整替换明显超过 task-only、单独替换与打乱对照，并在确认场景保持正向”的全部判定条件。'),
              'Teacher 原生预测使用未来图像和其自己的解码结果，只能作为特权参考；它高于 Student 并不保证替换 query 后也能高于 Student。',
              '若 Teacher 原生预测较好但替换后下降，应优先检查表示／解码器兼容、匹配覆盖和有限训练预算，不能断言 Teacher query 没有信息。',
              '本实验不训练从历史预测 Teacher query 的生成器，也不验证部署时是否可获得 Teacher query。',
              '多个比较共享同一验证集；区间是按验证场景配对重采样，训练随机性由三个种子分别列出。', '',
              '## 原始证据', '', f'报告：`{path}`；汇总：`{destination}`。',
              '输出目录另有每组检查点和逐样本混淆矩阵，可复核每个种子、时距和确认场景。', '']
    Path(args.markdown).write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(preliminary_gate=passes, mean_gain_vs_task_only=gain,
                          scene_interval=ci, summary=str(destination),
                          markdown=args.markdown), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='out/future_alignment_hypotheses/teacher_query_replacement_512x256/report.json')
    parser.add_argument('--markdown', default='analysis/hypotheses/h2_spatial_correspondence/teacher_query_replacement_results.md')
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    main(parser.parse_args())
