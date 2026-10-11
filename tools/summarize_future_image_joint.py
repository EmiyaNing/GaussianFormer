#!/usr/bin/env python3
"""Summarize paired joint-image experiments and scene-cluster uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np


HORIZONS = ('1s', '2s', '3s')
ARMS = ('query_only', 'current_image', 'future_image', 'future_shuffled')


def score(hist, intersection, union):
    diagonal = np.diagonal(hist, axis1=-2, axis2=-1)
    denominator = hist.sum(-2) + hist.sum(-1) - diagonal
    iou = np.divide(diagonal, denominator, out=np.full(diagonal.shape, np.nan),
                    where=denominator > 0)
    return dict(mIoU=np.nanmean(iou[..., :17], axis=-1) * 100.,
                OccIoU=100. * intersection / np.maximum(union, 1))


def read_arrays(root, entry):
    with np.load(root / entry['confusion_file']) as data:
        return {key: data[key].astype(np.float64) for key in ('hist', 'intersection', 'union')}


def scene_arrays(arrays, scenes, indices):
    unique = sorted({scenes[i] for i in indices})
    groups = [[i for i in indices if scenes[i] == scene] for scene in unique]
    return {key: np.stack([value[group].sum(0) for group in groups])
            for key, value in arrays.items()}


def bootstrap_pair(first, second, scenes, indices, repeats, seed):
    """Resample scenes identically in both arms and all training seeds.

    Recalculate mIoU from counts, never average per-sample mIoUs. Average
    scores across training seeds after aggregation; training seeds themselves
    are not resampled, so this interval is conditional on those seeds.
    """
    left = [scene_arrays(a, scenes, indices) for a in first]
    right = [scene_arrays(a, scenes, indices) for a in second]
    count = len(left[0]['hist'])
    rng = np.random.RandomState(seed)
    weights = rng.multinomial(count, np.ones(count) / count, size=repeats)
    deltas = []
    for a, b in zip(left, right):
        scores = []
        for arrays in (a, b):
            aggregated = {key: np.tensordot(weights, value, axes=(1, 0))
                          for key, value in arrays.items()}
            scores.append(score(**aggregated))
        deltas.append({key: scores[0][key] - scores[1][key] for key in scores[0]})
    result = {}
    for metric in ('mIoU', 'OccIoU'):
        values = np.stack([delta[metric] for delta in deltas]).mean(0)
        result[metric] = {
            h: np.percentile(values[:, j], [2.5, 97.5]).tolist()
            for j, h in enumerate(HORIZONS)}
        result[metric]['mean'] = np.percentile(values.mean(-1), [2.5, 97.5]).tolist()
    return result


def main(args):
    path = Path(args.report)
    report = json.loads(path.read_text())
    seeds = [str(seed) for seed in report['args']['training_seeds']]
    for seed in seeds:
        if any(arm not in report['runs'].get(seed, {}) for arm in ARMS):
            raise RuntimeError(f'Experiment incomplete for seed {seed}; do not publish final conclusions')
    root = path.parent
    arrays = {arm: [read_arrays(root, report['runs'][seed][arm]) for seed in seeds] for arm in ARMS}
    for intervention in ('zero', 'shuffle'):
        arrays[intervention] = [read_arrays(root, report['runs'][seed]['future_image'][intervention]) for seed in seeds]
    summary = dict(bootstrap_repeats=args.bootstrap_repeats,
                   interval_scope='paired validation-scene resampling conditional on the three trained seeds',
                   subsets={})
    scenes = report['validation']['scenes']
    for name, indices in [('full', list(range(len(scenes)))),
                          ('confirmation', report['confirmation_indices'])]:
        if not indices:
            continue
        results = {}
        for arm, runs in arrays.items():
            measurements = [score(**{key: value[indices].sum(0) for key, value in run.items()}) for run in runs]
            results[arm] = {metric: np.stack([v[metric] for v in measurements]).mean(0).tolist()
                           for metric in ('mIoU', 'OccIoU')}
        comparisons = {}
        for control in ('query_only', 'current_image', 'future_shuffled', 'zero', 'shuffle'):
            per_seed = []
            for first, second in zip(arrays['future_image'], arrays[control]):
                a = score(**{key: value[indices].sum(0) for key, value in first.items()})
                b = score(**{key: value[indices].sum(0) for key, value in second.items()})
                per_seed.append({metric: float((a[metric]-b[metric]).mean()) for metric in a})
            comparisons[control] = dict(
                per_seed_average_gain=per_seed,
                mean_gain={metric: float(np.mean([v[metric] for v in per_seed])) for metric in ('mIoU', 'OccIoU')},
                scene_bootstrap_95=bootstrap_pair(arrays['future_image'], arrays[control], scenes,
                                                  indices, args.bootstrap_repeats, 20260927))
        summary['subsets'][name] = dict(samples=len(indices), scenes=len({scenes[i] for i in indices}),
                                       metrics=results, comparisons=comparisons)
    destination = root / 'summary.json'
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    full = summary['subsets']['full']
    main_gain = full['comparisons']['query_only']['mean_gain']['mIoU']
    lower = full['comparisons']['query_only']['scene_bootstrap_95']['mIoU']['mean'][0]
    confirmation = summary['subsets'].get('confirmation')
    passes = (main_gain >= .2 and lower > 0 and
              all(v['mIoU'] > 0 for v in full['comparisons']['query_only']['per_seed_average_gain']) and
              all(full['comparisons'][c]['mean_gain']['mIoU'] > 0 for c in ('current_image', 'future_shuffled', 'zero', 'shuffle')) and
              full['comparisons']['query_only']['mean_gain']['OccIoU'] >= -.2 and
              confirmation is not None and confirmation['comparisons']['query_only']['mean_gain']['mIoU'] > 0)
    lines = ['# H4 联合优化：实验结果', '', f'状态：{len(seeds)} 个种子、四组训练及输入干预全部完成。', '',
             '设计和初学者版解释见[实验说明](h4_future_image_joint_analysis.md)。', '',
             '## 数据与资源', '',
             f"训练 {report['args']['train_samples']} 个样本、{report['unique_train_scenes']} 个场景；验证 {full['samples']} 个样本、{full['scenes']} 个场景。",
             f"每组 {report['args']['updates']} 次更新，{len(seeds)} 个配对种子；{report['args']['rounds']} 轮共享权重的特征与位置修正。保留 future-born queries。",
             f"CUDA 分配峰值 {report['peak_cuda_memory_bytes']/2**30:.2f} GiB，累计运行 {report['elapsed_seconds']/3600:.2f} 小时。",
             '该值是 PyTorch 分配峰值，不包括桌面或 CUDA context；实际 nvidia-smi 占用会更高。', '',
             '## 训练链路检查', '',
             '预检查通过：零输出初始化复现原预测；采样偏移、尺度选择和回归头均收到非零任务梯度。',
             f"三个更新后，预测点平均移动 {report['preflight']['mean_point_movement_after_three_updates_m']:.3f} 米。",
             '初始第一步的采样梯度为零是零输出融合初始化的结果，从后续更新开始传递。', '']
    for subset_name, title in [('full', '全部验证样本'), ('confirmation', '新增确认场景')]:
        if subset_name not in summary['subsets']:
            continue
        subset = summary['subsets'][subset_name]
        lines += [f'## {title}', '', f"{subset['samples']} 个样本，{subset['scenes']} 个场景。以下为 {len(seeds)} 个种子的平均值，单位为百分数。", '',
                  '| 组别 | 1 秒 mIoU | 2 秒 mIoU | 3 秒 mIoU | 平均 mIoU | 平均 OccIoU |',
                  '|---|---:|---:|---:|---:|---:|']
        baseline = report['baseline' if subset_name == 'full' else 'baseline_confirmation']
        baseline_miou = [baseline[h]['mIoU'] for h in HORIZONS]
        lines.append(f"| 原始冻结 SparseWorld | {baseline_miou[0]:.3f} | {baseline_miou[1]:.3f} | {baseline_miou[2]:.3f} | {np.mean(baseline_miou):.3f} | {np.mean([baseline[h]['OccIoU'] for h in HORIZONS]):.3f} |")
        names = dict(query_only='不读图像', current_image='当前图像', future_image='真实未来图像', future_shuffled='训练时打乱', zero='未来组验证时关闭图像', shuffle='未来组验证时打乱')
        for arm, name in names.items():
            values = subset['metrics'][arm]
            scores = values['mIoU']
            lines.append(f"| {name} | {scores[0]:.3f} | {scores[1]:.3f} | {scores[2]:.3f} | {np.mean(scores):.3f} | {np.mean(values['OccIoU']):.3f} |")
        lines += ['', '下面均为“正确未来图像成绩减去对照成绩”，正数表示未来图像更好；单位为百分点。', '',
                  '| 对照 | 平均 mIoU 增益 | 95% 场景 bootstrap 区间 | 平均 OccIoU 增益 |', '|---|---:|---|---:|']
        for control, comparison in subset['comparisons'].items():
            lo, hi = comparison['scene_bootstrap_95']['mIoU']['mean']
            lines.append(f"| {names[control]} | {comparison['mean_gain']['mIoU']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {comparison['mean_gain']['OccIoU']:+.3f} |")
        lines += ['', '各种子相对不读图像的平均 mIoU 增益：' + '、'.join(
            f"{value['mIoU']:+.3f}" for value in subset['comparisons']['query_only']['per_seed_average_gain']) + ' 点。', '']
    lines += ['## 结论与初学者版解读', '',
              ('本轮达到项目的初步正对照门槛：未来图像带来至少 0.2 点收益，控制和新增场景检查方向一致。' if passes else
               '本轮尚未同时满足项目的全部正对照条件。请结合增益大小、三个种子、输入干预和新增场景逐项判断，不能把未过门槛等同于未来图像无用。'),
              '模型多了参数后整体变好，并不自动证明未来图像有用；关键是未来图像是否比相同可训练解码器的不读图像组更好。',
              '验证时关闭或打乱输入后的差值用于检验模型是否实际依赖图像；训练时打乱组用于排除单纯的额外输入统计收益。',
              '按场景重采样的区间只衡量验证场景的不确定性，不代表三种训练随机性的完整置信区间。',
              '本轮仍是冻结 backbone、单时刻图像、独立时距修正、四遍子集训练；不是全数据 16 epoch，也不是未来感知性能上限。', '',
              '## 原始证据', '',
              f'主报告：`{path}`；汇总与全部配对区间：`{destination}`。',
              '每组保存最后一次训练的检查点与逐样本混淆矩阵；报告包含采样权重的实际变化、点位移和可见比例。',
              '旧 R1、R2 的采样误冻结和固定几何限制已更正，不能与本轮扩大验证集的绝对分数直接比较。', '']
    Path(args.markdown).write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(passes_preliminary_gate=passes, mean_mIoU_gain=main_gain,
                          summary=str(destination), markdown=args.markdown), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='out/future_alignment_hypotheses/H4_joint_512x256/report.json')
    parser.add_argument('--markdown', default='analysis/hypotheses/h4_future_image_interface/joint_results.md')
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    main(parser.parse_args())
