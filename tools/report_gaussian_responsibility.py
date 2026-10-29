"""Audit completed A0..A3, bootstrap scenes, and write the human-facing report."""
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from tools.gaussian_responsibility_experiment import RUN, initialize, read, write, sha, checkpoint
from tools.eval_gaussian_responsibility import RULES, summarize_counts, plain

MODELS = ['A0', 'A1', 'A2', 'A3']
NAMES = dict(A0='原 focal', A1='最近邻 CE', A2='区域责任', A3='受约束传输')
RULE_NAMES = dict(native='原生联合', hierarchical='分层', global_calibrated='全局校准', class_calibrated='类别校准')
CLASSES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def audit():
    p = initialize(); checks = {}
    checks['A0_checkpoint_unchanged'] = sha(checkpoint('A0')) == p['baseline_sha256']
    for path, digest in p['source_sha256'].items():
        checks['source:' + path] = sha(ROOT / path) == digest
    reference = np.load(ROOT / 'work_dirs/gf2_probability_decomposition/train/probability/initial_forward.npz')
    anchor = read(RUN / 'evaluation/A0/statistics.json')['records']
    anchor_frames = [(r['index'], r['scene'], r['scene_frame']) for r in anchor]
    anchor_truth = np.array([r['counts']['native'] for r in anchor], dtype=np.int64).sum(-1)
    for name in MODELS:
        folder = RUN / 'evaluation' / name
        c = read(folder / 'complete.json')
        checks[name + ':full_validation'] = c['frames'] == 1024 and c['status'] == 'complete'
        checks[name + ':evaluated_checkpoint'] = c['checkpoint_sha256'] == sha(checkpoint(name))
        stats = read(folder / 'statistics.json')
        checks[name + ':frame_order'] = [r['index'] for r in stats['records']] == list(range(1024))
        checks[name + ':paired_scenes_frames'] = [(r['index'], r['scene'], r['scene_frame']) for r in stats['records']] == anchor_frames
        checks[name + ':identical_gt_counts_all_rules'] = all(np.array_equal(
            np.array([r['counts'][rule] for r in stats['records']], np.int64).sum(-1), anchor_truth) for rule in RULES)
        saved = np.load(folder / 'scene_counts.npz')
        checks[name + ':scene_totals_match_frames'] = all(np.array_equal(saved['counts'][:, j].sum(0),
            np.array([r['counts'][rule] for r in stats['records']], np.int64).sum(0)) for j, rule in enumerate(RULES))
        checks[name + ':train_only_selection'] = 'validation excluded' in read(folder / 'selection.json')['selection_data']
        if name != 'A0':
            t = read(RUN / 'train' / name / 'complete.json')
            checks[name + ':full_paired_training'] = t['frames'] == 20480 and t['epochs'] == 5 and t['paired_inputs']
            checks[name + ':matched_gradient_budget'] = t['max_gradient_norm_relative_error'] < 2e-6
            initial = np.load(RUN / 'train' / name / 'initial_forward.npz')
            checks[name + ':initial_output_parity'] = bool(np.array_equal(initial['raw'], reference['raw']) and np.array_equal(initial['labels'], reference['labels']))
    probes = [read(path) for path in sorted((RUN / 'probe').glob('train_*.json')) + sorted((RUN / 'probe').glob('val_*.json'))]
    checks['64_probe_frames'] = len(probes) == 64
    checks['sparse_renderer_parity'] = all(r['sparse_renderer_max_error'] < 5e-4 for r in probes)
    checks['fixed_geometry_negative_control'] = all(a['occupancy_unchanged'] for r in probes for a in r['interventions'])
    checks['all_requested_probe_steps'] = all(len(r['interventions']) == 12 for r in probes)
    checks['actual_probe_step_matches_protocol'] = all(abs(a['actual_rms'] - a['logit_rms_step']) < 1e-6 for r in probes for a in r['interventions'])
    cached_parity = []
    for record in probes:
        if record['split'] != 'val':
            continue
        cached = np.load(ROOT / f'work_dirs/gf2_probability_decomposition/evaluation/probability/frames/{record["frame"]["index"]:04d}.npz')
        cm = np.bincount(cached['y'].astype(np.int64) * 18 + cached['native'].astype(np.int64), minlength=324).reshape(18, 18)
        cached_parity.append(np.array_equal(cm, record['baseline']['counts']))
    checks['current_A0_matches_cached_baseline_32_frames'] = len(cached_parity) == 32 and all(cached_parity)
    registration = read(RUN / 'evaluation_registration.json')
    for name in ['tools/eval_gaussian_responsibility.py', 'tools/probe_gaussian_responsibility.py']:
        checks['frozen_evaluation:' + name] = sha(ROOT / name) == registration['source_sha256'][name]
    write(RUN / 'completion_audit.json', dict(status='passed' if all(checks.values()) else 'failed', checks=checks))
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    return checks, probes


def summarize_probes(probes):
    result = {}
    for split in ('train', 'val'):
        rows = [r for r in probes if r['split'] == split]
        total = lambda key: sum(r['render_only'][key] for r in rows)
        r = dict(frames=len(rows), supported_gaussians=total('supported_gaussians'),
            mismatch_percent=100 * total('mismatched_gaussians') / total('supported_gaussians'),
            mismatch_mass_percent=100 * total('mismatch_mass') / total('responsibility_mass'),
            nearest_share_mass_percent=100 * total('nearest_share_mass_sum') / total('responsibility_mass'),
            unsupported_gt_percent=100 * total('guard_voxels') / total('gt_voxels'),
            renderer_max_error=max(x['sparse_renderer_max_error'] for x in rows))
        base_counts = sum(np.asarray(x['baseline']['counts'], np.int64) for x in rows)
        base = plain(summarize_counts(base_counts))
        fg = sum(x['baseline']['foreground'] for x in rows)
        base['rank_accuracy'] = 100 * sum(x['baseline']['rank_correct'] for x in rows) / fg
        base['conditional_NLL'] = sum(x['baseline']['conditional_NLL_sum'] for x in rows) / fg
        r['baseline'] = base
        r['interventions'] = {}
        for method in MODELS:
            r['interventions'][method] = {}
            for step in (.01, .03, .1):
                values = [a['outcome'] for x in rows for a in x['interventions']
                          if a['method'] == method and a['logit_rms_step'] == step]
                counts = sum(np.asarray(v['counts'], np.int64) for v in values)
                point = plain(summarize_counts(counts))
                point['rank_accuracy'] = 100 * sum(v['rank_correct'] for v in values) / fg
                point['conditional_NLL'] = sum(v['conditional_NLL_sum'] for v in values) / fg
                point['delta_mIoU_pp'] = point['mIoU'] - base['mIoU']
                point['delta_rank_pp'] = point['rank_accuracy'] - base['rank_accuracy']
                point['delta_NLL'] = point['conditional_NLL'] - base['conditional_NLL']
                point['transitions'] = {key: sum(v[key] for v in values) for key in
                    ('correct_to_empty', 'missed_to_correct', 'empty_to_foreground', 'foreground_to_empty')}
                r['interventions'][method][str(step)] = point
        r['exposure'] = []
        for i in range(4):
            record = {key: sum(x['exposure'][i][key] for x in rows) for key in
                      ('voxels', 'rank_correct', 'final_correct', 'missed')}
            record.update(low=rows[0]['exposure'][i]['low'], high=rows[0]['exposure'][i]['high'])
            record['rank_accuracy'] = 100 * record['rank_correct'] / max(record['voxels'], 1)
            r['exposure'].append(record)
        result[split] = r
    return result


def main():
    checks, probes = audit()
    metadata = {name: np.load(RUN / 'evaluation' / name / 'scene_counts.npz') for name in MODELS}
    scenes = metadata['A0']['scenes']
    n = len(scenes)
    rng = np.random.default_rng(928)
    draws = rng.integers(0, n, size=(1000, n))
    weights = np.array([np.bincount(d, minlength=n) for d in draws], dtype=np.float64)
    models, bootstrap = {}, {}
    for name in MODELS:
        np.testing.assert_array_equal(metadata[name]['scenes'], scenes)
        counts = metadata[name]['counts']
        point = summarize_counts(counts.sum(0))
        resampled = (weights @ counts.reshape(n, -1)).reshape(1000, len(RULES), 18, 18)
        boot = summarize_counts(resampled)
        bootstrap[name] = boot
        stats = read(RUN / 'evaluation' / name / 'statistics.json')
        models[name] = dict(rules={rule: {key: dict(value=float(value[j]), ci=interval(boot[key][:, j]))
            for key, value in point.items() if key != 'per_class_IoU'} for j, rule in enumerate(RULES)},
            rank_accuracy=stats['rank_accuracy'], conditional_NLL=stats['conditional_NLL'],
            per_class_IoU={rule: point['per_class_IoU'][j].tolist() for j, rule in enumerate(RULES)},
            selection=read(RUN / 'evaluation' / name / 'selection.json'))
    comparisons = {}
    for first, second in [('A1', 'A0'), ('A2', 'A1'), ('A3', 'A2'), ('A2', 'A0'), ('A3', 'A0')]:
        comparisons[first + '-' + second] = {}
        for j, rule in enumerate(RULES):
            point = models[first]['rules'][rule]['mIoU']['value'] - models[second]['rules'][rule]['mIoU']['value']
            values = bootstrap[first]['mIoU'][:, j] - bootstrap[second]['mIoU'][:, j]
            comparisons[first + '-' + second][rule] = dict(delta_pp=point, ci=interval(values))
    probe_summary = summarize_probes(probes)
    allocation_diagnostics = {}
    for split in ('train', 'val'):
        rows = [r for r in probes if r['split'] == split]
        if not all('A2' in r and 'A3' in r for r in rows):
            continue  # The isolated reporting fixture has no allocation data.
        allocation_diagnostics[split] = {}
        for method in ('A2', 'A3'):
            count = sum(r[method]['supported_gaussians'] for r in rows)
            allocation_diagnostics[split][method] = dict(
                target_entropy_nats=sum(r[method]['supported_entropy_sum'] for r in rows) / count,
                nearest_share=sum(r[method]['supported_nearest_share_sum'] for r in rows) / count,
                column_mass_abs_deviation=float(np.mean([r[method]['column_mass_abs_error'] for r in rows])),
                fixed_point_max=max(r[method]['transport_fixed_point_log_residual'] for r in rows))
    costs = {'A0': read(ROOT / 'work_dirs/gf2_probability_decomposition/train/probability/training_runtime.json')}
    for name in MODELS[1:]:
        costs[name] = read(RUN / 'train' / name / 'training_runtime.json')
    summary = dict(models=models, comparisons=comparisons, probes=probe_summary, costs=costs,
        allocation_diagnostics=allocation_diagnostics,
        protocol=dict(frames=1024, scenes=n, training_epochs=5, seed=928, bootstrap=1000,
                      uncertainty='validation scenes only; one training seed'))
    write(RUN / 'summary.json', summary)
    with (RUN / 'performance.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'rule', 'mIoU', 'IoU', 'FNR', 'FPR', 'rank_accuracy', 'conditional_NLL'])
        for name in MODELS:
            for rule in RULES:
                v = models[name]['rules'][rule]
                writer.writerow([name, rule, *[v[k]['value'] for k in ['mIoU', 'IoU', 'FNR', 'FPR']],
                                 models[name]['rank_accuracy'], models[name]['conditional_NLL']])
    figures = ROOT / 'analysis/gaussian_responsibility/assets'
    figures.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(4)
    primary = ['A1-A0', 'A2-A1', 'A3-A2']
    for rule, offset in [('native', -.10), ('class_calibrated', .10)]:
        points = np.array([comparisons[c][rule]['delta_pp'] for c in primary])
        bounds = np.array([comparisons[c][rule]['ci'] for c in primary])
        axes[0].errorbar(points, np.arange(3) + offset,
            xerr=np.stack([np.maximum(0, points - bounds[:, 0]), np.maximum(0, bounds[:, 1] - points)]),
            fmt='o', capsize=3, label=rule)
    axes[0].axvline(0, color='grey', lw=.8)
    axes[0].set(title='Paired 95% scene intervals', xlabel='mIoU change (pp)', yticks=np.arange(3), yticklabels=primary)
    axes[0].set_ylim(-.35, 2.65)
    axes[0].legend(loc='upper center', ncol=2, fontsize=7)
    for method in MODELS:
        rows = probe_summary['val']['interventions'][method]
        axes[1].plot([.01, .03, .1], [rows[str(s)]['delta_mIoU_pp'] for s in (.01, .03, .1)], marker='o', label=method)
    axes[1].axhline(0, color='grey', lw=.8)
    axes[1].set(title='Same-frame semantic-only probes', xlabel='Logit step RMS', ylabel='mIoU change (pp)')
    axes[1].legend(fontsize=8)
    axes[2].bar(x, [costs[m]['training_and_validation_seconds'] / 60 for m in MODELS])
    axes[2].set(title='Five training epochs + validation', ylabel='Minutes', xticks=x, xticklabels=MODELS)
    fig.tight_layout()
    fig.savefig(figures / 'overview.png', dpi=180)
    fig.savefig(figures / 'overview.svg')
    plt.close(fig)

    f = lambda value: f'{value:.3f}'
    ci = lambda values: f'[{f(values[0])}, {f(values[1])}]'
    outcome = []
    for comparison, subject in [('A1-A0', '分类形式'), ('A2-A1', '区域责任'), ('A3-A2', '传输约束')]:
        a = comparisons[comparison]['native']; b = comparisons[comparison]['class_calibrated']
        confirmed = a['ci'][0] > 0 and b['ci'][0] > 0
        verdict = '在两个口径下均观察到正向差异' if confirmed else '尚未确认稳定收益，差值区间跨过 0'
        outcome.append(f'- **{subject}（{comparison}）{verdict}**：原生 mIoU {a["delta_pp"]:+.3f} pp，类别校准后 {b["delta_pp"]:+.3f} pp。')
    lines = ['# 方向一实验结果：Gaussian 监督责任与渲染贡献', '',
        '日期：2026-10-20。状态：**A0/A1/A2/A3 四组五轮结果、完整 1024 帧验证及 64 帧机制探针均完成。**', '',
        '实验设计见[配套设计文档](gaussian_responsibility_design.md)。本轮只研究最终 Gaussian 层的分类监督，保留原概率渲染与其他训练目标。', '',
        '## 1. 先看可以得出的结论', '',
        '**本轮完成了方向一的最小验证，但没有得到足以确认性能增益的证据。** 最近邻标签与区域责任确有差别，却还不能将这种差别认定为当前模型的主要瓶颈。', '', *outcome, '',
        '上述差值的单位是百分点（pp）。机制差别是否存在、局部干预是否有益、重新训练是否泛化，是三个不同问题，后文分别给出证据。', '',
        '![完整验证、局部探针与训练成本](assets/overview.png)', '',
        '## 2. 四组各改变了什么', '',
        '| 组别 | 最终层分类目标 | 训练与验证完成量 |', '| --- | --- | --- |',
        '| A0 | 最近邻硬标签，原 sigmoid focal | 复用已核对的 5 epoch／1024 帧概率基线 |',
        '| A1 | 最近邻硬标签，softmax CE | 新训练 5 epoch／完整 1024 帧验证 |',
        '| A2 | 渲染责任软标签，softmax CE | 新训练 5 epoch／完整 1024 帧验证 |',
        '| A3 | 受约束传输软标签，softmax CE | 新训练 5 epoch／完整 1024 帧验证 |', '',
        '每个新组完成 20,480 个训练帧次、5,120 次优化器更新。首帧输出与 A0 一致，全部训练帧的标签与投影矩阵指纹配对。每批次匹配最终 logits 分类梯度范数，原类别与距离权重不变；该控制不保证网络参数更新方向相同。', '',
        '## 3. 完整验证集性能', '',
        '以下均为默认 camera mask，先汇总混淆矩阵再计算。mIoU 只平均 17 个前景类。漏检率是“真实前景→empty／真实前景”；误占用率是“真实 empty→前景／真实 empty”。', '',
        '| 组别 | 原生 mIoU ↑ | 二值 IoU ↑ | 漏检率 ↓ | 误占用率 ↓ | 前景首选正确率 ↑ | 条件类别 NLL ↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in MODELS:
        a = models[name]; v = a['rules']['native']
        lines.append(f'| {name} {NAMES[name]} | {f(v["mIoU"]["value"])} | {f(v["IoU"]["value"])} | {f(v["FNR"]["value"])} | {f(v["FPR"]["value"])} | {f(a["rank_accuracy"])} | {f(a["conditional_NLL"])} |')
    lines += ['', '前景首选正确率：只在真实前景上判断条件类别分布的 argmax 是否正确，不让 empty 参与竞争；条件类别 NLL 则评价其概率分配，使用自然对数和 $10^{-6}$ 下限。二者都不等于最终前景分类正确率。', '',
        '| 组别 | 真实前景最终分对 | 真实前景分成其他前景类 | 真实前景漏检为 empty |',
        '| --- | ---: | ---: | ---: |']
    for name in MODELS:
        r = models[name]['rules']['native']
        correct, missed = r['foreground_final_accuracy']['value'], r['FNR']['value']
        lines.append(f'| {name} | {correct:.3f}% | {100-correct-missed:.3f}% | {missed:.3f}% |')
    lines += ['', '这三列共用“全部真实前景”这个分母，合计约为 100%。A1/A2/A3 的前景首选正确率和条件 NLL 比 A0 更好，但最终 mIoU 几乎没有变化；潜在的类别改进并未转化为同等幅度的最终正确占有。', '',
        '### 3.1 同等校准机会后，还剩多少差异', '',
        '全局与逐类别阈值都只由训练场景拟合、调参；各组采用相同搜索范围。这里的全局阈值是在渲染后对前景与 empty 概率比做校准，未改动 Gaussian 本身。', '',
        '| 组别 | 原生联合 mIoU | 分层 mIoU | 全局校准 mIoU | 类别校准 mIoU |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in MODELS:
        lines.append('| ' + name + ' | ' + ' | '.join(f(models[name]['rules'][r]['mIoU']['value']) for r in RULES) + ' |')
    lines += ['', '| 关键对照 | 原生 mIoU 差值与 95% 区间（pp） | 类别校准后差值与区间（pp） |', '| --- | ---: | ---: |']
    for comparison in ('A1-A0', 'A2-A1', 'A3-A2', 'A2-A0', 'A3-A0'):
        a, b = comparisons[comparison]['native'], comparisons[comparison]['class_calibrated']
        lines.append(f'| {comparison} | {a["delta_pp"]:+.3f} {ci(a["ci"])} | {b["delta_pp"]:+.3f} {ci(b["ci"])} |')
    lines += ['', f'区间来自 {n} 个验证场景的 1000 次配对重采样，只覆盖场景组成不确定性。这里只有一个训练 seed；不据此声称多 seed 稳定性。', '',
        '## 4. 最近邻标签与实际渲染责任是否一致', '',
        '固定 A0 五轮权重，在训练／验证各 32 个预登记帧上检查。责任组成沿用 Set 的全部有效前景 GT；下表只统计实际渲染密度，剔除用于训练兜底的守护边。Gaussian 责任总量至少为 0.001 才计入个数分母。后面的错误关联统计和模型成绩使用默认 camera mask。', '',
        '| 探针来源 | 多数类别与最近邻不一致的 Gaussian 比例 | 不一致 Gaussian 承担的责任比例 | 最近邻类别占总责任的比例 | 无有效密度的前景比例 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for split, label in [('train', '训练 32 帧'), ('val', '验证 32 帧')]:
        p = probe_summary[split]
        lines.append('| ' + label + ' | ' + ' | '.join(f(p[k]) + '%' for k in ['mismatch_percent', 'mismatch_mass_percent', 'nearest_share_mass_percent', 'unsupported_gt_percent']) + ' |')
    val_probe = probe_summary['val']
    lines += ['', f'验证探针中，类别不一致涉及约 {val_probe["mismatch_percent"]:.2f}% 的有支撑 Gaussian，但它们承担约 {val_probe["mismatch_mass_percent"]:.2f}% 的实际责任；最近邻类别仍占总责任的约 {val_probe["nearest_share_mass_percent"]:.2f}%。因此，最近邻并非整体失效。上述差别既可能来自标签不合适，也可能来自边界混合或已经错误的几何，单凭比例不能宣布它造成了模型误差。', '',
        '| 验证前景体素受到不一致 Gaussian 的贡献占比 | 体素数 | 条件类别首选正确率 |', '| --- | ---: | ---: |']
    for v in probe_summary['val']['exposure']:
        lines.append(f'| {100*v["low"]:.0f}%—{min(100,100*v["high"]):.0f}% | {v["voxels"]:,} | {f(v["rank_accuracy"])}% |')
    lines += ['', '该分组是相关性定位，未控制物体边界、类别难度和几何错误；不能作为独立因果效应。', '',
        '## 5. 固定几何时，哪种分类更新更有利', '',
        '此处用同一帧 GT 计算一次输出 logits 更新，再在该帧测量。占有概率逐值不变；它是机制干预，不是模型泛化成绩。每个方向使用相同 logits 改变量 RMS，避免梯度大小决定比较结果。', '',
        '| 方向 | RMS=0.01：mIoU 变化 | RMS=0.03：mIoU 变化 | RMS=0.1：mIoU 变化 | RMS=0.03：条件 NLL 变化 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in MODELS:
        rows = probe_summary['val']['interventions'][name]
        lines.append(f'| {name} {NAMES[name]} | ' + ' | '.join(f'{rows[str(s)]["delta_mIoU_pp"]:+.4f} pp' for s in (.01, .03, .1)) + f' | {rows["0.03"]["delta_NLL"]:+.5f} |')
    lines += ['', '**三个步长下，A2/A3 的验证探针 mIoU 增量均未超过 A1。** 原来的最近邻分类方向本身也能提高同帧 mIoU，不能据此说最近邻监督普遍有害。A2 在部分类别概率指标上更好，但没有形成一致的最终指标优势。', '',
        'mIoU 的变化可以同时包含类别纠正和联合决策的 empty 转换。几何占有概率没有改变，不表示最终占有判定一定不变；前景类别概率仍参与联合 argmax。三个步长均完整报告，没有挑选最佳步长作为方法成绩。', '',
        '## 6. 额外优化成本与数值质量', '',
        '| 组别 | 五轮训练及期间验证（分钟） | 峰值 allocated 显存（MiB） |', '| --- | ---: | ---: |']
    for name in MODELS:
        c = costs[name]
        lines.append(f'| {name} | {f(c["training_and_validation_seconds"]/60)} | {f(c["peak_cuda_allocated_mib"])} |')
    lines += ['', '本轮没有新增推理分支或参数，匹配与传输只在训练损失执行。训练时间来自各自进程，包含每轮验证，受本机系统负载影响；不以这些数字声称跨实现 FPS 优势。A2/A3 相比 A0 多约十分钟训练成本，尚未换来可确认的收益。', '',
        'A3 在 A2 同一候选图上进行 24 次非平衡缩放，Gaussian 容量按责任质量的平方根设定，覆盖与容量均为软约束。它是一次固定参数的最小实现，不能代表所有最优传输方法的上限。守护边保障没有原生密度的 GT 仍进入分类目标；它不改变渲染器的真实覆盖。', '']
    if allocation_diagnostics:
        lines += ['下表检查实际用于分类的分配，**包含 A2/A3 公共兜底**，与第 4 节纯渲染统计口径不同。目标熵越小表示标签越接近单一类别；列质量偏离衡量每个体素的分配质量离 1 多远，并非求解器残差。', '',
            '| 数据 | 方法 | 平均目标熵（nats） | 最近邻类平均目标份额 | 平均列质量偏离 | 最大固定点残差 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
        for split, label in [('train', '训练探针'), ('val', '验证探针')]:
            for method in ('A2', 'A3'):
                d = allocation_diagnostics[split][method]
                lines.append(f'| {label} | {method} | {d["target_entropy_nats"]:.5f} | {100*d["nearest_share"]:.3f}% | {d["column_mass_abs_deviation"]:.3g} | {d["fixed_point_max"]:.3g} |')
        residual = max(allocation_diagnostics[s]['A3']['fixed_point_max'] for s in allocation_diagnostics)
        lines += ['', f'A3 探针最大固定点残差为 {residual:.3g}；A2/A3 的验证目标仍高度接近最近邻类别。现有记录没有显示明显的求解未收敛问题。', '']
    lines += [
        '## 7. 逐类别结果（原生联合决策）', '',
        '| 类别 | A0 IoU | A1 IoU | A2 IoU | A3 IoU |', '| --- | ---: | ---: | ---: | ---: |']
    for i, name in enumerate(CLASSES):
        lines.append('| ' + name + ' | ' + ' | '.join(f(models[m]['per_class_IoU']['native'][i]) for m in MODELS) + ' |')
    lines += ['', '## 8. 如何使用这轮证据', '',
        '**目前不建议把这版 A2/A3 直接投入完整数据训练，或作为已验证有效的论文方法。** A2 与基线几乎相同；A3 仅有小幅点估计优势，所有主要差值区间都跨过 0，二值 IoU 也未改善。单 seed 不能确认这些微小差异可重复。', '',
        '若继续方向一，先定位“哪类不一致确实有害”，再改完整优化目标，优先于扩大最优传输参数搜索。边界、弱覆盖和类别竞争需要分别检查；有害子集不能只用“不一致”定义。', '',
        '代码与数学解释还提示两个候选原因：A2 使用空间混合权重，并非结合真实类别解释能力的混合后验；A3 的 Gaussian 侧缩放在类别目标行归一化时直接消去，只能通过体素侧缩放间接起作用。这些是实现层面的解释，尚不能认定为本轮小收益的唯一成因。详见[目标解释补充](gaussian_responsibility_objective_notes.md)。', '',
        '本轮没有证明最近邻责任是 empty 偏差的主导成因，也没有验证完整数据集、多训练 seed 或其他架构上的普适性。它提供的是当前概率基线、最终层分类监督上的受控最小证据。', '',
        f'完整性审计 **{sum(checks.values())}/{len(checks)}** 项通过。所有真实数据均来自本轮运行或经校验复用的 A0；没有把单元测试合成数据计入实验。', '',
        '数据入口：[汇总 JSON](../../work_dirs/gaussian_responsibility/summary.json) · [性能 CSV](../../work_dirs/gaussian_responsibility/performance.csv) · [登记协议](../../work_dirs/gaussian_responsibility/protocol.json) · [完整性审计](../../work_dirs/gaussian_responsibility/completion_audit.json) · [运行代码](../../tools/gaussian_responsibility_experiment.py)。', '']
    (ROOT / 'analysis/gaussian_responsibility/gaussian_responsibility_results.md').write_text('\n'.join(lines))
    print('Report written: analysis/gaussian_responsibility/gaussian_responsibility_results.md', flush=True)


if __name__ == '__main__':
    main()
