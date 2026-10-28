"""Explain B using saved evidence and exact same-output CE reweighting."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import runpy
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F

from ..metrics import CLASS_NAMES
from .analyze_a import class_conditionals
from .protocol import ANALYSIS, ARMS, ROOT
from .report import read, verify_arm


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def reweight_ce_record(record, target_free_weight=.5):
    """Reconstruct loss and bias gradient at identical outputs, not retraining."""
    if record['gamma'] != 0 or record['objective'] != 'weighted_ce':
        raise ValueError('exact bucket reweighting requires unmodulated CE')
    buckets = record['buckets']
    denominator = record['denominator']
    weighted_sum = sum(b['label_weight_sum'] for b in buckets.values())
    if not np.isclose(weighted_sum, denominator, rtol=1e-6, atol=1e-6):
        raise ValueError('bucket label weights do not reproduce the denominator')
    multiplier = target_free_weight / record['empty_class_relative_weight']
    free_sum = sum(b['label_weight_sum'] for k, b in buckets.items() if k.endswith('/free'))
    old_denominator = denominator + (multiplier - 1) * free_sum
    ratio = denominator / old_denominator
    changed = {}
    for name, bucket in buckets.items():
        factor = ratio * (multiplier if name.endswith('/free') else 1.)
        changed[name] = dict(loss=bucket['classification_loss_contribution'] * factor,
                             signed_bias_gradient=bucket['signed_bias_gradient'] * factor)
    return dict(index=record['index'], actual_free_weight=record['empty_class_relative_weight'],
                counterfactual_free_weight=target_free_weight,
                actual_denominator=denominator, counterfactual_denominator=old_denominator,
                B_to_old_occupied_ce_coefficient=1 / ratio,
                B_to_old_free_ce_coefficient=1 / (ratio * multiplier),
                actual_loss=record['current_classification_loss'],
                actual_signed_bias_gradient=record['signed_bias_gradient'],
                counterfactual_loss=sum(v['loss'] for v in changed.values()),
                counterfactual_signed_bias_gradient=sum(v['signed_bias_gradient'] for v in changed.values()),
                actual_free_bias_gradient=sum(b['signed_bias_gradient'] for k, b in buckets.items() if k.endswith('/free')),
                actual_occupied_bias_gradient=sum(b['signed_bias_gradient'] for k, b in buckets.items() if k.endswith('/occupied')),
                counterfactual_free_bias_gradient=sum(b['signed_bias_gradient'] for k, b in changed.items() if k.endswith('/free')),
                counterfactual_occupied_bias_gradient=sum(b['signed_bias_gradient'] for k, b in changed.items() if k.endswith('/occupied')),
                counterfactual_buckets=changed)


def verify_counterfactual_math():
    """Check the analytical reconstruction against independent CPU autograd."""
    generator = torch.Generator().manual_seed(17)
    base = torch.randn(2, 18, 53, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 17, (2, 53), generator=generator)
    labels[:, ::3] = 17
    labels[:, ::19] = 255
    bias = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    logits = torch.cat((base[:, :17], base[:, 17:] - bias), dim=1)
    weights = torch.cat((torch.linspace(.85, 1.3, 17, dtype=torch.float64), torch.tensor([.1], dtype=torch.float64)))
    valid = labels != 255
    sample_weights = weights[labels.masked_fill(~valid, 0)] * valid
    denominator = sample_weights.sum()
    weighted_losses = F.cross_entropy(logits, labels, weight=weights, ignore_index=255, reduction='none')
    buckets = {}
    for name, mask in (('full/free', valid & (labels == 17)), ('full/occupied', valid & (labels != 17))):
        loss = 10 * weighted_losses[mask].sum() / denominator
        buckets[name] = dict(label_weight_sum=float(sample_weights[mask].sum()),
                             classification_loss_contribution=float(loss.detach()),
                             signed_bias_gradient=float(torch.autograd.grad(loss, bias, retain_graph=True)[0]))
    record = dict(index=0, gamma=0., objective='weighted_ce', empty_class_relative_weight=.1,
                  denominator=float(denominator), buckets=buckets,
                  current_classification_loss=sum(b['classification_loss_contribution'] for b in buckets.values()),
                  signed_bias_gradient=sum(b['signed_bias_gradient'] for b in buckets.values()))
    counterfactual = reweight_ce_record(record)
    old_weights = weights.clone()
    old_weights[17] = .5
    reference = 10 * F.cross_entropy(logits, labels, weight=old_weights, ignore_index=255)
    reference_gradient = torch.autograd.grad(reference, bias, retain_graph=True)[0]
    if not np.isclose(counterfactual['counterfactual_loss'], float(reference.detach()), rtol=1e-12):
        raise ValueError('counterfactual CE loss reconstruction failed')
    if not np.isclose(counterfactual['counterfactual_signed_bias_gradient'], float(reference_gradient), rtol=1e-12):
        raise ValueError('counterfactual CE bias-gradient reconstruction failed')
    normalized = 10 * F.cross_entropy(logits, labels, weight=old_weights * 18 / old_weights.sum(), ignore_index=255)
    torch.testing.assert_close(reference, normalized, rtol=1e-12, atol=1e-12)
    foreground = valid & (labels != 17)
    conditional = F.cross_entropy(logits.transpose(1, 2)[foreground][:, :17], labels[foreground])
    conditional_gradient = torch.autograd.grad(conditional, logits, retain_graph=True)[0]
    torch.testing.assert_close(conditional_gradient[:, 17], torch.zeros_like(conditional_gradient[:, 17]))
    torch.testing.assert_close(conditional_gradient.sum(1), torch.zeros_like(conditional_gradient.sum(1)), atol=1e-12, rtol=0.)
    log_prob = logits.log_softmax(1).transpose(1, 2)[foreground]
    direct = -log_prob.gather(1, labels[foreground].unsqueeze(1)).squeeze(1)
    occupied_part = -torch.logsumexp(log_prob[:, :17], dim=1)
    semantic_part = F.cross_entropy(logits.transpose(1, 2)[foreground][:, :17], labels[foreground], reduction='none')
    torch.testing.assert_close(direct, occupied_part + semantic_part, rtol=1e-12, atol=1e-12)
    return dict(counterfactual_loss_and_bias_gradient_match_cpu_autograd=True,
                common_weight_rescaling_leaves_weighted_mean_ce_unchanged=True,
                conditional_semantic_loss_has_zero_direct_empty_logit_gradient=True,
                occupied_ce_equals_occupancy_plus_conditional_semantic_ce=True)


def analyze():
    original = Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e')
    roots = dict(Strict=original / 'sparseworld_strict', Camera_Gaussian=original / 'sparseworld_opus_gaussian_bias',
                 Fullspace_CE=Path('out/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e/sparseworld_opus_gaussian_bias'),
                 **{name: ROOT / name for name in ARMS})
    arms = {name: verify_arm(ROOT / name, ANALYSIS / name) for name in ARMS}
    a, b = arms.values()
    if (a['boundary']['manifest_sha256'] != b['boundary']['manifest_sha256'] or
            a['initialization']['initial_model_sha256'] != b['initialization']['initial_model_sha256']):
        raise ValueError('A and B did not use identical subsets and initial model weights')
    config_audit = read(ANALYSIS / 'configuration_audit.json')
    for name, config in ARMS.items():
        if hashlib.sha256(config.read_bytes()).hexdigest() != config_audit['arms'][name]['config_sha256']:
            raise ValueError('validated standalone configuration changed')
    metrics, matrices, binary, conditionals = {}, {}, {}, {}
    for name, work in roots.items():
        metric = read(work / 'epoch_01_metrics.json')
        if metric['epoch'] != 1 or metric['num_samples'] != 1024 or metric['miou_divisor'] != 17 or metric['evaluation_mask'] != 'none':
            raise ValueError('formal evaluation protocol differs')
        with np.load(work / 'epoch_01_confusion.npz') as saved:
            matrix = saved['confusion'].copy()
        if matrix.shape != (7, 18, 18):
            raise ValueError('expected seven half-second horizons and eighteen classes')
        for h, hist in enumerate(matrix):
            union = hist.sum(0) + hist.sum(1) - hist.diagonal()
            iou = np.divide(hist.diagonal(), union, out=np.zeros(18), where=union > 0) * 100
            if (not np.array_equal(hist.sum(1), metric['horizons'][h]['class_gt_count']) or
                    not np.allclose(iou, metric['horizons'][h]['class_iou'], rtol=1e-12, atol=1e-12)):
                raise ValueError('saved metrics do not match formal confusion matrices')
        metrics[name], matrices[name] = metric, matrix
        hist = matrix[0]
        tp, fp, fn = int(hist[:17, :17].sum()), int(hist[17, :17].sum()), int(hist[:17, 17].sum())
        binary[name] = dict(tp=tp, fp=fp, fn=fn, predicted_occupied_percent=100 * (tp + fp) / int(hist.sum()),
                            gt_occupied_percent=100 * (tp + fn) / int(hist.sum()),
                            conditional_semantic_accuracy_percent=100 * np.trace(hist[:17, :17]) / max(tp, 1))
        conditionals[name] = class_conditionals(hist)
    if any(not np.array_equal(m.sum(2), matrices['Strict'].sum(2)) for m in matrices.values()):
        raise ValueError('GT counts differ across experiments')
    for region in ('full', 'camera', 'noncamera'):
        counts = []
        for name in ARMS:
            with np.load(ANALYSIS / name / 'epoch_diagnostics/epoch_01_visibility_confusion.npz') as saved:
                counts.append(saved[region].sum(2))
        if not np.array_equal(*counts):
            raise ValueError('A/B regional GT counts differ')
    full_diag = read(Path('analysis/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e/epoch_diagnostics/epoch_01_diagnostics.json'))
    if len({tuple(v['attribute_sample_indices']) for v in (full_diag, a['diagnostic'], b['diagnostic'])}) != 1:
        raise ValueError('render/attribute probes are not matched')
    diagnostics = dict(Fullspace_CE=full_diag, **{name: arm['diagnostic'] for name, arm in arms.items()})
    attributes = {}
    for name, diagnostic in diagnostics.items():
        current = [r for r in diagnostic['attributes'] if r['horizon'] == 0]
        attributes[name] = {key: float(np.mean([r[key] for r in current])) for key in
                            ('empty_bias', 'empty_scalar', 'in_range_percent', 'scale_upper_saturation_percent', 'opacity_upper_saturation_percent')}
        attributes[name]['softplus_semantic_max_median_mean'] = float(np.mean([r['softplus_semantic_max_quantiles'][2] for r in current]))
    counterfactuals = [reweight_ce_record(record) for record in b['classification']['records']]
    numeric_keys = [key for key in counterfactuals[0] if key not in ('index', 'counterfactual_buckets')]
    counterfactual_mean = {key: float(np.mean([row[key] for row in counterfactuals])) for key in numeric_keys}
    counterfactual_mean['actual_negative_gradient_samples'] = sum(r['actual_signed_bias_gradient'] < 0 for r in counterfactuals)
    counterfactual_mean['old_weight_positive_gradient_samples'] = sum(r['counterfactual_signed_bias_gradient'] > 0 for r in counterfactuals)
    weights = np.asarray(runpy.run_path(str(ARMS['B_ce_free_0_1']))['loss']['occupancy_loss']['manual_class_weight'])
    counts = matrices['B_ce_free_0_1'][0].sum(1)
    denominator = float((counts * weights).sum())
    old_denominator = denominator + .4 * int(counts[17])
    pooled = dict(GT_occupied_percent=100 * counts[:17].sum() / counts.sum(),
                  B_weighted_free_GT_mass_percent=100 * counts[17] * .1 / denominator,
                  old_weighted_free_GT_mass_percent=100 * counts[17] * .5 / old_denominator,
                  B_to_old_nonempty_ce_coefficient=old_denominator / denominator)
    math_checks = verify_counterfactual_math()
    snapshot = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    scope='first_epoch_only', planned_epochs=15, completed_epochs=1,
                    metrics=metrics, binary_counts=binary, conditional_semantics=conditionals,
                    same_output_B_old_weight_counterfactuals=counterfactuals,
                    same_output_counterfactual_mean=counterfactual_mean,
                    pooled_GT_label_weight_evidence=pooled, attributes_same_16_samples=attributes,
                    arms=arms, math_verification=math_checks,
                    verification=dict(same_AB_initial_model=True, same_AB_subset=True,
                                      checkpoint_and_config_hashes=True, class_ious_recomputed=True,
                                      all_horizons_same_GT=True, regional_AB_same_GT=True,
                                      exact_formal_visibility_partition=True),
                    recommendations_are_proposals=True,
                    limitations=['One seed and one epoch; not a completed 15-epoch result.',
                                 'Formal metrics: 1024 samples; counterfactual/render probes: 16; output gradients: two.',
                                 'Counterfactual weights use identical B outputs, excluding Lovasz/set/trajectory terms.',
                                 'Label-weight masses are not observed gradient or actual loss proportions.',
                                 'Shared network parameter gradients and clipping frequency were not recorded.'])
    write(ANALYSIS / 'B_first_epoch_analysis.json', snapshot)
    csv_rows = []
    for h in range(7):
        for c, name in enumerate(CLASS_NAMES):
            row = dict(seconds=h * .5, class_index=c, class_name=name, gt_voxels=int(matrices['Strict'][h, c].sum()))
            row.update({f'{arm}_iou': value['horizons'][h]['class_iou'][c] for arm, value in metrics.items()})
            row.update(B_minus_A_pp=row['B_ce_free_0_1_iou'] - row['A_softmax_focal_iou'],
                       B_minus_Strict_pp=row['B_ce_free_0_1_iou'] - row['Strict_iou'],
                       B_minus_Fullspace_CE_pp=row['B_ce_free_0_1_iou'] - row['Fullspace_CE_iou'])
            csv_rows.append(row)
    with (ANALYSIS / 'B_first_epoch_per_horizon_per_class_iou.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    create_figure(snapshot)
    create_markdown(snapshot, diagnostics)
    print(json.dumps(dict(report=str(ANALYSIS / 'B_first_epoch_analysis.md'), csv_rows=len(csv_rows),
                          math_verification=math_checks, same_output_bias_gradient=counterfactual_mean,
                          current_B_miou=metrics['B_ce_free_0_1']['horizons'][0]['miou']), ensure_ascii=False), flush=True)


def create_figure(snapshot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = snapshot['metrics']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ['Strict', 'Camera', 'Full CE', 'A focal', 'B CE 0.1']
    miou = [m['horizons'][0]['miou'] for m in metrics.values()]
    ax = axes[0, 0]
    ax.bar(np.arange(5), miou, color=['#667085', '#8a95a5', '#b0bbc5', '#df9e37', '#269c85'])
    ax.set_xticks(np.arange(5), labels)
    ax.set_title('Current mIoU: epoch 1, 1024 samples (%)')
    ax.set_ylim(0, max(miou) * 1.2)
    for i, value in enumerate(miou):
        ax.text(i, value, f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    classes = [4, 11, 13, 14, 15, 16]
    ax = axes[0, 1]
    for offset, arm, color in zip((-.25, 0, .25), ('Strict', 'A_softmax_focal', 'B_ce_free_0_1'), ('#667085', '#df9e37', '#269c85')):
        ax.bar(np.arange(6) + offset, [metrics[arm]['horizons'][0]['class_iou'][c] for c in classes], width=.25,
               label={'Strict': 'Strict', 'A_softmax_focal': 'A', 'B_ce_free_0_1': 'B'}[arm], color=color)
    ax.set_xticks(np.arange(6), ['car', 'road', 'sidewalk', 'terrain', 'manmade', 'vegetation'], rotation=15)
    ax.set_title('Current class IoU: 1024 samples (%)')
    ax.legend()
    cf = snapshot['same_output_counterfactual_mean']
    ax = axes[1, 0]
    for offset, key, color, label in ((-.25, 'free', '#c1c7cf', 'free'), (0, 'occupied', '#269c85', 'occupied'), (.25, None, '#df9e37', 'net')):
        old = cf['counterfactual_signed_bias_gradient'] if key is None else cf[f'counterfactual_{key}_bias_gradient']
        actual = cf['actual_signed_bias_gradient'] if key is None else cf[f'actual_{key}_bias_gradient']
        ax.bar(np.arange(2) + offset, [old, actual], width=.25, color=color, label=label)
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_xticks(np.arange(2), ['CE free=0.5', 'CE free=0.1 (B)'])
    ax.set_title('Same B logits: classification dL/db (16 samples)')
    ax.legend()
    ax = axes[1, 1]
    probes = snapshot['arms']['B_ce_free_0_1']['diagnostic']['render_probes']
    values = [probes[f'h0/empty_bias_delta_{d:+d}/full'] if d else probes['h0/default/full'] for d in (-2, -1, 0, 1, 2)]
    ax.plot([-2, -1, 0, 1, 2], [v['miou'] for v in values], '-o', color='#269c85')
    ax.set_xlabel('Empty bias delta: higher means lower empty logit')
    ax.set_ylabel('mIoU (%)', color='#269c85')
    ax.set_title('B bias probe tradeoff: 16 identical samples')
    other = ax.twinx()
    other.plot([-2, -1, 0, 1, 2], [v['occupied_fp'] / 1e6 for v in values], '--s', color='#bd6058')
    other.set_ylabel('False positives (million voxels)', color='#bd6058')
    fig.tight_layout()
    fig.savefig(ANALYSIS / 'B_first_epoch_evidence.png', dpi=180)
    fig.savefig(ANALYSIS / 'B_first_epoch_evidence.pdf')
    plt.close(fig)


def create_markdown(snapshot, diagnostics):
    metrics = snapshot['metrics']
    binary = snapshot['binary_counts']
    conditionals = snapshot['conditional_semantics']
    b = metrics['B_ce_free_0_1']['horizons'][0]
    a = metrics['A_softmax_focal']['horizons'][0]
    strict = metrics['Strict']['horizons'][0]
    ce = metrics['Fullspace_CE']['horizons'][0]
    cf = snapshot['same_output_counterfactual_mean']
    pooled = snapshot['pooled_GT_label_weight_evidence']
    b_diag = snapshot['arms']['B_ce_free_0_1']['diagnostic']
    attrs = snapshot['attributes_same_16_samples']
    lines = ['# B 第一轮有效的原因与后续解决策略', '',
             f"记录时间：{snapshot['created_at']}。两组都完成第一轮 4096 样本训练和 1024 样本评估，并在第一轮边界暂停。"
             '配置保留原 15 轮/15360 更新的学习率计划。下述后续实验是建议，本次只分析已保存的数据。', '',
             'B 仅将全空间体素加权 CE 的 free 相对权重由 0.5 改为 0.1，保留 CE 系数 10、Lovasz 系数 1、集合监督和时间课程。'
             'A 是原 free=0.5 下将体素 CE 替换为 softmax focal gamma=2。A/B 子集、seed、完整模型初始权重一致。', '',
             '## 1. B 的改善同时体现在语义与占用指标', '',
             '| 第一轮模型 | 当前 mIoU % | 占用 IoU % | precision % | recall % | 预测占用比例 % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, metric in metrics.items():
        m = metric['horizons'][0]
        lines.append(f"| {name} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {binary[name]['predicted_occupied_percent']:.4f} |")
    lines += ['', f"B 相对全空间 CE：mIoU +{b['miou']-ce['miou']:.4f} pp（相对 +{100*(b['miou']/ce['miou']-1):.2f}%），"
              f"召回 +{b['occupied_recall']-ce['occupied_recall']:.4f} pp。相对 A：mIoU +{b['miou']-a['miou']:.4f} pp。", '',
              f"B 相对同轮 Strict：mIoU +{b['miou']-strict['miou']:.4f} pp（+{100*(b['miou']/strict['miou']-1):.2f}%），"
              f"占用 IoU 相对 +{100*(b['occupied_iou']/strict['occupied_iou']-1):.2f}%，"
              f"召回 +{b['occupied_recall']-strict['occupied_recall']:.4f} pp。"
              f"B 的 TP 比 Strict 多 {binary['B_ce_free_0_1']['tp']-binary['Strict']['tp']:,} 个体素，"
              f"FP 仅多 {100*(binary['B_ce_free_0_1']['fp']/binary['Strict']['fp']-1):.2f}%，precision 也高于 Strict。"
              '因此这次收益并非仅来自放宽占用判定造成的召回上升。', '',
              '这是单 seed 的第一轮小幅领先，不能据此判定 15 轮最优成绩或最终预测能力已优于 Strict。'
              'Gaussian 未来直接体素监督尚未启用，未来近零不作为本轮方案优劣的主证据。', '',
              '![B 第一轮证据](B_first_epoch_evidence.png)', '',
              '## 2. 最强机制：权重同时改变正负比例和归一化尺度', '',
              '本代码对整批体素使用按类别权重归一化的 mean CE：', '',
              '`L_CE = 10 * sum(w_y * (-log(p_y))) / sum(w_y)`', '',
              '类别索引形式的加权 CE 按目标类别权重之和归一化，而非简单按体素数平均；'
              '这一行为可由当前 `loss/occupancy_loss.py` 核实，也与 '
              '[PyTorch 官方 CE 定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)一致。'
              '类权重整体归一化到和为 18 的公共倍数会在分子/分母中相消。', '',
              f"正式验证集 GT 占用比例只有 {pooled['GT_occupied_percent']:.4f}%。"
              f"free=0.5 时空类标签加权质量为 {pooled['old_weighted_free_GT_mass_percent']:.2f}%，"
              f"free=0.1 时为 {pooled['B_weighted_free_GT_mass_percent']:.2f}%。"
              '这些是 GT 标签权重统计，既不是实际 loss 占比，也不是实测训练梯度占比。', '',
              '设非空标签权重之和为 S、空体素数为 N，旧分母为 `S+0.5N`，B 分母为 `S+0.1N`。'
              '因此在相同输出和标签上：', '',
              '- 非空体素的 CE loss/梯度系数放大为 `(S+0.5N)/(S+0.1N)`。',
              '- 空体素的 CE 系数变为上述比值的 0.2 倍。', '',
              f"固定 16 样本上，非空系数比的逐样本均值为 {cf['B_to_old_occupied_ce_coefficient']:.3f}，"
              f"空系数比均值为 {cf['B_to_old_free_ce_coefficient']:.3f}。完整验证集池化非空系数比为 "
              f"{pooled['B_to_old_nonempty_ce_coefficient']:.3f}。"
              '所以“free 权重降低 80%”不等于其归一化后绝对梯度降低 80%；B 的核心效果包括显著增强非空监督。', '',
              '## 3. 同一 B 输出下，旧权重会反转空类 bias 的分类梯度', '',
              '无需重训或再次 GPU 前向，可以从 B 保存的 CE 分桶精确重建旧权重反事实。'
              '若 B 分母为 Z、free 标签权重和为 F，旧分母为 `Z+4F`；'
              '非空桶乘 `Z/(Z+4F)`、空桶乘 `5Z/(Z+4F)`，loss 与输出分类梯度都按同一因子缩放。', '',
              '| 同一 B checkpoint 输出，固定 16 样本 | free 分类 dL/db | occupied 分类 dL/db | 总分类 dL/db |',
              '|---|---:|---:|---:|',
              f"| 还原 free=0.5 CE | {cf['counterfactual_free_bias_gradient']:+.6f} | {cf['counterfactual_occupied_bias_gradient']:+.6f} | {cf['counterfactual_signed_bias_gradient']:+.6f} |",
              f"| B 实际 free=0.1 CE | {cf['actual_free_bias_gradient']:+.6f} | {cf['actual_occupied_bias_gradient']:+.6f} | {cf['actual_signed_bias_gradient']:+.6f} |", '',
              '`empty_logit = rendered_empty_logit - bias`。正 dL/db 在梯度下降时降低 bias、提高空类分数；'
              '负 dL/db 提高 bias、降低空类分数。', '',
              f"在 {cf['old_weight_positive_gradient_samples']}/16 个样本上，旧权重产生正分类 bias 梯度；"
              f"B 权重在 {cf['actual_negative_gradient_samples']}/16 个样本上产生负梯度。"
              '这直接表明，在 B 已学到的同一输出上，两组权重提供的空类优化方向不同，'
              '比对不同 checkpoint 的最终 bias 作因果推断更可靠。', '',
              '反事实只包含当前体素 CE 分类项，排除 Lovasz、集合/轨迹项，未包含优化器动量、共享参数耦合及训练输出变化。'
              '它不是旧权重重新训练后的结果，也不能代表整个训练过程中每一步的梯度。'
              '重建公式已通过独立 CPU CE/autograd 数值验证。', '',
              '## 4. B 的前景幅值更强，改善不由更大的最终 bias 单独解释', '',
              '| 相同 16 样本当前属性均值 | 全空间 CE | A | B |', '|---|---:|---:|---:|']
    for key in attrs['B_ce_free_0_1']:
        lines.append(f"| {key} | {attrs['Fullspace_CE'][key]:.6f} | {attrs['A_softmax_focal'][key]:.6f} | {attrs['B_ce_free_0_1'][key]:.6f} |")
    lines += ['', f"B 的 bias={attrs['B_ce_free_0_1']['empty_bias']:.4f}，低于 A 的 {attrs['A_softmax_focal']['empty_bias']:.4f}；"
              f"但 softplus 语义最大值的样本内中位数均值从 A 的 {attrs['A_softmax_focal']['softplus_semantic_max_median_mean']:.4f} "
              f"升至 B 的 {attrs['B_ce_free_0_1']['softplus_semantic_max_median_mean']:.4f}。"
              '这是点级语义属性幅值证据，不是最终体素 logits 的统一放大倍数。'
              'B 还改变了点位置、尺度和 opacity，因此收益涉及前景学习，不只是 bias 标定。', '',
              'ROI 内点比例反而降低，尺度/opacity 更接近上界；这不支持“ROI 内点更多”或“属性饱和已经缓解”作为 B 改善的解释。'
              '尺度 >0.395m、opacity >0.99 的比例都超过 99%，仍值得在完整训练中持续诊断。', '',
              '| 样本 | A 体素/集合语义梯度范数比 | B 比值 | A 中心梯度比 | B 中心梯度比 | B 语义余弦 | B 中心余弦 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for old, new in zip(snapshot['arms']['A_softmax_focal']['diagnostic']['gradient_probes'], b_diag['gradient_probes']):
        if old['index'] != new['index']:
            raise ValueError('gradient sample mismatch')
        lines.append(f"| {new['index']} | {old['semantic']['voxel_to_set_norm_ratio']:.3f} | {new['semantic']['voxel_to_set_norm_ratio']:.3f} | {old['geometry']['voxel_to_set_norm_ratio']:.3f} | {new['geometry']['voxel_to_set_norm_ratio']:.3f} | {new['semantic']['cosine']:.3f} | {new['geometry']['cosine']:.3f} |")
    lines += ['', 'B 的末层体素梯度相对集合梯度更强，与非空监督增强的机制一致。'
              '这些比值比较不同 checkpoint，且只覆盖 2 样本的末层 raw logits/world centers；'
              '不能等同于全网络梯度占比，也不能据此断言集合与体素损失冲突。共享参数梯度和裁剪触发频率尚未记录。', '',
              'A 主要降低容易空体素的损失，但保留 free=0.5 的标签权重分母。'
              'B 直接改变分母与正负类别系数，两者处理的是不同方面。'
              '[Focal 原论文](https://arxiv.org/abs/1708.02002)的机制是降低容易样本贡献；'
              'A 的结果不能推导出“focal 对这个模型永远无效”。free=0.1 加 focal 尚未进行单变量实验。', '',
              '## 5. 类别差距已经从普遍漏检转向具体类别的问题', '',
              '| 当前类别 | Strict IoU % | A IoU % | B IoU % | B−Strict pp | Strict 占用召回 % | B 占用召回 % | Strict 命中后类别正确率 % | B 命中后类别正确率 % |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in (4, 10, 11, 13, 14, 15, 16):
        s, m = conditionals['Strict'][c], conditionals['B_ce_free_0_1'][c]
        lines.append(f"| {CLASS_NAMES[c]} | {strict['class_iou'][c]:.4f} | {a['class_iou'][c]:.4f} | {b['class_iou'][c]:.4f} | {b['class_iou'][c]-strict['class_iou'][c]:+.4f} | {s['occupied_recall_percent']:.3f} | {m['occupied_recall_percent']:.3f} | {s['conditional_semantic_accuracy_percent']:.3f} | {m['conditional_semantic_accuracy_percent']:.3f} |")
    lines += ['', '占用召回只判断该 GT 类体素是否被预测成任意非空；命中后类别正确率的分母是该类已预测为任意非空的 GT 体素。'
              '后者不是独立于定位的分类准确率。', '',
              '- sidewalk：占用召回已略高于 Strict，但命中后类别正确率约 15%，Strict 约 33%。大量 sidewalk 预测成道路/terrain；优先加强非空类别区分。',
              '- car：占用召回约 10.5%，Strict 约 21.3%；命中后类别正确率也偏低。因此同时存在空间覆盖/检出不足和道路混淆。',
              '- 道路、terrain、manmade、vegetation 的 IoU 已超过同轮 Strict；它们的增益抵消 car、sidewalk、truck 的差距，使整体均值略领先。',
              '- B 仍有 11 个非空类 IoU 为零。第一轮类别学习覆盖有限，需要后续课程和各类 GT 数量一起判断，不能以总均值替代类别检查。', '',
              '## 6. 空类约束仍需保留，误报集中在 camera mask 外', '',
              '| 模型 | 区域 | mIoU % | 占用 IoU % | precision % | recall % | FP | FN |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, diagnostic in diagnostics.items():
        for region in ('camera', 'noncamera'):
            m = diagnostic['visibility']['h0/' + region]
            lines.append(f"| {name} | {region} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {m['occupied_fp']:,} | {m['occupied_fn']:,} |")
    hidden = b_diag['visibility']['h0/noncamera']
    total = b_diag['visibility']['h0/full']
    lines += ['', f"B 的 {100*hidden['occupied_fp']/total['occupied_fp']:.2f}% 误报来自 camera mask 外；"
              f"该区 precision 仅 {hidden['occupied_precision']:.4f}%。"
              '这是数据标注的可见性分区，不等于完全没有历史图像信息。'
              'Strict 第一轮没有同口径的完整区域矩阵，不能声称 B 在该区域比 Strict 更差。', '',
              '| 同一 B checkpoint，固定 16 样本渲染干预 | mIoU % | 占用 IoU % | precision % | recall % | FP |',
              '|---|---:|---:|---:|---:|---:|']
    for name in ('default', 'empty_bias_delta_+1', 'empty_bias_delta_+2', 'point_rasterizer', 'fixed_scale_0_2', 'opacity_times_0_5'):
        m = b_diag['render_probes']['h0/' + name + '/full']
        lines.append(f"| {name} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {m['occupied_fp']:,} |")
    lines += ['', '进一步提高 bias 的 mIoU 收益趋于平缓，FP 却快速增加。该诊断不能直接选作验证集最优阈值。'
              '点栅格化也未提高 B 的整体指标；固定尺度 0.2m 或 opacity 减半几乎全空。'
              '这说明不能单独通过缩尺度、降低 opacity 或继续大幅削弱 free 来解决剩余问题。', '',
              '## 7. 推荐的解决策略与可检验实验', '',
              '### 第一优先：B 作为完整训练基线', '',
              '推荐下一项实验从 B 的 epoch_01.pth 继续第 2–15 轮，恢复优化器、RNG、查询时间分配和原 15 轮学习率；'
              '保持 free=0.1、CE=10、Lovasz=1 以及其余模型/课程配置。当前 A/B 仍按原要求暂停，尚未执行这项建议。', '',
              '每轮记录全部 7 时刻 ×18 类 IoU、占用 precision/recall，以及 camera/noncamera FP/FN。'
              '特别检查预训练结束和 1/2/3 秒直接体素监督启用时的变化。'
              '按原 current/1s/2s/3s mIoU 均值选择 checkpoint，并与 Strict 相同 15 轮预算下的最优成绩比较。'
              '若 B 的前景幅值持续增长、属性继续饱和而不可见 FP 上升，第一轮领先可能无法保持。', '',
              '### 第二优先：单独加强非空语义区分', '',
              '建议新的 C_conditional_semantic，在 B 的损失上只添加一个非空条件语义 CE；模型、free 权重和其他损失保持一致。', '',
              '`L = L_B + lambda_sem * sum_GT_occupied(w_y * CE(z_foreground, y)) / sum_GT_occupied(w_y)`', '',
              '`z_foreground` 只含 17 个非空渲染 logits，CE 内部只在这 17 类之间做 log-softmax；传入原始 logits，避免重复 softmax。'
              '只在 GT 非空体素计算该辅助项；原 B 的 18 类全空间 CE 继续监督所有空/非空体素。'
              '不需要新增模型头，不改变全空间监督要求。该项对 free logit 的直接梯度为 0，'
              '且对 17 个前景 logits 的梯度和为 0，因此在固定体素输出上主要改变非空类别间的相对分数。'
              '共享网络参数仍可能间接影响占用，不能声称两任务已完全解耦。', '',
              '这是有数学依据的待验证方案：在 GT 非空时，18 类 CE 可分解为“占用概率负对数 + 条件类别 CE”。'
              'B 的 sidewalk 召回接近 Strict、条件类别正确率较低，正好给这一辅助项提供了针对性目标。', '',
              '候选 lambda_sem=1.0，可再单独比较 0.5；这些是初始实验值，尚未证明最优。'
              '采用相同 scratch 初始化、15 轮计划，在第一轮先诊断 car/sidewalk IoU、条件类别正确率和不可见 FP。'
              '无 GT 非空体素时辅助项返回与 logits 连接的零，忽略 255。通过数值测试确认 free 梯度为零，且不影响原全空间 CE。'
              '如果只改变类别预测而 car 占用召回仍低，下一步再处理 car 的点覆盖/位置与局部聚合。', '',
              '### 第三优先：稳定空/非空监督比例', '',
              '若后续不同场景或未来时刻的 GT 占用比例变化使 B 的有效系数波动，建议独立的 D_group_normalized，'
              '仅将 CE 改为按 GT occupied/free 两组分别归一化，保留全部体素。', '',
              '`L_cls = 10 * [rho_occ * weighted_mean_CE_GT_occupied + (1-rho_occ) * mean_CE_GT_free]`', '',
              f"初始 rho_occ 可设约 {1-pooled['B_weighted_free_GT_mass_percent']/100:.3f}（即 0.33），"
              '来自 B 的验证集池化标签权重比例，仅作为诊断中的数值示例。正式训练应先由 4096 个训练样本的标签统计计算相应参考值，'
              '并非直接采用验证标签统计得到的常数；该起点也不是实测最优值。'
              '固定组系数可避免空体素数量改变时正样本组权重随之漂移，随后能独立调节占用与空类约束。'
              '非空组保留原 17 类权重，free 组按体素取均值。缺失某组时只计算存在组，并对存在组的系数重新归一化，避免除零。'
              '该方案需单独验证；不能把标签质量比例当成 loss/梯度质量比例。', '',
              '若 FP 仍成为主要瓶颈，再在该独立基线上增加难空体素的辅助约束，例如保留所有 free 的均值项，'
              '再混合一部分 high-CE free 项，关注高置信度假占用。'
              '先固定总空类组系数，避免重新退回正样本被压制的情况；也避免只保留少数负样本而丢失全空间监督。'
              '分组归一化与难负例约束分成两次实验，不能同时改。', '',
              '### 第四优先：在损失稳定后处理渲染与属性', '',
              '先增加类别级覆盖/聚合诊断：GT car/sidewalk 邻域的 child 点覆盖、有效 Gaussian 贡献数、'
              '前景最大值与 empty logit 的差、道路/sidewalk 的相对分数。当前 ROI 内点比例无法代替这些指标。', '',
              '若点栅格化与 Gaussian 的类别差距稳定存在，再考虑将局部密度负责占用、归一化语义负责类别的渲染形式。'
              '直接将当前正前景 sum 除以密度，会降低与现有空类分数竞争的幅值；需同步设计占用标定，并作为独立结构实验。'
              '不要同时改尺度范围、opacity 参数化、聚合方式和损失。尺度/opacity 饱和需要归因实验，不能仅凭边界统计认定它是唯一原因。', '',
              '### focal 的后续位置', '',
              '不建议直接把 A、B 与其他结构改动全部组合。若测试 focal，使用单独的 B_focal：'
              '固定 B 的 free=0.1 和全部设置，只把 CE 替换为 focal；可先测试 gamma=1。'
              '这才能回答“降低 free 权重后 focal 是否仍有增益”。目前已测结果仅能支持 B 优于 A 的具体设置。', '',
              '## 8. 验证范围与文件', '',
              '正式比较均为第一轮完整 1024 样本，mIoU 固定平均 17 个非空类，free 单列。'
              '全部 7 时刻的 GT 类计数一致，逐类 IoU 已从原混淆矩阵重新计算；'
              'A/B 可见性矩阵严格重建正式矩阵，并核验了 checkpoint、独立配置和完整初始模型 SHA256。'
              '反事实与渲染干预均是同一组 16 样本，末层梯度是两样本，不代表全网络训练梯度。', '',
              '- [B 分析原始数据、16 样本反事实与数值验证](B_first_epoch_analysis.json)',
              '- [126 行逐时刻逐类别 IoU 对比](B_first_epoch_per_horizon_per_class_iou.csv)',
              '- [证据图 PNG](B_first_epoch_evidence.png) / [PDF](B_first_epoch_evidence.pdf)',
              '- [A/B 完整逐类别报告](report.md)',
              '- [B 原始诊断](B_ce_free_0_1/epoch_diagnostics/epoch_01_diagnostics.md)',
              '- [B 分类分桶](B_ce_free_0_1/epoch_diagnostics/epoch_01_classification_components.json)', '']
    (ANALYSIS / 'B_first_epoch_analysis.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    analyze()
