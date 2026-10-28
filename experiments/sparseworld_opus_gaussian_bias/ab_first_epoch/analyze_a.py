"""Analyze completed A against matched epoch-one baselines using saved evidence."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import runpy
from zoneinfo import ZoneInfo

import numpy as np

from ..metrics import CLASS_NAMES
from .protocol import ANALYSIS, ARMS, ROOT
from .report import read, verify_arm


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def class_conditionals(hist):
    """Separate occupied detection from semantic correctness on detected GT."""
    rows = []
    for c in range(17):
        occupied = int(hist[c, :17].sum())
        gt = int(hist[c].sum())
        wrong = hist[c, :17].copy()
        wrong[c] = 0
        top = np.argsort(wrong)[-3:][::-1]
        rows.append(dict(class_index=c, class_name=CLASS_NAMES[c], gt_voxels=gt,
                         predicted_free_voxels=int(hist[c, 17]),
                         predicted_free_percent=100 * hist[c, 17] / max(gt, 1),
                         occupied_recall_percent=100 * occupied / max(gt, 1),
                         conditional_semantic_accuracy_percent=100 * hist[c, c] / max(occupied, 1),
                         detected_gt_occupied_voxels=occupied,
                         correct_semantic_voxels=int(hist[c, c]),
                         top_wrong_occupied_predictions=[
                             dict(class_name=CLASS_NAMES[k], voxels=int(wrong[k])) for k in top if wrong[k]]))
    return rows


def analyze():
    original = Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e')
    full_analysis = Path('analysis/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e')
    roots = dict(Strict=original / 'sparseworld_strict',
                 Camera_Gaussian=original / 'sparseworld_opus_gaussian_bias',
                 Fullspace_CE=Path('out/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e/sparseworld_opus_gaussian_bias'),
                 A_softmax_focal=ROOT / 'A_softmax_focal')
    a_verified = verify_arm(roots['A_softmax_focal'], ANALYSIS / 'A_softmax_focal')
    audit = read(ANALYSIS / 'configuration_audit.json')
    for arm, config in ARMS.items():
        if hashlib.sha256(config.read_bytes()).hexdigest() != audit['arms'][arm]['config_sha256']:
            raise ValueError('validated configuration changed')
    metrics, histograms, binary, conditional = {}, {}, {}, {}
    for name, work in roots.items():
        metrics[name] = read(work / 'epoch_01_metrics.json')
        if metrics[name]['epoch'] != 1 or metrics[name]['num_samples'] != 1024:
            raise ValueError('only complete matched epoch-one evaluations are comparable')
        with np.load(work / 'epoch_01_confusion.npz') as stored:
            histograms[name] = stored['confusion'].copy()
        hist = histograms[name][0]
        tp, fp, fn = int(hist[:17, :17].sum()), int(hist[17, :17].sum()), int(hist[:17, 17].sum())
        binary[name] = dict(tp=tp, fp=fp, fn=fn, tn=int(hist[17, 17]),
                            predicted_occupied_percent=100 * (tp + fp) / int(hist.sum()),
                            gt_occupied_percent=100 * (tp + fn) / int(hist.sum()),
                            conditional_semantic_accuracy_percent=100 * np.trace(hist[:17, :17]) / max(tp, 1))
        conditional[name] = class_conditionals(hist)
        hgt = histograms[name].sum(2)
        if not np.array_equal(hgt, np.array([v['class_gt_count'] for v in metrics[name]['horizons']])):
            raise ValueError('metrics and confusion matrix GT counts differ')
        for h, matrix in enumerate(histograms[name]):
            union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
            iou = np.divide(matrix.diagonal(), union, out=np.zeros(18), where=union > 0) * 100
            if not np.allclose(iou, metrics[name]['horizons'][h]['class_iou'], rtol=1e-12, atol=1e-12):
                raise ValueError('saved class IoUs do not match confusion counts')
    if any(not np.array_equal(v.sum(2), histograms['Strict'].sum(2)) for v in histograms.values()):
        raise ValueError('ground-truth counts differ across experiments')
    initializations = {name: read(work / 'initialization.json') for name, work in roots.items()}
    if len({v['sha256'] for v in initializations.values()}) != 1:
        raise ValueError('image initialization checkpoint differs')
    diagnostics = dict(Fullspace_CE=read(full_analysis / 'epoch_diagnostics/epoch_01_diagnostics.json'),
                       A_softmax_focal=a_verified['diagnostic'])
    visibility = {}
    for name, diag_dir in (
            ('Fullspace_CE', full_analysis / 'epoch_diagnostics'),
            ('A_softmax_focal', ANALYSIS / 'A_softmax_focal/epoch_diagnostics')):
        with np.load(diag_dir / 'epoch_01_visibility_confusion.npz') as stored:
            matrices = {k: stored[k].copy() for k in stored.files}
        if (not np.array_equal(matrices['full'], histograms[name]) or
                not np.array_equal(matrices['full'], matrices['camera'] + matrices['noncamera'])):
            raise ValueError('visibility partition does not reproduce formal evaluation')
        visibility[name] = matrices
    if any(not np.array_equal(visibility['Fullspace_CE'][k].sum(2), visibility['A_softmax_focal'][k].sum(2))
           for k in visibility['Fullspace_CE']):
        raise ValueError('visibility-region GT counts differ')
    if diagnostics['Fullspace_CE']['attribute_sample_indices'] != diagnostics['A_softmax_focal']['attribute_sample_indices']:
        raise ValueError('attribute/render probes do not use identical samples')
    attributes = {}
    for name, diagnostic in diagnostics.items():
        current = [row for row in diagnostic['attributes'] if row['horizon'] == 0]
        attributes[name] = {key: float(np.mean([row[key] for row in current])) for key in
                            ('in_range_percent', 'scale_upper_saturation_percent',
                             'opacity_upper_saturation_percent', 'empty_bias', 'empty_scalar')}
        attributes[name].update(softplus_semantic_max_median_mean=float(np.mean([
            row['softplus_semantic_max_quantiles'][2] for row in current])))
    classifier = a_verified['classification']
    records = classifier['records']
    bucket_summary = {
        bucket: {key: float(np.mean([r['buckets'][bucket][key] for r in records]))
                 for key in records[0]['buckets'][bucket]}
        for bucket in records[0]['buckets']}
    aggregates = {}
    for group in ('free', 'occupied'):
        values = [v for k, v in bucket_summary.items() if k.endswith('/' + group)]
        focal = sum(v['classification_loss_contribution'] for v in values)
        ce = sum(v['ce_counterfactual_loss_contribution'] for v in values)
        aggregates[group] = dict(focal_loss=focal, same_output_ce_loss=ce,
                                 loss_reduction_percent=100 * (1 - focal / ce),
                                 signed_bias_gradient=sum(v['signed_bias_gradient'] for v in values))
    aggregates['free_actual_loss_share_percent'] = 100 * aggregates['free']['focal_loss'] / sum(
        aggregates[k]['focal_loss'] for k in ('free', 'occupied'))
    aggregates['free_ce_counterfactual_loss_share_percent'] = 100 * aggregates['free']['same_output_ce_loss'] / sum(
        aggregates[k]['same_output_ce_loss'] for k in ('free', 'occupied'))
    aggregates['signed_bias_gradient_total'] = float(np.mean([r['signed_bias_gradient'] for r in records]))
    weights = np.asarray(runpy.run_path(str(ARMS['A_softmax_focal']))['loss']['occupancy_loss']['manual_class_weight'])
    counts = histograms['A_softmax_focal'][0].sum(1)
    a_denominator = float((counts * weights).sum())
    b_weights = weights.copy()
    b_weights[17] = .1
    b_denominator = float((counts * b_weights).sum())
    denominator_evidence = dict(
        A_weighted_free_gt_mass_percent=100 * counts[17] * weights[17] / a_denominator,
        B_weighted_free_gt_mass_percent=100 * counts[17] * b_weights[17] / b_denominator,
        B_to_A_occupied_ce_coefficient_same_output_pooled=a_denominator / b_denominator,
        note='Pooled held-out GT label-weight calculation, not observed training gradient share. '
             'Focal modulation derivative is excluded from this coefficient comparison.')
    current_a = metrics['A_softmax_focal']['horizons'][0]
    current_strict = metrics['Strict']['horizons'][0]
    gap = np.asarray(current_strict['class_iou'][:17]) - np.asarray(current_a['class_iou'][:17])
    top = np.argsort(gap)[-4:][::-1]
    gap_summary = dict(current_miou_gap_pp=current_strict['miou'] - current_a['miou'],
                       top_four_classes=[CLASS_NAMES[c] for c in top],
                       top_four_net_gap_share_percent=100 * gap[top].sum() / gap.sum(),
                       per_class_strict_minus_A_pp=gap.tolist())
    queue_status = read(ANALYSIS / 'status.json')
    snapshot = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    scope='epoch_one_diagnosis_only', planned_epochs=15, trained_epochs=1,
                    formal_sample_count=1024, metrics=metrics, binary_counts=binary,
                    conditional_semantics=conditional, attributes_mean_over_same_16_samples=attributes,
                    focal_classification_bucket_means=bucket_summary,
                    focal_classification_aggregates=aggregates, denominator_evidence=denominator_evidence,
                    strict_gap=gap_summary, queue_status_at_analysis=queue_status,
                    A_boundary=a_verified['boundary'],
                    verification=dict(config_sha256=True, checkpoint_sha256=True,
                                      same_image_initialization=True,
                                      all_horizons_identical_gt_counts=True,
                                      class_ious_recomputed_from_confusion=True,
                                      formal_visibility_exact_partition=True,
                                      matching_16_sample_attribute_and_render_probes=True),
                    limitations=['Only a single seed and a first-epoch checkpoint.',
                                 'Formal evaluation is 1024 samples; rendering/classifier probes are 16; output gradient probes are two.',
                                 'Baseline full model initialization fingerprints were not recorded; only the same image checkpoint is verified.',
                                 'This is an A analysis; B has not completed its first-epoch evaluation.',
                                 'Bias probes are diagnostics, not selected validation-tuned final model results.'])
    write(ANALYSIS / 'A_first_epoch_analysis.json', snapshot)
    rows = []
    for h in range(7):
        for c, name in enumerate(CLASS_NAMES):
            row = dict(seconds=h * .5, class_index=c, class_name=name,
                       gt_voxels=metrics['Strict']['horizons'][h]['class_gt_count'][c])
            row.update({f'{arm}_iou': metric['horizons'][h]['class_iou'][c] for arm, metric in metrics.items()})
            row.update(A_minus_camera_pp=row['A_softmax_focal_iou'] - row['Camera_Gaussian_iou'],
                       A_minus_fullspace_ce_pp=row['A_softmax_focal_iou'] - row['Fullspace_CE_iou'],
                       A_minus_strict_pp=row['A_softmax_focal_iou'] - row['Strict_iou'])
            rows.append(row)
    with (ANALYSIS / 'A_first_epoch_per_horizon_per_class_iou.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ['# A 第一轮：focal 改善了什么，为什么仍弱于 Strict', '',
             f"记录时间：{snapshot['created_at']}。A 完成 4096 样本训练及 1024 样本正式评估，在第一轮边界暂停。"
             '配置保留 15 轮/15360 次更新的学习率计划，本报告比较所有模型的第一轮，不代表 15 轮最优结果。', '',
             'A 的体素加权 CE 替换为 softmax focal（gamma=2），free 相对权重保留 0.5，Lovasz、集合监督及其权重保持原设置。', '',
             '## 1. 改善需要按指标区分', '',
             '| 第一轮模型 | 当前 mIoU % | 占用 IoU % | precision % | recall % | 预测占用体素比例 % |',
             '|---|---:|---:|---:|---:|---:|']
    for name, metric in metrics.items():
        m = metric['horizons'][0]
        lines.append(f"| {name} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {binary[name]['predicted_occupied_percent']:.4f} |")
    camera = metrics['Camera_Gaussian']['horizons'][0]
    ce = metrics['Fullspace_CE']['horizons'][0]
    lines += ['', f"A 相对原 camera Gaussian 的语义 mIoU 提高 {current_a['miou']-camera['miou']:.4f} pp"
              f"（{100*(current_a['miou']/camera['miou']-1):.2f}%），相对全空间 CE 提高 "
              f"{100*(current_a['miou']/ce['miou']-1):.2f}%。"
              f"但 A 的占用 IoU 相对 camera 下降 {100*(1-current_a['occupied_iou']/camera['occupied_iou']):.2f}%，"
              '不能将语义均值的轻微改善解释为所有指标都变好。', '',
              f"相对 Strict，A 当前 mIoU 低 {gap_summary['current_miou_gap_pp']:.4f} pp"
              f"（{100*(1-current_a['miou']/current_strict['miou']):.2f}%）。"
              f"相同 GT 占用比例为 {binary['Strict']['gt_occupied_percent']:.4f}%。"
              f"A 检出占用 TP={binary['A_softmax_focal']['tp']:,}，漏检 FN={binary['A_softmax_focal']['fn']:,}；"
              f"Strict TP={binary['Strict']['tp']:,}。单 seed 第一轮的 0.0330 pp 改善还不足以证明稳定收益。", '',
              '## 2. focal 确实削弱了容易空体素，尚未恢复充分召回', '',
              '以下是固定 16 样本、同一 checkpoint、同一渲染输出的分类项统计，包含分类项系数 10。'
              'CE 反事实沿用 A 的类别权重及分母，和 focal 在同一输出上计算；它不是另一次 CE 训练。', '',
              '| GT 分组 | focal loss 均值 | 同输出 CE loss 均值 | 损失减少 % | focal dL/db 均值 |',
              '|---|---:|---:|---:|---:|']
    for group in ('free', 'occupied'):
        m = aggregates[group]
        lines.append(f"| {group} | {m['focal_loss']:.6f} | {m['same_output_ce_loss']:.6f} | {m['loss_reduction_percent']:.2f} | {m['signed_bias_gradient']:+.6f} |")
    lines += ['', f"空体素占实际 focal 分类 loss 的 {aggregates['free_actual_loss_share_percent']:.2f}%，"
              f"同输出 CE 中为 {aggregates['free_ce_counterfactual_loss_share_percent']:.2f}%。"
              '此前的约 91% 是 GT 标签加权质量，不能当作实际 loss 或训练梯度占比。', '',
              '`L = 10 * sum(w_y * (1-p_y)^2 * (-log(p_y))) / sum(w_y)`。'
              'focal 会降低容易样本贡献；固定分母保留了原全空间标签权重的归一化尺度，'
              '并没有同时恢复 camera 范围下的可见正样本系数。不能仅看非空损失占比上升，就认定正样本绝对梯度足够。', '',
              f"作为待验证的 B 机制，按完整验证集 GT 池化计算，free 权重降到 0.1 后，"
              f"空类标签加权质量从 {denominator_evidence['A_weighted_free_gt_mass_percent']:.2f}% 变成 "
              f"{denominator_evidence['B_weighted_free_gt_mass_percent']:.2f}%；同输出 CE 非空样本系数约变为 "
              f"{denominator_evidence['B_to_A_occupied_ce_coefficient_same_output_pooled']:.3f} 倍。"
              '这是池化分布推导，实际 CE 按 batch 归一化，且 A 还含 focal 导数因子，不能称为 B/A 实测梯度比。', '',
              f"A 当前分类项总 dL/db={aggregates['signed_bias_gradient_total']:+.6f}。"
              '`empty_logit = rendered_empty_logit - bias`，负梯度在梯度下降时提高 bias、降低空类分数。'
              '该统计排除 Lovasz、集合损失、轨迹项以及训练期间的输出变化，不能代表整个训练轨迹。', '',
              '| 区域/标签 | focal loss | CE 反事实 | focal dL/db | 正确类别概率均值 |',
              '|---|---:|---:|---:|---:|']
    for bucket, m in bucket_summary.items():
        lines.append(f"| {bucket} | {m['classification_loss_contribution']:.6f} | {m['ce_counterfactual_loss_contribution']:.6f} | {m['signed_bias_gradient']:+.6f} | {m['mean_correct_class_probability']:.6f} |")
    lines += ['', '## 3. 剩余差距：漏检为主，car 和 sidewalk 还存在类别混淆', '',
              f"car、sidewalk、vegetation、manmade 合计解释当前 mIoU 净差距的 "
              f"{gap_summary['top_four_net_gap_share_percent']:.2f}%。道路与 terrain 已较接近 Strict。", '',
              '| 当前类别 | Strict IoU % | A IoU % | A−Strict pp | Strict 占用召回 % | A 占用召回 % | Strict 命中占用后类别正确率 % | A 命中占用后类别正确率 % |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in (4, 11, 13, 14, 15, 16):
        s, a = conditional['Strict'][c], conditional['A_softmax_focal'][c]
        lines.append(f"| {CLASS_NAMES[c]} | {current_strict['class_iou'][c]:.4f} | {current_a['class_iou'][c]:.4f} | {current_a['class_iou'][c]-current_strict['class_iou'][c]:+.4f} | {s['occupied_recall_percent']:.3f} | {a['occupied_recall_percent']:.3f} | {s['conditional_semantic_accuracy_percent']:.3f} | {a['conditional_semantic_accuracy_percent']:.3f} |")
    lines += ['', '占用召回在每个 GT 类别中只判断是否预测为任意非空类；条件类别正确率的分母为该类已被预测成任意非空的 GT 体素。'
              '这不是 IoU，也不是排除定位误差后的独立分类准确率。', '',
              'A 的 car 约 96.5% 被预测为空；已判成非空的 car 体素中，大部分仍预测为道路。'
              'sidewalk 已判为非空的体素也大量预测成道路/terrain。'
              'vegetation 的命中后正确率与 Strict 接近，其差距更集中在覆盖/占用检出。'
              '因此只调全局空类阈值，有望缓解漏检，但不会直接改变非空类别之间的相对分数。', '',
              '## 4. 全空间监督的误报压力依然存在', '',
              '| 模型 | 区域 | mIoU % | 占用 IoU % | precision % | recall % | FP 体素数 | FN 体素数 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, diag in diagnostics.items():
        for region in ('camera', 'noncamera'):
            m = diag['visibility']['h0/' + region]
            lines.append(f"| {name} | {region} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {m['occupied_fp']:,} | {m['occupied_fn']:,} |")
    a_diag = diagnostics['A_softmax_focal']
    hidden = a_diag['visibility']['h0/noncamera']
    lines += ['', f"A 的 {100*hidden['occupied_fp']/binary['A_softmax_focal']['fp']:.2f}% 占用误报来自 camera mask 外。"
              '相对全空间 CE，focal 恢复了部分召回，也增加了误报。'
              '这支持同时观察可见/不可见区域，而不能无限削弱空类约束。camera mask 外是数据标注分区，不等于该体素完全没有任何历史图像信息。', '',
              '## 5. 渲染和属性仍是候选瓶颈', '',
              '| 相同 16 样本当前属性均值 | 全空间 CE | A |', '|---|---:|---:|']
    for key in attributes['A_softmax_focal']:
        lines.append(f"| {key} | {attributes['Fullspace_CE'][key]:.6f} | {attributes['A_softmax_focal'][key]:.6f} |")
    lines += ['', 'A 的尺度超过 0.395m 比例仍约 98%，opacity 超过 0.99 比例约 94%。'
              'sigmoid 属性接近边界时导数变小。该饱和现象在旧模型也存在，是待处理的结构问题，不能单凭饱和就断言它解释全部成绩差距。'
              'ROI 内点比例也不是 GT 占用区域的覆盖率。', '',
              '渲染器对邻域的 `opacity * Gaussian_kernel * softplus(class_logit)` 做求和，'
              '空类来自全局 Gaussian 再减全局 bias。前景强度同时依赖点密度、尺度、不透明度和语义幅值。'
              '集合监督则作用于原始 logits 和点坐标，两者没有天然一致的体素分数标定。'
              '这是代码结构支持的候选解释，因果作用仍需单变量消融。', '',
              '以下干预固定同一 A checkpoint，只改变渲染参数，均为同一 16 样本，不能拿它们与 Strict 的 1024 样本结果直接判胜负。', '',
              '| A 当前渲染诊断 | mIoU % | 占用 IoU % | precision % | recall % | FP |',
              '|---|---:|---:|---:|---:|---:|']
    probe_names = ('default', 'point_rasterizer', 'empty_bias_delta_+1', 'empty_bias_delta_+2',
                   'fixed_scale_0_2', 'opacity_times_0_5')
    for name in probe_names:
        m = a_diag['render_probes']['h0/' + name + '/full']
        lines.append(f"| {name} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {m['occupied_fp']:,} |")
    default = a_diag['render_probes']['h0/default/full']
    point = a_diag['render_probes']['h0/point_rasterizer/full']
    lines += ['', '提高 bias 即降低空类 logit，能明显提高这 16 样本的召回和 mIoU，说明当前决策仍偏向空类；'
              '但误报也快速上升。不能据此直接把 +2 当作部署阈值或训练最优 bias。', '',
              f"A 改成点栅格化时，总 mIoU 从 {default['miou']:.4f}% 变成 {point['miou']:.4f}%，没有提升；"
              f"但 car IoU 从 {default['class_iou'][4]:.4f}% 升到 {point['class_iou'][4]:.4f}%，"
              f"道路从 {default['class_iou'][11]:.4f}% 降到 {point['class_iou'][11]:.4f}%。"
              '渲染对类别有明显不同的影响，不能延用旧 CE checkpoint 的“点栅格化整体更好”作为 A 的结论。', '',
              '固定尺度 0.2m 或 opacity 减半几乎全空，说明当前前景幅值依赖现有大尺度/高不透明度。'
              '直接缩尺度可能进一步降低召回，需要同时处理前景幅值与空类标定。', '',
              '| A 样本 | 体素/集合语义梯度范数比 | 语义余弦 | 中心梯度范数比 | 中心余弦 |',
              '|---|---:|---:|---:|---:|']
    for row in a_diag['gradient_probes']:
        lines.append(f"| {row['index']} | {row['semantic']['voxel_to_set_norm_ratio']:.3f} | {row['semantic']['cosine']:.3f} | {row['geometry']['voxel_to_set_norm_ratio']:.3f} | {row['geometry']['cosine']:.3f} |")
    lines += ['', '两样本末层输出梯度的整体余弦为正，不能声称集合与体素损失全面冲突；'
              '这些不是共享网络参数梯度，也不能用梯度范数比证明某项主导整个优化器更新。', '',
              '## 6. 下一步判断顺序', '',
              '先完成已授权的 B 第一轮。若 B 的召回及语义 IoU 优于 A，重点验证降低 free 权重及正样本归一化尺度的效果；'
              '若 B 主要增加不可见误报，继续削弱 free 约束的收益有限。当前还不能从 A 推断 B 的胜负。', '',
              'A 已证明 focal 会减少容易空体素的分类贡献；后续应分别观察占用检出、非空类分数混淆、属性饱和，'
              '通过单变量实验区分全局 bias 标定与局部密度/语义聚合的作用。'
              '本次分析不改训练损失、模型结构或配置。', '',
              '第一轮仍处于原时间扩展预训练阶段；Gaussian 直接体素监督只有当前。'
              '1/2/3 秒监督按原课程从后续轮次逐步启用，近零未来 IoU 不能作为此时 A 的主要失败证据。', '',
              f"分析时 B 队列状态：`{queue_status['status']}`，阶段 `{queue_status.get('phase', '')}`。"
              'B 原启动返回 -11（SIGSEGV），日志为空，无初始化或 checkpoint；故障原因尚未定位。'
              '已保留故障记录并开启 faulthandler 重启 B，保持 A 暂停及原配置。', '',
              '## 7. 各时刻整体指标与每类别 IoU', '',
              '| 时刻 | 模型 | mIoU % | 占用 IoU % |', '|---|---|---:|---:|']
    for h in range(7):
        for name, metric in metrics.items():
            m = metric['horizons'][h]
            lines.append(f"| {h*.5:g}s | {name} | {m['miou']:.6f} | {m['occupied_iou']:.6f} |")
    for h in range(7):
        lines += ['', f'### {h*.5:g} 秒（IoU %）', '',
                  '| 类别 | Strict | camera Gaussian | 全空间 CE | A | A−Strict pp |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in rows[h * 18:(h + 1) * 18]:
            lines.append(f"| {row['class_name']} | {row['Strict_iou']:.6f} | {row['Camera_Gaussian_iou']:.6f} | {row['Fullspace_CE_iou']:.6f} | {row['A_softmax_focal_iou']:.6f} | {row['A_minus_strict_pp']:+.6f} |")
    lines += ['', 'mIoU 固定平均 17 个非空类；free 类单列，不参与 mIoU。所有正式对比为全空间 1024 样本。', '',
              '## 8. 数据与可复现性', '',
              '- [原始指标、混淆分解、损失分桶与验证记录](A_first_epoch_analysis.json)',
              '- [126 行逐时刻逐类别 IoU CSV](A_first_epoch_per_horizon_per_class_iou.csv)',
              '- [A 第一轮完整诊断](A_softmax_focal/epoch_diagnostics/epoch_01_diagnostics.md)',
              '- [A 同输出分类损失分桶](A_softmax_focal/epoch_diagnostics/epoch_01_classification_components.json)',
              '- [独立配置差异审计](configuration_audit.json)', '',
              'GT 类别计数在全部 7 时刻、4 个模型之间完全一致；IoU 已从原混淆矩阵重新计算。'
              '可见/不可见矩阵之和与正式矩阵完全一致，A 和全空间 CE 采用同一组 16 样本进行属性/渲染诊断。'
              'A checkpoint 和独立配置 SHA256 已核验。图像初始化文件相同；旧基线未记录完整初始模型指纹，'
              '不能声称所有不同结构模型有完全相同初始权重。', '']
    (ANALYSIS / 'A_first_epoch_analysis.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(ANALYSIS / 'A_first_epoch_analysis.md'), csv_rows=len(rows),
                          focal_free_loss_reduction_percent=aggregates['free']['loss_reduction_percent'],
                          strict_gap=gap_summary), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    analyze()
