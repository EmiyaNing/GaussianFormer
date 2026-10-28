"""Analyze existing epoch-one evidence without using GPU or changing training."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import runpy
from zoneinfo import ZoneInfo

import numpy as np

from .metrics import CLASS_NAMES


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def analyze():
    old_root = Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e')
    new_root = Path('out/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e')
    output = Path('analysis/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e')
    old_analysis = Path('analysis/sparseworld_opus_gaussian_bias_4096_1024_15e')
    roots = dict(strict=old_root / 'sparseworld_strict', camera=old_root / 'sparseworld_opus_gaussian_bias',
                 fullspace=new_root / 'sparseworld_opus_gaussian_bias')
    config_path = Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e.py')
    cfg = runpy.run_path(str(config_path))
    audit = read(output / 'configuration_audit.json')
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != audit['new_sha256']:
        raise ValueError('full-space config changed since launch')
    if cfg['loss']['mask_type'] != 'none' or cfg['max_epochs'] != 15:
        raise ValueError('unexpected full-space training protocol')
    arms, histograms, binary = {}, {}, {}
    for name, directory in roots.items():
        arms[name] = read(directory / 'epoch_01_metrics.json')
        with np.load(directory / 'epoch_01_confusion.npz') as stored:
            histograms[name] = stored['confusion'].copy()
        if arms[name]['num_samples'] != 1024:
            raise ValueError('epoch-one comparisons require 1024 samples')
        hist = histograms[name][0]
        tp, fp, fn = int(hist[:17, :17].sum()), int(hist[17, :17].sum()), int(hist[:17, 17].sum())
        binary[name] = dict(tp=tp, fp=fp, fn=fn, tn=int(hist[17, 17]),
                            predicted_occupied_percent=100 * (tp + fp) / int(hist.sum()),
                            gt_occupied_percent=100 * (tp + fn) / int(hist.sum()))
    if any(not np.array_equal(h.sum(2), histograms['strict'].sum(2)) for h in histograms.values()):
        raise ValueError('ground-truth voxel counts differ between arms')
    init = {name: read(directory / 'initialization.json') for name, directory in roots.items()}
    if len({v['sha256'] for v in init.values()}) != 1:
        raise ValueError('image initialization differs')
    diagnostic = read(output / 'epoch_diagnostics' / 'epoch_01_diagnostics.json')
    with np.load(output / 'epoch_diagnostics' / 'epoch_01_visibility_confusion.npz') as stored:
        visibility = {k: stored[k].copy() for k in stored.files}
    if (not np.array_equal(visibility['full'], histograms['fullspace']) or
            not np.array_equal(visibility['full'], visibility['camera'] + visibility['noncamera'])):
        raise ValueError('visibility evidence is inconsistent')
    weights = np.asarray(cfg['loss']['occupancy_loss']['manual_class_weight'])
    mass = {}
    for region, hist in visibility.items():
        counts = hist[0].sum(1)
        weighted = counts * weights
        mass[region] = dict(voxels=int(counts.sum()), empty_voxels=int(counts[17]),
                            occupied_percent=100 * counts[:17].sum() / counts.sum(),
                            weighted_sum=float(weighted.sum()),
                            weighted_empty_percent=100 * weighted[17] / weighted.sum(),
                            weighted_occupied_percent=100 * weighted[:17].sum() / weighted.sum())
    coefficient = mass['camera']['weighted_sum'] / mass['full']['weighted_sum']
    target = mass['camera']['weighted_occupied_percent'] / 100
    occupied_mass = float((visibility['full'][0].sum(1) * weights)[:17].sum())
    reference_free_weight = occupied_mass * (1 - target) / (target * mass['full']['empty_voxels'])
    attrs = {}
    for h in range(7):
        records = [r for r in diagnostic['attributes'] if r['horizon'] == h]
        attrs[f'h{h}'] = {key: float(np.mean([r[key] for r in records])) for key in
                         ('in_range_percent', 'scale_upper_saturation_percent', 'opacity_upper_saturation_percent',
                          'empty_bias', 'empty_scalar')}
        attrs[f'h{h}']['feature_norm_median_across_samples'] = float(np.median([r['feature_norm_quantiles'][2] for r in records]))
    old_diagnostic = read(old_analysis / 'first_epoch_diagnostics.json')['models']['sparseworld_opus_gaussian_bias']
    old_records = {r['index']: r for r in old_diagnostic['records']}
    new_records = {r['index']: r for r in diagnostic['attributes'] if r['horizon'] == 0}
    matched = []
    for index in sorted(old_records.keys() & new_records.keys()):
        a, b = old_records[index], new_records[index]
        matched.append(dict(index=index, camera_in_range_percent=a['in_range_percent'],
                            fullspace_in_range_percent=b['in_range_percent'],
                            camera_softplus_semantic_median=a['softplus_semantic_max_quantiles'][2],
                            fullspace_softplus_semantic_median=b['softplus_semantic_max_quantiles'][2],
                            camera_point_score_median=a['valid_point_score_quantiles'][2],
                            fullspace_point_score_median=b['point_score_quantiles'][2]))
    losses = {}
    for name in ('camera', 'fullspace'):
        rows = [json.loads(line[6:]) for line in (roots[name] / 'train.log').read_text().splitlines() if line.startswith('TRAIN ')]
        epoch_rows = [r for r in rows if r['epoch'] == 1]
        losses[name] = dict(first=epoch_rows[0], last=epoch_rows[-1],
                            mean_of_logged_windows=float(np.mean([r['mean_loss'] for r in epoch_rows])))
    snapshot = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), epoch=1,
                    training_progress=read(roots['fullspace'] / 'progress.json'),
                    formal_sample_count=1024, arms=arms, binary_counts=binary,
                    ce_label_weight_mass=mass, visible_ce_coefficient_ratio=coefficient,
                    reference_free_class_weight=reference_free_weight,
                    attributes_summary=attrs, matched_attribute_samples=matched,
                    training_loss_windows=losses,
                    verification=dict(config_hash_verified=True, common_image_initialization=True,
                                      all_seven_horizon_gt_counts_identical=True,
                                      exact_visibility_partition_and_formal_histograms=True),
                    limitations=['Only epoch one is complete at this snapshot.',
                                 'CE masses are pooled held-out label-weight masses, not observed gradient shares.',
                                 'Rendering probes use 16 samples, gradients use two output-level probes.',
                                 'Changes across checkpoints do not isolate the mask effect at identical outputs.'])
    write(output / 'fullspace_first_epoch_analysis.json', snapshot)
    class_rows = []
    for h in range(7):
        for c, name in enumerate(CLASS_NAMES):
            values = {arm: arms[arm]['horizons'][h]['class_iou'][c] for arm in roots}
            class_rows.append(dict(seconds=h * .5, class_index=c, class_name=name,
                                   **{f'{arm}_iou': value for arm, value in values.items()},
                                   delta_fullspace_camera_pp=values['fullspace'] - values['camera'],
                                   delta_fullspace_strict_pp=values['fullspace'] - values['strict'],
                                   gt_voxels=arms['strict']['horizons'][h]['class_gt_count'][c]))
    with (output / 'fullspace_first_epoch_class_iou.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(3)
    colors = ['#536878', '#cc8a37', '#4a91a5']
    for ax, key, title in zip(axes.flat[:2], ('miou', 'occupied_recall'), ('Current mIoU (%)', 'Current occupied recall (%)')):
        values = [arms[arm]['horizons'][0][key] for arm in roots]
        ax.bar(x, values, color=colors)
        ax.set_xticks(x, list(roots))
        ax.set_title(title)
        for i, value in enumerate(values):
            ax.text(i, value, f'{value:.3f}', ha='center', va='bottom')
        ax.set_ylim(0, max(values) * 1.25)
    values = [binary[arm]['predicted_occupied_percent'] for arm in roots]
    ax = axes[1, 0]
    ax.bar(x, values, color=colors)
    ax.axhline(binary['strict']['gt_occupied_percent'], color='black', linestyle='--', label='GT occupied')
    ax.set_xticks(x, list(roots))
    ax.set_title('Predicted occupied voxel share (%)')
    ax.legend()
    for i, value in enumerate(values):
        ax.text(i, value, f'{value:.3f}', ha='center', va='bottom')
    ax.set_ylim(0, max(values) * 1.3)
    ax = axes[1, 1]
    fg = [mass[region]['weighted_occupied_percent'] for region in ('camera', 'full')]
    ax.bar([0, 1], fg, color='#4a91a5', label='occupied')
    ax.bar([0, 1], [100 - value for value in fg], bottom=fg, color='#d6d8da', label='free')
    ax.set_xticks([0, 1], ['camera region', 'full space'])
    ax.set_title('CE label-weight mass (%)')
    ax.legend(loc='upper right')
    for i, value in enumerate(fg):
        ax.text(i, value / 2, f'{value:.2f}%', ha='center')
    fig.tight_layout()
    fig.savefig(output / 'fullspace_first_epoch_evidence.png', dpi=180)
    fig.savefig(output / 'fullspace_first_epoch_evidence.pdf')
    plt.close(fig)

    a, b, strict = arms['camera']['horizons'][0], arms['fullspace']['horizons'][0], arms['strict']['horizons'][0]
    miou_drop = 100 * (1 - b['miou'] / a['miou'])
    recall_drop = 100 * (1 - b['occupied_recall'] / a['occupied_recall'])
    fp_drop = 100 * (1 - binary['fullspace']['fp'] / binary['camera']['fp'])
    p = snapshot['training_progress']
    lines = ['# 全空间体素监督第一轮变差的原因分析', '',
             f"记录时间：{snapshot['created_at']}。本次训练正在第{p['epoch']}轮，最近日志为{p['sample']}/4096样本。只有第1轮完整1024样本评估可供比较。", '',
             f"当前mIoU从camera-mask Gaussian的{a['miou']:.4f}%降至{b['miou']:.4f}%（相对下降{miou_drop:.2f}%）。"
             f"主要表象是严重漏检：召回下降{recall_drop:.2f}%，同时误报减少{fp_drop:.2f}%，精度上升。"
             '当前证据支持空类比例变化和渲染前景分数不足的解释；还不能据第一轮判断15轮最优性能。', '',
             '## 1. 同轮完整验证集结果', '',
             '| 模型 | 当前mIoU (%) | 占据IoU (%) | 占据precision (%) | 占据recall (%) | 预测占据比例 (%) | free IoU (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in roots:
        m = arms[arm]['horizons'][0]
        lines.append(f"| {arm} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {binary[arm]['predicted_occupied_percent']:.4f} | {m['class_iou'][17]:.4f} |")
    lines += ['', f"相同GT占据比例为{binary['strict']['gt_occupied_percent']:.4f}%。全空间模型预测99.3777%的体素为空，"
              '但这不是所有输出严格全空；free IoU上升并不意味着17个非空类mIoU变好。', '',
              '| 当前类别 | camera IoU (%) | 全空间IoU (%) | Δ pp | Strict IoU (%) |',
              '|---|---:|---:|---:|---:|']
    for c in (4, 11, 13, 14, 15, 16):
        lines.append(f"| {CLASS_NAMES[c]} | {a['class_iou'][c]:.4f} | {b['class_iou'][c]:.4f} | {b['class_iou'][c]-a['class_iou'][c]:+.4f} | {strict['class_iou'][c]:.4f} |")
    lines += ['', '道路和terrain等静态类别贡献主要降幅；car和sidewalk略有提升，因此不能声称所有类别统一变差。', '',
              '## 2. 最强解释：扩大监督范围同时改变了有效正负比例', '',
              '`OccupancyLoss`的CE仍使用按类别权重归一化的mean，free权重0.5；非空权重约0.85–1.30。CE权重10、Lovasz权重1未调整。', '',
              '| 统计范围 | GT占据比例 (%) | CE空类标签加权质量 (%) | CE非空类标签加权质量 (%) |',
              '|---|---:|---:|---:|']
    for region, m in mass.items():
        lines.append(f"| {region} | {m['occupied_percent']:.4f} | {m['weighted_empty_percent']:.4f} | {m['weighted_occupied_percent']:.4f} |")
    lines += ['', f"camera只覆盖{100 * mass['camera']['voxels'] / mass['full']['voxels']:.3f}%空间。移除mask后，"
              '大量不可见空体素进入CE，free权重0.5仍不足以抵消其数量。', '',
              '加权CE为`sum(w[y] * CE) / sum(w[y])`。在同一输出和同一可见体素上，扩大归一化分母会稀释其单体素梯度。', '',
              f"按该验证集池化标签估算，可见体素CE系数仅剩原来的{coefficient:.4f}（约1/{1/coefficient:.2f}）。"
              '这是标签分布/归一化证据，不是实际训练梯度占比，也没有包含变化后的预测误差与Lovasz。训练CE按每个batch归一化，不能把该池化比例当作所有batch的精确倍数。', '',
              '因此，原第10轮不可见区域误报较多这一判断仍成立，但“给所有体素相同的采样范围并沿用原权重”会带来新的漏检压力。监督范围正确不保证数值权重适合新的分布。', '',
              '## 3. 强空类先验、前景强度和全局bias共同形成渲染瓶颈', '',
              '本渲染器将各Gaussian的`opacity × softplus(raw_logits) × spatial_kernel`直接相加作为前景logit，没有按贡献数归一化。'
              '空类来自全局大Gaussian，之后减去一个learned bias。二者不仅比较类别排序，还比较绝对强度。', '',
              f"第1轮全空间empty_scalar={attrs['h0']['empty_scalar']:.6f}、empty_bias={attrs['h0']['empty_bias']:.6f}；"
              f"camera模型分别为{old_diagnostic['records'][0]['empty_scalar']:.6f}、{old_diagnostic['records'][0]['empty_bias']:.6f}。"
              'bias更小意味着减掉的空类分数更少；差异只有约0.137，不能独自解释全部召回下降。', '',
              '| 相同样本 | camera child softplus前景最大值中位数 | 全空间中位数 | camera点分数中位数 | 全空间点分数中位数 |',
              '|---|---:|---:|---:|---:|']
    for row in matched:
        lines.append(f"| {row['index']} | {row['camera_softplus_semantic_median']:.6f} | {row['fullspace_softplus_semantic_median']:.6f} | {row['camera_point_score_median']:.6f} | {row['fullspace_point_score_median']:.6f} |")
    lines += ['', '这些是同样本、不同已训练checkpoint的属性比较，不是把两个mask作用于同一checkpoint的因果实验。', '',
              '固定16样本当前渲染探针（保持模型权重不变）：', '',
              '| 方式 | mIoU (%) | 占据IoU (%) | recall (%) | 预测占据比例 (%) |',
              '|---|---:|---:|---:|---:|']
    for mode in ('default', 'empty_bias_delta_+1', 'empty_bias_delta_+2', 'point_rasterizer', 'fixed_scale_0_2', 'opacity_times_0_5'):
        m = diagnostic['render_probes'][f'h0/{mode}/full']
        lines.append(f"| {mode} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_recall']:.4f} | {m['predicted_occupied_percent']:.4f} |")
    lines += ['', '`empty_bias_delta_+2`代表将空类logit再降低2；它能明显恢复当前召回。点栅格化也优于默认Gaussian渲染，'
              '说明当前点/语义输出中仍有可用信号，渲染的前景–空类决策在压低输出。'
              '这些仅是小样本诊断，不能作为正式改进成绩、最优阈值或Strict比较结论。', '',
              '## 4. 属性仍饱和，不能把此次下降解释成Gaussian都变小了', '',
              f"16样本当前尺度>0.395m比例平均{attrs['h0']['scale_upper_saturation_percent']:.3f}%，"
              f"opacity>0.99比例平均{attrs['h0']['opacity_upper_saturation_percent']:.3f}%。"
              '尺度仍几乎全部靠近0.4m上限，不透明度也接近1。缩至固定0.2m后输出几乎全空，是前景幅值/覆盖依赖大尺度的表现，不是缩尺度改善性能的证据。', '',
              'sigmoid属性接近边界时局部导数变小。这是原camera实验也存在的结构性问题，本轮没有解决；仅凭该事实不能认定它造成两次训练之间的全部降幅。', '',
              f"当前ROI内child比例平均{attrs['h0']['in_range_percent']:.3f}%；共同样本0的比例从"
              f"{matched[0]['camera_in_range_percent']:.3f}%升至{matched[0]['fullspace_in_range_percent']:.3f}%。"
              '这不支持“新一轮点普遍更越界”作为主要解释；ROI内比例也不等于真实占据区域覆盖率。', '',
              '## 5. 损失与梯度需要区别看待', '',
              f"第1轮最后记录的总loss：camera {losses['camera']['last']['mean_loss']:.4f}，"
              f"全空间{losses['fullspace']['last']['mean_loss']:.4f}。监督样本分布改变后，loss降低不能直接代表非空类性能改善。", '',
              '| 样本 | 体素/集合语义梯度范数比 | 语义余弦 | 中心梯度范数比 | 中心余弦 |',
              '|---|---:|---:|---:|---:|']
    for row in diagnostic['gradient_probes']:
        lines.append(f"| {row['index']} | {row['semantic']['voxel_to_set_norm_ratio']:.3f} | {row['semantic']['cosine']:.3f} | {row['geometry']['voxel_to_set_norm_ratio']:.3f} | {row['geometry']['cosine']:.3f} |")
    lines += ['', '这是固定features/covariance/opacity后的末层raw logits和world centers梯度，覆盖2样本；整体余弦为正，不能声称两种损失全面冲突。'
              '新模型的语义范数比约5.65–6.92，也不能据此声称全空间使梯度比原camera模型更大。'
              '监督当前完整网格与预训练时间扩展集合在目标、归一化、分类方式上不同，是可能影响优化的候选因素，需要共享参数梯度或训练消融验证。', '',
              '## 6. 未来成绩和早期训练的解释边界', '',
              '第1–5轮为时间扩展预训练，实际Gaussian体素监督仅当前；未来1/2/3秒直接体素监督分别从第7/9/11轮启用。'
              '原camera模型和Strict在第1轮的未来1/2秒也接近零，不能用当前未来近零判断新方案失败。', '',
              '三组使用相同标签、图像初始化和评测口径。新旧Gaussian的配置、模型、优化器、seed、子集已审计，训练语义只改变mask；'
              '新增epoch诊断不更新参数，并已验证随机数状态及参数梯度隔离。当前证据没有指向数据集变化、加载错模型或训练数值发散。', '',
              '## 7. 建议的下一步', '',
              '1. 保留本次全空间实验和逐轮诊断，观察第2–5轮当前召回及第6轮后未来学习；第一轮表现不足以判定15轮最优性能。',
              '2. 若开展下一次消融，优先保持全部有效体素参与监督，对occupied/free分别归一化后混合，或降低free类相对权重。可进一步分可见/不可见组，防止一个组凭体素数量主导。',
              f'3. 若仅按此验证标签分布恢复原camera区域约33%的非空CE加权质量，free权重参考值约{reference_free_weight:.4f}，当前为0.5。该数值仅是设计参考，实际配置应依据训练集统计，不是已验证最优值。',
              '4. 然后单独测试Gaussian损失warm-up或课程、empty prior/learned bias初始幅值；每次只改变一个因素。小样本bias探针不直接写入训练或正式评测。',
              '5. 继续跟踪属性饱和及密度和语义幅值耦合；这些属于后续结构消融，不能通过一次mask替换解决。', '',
              '## 原始证据与实现', '',
              '- [三组完整指标、计数及审计JSON](fullspace_first_epoch_analysis.json)',
              '- [7时刻×18类别三组IoU比较CSV](fullspace_first_epoch_class_iou.csv)',
              '- [第一轮定量证据图](fullspace_first_epoch_evidence.png)',
              '- [第一轮原始诊断](epoch_diagnostics/epoch_01_diagnostics.md)',
              '- [监督范围与未来损失](../../experiments/sparseworld_opus_gaussian_bias/loss.py)',
              '- [CE与Lovasz实现](../../loss/occupancy_loss.py)',
              '- [Gaussian属性映射](../../model/encoder/opus_v1_gaussian_encoder.py)',
              '- [Gaussian局部累加CUDA实现](../../model/head/localagg/src/forward.cu)', '',
              '本分析未修改配置、模型、checkpoint或运行中的训练。']
    (output / 'fullspace_first_epoch_analysis.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str(output / 'fullspace_first_epoch_analysis.md'),
                          current_miou= b['miou'], camera_miou=a['miou'],
                          reference_free_class_weight=reference_free_weight,
                          verified=True), ensure_ascii=False))


if __name__ == '__main__':
    analyze()
