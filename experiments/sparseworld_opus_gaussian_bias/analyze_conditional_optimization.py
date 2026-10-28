"""Read saved C/B evidence and write an optimization assessment; no training."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import runpy
from zoneinfo import ZoneInfo

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / 'analysis/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e'
C_WORK = REPO / 'out/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e/C_conditional_semantic_ce'
B_WORK = REPO / 'out/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e/B_ce_free_0_1'
CONFIG = REPO / 'config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_conditional_ce_C_4096_1024_15e.py'


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def binary_counts(hist):
    return dict(tp=int(hist[:17, :17].sum()), fp=int(hist[17, :17].sum()),
                fn=int(hist[:17, 17].sum()))


def main():
    comparison = read(ANALYSIS / 'comparison.json')
    status = read(ANALYSIS / 'status.json')
    require(status['status'] == 'paused_after_first_epoch_diagnosis', 'expected paused C run')
    require(status['completed_training_epochs'] == 1, 'assessment is for epoch one only')
    audit = read(ANALYSIS / 'configuration_audit.json')
    require(hashlib.sha256(CONFIG.read_bytes()).hexdigest() == audit['config_sha256'], 'C configuration changed')
    for key in ('initial_model_sha256',):
        require(comparison['C']['initialization'][key] == comparison['B']['initialization'][key],
                'C/B model initialization differs')
    require(comparison['C']['boundary']['manifest_sha256'] == comparison['B']['boundary']['manifest_sha256'],
            'C/B subset manifests differ')

    matrices = {name: load_npz(work / 'epoch_01_confusion.npz')['confusion']
                for name, work in (('B_ce_free_0_1', B_WORK), ('C_conditional_semantic_ce', C_WORK))}
    for name, matrix in matrices.items():
        metric = comparison['metrics'][name]
        require(matrix.shape == (7, 18, 18), 'expected seven horizons and eighteen classes')
        require(metric['num_samples'] == 1024 and metric['miou_divisor'] == 17 and metric['evaluation_mask'] == 'none',
                'formal evaluation protocol differs')
        for h, hist in enumerate(matrix):
            gt = hist.sum(1)
            union = gt + hist.sum(0) - hist.diagonal()
            iou = np.divide(hist.diagonal(), union, out=np.zeros(18), where=union > 0) * 100
            require(np.array_equal(gt, metric['horizons'][h]['class_gt_count']), 'GT counts do not match metrics')
            require(np.allclose(iou, metric['horizons'][h]['class_iou'], rtol=1e-12, atol=1e-12), 'IoU reconstruction failed')
            require(np.isclose(iou[:17].mean(), metric['horizons'][h]['miou'], rtol=1e-12), 'mIoU reconstruction failed')
    b_hist, c_hist = matrices.values()
    require(np.array_equal(b_hist.sum(2), c_hist.sum(2)), 'C/B GT differs across horizons')
    visibility = load_npz(ANALYSIS / 'epoch_diagnostics/epoch_01_visibility_confusion.npz')
    require(np.array_equal(visibility['full'], c_hist), 'C regional and formal matrices differ')
    require(np.array_equal(visibility['camera'] + visibility['noncamera'], c_hist), 'visibility partition failed')
    conditional = load_npz(ANALYSIS / 'epoch_diagnostics/epoch_01_conditional_semantic_confusion.npz')
    require(np.array_equal(conditional['camera'] + conditional['noncamera'], conditional['full']), 'conditional partition failed')
    cond = conditional['full']
    require(cond.shape == (17, 17), 'expected a seventeen-class conditional matrix')
    counts = c_hist[0].sum(1)
    require(np.array_equal(cond.sum(1), counts[:17]), 'conditional predictions must cover all GT occupied voxels')

    config = runpy.run_path(str(CONFIG))
    weights = np.asarray(config['loss']['occupancy_loss']['manual_class_weight'][:17])
    diagnostic = comparison['C']['diagnostic']
    names = diagnostic['class_names']
    attributes = [row for row in diagnostic['attributes'] if row['horizon'] == 0]
    attrs = {key: float(np.mean([row[key] for row in attributes])) for key in
             ('scale_upper_saturation_percent', 'opacity_upper_saturation_percent', 'in_range_percent', 'empty_bias', 'empty_scalar')}
    records = comparison['conditional']['records']
    auxiliary = {key: float(np.mean([row[key] for row in records])) for key in
                 ('conditional_semantic_loss', 'auxiliary_to_base_ce_logit_gradient_norm_ratio', 'auxiliary_base_ce_logit_gradient_cosine')}
    b, c = (comparison['metrics'][name]['horizons'][0] for name in matrices)
    binary = {name: binary_counts(matrix[0]) for name, matrix in matrices.items()}
    b_counts, c_counts = binary.values()
    delta_tp, delta_fp = c_counts['tp'] - b_counts['tp'], c_counts['fp'] - b_counts['fp']
    evidence = dict(current_miou_gain_pp=c['miou'] - b['miou'],
                    current_miou_gain_relative_percent=100 * (c['miou'] / b['miou'] - 1),
                    delta_tp=delta_tp, delta_fp=delta_fp,
                    additional_fp_per_additional_tp=delta_fp / delta_tp,
                    fp_relative_increase_percent=100 * delta_fp / b_counts['fp'],
                    c_non_camera_fp_share_percent=100 * binary_counts(visibility['noncamera'][0])['fp'] / c_counts['fp'],
                    conditional_micro_accuracy_percent=100 * float(np.trace(cond)) / int(cond.sum()),
                    conditional_macro_class_accuracy_percent=100 * float(np.mean(cond.diagonal() / cond.sum(1))),
                    gt_occupied_percent=100 * float(counts[:17].sum()) / int(counts.sum()),
                    road_manmade_vegetation_gt_share_percent=100 * float(counts[[11, 15, 16]].sum()) / int(counts[:17].sum()),
                    road_manmade_vegetation_weighted_label_share_percent=100 * float((counts[[11, 15, 16]] * weights[[11, 15, 16]]).sum()) / float((counts[:17] * weights).sum()),
                    zero_iou_class_names=[names[i] for i in range(17) if c['class_iou'][i] == 0])
    class_rows = []
    for i in range(17):
        order = np.argsort(cond[i])[::-1][:3]
        class_rows.append(dict(class_name=names[i], gt_voxels=int(counts[i]),
                               B_iou_percent=b['class_iou'][i], C_iou_percent=c['class_iou'][i],
                               delta_iou_pp=c['class_iou'][i] - b['class_iou'][i],
                               C_conditional_recall_percent=100 * int(cond[i, i]) / int(counts[i]),
                               C_conditional_prediction_share_percent=100 * int(cond[:, i].sum()) / int(cond.sum()),
                               C_top_conditional_predictions=[dict(class_name=names[j], percent=100 * int(cond[i, j]) / int(counts[i])) for j in order]))
    probes = {key.split('/')[1]: value for key, value in diagnostic['render_probes'].items() if key.startswith('h0/') and key.endswith('/full')}
    tracked_inputs = [CONFIG, ANALYSIS / 'comparison.json', ANALYSIS / 'status.json',
                      ANALYSIS / 'configuration_audit.json', C_WORK / 'epoch_01_confusion.npz', B_WORK / 'epoch_01_confusion.npz',
                      ANALYSIS / 'epoch_diagnostics/epoch_01_visibility_confusion.npz',
                      ANALYSIS / 'epoch_diagnostics/epoch_01_conditional_semantic_confusion.npz']
    payload = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), scope='C/B epoch-one optimization assessment',
                   formal_samples=1024, attribute_and_render_samples=len(attributes), output_gradient_samples=len(diagnostic['gradient_probes']),
                   evidence=evidence, binary_counts=binary, class_conditionals=class_rows,
                   attributes=attrs, auxiliary=auxiliary, render_probes_16_samples=probes,
                   proposed_experiments_not_executed=True, training_remains_paused=True,
                   verification=dict(config_hash=True, same_initial_model=True, same_subset_manifests=True,
                                     seven_horizon_ious_recomputed=True, same_all_horizon_gt=True,
                                     exact_visibility_partition=True, conditional_covers_all_gt_occupied=True),
                   input_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked_inputs})
    (ANALYSIS / 'C_optimization_analysis.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    lines = [
        '# C 条件语义 CE：进一步优化空间与实验顺序', '',
        '有进一步提升空间。第一轮结果支持继续验证条件语义监督，但还没有证明完整训练后的优势或多 seed 稳定性。当前 C 仍暂停在第一轮，本次只读取已有结果，新增分析文件。', '',
        '## 同轮结果与主要限制', '',
        '| 第一轮模型 | 当前 mIoU % | 占用 IoU % | precision % | recall % |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, label in [('Strict', 'Strict'), ('B_ce_free_0_1', 'B：free=0.1'), ('C_conditional_semantic_ce', 'C：B+17类条件CE')]:
        m = comparison['metrics'][name]['horizons'][0]
        lines.append(f"| {label} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} |")
    lines += ['',
        f"C 比 B 的当前 mIoU 高 {evidence['current_miou_gain_pp']:.4f} 个百分点，相对提升 {evidence['current_miou_gain_relative_percent']:.2f}%。但新增 TP {delta_tp:,}，新增 FP {delta_fp:,}，每增加一个正确占用检出对应约 {evidence['additional_fp_per_additional_tp']:.2f} 个新增误报；FP 相对增加 {evidence['fp_relative_increase_percent']:.2f}%。这是 B/C 聚合差值，不是逐体素转移追踪。",
        f"C 的误报有 {evidence['c_non_camera_fp_share_percent']:.2f}% 位于 noncamera 区域；该区域 precision 仅 6.5759%。不能只靠整体增加前景分数追求召回。", '',
        f"17 类正式 mIoU 中有 {len(evidence['zero_iou_class_names'])} 类仍为零。C 全部 GT 非空体素的条件分类 micro accuracy 为 {evidence['conditional_micro_accuracy_percent']:.2f}%，先按类别计算召回再平均的 macro accuracy 只有 {evidence['conditional_macro_class_accuracy_percent']:.2f}%。这两个准确率都不是 mIoU，也不是只统计已经预测成非空的体素。B 没有保存同口径的完整条件矩阵，不能声称这两个指标比 B 提升了多少。",
        f"评估集道路、manmade、vegetation 占 GT 非空体素的 {evidence['road_manmade_vegetation_gt_share_percent']:.2f}%，按当前辅助 CE 权重计算仍占标签权重质量的 {evidence['road_manmade_vegetation_weighted_label_share_percent']:.2f}%。这不是观测到的实际 loss 或梯度占比；训练集类别频率仍需另行统计。", '',
        '| 类别 | B IoU % | C IoU % | Δ pp | C 全部 GT 非空条件类别召回 % |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in class_rows:
        lines.append(f"| {row['class_name']} | {row['B_iou_percent']:.4f} | {row['C_iou_percent']:.4f} | {row['delta_iou_pp']:+.4f} | {row['C_conditional_recall_percent']:.4f} |")
    lines += ['',
        'car 的占用召回从 10.501% 升至 12.678%，但命中后类别正确率从 41.898% 降至 29.835%。全部 GT car 的条件预测中，29.10% 被分成道路、22.38% 被分成 manmade。收益并不是所有类别都变得更容易区分；大面积静态类别收益与小目标语义混淆同时存在。', '',
        '## 优先方向一：让条件语义监督更符合按类别平均的评价目标', '',
        '当前辅助 CE 对 GT 非空体素按已有类别权重求平均，权重约 0.85–1.30，仍容易由大类别决定优化方向。建议先统计 4096 训练样本中的各类体素数、出现帧数与场景数，然后只改变辅助项的类别平衡方式。原 18 类全空间 CE、free=0.1、Lovasz 和集合/时序损失保持一致。', '',
        '首选一个有限幅度的类内平均、再类间平均方案，或按训练频率计算并截断权重的条件 CE。一个可验证的类间平均目标为：', '',
        '$$L_{sem}^{macro}=\frac{\sum_{c\in C_{present}}a_c\left(\frac{1}{N_c}\sum_{v:y_v=c}-\log r_c(v)\right)}{\sum_{c\in C_{present}}a_c}.$$','',
        '其中 r 是 17 类条件 softmax，C_present 只包含当前批次实际出现的 GT 类。小类别只有少数体素时应限制权重，避免单个噪声标签支配梯度；不直接使用无上限的 1/N_c。权重必须来自训练集，不从 1024 评估样本反推。这里的验证集统计用于定位问题。', '',
        '有效样本数重加权可作为另一种候选，其类别平衡依据来自 [Class-Balanced Loss](https://arxiv.org/abs/1901.05555)。这是将分类论文的方法迁移到当前体素任务的建议，尚未在本模型验证。', '',
        '若后续尝试 focal，建议只替换这条 17 类辅助 CE，并固定其余项。Focal 对容易样本降权、强调难例的机制见 [Focal Loss](https://arxiv.org/abs/1708.02002)。先以较小 gamma=1 做单因素对照；A 的较差结果不能证明条件语义 focal 无效，A 同时使用了不同的 free 权重和全 18 类 focal。', '',
        '## 优先方向二：占用判断和语义判别进一步解耦', '',
        '辅助项对 free logit 的直接梯度为零，但当前仍用 18 类 argmax。占用判定实际是 max(z_0,…,z_16)>z_free。条件 CE 会改变前景最大分数；前景还通过 Gaussian 语义、中心、尺度、opacity 和共享特征间接影响占用。因此“辅助项不直接更新 free”不等于“不会增加误报”。', '',
        '先做低成本离线校准诊断：绘制完整 precision/recall/mIoU 随占用阈值变化的曲线，分别看 camera/noncamera、距离与类别。用训练集内部留出的样本确定候选参数，随后在固定 1024 评估子集报告。现有 16 样本探针只能定位敏感性，不能直接挑出最终阈值。', '',
        '中期建立独立的占用/密度输出，条件语义输出只负责 17 类分布；由二元全空间占用损失监督空与非空，再由条件 CE 监督非空类别。两个任务可共享特征，但分数参数与决策应明确分开。简单用 logsumexp 前景替换 max 会引入类别数和分布的影响，需要校准与训练验证。', '',
        '仓库 GaussianHead 已有 localaggprob 分支，但当前 GaussianOPUSHead 明确禁止该模式，不能只改一个配置开关；若实施，需要新 renderer/model/loss 文件并验证 batch、概率/损失契约和时序输出。', '',
        '## 优先方向三：改善高斯属性的可学习性和几何覆盖', '',
        f"固定 16 样本中，scale>.395 的比例均值为 {attrs['scale_upper_saturation_percent']:.2f}%，opacity>.99 为 {attrs['opacity_upper_saturation_percent']:.2f}%；中心在空间范围内的比例均值为 {attrs['in_range_percent']:.2f}%。接近 sigmoid 上界会降低对应 raw 参数的局部梯度，也可能让网络依靠前景幅值补偿几何覆盖。饱和是需要验证的优化限制，不是已证实的性能差原因。", '',
        '建议先记录尺度/opacity raw 值、各属性分支参数梯度、实际学习率、裁剪前总梯度范数以及各类 GT 到高斯中心/有效支撑域的距离。已有探针只覆盖末层语义和中心，不代表共享参数，更没有证明全局梯度冲突。', '',
        '随后分别验证属性分支较小学习率、适度的尺度/opacity 正则，或保持上界约束但减缓饱和的参数化。当前渲染空间支撑和前景幅值耦合，必须同步检查检出率，不能直接减半尺度或 opacity。中心/Query 覆盖若偏向大面积类别，可在总点数预算固定时验证按目标分配 child 的方案。', '',
        '| 固定16样本探针 | 当前 mIoU % | recall % | FP |',
        '|---|---:|---:|---:|',
    ]
    for name in ('default', 'point_rasterizer', 'empty_bias_delta_+1', 'empty_bias_delta_+2', 'fixed_scale_0_2', 'opacity_times_0_5'):
        row = probes[name]
        lines.append(f"| {name} | {row['miou']:.6f} | {row['occupied_recall']:.4f} | {row['occupied_fp']:,} |")
    lines += ['',
        'bias +1 到 +2 仅带来约 0.046 pp 的额外 mIoU，却明显增加误报；固定 scale=0.2 或 opacity×0.5 几乎清空预测。点栅格化也弱于默认高斯渲染。这些结果支持协调密度、分数和几何，而不是简单扩大 bias、压缩支撑或移除高斯渲染。', '',
        '## 优先方向四：辅助项强度与时序训练', '',
        f"固定 16 样本中辅助/原18类CE输出梯度范数比均值为 {auxiliary['auxiliary_to_base_ce_logit_gradient_norm_ratio']:.4f}，余弦均值为 {auxiliary['auxiliary_base_ce_logit_gradient_cosine']:.4f}。辅助项与原分类项总体同向，但 car 下降意味着不能据此无限加大 lambda。候选 lambda=0.5/1/2 应各自独立比较，同时报告 car、小类别与误报。", '',
        '第一轮只对当前时刻施加 Gaussian 体素监督。原课程前5轮是时间扩展预训练，训练预测时域在第6轮后增加；由于 Gaussian 监督只选 h=0/2/4/6，第7/9/11轮才分别直接监督1/2/3秒。因此第一轮的未来近零结果不能用于判断时序优化方向。', '',
        '应先完成 B/C 同预算15轮的曲线和每轮诊断，再验证未来条件语义的权重/渐进启用；必要时才尝试有限步反传、当前与未来 Gaussian 属性的适配。原预测过程在跨时刻点和局部历史特征处有 detach，改变它会增加显存和改变优化问题，需要独立对照。', '',
        '## 建议的实际顺序与判定标准', '',
        '1. 保留 C 为基线，优先确认 B/C 完整15轮的收益能否持续，尤其第7/9/11轮未来体素监督开启之后；运行结束后做匹配预算对照，不用 C 第一轮对比 Strict 第十四轮。',
        '2. 下一项新的第一轮筛选优先做“仅辅助17类CE的温和类别平衡”；初始化、4096/1024数据、15轮学习率计划和其他损失保持一致。',
        '3. 单独评估辅助权重，或只在辅助分支评估条件 focal，避免同时改 free、lambda、类别权重和渲染属性。',
        '4. 离线校准并建立占用/语义分开输出的对照，重点控制 noncamera 误报；保持全空间空体素监督。',
        '5. 根据属性梯度与覆盖诊断再调整参数化、Query 分配和未来训练课程。', '',
        '正式报告至少包含当前/1/2/3秒 mIoU 和占用 IoU、7时刻×18类IoU、各类占用召回/条件正确率、camera/noncamera FP/FN、17类条件混淆与宏平均准确率。候选方案通过第一轮筛选后还需相同15轮预算验证；关键结论再做多个随机种子，计算稳定性。', '',
        '本次仅有一个 seed、一个 epoch；属性/渲染16样本、输出梯度2样本。已使用1024评估子集进行多次方案诊断；若结论要作为最终泛化报告，还需独立留出数据。本文建议没有执行任何新训练，也没有修改已有训练代码、配置或结果。', '',
        '- [重算证据和输入SHA256](C_optimization_analysis.json)',
        '- [第一轮正式报告](report.md)',
        '- [完整每时刻每类别IoU](per_horizon_per_class_iou.csv)',
        '- [原属性/渲染/末层梯度诊断](epoch_diagnostics/epoch_01_diagnostics.md)',
    ]
    report = ANALYSIS / 'C_optimization_analysis.md'
    report.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str(report.relative_to(REPO)), evidence=evidence, verification=payload['verification']), ensure_ascii=False))


if __name__ == '__main__':
    main()
