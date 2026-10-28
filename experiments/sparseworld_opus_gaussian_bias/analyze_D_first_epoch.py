"""Audit D's epoch-one improvement from saved evidence, without training."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import runpy
from zoneinfo import ZoneInfo

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / 'analysis/sparseworld_opus_gaussian_bias_semantic_occupancy_DE_4096_1024_15e'
C_ANALYSIS = REPO / 'analysis/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e'
C_WORK = REPO / 'out/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e/C_conditional_semantic_ce'
D_WORK = REPO / 'out/sparseworld_opus_gaussian_bias_semantic_occupancy_DE_4096_1024_15e/D_balanced_conditional_ce'
D_ARM = 'D_balanced_conditional_ce'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binary(hist):
    return dict(tp=int(hist[:17, :17].sum()), fp=int(hist[17, :17].sum()),
                fn=int(hist[:17, 17].sum()))


def check_gradient_identity():
    """Finite differences verify the weighted-mean formula, including free."""
    logits = np.random.default_rng(17).normal(size=(6, 18))
    labels = np.array([4, 11, 13, 10, 17, 255])
    weights = np.linspace(.4, 3., 17)
    selected = labels < 17

    def objective(value):
        foreground = value[selected, :17]
        maximum = foreground.max(1, keepdims=True)
        logp = foreground - maximum - np.log(np.exp(foreground - maximum).sum(1, keepdims=True))
        return -(logp[np.arange(selected.sum()), labels[selected]] * weights[labels[selected]]).sum() / weights[labels[selected]].sum()

    foreground = logits[selected, :17]
    probability = np.exp(foreground - foreground.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    probability[np.arange(selected.sum()), labels[selected]] -= 1
    gradient = np.zeros_like(logits)
    gradient[selected, :17] = probability * (weights[labels[selected]] / weights[labels[selected]].sum())[:, None]
    numerical = np.zeros_like(logits)
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numerical[index] = (objective(plus) - objective(minus)) / 2e-5
    require(np.allclose(gradient, numerical, atol=1e-9, rtol=1e-7), 'weighted-mean gradient formula failed')
    require(np.all(gradient[:, 17] == 0) and np.all(gradient[~selected] == 0), 'auxiliary must exclude free/ignored targets')
    return float(np.max(np.abs(gradient - numerical)))


def main():
    hashes = {}

    def read(path):
        payload = path.read_bytes()
        hashes[str(path.relative_to(REPO))] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    def npz(path):
        hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    comparison = read(ANALYSIS / 'comparison.json')
    status = read(ANALYSIS / 'status.json')
    final = read(ANALYSIS / 'final_verification.json')
    audit = read(ANALYSIS / 'configuration_audit.json')
    c_comparison = read(C_ANALYSIS / 'comparison.json')
    require(status['status'] == 'paused_after_first_epoch_diagnosis', 'expected paused D/E runs')
    require(status['completed_training_epochs_per_arm'] == 1 and final['status'] == 'passed', 'expected verified epoch-one results')
    require(comparison['C_reference']['exact_formal_confusion_matches_C'], 'C re-evaluation did not reproduce the original')
    require(comparison['arms'][D_ARM]['initialization']['initial_model_sha256'] ==
            c_comparison['C']['initialization']['initial_model_sha256'], 'C/D initialization differs')
    require(comparison['arms'][D_ARM]['boundary']['manifest_sha256'] ==
            c_comparison['C']['boundary']['manifest_sha256'], 'C/D subset manifests differ')

    configs = {}
    for label, key in [('C', 'C_reference_eval'), ('D', D_ARM)]:
        entry = audit['arms'][key]
        path = REPO / (entry['source_config'] if label == 'C' else entry['config'])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == entry['source_sha256' if label == 'C' else 'config_sha256'], 'configuration hash differs')
        hashes[str(path.relative_to(REPO))] = actual
        configs[label] = runpy.run_path(str(path))
    c_loss, d_loss = (configs[label]['loss']['occupancy_loss'] for label in ('C', 'D'))
    weights = np.asarray(c_loss['manual_class_weight'][:17], dtype=float)
    require(c_loss['manual_class_weight'] == d_loss['manual_class_weight'], 'base CE weights changed')
    require(c_loss['conditional_semantic_weight'] == d_loss['conditional_semantic_weight'] == 1., 'auxiliary coefficient changed')
    frequency = read(ANALYSIS / 'training_class_frequencies_mean_balance.json')
    names = frequency['class_names']
    counts = np.asarray(frequency['voxel_counts'][:17], dtype=np.int64)
    multipliers = np.asarray(frequency['auxiliary_multipliers'], dtype=float)
    require(frequency['split'] == 'train' and frequency['horizons'] == [0] and frequency['num_samples'] == 4096,
            'class frequencies must come from the current training subset only')
    require(np.allclose(multipliers, np.clip(np.sqrt(counts.mean() / counts), .5, 3.), atol=1e-12), 'frequency weighting formula differs')
    require(np.array_equal(multipliers, d_loss['conditional_class_multipliers']), 'configured multipliers differ')
    mean_multiplier = float(np.sum(counts * weights * multipliers) / np.sum(counts * weights))
    masses = {label: counts * weights * a for label, a in [('C', np.ones(17)), ('D', multipliers)]}
    for label in masses:
        masses[label] = masses[label] / masses[label].sum()

    matrices, visibility, conditional, metrics = {}, {}, {}, {}
    for label, work, directory, metric_key in [
        ('C', C_WORK, C_ANALYSIS, 'C_conditional_semantic_ce'),
        ('D', D_WORK, ANALYSIS / D_ARM, D_ARM),
    ]:
        matrix = npz(work / 'epoch_01_confusion.npz')['confusion']
        metric = comparison['metrics'][metric_key]
        require(matrix.shape == (7, 18, 18) and metric['num_samples'] == 1024 and
                metric['miou_divisor'] == 17 and metric['evaluation_mask'] == 'none', 'formal evaluation protocol differs')
        for index, hist in enumerate(matrix):
            gt = hist.sum(1)
            union = gt + hist.sum(0) - hist.diagonal()
            iou = np.divide(hist.diagonal(), union, out=np.zeros(18), where=union > 0) * 100
            require(np.allclose(iou, metric['horizons'][index]['class_iou'], atol=1e-12, rtol=1e-12), 'IoU reconstruction failed')
            require(np.isclose(iou[:17].mean(), metric['horizons'][index]['miou'], atol=1e-12, rtol=1e-12), 'mIoU reconstruction failed')
        regions = npz(directory / 'epoch_diagnostics/epoch_01_visibility_confusion.npz')
        cond = npz(directory / 'epoch_diagnostics/epoch_01_conditional_semantic_confusion.npz')
        require(np.array_equal(regions['full'], matrix) and np.array_equal(regions['camera'] + regions['noncamera'], matrix), 'formal region partition differs')
        require(np.array_equal(cond['camera'] + cond['noncamera'], cond['full']) and
                np.array_equal(cond['full'].sum(1), matrix[0].sum(1)[:17]), 'conditional GT counts differ')
        require(np.all(cond['full'].diagonal() >= matrix[0].diagonal()[:17]), 'conditional correct predictions must include formal correct predictions')
        matrices[label], visibility[label], conditional[label] = matrix, regions, cond
        metrics[label] = metric['horizons'][0]
    require(np.array_equal(matrices['C'].sum(2), matrices['D'].sum(2)), 'C/D target counts differ')

    class_rows = []
    tp_effects, fp_effects = [], []
    for i, name in enumerate(names[:17]):
        row = dict(class_name=name, train_gt_voxels=int(counts[i]), multiplier=float(multipliers[i]),
                   pooled_train_coefficient_ratio_D_over_C=float(multipliers[i] / mean_multiplier),
                   C_train_target_weight_mass_percent=float(100 * masses['C'][i]),
                   D_train_target_weight_mass_percent=float(100 * masses['D'][i]))
        gt = int(matrices['C'][0][i].sum())
        for label in ('C', 'D'):
            hist, cond = matrices[label][0], conditional[label]['full']
            tp, predicted = int(hist[i, i]), int(hist[:, i].sum())
            hit = int(hist[i, :17].sum())
            row[label] = dict(iou_percent=metrics[label]['class_iou'][i], tp=tp, fp=predicted-tp,
                              fp_on_gt_free=int(hist[17, i]), fn=gt-tp,
                              occupied_recall_percent=100*hit/gt,
                              correct_given_occupied_hit_percent=100*tp/max(hit, 1),
                              conditional_correct=int(cond[i, i]), conditional_recall_percent=100*int(cond[i, i])/gt,
                              conditionally_correct_but_formally_free=int(cond[i, i])-tp,
                              top_conditional_predictions=[dict(class_name=names[j], percent=100*int(cond[i, j])/gt)
                                  for j in np.argsort(cond[i])[::-1][:4]])
        c, d = row['C'], row['D']
        denom_c, denom_d = gt+c['fp'], gt+d['fp']
        # Exact symmetric accounting of I = TP / (GT + FP), not a causal intervention.
        tp_effect = 50*(d['tp']-c['tp'])*(1/denom_c+1/denom_d)
        fp_effect = 50*(d['tp']+c['tp'])*(1/denom_d-1/denom_c)
        delta = d['iou_percent']-c['iou_percent']
        require(np.isclose(tp_effect+fp_effect, delta, atol=1e-12), 'IoU decomposition failed')
        row.update(delta_iou_pp=delta, contribution_to_delta_miou_pp=delta/17,
                   tp_change_contribution_to_iou_pp=tp_effect, fp_change_contribution_to_iou_pp=fp_effect)
        tp_effects.append(tp_effect)
        fp_effects.append(fp_effect)
        class_rows.append(row)

    records = {
        'C': read(C_ANALYSIS / 'epoch_diagnostics/epoch_01_conditional_semantics.json')['records'],
        'D': read(ANALYSIS / D_ARM / 'epoch_diagnostics/epoch_01_conditional_semantics.json')['records'],
    }
    require([r['index'] for r in records['C']] == [r['index'] for r in records['D']] and len(records['D']) == 16,
            'diagnostic samples differ')
    counterfactual, normalization_ratios = [], []
    for record in records['D']:
        n = np.asarray([row['gt_voxels'] for row in record['classes']])
        wd = np.asarray([row['auxiliary_weight'] for row in record['classes']])
        contributions = np.asarray([row['auxiliary_loss_contribution'] for row in record['classes']])
        denominator_d = record['label_weight_sum']
        require(np.allclose(wd, weights*multipliers, rtol=2e-7), 'diagnostic weights differ')
        require(np.isclose(denominator_d, np.sum(n*wd), rtol=2e-7), 'diagnostic denominator reconstruction failed')
        require(np.isclose(contributions.sum(), record['conditional_semantic_loss'], atol=1e-6), 'class loss contributions do not sum')
        # Saved class CE sums are sufficient for scalar loss; they are not sufficient for gradient norms.
        ce_sums = contributions * denominator_d / wd
        counterfactual.append(float(np.sum(weights*ce_sums) / np.sum(n*weights)))
        normalization_ratios.append(float(denominator_d / np.sum(n*weights)))
    aux_keys = ('conditional_semantic_loss', 'auxiliary_to_base_ce_logit_gradient_norm_ratio',
                'auxiliary_base_ce_logit_gradient_cosine')
    auxiliary = {label: {key: float(np.mean([r[key] for r in values])) for key in aux_keys}
                 for label, values in records.items()}
    c_attrs = read(C_ANALYSIS / 'C_optimization_analysis.json')['attributes']
    d_attr_rows = [r for r in comparison['arms'][D_ARM]['diagnostic']['attributes'] if r['horizon'] == 0]
    d_attrs = {key: float(np.mean([r[key] for r in d_attr_rows])) for key in c_attrs}
    binary_counts = {label: binary(matrix[0]) for label, matrix in matrices.items()}
    region_counts = {label: {region: binary(data[region][0]) for region in ('camera', 'noncamera')}
                     for label, data in visibility.items()}
    delta_binary = {key: binary_counts['D'][key]-binary_counts['C'][key] for key in ('tp', 'fp', 'fn')}
    common = [11, 15, 16]
    statistics = dict(
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        scope='Epoch one, current horizon; 4096 training / 1024 full-space evaluation; one seed; no training advanced.',
        metrics=metrics, paired_scene_bootstrap=comparison['judgements'][D_ARM], binary_counts=binary_counts,
        binary_delta_D_minus_C=delta_binary, region_binary_counts=region_counts,
        false_positive_relative_change_percent=100*delta_binary['fp']/binary_counts['C']['fp'],
        mean_train_weighted_multiplier=mean_multiplier,
        pooled_train_common_target_weight_mass_percent={label: float(100*masses[label][common].sum()) for label in masses},
        class_rows=class_rows, auxiliary_probe_16_samples=auxiliary, attributes_probe_16_samples={'C': c_attrs, 'D': d_attrs},
        same_D_outputs_C_auxiliary_weights_loss_mean=float(np.mean(counterfactual)),
        probe_denominator_D_over_C=dict(mean=float(np.mean(normalization_ratios)), minimum=min(normalization_ratios), maximum=max(normalization_ratios)),
        macro_iou_symmetric_accounting_pp=dict(tp=float(np.mean(tp_effects)), fp=float(np.mean(fp_effects))),
        verification=dict(all_seven_horizon_metrics_reconstructed=True, identical_GT=True,
                          conditional_and_visibility_partitions=True, preserved_configs_verified=True,
                          identical_initialization_and_manifests=True, scalar_counterfactual_reconstructed=True,
                          analytic_gradient_max_abs_error=check_gradient_identity()),
        training_remains_paused=True, input_sha256=hashes,
    )
    report = build_report(statistics)
    (ANALYSIS / 'D_first_epoch_analysis.json').write_text(json.dumps(statistics, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (ANALYSIS / 'D_first_epoch_analysis.md').write_text(report)
    print(json.dumps(dict(report=str(ANALYSIS/'D_first_epoch_analysis.md'), verification=statistics['verification'],
                          delta_miou_pp=metrics['D']['miou']-metrics['C']['miou']), ensure_ascii=False, indent=2))


def build_report(data):
    c, d = data['metrics']['C'], data['metrics']['D']
    judgement = data['paired_scene_bootstrap']
    rows = data['class_rows']
    lines = ['# D第一轮提升的原因分析', '',
        'D在C的17类条件语义CE上加入训练频率平衡，当前mIoU提高主要对应两种变化：中小类语义识别更好，以及部分大类别在GT空体素上的误报减少。占用召回有所下降，几何饱和没有缓解。', '',
        '正式指标使用同一1024样本的全空间混淆矩阵。C/D同初始化、seed=17、子集和15轮学习率计划，各只完成第一轮1024次optimizer更新；以下不是15轮最优结果。属性和梯度探针只覆盖固定16样本。', '',
        '## 完整评估结果', '', '| 指标 | C % | D % | 变化 pp |', '|---|---:|---:|---:|']
    for key, label in [('miou', '当前17类mIoU'), ('occupied_iou', '二元占用IoU'),
                       ('occupied_precision', '占用precision'), ('occupied_recall', '占用recall')]:
        lines.append(f'| {label} | {c[key]:.6f} | {d[key]:.6f} | {d[key]-c[key]:+.6f} |')
    interval = judgement['Bonferroni_two_routes_CI97_5_pp']
    lines += ['', f"mIoU相对提升{judgement['relative_miou_percent']:.2f}%。150个场景、2000次配对bootstrap，两路线调整区间为[{interval[0]:+.4f}, {interval[1]:+.4f}] pp。区间仅反映评估数据重采样变化，不能代替多训练seed验证。", '',
        '## 1. 条件语义CE仍受非空类别频率不均衡影响', '',
        'C已经排除了GT空类对辅助语义项的直接竞争，但条件CE仍然按体素加权平均。在4096训练样本的当前GT中，道路、manmade、vegetation占非空体素77.50%；原辅助项中这三类的目标权重质量占75.82%。因此，把18类竞争改成17类竞争，并没有解决17类内部的不均衡。', '',
        'D只将辅助项权重由w_c改成w_c a_c：', '',
        r'\[a_c=\operatorname{clip}(\sqrt{\bar f/f_c},0.5,3),\qquad L_{aux}=\frac{\sum_{v\in GT_{occ}}w_{y_v}a_{y_v}[-\log r_v(y_v)]}{\sum_{v\in GT_{occ}}w_{y_v}a_{y_v}}.\]', '',
        'f只来自4096训练样本的当前帧。原18类全空间CE、free权重0.1、CE系数10、Lovasz系数1、辅助lambda=1、模型结构和推理规则保持一致；D没有新增参数，也没有使用Focal Loss。', '',
        '这三类在辅助项的训练目标权重质量降至53.39%。这里的“质量”是频率×类别权重之和占比，不等于实际梯度范数或实际损失占比。它说明监督预算分配发生了变化，方向更符合17类等权平均的mIoU。', '',
        '## 2. 加权平均的分母也变了，不能只看乘数', '',
        r'\[\frac{\partial L_{aux}}{\partial z_{v,k}}=\frac{w_{y_v}a_{y_v}}{\sum_{u\in GT_{occ}}w_{y_u}a_{y_u}}(r_v(k)-\mathbf{1}_{k=y_v}).\]', '',
        '在相同输出与标签上，D/C梯度前系数比为a_y/μ，μ=Σw_y a_y/Σw_y。按整个训练集直方图合并计算，μ=0.714213；这不是训练时每个batch的实际分母，实际系数随batch组成变化。固定16个评估探针的μ范围为0.564944～0.928383。', '',
        '| 类别 | 辅助权重乘数a | 合并训练直方图的D/C梯度前系数比 |', '|---|---:|---:|']
    for i in [3, 4, 10, 11, 13, 14, 15, 16]:
        r = rows[i]
        lines.append(f"| {r['class_name']} | {r['multiplier']:.6f} | {r['pooled_train_coefficient_ratio_D_over_C']:.6f} |")
    lines += ['', 'sidewalk和terrain的原始乘数虽小于1，归一化后的相对系数仍可提高；这解释了D不只是给极少数类加权。上述比值只描述同输出下的梯度前系数，不能解释为最终梯度范数或IoU同比增加。', '',
        '## 3. 语义区分确实改善，且收益集中在部分中小类', '',
        '条件17类指标覆盖所有GT非空体素，先在17个非空logit中取argmax，包括正式18类预测为空的体素。条件micro准确率63.7323%→63.6361%，macro类别召回18.7814%→20.2107%。整体micro略降、按类别平均的指标提高，与类别再平衡的预期一致。', '',
        '| 类别 | C条件17类召回 % | D条件17类召回 % | C正式IoU % | D正式IoU % | ΔIoU pp |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['class_name']} | {r['C']['conditional_recall_percent']:.4f} | {r['D']['conditional_recall_percent']:.4f} | {r['C']['iou_percent']:.6f} | {r['D']['iou_percent']:.6f} | {r['delta_iou_pp']:+.6f} |")
    lines += ['', 'sidewalk误判为道路的比例从42.1978%降至32.7945%，自身条件召回12.5509%→18.9590%；terrain误判为道路25.5959%→18.3172%，自身条件召回49.6369%→54.3679%。car条件召回29.5582%→35.3577%，truck从接近0升至9.9819%。这些混淆矩阵变化支持“纠正了部分大类吸收其他类别”的解释。', '',
        '正式IoU为零的非空类从11个减少为7个，新增正IoU的是barrier、bus、truck、other_flat；barrier和other_flat数值仍很小。最大的IoU收益是sidewalk +0.8228 pp、道路 +0.5908 pp、terrain +0.4910 pp；manmade下降0.3957 pp。不能把提升全部归于极少数类。', '',
        '## 4. 空体素误报下降，但部分中小类误报反而增加', '',
        '| 全空间二元占用计数 | C | D | Δ |', '|---|---:|---:|---:|']
    for key, label in [('tp', 'GT非空预测非空 TP'), ('fp', 'GT空预测非空 FP'), ('fn', 'GT非空预测空 FN')]:
        lines.append(f"| {label} | {data['binary_counts']['C'][key]:,} | {data['binary_counts']['D'][key]:,} | {data['binary_delta_D_minus_C'][key]:+,} |")
    lines += ['', f"误报减少{abs(data['false_positive_relative_change_percent']):.2f}%，同时多漏检274,359个体素。这是提升precision、牺牲少量recall的取舍。camera误报减少1,250,775，noncamera误报减少3,236,503；这些体素来自累积评估网格，并非独立样本。", '',
        '| 类别 | C正确分类TP | D正确分类TP | C的GT空误报 | D的GT空误报 |', '|---|---:|---:|---:|---:|']
    for i in [3, 4, 10, 11, 13, 14, 15, 16]:
        r = rows[i]
        lines.append(f"| {r['class_name']} | {r['C']['tp']:,} | {r['D']['tp']:,} | {r['C']['fp_on_gt_free']:,} | {r['D']['fp_on_gt_free']:,} |")
    lines += ['', '道路、manmade、vegetation在GT空区域的误报明显减少；car、sidewalk、terrain误报却增加。后者的正确分类TP增长仍足以抵消误报增长。因此D不是所有类别都更保守，也不能只用总FP解释mIoU。', '',
        '对于每类IoU=TP/(GT+FP)，GT固定。下表对TP和全部该类FP变化做对称代数分解（两种替换顺序取平均），两列之和严格等于IoU变化；它是指标记账，不是训练因果分解。', '',
        '| 类别 | TP变化贡献 pp | FP变化贡献 pp | ΔIoU pp |', '|---|---:|---:|---:|']
    for i in [4, 11, 13, 14, 15, 16]:
        r = rows[i]
        lines.append(f"| {r['class_name']} | {r['tp_change_contribution_to_iou_pp']:+.6f} | {r['fp_change_contribution_to_iou_pp']:+.6f} | {r['delta_iou_pp']:+.6f} |")
    lines += ['', '道路虽然正确分类TP减少104,602，FP下降仍推动IoU提高；vegetation同样由FP减少抵消TP减少。sidewalk和terrain则靠TP增加取得收益。按17类平均，TP变化对应+0.123274 pp，FP变化对应+0.030188 pp；不能把二元占用FP的大幅下降直接等同于全部mIoU提升。', '',
        'D没有改变辅助项对空类logit的直接梯度：该梯度仍为0。但是正式预测仍是18类argmax，类别平衡会通过语义幅值、共享特征和几何学习间接影响空/非空竞争。现有统计支持这种间接耦合解释，尚不能单独量化每条路径。', '',
        '## 5. 不是通过解除几何饱和实现的', '',
        '| 固定16样本诊断 | C | D |', '|---|---:|---:|']
    for key, label in [('scale_upper_saturation_percent', 'scale上界饱和 %'), ('opacity_upper_saturation_percent', 'opacity上界饱和 %'),
                       ('in_range_percent', '合法高斯中心 %'), ('empty_bias', 'empty bias')]:
        lines.append(f"| {label} | {data['attributes_probe_16_samples']['C'][key]:.6f} | {data['attributes_probe_16_samples']['D'][key]:.6f} |")
    for key, label in [('auxiliary_to_base_ce_logit_gradient_norm_ratio', '辅助/原18CE输出梯度范数比'),
                       ('auxiliary_base_ce_logit_gradient_cosine', '两输出梯度cosine')]:
        lines.append(f"| {label} | {data['auxiliary_probe_16_samples']['C'][key]:.6f} | {data['auxiliary_probe_16_samples']['D'][key]:.6f} |")
    lines += ['', 'scale/opacity仍高度饱和，合法中心比例没有提高。empty bias略升，而renderer使用z_free−bias，单看该变化通常更利于预测非空；实际总误报却下降，因此不能把收益简单归因于bias降低或几何参数恢复梯度。', '',
        '辅助/原18CE输出梯度范数比从0.197897升至0.310799，说明在各自训练后输出上，辅助语义信号相对更强；cosine仍为正但下降。探针排除了Lovasz、集合损失和共享参数梯度，不能据此宣称参数梯度冲突已解决，也不能把差异全部归于权重。', '',
        f"同一D输出用D权重算辅助CE，16样本均值为{data['auxiliary_probe_16_samples']['D']['conditional_semantic_loss']:.6f}；仅改回C权重，按保存的逐类CE统计重构为{data['same_D_outputs_C_auxiliary_weights_loss_mean']:.6f}。C输出的原辅助CE是{data['auxiliary_probe_16_samples']['C']['conditional_semantic_loss']:.6f}。因此D日志中辅助CE变大很大程度来自评价目标的重加权，不能据此判定训练退化；重构只恢复标量损失，不能恢复反事实梯度范数。", '',
        '## 剩余瓶颈与后续判断', '',
        'D第一轮支持在SparseWorld基线内继续研究语义监督平衡，但7类正式IoU仍为0，且收益并不覆盖所有类别。truck的56,040个GT体素在条件17类中分类正确，只有689个成为正式正确预测，其余55,351个仍被正式预测为空；bus对应2,698→997。这说明提升条件语义后，空/非空竞争和占用检出仍会限制最终IoU。', '',
        '下一步判断宜先看D原15轮计划中优势是否保持、manmade是否恢复、零IoU类别是否逐渐出现；在此之前不能认定收敛上限更高。若进一步消融，可保持同一协议比较权重平衡强度，以及使用C分母的重加权辅助项或C中单纯调整lambda，以区分“类别重新分配”与“辅助项有效强度”的贡献。占用校准应单独验证，不直接把E合并进D；本次没有运行这些后续实验。', '',
        '已核验七个时刻全部类别IoU重构、GT一致、条件/可见性分区、配置哈希、初始化与子集一致、逐类损失重构及梯度公式的数值差分。所有训练保持第一轮诊断后暂停。', '',
        '- [可追溯统计与输入SHA256](D_first_epoch_analysis.json)',
        '- [原始D/E正式报告与每时刻类别IoU](report.md)',
        '- [逐时刻逐类别CSV](per_horizon_per_class_iou.csv)',
        '- [正式对比图](first_epoch_comparison.png)',
        '- [新增分析脚本](../../experiments/sparseworld_opus_gaussian_bias/analyze_D_first_epoch.py)',
        '- [D辅助损失代码](../../experiments/sparseworld_opus_gaussian_bias/semantic_occupancy_first_epoch/loss.py)',
        '- [D独立配置](../../config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_balanced_conditional_D_4096_1024_15e.py)', '',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
