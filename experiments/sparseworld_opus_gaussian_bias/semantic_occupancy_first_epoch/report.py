"""Compare D/E with C using exact counts and paired scene bootstrap intervals."""
import csv
from pathlib import Path

import numpy as np

from ..ab_first_epoch.report import read, verify_arm
from ..metrics import CLASS_NAMES
from ..train import atomic_json, digest
from .protocol import ANALYSIS, ARMS, C_ANALYSIS, C_WORK, D, E, ROOT


def batch_miou(histograms):
    diagonal = np.diagonal(histograms, axis1=-2, axis2=-1)
    union = histograms.sum(-2) + histograms.sum(-1) - diagonal
    iou = np.divide(diagonal, union, out=np.zeros_like(diagonal, dtype=np.float64), where=union > 0)
    return iou[..., :17].mean(-1) * 100


def paired_bootstrap(candidate, baseline, draws=2000):
    if candidate.shape != baseline.shape or candidate.ndim != 3 or candidate.shape[1:] != (18, 18):
        raise ValueError('paired scene counts must have identical [scene,18,18] shapes')
    if not np.array_equal(candidate.sum(2), baseline.sum(2)):
        raise ValueError('paired scene GT differs')
    n = len(candidate)
    if n < 2:
        raise ValueError('scene bootstrap needs at least two scenes')
    rng = np.random.default_rng(17)
    weights = rng.multinomial(n, np.ones(n) / n, size=draws)
    c = np.einsum('bs,sij->bij', weights, candidate, optimize=True)
    b = np.einsum('bs,sij->bij', weights, baseline, optimize=True)
    differences = batch_miou(c) - batch_miou(b)
    delta = float(batch_miou(candidate.sum(0)) - batch_miou(baseline.sum(0)))
    lower, upper = np.quantile(differences, [.025, .975])
    corrected_lower, corrected_upper = np.quantile(differences, [.0125, .9875])
    baseline_miou = float(batch_miou(baseline.sum(0)))
    relative = 100 * delta / baseline_miou if baseline_miou else 0.
    return dict(seed=17, draws=draws, resampling_unit='scene', scenes=n,
        delta_miou_pp=delta, relative_miou_percent=relative, CI95_pp=[float(lower), float(upper)],
        Bonferroni_two_routes_CI97_5_pp=[float(corrected_lower), float(corrected_upper)],
        data_resampling_evidence_positive=bool(corrected_lower > 0),
        practical_gain_at_least_5_percent=bool(relative >= 5),
        meets_prespecified_first_epoch_screen=bool(corrected_lower > 0 and relative >= 5),
        limitation='One training seed: scene bootstrap measures evaluation data variability, not training-run stability.')


def checked_matrix(work, metrics):
    with np.load(work / 'epoch_01_confusion.npz', allow_pickle=False) as saved:
        matrix = saved['confusion'].copy()
    if (matrix.shape != (7, 18, 18) or metrics['num_samples'] != 1024 or metrics['epoch'] != 1 or
            metrics['miou_divisor'] != 17 or metrics['evaluation_mask'] != 'none'):
        raise ValueError('formal protocol differs')
    for h, hist in enumerate(matrix):
        union = hist.sum(0) + hist.sum(1) - hist.diagonal()
        iou = np.divide(hist.diagonal(), union, out=np.zeros(18), where=union > 0) * 100
        if not np.allclose(iou, metrics['horizons'][h]['class_iou'], rtol=1e-12, atol=1e-12):
            raise ValueError('formal IoUs differ from confusion counts')
        if not np.array_equal(hist.sum(1), metrics['horizons'][h]['class_gt_count']):
            raise ValueError('formal GT counts differ')
    return matrix


def load_scenes(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def create_report():
    reference = read(ANALYSIS / 'C_reference_eval/reference_verified.json')
    if reference['status'] != 'passed' or not reference['exact_formal_confusion_matches_C']:
        raise ValueError('paired baseline must reproduce the preserved C checkpoint exactly')
    original = Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e')
    roots = dict(Strict=original / 'sparseworld_strict',
        B_ce_free_0_1=Path('out/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e/B_ce_free_0_1'),
        C_conditional_semantic_ce=C_WORK)
    verified = {}
    config_audit = read(ANALYSIS / 'configuration_audit.json')
    for name in ARMS:
        work = ROOT / name
        if not (work / 'first_epoch_completed.json').exists():
            continue
        arm = verify_arm(work, ANALYSIS / name)
        audit = config_audit['arms'][name]
        if digest(ARMS[name]) != audit['config_sha256'] or arm['boundary']['config_sha256'] != audit['config_sha256']:
            raise ValueError('validated route configuration changed')
        if arm['initialization']['shared_initial_model_sha256'] != read(C_WORK / 'initialization.json')['initial_model_sha256']:
            raise ValueError('unchanged model tensors differ from C initialization')
        if arm['boundary']['manifest_sha256'] != read(C_WORK / 'first_epoch_completed.json')['manifest_sha256']:
            raise ValueError('immutable subset differs from C')
        roots[name] = work
        verified[name] = arm
    if not verified:
        raise ValueError('no route has a complete epoch-one diagnosis')
    metrics = {name: read(work / 'epoch_01_metrics.json') for name, work in roots.items()}
    matrices = {name: checked_matrix(work, metrics[name]) for name, work in roots.items()}
    c_matrix = matrices['C_conditional_semantic_ce']
    if any(not np.array_equal(hist.sum(2), c_matrix.sum(2)) for hist in matrices.values()):
        raise ValueError('formal targets differ across routes')
    c_scenes = load_scenes(ROOT / 'C_reference_eval/epoch_01_scene_confusion.npz')
    if not np.array_equal(c_scenes['confusion'].sum(0), c_matrix):
        raise ValueError('paired C scene counts differ from formal C')
    judgements = {}
    for name in verified:
        scenes = load_scenes(ROOT / name / 'epoch_01_scene_confusion.npz')
        if (not np.array_equal(scenes['scenes'], c_scenes['scenes']) or
                not np.array_equal(scenes['sample_counts'], c_scenes['sample_counts']) or
                not np.array_equal(scenes['confusion'].sum(0), matrices[name])):
            raise ValueError('route scene pairing differs')
        conditional = read(ANALYSIS / name / 'epoch_diagnostics/epoch_01_conditional_semantics.json')
        if conditional['num_samples'] != 16 or conditional['formal_sample_count'] != 1024 or conditional['is_smoke']:
            raise ValueError('conditional diagnosis incomplete')
        judgement = paired_bootstrap(scenes['confusion'][:, 0], c_scenes['confusion'][:, 0])
        m, c = metrics[name]['horizons'][0], metrics['C_conditional_semantic_ce']['horizons'][0]
        judgement.update(occupied_iou_delta_pp=m['occupied_iou']-c['occupied_iou'],
            occupied_precision_delta_pp=m['occupied_precision']-c['occupied_precision'],
            occupied_recall_delta_pp=m['occupied_recall']-c['occupied_recall'],
            current_conditional_micro_accuracy_percent=conditional['formal_current_conditional_class_accuracy_percent'],
            current_conditional_macro_accuracy_percent=conditional['formal_current_conditional_macro_accuracy_percent'],
            zero_iou_classes=[CLASS_NAMES[i] for i in range(17) if m['class_iou'][i] == 0])
        judgements[name] = judgement
    atomic_json(ANALYSIS / 'comparison.json', dict(scope='first_epoch_only', planned_epochs_per_arm=15, actual_epochs_per_arm=1,
        metrics=metrics, arms=verified, judgements=judgements, C_reference=reference,
        significance_protocol='Primary current mIoU, paired scene bootstrap 2000, Bonferroni-adjusted two-route CI; practical relative gain >=5%.',
        geometry_route_deferred=True, configuration_audit=config_audit))
    rows = []
    for h in range(7):
        for c, name in enumerate(CLASS_NAMES):
            row = dict(seconds=h * .5, class_index=c, class_name=name, gt_voxels=int(c_matrix[h, c].sum()))
            row.update({f'{key}_iou': value['horizons'][h]['class_iou'][c] for key, value in metrics.items()})
            row.update({f'{key}_minus_C_pp': value['horizons'][h]['class_iou'][c]-metrics['C_conditional_semantic_ce']['horizons'][h]['class_iou'][c] for key, value in metrics.items() if key in verified})
            rows.append(row)
    with (ANALYSIS / 'per_horizon_per_class_iou.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 类别平衡与独立占用：D/E第一轮诊断', '',
        '两组分别从头训练，保留15轮/15360次更新的学习率计划；实际各训练一轮（1024次更新），完整评估1024样本后暂停。C为直接对照，Strict和B同轮结果同时列出。几何研究路线暂存。', '',
        '- D只重加权17类条件辅助CE：乘数clip(sqrt(训练集非空类平均体素数/该类体素数),0.5,3)。原18类CE、free=0.1、Lovasz与lambda=1保持一致。',
        '- E只改变占用输出结构：独立的256→128 child证据分支，复用原高斯中心/协方差/opacity及语义渲染；占用概率与17类条件softmax组合为18类联合概率。联合CE等价于占用二元CE与条件语义CE，原权重/归一化保持一致。',
        '- E固定sigmoid占用阈值0.5，预测非空后再取17类语义argmax。新增32,896参数；原C参数初始化逐张量一致，新增分支零权重、raw bias=-4.59512。不同结构不宣称完整state_dict完全相同。',
        '- E固定log(17)占用先验偏移，用于对应原18类CE中的17类前景总质量；未使用评估数据选择偏移或阈值。几何和共享特征仍可间接耦合，独立性指固定它们时语义logit不改变占用概率。',
        '- C仅重新评估已保存第一轮checkpoint，场景计数之和必须与原C正式混淆矩阵完全一致；不推进C训练、不修改C结果。', '',
        '## 当前时刻正式指标', '',
        '| 模型 | mIoU % | 占用IoU % | precision % | recall % |', '|---|---:|---:|---:|---:|']
    for name, value in metrics.items():
        m = value['horizons'][0]
        lines.append(f"| {name} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} |")
    lines += ['', '## 相对C的第一轮判断', '',
        '在看正式结果前固定筛选标准：当前mIoU相对提高至少5%，且两条路线Bonferroni调整后的配对场景bootstrap区间下界大于0。它是首轮筛选标准，不代表多训练seed的统计显著性。', '',
        '| 路线 | ΔmIoU pp | 相对变化 % | 场景配对95%区间 pp | 两路线调整区间 pp | 达到预设首轮筛选 |',
        '|---|---:|---:|---|---|---|']
    for name, j in judgements.items():
        lines.append(f"| {name} | {j['delta_miou_pp']:+.4f} | {j['relative_miou_percent']:+.2f} | [{j['CI95_pp'][0]:+.4f},{j['CI95_pp'][1]:+.4f}] | [{j['Bonferroni_two_routes_CI97_5_pp'][0]:+.4f},{j['Bonferroni_two_routes_CI97_5_pp'][1]:+.4f}] | {'是' if j['meets_prespecified_first_epoch_screen'] else '否'} |")
    for name, j in judgements.items():
        value = verified[name]['diagnostic']['visibility']
        c_diag = read(C_ANALYSIS / 'epoch_diagnostics/epoch_01_diagnostics.json')['visibility']
        lines += ['', f"{name}：占用IoU {j['occupied_iou_delta_pp']:+.4f} pp、precision {j['occupied_precision_delta_pp']:+.4f} pp、recall {j['occupied_recall_delta_pp']:+.4f} pp。条件类别micro/macro准确率为{j['current_conditional_micro_accuracy_percent']:.3f}%/{j['current_conditional_macro_accuracy_percent']:.3f}%，IoU为零的非空类有{len(j['zero_iou_classes'])}个。"]
        for region in ('camera', 'noncamera'):
            current, old = value['h0/'+region], c_diag['h0/'+region]
            lines.append(f"{region}：FP {current['occupied_fp']:,}（Δ {current['occupied_fp']-old['occupied_fp']:+,}），FN {current['occupied_fn']:,}（Δ {current['occupied_fn']-old['occupied_fn']:+,}）。")
        delta = np.asarray(metrics[name]['horizons'][0]['class_iou'][:17]) - np.asarray(metrics['C_conditional_semantic_ce']['horizons'][0]['class_iou'][:17])
        lines += ['收益最大的类：'+'、'.join(f'{CLASS_NAMES[c]} {delta[c]:+.4f} pp' for c in np.argsort(delta)[-3:][::-1])+'。',
                  '差值最小的类：'+'、'.join(f'{CLASS_NAMES[c]} {delta[c]:+.4f} pp' for c in np.argsort(delta)[:3])+'。']
    if E in verified:
        gate = read(ANALYSIS / E / 'epoch_diagnostics/epoch_01_occupancy_gate.json')
        joint = gate['same_logits_alternative'][0]
        lines += ['', f"E同一输出改回18类联合概率argmax：当前mIoU={joint['miou']:.4f}%、占用IoU={joint['occupied_iou']:.4f}%。这是同checkpoint的决策诊断，不是另一组训练；正式E使用预先固定的独立二元判断。"]
    lines += ['', '## 每时刻每类别IoU（%）', '',
        '第一轮只直接监督当前Gaussian；未来时刻全部报告，但不用于首轮路线胜负。mIoU平均17个非空类，free另列。']
    keys = list(metrics)
    for h in range(7):
        lines += ['', f'### {h*.5:g}秒', '', '| 类别 | '+' | '.join(keys)+' |', '|---|'+'---:|'*len(keys)]
        for row in rows[h*18:(h+1)*18]:
            lines.append('| '+row['class_name']+' | '+' | '.join(f"{row[key+'_iou']:.6f}" for key in keys)+' |')
    lines += ['', '## 证据与限制', '',
        '一个训练seed和一个epoch，不能据此证明最终15轮收益或训练稳定性。场景bootstrap考虑同场景帧的相关性，但仍只反映这份评估数据的波动。属性/渲染探针固定16样本，末层梯度固定2样本。', '',
        '- [126行逐时刻逐类别CSV](per_horizon_per_class_iou.csv)',
        '- [完整配置、边界、指标和区间](comparison.json)',
        '- [训练集类别统计](training_class_frequencies_mean_balance.json)',
        '- [D诊断](D_balanced_conditional_ce/epoch_diagnostics/epoch_01_diagnostics.md)',
        '- [E诊断](E_factorized_occupancy/epoch_diagnostics/epoch_01_diagnostics.md)',
        '- [独立配置审计](configuration_audit.json)']
    (ANALYSIS / 'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(ANALYSIS / 'diagnosis_completed.json', dict(status='completed' if len(verified)==2 else 'partial',
        compared_epoch=1, planned_epochs=15, completed_arms=list(verified), judgements=judgements))
    print(f'DE_REPORT_COMPLETE arms={list(verified)} report={ANALYSIS / "report.md"}', flush=True)


if __name__ == '__main__':
    create_report()
