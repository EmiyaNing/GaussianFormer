"""Compare epoch one only; never present it as a completed 15-epoch result."""
import csv
import json
from pathlib import Path

import numpy as np

from ..metrics import CLASS_NAMES
from ..train import atomic_json, digest
from .protocol import ANALYSIS, ARMS, ROOT


def read(path):
    return json.loads(Path(path).read_text())


def verify_arm(work, output):
    boundary = read(work / 'first_epoch_completed.json')
    budget = read(work / 'optimization_budget.json')
    metric = read(work / 'epoch_01_metrics.json')
    if (boundary['planned_epochs'] != 15 or boundary['trained_epochs'] != 1 or
            boundary['updates'] != 1024 or boundary['next_iter'] != 0 or boundary['phase'] != 'train' or
            budget['planned_epochs'] != 15 or budget['total_updates'] != 15360 or
            metric['num_samples'] != 1024 or metric['epoch'] != 1):
        raise ValueError('first-epoch experiment did not preserve the 15-epoch protocol')
    if digest(work / 'epoch_01.pth') != boundary['checkpoint_sha256']:
        raise ValueError('epoch-one checkpoint changed')
    diagnostic_dir = output / 'epoch_diagnostics'
    diagnostic = read(diagnostic_dir / 'epoch_01_diagnostics.json')
    classification = read(diagnostic_dir / 'epoch_01_classification_components.json')
    if (diagnostic['is_smoke'] or diagnostic['num_samples'] != 1024 or
            diagnostic['voxel_supervision_mask'] != 'none' or
            diagnostic['config_sha256'] != boundary['config_sha256'] or
            len(diagnostic['attributes']) != 16 * 7 or len(diagnostic['gradient_probes']) != 2 or
            classification['is_smoke'] or classification['num_samples'] != 16 or
            classification['formal_sample_count'] != 1024):
        raise ValueError('first-epoch diagnosis is incomplete')
    with np.load(diagnostic_dir / 'epoch_01_visibility_confusion.npz') as visibility:
        with np.load(work / 'epoch_01_confusion.npz') as formal:
            if (not np.array_equal(visibility['full'], formal['confusion']) or
                    not np.array_equal(visibility['full'], visibility['camera'] + visibility['noncamera'])):
                raise ValueError('diagnostic counts differ from formal evaluation')
    return dict(metric=metric, boundary=boundary, budget=budget, diagnostic=diagnostic,
                classification=classification, initialization=read(work / 'initialization.json'))


def create_report(root=ROOT, output=ANALYSIS, baseline_roots=None):
    output.mkdir(parents=True, exist_ok=True)
    arms = {arm: verify_arm(root / arm, output / arm) for arm in ARMS}
    a, b = [arms[arm] for arm in ARMS]
    if a['boundary']['manifest_sha256'] != b['boundary']['manifest_sha256']:
        raise ValueError('A and B used different subsets')
    if a['initialization']['initial_model_sha256'] != b['initialization']['initial_model_sha256']:
        raise ValueError('A and B did not start from identical model weights')
    if baseline_roots is None:
        old = Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e')
        baseline_roots = dict(Strict=old / 'sparseworld_strict',
            Camera_Gaussian=old / 'sparseworld_opus_gaussian_bias',
            Fullspace_CE=Path('out/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e/sparseworld_opus_gaussian_bias'))
    metrics = {name: read(path / 'epoch_01_metrics.json') for name, path in baseline_roots.items()}
    metrics.update({arm: value['metric'] for arm, value in arms.items()})
    if any(value['num_samples'] != 1024 or value['epoch'] != 1 for value in metrics.values()):
        raise ValueError('all comparisons must use full epoch-one evaluations')
    for h in range(7):
        gt_counts = [value['horizons'][h]['class_gt_count'] for value in metrics.values()]
        if any(counts != gt_counts[0] for counts in gt_counts):
            raise ValueError('horizon/class target counts differ')
    atomic_json(output / 'comparison.json', dict(
        scope='epoch_one_diagnosis_only', planned_epochs_per_arm=15, compared_epoch=1,
        metrics=metrics, arms=arms, same_initial_model_sha256=a['initialization']['initial_model_sha256']))
    rows = []
    for h in range(7):
        for c, name in enumerate(CLASS_NAMES):
            row = dict(seconds=h * .5, class_index=c, class_name=name,
                       gt_voxels=metrics['Fullspace_CE']['horizons'][h]['class_gt_count'][c])
            row.update({f'{key}_iou': value['horizons'][h]['class_iou'][c] for key, value in metrics.items()})
            row.update(A_minus_fullspace_ce_pp=row['A_softmax_focal_iou'] - row['Fullspace_CE_iou'],
                       B_minus_fullspace_ce_pp=row['B_ce_free_0_1_iou'] - row['Fullspace_CE_iou'],
                       A_minus_B_pp=row['A_softmax_focal_iou'] - row['B_ce_free_0_1_iou'])
            rows.append(row)
    with (output / 'per_horizon_per_class_iou.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# A/B全空间监督：15轮预算下的第一轮诊断', '',
             '本报告只比较第1轮完整4096训练样本之后、1024验证样本上的成绩。'
             '两组都使用15轮/15360次更新的学习率计划和原未来课程，当前结果不代表15轮最优性能。', '',
             '- A：体素CE替换为softmax focal，gamma=2，free权重仍为0.5；固定标签权重归一化，Lovasz保持原设置。',
             '- B：保留体素CE，仅将free相对权重从0.5改为0.1；不改变归一化方式。',
             '- 相同seed 17、子集、图像初始化、模型初始权重、优化器和空类bias初值；A/B初始完整state_dict SHA256一致。',
             '- 第1轮完成后保存可继续第2–15轮的完整模型/优化器/RNG/timestamp checkpoint。',
             '- 正式指标均为完整空间；mIoU平均17个非空类，free类另列。', '',
             '## 当前时刻整体性能', '',
             '| 模型 | mIoU (%) | 占据IoU (%) | precision (%) | recall (%) | free IoU (%) |',
             '|---|---:|---:|---:|---:|---:|']
    for name, value in metrics.items():
        m = value['horizons'][0]
        lines.append(f"| {name} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {m['class_iou'][17]:.4f} |")
    baseline = metrics['Fullspace_CE']['horizons'][0]
    lines += ['', '## 第一轮观察', '']
    for arm, value in arms.items():
        m = value['metric']['horizons'][0]
        full = value['diagnostic']['visibility']['h0/full']
        hidden = value['diagnostic']['visibility']['h0/noncamera']
        current = [v for v in value['diagnostic']['attributes'] if v['horizon'] == 0]
        lines += [f"{arm}相对全空间CE：mIoU {m['miou']-baseline['miou']:+.4f} pp，"
                  f"召回{m['occupied_recall']-baseline['occupied_recall']:+.4f} pp；"
                  f"当前FP={full['occupied_fp']}，不可见FP={hidden['occupied_fp']}，预测占据比例{full['predicted_occupied_percent']:.4f}%。", '',
                  f"固定16样本当前尺度饱和比例均值{np.mean([v['scale_upper_saturation_percent'] for v in current]):.3f}%，"
                  f"opacity饱和{np.mean([v['opacity_upper_saturation_percent'] for v in current]):.3f}%，"
                  f"empty_bias={current[0]['empty_bias']:.6f}。", '']
        delta = np.array(m['class_iou'][:17]) - np.array(baseline['class_iou'][:17])
        gained, lost = np.argsort(delta)[-3:][::-1], np.argsort(delta)[:3]
        lines += ['与全空间CE相比差值最大的类别：' + '、'.join(f'{CLASS_NAMES[c]} {delta[c]:+.4f} pp' for c in gained) + '。',
                  '差值最小的类别：' + '、'.join(f'{CLASS_NAMES[c]} {delta[c]:+.4f} pp' for c in lost) + '。', '']
    winner = max(ARMS, key=lambda name: metrics[name]['horizons'][0]['miou'])
    lines += [f'第一轮当前mIoU较高的是{winner}；此结论只限当前checkpoint、当前子集和单seed。'
              '若召回恢复而不可见FP明显增加，需要同时评估precision与各类IoU，不能仅看召回。', '',
              '## 各时刻mIoU与占据IoU', '',
              '| 时刻 | 模型 | mIoU (%) | 占据IoU (%) |', '|---|---|---:|---:|']
    for h in range(7):
        for name, value in metrics.items():
            m = value['horizons'][h]
            lines.append(f"| {h*.5:g}s | {name} | {m['miou']:.4f} | {m['occupied_iou']:.4f} |")
    lines += ['', '第1–5轮为时间扩展预训练，Gaussian直接体素监督只有当前；未来近零不作为这次方案优劣的主要依据。', '',
              '## 分类损失与空类bias梯度', '',
              '固定16样本，仅统计当前分类项，排除Lovasz/集合/轨迹项。'
              '正dL/db会在梯度下降时降低bias、提高空类分数；负dL/db会提高bias、降低空类分数。数值为逐样本均值。', '',
              '| 方案 | 区域/标签 | 分类loss贡献 | CE同输出反事实loss贡献 | dL/db | 正确类别概率均值 |',
              '|---|---|---:|---:|---:|---:|']
    for arm, value in arms.items():
        records = value['classification']['records']
        for bucket in ('camera/free', 'camera/occupied', 'noncamera/free', 'noncamera/occupied'):
            values = [v['buckets'][bucket] for v in records]
            mean = lambda key: float(np.mean([v[key] for v in values]))
            lines.append(f"| {arm} | {bucket} | {mean('classification_loss_contribution'):.6f} | {mean('ce_counterfactual_loss_contribution'):.6f} | {mean('signed_bias_gradient'):.6f} | {mean('mean_correct_class_probability'):.6f} |")
    lines += ['', 'A的CE反事实与focal来自相同渲染输出及类别权重，可用于判断容易空体素损失是否被削弱；'
              '它不等同于用CE重新训练后的效果。B的该反事实与实际CE相同。', '',
              '## 每时刻每类别IoU', '']
    for h in range(7):
        lines += [f'### {h*.5:g}秒', '',
                  '| 类别 | Strict | camera | 全空间CE | A | B | A−全空间CE pp | B−全空间CE pp |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows[h * 18:(h + 1) * 18]:
            lines.append(f"| {row['class_name']} | {row['Strict_iou']:.4f} | {row['Camera_Gaussian_iou']:.4f} | {row['Fullspace_CE_iou']:.4f} | {row['A_softmax_focal_iou']:.4f} | {row['B_ce_free_0_1_iou']:.4f} | {row['A_minus_fullspace_ce_pp']:+.4f} | {row['B_minus_fullspace_ce_pp']:+.4f} |")
    lines += ['', '## 证据', '',
              '- [126行逐时刻逐类别CSV](per_horizon_per_class_iou.csv)',
              '- [配置、checkpoint和原始指标JSON](comparison.json)',
              '- [A完整诊断](A_softmax_focal/epoch_diagnostics/epoch_01_diagnostics.md)',
              '- [B完整诊断](B_ce_free_0_1/epoch_diagnostics/epoch_01_diagnostics.md)',
              '- [配置差异审计](configuration_audit.json)', '']
    (output / 'report.md').write_text('\n'.join(lines))
    atomic_json(output / 'diagnosis_completed.json', dict(
        status='completed', scope='first_epoch_only', planned_epochs=15, compared_epoch=1,
        epoch_one_current_miou_winner=winner,
        current_miou={name: value['horizons'][0]['miou'] for name, value in metrics.items()}))
    print(f'A/B first-epoch report: {output / "report.md"}', flush=True)


if __name__ == '__main__':
    create_report()
