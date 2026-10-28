"""Generate the final seven-horizon, eighteen-class comparison from evidence."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..metrics import CLASS_NAMES
from ..train import atomic_json, digest
from .protocol import ROOT, ANALYSIS, SOURCE_WORK

VARIANTS = ('sparseworld_strict', 'D_balanced_conditional_ce')


def read(path):
    return json.loads(Path(path).read_text())


def gather(directory):
    completion = read(directory / 'completed.json')
    if completion['epochs'] != 15:
        raise ValueError('final reporting requires all 15 epochs')
    epochs = [read(directory / f'epoch_{e:02d}_metrics.json') for e in range(1, 16)]
    if any(e['num_samples'] != 1024 for e in epochs):
        raise ValueError('an epoch did not evaluate the full held-out subset')
    best = read(directory / 'best.json')
    winner = max(epochs, key=lambda e: e['selection_mean_miou'])
    if best != winner:
        raise ValueError('saved best differs from declared common selection rule')
    recheck = read(directory / 'best_recheck' / f"epoch_{best['epoch']:02d}_metrics.json")
    original_confusion = np.load(directory / f"epoch_{best['epoch']:02d}_confusion.npz")['confusion']
    recheck_confusion = np.load(directory / 'best_recheck' / f"epoch_{best['epoch']:02d}_confusion.npz")['confusion']
    if not np.array_equal(original_confusion, recheck_confusion):
        raise ValueError('best checkpoint re-evaluation confusion matrices differ')
    return dict(completion=completion, epochs=epochs, best=best,
                initialization=read(directory / 'initialization.json'),
                best_checkpoint_sha256=digest(directory / 'best_mIoU.pth'))


def verify_diagnostics(work, output):
    records = []
    for epoch in range(1, 16):
        directory = output / 'epoch_diagnostics'
        diagnostic = read(directory / f'epoch_{epoch:02d}_diagnostics.json')
        metrics = read(work / f'epoch_{epoch:02d}_metrics.json')
        semantics = read(directory / f'epoch_{epoch:02d}_conditional_semantics.json')
        if (diagnostic['epoch'] != epoch or diagnostic['num_samples'] != 1024 or diagnostic['is_smoke'] or
                diagnostic['voxel_supervision_mask'] != 'none' or len(diagnostic['attribute_sample_indices']) != 16 or
                len(diagnostic['attributes']) != 112 or semantics['num_samples'] != 16 or semantics['formal_sample_count'] != 1024):
            raise ValueError(f'incomplete D diagnostics at epoch {epoch}')
        with np.load(work / f'epoch_{epoch:02d}_confusion.npz', allow_pickle=False) as formal:
            hist = formal['confusion']
        with np.load(directory / f'epoch_{epoch:02d}_visibility_confusion.npz', allow_pickle=False) as visible:
            if not np.array_equal(visible['full'], hist) or not np.array_equal(visible['camera']+visible['noncamera'], hist):
                raise ValueError('formal and diagnostic visibility counts differ')
        with np.load(directory / f'epoch_{epoch:02d}_conditional_semantic_confusion.npz', allow_pickle=False) as cond:
            if not np.array_equal(cond['full'].sum(1), hist[0].sum(1)[:17]) or not np.array_equal(cond['camera']+cond['noncamera'], cond['full']):
                raise ValueError('conditional diagnosis must cover all GT occupied voxels')
        for horizon, h in enumerate(hist):
            gt = h.sum(1)
            union = gt+h.sum(0)-h.diagonal()
            iou = np.divide(h.diagonal(), union, out=np.zeros(18), where=union>0)*100
            if not np.allclose(iou, metrics['horizons'][horizon]['class_iou'], atol=1e-12, rtol=1e-12):
                raise ValueError('saved class IoU reconstruction failed')
        records.append(diagnostic)
    atomic_json(output/'diagnostics_verification.json', dict(status='passed', verified_epochs=15,
        formal_samples_per_epoch=1024, attribute_samples_per_epoch=16, seven_horizons_verified=True,
        fullspace_visibility_partitions_verified=True, conditional_GT_occupied_partitions_verified=True))
    return records


def diagnostic_table(records):
    lines = ['', '## 每轮诊断趋势', '',
             '| epoch | 选择平均mIoU % | 当前mIoU % | 当前FP | 当前FN | 当前scale饱和 % | 当前opacity饱和 % | 诊断 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for record in records:
        current = [a for a in record['attributes'] if a['horizon']==0]
        epoch = record['epoch']
        v = record['visibility']['h0/full']
        scale = np.mean([a['scale_upper_saturation_percent'] for a in current])
        opacity = np.mean([a['opacity_upper_saturation_percent'] for a in current])
        lines.append(f"| {epoch} | {record['selection_mean_miou']:.4f} | {v['miou']:.4f} | {v['occupied_fp']:,} | {v['occupied_fn']:,} | {scale:.3f} | {opacity:.3f} | [诊断](epoch_diagnostics/epoch_{epoch:02d}_diagnostics.md) |")
    return lines


def refresh(work, output):
    """Update progress from completed evaluations, without declaring completion."""
    files = sorted(work.glob('epoch_[0-9][0-9]_metrics.json'))
    epochs = [read(path) for path in files]
    if not epochs:
        return
    fields = ['epoch', 'selection_mean_miou', 'current_occupied_iou', 'current_precision', 'current_recall'] + [f'h{h}_miou' for h in range(7)]
    with (output/'progress_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in epochs:
            current = epoch['horizons'][0]
            writer.writerow(dict(epoch=epoch['epoch'], selection_mean_miou=epoch['selection_mean_miou'],
                current_occupied_iou=current['occupied_iou'], current_precision=current['occupied_precision'], current_recall=current['occupied_recall'],
                **{f'h{h}_miou':epoch['horizons'][h]['miou'] for h in range(7)}))
    best = max(epochs, key=lambda item:item['selection_mean_miou'])
    lines = ['# D完整15轮训练进度', '',
             f"已完成正式评估{len(epochs)}/15轮。最优已评估epoch={best['epoch']}，当前/1/2/3秒平均mIoU={best['selection_mean_miou']:.4f}%。", '',
             'D从原第一轮检查点继续，保留15轮学习率计划。当前/1/2/3秒平均mIoU仍作为共同checkpoint选择规则；单时刻最优仅作诊断。', '',
             '| epoch | 当前mIoU % | 1秒mIoU % | 2秒mIoU % | 3秒mIoU % | 选择平均mIoU % | 同轮Strict平均mIoU % | 诊断 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for epoch in epochs:
        index = epoch['epoch']
        strict = read(ROOT/'sparseworld_strict'/f'epoch_{index:02d}_metrics.json')
        values = ' | '.join(f"{epoch['horizons'][h]['miou']:.4f}" for h in (0,2,4,6))
        lines.append(f"| {index} | {values} | {epoch['selection_mean_miou']:.4f} | {strict['selection_mean_miou']:.4f} | [诊断](epoch_diagnostics/epoch_{index:02d}_diagnostics.md) |")
    lines += ['', '- [逐轮进度CSV](progress_metrics.csv)', '- [续训来源审计](preparation.json)', '',
              '最终报告将在15轮训练、最佳checkpoint独立复评和完整诊断验证后生成。', '']
    (output/'progress_report.md').write_text('\n'.join(lines))
    atomic_json(output/'status.json', dict(status='running', phase='train', completed_training_epochs=len(epochs),
        last_completed_epoch=epochs[-1]['epoch'], planned_epochs=15, best_evaluated_epoch=best['epoch'],
        best_evaluated_selection_miou=best['selection_mean_miou'], work_directory=str(work)))


def create_report(root, output):
    arms = {v: gather(root / v) for v in VARIANTS}
    left, right = [arms[v] for v in VARIANTS]
    if left['completion']['manifest_sha256'] != right['completion']['manifest_sha256']:
        raise ValueError('experiment arms used different subsets')
    if left['initialization']['sha256'] != right['initialization']['sha256']:
        raise ValueError('experiment arms used different image initialization')
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = verify_diagnostics(root / VARIANTS[1], output)
    atomic_json(output / 'comparison.json', arms)
    rows = []
    for h in range(7):
        a, b = left['best']['horizons'][h], right['best']['horizons'][h]
        for c, name in enumerate(CLASS_NAMES):
            if a['class_gt_count'][c] != b['class_gt_count'][c]:
                raise ValueError('per-class target voxel counts differ')
            rows.append(dict(horizon_index=h, seconds=h*.5, class_index=c, class_name=name,
                strict_iou=a['class_iou'][c], gaussian_iou=b['class_iou'][c],
                delta_percentage_points=b['class_iou'][c]-a['class_iou'][c],
                gt_voxels=a['class_gt_count'][c], strict_union=a['class_union'][c],
                gaussian_union=b['class_union'][c],
                strict_epoch=left['best']['epoch'], gaussian_epoch=right['best']['epoch']))
    with open(output / 'per_horizon_per_class_iou.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / 'per_epoch_metrics.csv', 'w', newline='') as f:
        fields = ['model', 'epoch', 'selection_mean_miou'] + [f'h{h}_miou' for h in range(7)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant in VARIANTS:
            for epoch in arms[variant]['epochs']:
                writer.writerow(dict(model=variant, epoch=epoch['epoch'],
                    selection_mean_miou=epoch['selection_mean_miou'],
                    **{f'h{h}_miou':epoch['horizons'][h]['miou'] for h in range(7)}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delta = np.array([[b['class_iou'][c]-a['class_iou'][c] for c in range(17)]
        for a,b in zip(left['best']['horizons'],right['best']['horizons'])])
    fig, ax = plt.subplots(figsize=(16, 5))
    bound = max(float(np.abs(delta).max()), .1)
    im = ax.imshow(delta, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_xticks(range(17), CLASS_NAMES[:17], rotation=55, ha='right')
    ax.set_yticks(range(7), [f'{h*.5:g}s' for h in range(7)])
    ax.set_title('Gaussian minus Strict: class IoU difference (percentage points)')
    for h in range(7):
        for c in range(17):
            ax.text(c,h,f'{delta[h,c]:+.1f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im, ax=ax, label='percentage points')
    fig.tight_layout()
    fig.savefig(output / 'class_iou_delta.png',dpi=180)
    fig.savefig(output / 'class_iou_delta.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2,4,figsize=(15,7),sharex=True)
    for h, ax in enumerate(axes.flat):
        for v in VARIANTS:
            values=[e['selection_mean_miou'] if h==7 else e['horizons'][h]['miou'] for e in arms[v]['epochs']]
            ax.plot(range(1,16),values,label=v.replace('sparseworld_',''),marker='.',markersize=4)
        ax.set_title('Selection mean' if h==7 else f't + {h*.5:g}s')
        ax.set_ylabel('mIoU (%)')
        ax.set_xlabel('epoch')
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png',dpi=180)
    plt.close(fig)

    a, b = left['best'], right['best']
    difference=b['selection_mean_miou']-a['selection_mean_miou']
    lines=['# SparseWorld OPUS Gaussian bias D类别平衡：4096 / 1024，15 epoch 对比', '',
        f"两组均完成 15 个完整 epoch。按照当前与未来 1/2/3 秒平均 mIoU 选择一个共同最优 checkpoint，"
        f"Strict 最优 epoch 为 {a['epoch']}，Gaussian 最优 epoch 为 {b['epoch']}。平均 mIoU："
        f"{a['selection_mean_miou']:.4f}% → {b['selection_mean_miou']:.4f}%，差值 {difference:+.4f} 个百分点。", '',
        '## 实验协议与解释范围', '',
        '- D保留原15轮计划，从已核验第一轮边界继续第2～15轮；第1轮结果复制自原实验。原第一轮checkpoint、配置和分析目录均保留。',
        '- 新独立配置仅改变report_directory；迁移checkpoint只更新config_sha256，模型、优化器、RNG、query统计和1024次更新进度逐值一致。',
        '- 原18类全空间CE权重10、free权重0.1、Lovasz权重1、条件语义lambda=1保持不变；训练频率乘数沿用D第一轮。',
        '- 每轮完成1024样本正式评估和固定16样本属性/语义诊断，前2样本检查实际监督输出梯度；未启用几何研究路线。',
        '- 两组共用 4096 个训练样本、1024 个 held-out 样本、5 帧图像、seed 17、COCO 图像初始化、15 epoch 和相同优化器主设置。',
        '- 1024 子集用于逐 epoch 选最优，因此是验证角色；本结果不是额外独立测试集结果。',
        '- 全空间评测 17 个非空类别；empty 类单列。mIoU 固定除以 17，union=0 类置零并披露。',
        '- 新模型将子点数从 48 提升至 128，并使用全空间Gaussian体素监督、free=0.1及频率平衡17类条件辅助CE；本对比反映整个替换方案，不能单独归因于 Gaussian 表示。',
        '- 单一随机种子，没有多次重复实验；小幅差异不能据此解释为统计显著。',
        '- 每个模型的最终类别数值全部来自同一个最优 epoch；没有拼接类别最优值。', '',
        '## 各时刻整体性能', '',
        '| 时刻 | Strict mIoU | Gaussian mIoU | Δ pp | Strict occupied IoU | Gaussian occupied IoU |',
        '|---|---:|---:|---:|---:|---:|']
    for h in range(7):
        x,y=a['horizons'][h],b['horizons'][h]
        lines.append(f"| {h*.5:g}s | {x['miou']:.4f} | {y['miou']:.4f} | {y['miou']-x['miou']:+.4f} | {x['occupied_iou']:.4f} | {y['occupied_iou']:.4f} |")
    lines += ['', '## 每个时刻、每个类别 IoU', '']
    for h in range(7):
        lines += [f'### 当前后 {h*.5:g} 秒', '',
            '| 类别 | Strict IoU (%) | Gaussian IoU (%) | Δ pp | GT 体素数 |',
            '|---|---:|---:|---:|---:|']
        for row in [r for r in rows if r['horizon_index']==h]:
            lines.append(f"| {row['class_name']} | {row['strict_iou']:.4f} | {row['gaussian_iou']:.4f} | {row['delta_percentage_points']:+.4f} | {row['gt_voxels']} |")
        class_delta = delta[h]
        gained = sorted(range(17),key=lambda c:class_delta[c],reverse=True)[:3]
        lost = sorted(range(17),key=lambda c:class_delta[c])[:3]
        lines += ['', '差值最大的三个类别：' + '、'.join(f'{CLASS_NAMES[c]} {class_delta[c]:+.4f} pp' for c in gained) + '。',
            '差值最小的三个类别：' + '、'.join(f'{CLASS_NAMES[c]} {class_delta[c]:+.4f} pp' for c in lost) + '。',
            f"union=0 类：Strict {a['horizons'][h]['absent_union_classes']}；Gaussian {b['horizons'][h]['absent_union_classes']}。", '']
    lines += ['## 轨迹与占据诊断', '', '| 未来时刻 | Strict L2 (m) | Gaussian L2 (m) | Strict precision / recall (%) | Gaussian precision / recall (%) |',
              '|---|---:|---:|---:|---:|']
    for h in range(1,7):
        x,y=a['horizons'][h],b['horizons'][h]
        lines.append(f"| {h*.5:g}s | {a['trajectory_l2_m'][h-1]:.4f} | {b['trajectory_l2_m'][h-1]:.4f} | {x['occupied_precision']:.4f} / {x['occupied_recall']:.4f} | {y['occupied_precision']:.4f} / {y['occupied_recall']:.4f} |")
    lines += ['', '## 各时刻单独最优 epoch（诊断）', '',
        '以下结果允许每个时刻选择不同 epoch，与上面的共同 checkpoint 主比较不同。', '',
        '| 时刻 | Strict epoch / mIoU | Gaussian epoch / mIoU |', '|---|---:|---:|']
    for h in range(7):
        x=max(left['epochs'],key=lambda e:e['horizons'][h]['miou'])
        y=max(right['epochs'],key=lambda e:e['horizons'][h]['miou'])
        lines.append(f"| {h*.5:g}s | {x['epoch']} / {x['horizons'][h]['miou']:.4f} | {y['epoch']} / {y['horizons'][h]['miou']:.4f} |")
    lines += ['', '## 原始证据', '',
        '- [逐时刻逐类别 CSV](per_horizon_per_class_iou.csv)',
        '- [逐 epoch CSV](per_epoch_metrics.csv)',
        '- [完整指标及配置/manifest/checkpoint SHA256](comparison.json)',
        '- [类别 IoU 差值图](class_iou_delta.png)',
        '- [训练曲线](training_curves.png)',
        '', '两组最优 checkpoint 均完成独立重新评测，混淆矩阵与对应 epoch 逐元素一致。', '']
    lines += diagnostic_table(diagnostics)
    (output / 'report.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(output / 'status.json',dict(status='completed',epochs_per_model=15, diagnostics_verified_epochs=15,
        strict_best_epoch=a['epoch'],gaussian_best_epoch=b['epoch'],delta_mean_miou_pp=difference))
    print(f'Final report: {output / "report.md"}',flush=True)



def main():
    create_report(ROOT, ANALYSIS)


if __name__ == '__main__':
    main()
