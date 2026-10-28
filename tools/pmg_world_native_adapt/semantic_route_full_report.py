"""Per-epoch full-validation tables and paired fixed-subset S2 diagnostics."""

import csv
import json
from pathlib import Path

import numpy as np

from tools.pmg_world_native_adapt import gate_report
from tools.pmg_world_native_adapt.gate_optimization_report import (
    CONTROLS, validate_metric_fields, build_statistics,
    write_class_results, write_regional_results, write_summary,
)
from tools.pmg_world_native_adapt.report import regional
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


def diagnostic_report(folder, epoch, updates):
    folder=Path(folder)
    sources=(*CONTROLS,('S2',folder,f'epoch_{epoch:02d}'))
    metrics,global_counts,parts,raw_regions,hashes=gate_report.load_sources(sources)
    validate_metric_fields(metrics,global_counts,raw_regions,sources)
    statistics=build_statistics(metrics,global_counts,parts,raw_regions,sources,hashes,'trained-candidate')
    statistics.update(additional_training_updates=updates,full_training_epochs=epoch,
        diagnostic_scope='Fixed 1024 development samples within full 4219 validation; not full-split regional accuracy')
    description=('S2全量训练后的固定1024样本监测；共享感知和原生未来参考冻结，仅候选预测器及门可训练。'
                 '原生点集/轨迹损失监督候选，体素损失监督路由后的完整Gaussian。'
                 '本报告的区域和配对统计仅覆盖旧1024样本，完整验证mIoU另见上层training_report.md。')
    statistics['description']=description
    runtime.atomic_json(folder/'comparison_statistics.json',statistics)
    write_class_results(folder,sources,metrics)
    rows=write_regional_results(folder,sources,metrics,raw_regions)
    write_summary(folder,sources,metrics,statistics,rows,description)
    return statistics


def refresh(directory):
    directory=Path(directory)
    epochs=[json.loads(p.read_text()) for p in sorted(directory.glob('epoch_*_metrics.json'))]
    if not epochs:return
    best=max(epochs,key=lambda x:x['selection_mean_miou'])
    runtime.atomic_json(directory/'training_summary.json',dict(
        model='S2-semantic-route-full15e',completed_epochs=len(epochs),planned_epochs=15,
        best_epoch_by_full_primary=best['epoch'],best_selection_mean_miou=best['selection_mean_miou'],
        validation_samples=best['num_samples'],initialization='T-A0 subset epoch10',
        training_scope='Candidate six forecast modules plus gate; perception and native reference frozen',
        best_metrics=best))
    rows=[]
    lines=['# S2全量15轮训练：逐轮结果','','训练19,730样本，完整验证4,219样本。mIoU平均17个非空类别；主指标为0/1/2/3秒均值。',
           '','从T-A0子集epoch10权重开始适配；感知与原生参考冻结。不是全网络从随机初始化训练。',
           '','| Epoch | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 主mIoU |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for metric in epochs:
        lines.append(f"| {metric['epoch']} | "+' | '.join(f"{h['miou']:.4f}" for h in metric['horizons'])+f" | {metric['selection_mean_miou']:.4f} |")
        for horizon in metric['horizons']:
            for c,name in enumerate(CLASS_NAMES):
                rows.append(dict(epoch=metric['epoch'],seconds=horizon['seconds'],class_index=c,
                    class_name=name,iou_percent=horizon['class_iou'][c],miou_percent=horizon['miou'],
                    occupied_iou_percent=horizon['occupied_iou']))
    with (directory/'epoch_horizon_class_iou.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines+=['',f"按全量主mIoU选择的当前最佳轮次：{best['epoch']}（{best['selection_mean_miou']:.4f}%）。",'',
            '## 固定1024样本的区域诊断','','下面数值不是全量4,219样本的区域指标；F-only也不等于静态区域。',
            '','| Epoch | fast平均准确率 | F-only 1s | 2s | 3s | 原联合门槛 |',
            '|---|---:|---:|---:|---:|---|']
    for metric in epochs:
        epoch=metric['epoch'];folder=directory/'diagnostics'/f'epoch_{epoch:02d}'
        path=folder/f'epoch_{epoch:02d}_metrics.json'
        if not path.exists():continue
        sub=json.loads(path.read_text());fonly=regional(sub,'F_only','occupied_semantic_accuracy')
        stats=json.loads((folder/'comparison_statistics.json').read_text())
        lines.append(f"| {epoch} | {regional(sub,'fast','occupied_semantic_accuracy').mean():.4f} | "+
            ' | '.join(f'{x:.4f}' for x in fonly)+f" | {'通过' if stats['joint_target_passed'] else '未通过'} |")
    lines+=['','每轮诊断用于观察过程，不触发提前停止。场景区间不覆盖训练随机性或重复开发选择。','',
            '- [逐轮、七时刻、18类IoU](epoch_horizon_class_iou.csv)',
            '- [固定子集详细诊断](diagnostics/)',
            '- [原生参考的全量验证](reference_full/)',
            '- [实验设计](experiment_design.md)']
    (directory/'training_report.md').write_text('\n'.join(lines)+'\n')
