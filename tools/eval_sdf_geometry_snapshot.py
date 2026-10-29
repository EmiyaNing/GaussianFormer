"""Evaluate a saved partial-epoch SDF checkpoint without resuming training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.prepare_sdf_geometry_cache import sha, write
import tools.run_sdf_geometry_finetune as shared
import tools.report_gaussian_geometry_causal as metrics
from tools.report_sdf_geometry_finetune import geometry, csv_write

PARENT=ROOT/'work_dirs/sdf_geometry_replacement_full'
REPORT=ROOT/'analysis/sdf_geometry_replacement_interim_results.md'


def report(folder, record):
    metrics.RUN=folder
    (folder/'aggregate').mkdir(exist_ok=True)
    models={}; arrays={}; rows=[]; classes=[]; scenes={}
    for name in ('before','after'):
        a,s,scene_rows=metrics.aggregate(name,'val',dict(frames=6019,scenes=150))
        arrays[name]=a; scenes[name]=s
        models[name]={**metrics.metrics(a),**geometry(a)}
        for row in scene_rows:
            with np.load(folder/'eval'/name/'val'/(row['scene']+'.npz')) as data:
                row.update(geometry(data))
            rows.append(row)
        classes.extend(metrics.class_rows(name,'val',a['confusion']))
    assert np.array_equal(arrays['before']['confusion'].sum(1),arrays['after']['confusion'].sum(1))
    delta={k:models['after'][k]-models['before'][k] for k in models['before']}
    summary=dict(status='complete',checkpoint=record,validation_frames=6019,
                 validation_scenes=150,models=models,delta=delta,
                 scene_bootstrap_95=metrics.bootstrap(scenes['before'],scenes['after']))
    write(folder/'summary.json',summary)
    csv_write(folder/'scene_metrics.csv',rows);csv_write(folder/'class_metrics.csv',classes)
    def table(keys):
        text='| 指标 | 原始 tiny 权重 | 暂停前已保存权重 | 变化 |\n| --- | ---: | ---: | ---: |\n'
        for key,label in keys:
            text+=f"| {label} | {models['before'][key]:.4f} | {models['after'][key]:.4f} | {delta[key]:+.4f} |\n"
        return text
    text='# SDF 替换实验：暂停训练后的中间权重验证\n\n'
    text+=f"**完整验证已完成：150 场景、6,019 帧。评估权重训练了 {record['microbatches']:,} / 28,130 帧（{100*record['microbatches']/28130:.2f}% epoch），不是完成一整轮后的权重。**\n\n"
    text+=f"mIoU 从 **{models['before']['mIoU']:.4f}%** 变为 **{models['after']['mIoU']:.4f}%**，变化 **{delta['mIoU']:+.4f} 个百分点**；二值 IoU 变化 **{delta['IoU']:+.4f} 个百分点**。\n\n"
    text+='## 性能变化\n\n'+table([
        ('mIoU','mIoU（%）'),('IoU','二值 IoU（%）'),('FN_rate','漏检率：FN／GT 前景（%）'),
        ('FPR','误占用率：FP／GT empty（%）'),('FP_over_GTfg','FP／GT 前景（%）'),
        ('FP_over_predfg','FP／预测前景（%）'),('semantic_error','GT 前景错类率（%）')])
    text+='\n性能只在 camera mask 内汇总。未额外修改 empty bias、类别 bias 或温度。\n\n'
    text+='## 几何变化\n\n'+table([
        ('GS_camera_foreground_percent','中心位于可见前景／全部 GS（%）'),
        ('GS_camera_empty_percent','中心位于可见 empty／全部 GS（%）'),
        ('GS_camera_unknown_percent','中心位于 camera mask 外／全部 GS（%）'),
        ('SDF_center_valid_percent','中心约束有效 GS／全部 GS（%）'),
        ('SDF_shape_valid_percent','尺度约束有效 GS／全部 GS（%）'),
        ('SDF_outside_percent','中心在前景外／SDF 有效 GS（%）'),
        ('SDF_positive_distance_m','SDF 正距离均值，内部计 0（m）'),
        ('SDF_normal_scale_m','法向尺度／形状有效 GS（m）'),
        ('no_1sigma','GT 前景无 1σ 覆盖（%）'),('no_2sigma','GT 前景无 2σ 覆盖（%）'),
        ('n_1sigma','GT 前景平均 1σ 覆盖数'),('GS_sum_A','语义幅度均值'),
        ('GS_sum_alpha','不透明度均值'),('GS_sum_sigma','三轴尺度均值（m）'),
        ('GS_maxscale_ge_03999','最大尺度饱和 GS（%）')])
    text+='\nSDF 有效集合会随预测位置变化。因此，距离均值必须与有效比例一起解读，优先结合固定 GT 前景上的覆盖指标评价几何。\n\n'
    text+='## 如何理解这次结果\n\n'
    text+='本组移除了六个阶段的原始双向 Set 几何损失，仅对最终层中通过严格可靠性筛选的 GS 施加 SDF 中心和法向尺度约束。训练日志已显示其直接监督范围很小。此次验证检验这一整体替换方案的当前表现，不能单独归因于 SDF 理论是否有效。\n\n'
    text+='本轮没有同预算的原始目标继续训练对照；微调前后差值也包含继续训练的影响。训练进程仍暂停，不会因验证完成自动恢复。下一组“恢复原 Set 几何＋附加 SDF”尚未启动。\n\n'
    text+='## 复核文件\n\n'
    for label,path in [('完整汇总',folder/'summary.json'),('逐场景统计',folder/'scene_metrics.csv'),
                       ('逐类别统计',folder/'class_metrics.csv'),('评估权重',folder/'checkpoint.pth'),
                       ('快照记录',folder/'snapshot.json')]:
        text+=f'- [{label}](../{path.relative_to(ROOT)})\n'
    REPORT.write_text(text)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=PARENT/'train/latest.pth')
    args=parser.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    steps=int(state['finetune_microbatches']);updates=int(state['finetune_optimizer_updates'])
    del state
    folder=PARENT/f'interim_{steps}'
    folder.mkdir(exist_ok=True)
    snapshot=folder/'checkpoint.pth'
    if not snapshot.exists():os.link(args.checkpoint,snapshot)
    assert sha(snapshot)==sha(args.checkpoint)
    record=dict(source=str(args.checkpoint.relative_to(ROOT)),microbatches=steps,
                optimizer_updates=updates,epoch_fraction=steps/28130,
                checkpoint_sha256=sha(snapshot),initial_sha256=sha(shared.INITIAL))
    write(folder/'snapshot.json',record)
    (folder/'eval').mkdir(exist_ok=True)
    for dest,source in [(folder/'manifest.json',PARENT/'manifest.json'),
                        (folder/'eval/before',PARENT/'eval/before')]:
        if not dest.exists():dest.symlink_to(source,target_is_directory=source.is_dir())
    write(folder/'status.json',dict(status='running',phase='完整验证暂停前已保存权重',pid=os.getpid()))
    REPORT.write_text('# SDF 替换实验：中间权重验证\n\n'
        f'训练已暂停。本次评估最近保存的 {steps:,} 帧／{updates:,} 次更新权重，'
        '在完整 150 场景、6,019 帧验证集上执行。\n\n'
        '**评估正在进行，尚无完整性能结论。**\n')
    try:
        shared.RUN=folder
        import tools.eval_gaussian_geometry_causal as evaluation
        original=evaluation.evaluate
        def use_snapshot(name, checkpoint, **kwargs):
            return original(name,snapshot,**kwargs)
        evaluation.evaluate=use_snapshot
        shared.evaluate('after')
        result=report(folder,record)
        write(folder/'status.json',dict(status='complete',phase='完整验证与报告已完成',pid=os.getpid()))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as error:
        write(folder/'status.json',dict(status='failed',error=repr(error),pid=os.getpid()))
        raise


if __name__=='__main__':main()
