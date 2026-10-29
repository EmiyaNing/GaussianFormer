"""Full-val diagnostics of the trained core-coverage auxiliary model."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from mmengine import Config
import tools.gaussian_geometry_causal as core
import tools.eval_gaussian_geometry_causal as evaluation
import tools.report_gaussian_geometry_causal as base

TRAIN=ROOT/'work_dirs/coverage_full_tiny_1ep'
RUN=TRAIN/'diagnostics'
REPORT=ROOT/'analysis/core_coverage_full_finetune_results.md'
CONFIG=ROOT/'config/full_occ3d_coverage/opusv1_gaussian_tiny_8f_coverage_1ep.py'
BASELINE=ROOT/'work_dirs/sdf_geometry_replacement_full/eval/before'


def csv_write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report():
    base.RUN=RUN;(RUN/'aggregate').mkdir(exist_ok=True)
    arrays={};scenes={};models={};scene_rows=[];class_rows=[]
    for name in ('before','after'):
        a,s,rows=base.aggregate(name,'val',dict(frames=6019,scenes=150))
        arrays[name]=a;scenes[name]=s;models[name]=base.metrics(a)
        scene_rows.extend(rows);class_rows.extend(base.class_rows(name,'val',a['confusion']))
    assert np.array_equal(arrays['before']['confusion'].sum(1),arrays['after']['confusion'].sum(1))
    assert int(arrays['after']['loss'][2])==6019
    native=(TRAIN/'eval/console.log').read_text()
    native_miou=float(re.findall(r'mIoU of 6019 samples:\s*([\d.]+)',native)[-1])
    assert abs(models['after']['mIoU']-native_miou)<.0051,'Diagnostic/native predictions differ'
    interval=base.bootstrap(scenes['before'],scenes['after'])
    delta={k:models['after'][k]-models['before'][k] for k in models['before']}
    traces=[json.loads(x) for x in (TRAIN/'train/trace.jsonl').read_text().splitlines()]
    assert len(traces)==28130 and len({x['index'] for x in traces})==28130
    def trace_summary(rows):
        k=['GaussianCoreCoverageLoss','GaussianCoreCoverageLoss/raw_loss',
           'GaussianCoreCoverageLoss/no_core_fraction','GaussianCoreCoverageLoss/mean_gap_weight',
           'GaussianCoreCoverageLoss/active_gt','OPUSSetLoss/loss_pts']
        return {name:float(np.mean([r['losses'][name] for r in rows])) for name in k}
    training={name:trace_summary(rows) for name,rows in [('all',traces),('first1000',traces[:1000]),('last1000',traces[-1000:])]}
    training['coverage_to_total_set_geometry_scalar_ratio']=training['all']['GaussianCoreCoverageLoss']/(6*training['all']['OPUSSetLoss/loss_pts'])
    by_scene={name:{r['scene']:r for r in scene_rows if r['name']==name} for name in models}
    signs={key:{'increased':sum(by_scene['after'][s][key]>by_scene['before'][s][key] for s in by_scene['before']),
                'decreased':sum(by_scene['after'][s][key]<by_scene['before'][s][key] for s in by_scene['before'])}
           for key in ['mIoU','IoU','FN_rate','FPR','no_1sigma','n_1sigma','GS_center_camera_empty']}
    result=dict(status='complete',validation_scenes=150,validation_frames=6019,models=models,delta=delta,
                scene_bootstrap_95=interval,scene_sign_counts=signs,training=training,
                native_miou_rounded=native_miou,diagnostic_matches_native_at_logged_precision=True)
    core.write(RUN/'summary.json',result)
    csv_write(RUN/'scene_metrics.csv',scene_rows);csv_write(RUN/'class_metrics.csv',class_rows)
    def table(keys):
        out='| 指标 | 原始 tiny | 完整目标＋覆盖项 | 差值 |\n| --- | ---: | ---: | ---: |\n'
        for key,label in keys:
            out+=f"| {label} | {models['before'][key]:.5f} | {models['after'][key]:.5f} | {delta[key]:+.5f} |\n"
        return out
    text='# 完整目标＋核心覆盖附加项：完整验证诊断\n\n'
    text+='**训练与诊断均已完成。微调覆盖 700 场景、28,130 帧，恰好 1 epoch、7,033 次更新；诊断覆盖完整 150 场景、6,019 帧。**\n\n'
    text+=f"mIoU 为 **{models['after']['mIoU']:.5f}%**，相对原始权重变化 **{delta['mIoU']:+.5f} 个百分点**；二值 IoU 变化 **{delta['IoU']:+.5f} 个百分点**。诊断 mIoU 与原生 eval 日志的 {native_miou:.2f}% 在记录精度内一致。\n\n"
    text+='## 1. 性能：哪些错误改变了\n\n'+table([
        ('mIoU','mIoU（%）'),('IoU','二值 IoU（%）'),('FN_rate','漏检率：FN／GT 前景（%）'),
        ('FPR','误占用率：FP／GT empty（%）'),('FP_over_GTfg','FP／GT 前景（%）'),
        ('FP_over_predfg','FP／预测前景（%）'),('semantic_error','GT 前景错类率（%）'),
        ('detected_semantic_accuracy','已检出前景的语义正确率（%）')])
    text+=f"\nFN 净变化 {int(delta['FN']):+,} 个体素；FP 净变化 {int(delta['FP']):+,} 个体素。它们是两次预测计数的净差，不是逐体素追踪的救回／新增错误数量。\n\n"
    text+='## 2. 几何：核心覆盖是否真正改善\n\n'+table([
        ('no_1sigma','GT 前景没有 1σ 覆盖（%）'),('no_2sigma','GT 前景没有 2σ 覆盖（%）'),
        ('no_native_support','GT 前景没有原生渲染支持（%）'),('n_1sigma','GT 前景平均 1σ 覆盖数'),
        ('n_2sigma','GT 前景平均 2σ 覆盖数'),('kernel_mass','GT 前景高斯核权重和均值'),
        ('GS_center_camera','中心可见／全部 GS（%）'),('GS_center_camera_empty','中心位于可见 empty／全部 GS（%）'),
        ('GS_sum_sigma','GS 三轴尺度均值（m）'),('GS_maxscale_ge_03999','最大尺度饱和 GS（%）'),
        ('GS_sum_A','GS 总语义幅度均值'),('GS_sum_alpha','GS 不透明度均值'),
        ('semantic_amplitude','有效前景上的贡献加权语义幅度'),('class_concentration','有效前景上的类别集中度')])
    text+='\n覆盖统计在相同的全部 camera-valid GT 前景上进行，不以新模型检出或漏检集合为固定比较对象。这里使用实际原生候选范围和非零核贡献；训练覆盖权重使用不依赖不透明度的纯几何椭球，两者定义需区分。\n\n'
    text+='## 3. 训练中附加项的实际强度\n\n'
    t=training['all'];first=training['first1000'];last=training['last1000']
    text+=f"每帧平均 {t['GaussianCoreCoverageLoss/active_gt']:.1f} 个前景体素参与附加吸引。逐帧平均核心缺口比例为 {100*t['GaussianCoreCoverageLoss/no_core_fraction']:.4f}%；最初 1,000 帧为 {100*first['GaussianCoreCoverageLoss/no_core_fraction']:.4f}%，最后 1,000 帧为 {100*last['GaussianCoreCoverageLoss/no_core_fraction']:.4f}%。前后帧不同，不能把这两个数直接当作固定训练样本的学习曲线。\n\n"
    text+=f"附加项加权后的平均值为 {t['GaussianCoreCoverageLoss']:.6f}，约为六阶段 Set 几何项标量总和的 {100*training['coverage_to_total_set_geometry_scalar_ratio']:.4f}%。这提示干预强度可能较小，但损失值比例不等于中心梯度比例；本轮没有保存逐损失参数梯度，不能据此认定梯度主导关系。\n\n"
    text+='## 4. 场景差异与结果解释\n\n| 指标差值 | 配对场景 bootstrap 95% 区间 |\n| --- | --- |\n'
    for k,v in interval.items():text+=f'| {k} | [{v[0]:+.5f}, {v[1]:+.5f}] |\n'
    text+=f"\nmIoU 上升／下降的场景数为 {signs['mIoU']['increased']}／{signs['mIoU']['decreased']}；无 1σ 覆盖比例下降／上升的场景数为 {signs['no_1sigma']['decreased']}／{signs['no_1sigma']['increased']}。区间描述固定权重间的场景不确定性，不包含多次训练的方差。\n\n"
    if delta['no_1sigma']<0:
        text+='新增方案减少了全局核心覆盖缺口。仍需结合变化幅度、漏检和 mIoU，区分可测的几何变化与实际性能收益。\n\n'
    else:text+='新增方案尚未减少全局核心覆盖缺口，因此没有达到本轮最直接的几何目标。\n\n'
    if interval['mIoU'][0]<=0<=interval['mIoU'][1]:
        text+='mIoU 差值的场景区间跨过零，目前没有确认可靠的性能提升。\n\n'
    elif delta['mIoU']<0:text+='mIoU 差值区间位于零以下，当前两份权重的验证表现支持退化，而非收益。\n\n'
    else:text+='mIoU 差值区间位于零以上，支持当前两份权重在验证场景上存在性能提升。\n\n'
    text+='本轮没有相同预算、只保留原始完整目标继续训练的对照，因此不能单独分离继续训练与覆盖附加项的净贡献。当前附加项只吸引缺口附近的最近中心，未显式招募远处低贡献 GS，也没有限制局部高斯预算集中，不能将其视为已经实现全局预算重新分配。\n\n'
    text+='## 5. 逐类别结果\n\n| 类别 | 原始 IoU（%） | 新 IoU（%） | 差值（百分点） |\n| --- | ---: | ---: | ---: |\n'
    old={x['class_id']:x for x in class_rows if x['model']=='before'}
    for x in class_rows:
        if x['model']=='after':
            b=old[x['class_id']]['IoU_percent'];a=x['IoU_percent']
            text+=f"| {x['class_name']} | {b:.4f} | {a:.4f} | {a-b:+.4f} |\n"
    text+='\n## 6. 复核资料\n\n'
    for label,path in [('汇总结果',RUN/'summary.json'),('逐场景指标',RUN/'scene_metrics.csv'),
                       ('逐类别计数',RUN/'class_metrics.csv'),('诊断协议',RUN/'protocol.json'),
                       ('训练完成记录',TRAIN/'train/complete.json'),('原生 eval 日志',TRAIN/'eval/console.log')]:
        text+=f'- [{label}](../{path.relative_to(ROOT)})\n'
    REPORT.write_text(text)
    print(json.dumps(dict(status='complete',models=models,delta=delta),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=['run','report'],default='run')
    args=parser.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    if args.action=='report':report();return
    train_protocol=json.loads((TRAIN/'protocol.json').read_text())
    for p,h in train_protocol['source_sha256'].items():assert core.sha(ROOT/p)==h,p
    checkpoint=TRAIN/'train/epoch_1.pth'
    native=json.loads((TRAIN/'eval/complete.json').read_text())
    assert core.sha(checkpoint)==native['checkpoint_sha256']
    before=json.loads((BASELINE/'complete.json').read_text())
    assert before['checkpoint_sha256']==train_protocol['initial_sha256']
    protocol=dict(checkpoint_sha256=native['checkpoint_sha256'],initial_sha256=train_protocol['initial_sha256'],
        train_protocol_sha256=core.sha(TRAIN/'protocol.json'),config_sha256=core.sha(CONFIG),
        sources={str(p.relative_to(ROOT)):core.sha(p) for p in [Path(__file__),ROOT/'tools/eval_gaussian_geometry_causal.py',ROOT/'tools/stat_gaussian_voxel_coverage.py']},
        frames=6019,scenes=150,calibration='none',training=False)
    if (RUN/'protocol.json').exists():assert json.loads((RUN/'protocol.json').read_text())==protocol
    else:core.write(RUN/'protocol.json',protocol)
    (RUN/'eval').mkdir(exist_ok=True)
    target=RUN/'eval/before'
    if not target.exists():target.symlink_to(BASELINE,target_is_directory=True)
    core.CONFIG=CONFIG;evaluation.RUN=RUN
    core.write(RUN/'status.json',dict(status='running',phase='完整验证几何与错误诊断',pid=os.getpid()))
    REPORT.write_text('# 完整目标＋覆盖附加项：诊断进行中\n\n'
        '已完成完整 1 epoch 训练，原生完整验证 mIoU 为 39.81%。'
        '目前正在补充完整 150 场景／6,019 帧的混淆矩阵、核心覆盖和高斯分布统计；详细结论等待诊断完成。\n')
    try:
        evaluation.evaluate('after',checkpoint,workers=4,splits=('val',))
        report()
        core.write(RUN/'status.json',dict(status='complete',phase='完整诊断与报告完成',pid=os.getpid()))
    except BaseException as error:
        core.write(RUN/'status.json',dict(status='failed',error=repr(error),pid=os.getpid()));raise


if __name__=='__main__':main()
