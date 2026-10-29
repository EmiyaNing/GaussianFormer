"""DCD mini-set experiment: 4096 training frames, full validation, readable report."""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
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
from tools.boundary_gaussian_stats.data import EvaluationFrames as FullFrames

RUN=ROOT/'work_dirs/dcd_mini_1ep'
CONFIG=ROOT/'config/mini_dcd_finetune/opusv1_gaussian_tiny_8f_dcd_1ep.py'
INITIAL=ROOT/'ckpts/opus_gaussian_8f_tiny.pth'
REPORT=ROOT/'analysis/dcd_mini_1ep_results.md'


def progress_report():
    if (RUN/'summary.json').exists():return
    status=json.loads((RUN/'status.json').read_text()) if (RUN/'status.json').exists() else {}
    p=RUN/'train/progress.json';training=json.loads(p.read_text()) if p.exists() else {}
    text='# DCD 几何替换：mini set 微调结果\n\n'
    text+='**实验尚未全部完成；当前不报告微调后性能结论。**\n\n'
    text+=f"当前阶段：{status.get('phase','准备')}；状态：{status.get('status','running')}。\n\n"
    if training:
        text+=f"训练进度：{training.get('microbatches',training.get('frames',0)):,}／4,096 帧；更新次数：{training.get('optimizer_updates',0):,}。\n\n"
        if training.get('status')=='running':text+=f"当前估计剩余训练 {training.get('remaining_hours',0):.2f} 小时，之后自动验证。\n\n"
    if status.get('error'):text+=f"错误：`{status['error']}`。\n\n"
    text+='训练使用既定 4,096 帧、验证使用完整验证集 6,019 帧。保留 8 帧 tiny 架构、Set 分类与 Occupancy 损失，只替换六阶段 Set 几何距离。\n\n'
    text+='[实验设计](dcd_mini_1ep_design.md) · [运行状态](../work_dirs/dcd_mini_1ep/status.json)\n'
    REPORT.write_text(text)


def evaluate(name):
    core.CONFIG=CONFIG;evaluation.RUN=RUN
    checkpoint=INITIAL if name=='before' else RUN/'train/epoch_1.pth'
    complete=RUN/'eval'/name/'complete.json'
    if name=='before' and not complete.exists():
        baseline=ROOT/'work_dirs/sdf_geometry_replacement_full/eval/before'
        record=json.loads((baseline/'complete.json').read_text())
        assert record['checkpoint_sha256']==core.sha(INITIAL)
        paths=list((baseline/'val').glob('*.json'))
        assert len(paths)==150 and sum(json.loads(p.read_text())['frames'] for p in paths)==6019
        complete.parent.parent.mkdir(parents=True,exist_ok=True)
        complete.parent.symlink_to(baseline,target_is_directory=True)
        core.write(RUN/'baseline_reuse.json',dict(source=str(baseline),checkpoint_sha256=record['checkpoint_sha256'],frames=6019,scenes=150))
    if complete.exists():
        record=json.loads(complete.read_text());assert record['checkpoint_sha256']==core.sha(checkpoint)
        paths=list((complete.parent/'val').glob('*.json'))
        assert len(paths)==150 and sum(json.loads(p.read_text())['frames'] for p in paths)==6019
        return
    evaluation.EvaluationFrames=FullFrames
    evaluation.evaluate(name,checkpoint,workers=4,splits=('val',))


def report():
    base.RUN=RUN;(RUN/'aggregate').mkdir(exist_ok=True)
    arrays={};scenes={};models={};rows=[];classes=[]
    for name in ('before','after'):
        a,s,r=base.aggregate(name,'val',dict(frames=6019,scenes=150))
        arrays[name]=a;scenes[name]=s;models[name]=base.metrics(a);rows.extend(r)
        classes.extend(base.class_rows(name,'val',a['confusion']))
    assert np.array_equal(arrays['before']['confusion'].sum(1),arrays['after']['confusion'].sum(1))
    delta={k:models['after'][k]-models['before'][k] for k in models['before']}
    interval=base.bootstrap(scenes['before'],scenes['after'])
    result=dict(status='complete',train_frames=4096,optimizer_updates=1024,epochs=1,
        val_frames=6019,val_scenes=150,models=models,delta=delta,scene_bootstrap_95=interval)
    core.write(RUN/'summary.json',result)
    for path,data in [(RUN/'scene_metrics.csv',rows),(RUN/'class_metrics.csv',classes)]:
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    def table(keys):
        text='| 指标 | 原始 tiny | DCD 微调 1 epoch | 差值 |\n| --- | ---: | ---: | ---: |\n'
        for k,label in keys:text+=f"| {label} | {models['before'][k]:.5f} | {models['after'][k]:.5f} | {delta[k]:+.5f} |\n"
        return text
    text='# DCD 几何替换：mini set 微调结果\n\n'
    text+='**已完成：4,096 个训练帧恰好 1 epoch、1,024 次更新；微调前后均评估完整验证集的 6,019 帧、150 场景。**\n\n'
    text+=f"mIoU 从 {models['before']['mIoU']:.5f}% 变为 **{models['after']['mIoU']:.5f}%**，变化 **{delta['mIoU']:+.5f} 个百分点**。\n\n"
    text+='## 性能变化\n\n'+table([('mIoU','mIoU（%）'),('IoU','二值 IoU（%）'),
        ('FN_rate','漏检率：FN／GT 前景（%）'),('FPR','误占用率：FP／GT empty（%）'),
        ('FP_over_GTfg','FP／GT 前景（%）'),('FP_over_predfg','FP／预测前景（%）'),
        ('semantic_error','GT 前景错类率（%）')])
    text+='\n## 高斯分布与覆盖\n\n'+table([('no_1sigma','GT 前景无 1σ 核心覆盖（%）'),
        ('no_2sigma','GT 前景无 2σ 覆盖（%）'),('n_1sigma','GT 前景平均 1σ 覆盖数'),
        ('n_2sigma','GT 前景平均 2σ 覆盖数'),('GS_center_camera','可见中心／全部 GS（%）'),
        ('GS_center_camera_empty','可见 empty 中心／全部 GS（%）'),('GS_sum_sigma','GS 平均尺度（m）'),
        ('GS_sum_A','GS 总语义幅度'),('GS_sum_alpha','GS 不透明度')])
    text+='\n## 怎样解释这次结果\n\n'
    if delta['no_1sigma']<0:text+='固定 GT 前景上的核心覆盖缺口有所减少，需结合漏检和 mIoU 判断是否转化为性能收益。\n\n'
    else:text+='固定 GT 前景上的核心覆盖缺口没有减少，本轮尚未证明 DCD 改善了核心覆盖。\n\n'
    text+='原 Set 分类与 Occupancy 损失保持原值，六阶段双向几何项改用 DCD，并保留 OPUS 原 GT 加权。模型、渲染器、empty 参数的学习方式不变；未叠加此前覆盖或 SDF 附加项，也未额外校准输出。\n\n'
    text+='本轮仅有一条微调轨迹，没有相同预算的原 Set 继续训练对照。前后差值包含继续训练与几何目标改变的共同影响，不能单独视为 DCD 的净因果收益。本轮训练范围为 mini set，验证范围为完整验证集；不将其与全训练集微调的训练预算混同。\n\n'
    text+='## 配对场景区间\n\n| 指标差值 | bootstrap 95% 区间 |\n| --- | --- |\n'
    for k,v in interval.items():text+=f'| {k} | [{v[0]:+.5f}, {v[1]:+.5f}] |\n'
    text+='\n区间只描述这两份权重的场景组成不确定性，不包含训练 seed 方差。\n\n'
    text+='## 复核资料\n\n'
    for label,path in [('实验设计',ROOT/'analysis/dcd_mini_1ep_design.md'),('汇总数据',RUN/'summary.json'),
        ('逐场景指标',RUN/'scene_metrics.csv'),('逐类别指标',RUN/'class_metrics.csv'),
        ('微调权重',RUN/'train/epoch_1.pth'),('训练完成记录',RUN/'train/complete.json')]:
        text+=f'- [{label}](../{path.relative_to(ROOT)})\n'
    REPORT.write_text(text)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def queue():
    RUN.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),CONFIG,ROOT/'tools/train_dcd_mini.py',ROOT/'loss/opus_set_dcd_loss.py',
        ROOT/'loss/opus_set_loss.py',ROOT/'tools/gaussian_geometry_causal.py',
        ROOT/'tools/eval_gaussian_geometry_causal.py',ROOT/'tools/boundary_gaussian_stats/data.py']
    protocol=dict(initial_sha256=core.sha(INITIAL),config=Config.fromfile(CONFIG).to_dict(),
        sources={str(p.relative_to(ROOT)):core.sha(p) for p in files},
        train_annotation_sha256=core.sha(ROOT/'data/mini_occ3d_4096_1024/train.pkl'),
        val_annotation_sha256=core.sha(ROOT/'data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl'))
    p=RUN/'experiment_protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(protocol))
    else:
        core.write(p,protocol)
        for source in files:
            dest=RUN/'sources'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    try:
        for action,phase in [('before','原始权重：复用已核验的完整验证基线'),
            ('train','DCD 几何：4,096 帧微调 1 epoch'),('after','微调权重：完整验证 6,019 帧与几何统计'),
            ('report','汇总性能与几何变化')]:
            core.write(RUN/'status.json',dict(status='running',phase=phase,pid=os.getpid()))
            progress_report()
            command=([sys.executable,'-u',str(ROOT/'tools/train_dcd_mini.py'),'train'] if action=='train'
                     else [sys.executable,'-u',str(Path(__file__)),action])
            with (RUN/(action+'_console.log')).open('ab',buffering=0) as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    progress_report();time.sleep(10)
                if child.returncode:raise RuntimeError(f'{action} failed with code {child.returncode}')
        core.write(RUN/'status.json',dict(status='complete',phase='微调、验证和报告已完成',pid=os.getpid()))
    except BaseException as error:
        core.write(RUN/'status.json',dict(status='failed',error=repr(error),pid=os.getpid()));progress_report();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',choices=['queue','before','after','report'],default='queue')
    args=p.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    if args.action=='queue':
        with (RUN/'experiment.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);queue()
    elif args.action=='report':report()
    else:evaluate(args.action)


if __name__=='__main__':main()
