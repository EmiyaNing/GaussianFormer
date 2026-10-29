"""Matched full-validation report for original CD plus two literal center terms."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.semantic_neighborhood_experiment import *
from tools.report_semantic_field_mini import metrics as original_metrics,paired_interval
from tools.report_gaussian_geometry_causal import CLASS_NAMES
ARM=ARMS[0];NAMES=('baseline',ARM)
REPORT=ROOT/'analysis/semantic_neighborhood_mean_mini_results.md'
LABELS={'baseline':'B：原目标继续训练',ARM:'M：原目标＋两个新项'}

def progress():
    lines=['# 3σ同类平均距离与1σ未覆盖吸引：实验结果','',
        '**状态：实验尚未完成，暂不判断方案是否有效。**','',
        '本轮保留原有六级双向CD、Set分类和Occupancy损失，移除上一轮新增项，仅添加两个中心距离目标。设计见[实验设计](semantic_neighborhood_mean_mini_design.md)。','',
        '| 对象 | 训练 | 本轮完整验证 |','| --- | --- | --- |']
    for n in NAMES:
        t=RUN/'train'/n/'progress.json';e=RUN/'eval'/n/'progress.json'
        td=json.loads(t.read_text()) if t.exists() else {};ed=json.loads(e.read_text()) if e.exists() else {}
        train='复用已核验同预算权重' if n=='baseline' else ('4096/4096帧' if td.get('status')=='complete' else f"{td.get('microbatches',0)}/4096帧")
        val='6019/6019帧' if ed.get('status')=='complete' else f"{ed.get('scenes',0)}/150场景"
        lines.append(f'| {LABELS[n]} | {train} | {val} |')
    q=RUN/'queue_progress.json'
    if q.exists():
        state=json.loads(q.read_text());lines+=['',f"当前阶段：`{state.get('phase','准备')}`；状态：`{state.get('status','准备')}`。"]
        if state.get('status')=='failed':lines+=['',f"运行错误：`{state.get('error')}`。"]
    lines+=['','全部结果按完整6019帧验证集汇总；不会用训练损失下降代替有效性结论。','']
    REPORT.write_text('\n'.join(lines))

def aggregate(name):
    folder=RUN/'eval'/name;meta=json.loads((folder/'complete.json').read_text())
    paths=sorted((folder/'scenes').glob('*.npz'));assert len(paths)==150
    total={};scenes=[];indices=[];loads=[];cores=[];tokens=[]
    for path in paths:
        info=json.loads(path.with_suffix('.json').read_text());indices+=info['indices'];tokens+=info['tokens']
        assert info['checkpoint_sha256']==meta['checkpoint_sha256']
        with np.load(path) as z:
            row={k:z[k].sum(0) for k in z.files if k not in ('load','query_core')}
            loads.append(z['load']);cores.append(z['query_core'])
        for k,v in row.items():total[k]=total.get(k,np.zeros_like(v))+v
        scenes.append((path.stem,row))
    assert sorted(indices)==list(range(6019)) and len(set(tokens))==6019
    folder=RUN/'aggregate';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{name}.npz',**total,load=np.concatenate(loads),query_core=np.concatenate(cores))
    return total,scenes

def metrics(a):
    m=original_metrics(a);cc=a['class_coverage'];gt=cc[:,0];other=np.arange(17)!=11
    m['macro_core1']=100*float((cc[:,1]/gt).mean())
    m['nonroad_core1']=100*float(cc[other,1].sum()/gt[other].sum())
    n,j,active,pairs,mean_total,u,uncovered_total,same,idle,zero=a['aux_geometry']
    m.update(active_same3_percent=100*active/max(n,1),same3_points_per_active_GS=pairs/max(active,1),
        mean3_m=mean_total/max(n,1),mean3_active_m=mean_total/max(active,1),
        uncovered1_m=uncovered_total/max(j,1),uncovered_center_distance_m=uncovered_total/max(u,1),
        loss_uncovered_GT_percent=100*u/max(j,1),loss_same_core_percent=100*same/max(j,1),
        idle_core_GS_percent=100*idle/max(n,1),opacity_zero_percent=100*zero/max(n,1))
    return {k:float(v) for k,v in m.items()}

def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def plot(m,interval):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b,v=m['baseline'],m[ARM]
    panels=[('mIoU','mIoU','mIoU_delta_95CI'),('geo1','Geometric core coverage','geo1_delta_95CI'),
        ('same1','Predicted same-class core coverage',None),('FN_rate','Misses / GT foreground',None),
        ('FP_over_GTfg','False positives / GT foreground',None),('high_core','Initial high-load GT: core coverage',None)]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(10.5,5.7),constrained_layout=True)
    for ax,(key,title,ci) in zip(axes.flat,panels):
        value=v[key]-b[key];ax.axhline(0,color='.5',lw=.8);ax.bar([0],[value],width=.4,color='#397aa4')
        anchor=value
        if ci:
            lo,hi=interval[ci];ax.vlines(0,lo,hi,color='black');ax.scatter([0,0],[lo,hi],marker='_',color='black')
            anchor=max(value,hi) if value>=0 else min(value,lo)
        ax.annotate(f'{value:+.3f}',(0,anchor),xytext=(0,5 if value>=0 else -5),textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top')
        ax.set_xticks([0],['Two added center terms']);ax.set_ylabel('Change vs baseline (pp)');ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
        low,high=ax.get_ylim();margin=max(.04,(high-low)*.3);ax.set_ylim(low-margin,high+margin);ax.set_xlim(-.6,.6)
    fig.suptitle('Original CD retained: full Occ3D validation after 4096-frame fine-tuning')
    folder=ROOT/'analysis/figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(folder/f'semantic_neighborhood_mean_mini_comparison.{ext}',dpi=180)
    plt.close(fig)

def report():
    if not all((RUN/'eval'/n/'complete.json').exists() for n in NAMES):progress();return
    arrays={};scenes={};m={}
    for n in NAMES:arrays[n],scenes[n]=aggregate(n);m[n]=metrics(arrays[n])
    old=np.load(REFERENCE_RUN/'aggregate/baseline.npz')
    np.testing.assert_array_equal(old['confusion'],arrays['baseline']['confusion'])
    np.testing.assert_array_equal(old['class_coverage'],arrays['baseline']['class_coverage'])
    interval=paired_interval(scenes['baseline'],scenes[ARM]);csv_write(RUN/'aggregate/metrics.csv',[dict(arm=n,**m[n]) for n in NAMES])
    rows=[]
    for n in NAMES:
        cm=arrays[n]['confusion'];cc=arrays[n]['class_coverage']
        for c,name in enumerate(CLASS_NAMES):
            tp=cm[c,c];union=cm[c].sum()+cm[:,c].sum()-tp;gt=cc[c,0]
            rows.append(dict(arm=n,class_id=c,name=name,GT=int(gt),IoU=100*tp/max(union,1),
                core1=100*cc[c,1]/max(gt,1),same_core1=100*cc[c,4]/max(gt,1),FN=100*cm[c,17]/max(gt,1),FP_from_empty=int(cm[17,c])))
    csv_write(RUN/'aggregate/per_class.csv',rows)
    trace=[json.loads(x) for x in (RUN/'train'/ARM/'trace.jsonl').read_text().splitlines()]
    oldtrace=[json.loads(x) for x in (REFERENCE_RUN/'train/baseline/trace.jsonl').read_text().splitlines()]
    assert len(trace)==len(oldtrace)==4096 and [x['index'] for x in trace]==[x['index'] for x in oldtrace]
    first_error=max(abs(trace[0]['losses'][k]-v) for k,v in oldtrace[0]['losses'].items());assert first_error<1e-5
    train_mean={k:float(np.mean([x['losses'][k] for x in trace])) for k in trace[0]['losses'] if k.startswith('SemanticNeighborhoodMeanLoss')}
    summary=dict(status='complete',metrics=m,paired_vs_baseline=interval,train_frame_means=train_mean,
        original_first_frame_max_difference=first_error,baseline_historical_confusion_exact=True,
        baseline_historical_coverage_exact=True,matched_training_order=True)
    write(RUN/'summary.json',summary);plot(m,interval)
    b,v=m['baseline'],m[ARM];dm=v['mIoU']-b['mIoU'];dg=v['geo1']-b['geo1']
    if interval['mIoU_delta_95CI'][0]>0 and dg>0:
        verdict='本轮同时观察到性能和核心覆盖收益；是否缓解query压力，还需结合固定高负载区域与负载集中度判断。'
    elif dg<=0:
        verdict='本轮没有改善整体核心覆盖，尚不支持这两个附加项能够解决query负载压力。'
    else:verdict='本轮核心覆盖有所提高，但mIoU收益尚不明确；不能据此认定已经解决query负载压力。'
    lines=['# 3σ同类平均距离与1σ未覆盖吸引：实验结果','',f'**{verdict}**','',
        f'相对同预算原目标基线，mIoU变化{dm:+.3f}个百分点，1σ几何覆盖变化{dg:+.3f}个百分点。训练为既有4096帧mini集、1epoch；评估覆盖完整6019帧、150场景，均使用camera有效区域。','',
        '![相对基线的完整验证变化](figures/semantic_neighborhood_mean_mini_comparison.png)','',
        '## 1. 本轮实际改了什么','',
        '保留基础六级双向CD、Set分类和Occupancy CE＋Lovasz，去掉上一轮新增的覆盖、局部区域拟合和泄漏项。最终级只增加两个中心目标：GS到严格3σ内全部同预测类别GT点的平均欧氏距离；GT未被任何GS的1σ覆盖时，到最近GS中心的欧氏距离。两项权重均为0.5。', '',
        'GS没有局部同类GT时，该新增项记零；原有CD仍提供监督。已覆盖GT停止的是新增吸引，基础CD保持原样。离散类别、3σ集合和1σ掩码在当前反向固定；新增项不直接监督尺度或旋转。公式和空邻域规则见[实验设计](semantic_neighborhood_mean_mini_design.md)。','',
        '## 2. 预测性能','', '| 模型 | mIoU | 二值IoU | FN/GT前景 | FP/GT前景 | FP/预测前景 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        z=m[n];lines.append(f"| {LABELS[n]} | {z['mIoU']:.3f} | {z['IoU']:.3f} | {z['FN_rate']:.3f}% | {z['FP_over_GTfg']:.3f}% | {z['FP_over_predfg']:.3f}% |")
    lo,hi=interval['mIoU_delta_95CI'];cl,ch=interval['geo1_delta_95CI']
    lines+=['',f'mIoU差值的场景配对95%区间为[{lo:+.3f}, {hi:+.3f}]个百分点；1σ覆盖差值为[{cl:+.3f}, {ch:+.3f}]个百分点。按150个场景配对重采样1500次，seed928；该区间不能替代多训练种子的方差评估。','',
        '## 3. 几何与同类覆盖','', '| 模型 | 1σ | 2σ | 严格3σ | 同预测类别1σ | 17类等权1σ | 非道路1σ |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        z=m[n];lines.append(f"| {LABELS[n]} | {z['geo1']:.3f}% | {z['geo2']:.3f}% | {z['geo3']:.3f}% | {z['same1']:.3f}% | {z['macro_core1']:.3f}% | {z['nonroad_core1']:.3f}% |")
    p=arrays[ARM]['paired_baseline'];pp=100*p[:4]/p[4]
    lines+=['',f'相对B，新获1σ几何覆盖占GT前景{pp[0]:.3f}%，丢失{pp[1]:.3f}%；新获同类核心覆盖{pp[2]:.3f}%，丢失{pp[3]:.3f}%。新获与丢失均在同一固定GT集合上统计。','',
        '每个GT被多个GS覆盖时只计一次。同类覆盖要求GS的当前预测类别等于GT标签；单纯改变类别也可能改变同类覆盖，因此与纯几何覆盖分开报告。','',
        '## 4. Query负载是否缓解','', '| 模型 | 最近归属top20%份额 | 零核心接触query | 接触≤20体素query | 固定高负载区核心覆盖 | 固定高负载区二值召回 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        z=m[n];lines.append(f"| {LABELS[n]} | {z['top20_nearest_share']:.3f}% | {z['zero_core']:.3f}% | {z['core_le20']:.3f}% | {z['high_core']:.3f}% | {z['high_recall']:.3f}% |")
    lines+=['','最近归属按GT最近的第六级GS回溯父query；实际接触按该query的64个最终GS的1σ覆盖并集计数。两者分开，最近归属集中度下降本身不代表新增覆盖。高负载GT区域固定为同一初始模型中top20% query对应的GT集合，不随新模型重新划分。','',
        '本轮未改变1200个query、每query64个GS的结构。即使出现覆盖收益，也应区分几何位置改善与真正解除固定输出容量限制。','',
        '## 5. 新增项是否在真正拟合同类点、补足缺口','', '| 模型 | 有同类3σ目标的GS比例 | 每个有效GS的同类GT数 | GS均距/全部GS | GS均距/有效GS | 未覆盖GT比例 | 未覆盖GT最近中心距 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        z=m[n];lines.append(f"| {LABELS[n]} | {z['active_same3_percent']:.3f}% | {z['same3_points_per_active_GS']:.3f} | {z['mean3_m']:.5f}m | {z['mean3_active_m']:.5f}m | {z['loss_uncovered_GT_percent']:.3f}% | {z['uncovered_center_distance_m']:.5f}m |")
    lines+=['','这张表在完整验证集上对两个模型使用同样规则重新计算。全部GS均距变低，可能来自有效GS减少，不能单独解释为贴合同类区域；有效GS均距和目标点数同时列出。所有GT、GS分别按固定总数归一化，不因有效集合变小而重新归一。','',
        '## 6. 中心、尺度与语义幅度','', '| 模型 | 中心在有效前景 | 中心在有效empty | 中心在未知区 | 平均最大轴 | 最大轴饱和 | 平均opacity | 平均前景幅度和 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        z=m[n];lines.append(f"| {LABELS[n]} | {z['center_camera_fg']:.3f}% | {z['center_camera_empty']:.3f}% | {z['center_unknown']:.3f}% | {z['major_scale']:.5f}m | {z['major_scale_saturated']:.3f}% | {z['opacity']:.5f} | {z['semantic_amplitude']:.5f} |")
    lines+=['','未知包括camera无效、标签无效或场景外。新增两项的直接梯度只到中心；尺度与语义等属性仍可能经Occupancy及共享网络间接改变。硬筛选使目标分段可导，在1σ和3σ集合切换处不保证连续，不能宣称全局光滑。','',
        '## 7. 对照、实现与可追溯性','',
        f'基线复用上一轮已核验的同初始化、同4096帧顺序、同优化器和调度、同1epoch预算的继续训练权重，本轮重新完成其6019帧验证以计算新增目标诊断。新旧基线混淆矩阵与核心覆盖逐项一致。新实验首帧所有原有损失最大差值为{first_error:.8f}，数据顺序也完全一致。没有再次训练一个相同基线。','',
        '数值检查包括GPU全邻域与FP64穷举、中心梯度、CUDA有限差分、均距与质心距离的区别、空同类集合、最近GS漏覆盖但较远GS已覆盖的情况。4个预定训练帧上验证总损失联合反向有限，未使用验证结果调权重。','',
        '本轮只是单种子微调，不能把负结果扩大为所有局部集合监督都无效，也不能把正结果直接写成固定配额的因果证明。','',
        '- [执行协议](../work_dirs/semantic_neighborhood_mean_mini_v1/protocol.json)；[完整指标与区间](../work_dirs/semantic_neighborhood_mean_mini_v1/summary.json)。',
        '- [逐类CSV](../work_dirs/semantic_neighborhood_mean_mini_v1/aggregate/per_class.csv)；[数值检查](../work_dirs/semantic_neighborhood_mean_mini_v1/numerical_checks.json)；[真实帧梯度检查](../work_dirs/semantic_neighborhood_mean_mini_v1/pilot.json)。',
        '- 权重：`work_dirs/semantic_neighborhood_mean_mini_v1/train/mean3_uncovered1/epoch_1.pth`；逐帧日志与逐场景数组位于同一实验目录。','']
    REPORT.write_text('\n'.join(lines));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':report()
