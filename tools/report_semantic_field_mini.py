"""Readable full-validation report for the matched local semantic field experiment."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.semantic_field_experiment_common import RUN,ARMS,write,sha
from tools.report_gaussian_geometry_causal import cm_metrics,CLASS_NAMES
NAMES=('initial',*ARMS)
LABELS=dict(initial='微调起点',baseline='B：原目标继续训练',geometry='G：附加几何场',semantic='S：附加局部同类场')
REPORT=ROOT/'analysis/semantic_field_mini_results.md'

def aggregate(name):
    folder=RUN/'eval'/name;meta=json.loads((folder/'complete.json').read_text())
    paths=sorted((folder/'scenes').glob('*.npz'));assert len(paths)==150
    total={};scenes=[];indices=[];qload=[];qcore=[]
    for path in paths:
        info=json.loads(path.with_suffix('.json').read_text())
        assert info['checkpoint_sha256']==meta['checkpoint_sha256'];indices+=info['indices']
        with np.load(path) as z:
            row={k:z[k].sum(0) for k in z.files if k not in ('load','query_core')}
            qload.append(z['load']);qcore.append(z['query_core'])
        for k,v in row.items():total[k]=total.get(k,np.zeros_like(v))+v
        scenes.append((path.stem,row))
    assert sorted(indices)==list(range(6019))
    load=np.concatenate(qload);core=np.concatenate(qcore);assert core.shape==(6019,1200)
    folder=RUN/'aggregate';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{name}.npz',**total,load=load,query_core=core)
    return total,scenes

def metrics(a):
    m={k:float(v) for k,v in cm_metrics(a['confusion']).items()}
    cc=a['class_coverage'].sum(0);n=cc[0];q=a['queries'];g=a['gaussians'];h=a['high']
    for j,k in enumerate(('geo1','geo2','geo3','same1','same2','same3','initial_class_geo1','native'),1):
        m[k]=100*cc[j]/max(n,1)
    for j,k in enumerate(('zero_core','core_le5','core_le10','core_le20'),1):m[k]=100*q[j]/q[0]
    m['top20_nearest_share']=100*q[5]/max(n,1);m['top20_core_union']=100*q[6]/max(n,1)
    m['query_core_pairs_per_GT']=q[7]/max(n,1)
    m['zero_nearest']=100*q[8]/q[0]
    m['high_core']=100*h[1]/max(h[0],1);m['high_same_core']=100*h[2]/max(h[0],1)
    m['high_recall']=100*h[3]/max(h[0],1);m['high_semantic_recall']=100*h[4]/max(h[0],1)
    for j,k in enumerate(('center_camera_fg','center_camera_empty','center_unknown','major_scale_saturated'),1):m[k]=100*g[j]/g[0]
    for j,k in enumerate(('major_scale','opacity','semantic_amplitude'),5):m[k]=g[j]/g[0]
    return {k:float(v) for k,v in m.items()}

def paired_interval(b,t,repeat=1500):
    assert [x[0] for x in b]==[x[0] for x in t]
    rng=np.random.default_rng(928);selection=rng.integers(0,150,(repeat,150))
    bcm=np.stack([x[1]['confusion'] for x in b]);tcm=np.stack([x[1]['confusion'] for x in t])
    bc=np.stack([x[1]['class_coverage'] for x in b]).sum(1);tc=np.stack([x[1]['class_coverage'] for x in t]).sum(1)
    diffs=[];cover=[]
    for ids in selection:
        diffs.append(float(cm_metrics(tcm[ids].sum(0))['mIoU']-cm_metrics(bcm[ids].sum(0))['mIoU']))
        x,y=bc[ids].sum(0),tc[ids].sum(0)
        cover.append(100*(y[1]/max(y[0],1)-x[1]/max(x[0],1)))
    return dict(mIoU_delta_95CI=np.quantile(diffs,[.025,.975]).tolist(),
        geo1_delta_95CI=np.quantile(cover,[.025,.975]).tolist(),unit='scene',resamples=repeat,seed=928)

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def progress():
    lines=['# 局部同类场监督：实验结果','', '**状态：实验尚未全部完成，当前不能判断方案是否有效。**','',
        '设计、公式和判据见[实验设计](semantic_field_mini_design.md)。每组在4096个训练帧上微调1个epoch，最终在完整6019帧验证集比较。','',
        '| 模型 | 训练进度 | 完整验证进度 |','| --- | --- | --- |']
    for name in NAMES:
        train=RUN/'train'/name/'progress.json';evaluation=RUN/'eval'/name/'progress.json'
        t=json.loads(train.read_text()) if train.exists() else {};e=json.loads(evaluation.read_text()) if evaluation.exists() else {}
        ts='已有权重' if name=='initial' else ('4096/4096' if t.get('status')=='complete' else f"{t.get('microbatches',0)}/4096")
        es='6019/6019' if e.get('status')=='complete' else f"{e.get('scenes',0)}/150 场景"
        lines.append(f'| {LABELS[name]} | {ts} | {es} |')
    lines+=['','已完成椭球距离的有限差分检查，以及真实训练帧上中心、尺度、旋转的梯度检查。性能结论必须等到同预算基线和两组场监督均完成。',
        '', '运行数据：[实验目录](../work_dirs/semantic_field_mini_v1/)。本报告由汇总脚本更新，进度以该目录内的日志为准。','']
    REPORT.parent.mkdir(exist_ok=True);REPORT.write_text('\n'.join(lines))

def report():
    if not all((RUN/'eval'/n/'complete.json').exists() for n in NAMES):progress();return
    combined={};scenes={};m={}
    for n in NAMES:combined[n],scenes[n]=aggregate(n);m[n]=metrics(combined[n])
    intervals={n:paired_interval(scenes['baseline'],scenes[n]) for n in ('geometry','semantic')}
    s_vs_g=paired_interval(scenes['geometry'],scenes['semantic'])
    rows=[dict(arm=n,**m[n]) for n in NAMES];write_csv(RUN/'aggregate/metrics.csv',rows)
    classrows=[]
    for n in NAMES:
        a=combined[n];cm=a['confusion'];cc=a['class_coverage']
        for c,label in enumerate(CLASS_NAMES):
            tp=cm[c,c];union=cm[c].sum()+cm[:,c].sum()-tp;gt=cc[c,0]
            classrows.append(dict(arm=n,class_id=c,class_name=label,GT=int(gt),
                IoU=100*tp/max(union,1),core1=100*cc[c,1]/max(gt,1),same_core1=100*cc[c,4]/max(gt,1),
                FN=100*cm[c,17]/max(gt,1),FP_from_empty=int(cm[17,c])))
    write_csv(RUN/'aggregate/per_class.csv',classrows)
    traces={}
    for n in ARMS:
        r=[json.loads(s) for s in (RUN/'train'/n/'trace.jsonl').read_text().splitlines()]
        assert len(r)==4096 and len({x['index'] for x in r})==4096
        d={}
        keys=sorted({k for x in r for k in x['losses'] if k.startswith('SemanticGaussianFieldLoss')})
        for k in keys:d[k]=float(np.mean([x['losses'].get(k,0) for x in r]))
        traces[n]=d
    pilot=json.loads((RUN/'pilot.json').read_text())
    summary=dict(status='complete',metrics=m,paired_vs_baseline=intervals,semantic_vs_geometry=s_vs_g,
        training_frame_mean_diagnostics=traces,pilot=pilot,
        artifacts={n:sha(RUN/'eval'/n/'complete.json') for n in NAMES})
    write(RUN/'summary.json',summary)
    b=m['baseline'];g=m['geometry'];s=m['semantic']
    fmt=lambda v:f'{v:.3f}'
    lines=['# 局部同类场监督：实验结果','',
        '本实验检验4.1节的场监督能否改善现有small模型。三组使用相同预训练权重、4096个训练帧和1个epoch；以下结果均来自完整6019帧、150个场景的camera有效区域。', '',
        f"**几何场版相对同预算基线：mIoU {g['mIoU']-b['mIoU']:+.3f}个百分点，1σ几何覆盖 {g['geo1']-b['geo1']:+.3f}个百分点；局部同类版分别为 {s['mIoU']-b['mIoU']:+.3f} 和 {s['geo1']-b['geo1']:+.3f}个百分点。**",'',
        '这里将“性能改善”“几何覆盖改善”和“负载压力改善”分别判断。微调起点用于观察绝对变化，方法收益以同预算的B组为对照。','',
        '## 1. 预测性能','',
        '| 模型 | mIoU ↑ | 二值IoU ↑ | 漏检/GT前景 ↓ | FP/GT前景 ↓ | FP/预测前景 ↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {LABELS[n]} | {v['mIoU']:.3f} | {v['IoU']:.3f} | {v['FN_rate']:.3f}% | {v['FP_over_GTfg']:.3f}% | {v['FP_over_predfg']:.3f}% |")
    lines+=['','mIoU为17个前景类别IoU的算术平均；二值IoU忽略前景类别。漏检指真实前景被预测成empty，FP指真实empty被预测成前景。上表均先合并全验证集计数，再计算指标。','',
        '| 与B比较 | mIoU差值 | 场景配对bootstrap 95%区间 | 1σ覆盖差值 | 95%区间 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for n in ('geometry','semantic'):
        v=m[n];i=intervals[n];a,z=i['mIoU_delta_95CI'];c,d=i['geo1_delta_95CI']
        lines.append(f"| {LABELS[n]} | {v['mIoU']-b['mIoU']:+.3f} | [{a:+.3f}, {z:+.3f}] | {v['geo1']-b['geo1']:+.3f} | [{c:+.3f}, {d:+.3f}] |")
    lines+=['','区间以场景为重采样单位，配对重采样1500次。它反映场景差异，不能替代多训练种子的方差评估。S与G的直接配对区间也保存在summary.json。','',
        '## 2. 核心覆盖是否真正改善','',
        '| 模型 | 1σ几何覆盖 | 2σ几何覆盖 | 3σ几何覆盖 | 同预测类别1σ覆盖 | 固定初始类别1σ覆盖 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {LABELS[n]} | {v['geo1']:.3f}% | {v['geo2']:.3f}% | {v['geo3']:.3f}% | {v['same1']:.3f}% | {v['initial_class_geo1']:.3f}% |")
    lines+=['','所有覆盖率的分母均为camera有效GT前景体素数；一个GT被多个GS覆盖只计一次。几何覆盖不检查类别；同预测类别覆盖要求GS预测类别与GT一致。最后一列固定每个GS输出槽在初始模型中的类别，只改变其几何，作为额外诊断。输出槽并非固定物体身份，因此该列不能解释为独立的几何因果效应。', '',
        '| 相对B | 新获1σ几何覆盖/GT | 丢失1σ几何覆盖/GT | 新获同类覆盖/GT | 丢失同类覆盖/GT |',
        '| --- | ---: | ---: | ---: | ---: |']
    for n in ('geometry','semantic'):
        p=combined[n]['paired_baseline'];values=100*p[:4]/max(p[4],1)
        lines.append('| '+LABELS[n]+' | '+' | '.join(f'{x:.3f}%' for x in values)+' |')
    lines+=['','这里同时列出新增和丢失，避免只报告净覆盖率而忽略原有覆盖被破坏。统计基于最终第六级GS与实际渲染候选盒内的严格椭球支持，未使用第五级中心或最近中心距离代替覆盖。','',
        '## 3. Query负载压力','',
        '| 模型 | 最近归属top20%的GT份额 | 零核心接触query | 核心接触≤20体素query | 核心覆盖top20%的GT并集 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {LABELS[n]} | {v['top20_nearest_share']:.3f}% | {v['zero_core']:.3f}% | {v['core_le20']:.3f}% | {v['top20_core_union']:.3f}% |")
    lines+=['','最近归属：每个GT体素选最近的最终GS，再回溯其父query；每帧负载最大的240个query占top20%。这不要求GS真正覆盖该GT。核心接触：对一个query的64个GS取1σ覆盖并集，得到它实际接触的GT数。不同query之间允许重叠，所以接触数不构成互斥分配。','',
        '**固定初始高负载区域。** 先用初始模型逐帧确定高负载query及归属于它们的GT集合，之后所有模型在同一GT集合上比较：','',
        '| 模型 | 1σ几何覆盖 | 同预测类别1σ覆盖 | 二值召回 | 正确语义召回 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {LABELS[n]} | {v['high_core']:.3f}% | {v['high_same_core']:.3f}% | {v['high_recall']:.3f}% | {v['high_semantic_recall']:.3f}% |")
    lines+=['','只有最近归属变均衡，不足以说明压力得到缓解。还需要实际核心缺口减少、固定高负载区域覆盖或召回改善。该损失没有改变每query固定64个GS的结构；即使有效，也不能直接宣称解决了自适应配额问题。','',
        '## 4. 是否通过退缩或改变语义幅度降损','',
        '| 模型 | 中心在有效前景 | 中心在有效empty | 中心在未知区域 | 最大轴≥0.3999m | 平均最大轴/m | 平均opacity | 平均前景幅度和 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {LABELS[n]} | {v['center_camera_fg']:.3f}% | {v['center_camera_empty']:.3f}% | {v['center_unknown']:.3f}% | {v['major_scale_saturated']:.3f}% | {v['major_scale']:.4f} | {v['opacity']:.4f} | {v['semantic_amplitude']:.4f} |")
    lines+=['','未知包括camera无效、标签无效和场景范围外的位置。场损失不乘opacity或语义幅度，直接梯度只进入几何；共享网络仍可间接改变其他属性。尺度有界参数化保持原样。','',
        '## 5. 训练与实现核对','',
        '原有六级中心CD、Set分类和camera内CE＋Lovasz全部保留。G新增覆盖＋泄漏；S新增同类覆盖＋局部同类拟合＋不确定泄漏。所有新增分项权重均为0.5，使用训练帧梯度预检统一确定，未按验证结果选权重。','',
        '| 训练帧平均诊断 | G | S |','| --- | ---: | ---: |']
    for suffix,label in [('raw','附加项未加权总值'),('known_sample_fraction','有效场内样本比例'),
        ('known_empty_sample_fraction','已知空区样本比例'),('reliable_fraction','可靠同类对应GS比例'),
        ('mean_confidence','平均对应置信度'),('fallback_fraction','GT同类候选回退比例')]:
        vals=[]
        for n in ('geometry','semantic'):
            found=[v for k,v in traces[n].items() if k.endswith('/'+suffix)]
            vals.append(f'{found[0]:.5f}' if found else '—')
        if suffix=='fallback_fraction':vals[0]='不适用'
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines+=['','上表按训练帧等权平均；回退指缺少可靠同类候选，不能计作同类覆盖。每GS固定8个3σ截断高斯样本，固定全部NK为分母；有效样本比例另列，防止将迁移到未知区误读为拟合改善。','',
        '数值检查包括：FP64有限差分验证中心/尺度/四元数梯度；GPU精确最近椭球对照穷举；局部连通片段的最近体素立方体查询；缺失同类目标回退；评估CUDA覆盖与历史位图逐位一致，并在16个GT位置穷举全部GS复核。真实训练帧上物理尺度梯度非零，但原始尺度输出存在饱和，不能只凭物理梯度认定尺度分支已充分更新。','',
        '## 6. 本次实验能回答到哪里','',
        '这是单一初始化、单一训练种子的1epoch微调验证。性能与覆盖的改善应同时对照B组，语义扩展的额外价值还需对照G组。训练损失下降本身不能证明缓解了query负载压力。','',
        '同类对应采用GT主导的单连通片段特例，范围固定2米，预测份额仅弱调置信度；没有训练新的类别对应网络。empty泄漏距离在网格中心精确、非网格查询使用三线性近似。离散匹配与camera边界固定在当前反向步骤，未对边界选择求导。', '',
        '## 7. 可追溯产物','',
        '- [实验设计](semantic_field_mini_design.md)；[论文框架4.1](../plans/cvpr2027_semantic_gaussian_evidence_support_paper_framework.md#411-先区分体素空间区域与高斯核心)。',
        '- [完整统计与置信区间](../work_dirs/semantic_field_mini_v1/summary.json)；[指标CSV](../work_dirs/semantic_field_mini_v1/aggregate/metrics.csv)；[逐类CSV](../work_dirs/semantic_field_mini_v1/aggregate/per_class.csv)。',
        '- [冻结执行协议](../work_dirs/semantic_field_mini_v1/protocol.json)；[数值检查](../work_dirs/semantic_field_mini_v1/numerical_checks.json)；[训练帧梯度预检](../work_dirs/semantic_field_mini_v1/pilot.json)。',
        '- 每组完整权重位于 `work_dirs/semantic_field_mini_v1/train/<组名>/epoch_1.pth`；逐帧训练日志、逐场景验证计数和逐query数组一并保留。','']
    REPORT.write_text('\n'.join(lines));print(json.dumps(dict(status='complete',metrics=m,intervals=intervals),ensure_ascii=False))

if __name__=='__main__':report()
