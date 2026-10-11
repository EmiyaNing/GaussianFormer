#!/usr/bin/env python3
"""Aggregate only completed paired fast-motion KD runs and write readable reports."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.sparseworld_fast_motion_common import OUT,REPORT,HORIZONS,GROUPS,region_counts
from tools.sparseworld_region_utils import metric_from_hist,percentile_interval,write_json,sha256

LABELS={'native':'原始预测器','teacher':'未来感知 teacher','frozen_task':'固定感知器：task','frozen_fast_kd':'固定感知器：KD','full_task':'全量微调：task','full_fast_kd':'全量微调：KD'}

def load_base(root,name,manifest):
    hist=[];counts=[]
    for ix in manifest['val']['indices']:
        with np.load(root/'targets'/'val'/f'{ix:06d}.npz') as d:
            hs=[];cs=[]
            gt=d['GT'];pred=d['native' if name=='native' else 'teacher'];bins=d['speed']
            for j in range(3):
                h,c=region_counts(gt[j],pred[j],bins[j]);hs.append(h);cs.append(c)
        hist.append(hs);counts.append(cs)
    return dict(hist=np.asarray(hist),counts=np.asarray(counts))

def gather(rows,ids,weights):
    values=[]
    for key in ('hist','counts'):
        x=rows[key]
        clustered=np.zeros((len(ids),)+x.shape[1:],np.int64)
        for j,positions in enumerate(ids):clustered[j]=x[positions].sum(0)
        values.append(np.tensordot(weights,clustered,axes=(1,0)))
    return dict(hist=values[0],counts=values[1])

def metrics(hist,counts):
    m=metric_from_hist(hist)
    num=counts[...,1];den=counts[...,0]
    acc=np.divide(num,den,out=np.full(num.shape,np.nan,dtype=float),where=den>0)*100
    recall=np.divide(counts[...,2],den,out=np.full(num.shape,np.nan,dtype=float),where=den>0)*100
    return dict(mIoU=m['mIoU'],OccIoU=m['OccIoU'],accuracy=acc,occupancy_recall=recall)

def summary(root,report,write_documents=True):
    protocol=json.loads((root/'protocol.json').read_text());scenes=protocol['val']['scenes'];scene_names=sorted(set(scenes))
    ids=[np.flatnonzero(np.asarray(scenes)==scene) for scene in scene_names]
    weights=np.random.default_rng(20261008).multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000)
    rows={name:load_base(root,name,protocol) for name in ('native','teacher')}
    epoch=protocol['epochs']
    for arm in protocol['arms']:
        path=root/arm/f'epoch_{epoch}.npz';complete=root/arm/'completion.json'
        if not path.exists() or not complete.exists():raise RuntimeError(f'Cannot summarize incomplete arm: {arm}')
        with np.load(path) as d:
            assert d['indices'].tolist()==protocol['val']['indices'] and d['scenes'].tolist()==scenes
            rows[arm]=dict(hist=d['hist'].copy(),counts=d['counts'].copy())
        assert np.array_equal(rows[arm]['counts'][...,0],rows['native']['counts'][...,0])
    samples={};result={}
    for name,data in rows.items():
        assert np.all(data['hist'].sum((2,3))==640000)
        raw=metrics(data['hist'].sum(0),data['counts'].sum(0));boot=gather(data,ids,weights)
        samples[name]=metrics(boot['hist'],boot['counts'])
        result[name]={key:dict(value=value,ci=percentile_interval(samples[name][key])) for key,value in raw.items()}
        result[name]['support']=dict(voxels=data['counts'][...,0].sum(0),scenes=np.array([[(sum(data['counts'][idx,j,g,0])>0) for idx in ids] for j in range(3) for g in range(len(GROUPS))]).sum(-1).reshape(3,len(GROUPS)))
    pairs={}
    for prefix in ('frozen','full'):
        a=prefix+'_fast_kd';b=prefix+'_task'
        pairs[prefix]={key:dict(value=np.asarray(result[a][key]['value'])-result[b][key]['value'],ci=percentile_interval(samples[a][key]-samples[b][key])) for key in samples[a]}
    logs={}
    for arm in protocol['arms']:
        rows_log=[json.loads(line) for line in (root/arm/'train.jsonl').read_text().splitlines()]
        expected=len(protocol['train']['indices'])*epoch
        unique={r['step']:r for r in rows_log};assert len(unique)==expected and sorted(unique)==list(range(1,expected+1))
        # Tail entries from resumed old blocks are explicitly deduplicated.
        rows_log=[unique[k] for k in sorted(unique)]
        log=dict(updates=expected,task_mean=np.mean([r['task'] for r in rows_log]),kd_mean=np.mean([r['kd'] for r in rows_log]),completion=json.loads((root/arm/'completion.json').read_text()))
        if arm.endswith('kd'):
            vox=sum(d['voxels'] for r in rows_log for d in r['fast']);support=sum(d['score_supported'] for r in rows_log for d in r['fast']);free=sum(d['teacher_free'] for r in rows_log for d in r['fast'])
            log.update(fast_voxel_tasks=vox,score_supported_fraction=support/vox if vox else None,teacher_free_fraction=free/vox if vox else None,empty_samples=sum(sum(d['voxels'] for d in r['fast'])==0 for r in rows_log))
        logs[arm]=log
    versus_native={prefix:{key:dict(value=np.asarray(result[prefix+'_fast_kd'][key]['value'])-result['native'][key]['value'],ci=percentile_interval(samples[prefix+'_fast_kd'][key]-samples['native'][key])) for key in samples['native']} for prefix in ('frozen','full')}
    output=dict(status='complete',protocol=protocol,groups=GROUPS,models=result,kd_minus_task=pairs,kd_minus_native=versus_native,training=logs)
    write_json(root/'summary.json',output)
    if write_documents:write_report(report,output);plot(report,output)
    return output

def value(d,j,g=None,digits=2):
    v=np.asarray(d['value']);ci=np.asarray(d['ci'])
    if g is None:return f'{v[j]:.{digits}f} [{ci[j,0]:.{digits}f}, {ci[j,1]:.{digits}f}]'
    return f'{v[j,g]:.{digits}f} [{ci[j,g,0]:.{digits}f}, {ci[j,g,1]:.{digits}f}]'

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def write_report(report,s):
    report.mkdir(parents=True,exist_ok=True);m=s['models'];p=s['kd_minus_task']
    fast=table(['设置','1 秒快速准确率（%）','2 秒','3 秒'],[[LABELS[a]]+[value(v['accuracy'],j,1) for j in range(3)] for a,v in m.items()])
    full=table(['设置','1 秒整体 mIoU（%）','2 秒','3 秒'],[[LABELS[a]]+[value(v['mIoU'],j) for j in range(3)] for a,v in m.items()])
    delta=table(['KD − task（百分点）','1 秒','2 秒','3 秒'],[[('固定感知器' if a=='frozen' else '全量微调')+'：'+metric]+[value(v[key],j,g,3 if g is None else 2) for j in range(3)] for a,v in p.items() for metric,key,g in [('快速准确率','accuracy',1),('整体 mIoU','mIoU',None)]])
    paragraphs=[]
    for prefix,title in [('frozen','固定感知器'),('full','全量微调')]:
        f=np.asarray(p[prefix]['accuracy']['value'])[:,1];interval=np.asarray(p[prefix]['accuracy']['ci'])[:,1]
        signs=['区间完全高于零' if x[0]>0 else ('区间完全低于零' if x[1]<0 else '区间跨零') for x in interval]
        proof=('三个时距的 95% 区间均高于零' if all(x[0]>0 for x in interval) else '对应'+ '／'.join(signs))
        paragraphs.append(f'**{title}：**1／2／3 秒快速区域的 KD−task 分别为 '+ '／'.join(f'{v:+.2f}' for v in f)+' 个百分点，'+proof+'。')
    initial=table(['KD 相对原始预测器（百分点）','1 秒快速准确率','2 秒','3 秒'],[[('固定感知器' if prefix=='frozen' else '全量微调')]+[value(v['accuracy'],j,1) for j in range(3)] for prefix,v in s['kd_minus_native'].items()])
    logs=table(['组别','更新次数','平均 task','平均 KD','峰值显存（GiB）'],[[LABELS[a],v['updates'],f"{v['task_mean']:.3f}",f"{v['kd_mean']:.3f}",f"{v['completion']['peak_gib']:.2f}"] for a,v in s['training'].items()])
    support=table(['KD 学生','累计快速体素任务','有学生分数支持','teacher 输出 free','无快速体素的样本'],[[LABELS[a],v['fast_voxel_tasks'],f"{100*v['score_supported_fraction']:.2f}%",f"{100*v['teacher_free_fraction']:.2f}%",v['empty_samples']] for a,v in s['training'].items() if a.endswith('kd')])
    positive=all(np.asarray(v['accuracy']['ci'])[:,1,0].min()>0 for v in p.values())
    lead=('**核心结果：两种学生都从快速区域 KD 获得了相对 task 基线的局部增益，但这不等于所有时距都超过了原始 checkpoint。**' if positive else '**核心结果：KD 的收益需要结合时距、学生版本和原始 checkpoint 分别判断。**')
    below_original=all((np.asarray(s['kd_minus_native'][a]['mIoU']['value'])<0).all() for a in ('frozen','full'))
    initial_global=('本轮两种 KD 学生的全域 mIoU 在三个时距均仍低于原始 checkpoint，因此未建立绝对整体改进。' if below_original else '是否取得绝对整体改进，需结合原始 checkpoint 一行比较。')
    text=f'''# 实验结果｜快速运动区域的 occupancy 蒸馏

[返回总览](README.md) · [实验设计](design.md) · [执行协议](protocol.md)

> **四组训练及完整验证已完成。**2048 个训练样本、512 个验证样本；每组 {s['protocol']['epochs']} 个 epoch、单个配对种子。下表使用全部验证场景；方括号是场景级配对 bootstrap 的 95% 区间。

{lead}

## 先回答：蒸馏有没有比 task 基线更好？

{(chr(10)*2).join(paragraphs)}

{delta}

![快速区域及整体指标的蒸馏净变化](figures/kd_effect.png)

*读图：点表示 KD−task，横线为 95% 区间；右侧表示提高，左侧表示下降。局部准确率与全域 mIoU 分开阅读。*

## 快速运动区域的绝对成绩

{fast}

Teacher 输入真实未来图像，作为信息参考，不是学生推理时可用的模型。主比较仍是同一版本的 KD 与 task；与原 checkpoint 的变化同时包含继续训练的影响。

## 是否超过未继续训练的原始预测器？

{initial}

KD−task 为正，表示蒸馏比同预算的继续训练更好；是否改进了原模型，需要另看这张 KD−原始表。区间为负的时距仍低于原始预测器，不能把恢复部分训练退化写成全时距绝对进步。

## 整体成绩是否也改善？

{full}

{initial_global}

OccIoU、慢速、静态及速度未知区域的全部指标见[统计附表](statistics.md)。快速区域只覆盖真实占据体素，不能靠这一个指标判断空闲区域误报是否减少。

## 实际训练预算与蒸馏信号

{logs}

{support}

“有学生分数支持”表示该体素至少一个类别分数高于 1e-6，是有效蒸馏梯度覆盖的近似诊断；未覆盖的位置虽然计入平均损失，但可能没有直接 KD 梯度。Teacher 的 free 标签没有被 GT 替换或剔除，故 teacher 漏检可能与占据 GT task 监督冲突。

## 可以据此判断到什么程度？

这组配对实验检验了特定损失、权重和预算下的蒸馏增益，不等同于未来感知器全部优势都能迁移。固定版本的感知参数与缓冲区在训练后对账一致，全量版本保留原始结构并开放全部可学习参数；学生验证仅使用历史五帧。

本轮未加入同强度快速 GT 重加权或打乱 teacher 对照，因此即使有收益，也不能单凭它归因于未来信息的独有内容。只有一个训练种子，且验证场景已用于前期区域分析；bootstrap 描述场景采样不确定性，不能代替训练种子稳定性或独立测试。KD 的直接梯度作用于语义分数，几何仍由原 task loss 训练，因此结果也不回答可微几何 occupancy 蒸馏的潜力。

原始统计、逐样本验证与训练日志见 [`out/sparseworld_fast_motion_distillation/20261008_2048x512/`](../../../../out/sparseworld_fast_motion_distillation/20261008_2048x512/)。
'''
    (report/'results.md').write_text(text)
    appendix='# 统计附表｜三个时距与区域指标\n\n[结果报告](results.md) · [执行协议](protocol.md)\n\n'
    for key in ('mIoU','OccIoU'):
        appendix+=f'## {key}（%）\n\n'+table(['设置','1 秒','2 秒','3 秒'],[[LABELS[a]]+[value(v[key],j) for j in range(3)] for a,v in m.items()])+'\n'
    for g,name in enumerate(GROUPS):
        appendix+=f'## '+{'occupied':'全部真实占据','fast':'快速运动（≥2 m/s）','slow':'低速（<0.5 m/s）','moderate':'中速（0.5–2 m/s）','motion_unknown':'可运动类别：速度未知','static':'静态类别'}[name]+'：语义准确率（%）\n\n'+table(['设置','1 秒','2 秒','3 秒'],[[LABELS[a]]+[value(v['accuracy'],j,g) for j in range(3)] for a,v in m.items()])+'\n'
    appendix+='## 区域定义\n\noccupied：全部 GT 占据；fast：速度 ≥2 m/s；slow：<0.5 m/s；moderate：0.5–2 m/s；motion_unknown：可运动类别中速度未知；static：前期定义的静态类别。\n'
    (report/'statistics.md').write_text(appendix)

def plot(report,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
    out=report/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for ax,key,g,title in zip(axes,['accuracy','mIoU'],[1,None],['快速运动区域：准确率','整体 mIoU']):
        for i,(prefix,color) in enumerate([('frozen','#286b9e'),('full','#bf5c4c')]):
            d=s['kd_minus_task'][prefix][key];v=np.asarray(d['value']);ci=np.asarray(d['ci'])
            if g is not None:v=v[:,g];ci=ci[:,g]
            ax.errorbar(v,np.arange(3)+(.1 if i else -.1),xerr=np.stack([v-ci[:,0],ci[:,1]-v]),fmt='o',color=color,label='固定感知器' if i==0 else '全量微调')
        ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(3),['1 秒','2 秒','3 秒']);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('KD − task（百分点）');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=2,frameon=False);ax.grid(axis='x',alpha=.15)
    for ext in ('png','svg'):fig.savefig(out/f'kd_effect.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUT);p.add_argument('--report',type=Path,default=REPORT);args=p.parse_args();summary(args.root,args.report)
