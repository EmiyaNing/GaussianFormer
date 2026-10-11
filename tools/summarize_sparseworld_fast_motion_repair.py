#!/usr/bin/env python3
"""Report repair gains separately from KD gains, with paired scene intervals."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.summarize_sparseworld_fast_motion_distillation import summary,gather,metrics,table,value
from tools.sparseworld_region_utils import write_json,percentile_interval,metric_from_hist,CLASS_NAMES
from tools.check_sparseworld_fast_motion_repair import DEFAULT

REPORT=Path('analysis/hypotheses/prediction_perception_regions/fast_motion_distillation/repair')
NAMES={'native':'原始预测器','teacher':'未来感知 teacher','frozen_task':'固定感知器：修复 task','frozen_fast_kd':'固定感知器：修复 KD','full_task':'全量微调：修复 task','full_fast_kd':'全量微调：修复 KD'}

def main(root,report):
    report.mkdir(parents=True,exist_ok=True)
    s=summary(root,report,write_documents=False);oldroot=Path(s['protocol']['target_cache']);old=json.loads((oldroot/'summary.json').read_text())
    scenes=np.asarray(s['protocol']['val']['scenes']);ids=[np.flatnonzero(scenes==n) for n in sorted(set(scenes))]
    weights=np.random.default_rng(20261008).multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000)
    assert s['protocol']['train']==old['protocol']['train'] and s['protocol']['val']==old['protocol']['val']
    differences={};conditional={};signal={};class_iou={}
    for arm in s['protocol']['arms']:
        data=[]
        for directory in (root,oldroot):
            with np.load(directory/arm/'epoch_1.npz') as d:data.append(dict(hist=d['hist'].copy(),counts=d['counts'].copy()))
        a,b=[gather(d,ids,weights) for d in data];ma,mb=metrics(a['hist'],a['counts']),metrics(b['hist'],b['counts'])
        differences[arm]={k:dict(value=np.asarray(s['models'][arm][k]['value'])-np.asarray(old['models'][arm][k]['value']),ci=percentile_interval(ma[k]-mb[k])) for k in ma}
        class_iou[arm]=metric_from_hist(data[0]['hist'].sum(0))['per_class_iou']
        n=data[0]['counts'].sum(0);c=np.divide(n[:,1,1],n[:,1,2])*100
        boot=a['counts'];cb=np.divide(boot[:,:,1,1],boot[:,:,1,2])*100
        conditional[arm]=dict(value=c,ci=percentile_interval(cb))
        if arm.endswith('kd'):
            logs={r['step']:r for r in map(json.loads,(root/arm/'train.jsonl').read_text().splitlines())}
            signal[arm]={key:sum(d[key] for r in logs.values() for d in r['fast']) for key in ('voxels','teacher_free','score_supported','retained','positive_unsupported')}
    for name in ('native','teacher'):
        with np.load(oldroot/(name+'_baseline.npz')) as d:class_iou[name]=metric_from_hist(d['hist'].sum(0))['per_class_iou']
    s.update(repair_minus_legacy=differences,conditional_semantic_accuracy=conditional,repair_signal=signal,per_class_iou=class_iou)
    write_json(root/'summary.json',s)
    write_report(root,report,s,old);plot(report,s,old)
    export_csv(report,s,old)
    print(json.dumps({prefix:dict(fast_vs_task=np.asarray(s['kd_minus_task'][prefix]['accuracy']['value'])[:,1].tolist(),fast_vs_native=np.asarray(s['kd_minus_native'][prefix]['accuracy']['value'])[:,1].tolist(),miou_vs_native=np.asarray(s['kd_minus_native'][prefix]['mIoU']['value']).tolist()) for prefix in ('frozen','full')},ensure_ascii=False))

def write_report(root,report,s,old):
    models=s['models'];delta=s['kd_minus_task'];native=s['kd_minus_native'];repair=s['repair_minus_legacy']
    quick=[]
    for prefix,title in [('frozen','固定感知器'),('full','全量微调')]:
        task=np.asarray(delta[prefix]['accuracy']['value'])[:,1];orig=np.asarray(native[prefix]['accuracy']['value'])[:,1]
        ci=np.asarray(native[prefix]['accuracy']['ci'])[:,1];mc=np.asarray(native[prefix]['mIoU']['ci'])
        states=['高于零' if x[0]>0 else ('低于零' if x[1]<0 else '跨零') for x in ci]
        status=('三个时距均超过原始模型，场景级区间均高于零' if (ci[:,0]>0).all() else '相对原始模型的 1／2／3 秒区间分别'+ '／'.join(states)+'，尚未证明全时距超过原始模型')
        overall=('整体 mIoU 三个时距的区间均高于零' if (mc[:,0]>0).all() else ('整体 mIoU 三个时距的区间均低于零' if (mc[:,1]<0).all() else '整体 mIoU 的证据随时距而异，不能概括为全时距显著提高'))
        quick.append(f'**{title}：**修复 KD 相对匹配 task 的快速准确率提高 '+ '／'.join(f'{x:+.2f}' for x in task)+' 个百分点；相对原始模型为 '+ '／'.join(f'{x:+.2f}' for x in orig)+f' 个百分点。{status}；{overall}。')
    absolute=table(['模型','1 秒快速准确率（%）','2 秒','3 秒'],[[NAMES[a]]+[value(v['accuracy'],j,1) for j in range(3)] for a,v in models.items()])
    dtable=table(['比较（百分点）','1 秒','2 秒','3 秒'],[[title+'：'+label]+[value(d[key],j,g,3 if g is None else 2) for j in range(3)] for title,prefix in [('固定感知器','frozen'),('全量微调','full')] for label,d,key,g in [('KD − 修复 task',delta[prefix],'accuracy',1),('KD − 原始模型',native[prefix],'accuracy',1),('KD − 旧 KD',repair[prefix+'_fast_kd'],'accuracy',1),('整体 mIoU：KD − 原始',native[prefix],'mIoU',None)]])
    global_table=table(['模型','1 秒 mIoU（%）','2 秒','3 秒'],[[NAMES[a]]+[value(v['mIoU'],j,None,3) for j in range(3)] for a,v in models.items()])
    decomposition=table(['修复后的学生','3 秒占据召回（%）','3 秒条件分类正确率（%）','3 秒最终准确率（%）'],[[NAMES[a],value(models[a]['occupancy_recall'],2,1),value(s['conditional_semantic_accuracy'][a],2),value(models[a]['accuracy'],2,1)] for a in s['protocol']['arms']])
    original_comp=float(np.asarray(old['models']['native']['accuracy']['value'])[2,1])/float(np.asarray(old['models']['native']['occupancy_recall']['value'])[2,1])*100
    signal=table(['KD 版本','快速体素任务数','teacher 空：跳过','学生无支持且 teacher 非空：跳过','实际用于 KD'],[[NAMES[a],d['voxels'],f"{100*d['teacher_free']/d['voxels']:.2f}%",f"{100*d['positive_unsupported']/d['voxels']:.2f}%",f"{100*d['retained']/d['voxels']:.2f}%"] for a,d in s['repair_signal'].items()])
    legacy=table(['旧 KD → 修复 KD','1 秒快速准确率（%）','2 秒','3 秒'],[[title]+[f"{np.asarray(old['models'][prefix+'_fast_kd']['accuracy']['value'])[j,1]:.2f} → {np.asarray(models[prefix+'_fast_kd']['accuracy']['value'])[j,1]:.2f}" for j in range(3)] for title,prefix in [('固定感知器','frozen'),('全量微调','full')]])
    spillover=table(['3 秒区域：KD−匹配 task（百分点）','固定感知器','全量微调'],[[label]+[value(delta[prefix]['accuracy'],2,g) for prefix in ('frozen','full')] for g,label in [(0,'全部 GT 占据'),(1,'快速运动'),(2,'低速运动'),(3,'中速运动'),(4,'速度未知'),(5,'静态类别')]])
    audit=json.loads((root/'repair_gradient_diagnosis.json').read_text())
    gs=audit['snapshots']['frozen_fast_kd']['statistics']
    gradient_table=table(['KD 末次快照：参数组','task / KD 梯度余弦均值','负余弦数 / 有效探针数','加权 KD／task 梯度范数中位数'],[[group,f"{d['mean_cosine']:.3f}",f"{d['negative_cosines']} / {d['probes_with_cosine']}",f"{d['median_weighted_ratio']:.3f}"] for group,d in gs.items()])
    all_fast_positive=all((np.asarray(native[p]['accuracy']['ci'])[:,1,0]>0).all() for p in ('frozen','full'))
    all_global_positive=all((np.asarray(native[p]['mIoU']['ci'])[:,0]>0).all() for p in ('frozen','full'))
    fixed_fast_positive=(np.asarray(native['frozen']['accuracy']['ci'])[:,1,0]>0).all()
    lead=('**快速区域的提升已恢复：两种学生在三个时距均超过原始模型，场景级配对区间均高于零。**' if all_fast_positive else ('**固定感知器的快速区域收益已恢复；全量微调部分追回退化，整体提升尚未恢复。**' if fixed_fast_positive else '**修复效果需按学生版本和时距判断；未预设所有指标都会恢复。**'))
    scope=('本轮还支持两种学生全时距的整体 mIoU 提升。' if all_global_positive else '本轮未建立两种学生在所有时距上的整体 mIoU 提升，局部收益与整体结果需要分开判断。')
    text=f'''# 修复实验结果｜蒸馏是否重新带来提升？

[实验设计](design.md) · [统计附表](statistics.md) · [代码与复现](reproduction.md) · [返回原实验](../README.md)

> **已完成：2048 个训练样本、512 个验证样本，四组各 2048 次更新。**使用与原实验相同的样本、起始权重、种子、学习率和训练预算。方括号为 150 个场景、2000 次配对 bootstrap 的 95% 区间；所有差值使用百分点。

{lead}

{scope}

{(chr(10)*2).join(quick)}

## 先看绝对成绩，再看蒸馏净收益

{absolute}

Teacher 使用真实未来图像，只作为信息上限参考；学生训练与验证输入均截止当前。快速准确率的分母是 GT 快速占据体素数，预测为空也算错误。

![原始、旧 KD 与修复后的对照](figures/recovery.png)

*同一学生版本分别画图：原始模型是共同起点，旧 KD 是前一次实验，修复 task 与修复 KD 是本轮匹配对照。*

{dtable}

**KD−修复 task** 衡量本轮蒸馏的净收益；**KD−原始** 检验是否超过起点；**KD−旧 KD** 衡量整个修复组合的变化。第三项包含 task 梯度协议与 KD 策略的共同影响，不能归因于某一项修复。

## 是否同时守住了整体性能？

{global_table}

{scope}快速区域只占 GT 占据体素约 1.8%，局部准确率不能替代全域 mIoU 或 OccIoU。完整区域及 OccIoU 数据见[统计附表](statistics.md)。

{spillover}

KD 的监督位置只在快速 mask 内，但 `semantic/position/ego_attention` 等参数被所有区域共享；局部监督会改变其他区域的输出。这张表直接检查了这种影响，不能假定未被蒸馏的区域完全不变。

![与匹配 task 和原始模型比较的区间](figures/paired_effect.png)

*每个点是配对差值，横线为场景级 95% 区间；跨过零意味着尚无明确方向证据。图中的快速准确率与整体 mIoU 使用不同横轴刻度。*

## 召回和类别判断分别发生了什么？

{decomposition}

原始模型在 3 秒快速区域的占据召回为 **{np.asarray(old['models']['native']['occupancy_recall']['value'])[2,1]:.2f}%**，条件分类正确率为 **{original_comp:.2f}%**。这里严格满足：`最终准确率 = 占据召回 × 条件分类正确率 / 100`。

恢复原版 detach 行为后，未来几何头的参数保持原始值；但共享特征、语义运动门控以及全量版本的感知更新仍可能改变最终位置与召回。因此“几何头未更新”不等于“输出几何完全不变”。本轮 KD 仍无直接几何梯度，不能据此声称解决了所有未覆盖区域。

## 梯度冲突是否也消失了？

对同一批 12 个按身份哈希选择的非空快速训练样本，在初始、固定 task 末次、固定 KD 末次三种快照上做了只读回放。总计 36 次，没有优化器更新；其中有些探针的 teacher 目标或学生支持被全部跳过，故不进入余弦统计。

{gradient_table}

这些统计表明，修复 teacher 空标签和数值路径后，共享参数上的 task／KD 方向冲突仍存在。它提供了进一步限制更新范围或保护原模型行为的依据，但只是小样本 eval 快照，不能解释全部训练更新，也不能单独证明整体下降由某个模块造成。未来几何头在本轮 task 与 KD 回放中均无直接梯度，符合恢复后的原版协议。

## 哪些监督实际进入了损失？

{signal}

后三列按原始快速体素任务数计算，并且恰好合计 100%。Teacher 空预测不再要求学生压低所有类别；有 teacher 占据但没有学生分数支持的位置也跳过。有效体素的损失仍除以原始快速 mask 大小，未因筛选而放大单个体素权重。没有用 GT 类别挑选 teacher 正确答案，错误非空 teacher 仍可能进入 KD。

## 相比旧实现，修复了多少？

{legacy}

原实验与本轮在各自内部都有匹配 task 对照；跨轮差值用于说明修复组合效果。实现变化是：显式恢复 strict 原版的几何梯度、按原版硬筛点渲染、稳定 logit 空间 BCE，以及跳过 teacher 漏检和无梯度目标。学习率与训练时长没有随验证成绩调整。

## 结论的证据边界

这是同一数据划分、一个训练种子的修复组合实验。场景级区间反映验证场景采样变化，不能替代训练种子的稳定性；验证集也已用于前期分析。要将局部迁移收益写成论文核心方法结论，仍需要独立测试、多种子、同强度快速 GT 重加权和打乱 teacher 对照，并分别消融这几项修复。

本轮尝试的是硬 occupancy 的可训练分数蒸馏。没有增加特征蒸馏、新模型结构、未来学生图像，或针对验证样本的后处理。结果应按上面的配对差值和绝对指标解读。

若继续研究，优先在保持本轮原版 task 梯度协议的基础上，对“KD 只更新语义头”和“保护非快速区域的原模型输出”做匹配消融，再独立研究有直接几何梯度的占据迁移。本轮没有执行这些后续方法，不能把它们写成已证明的修复效果。

原始产物：[本轮输出目录](../../../../../{root}/)；验收记录：[final_validation.json](../../../../../{root}/final_validation.json)。
'''
    (report/'results.md').write_text(text)
    appendix='# 统计附表｜修复后所有时距与区域\n\n[结果报告](results.md) · [实验设计](design.md)\n\n所有指标单位为 %，括号为场景级 95% 区间。\n\n'
    for key in ('mIoU','OccIoU'):
        appendix+=f'## 全域 {key}\n\n'+table(['模型','1 秒','2 秒','3 秒'],[[NAMES[a]]+[value(v[key],j,None,3) for j in range(3)] for a,v in models.items()])+'\n'
    for g,label in enumerate(['全部 GT 占据','快速运动','低速运动','中速运动','可运动类别：速度未知','静态类别']):
        for key,title in [('accuracy','语义准确率'),('occupancy_recall','占据召回')]:
            appendix+=f'## {label}：{title}\n\n'+table(['模型','1 秒','2 秒','3 秒'],[[NAMES[a]]+[value(v[key],j,g) for j in range(3)] for a,v in models.items()])+'\n'
    appendix+='## 修复 task 相对旧 task\n\n这一比较同时改变未来几何梯度协议，不能作为 KD 净收益。\n\n'+table(['版本与指标','1 秒（百分点）','2 秒','3 秒'],[[title+': '+label]+[value(repair[prefix+'_task'][key],j,g,3 if g is None else 2) for j in range(3)] for title,prefix in [('固定感知器','frozen'),('全量微调','full')] for label,key,g in [('快速准确率','accuracy',1),('整体 mIoU','mIoU',None)]])
    columns=['native','frozen_task','frozen_fast_kd','full_task','full_fast_kd']
    def display_iou(a,j,c):
        x=np.asarray(s['per_class_iou'][a])[j,c]
        return f'{x:.3f}' if np.isfinite(x) else '—'
    for j in range(3):
        appendix+=f'\n## {j+1} 秒：全域各类别 IoU（%）\n\n这些数值覆盖整个体素空间，而非快速 mask；free 不参与上面的 mIoU。\n\n'+table(['类别']+[NAMES[a] for a in columns],[[CLASS_NAMES[c]]+[display_iou(a,j,c) for a in columns] for c in range(18)])
    (report/'statistics.md').write_text(appendix)
    (report/'README.md').write_text('# 快速区域占据蒸馏｜修复与重新训练\n\n四组 2048／512 实验已完成。\n\n| 阅读目的 | 入口 |\n|---|---|\n| 修复是否提升，证据到哪里 | [结果报告](results.md) |\n| 修改内容、损失与对照 | [实验设计](design.md) |\n| 完整区域指标 | [统计附表](statistics.md) · [CSV](metrics.csv) |\n| 运行哪些代码、如何复现 | [代码与复现](reproduction.md) |\n\n'+lead+' '+scope+'\n')

def plot(report,s,old):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(font.get_file());plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
    directory=report/'figures';directory.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,prefix,title in zip(axes,('frozen','full'),('固定感知器','全量微调')):
        series=[('原始模型',old['models']['native'],'#9aa4ad'),('旧 KD',old['models'][prefix+'_fast_kd'],'#d0a16c'),('修复 task',s['models'][prefix+'_task'],'#75a4bb'),('修复 KD',s['models'][prefix+'_fast_kd'],'#246b54')]
        for k,(label,d,color) in enumerate(series):ax.bar(np.arange(3)+(k-1.5)*.19,np.asarray(d['accuracy']['value'])[:,1],width=.18,color=color,label=label)
        ax.set_xticks(range(3),['1 秒','2 秒','3 秒']);ax.set_ylim(0,50);ax.set_title(title);ax.set_ylabel('快速区域语义准确率（%）');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    for ext in ('png','svg'):fig.savefig(directory/f'recovery.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,key,g,title in zip(axes,('accuracy','mIoU'),(1,None),('快速准确率','整体 mIoU')):
        for k,(prefix,comparison,label,color,marker) in enumerate([('frozen','kd_minus_task','固定：KD−task','#286b9e','o'),('full','kd_minus_task','全量：KD−task','#bf5c4c','o'),('frozen','kd_minus_native','固定：KD−原始','#286b9e','s'),('full','kd_minus_native','全量：KD−原始','#bf5c4c','s')]):
            d=s[comparison][prefix][key];v=np.asarray(d['value']);ci=np.asarray(d['ci'])
            if g is not None:v=v[:,g];ci=ci[:,g]
            ax.errorbar(v,np.arange(3)+(k-1.5)*.14,xerr=np.stack([v-ci[:,0],ci[:,1]-v]),fmt=marker,color=color,label=label,markersize=5)
        ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(3),['1 秒','2 秒','3 秒']);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('配对差值（百分点）');ax.grid(axis='x',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    for ext in ('png','svg'):fig.savefig(directory/f'paired_effect.{ext}',dpi=180)
    plt.close(fig)

def export_csv(report,s,old):
    import csv
    with (report/'metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['experiment','model','horizon_seconds','metric','region','value_percent','ci_low','ci_high'])
        for experiment,models in [('legacy',old['models']),('repair',s['models'])]:
            for name,m in models.items():
                for key in ('mIoU','OccIoU','accuracy','occupancy_recall'):
                    v=np.asarray(m[key]['value']);ci=np.asarray(m[key]['ci'])
                    for j in range(3):
                        if v.ndim==1:writer.writerow([experiment,name,j+1,key,'all',v[j],*ci[j]])
                        else:
                            for g,region in enumerate(s['groups']):writer.writerow([experiment,name,j+1,key,region,v[j,g],*ci[j,g]])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--report',type=Path,default=REPORT);args=p.parse_args();main(args.root,args.report)
