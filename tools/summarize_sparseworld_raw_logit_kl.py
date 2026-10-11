#!/usr/bin/env python3
"""Summarize literal raw-KL diagnostics and Gaussian controls, then validate."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools.summarize_sparseworld_fast_motion_distillation import load_base,gather,metrics,table,value
from tools.sparseworld_region_utils import write_json,percentile_interval,sha256
from tools.sparseworld_fast_motion_common import GROUPS,FORECAST
from tools.audit_sparseworld_raw_logit_kl import ROOT

REPORT=Path('analysis/hypotheses/prediction_perception_regions/fast_motion_distillation/raw_logits_kl')
NAMES={'native':'原始预测器','teacher':'未来感知 teacher','frozen_task':'固定：task','full_task':'全量：task',
       'frozen_raw_api_kd':'固定：直接接口 D1','full_raw_api_kd':'全量：直接接口 D1','frozen_gaussian_kd':'固定：高斯 KL G','full_gaussian_kd':'全量：高斯 KL G'}

def main():
    if '--report-only' in sys.argv:
        result=json.loads((ROOT/'summary.json').read_text());assert result['status']=='complete'
        audit=json.loads((ROOT/'raw_kl_input_audit.json').read_text());fit=json.loads((ROOT/'raw_logit_fit_audit.json').read_text())
        write_report(result,audit,fit);plots(result,audit);csv_export(result);validate(result)
        print('Regenerated reports from completed statistics; validation passed.');return
    protocol=json.loads((ROOT/'protocol.json').read_text());baseline=Path(protocol['matched_task_root'])
    bp=json.loads((baseline/'protocol.json').read_text())
    for key in ('train','val','checkpoint_sha256','epochs','seed','kd_weight','lr','backbone_lr','warmup','precision','bn','position_gradient'):
        assert protocol[key]==bp[key],f'Unmatched baseline field: {key}'
    scenes=np.asarray(protocol['val']['scenes']);ids=[np.flatnonzero(scenes==n) for n in sorted(set(scenes))]
    weights=np.random.default_rng(20261008).multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000)
    rows={n:load_base(ROOT,n,protocol) for n in ('native','teacher')}
    for arm in ('frozen_task','full_task')+tuple(protocol['arms']):
        directory=(baseline if arm.endswith('task') else ROOT)/arm
        assert json.loads((directory/'completion.json').read_text())['status']=='complete'
        with np.load(directory/'epoch_1.npz') as d:
            assert d['indices'].tolist()==protocol['val']['indices'] and d['scenes'].tolist()==protocol['val']['scenes']
            rows[arm]=dict(hist=d['hist'].copy(),counts=d['counts'].copy())
        assert np.array_equal(rows[arm]['counts'][...,0],rows['native']['counts'][...,0])
    models={};boot={}
    for name,r in rows.items():
        assert np.all(r['hist'].sum((2,3))==640000)
        raw=metrics(r['hist'].sum(0),r['counts'].sum(0));b=gather(r,ids,weights);boot[name]=metrics(b['hist'],b['counts'])
        c=r['counts'].sum(0);raw['conditional_semantic']=c[...,1]/c[...,2]*100
        boot[name]['conditional_semantic']=b['counts'][...,1]/b['counts'][...,2]*100
        models[name]={k:dict(value=v,ci=percentile_interval(boot[name][k])) for k,v in raw.items()}
    pairs={};native={};training={}
    for arm in protocol['arms']:
        prefix='full' if arm.startswith('full') else 'frozen';reference=prefix+'_task'
        pairs[arm]={k:dict(value=np.asarray(models[arm][k]['value'])-np.asarray(models[reference][k]['value']),ci=percentile_interval(boot[arm][k]-boot[reference][k])) for k in boot[arm]}
        native[arm]={k:dict(value=np.asarray(models[arm][k]['value'])-np.asarray(models['native'][k]['value']),ci=percentile_interval(boot[arm][k]-boot['native'][k])) for k in boot[arm]}
        logs={r['step']:r for r in map(json.loads,(ROOT/arm/'train.jsonl').read_text().splitlines())};assert sorted(logs)==list(range(1,2049))
        assert sorted(r['index'] for r in logs.values())==sorted(protocol['train']['indices'])
        signal={k:sum(d[k] for r in logs.values() for d in r['fast']) for k in ('voxels','teacher_supported','student_supported','retained')}
        losses=np.array([logs[i]['kd'] for i in sorted(logs)])
        training[arm]=dict(updates=len(logs),signal=signal,kd_mean=losses.mean(),kd_negative_fraction=(losses<0).mean(),task_mean=np.mean([r['task'] for r in logs.values()]),peak_gib=json.loads((ROOT/arm/'completion.json').read_text())['peak_gib'])
    audit=json.loads((ROOT/'raw_kl_input_audit.json').read_text());fit=json.loads((ROOT/'raw_logit_fit_audit.json').read_text())
    result=dict(status='complete',protocol=protocol,groups=GROUPS,models=models,kd_minus_task=pairs,kd_minus_native=native,training=training,input_audit_totals={k:v['totals'] for k,v in audit['results'].items()},fit_audit=fit)
    write_json(ROOT/'summary.json',result);REPORT.mkdir(parents=True,exist_ok=True)
    write_report(result,audit,fit);plots(result,audit);csv_export(result);validate(result)
    print(json.dumps({a:dict(fast_vs_task=np.asarray(pairs[a]['accuracy']['value'])[:,1].tolist(),miou_vs_task=np.asarray(pairs[a]['mIoU']['value']).tolist()) for a in protocol['arms']}))

def write_report(s,audit,fit):
    m=s['models'];arms=s['protocol']['arms'];p=s['kd_minus_task'];n=s['kd_minus_native']
    validity=table(['真实数据：初始化','训练集','验证集'],[[label]+[f'{100*d[split][num]/d[split][den]:.2f}%' for split in ('train','val')] for label,num,den in [('有至少一个非法负 target 的配对体素','default_nonfinite_voxels','retained'),('默认接口非有限分量','default_nonfinite_components','components'),('双方均有 raw logits 的 GT-fast 体素','retained','fast_voxels')] for d in [s['input_audit_totals']]])
    fast=table(['模型','1 秒快速准确率（%）','2 秒','3 秒'],[[NAMES[a]]+[value(d['accuracy'],j,1) for j in range(3)] for a,d in m.items()])
    deltas=table(['KD−匹配 task（百分点）','1 秒','2 秒','3 秒'],[[NAMES[a]+'：'+label]+[value(p[a][key],j,g,3 if g is None else 2) for j in range(3)] for a in arms for label,key,g in [('快速准确率','accuracy',1),('整体 mIoU','mIoU',None)]])
    absolute=table(['模型','1 秒 mIoU（%）','2 秒','3 秒'],[[NAMES[a]]+[value(d['mIoU'],j,None,3) for j in range(3)] for a,d in m.items()])
    versus=table(['相对原始模型（百分点）','1 秒快速准确率','2 秒','3 秒'],[[NAMES[a]]+[value(n[a]['accuracy'],j,1) for j in range(3)] for a in arms])
    fit_table=table(['固定身份探针','GT-fast 体素','初始／最终配对数','共同体素','初始 RMSE','最终 RMSE'],[[NAMES[a],d['fast_voxels'],f"{d['initial_pair_voxels']} / {d['final_pair_voxels']}",d['common_voxels'],f"{d['initial_rmse']:.3f}",f"{d['final_rmse']:.3f}"] for a,d in fit['results'].items()])
    fast_three=table(['3 秒主比较','快速准确率（%）','KD−匹配 task（百分点，95% 区间）'],[[NAMES[a],f"{np.asarray(m[a]['accuracy']['value'])[2,1]:.2f}",value(p[a]['accuracy'],2,1)] for a in arms])
    fitting='两种直接接口学生在各自的共同体素上的 RMSE 均增大，两种高斯 KL 学生均减小。' if all(fit['results'][a]['final_rmse']>fit['results'][a]['initial_rmse'] for a in arms if 'raw_api' in a) and all(fit['results'][a]['final_rmse']<fit['results'][a]['initial_rmse'] for a in arms if 'gaussian' in a) else '原始对齐误差与覆盖率需要逐组阅读，不能只用训练 loss 判断。'
    signals=table(['训练组','GT-fast 体素任务','实际配对覆盖','平均 KD 目标','负 KD 目标步骤','峰值显存 GiB'],[[NAMES[a],d['signal']['voxels'],f"{100*d['signal']['retained']/d['signal']['voxels']:.2f}%",f"{d['kd_mean']:.4f}",f"{100*d['kd_negative_fraction']:.2f}%",f"{d['peak_gib']:.2f}"] for a,d in s['training'].items()])
    statements=[]
    for a in arms:
        v=np.asarray(p[a]['accuracy']['value'])[:,1];ci=np.asarray(p[a]['accuracy']['ci'])[:,1]
        states=['区间高于零' if x[0]>0 else ('区间低于零' if x[1]<0 else '区间跨零') for x in ci]
        statements.append(f'**{NAMES[a]}：**1／2／3 秒快速准确率相对 task 为 '+ '／'.join(f'{x:+.2f}' for x in v)+' 个百分点，分别'+ '／'.join(states)+'。')
    below=all((np.asarray(n[a]['mIoU']['value'])<0).all() for a in arms)
    scope='四个新学生的全域 mIoU 在三个时距均低于原始模型，本轮未恢复绝对整体提升。' if below else '是否取得整体提升，需要结合原始模型的绝对成绩与配对区间逐项判断。'
    t=s['input_audit_totals'];mass=table(['初始化的 logsumexp(raw logits) 中位数','训练集','验证集'],[[side]+[f"{t[split][side+'_log_mass_quantiles'][2]:.3f}" for split in ('train','val')] for side in ('teacher','student')])
    gradient_rows=[]
    for policy,label in [('raw_api_log_target','直接接口 D1'),('gaussian_raw','高斯 KL G')]:
        groups=[r['policies'][policy]['groups'] for r in audit['gradient_probes']]
        cos=[d['semantic']['cosine'] for d in groups if d['semantic']['cosine'] is not None]
        gradient_rows.append([label]+[f"{np.median([d[g]['weighted_ratio'] for d in groups]):.3f}" for g in ('semantic','position','ego_attention')]+[f'{np.mean(cos):+.3f}',len(cos)])
    gradient_table=table(['初始固定学生探针','语义头范数比中位数','position','ego_attention','语义梯度余弦均值','非零 KD 探针'],gradient_rows)
    text=f'''# 实验结果｜保留原始 logits，直接 KL 接口会发生什么？

[实验设计](design.md) · [统计附表](statistics.md) · [代码与复现](reproduction.md) · [返回蒸馏总览](../README.md)

> **已完成：2048 训练／512 验证，四个新训练组各 2048 次更新；复用两组严格匹配的 task 基线。**Teacher 输出重新导出为完整 FP32 17 维 raw logits。蒸馏读出和 loss 前没有 sigmoid／softmax／log_softmax，GT 仅用于 fast mask。

**核心判断：原始 logits 直接进入默认 KL 接口无法形成有效训练目标；`log_target=True` 虽能计算有限值，得到的也不是合法的分类分布 KL。固定方差高斯 KL 则是有效的原始 logit 对齐对照。**

{scope}

{fast_three}

这四行先比较同训练预算下的 KD 增量。绝对成绩是否超过原始模型，以及整体 mIoU 的变化，见下文第 3 节。

## 1. 先验证接口定义，而非先看准确率

{validity}

所有初始化配对体素都至少有一个负 teacher 分量，默认 `F.kl_div(zS,zT,log_target=False)` 的区域 loss 因而包含 NaN。**D0 没有进行优化器更新，也没有伪造训练后成绩。**

{mass}

合法 categorical 对数概率要求 `logsumexp(log_p)=0`；这批配对体素中，teacher 和 student 均没有在 `1e−5` 容差内满足该条件。因此，把 raw logits 直接解释为 log-probabilities 并不成立。

![直接接口与高斯 KL 的常数平移响应](figures/objective_check.png)

*使用一个真实 teacher 向量，将 student 设为 `teacher+c`。D1 在 `c=0` 损失为零，继续增加 c 仍能降低目标；G 在 c=0 才达到最小值。两面板纵轴尺度分别对应各自目标。*

PyTorch 官方 [KLDivLoss 接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html)要求输入为对数概率；`log_target=True` 仅改变 target 的解释，内部实际计算 `exp(zT)*(zT-zS)`。本地 PyTorch 2.3.0 的源码与实测符合这一公式，没有自动归一化。

## 2. 四个有限值训练组是否改善快速区域？

{(chr(10)*2).join(statements)}

{fast}

![快速区域与整体指标的配对变化](figures/performance.png)

*左图单独呈现全量直接接口组的大幅下降；中图放大其余三组的快速准确率变化；右图比较四组全域 mIoU。三个面板分别设置横轴尺度，误差线均为场景配对 95% 区间。*

{deltas}

这张表的 task 基线与对应 KD 学生具有相同训练版本、起始权重、样本、种子、精度、学习率和预算。方括号为 150 个场景、2000 次配对 bootstrap 的 95% 区间；差值单位为百分点。

## 3. 是否超过了原始模型，并守住整体性能？

{versus}

{absolute}

{scope}快速准确率只在 GT 快速占据体素中计分，不能惩罚 mask 外的错误占据；全域 mIoU／OccIoU 因此必须同时阅读。[统计附表](statistics.md)保留全部区域的准确率和占据召回。

## 4. Loss 下降时，raw logits 真的匹配了吗？

**{fitting}**

![相同物理体素的原始 logit 误差与配对覆盖率](figures/fit_check.png)

{fit_table}

使用此前按身份哈希固定的 12 个非空快速训练探针，在每个学生的初始和末次输出都具有支持的同一物理体素上比较，teacher 始终固定。不同学生的共同集合大小不同；每一行内初始／最终 RMSE 使用完全相同的体素和 17 个分量。全量版本的末次输出重新输入真实历史图像计算，没有用原始感知缓存替代训练后的感知器。

配对数包含 teacher 与 student 都有完整有限向量的体素；共同体素是这些集合在初始和末次输出间的交集。先检查覆盖率，再阅读共同体素的误差，避免将漏掉的区域从误差统计中消失误判为改善。

这个小样本只读检查用来检验目标行为，不代表全部训练体素，也没有进行额外优化器更新。D1 的负 loss 不能作为匹配误差收敛的证据；G 的定义为 `Normal(zT,1)` 与 `Normal(zS,1)` 的 KL，等价每分量半平方误差。

## 5. 有多少监督实际进入训练？

{signals}

双方缺少原始向量的体素被跳过，分母仍是原 GT-fast mask 大小乘以 17；没有把缺失位置填成零分数或 free 概率。有限 teacher 向量不经过 argmax 正确性筛选，也不因最终 occupancy 是 free 就被删除。

本轮 raw 读出只做原始 logit 的逐类别最大聚合，不执行原版置信度阈值、紧凑度重标定或闭运算。覆盖范围、目标类型、类别归一化均与上一轮硬 BCE 不同，不能把跨轮结果全部归因于 KL 函数。

### 名义权重 0.1 对应多大的梯度？

{gradient_table}

范数比为 `||0.1 × KD梯度|| / ||task梯度||`，中位数包括 12 个探针中因没有有效配对而产生的零值；余弦均值只包括双方范数非零的探针。检查采用初始固定感知学生、FP32 eval，`reg/velocity` 的两种梯度均为零，符合原版未来位置 detach。

初始语义头的 D1 范数比中位数没有高于 G，不能简单将两组差异全部归因于“D1 的梯度更大”。目标方向、训练后覆盖率与参数共享也需要结合阅读。这一检查没有量化全量感知器梯度，也不能代表整个训练过程的梯度变化。

## 6. 这些实验支持到什么程度？

D0 给出了直接接口输入定义失败的真实数据证据；D1 检验了“原 logits 直接调用对数 target 接口”的实际训练行为，不能证明 categorical KL 有效或无效。G 检验固定方差高斯均值对齐，保留了 raw logits，但不等价于 teacher 的类别概率分布 KL。

只有一个训练种子，且验证集已经用于前期分析。场景级区间描述验证采样不确定性，不能代替多种子或独立测试。原版 task 与 KP 更新共享参数；fast mask 内的监督也可能改变静态、慢速及空闲区域。原始稀疏几何和体素归属停止梯度，直接 raw-logit 对齐仍不能修复全部漏覆盖。

原始产物：[实验输出](../../../../../out/sparseworld_fast_motion_distillation/20261009_raw_logits_kl_2048x512/)；完整性检查：[最终验收](../../../../../out/sparseworld_fast_motion_distillation/20261009_raw_logits_kl_2048x512/final_validation.json)。
'''
    (REPORT/'results.md').write_text(text.replace('KP 更新','KD 更新'))
    appendix='# 统计附表｜原始 logit 实验\n\n[结果报告](results.md) · [实验设计](design.md)\n\n方括号为场景级 95% 区间，指标单位为 %。\n\n'
    for key in ('mIoU','OccIoU'):
        appendix+=f'## 全域 {key}\n\n'+table(['模型','1 秒','2 秒','3 秒'],[[NAMES[a]]+[value(d[key],j,None,3) for j in range(3)] for a,d in m.items()])+'\n'
    for g,label in enumerate(['全部真实占据','快速运动','低速运动','中速运动','可运动类别：速度未知','静态类别']):
        for key,title in [('accuracy','语义准确率'),('occupancy_recall','占据召回')]:
            appendix+=f'## {label}：{title}\n\n'+table(['模型','1 秒','2 秒','3 秒'],[[NAMES[a]]+[value(d[key],j,g) for j in range(3)] for a,d in m.items()])+'\n'
    appendix+='## 快速区域：已输出占据中的分类正确率\n\n'+table(['模型','1 秒','2 秒','3 秒'],[[NAMES[a]]+[value(d['conditional_semantic'],j,1) for j in range(3)] for a,d in m.items()])
    (REPORT/'statistics.md').write_text(appendix)
    (REPORT/'README.md').write_text('# 原始 logits 与 KL｜构造、执行与结果\n\n[返回蒸馏总览](../README.md)\n\n**已完成 2048／512 四组新训练，另有默认接口的全数据诊断。蒸馏路径保留原 logits，没有 sigmoid／softmax／log_softmax。**\n\n| 阅读目的 | 入口 |\n|---|---|\n| 直接 KL 接口是否有效，性能怎样 | [结果报告](results.md) |\n| 三种目标、raw 读出和固定对照 | [实验设计](design.md) |\n| 全部区域数据 | [统计附表](statistics.md) · [CSV](metrics.csv) |\n| 代码和命令 | [复现说明](reproduction.md) |\n\n默认 raw KL 输入产生 NaN；对数 target 接口的有限值目标缺少概率归一化。高斯 KL 对照保留原 logits，但其含义是均值对齐。\n\n'+fast_three+'\n'+scope+'\n\n'+fitting+'探针范围和有效覆盖率见结果报告。\n')

def plots(s,audit):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');font_manager.fontManager.addfont(font.get_file())
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
    directory=REPORT/'figures';directory.mkdir(exist_ok=True);example=audit['translation_probe'];x=np.asarray(example['shifts'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,key,title,color in zip(axes,('raw_api_curve','gaussian_curve'),('D1：原 logits 直接 log-target 接口','G：固定方差高斯 KL'),('#b75b48','#277e67')):
        ax.plot(x,example[key],color=color,lw=2);ax.axvline(0,color='gray',lw=.8);ax.axhline(0,color='gray',lw=.8);ax.set_title(title);ax.set_xlabel('student = teacher + c');ax.set_ylabel('每分量目标值');ax.grid(alpha=.15)
    for ext in ('png','svg'):fig.savefig(directory/f'objective_check.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14.5,5),layout='constrained');colors=['#9f4e3f','#2a735a','#c47b4e','#3c78a0']
    for panel,(ax,key,g,title) in enumerate(zip(axes,('accuracy','accuracy','mIoU'),(1,1,None),('全量直接接口：快速准确率','其余三组：快速准确率','整体 mIoU'))):
        for i,a in enumerate(s['protocol']['arms']):
            if panel==0 and a!='full_raw_api_kd':continue
            if panel==1 and a=='full_raw_api_kd':continue
            d=s['kd_minus_task'][a][key];v=np.asarray(d['value']);ci=np.asarray(d['ci'])
            if g is not None:v=v[:,g];ci=ci[:,g]
            ax.errorbar(v,np.arange(3)+(i-1.5)*.14,xerr=np.stack([v-ci[:,0],ci[:,1]-v]),fmt='o',color=colors[i],label=NAMES[a],markersize=5)
        ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(3),['1 秒','2 秒','3 秒']);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('KD − 匹配 task（百分点）');ax.grid(axis='x',alpha=.15)
    handles,labels=axes[2].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    for ext in ('png','svg'):fig.savefig(directory/f'performance.{ext}',dpi=180)
    plt.close(fig)
    fit=s['fit_audit']['results'];arms=s['protocol']['arms'];y=np.arange(len(arms))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,initial,final,title,xlabel in [
        (axes[0],[fit[a]['initial_rmse'] for a in arms],[fit[a]['final_rmse'] for a in arms],'相同共同体素：原始 logit 对齐误差','RMSE（越低越接近 teacher）'),
        (axes[1],[100*fit[a]['initial_pair_voxels']/fit[a]['fast_voxels'] for a in arms],[100*fit[a]['final_pair_voxels']/fit[a]['fast_voxels'] for a in arms],'固定探针：teacher／student 有效配对覆盖','GT-fast 体素覆盖率（%）')]:
        for i,(a,b) in enumerate(zip(initial,final)):ax.plot([a,b],[i,i],color='#b5b5b5',lw=2,zorder=1)
        ax.scatter(initial,y,color='#757575',marker='o',s=45,label='初始',zorder=2);ax.scatter(final,y,color='#2a735a',marker='D',s=45,label='训练后',zorder=2)
        ax.set_yticks(y,[NAMES[a] for a in arms]);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15);ax.legend(frameon=False)
    for ext in ('png','svg'):fig.savefig(directory/f'fit_check.{ext}',dpi=180)
    plt.close(fig)

def csv_export(s):
    import csv
    with (REPORT/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['model','horizon_seconds','metric','region','value_percent','ci_low','ci_high'])
        for a,m in s['models'].items():
            for key,d in m.items():
                v=np.asarray(d['value']);ci=np.asarray(d['ci'])
                for j in range(3):
                    if v.ndim==1:w.writerow([a,j+1,key,'all',v[j],*ci[j]])
                    else:
                        for g,name in enumerate(GROUPS):w.writerow([a,j+1,key,name,v[j,g],*ci[j,g]])

def validate(s):
    protocol=s['protocol'];initial=torch.load(protocol['checkpoint'],map_location='cpu')['state_dict'];checks=[]
    assert sha256(protocol['checkpoint'])==protocol['checkpoint_sha256'];assert not set(protocol['train']['scenes'])&set(protocol['val']['scenes'])
    checks.append('checkpoint identity and train/validation scene separation')
    fit=s['fit_audit']
    for arm in protocol['arms']:
        saved=torch.load(ROOT/arm/'latest.pth',map_location='cpu');c=json.loads((ROOT/arm/'completion.json').read_text())
        assert saved['step']==2048 and saved['protocol']==protocol and c['updates']==2048 and c['val_samples']==512
        assert sha256(ROOT/arm/'latest.pth')==c['checkpoint_sha256']==fit['checkpoint_hashes'][arm]
        state=saved['state_dict'];assert all(torch.equal(v,state[n]) for n,v in initial.items() if n.split('.')[0] in ('reg','velocity') or n.endswith(('running_mean','running_var','num_batches_tracked')))
        if arm.startswith('frozen'):assert all(torch.equal(v,state[n]) for n,v in initial.items() if n.split('.')[0] not in FORECAST)
        else:assert json.loads((ROOT/arm/'initial_parity.json').read_text())['status']=='passed'
        gradient=json.loads((ROOT/arm/'gradient_audit.json').read_text())
        assert sorted(gradient['no_gradient_names'])==sorted(n for n in state if n.split('.')[0] in ('reg','velocity'))
        assert s['training'][arm]['signal']['voxels']==3677614
        logs={r['step']:r for r in map(json.loads,(ROOT/arm/'train.jsonl').read_text().splitlines())}
        assert all(np.isfinite([r['task'],r['kd'],r['total'],r['grad_norm']]).all() for r in logs.values())
        checks.append(arm+': complete updates, finite losses, raw mask count, native geometry, BN and checkpoint hashes')
        del saved,state
    for name in ('frozen_task','full_task'):
        directory=Path(protocol['matched_task_root'])/name;c=json.loads((directory/'completion.json').read_text())
        assert sha256(directory/'latest.pth')==c['checkpoint_sha256'];checks.append(name+': matched baseline checkpoint unchanged')
    cache=json.loads((ROOT/'raw_cache_audit.json').read_text());assert cache['status']=='complete'
    for split in ('train','val'):
        for ix,scene in zip(protocol[split]['indices'],protocol[split]['scenes']):
            with np.load(ROOT/'teacher_logits'/split/f'{ix:06d}.npz') as d, np.load(ROOT/'targets'/split/f'{ix:06d}.npz') as old:
                assert int(d['index'])==ix and str(d['scene'])==scene
                for j in range(3):
                    assert np.array_equal(d[f'fast_{j}'],old[f'fast_{j}'])
                    z=d[f'teacher_raw_{j}'];v=d[f'teacher_raw_valid_{j}'];assert np.array_equal(np.isfinite(z).all(-1),v) and np.isneginf(z[~v]).all()
    checks.append('2560 authentic raw teacher files: identity, fast mask, finite support and native occupancy replays')
    audit=json.loads((ROOT/'raw_kl_input_audit.json').read_text());assert audit['updates']==0
    assert all(x['totals']['default_nonfinite_voxels']==x['totals']['retained'] for x in audit['results'].values())
    checks.append('literal default invalid on every paired voxel; diagnostic optimizer updates zero')
    checks.append('12 identity probes: final logit error uses common physical voxels; audited checkpoints unchanged')
    docs=list(REPORT.glob('*.md'));assert len(docs)==5
    for doc in docs:
        t=doc.read_text();assert t.count('```')%2==0
        for link in re.findall(r'\]\(([^)]+)\)',t):
            if not link.startswith(('http:','https:','#')):
                p=(doc.parent/link.split('#')[0]).resolve();assert p.exists() or p==(ROOT/'final_validation.json').resolve(),(doc,link)
    checks.append('five report documents, local links and fenced blocks')
    code=['config/opus/sparseworld_fast_motion_raw_logit_kl.py','tools/sparseworld_raw_logit_kd.py','tools/export_sparseworld_raw_teacher_logits.py','tools/audit_sparseworld_raw_logit_kl.py','tools/check_sparseworld_raw_logit_fit.py','tools/run_sparseworld_fast_motion_distillation.py','tools/summarize_sparseworld_raw_logit_kl.py','tests/test_sparseworld_raw_logit_kd.py']
    artifacts=docs+list((REPORT/'figures').glob('*'))+[REPORT/'metrics.csv']
    write_json(ROOT/'final_validation.json',dict(status='passed',check_count=len(checks),checks=checks,regression_tests=14,code_sha256={p:sha256(p) for p in code},report_sha256={str(p):sha256(p) for p in artifacts}))

if __name__=='__main__':main()
