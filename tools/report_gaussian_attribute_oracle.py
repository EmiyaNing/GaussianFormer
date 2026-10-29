"""Report all Tiny attribute subsets; all fitted scores use each frame's GT."""
from pathlib import Path
from itertools import combinations
import csv
import json
import math
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.eval_gaussian_attribute_oracle import (
    RUN,HISTORY,CENTER_HISTORY,ARMS,ATTRIBUTES,STEPS,ATTR_STATS,OVERLAP_NAMES,CLASS_NAMES,CLASS_COLUMNS)
from tools.report_gaussian_sizes import performance
from tools.stat_boundary_gaussians import read,write,sha

AGG=RUN/'aggregate'
REPORT=ROOT/'analysis/gaussian_attribute_oracle_full_val_results.md'
FIGURES=ROOT/'analysis/figures/gaussian_attribute_oracle_full_val'


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def name(subset):
    return ''.join(x for x in ATTRIBUTES if x in subset) or 'baseline'


def describe(cm,cc):
    result=performance(cm)
    for j,key in enumerate(CLASS_COLUMNS[1:],1):
        result[key]=float(100*cc[:,j].sum()/cc[:,0].sum())
    return result


def changes(stats,maximum):
    d=dict(zip(ATTR_STATS,stats));m=dict(zip(ATTR_STATS,maximum));n=d['GS']
    return dict(center_mean_m=float(d['center_sum_m']/n),center_max_m=float(m['center_max_m']),
        center_moving_percent=float(100*d['center_moving']/n),center_at_limit_percent=float(100*d['center_at_limit']/n),
        scale_axis_mean_abs_change_m=float(d['scale_abs_change_sum_m']/(3*n)),
        scale_max_abs_change_m=float(m['scale_abs_change_max_m']),scale_moving_percent=float(100*d['scale_moving_GS']/n),
        scale_axes_at_min_percent=float(100*d['scale_at_min_axes']/(3*n)),
        scale_axes_at_max_percent=float(100*d['scale_at_max_axes']/(3*n)),
        rotation_mean_deg=float(d['rotation_sum_rad']/n*180/math.pi),
        rotation_max_deg=float(m['rotation_max_rad']*180/math.pi),
        rotation_moving_percent=float(100*d['rotation_moving']/n),
        rotation_at_limit_percent=float(100*d['rotation_at_limit']/n),
        opacity_mean_abs_change=float(d['opacity_abs_change_sum']/n),
        opacity_max_abs_change=float(m['opacity_abs_change_max']),opacity_moving_percent=float(100*d['opacity_moving']/n),
        opacity_at_zero_percent=float(100*d['opacity_at_zero']/n),opacity_at_one_percent=float(100*d['opacity_at_one']/n),
        covariance_mean_relative_change=float(d['covariance_relative_change_sum']/n),
        initial_near_spherical_percent=float(100*d['initial_near_spherical']/n),
        initial_mean_aspect_ratio=float(d['initial_mean_aspect_ratio_sum']/n),
        near_spherical_rotation_mean_deg=float(d['near_spherical_rotation_sum_rad']/max(d['initial_near_spherical'],1)*180/math.pi))


def shapley(values):
    result={}
    for a in ATTRIBUTES:
        others=[x for x in ATTRIBUTES if x!=a]
        total=0
        for k in range(4):
            w=math.factorial(k)*math.factorial(3-k)/24
            for subset in combinations(others,k):
                total=total+w*(values[name((*subset,a))]-values[name(subset)])
        result[a]=total
    np.testing.assert_allclose(sum(result.values()),values['CSRO']-values['baseline'],rtol=0,atol=1e-10)
    return result


def aggregate():
    protocol=read(RUN/'protocol.json');complete=read(RUN/'complete.json')
    assert complete['status']=='complete' and complete['frames']==6019 and complete['scenes']==150
    assert complete['arms']==15 and read(RUN/'pilot.json')['status']=='passed'
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']==complete['checkpoint_sha256']
    for rel,digest in protocol['source_sha256'].items(): assert sha(ROOT/rel)==digest,rel
    cm=np.zeros((15,4,18,18),np.int64);cc=np.zeros((15,4,17,8),np.int64)
    trans=np.zeros((15,4,8),np.int64);overlap=np.zeros((4,15,15),np.int64)
    stats=np.zeros((15,4,len(ATTR_STATS)),np.float64);stats_max=np.zeros_like(stats)
    loss=np.zeros((15,21,3),np.float64);above=np.zeros((15,21),np.int64)
    indices=[];tokens=[];records=[]
    for path in sorted((RUN/'scenes').glob('*.json')):
        meta=read(path);indices+=meta['indices'];tokens+=meta['tokens']
        assert meta['checkpoint_sha256']==protocol['checkpoint_sha256']
        with np.load(path.with_suffix('.npz')) as z:
            count=len(meta['indices'])
            assert z['confusion'].shape==(count,15,4,18,18)
            assert z['class_coverage'].shape==(count,15,4,17,8)
            assert z['correction_overlap'].shape==(count,4,15,15)
            assert np.isfinite(z['loss_trajectory']).all() and np.isfinite(z['attribute_stats']).all()
            assert np.isfinite(z['max_gradient']).all()
            c=z['confusion'].sum(0);v=z['class_coverage'].sum(0);t=z['transitions'].sum(0)
            cm+=c;cc+=v;trans+=t;overlap+=z['correction_overlap'].sum(0)
            stats+=z['attribute_stats'].sum(0);stats_max=np.maximum(stats_max,z['attribute_stats'].max(0))
            loss+=z['loss_trajectory'].sum(0,dtype=np.float64)
            above+=(z['loss_trajectory'][...,0]>z['loss_trajectory'][:,:,0:1,0]).sum(0)
            records.append(dict(scene=meta['scene'],frames=count,confusion=c,class_coverage=v))
    assert len(records)==150 and sorted(indices)==list(range(6019)) and len(set(tokens))==6019
    for i in range(15):
        np.testing.assert_array_equal(cm[i,0],cm[0,0]);np.testing.assert_array_equal(cc[i,0],cc[0,0])
        for j in range(4):
            np.testing.assert_array_equal(cm[i,j,:17].sum(1),cc[i,j,:,0])
            assert cm[i,0,:17,17].sum()-cm[i,j,:17,17].sum()==trans[i,j,0]-trans[i,j,1]
            assert cm[i,0,17,:17].sum()-cm[i,j,17,:17].sum()==trans[i,j,2]-trans[i,j,3]
            assert np.trace(cm[i,j])-np.trace(cm[i,0])==trans[i,j,4]-trans[i,j,5]
    for c,k in enumerate((0,2,6,4)):
        np.testing.assert_array_equal(np.diag(overlap[c]),trans[:,-1,k])
        np.testing.assert_array_equal(overlap[c],overlap[c].T)
    with np.load(HISTORY/'aggregate/tiny.npz') as old:
        np.testing.assert_array_equal(cm[0,0],old['confusion'])
        np.testing.assert_array_equal(cc[0,0],old['class_coverage'])
    with np.load(CENTER_HISTORY.parent/'aggregate/tiny.npz') as old:
        for key,current in (('confusion',cm[0]),('class_coverage',cc[0]),('transitions',trans[0])):
            np.testing.assert_array_equal(current,old[key].sum(0))
    return protocol,dict(confusion=cm,class_coverage=cc,transitions=trans,overlap=overlap,
                         stats=stats,stats_max=stats_max,loss=loss/6019,above=above),records


def bootstrap(records):
    weights=np.random.default_rng(928).multinomial(150,np.full(150,1/150),size=1500)
    samples={};metrics={};intervals={}
    cm=np.stack([r['confusion'] for r in records])
    cc=np.stack([r['class_coverage'] for r in records])
    for arm in ('baseline',*ARMS):
        i,j=(0,0) if arm=='baseline' else (ARMS.index(arm),3)
        c=(weights@cm[:,i,j].reshape(150,-1)).reshape(-1,18,18)
        tp=np.diagonal(c,axis1=1,axis2=2)[:,:17];gt=c[:,:17].sum(2)
        union=gt+c.sum(1)[:,:17]-tp
        class_iou=np.divide(100*tp,union,out=np.zeros_like(tp,float),where=union>0)
        btp=c[:,:17,:17].sum((1,2))
        cov=weights@cc[:,i,j][:,:,[0,1]].sum(1)
        samples[arm]=class_iou.mean(1)
        metrics[arm]=dict(mIoU=samples[arm],IoU=100*btp/(btp+c[:,:17,17].sum(1)+c[:,17,:17].sum(1)),geo1=100*cov[:,1]/cov[:,0])
    for arm in ARMS:
        intervals[arm]={k:np.quantile(metrics[arm][k]-metrics['baseline'][k],[.025,.975]).tolist() for k in metrics[arm]}
    phi=shapley(samples)
    shapley_ci={a:np.quantile(phi[a],[.025,.975]).tolist() for a in ATTRIBUTES}
    return intervals,shapley_ci,samples


def tables(totals,records,samples,intervals,phi_ci):
    cm,cc=totals['confusion'],totals['class_coverage']
    baseline=describe(cm[0,0],cc[0,0]);metrics={};rows=[];classes=[];differences=[];scenes=[];trajectories=[];attribute_rows=[]
    rows.append(dict(arm='baseline',step=0,**baseline))
    for i,arm in enumerate(ARMS):
        metrics[arm]=[]
        for j,step in enumerate(STEPS):
            m=describe(cm[i,j],cc[i,j]);metrics[arm].append(m)
            rows.append(dict(arm=arm,step=step,**m))
            attribute_rows.append(dict(arm=arm,step=step,**changes(totals['stats'][i,j],totals['stats_max'][i,j])))
            for c,cname in enumerate(CLASS_NAMES):
                gt=int(cm[i,j,c].sum());tp=int(cm[i,j,c,c]);fp=int(cm[i,j,:,c].sum())-tp
                r=dict(arm=arm,step=step,class_name=cname,GT=gt,TP=tp,FP=fp,FN_to_empty=int(cm[i,j,c,17]),
                    IoU=100*tp/(gt+fp),**{key:100*int(cc[i,j,c,k])/gt for k,key in enumerate(CLASS_COLUMNS[1:],1)})
                classes.append(r)
                if step:
                    bgt=int(cm[i,0,c].sum());btp=int(cm[i,0,c,c]);bfp=int(cm[i,0,:,c].sum())-btp
                    differences.append(dict(arm=arm,step=step,class_name=cname,delta_IoU=r['IoU']-100*btp/(bgt+bfp),
                        **{f'delta_{key}':r[key]-100*int(cc[i,0,c,k])/bgt for k,key in enumerate(CLASS_COLUMNS[1:],1)}))
        for step in range(21):
            l=totals['loss'][i,step]
            trajectories.append(dict(arm=arm,step=step,total_loss=float(l[0]),weighted_CE=float(l[1]),weighted_Lovasz=float(l[2]),
                                     frames_loss_above_initial=int(totals['above'][i,step])))
    for record in records:
        for i,arm in enumerate(ARMS):
            for j,step in enumerate(STEPS):
                scenes.append(dict(scene=record['scene'],frames=record['frames'],arm=arm,step=step,
                    **describe(record['confusion'][i,j],record['class_coverage'][i,j])))
    final={'baseline':baseline,**{a:metrics[a][-1] for a in ARMS}}
    phi=shapley({a:v['mIoU'] for a,v in final.items()})
    phi_rows=[dict(attribute=a,Shapley_mIoU_pp=float(phi[a]),CI_low=phi_ci[a][0],CI_high=phi_ci[a][1],
                   interpretation='Budget-specific allocation of GT-assisted fitting gain; not causal training-error share') for a in ATTRIBUTES]
    marginal=[]
    for a in ATTRIBUTES:
        others=[x for x in ATTRIBUTES if x!=a]
        for k in range(4):
            for subset in combinations(others,k):
                before,after=name(subset),name((*subset,a))
                ci=np.quantile(samples[after]-samples[before],[.025,.975])
                marginal.append(dict(attribute=a,before=before,after=after,
                    **{f'delta_{key}':final[after][key]-final[before][key] for key in ('mIoU','IoU','geo1','FN_rate','FP_over_GTfg')},
                    mIoU_CI_low=ci[0],mIoU_CI_high=ci[1]))
    interaction=[]
    for a,b in combinations(ATTRIBUTES,2):
        others=[x for x in ATTRIBUTES if x not in (a,b)]
        for k in range(3):
            for subset in combinations(others,k):
                s,sa,sb,sab=name(subset),name((*subset,a)),name((*subset,b)),name((*subset,a,b))
                value=final[sab]['mIoU']-final[sa]['mIoU']-final[sb]['mIoU']+final[s]['mIoU']
                vals=samples[sab]-samples[sa]-samples[sb]+samples[s]
                ci=np.quantile(vals,[.025,.975])
                interaction.append(dict(pair=a+b,context=s,interaction_mIoU_pp=value,CI_low=ci[0],CI_high=ci[1]))
    overlap=[]
    for c,cname in enumerate(OVERLAP_NAMES):
        o=totals['overlap'][c]
        for i in range(15):
            for j in range(i+1,15):
                x,na,nb=int(o[i,j]),int(o[i,i]),int(o[j,j]);union=na+nb-x
                overlap.append(dict(category=cname,arm_A=ARMS[i],arm_B=ARMS[j],
                    corrected_A=na,corrected_B=nb,intersection=x,union=union,
                    Jaccard_percent=100*x/union if union else 0.,
                    A_also_fixed_by_B_percent=100*x/na if na else 0.,
                    B_also_fixed_by_A_percent=100*x/nb if nb else 0.))
    csv_write(AGG/'metrics.csv',rows);csv_write(AGG/'per_class.csv',classes)
    csv_write(AGG/'per_class_differences.csv',differences);csv_write(AGG/'per_scene.csv',scenes)
    csv_write(AGG/'loss_trajectory.csv',trajectories);csv_write(AGG/'attribute_changes.csv',attribute_rows)
    csv_write(AGG/'shapley.csv',phi_rows);csv_write(AGG/'conditional_gains.csv',marginal)
    csv_write(AGG/'interactions.csv',interaction);csv_write(AGG/'error_overlaps.csv',overlap)
    return baseline,metrics,classes,attribute_rows,phi_rows,marginal,interaction,overlap


def figures(baseline,metrics,totals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    colors=['#4477aa' if len(a)==1 else '#66aa99' if len(a)<4 else '#dd9955' for a in ARMS]
    for ax,key,title in zip(axes,('mIoU','IoU','geo1'),('mIoU','Binary IoU','Active 1-sigma coverage')):
        vals=[metrics[a][-1][key]-baseline[key] for a in ARMS]
        bars=ax.bar(ARMS,vals,color=colors)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=7)
        ax.axhline(0,color='black',lw=.7);ax.tick_params(axis='x',rotation=65)
        ax.set(title=title,ylabel='Change from original prediction (pp)')
        ax.margins(y=.15)
    fig.suptitle('Tiny | same-frame GT-assisted output fitting | 20 updates | 6,019 frames')
    for ext in ('png','pdf'):fig.savefig(FIGURES/f'attribute_gains.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,12),layout='constrained')
    axes=axes.flatten()
    for c,ax in enumerate(axes):
        o=totals['overlap'][c];d=np.diag(o)
        union=d[:,None]+d[None,:]-o
        jac=np.divide(100*o,union,out=np.zeros_like(o,float),where=union>0)
        im=ax.imshow(jac,vmin=0,vmax=100,cmap='viridis')
        ax.set(xticks=range(15),yticks=range(15),xticklabels=ARMS,yticklabels=ARMS,
               title=['Rescued FN','Removed FP','Corrected FG class','All corrections'][c])
        ax.tick_params(axis='x',rotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=7)
    fig.colorbar(im,ax=axes,label='Jaccard of corrected baseline errors (%)',shrink=.7)
    fig.suptitle('Overlap of errors repaired by different attribute groups | same baseline')
    for ext in ('png','pdf'):fig.savefig(FIGURES/f'correction_overlap.{ext}',dpi=180)
    plt.close(fig)


def report(baseline,metrics,classes,attrs,phi,marginal,interaction,overlap,intervals):
    final={a:metrics[a][-1] for a in ARMS}
    best=max(ARMS,key=lambda a:final[a]['mIoU'])
    single=max(ATTRIBUTES,key=lambda a:final[a]['mIoU'])
    lines=['# Tiny各属性与组合：完整GT辅助诊断结果','',
        '**全部15个非空组合已完成150场景、6,019帧。每组都使用本帧自身的GT拟合20步，网络与GS语义值固定。下列结果不是正常推理性能，也不是新的训练权重。**','',
        f"原始Tiny的mIoU为 **{baseline['mIoU']:.3f}%**。单项中本次收益最高的是 **{single}**，第20步为 **{final[single]['mIoU']:.3f}%**；组合中最高的是 **{best}**，为 **{final[best]['mIoU']:.3f}%**。这个排名限于下述约束、参数化和20步优化预算。",'',
        'C为中心、S为尺度、R为旋转、O为不透明度。C最多移动0.4米；S每轴为0.08–0.4米；R相对旋转最多45°；O为0–1。每组从相同原始预测独立出发。完整约定见[设计文档](gaussian_attribute_oracle_full_val_design.md)。','',
        '## 1. 单项与组合的整体收益','',
        '变化单位为百分点；所有指标先汇总全验证集计数，再计算比值。第20步是预先指定的主要结果。','',
        '| 组别 | mIoU | ΔmIoU | 二值IoU | Δ二值IoU | 1σ覆盖 | Δ1σ覆盖 | 同类1σ | 漏检/GT前景 | 二值FP/GT前景 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines.append(f"| 原预测 | {baseline['mIoU']:.3f}% | — | {baseline['IoU']:.3f}% | — | {baseline['geo1']:.3f}% | — | {baseline['same1']:.3f}% | {baseline['FN_rate']:.3f}% | {baseline['FP_over_GTfg']:.3f}% |")
    for a in ARMS:
        r=final[a]
        lines.append(f"| {a} | {r['mIoU']:.3f}% | {r['mIoU']-baseline['mIoU']:+.3f} | {r['IoU']:.3f}% | {r['IoU']-baseline['IoU']:+.3f} | {r['geo1']:.3f}% | {r['geo1']-baseline['geo1']:+.3f} | {r['same1']:.3f}% | {r['FN_rate']:.3f}% | {r['FP_over_GTfg']:.3f}% |")
    lines+=['','覆盖沿用此前“原生候选范围内、opacity乘核值为正”的口径。因此O将GS置零时，活动支持覆盖也会下降，即使中心、尺度和朝向完全未变。它不能被直接解释为几何变差。','',
        '![各属性组合的收益](figures/gaussian_attribute_oracle_full_val/attribute_gains.png)','',
        '| 单项 | ΔmIoU的95%场景区间 | 第5步mIoU | 第10步mIoU | 第20步mIoU |',
        '|---|---|---:|---:|---:|']
    for a in ATTRIBUTES:
        ci=intervals[a]['mIoU']
        lines.append(f"| {a} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {metrics[a][1]['mIoU']:.3f}% | {metrics[a][2]['mIoU']:.3f}% | {metrics[a][3]['mIoU']:.3f}% |")
    lines+=['','区间按相同150个场景配对bootstrap，重采样1,500次、种子928。不代表重新训练的随机性。全部组别的四个记录步骤与区间见CSV和汇总JSON。','',
        '## 2. 已经调整中心后，其他属性还能增加多少收益','',
        '下表使用同一原始起点独立优化的组别作差，不是先优化前一组再接着优化后一组。有限20步的联合优化不保证优于所有子组。','',
        '| 比较 | 额外开放的属性 | ΔmIoU | Δ二值IoU | Δ1σ覆盖 | Δ漏检率 | Δ二值FP/GT前景 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for before,after in [('C','CS'),('C','CR'),('C','CO'),('S','SR'),('CS','CSR'),('CSR','CSRO')]:
        b,r=final[before],final[after];added=''.join(x for x in after if x not in before)
        lines.append('| '+after+'−'+before+' | '+added+' | '+' | '.join(f'{r[k]-b[k]:+.3f}' for k in ('mIoU','IoU','geo1','FN_rate','FP_over_GTfg'))+' |')
    lines+=['','完整32项条件边际收益及区间见[条件收益CSV](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/conditional_gains.csv)。','',
        '## 3. 各属性的收益是否互相替代','',
        '下面的交互值为“两项联合收益”减去“两项单独收益之和”。正值表示超出简单相加，负值表示联合收益小于相加；它受到约束和有限步优化影响，不是原训练的因果机制证明。','',
        '| 属性对 | mIoU交互/百分点 | 95%场景区间 |','|---|---:|---|']
    for r in interaction:
        if r['context']=='baseline': lines.append(f"| {r['pair']} | {r['interaction_mIoU_pp']:+.3f} | [{r['CI_low']:+.3f}, {r['CI_high']:+.3f}] |")
    lines+=['','Shapley汇总将四个属性的24种加入顺序逐一比较，平均每个属性加入时带来的mIoU变化。四项相加恰好等于CSRO相对原预测的收益。**它分摊的是本次GT辅助拟合收益，不是原训练错误的成因比例。**','',
        '| 属性 | 平均边际收益/百分点 | 95%场景区间 |','|---|---:|---|']
    for r in phi: lines.append(f"| {r['attribute']} | {r['Shapley_mIoU_pp']:+.3f} | [{r['CI_low']:+.3f}, {r['CI_high']:+.3f}] |")
    lines+=['','## 4. 修复的是不是同一批体素','',
        '重叠均相对同一份原始预测定义。Jaccard为双方修复体素交集除以并集；某组没有修复体素时相关比例记为0。下表展示全部正确修复的重叠，完整CSV另列漏检救回、FP消除和前景类别纠正。','',
        '| A与B | A修复数 | B修复数 | 共同修复数 | Jaccard | A修复中也被B修复 |',
        '|---|---:|---:|---:|---:|---:|']
    wanted=[('C','S'),('C','R'),('C','O'),('S','R'),('C','CSRO'),('CSR','CSRO')]
    for aa,bb in wanted:
        x=next(r for r in overlap if r['category']=='all_corrected_predictions' and set((r['arm_A'],r['arm_B']))==set((aa,bb)))
        if x['arm_A']==aa: na,nb,share=x['corrected_A'],x['corrected_B'],x['A_also_fixed_by_B_percent']
        else: na,nb,share=x['corrected_B'],x['corrected_A'],x['B_also_fixed_by_A_percent']
        lines.append(f"| {aa} / {bb} | {na:,} | {nb:,} | {x['intersection']:,} | {x['Jaccard_percent']:.3f}% | {share:.3f}% |")
    lines+=['','![不同属性组修复错误的重叠](figures/gaussian_attribute_oracle_full_val/correction_overlap.png)','',
        '## 5. 属性实际改变了多少','',
        '| 组别 | 中心平均位移/m | 每轴尺度平均绝对变化/m | 平均旋转/° | 不透明度平均绝对变化 | 协方差平均相对变化 |',
        '|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        r=next(x for x in attrs if x['arm']==a and x['step']==20)
        lines.append(f"| {a} | {r['center_mean_m']:.4f} | {r['scale_axis_mean_abs_change_m']:.4f} | {r['rotation_mean_deg']:.3f} | {r['opacity_mean_abs_change']:.4f} | {r['covariance_mean_relative_change']:.4f} |")
    r=next(x for x in attrs if x['arm']=='R' and x['step']==20)
    lines+=['',f"原预测有 **{r['initial_near_spherical_percent']:.3f}%** 的GS轴比不超过1.05。旋转对球形高斯没有几何作用，单独R的收益必须与SR相对S的增量一起看。原网络的sigmoid饱和没有用于限制本次S/O的输出优化，因此这里衡量的是可修正空间，而非原分支的梯度可达性。",'',
        '## 6. 逐类IoU与覆盖变化','',
        '先展示四个单项和全部属性联合组；15组全部步骤的绝对值与变化在逐类CSV中。每格为“ΔIoU / Δ1σ活动支持覆盖”，单位为百分点。','',
        '| 类别 | C | S | R | O | CSRO |','|---|---:|---:|---:|---:|---:|']
    lookup={(r['arm'],r['step'],r['class_name']):r for r in classes}
    for cname in CLASS_NAMES:
        vals=[]
        for a in ('C','S','R','O','CSRO'):
            b,r=lookup[a,0,cname],lookup[a,20,cname]
            vals.append(f"{r['IoU']-b['IoU']:+.3f} / {r['geo1']-b['geo1']:+.3f}")
        lines.append('| '+cname+' | '+' | '.join(vals)+' |')
    lines+=['','## 7. 如何据此决定下一步','',
        '本轮能回答：在当前语义值固定、给定调整范围和20步优化的条件下，哪种属性能修正更多错误；额外开放某属性后是否有增量；不同属性是否修复相同体素。','',
        '数据集只有体素GT，没有逐个GS的中心、尺度、朝向真值；同一占有分布也可能由不同高斯配置表示。因此这里没有直接测量逐属性的预测偏差。','',
        '单项收益大不等于该属性是原模型的主要成因。属性之间可以补偿，范围和自由度也不同；不透明度属于证据强度，不能把它的收益称为几何位置改善。旋转收益小也不等于朝向准确，尤其是接近球形的GS。','',
        '网络没有学习这些逐帧调整，所有拟合使用验证标签。这些分数不能用于SOTA对比，也不能直接证明哪个属性的泛化最差。训练目标或结构的可部署收益，仍需只在训练集学习，再用不参与任何更新的验证集评价。','',
        '## 8. 核验与数据入口','',
        '- 完整150场景、6,019帧；所有15组使用完全相同的原始预测、GT范围和帧身份。',
        '- 15组零更新logits均与原模型逐元素一致；每帧混淆计数和覆盖计数复现历史基线。',
        '- C组四次记录的混淆、覆盖、体素转变以及21次损失均逐帧精确复现已有中心诊断。',
        '- C、S、R、O的真实CUDA梯度均通过有限差分核对，球形协方差旋转不变性检查通过。',
        '- 全部损失和梯度有限，未开放属性保持原值，网络参数没有接收梯度；原权重未被覆盖。',
        '- 修复交集在首帧经独立枚举核对，全部交集矩阵对称且对角计数与修复统计一致。','',
        '| 文件 | 内容 |','|---|---|',
        '| [整体指标](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/metrics.csv) | 全部组别0/5/10/20步 |',
        '| [逐类绝对值](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/per_class.csv) | 各类IoU、覆盖、TP/FP/FN |',
        '| [逐类变化](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/per_class_differences.csv) | 相对原预测的变化 |',
        '| [条件边际收益](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/conditional_gains.csv) | 32种加入属性的比较 |',
        '| [交互项](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/interactions.csv) | 不同上下文下的两属性交互 |',
        '| [错误修复重叠](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/error_overlaps.csv) | 四类错误、每对组别的交集与并集 |',
        '| [属性变化](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/attribute_changes.csv) | 位移、尺度、角度、强度、约束命中 |',
        '| [优化轨迹](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/loss_trajectory.csv) | CE、Lovasz、损失异常变化 |',
        '| [逐场景指标](../work_dirs/gaussian_attribute_oracle_full_val_v1/aggregate/per_scene.csv) | 配对场景分析 |',
        '| [完整JSON](../work_dirs/gaussian_attribute_oracle_full_val_v1/summary.json) | 精确结果与场景置信区间 |',
        '| [完整性审计](../work_dirs/gaussian_attribute_oracle_full_val_v1/delivery_audit.json) | 输入、来源与产物哈希 |','']
    REPORT.write_text('\n'.join(lines))


def main():
    AGG.mkdir(parents=True,exist_ok=True)
    protocol,totals,records=aggregate()
    intervals,phi_ci,samples=bootstrap(records)
    b,m,c,a,p,g,i,o=tables(totals,records,samples,intervals,phi_ci)
    with (AGG/'totals.npz').open('wb') as f: np.savez_compressed(f,**totals)
    summary=dict(status='complete',model='tiny',frames=6019,scenes=150,arms=list(ARMS),
        baseline=b,metrics=m,step20_delta_intervals=intervals,shapley=p,
        conditional_gains=g,interactions=i,checkpoint_sha256=protocol['checkpoint_sha256'],
        interpretation='Same-frame GT-assisted fitting, not normal validation or causal error fractions')
    write(RUN/'summary.json',summary)
    figures(b,m,totals)
    report(b,m,c,a,p,g,i,o,intervals)
    files=[REPORT,ROOT/'analysis/gaussian_attribute_oracle_full_val_design.md',Path(__file__),
           ROOT/'tools/run_gaussian_attribute_oracle.py',
           ROOT/'tools/eval_gaussian_attribute_oracle.py',RUN/'summary.json',RUN/'protocol.json',RUN/'pilot.json',RUN/'complete.json',
           *sorted(AGG.glob('*')),*sorted(FIGURES.glob('*'))]
    write(RUN/'delivery_audit.json',dict(status='passed',frames=6019,scenes=150,arms=15,
        exact_native_baseline=True,exact_center_history=True,finite_losses_and_gradients=True,
        correction_overlaps_reconcile=True,Shapley_sum_reconciles=True,
        source_and_checkpoint_hashes_verified=True,artifact_sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    print(json.dumps(summary,indent=2),flush=True)
    print('All 15 attribute groups and report checks passed.',flush=True)


if __name__=='__main__': main()
