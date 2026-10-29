"""Summarize the complete frozen-geometry semantic-only GT-assisted diagnostic."""
from pathlib import Path
import csv
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.eval_gaussian_semantic_oracle import (
    RUN,HISTORY,ATTRIBUTE_HISTORY,STEPS,SEMANTIC_STATS,TRANSITIONS,CLASS_NAMES,CLASS_COLUMNS)
from tools.eval_gaussian_attribute_oracle import ARMS
from tools.report_gaussian_sizes import performance
from tools.stat_boundary_gaussians import read,write,sha

AGG=RUN/'aggregate'
REPORT=ROOT/'analysis/gaussian_semantic_oracle_full_val_results.md'
FIGURES=ROOT/'analysis/figures/gaussian_semantic_oracle_full_val'
COMPARE=('C','S','R','O','CS','CSRO')


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def describe(cm,cc):
    result=performance(cm)
    mis=int(cm[:17,:17].sum()-np.trace(cm[:17,:17]))
    result.update(foreground_class_errors=mis,class_error_over_GTfg=100*mis/result['GTfg'],
        valid_GT=int(cm.sum()),FN_over_valid_GT=100*int(cm[:17,17].sum())/cm.sum(),
        FP_over_valid_GT=100*int(cm[17,:17].sum())/cm.sum(),
        class_error_over_valid_GT=100*mis/cm.sum())
    for j,key in enumerate(CLASS_COLUMNS[1:],1):result[key]=100*int(cc[:,j].sum())/int(cc[:,0].sum())
    return result


def semantic_changes(total,maximum):
    d=dict(zip(SEMANTIC_STATS,total));m=dict(zip(SEMANTIC_STATS,maximum));n=d['GS']
    return dict(mean_semantic_mass=d['sum_semantic_mass']/n,
        initial_mean_semantic_mass=d['initial_sum_semantic_mass']/n,
        mean_abs_change_per_class=d['sum_abs_change']/d['entries'],
        max_abs_change=m['max_abs_change'],changed_GS_percent=100*d['changed_GS']/n,
        argmax_changed_GS_percent=100*d['changed_argmax_GS']/n,
        zero_entries_percent=100*d['zero_entries']/d['entries'],
        zero_mass_GS_percent=100*d['zero_mass_GS']/n,
        mean_normalized_entropy=d['entropy_sum']/n,
        initial_mean_normalized_entropy=d['initial_entropy_sum']/n,
        mass_increased_GS_percent=100*d['mass_increased_GS']/n,
        mass_decreased_GS_percent=100*d['mass_decreased_GS']/n)


def aggregate():
    protocol=read(RUN/'protocol.json');complete=read(RUN/'complete.json')
    assert complete['status']=='complete' and complete['frames']==6019 and complete['scenes']==150
    assert read(RUN/'pilot.json')['status']=='passed'
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']==complete['checkpoint_sha256']
    for p,digest in protocol['source_sha256'].items():assert sha(ROOT/p)==digest,p
    arrays={};records=[];indices=[];tokens=[]
    for path in sorted((RUN/'scenes').glob('*.json')):
        meta=read(path);indices+=meta['indices'];tokens+=meta['tokens']
        assert meta['checkpoint_sha256']==protocol['checkpoint_sha256']
        with np.load(path.with_suffix('.npz')) as z:
            assert z['confusion'].shape==(meta['frames'],4,18,18)
            assert z['class_coverage'].shape==(meta['frames'],4,17,8)
            for key in z.files:
                value=z[key];assert np.isfinite(value).all(),(path,key)
                arrays.setdefault(key,[]).append(value)
            for j in range(4):
                np.testing.assert_array_equal(z['class_coverage'][:,j][:,:,[0,1,2,3,7]],
                                               z['class_coverage'][:,0][:,:,[0,1,2,3,7]])
            records.append(dict(scene=meta['scene'],frames=meta['frames'],
                confusion=z['confusion'].sum(0),class_coverage=z['class_coverage'].sum(0)))
    assert len(records)==150 and sorted(indices)==list(range(6019)) and len(set(tokens))==6019
    order=np.argsort(indices);arrays={k:np.concatenate(v)[order] for k,v in arrays.items()}
    arrays['frame_indices']=np.array(indices)[order]
    cm=arrays['confusion'].sum(0);cc=arrays['class_coverage'].sum(0);tr=arrays['transitions'].sum(0)
    with np.load(HISTORY/'aggregate/tiny.npz') as old:
        np.testing.assert_array_equal(cm[0],old['confusion']);np.testing.assert_array_equal(cc[0],old['class_coverage'])
    np.testing.assert_array_equal(tr[0],0)
    for j in range(4):
        np.testing.assert_array_equal(cm[j,:17].sum(1),cc[j,:,0])
        assert cm[0,:17,17].sum()-cm[j,:17,17].sum()==tr[j,0]-tr[j,1]
        assert cm[0,17,:17].sum()-cm[j,17,:17].sum()==tr[j,2]-tr[j,3]
        assert np.trace(cm[j])-np.trace(cm[0])==tr[j,4]-tr[j,5]
    attr=[]
    for record in records:
        p=ATTRIBUTE_HISTORY/'scenes'/f"{record['scene']}.npz"
        with np.load(p) as z:
            np.testing.assert_array_equal(record['confusion'][0],z['confusion'][:,0,0].sum(0))
            attr.append(z['confusion'].sum(0)[:,-1])
    return protocol,arrays,records,np.stack(attr)


def sample_metrics(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)[...,:17]
    gt=cm[...,:17,:].sum(-1);union=gt+cm.sum(-2)[...,:17]-tp
    ciou=np.divide(100*tp,union,out=np.zeros_like(tp,float),where=union>0)
    btp=cm[...,:17,:17].sum((-2,-1));fn=cm[...,:17,17].sum(-1);fp=cm[...,17,:17].sum(-1)
    return dict(mIoU=ciou.mean(-1),IoU=100*btp/(btp+fn+fp),
        FN_rate=100*fn/gt.sum(-1),FP_over_GTfg=100*fp/gt.sum(-1),
        class_error_over_GTfg=100*(btp-tp.sum(-1))/gt.sum(-1))


def intervals(records,attribute_scenes):
    w=np.random.default_rng(928).multinomial(150,np.full(150,1/150),size=1500)
    scenes=np.stack([r['confusion'] for r in records]);samples=[]
    for j in range(4):
        cm=(w@scenes[:,j].reshape(150,-1)).reshape(-1,18,18)
        samples.append(sample_metrics(cm))
    own={str(STEPS[j]):{k:np.quantile(samples[j][k]-samples[0][k],[.025,.975]).tolist()
        for k in samples[j]} for j in range(1,4)}
    comparisons={}
    for i,arm in enumerate(ARMS):
        cm=(w@attribute_scenes[:,i].reshape(150,-1)).reshape(-1,18,18)
        other=sample_metrics(cm)
        comparisons[arm]={k:np.quantile(samples[-1][k]-other[k],[.025,.975]).tolist()
            for k in samples[-1]}
    return own,comparisons


def tables(arrays,records,attribute_scenes,pair_ci):
    cm=arrays['confusion'].sum(0);cc=arrays['class_coverage'].sum(0);tr=arrays['transitions'].sum(0)
    trajectory=arrays['loss_trajectory'];sem=arrays['semantic_statistics']
    metrics=[];classes=[];scenes=[];changes=[]
    for j,step in enumerate(STEPS):
        m=dict(arm='E',step=step,**describe(cm[j],cc[j]),
            mean_total_loss=float(trajectory[:,step,0].mean()),
            **{k:int(tr[j,i]) for i,k in enumerate(TRANSITIONS)})
        metrics.append(m)
        changes.append(dict(step=step,**semantic_changes(sem[:,j].sum(0),sem[:,j].max(0))))
        for i,name in enumerate(CLASS_NAMES):
            gt=int(cm[j,i].sum());tp=int(cm[j,i,i]);fpempty=int(cm[j,17,i]);fpother=int(cm[j,:17,i].sum())-tp
            classes.append(dict(step=step,class_id=i,class_name=name,GT=gt,TP=tp,FP_empty=fpempty,
                FP_other_foreground=fpother,FN_to_empty=int(cm[j,i,17]),IoU=100*tp/(gt+fpempty+fpother),
                **{k:100*int(cc[j,i,n])/gt for n,k in enumerate(CLASS_COLUMNS[1:],1)}))
        for r in records:
            scenes.append(dict(scene=r['scene'],frames=r['frames'],step=step,
                **describe(r['confusion'][j],r['class_coverage'][j])))
    with np.load(ATTRIBUTE_HISTORY/'aggregate/totals.npz') as z:
        other_cm=z['confusion'][:,-1];other_cc=z['class_coverage'][:,-1]
    comparators={arm:describe(other_cm[i],other_cc[i]) for i,arm in enumerate(ARMS)}
    pairs=[]
    for arm in ARMS:
        pairs.append(dict(comparison=f'E-{arm}',
            **{f'delta_{k}':metrics[-1][k]-comparators[arm][k] for k in
                ('mIoU','IoU','FN_rate','FP_over_GTfg','class_error_over_GTfg','geo1','same1')},
            mIoU_CI_low=pair_ci[arm]['mIoU'][0],mIoU_CI_high=pair_ci[arm]['mIoU'][1]))
    losses=[dict(step=i,total_loss=float(trajectory[:,i,0].mean()),
        weighted_CE=float(trajectory[:,i,1].mean()),weighted_Lovasz=float(trajectory[:,i,2].mean()),
        frames_above_initial=int((trajectory[:,i,0]>trajectory[:,0,0]).sum())) for i in range(21)]
    for name,rows in [('metrics',metrics),('per_class',classes),('per_scene',scenes),
        ('comparisons',pairs),('semantic_changes',changes),('loss_trajectory',losses)]:
        csv_write(AGG/f'{name}.csv',rows)
    return metrics,classes,changes,comparators,pairs


def figures(metrics,comparators):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    FIGURES.mkdir(parents=True,exist_ok=True)
    keys=['baseline','C','S','R','O','CS','CSRO','E']
    values=[metrics[0],*[comparators[k] for k in keys[1:-1]],metrics[-1]]
    fig,axes=plt.subplots(1,3,figsize=(15,4.1),layout='constrained')
    colors=['#999999']+['#4477aa']*6+['#dd9955']
    for ax,key,title in zip(axes,['mIoU','FN_rate','FP_over_GTfg'],
        ['mIoU','Misses / GT foreground','Binary FP / GT foreground']):
        heights=[v[key] for v in values];ax.bar(keys,heights,color=colors)
        ax.bar_label(ax.containers[0],fmt='%.2f',fontsize=8,padding=2)
        ax.set(title=title,ylabel='%',ylim=(0,max(heights)*1.15))
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Tiny | same-frame GT-assisted output fitting | 20 updates | 6,019 frames\nE: semantics only; geometry and opacity fixed')
    for ext in ('png','pdf'):fig.savefig(FIGURES/f'semantic_comparison.{ext}',dpi=180)
    plt.close(fig)


def report(metrics,classes,changes,comparators,pairs,own_ci):
    b,e=metrics[0],metrics[-1];c=changes[-1]
    lines=['# 固定几何，只优化语义：完整验证集诊断结果','',
        '**实验已完成：Tiny 原始权重、150 个场景、6,019 帧，每帧独立优化 20 步。**',
        '',f"语义单项 E 的 mIoU 为 **{b['mIoU']:.3f}% → {e['mIoU']:.3f}%（{e['mIoU']-b['mIoU']:+.3f} 个百分点）**；二值 IoU 为 **{b['IoU']:.3f}% → {e['IoU']:.3f}%**。",'',
        '**每帧使用自身验证 GT 调整输出。网络没有训练，结果不能作为正常推理性能，也不生成新的训练权重。**',
        '', 'E 表示每个 GS 的 17 类非负语义值，类别比例与总幅度均可改变。中心、尺度、旋转、opacity、empty 参数和 GS 数量固定。',
        '执行范围与公式见[实验设计](gaussian_semantic_oracle_full_val_design.md)。','',
        '## 1. 与几何及 opacity 调整的对照','',
        '表内均为第 20 步；漏检、前景类别错分和二值 FP 统一除以 GT 前景体素数。它们互斥：前景→empty、前景→其他前景类别、empty→前景。','',
        '| 组别 | mIoU | ΔmIoU/百分点 | 二值IoU | 漏检/GT前景 | 类别错分/GT前景 | FP/GT前景 | 1σ几何覆盖 | 同类1σ覆盖 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,m in [('原预测',b),*[(a,comparators[a]) for a in COMPARE],('E：仅语义',e)]:
        lines.append(f"| {arm} | {m['mIoU']:.3f}% | {m['mIoU']-b['mIoU']:+.3f} | "+
            ' | '.join(f"{m[k]:.3f}%" for k in ['IoU','FN_rate','class_error_over_GTfg','FP_over_GTfg','geo1','same1'])+' |')
    lines+=['','C＝中心，S＝尺度，R＝旋转，O＝opacity。E 的几何覆盖逐帧严格不变；同类覆盖可以因 GS 类别改变而变化。全部 15 个属性组的比较见 CSV。','',
        '![语义与属性优化对照](figures/gaussian_semantic_oracle_full_val/semantic_comparison.png)','',
        '## 2. 20 步内如何变化','',
        '| 更新步数 | mIoU | 二值IoU | 漏检/GT前景 | 类别错分/GT前景 | FP/GT前景 |',
        '|---|---:|---:|---:|---:|---:|']
    for m in metrics:
        lines.append(f"| {m['step']} | "+' | '.join(f"{m[k]:.3f}%" for k in
            ['mIoU','IoU','FN_rate','class_error_over_GTfg','FP_over_GTfg'])+' |')
    lo,hi=own_ci['20']['mIoU']
    lines+=['',f'第20步相对原预测的 mIoU 差值，场景配对95%区间为 [{lo:+.3f}, {hi:+.3f}] 个百分点。按相同150个场景重采样1,500次，种子928；不包含重新训练的随机性。','',
        '| 比较 | ΔmIoU/百分点 | 95%场景区间 | Δ漏检率/百分点 | ΔFP/GT前景/百分点 |',
        '|---|---:|---|---:|---:|']
    for p in pairs:
        if p['comparison'][2:] in COMPARE:
            lines.append(f"| {p['comparison']} | {p['delta_mIoU']:+.3f} | [{p['mIoU_CI_low']:+.3f}, {p['mIoU_CI_high']:+.3f}] | {p['delta_FN_rate']:+.3f} | {p['delta_FP_over_GTfg']:+.3f} |")
    lines+=['','## 3. 语义值实际改变了什么','',
        f"- GS 平均语义总幅度：{c['initial_mean_semantic_mass']:.5f} → {c['mean_semantic_mass']:.5f}。",
        f"- 17维数值向量 argmax 改变的 GS：{c['argmax_changed_GS_percent']:.3f}%。该计数按数值 argmax 定义，全零向量返回0；全零比例单独列出，不将其解释为 others 类别。",
        f"- 语义总幅度增加/减少的 GS：{c['mass_increased_GS_percent']:.3f}% / {c['mass_decreased_GS_percent']:.3f}%。",
        f"- 全部17类语义值均为0的 GS：{c['zero_mass_GS_percent']:.3f}%。这些 GS 不计作任何类别的同类支持，但保留其固定几何支持统计。",
        f"- 对17类值归一化后，平均熵：{c['initial_mean_normalized_entropy']:.5f} → {c['mean_normalized_entropy']:.5f} nats；全零向量的熵记0。",'',
        '这组实验同时开放类别比例与总幅度。收益不能全部解释为类别判断改善；改变幅度也会改变前景与 empty 的竞争。没有把语义替换成 softmax 概率。','',
        '## 4. 逐类 IoU 与同类覆盖','',
        '几何1σ覆盖各类均与基线严格相同，因此只列一次。同类覆盖按当前 GS 语义 argmax 重新计算，语义全零的 GS 无类别。','',
        '| 类别 | 原IoU | E IoU | ΔIoU/百分点 | 固定1σ几何覆盖 | 原同类1σ | E同类1σ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for i,name in enumerate(CLASS_NAMES):
        x,y=classes[i],classes[3*17+i]
        lines.append(f"| {name} | {x['IoU']:.3f}% | {y['IoU']:.3f}% | {y['IoU']-x['IoU']:+.3f} | {x['geo1']:.3f}% | {x['same1']:.3f}% | {y['same1']:.3f}% |")
    pc=next(p for p in pairs if p['comparison']=='E-C')
    pall=next(p for p in pairs if p['comparison']=='E-CSRO')
    lines+=['','## 5. 可以得出什么结论','',
        f"**固定几何，语义调整仍有很大的修正空间。** mIoU 提高 {e['mIoU']-b['mIoU']:.3f} 个百分点，漏检由 {b['FN_rate']:.3f}% 降至 {e['FN_rate']:.3f}%，二值FP由 {b['FP_over_GTfg']:.3f}% 降至 {e['FP_over_GTfg']:.3f}%，前景类别错分由 {b['class_error_over_GTfg']:.3f}% 降至 {e['class_error_over_GTfg']:.3f}%。这说明巨大GT辅助收益并非几何调整所独有。",
        '',f"**语义组与中心组修复错误的侧重不同。** E 相对 C 的 mIoU 为 {pc['delta_mIoU']:+.3f} 个百分点，二值IoU却为 {pc['delta_IoU']:+.3f} 个百分点；E的前景类别错分更少，但漏检与二值FP均更高。mIoU还取决于逐类别TP、FP及类别之间的分布，不能用总体二值错误数量直接替代。",
        '',f"**本次数据不足以认定 E 的 mIoU 稳定高于 CSRO。** 点估计差值为 {pall['delta_mIoU']:+.3f} 个百分点，场景配对95%区间为 [{pall['mIoU_CI_low']:+.3f}, {pall['mIoU_CI_high']:+.3f}]，跨过0；E相对CSRO的二值IoU为 {pall['delta_IoU']:+.3f} 个百分点、二值FP/GT前景为 {pall['delta_FP_over_GTfg']:+.3f} 个百分点。",
        '',f"**语义优化没有表现为全局幅度统一增强。** 平均语义总幅度由 {c['initial_mean_semantic_mass']:.5f} 降至 {c['mean_semantic_mass']:.5f}，{c['mass_decreased_GS_percent']:.3f}% 的GS总幅度下降，同时归一化后的平均熵降低。这记录了幅度和类别分布共同变化，不能单独分离它们各自的贡献。",
        '', '每个 GS 的语义有17个自由度，中心有3个；学习率、量纲和合法范围不同。E 允许非负语义幅度增加而没有额外上界，20步也不是全局收敛。因此，排名只能解释为本次有限预算下的可修正收益。',
        '', '几何固定时，语义仍然可以改变错误空间影响的强度。1σ覆盖不变而性能改变，并不否定几何预测的重要性，也不能证明语义分支或几何分支能在训练后泛化这些逐帧调整。',
        '', '## 6. 完整性与数据','',
        '- 完整150场景、6,019帧，每帧身份只出现一次，原始权重及测量源码哈希保持一致。',
        '- 零更新 logits 与原模型逐元素一致；每帧基线混淆矩阵和覆盖统计与历史完整验证一致。',
        '- 原生CUDA语义梯度通过有限差分；缓存固定empty后的语义梯度与完整原生渲染逐元素一致。',
        '- 所有几何与opacity固定，网络参数无梯度，语义非负，全部损失和梯度有限。',
        '- 每帧各步的1σ、2σ、3σ及原生几何支持逐元素相同，动态类别覆盖在真实GT位置枚举所有GS核验。',
        '- 漏检、FP及正确预测的净变化与体素转变计数一致。','',
        '| 文件 | 内容 |','|---|---|',
        '| [整体指标](../work_dirs/gaussian_semantic_oracle_full_val_v1/aggregate/metrics.csv) | 0/5/10/20步、错误数量和两种分母 |',
        '| [逐类指标](../work_dirs/gaussian_semantic_oracle_full_val_v1/aggregate/per_class.csv) | IoU、覆盖、empty与其他前景产生的类别FP |',
        '| [与15个属性组的差值](../work_dirs/gaussian_semantic_oracle_full_val_v1/aggregate/comparisons.csv) | 性能、错误、覆盖变化及mIoU区间 |',
        '| [语义变化](../work_dirs/gaussian_semantic_oracle_full_val_v1/aggregate/semantic_changes.csv) | 总幅度、类别、熵及全零比例 |',
        '| [逐场景指标](../work_dirs/gaussian_semantic_oracle_full_val_v1/aggregate/per_scene.csv) | 所有场景与记录步骤 |',
        '| [优化轨迹](../work_dirs/gaussian_semantic_oracle_full_val_v1/aggregate/loss_trajectory.csv) | 原CE与Lovasz、损失高于初始的帧数 |',
        '| [完整JSON](../work_dirs/gaussian_semantic_oracle_full_val_v1/summary.json) | 精确指标与配对区间 |',
        '| [完整性核验](../work_dirs/gaussian_semantic_oracle_full_val_v1/delivery_audit.json) | 来源与产物哈希 |','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def main():
    AGG.mkdir(parents=True,exist_ok=True)
    protocol,arrays,records,attrs=aggregate()
    np.savez_compressed(AGG/'tiny.npz',**arrays)
    own_ci,pair_ci=intervals(records,attrs)
    metrics,classes,changes,comparators,pairs=tables(arrays,records,attrs,pair_ci)
    figures(metrics,comparators);report(metrics,classes,changes,comparators,pairs,own_ci)
    summary=dict(status='complete',frames=6019,scenes=150,metrics=metrics,
        semantic_changes=changes,baseline_delta_CI=own_ci,paired_delta_CI=pair_ci,comparisons=pairs,
        checkpoint_sha256=protocol['checkpoint_sha256'],
        interpretation='Same-frame GT-assisted semantic output fitting; not normal inference')
    write(RUN/'summary.json',summary)
    artifacts=[REPORT,ROOT/'analysis/gaussian_semantic_oracle_full_val_design.md',
        ROOT/'tools/eval_gaussian_semantic_oracle.py',Path(__file__),ROOT/'tools/run_gaussian_semantic_oracle.py',
        RUN/'protocol.json',RUN/'pilot.json',RUN/'complete.json',RUN/'summary.json',
        *sorted(AGG.glob('*')),*sorted(FIGURES.glob('*'))]
    audit=dict(status='passed',frames=6019,scenes=150,
        native_baseline_exact=True,geometry_and_opacity_fixed=True,
        geometric_support_exactly_constant=True,finite_losses_and_gradients=True,
        source_and_checkpoint_hashes_verified=True,
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts})
    write(RUN/'delivery_audit.json',audit)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
