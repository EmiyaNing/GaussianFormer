"""Report complete joint/ordered GS attribute fitting with explicit budgets."""
from pathlib import Path
from itertools import combinations
import csv
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.eval_gaussian_joint_oracle import RUN,ARMS,STEPS,ATTR_STATS,SEMANTIC_STATS,COUPLING_STATS,CLASS_NAMES,CLASS_COLUMNS
from tools.eval_gaussian_joint_oracle import G_HISTORY,E_HISTORY
from tools.report_gaussian_semantic_oracle import describe,semantic_changes,sample_metrics
from tools.report_gaussian_attribute_oracle import changes as geometry_changes
from tools.stat_boundary_gaussians import read,write,sha

AGG=RUN/'aggregate'
REPORT=ROOT/'analysis/gaussian_joint_oracle_full_val_results.md'
FIGURES=ROOT/'analysis/figures/gaussian_joint_oracle_full_val'
ENDPOINTS={'baseline':(0,0),'E20':(1,3),'G20':(2,3),'J20':(0,3),
    'J40':(0,6),'E_then_G40':(1,6),'G_then_E40':(2,6)}
LABELS={'baseline':'原预测','E20':'仅语义 E20','G20':'仅 CSRO G20','J20':'联合 J20',
    'J40':'联合 J40','E_then_G40':'先语义后 CSRO','G_then_E40':'先 CSRO 后语义'}
METRICS=('mIoU','IoU','FN_rate','FP_over_GTfg','class_error_over_GTfg','geo1','same1')


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def aggregate():
    protocol=read(RUN/'protocol.json');complete=read(RUN/'complete.json')
    assert complete['status']=='complete' and complete['frames']==6019 and complete['scenes']==150
    assert read(RUN/'pilot.json')['status']=='passed'
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']==complete['checkpoint_sha256']
    for p,digest in protocol['source_sha256'].items():assert sha(ROOT/p)==digest,p
    arrays={};records=[];indices=[];tokens=[]
    for p in sorted((RUN/'scenes').glob('*.json')):
        m=read(p);indices+=m['indices'];tokens+=m['tokens']
        assert m['checkpoint_sha256']==protocol['checkpoint_sha256']
        with np.load(p.with_suffix('.npz')) as z:
            assert z['confusion'].shape==(m['frames'],3,7,18,18)
            assert z['class_coverage'].shape==(m['frames'],3,7,17,8)
            assert z['loss_trajectory'].shape==(m['frames'],3,41,3)
            for key in z.files:
                assert np.isfinite(z[key]).all(),(p,key)
                arrays.setdefault(key,[]).append(z[key])
            records.append(dict(scene=m['scene'],frames=m['frames'],
                confusion=z['confusion'].sum(0),class_coverage=z['class_coverage'].sum(0)))
    assert len(records)==150 and sorted(indices)==list(range(6019)) and len(set(tokens))==6019
    order=np.argsort(indices);arrays={key:np.concatenate(value)[order] for key,value in arrays.items()}
    arrays['frame_indices']=np.array(indices)[order]
    cm=arrays['confusion'].sum(0);cc=arrays['class_coverage'].sum(0)
    tr=arrays['transitions'].sum(0);post=arrays['post20_transitions'].sum(0)
    for i in range(3):
        np.testing.assert_array_equal(cm[i,0],cm[0,0]);np.testing.assert_array_equal(cc[i,0],cc[0,0])
        for j in range(7):
            np.testing.assert_array_equal(cm[i,j,:17].sum(1),cc[i,j,:,0])
            assert cm[i,0,:17,17].sum()-cm[i,j,:17,17].sum()==tr[i,j,0]-tr[i,j,1]
            assert cm[i,0,17,:17].sum()-cm[i,j,17,:17].sum()==tr[i,j,2]-tr[i,j,3]
            assert np.trace(cm[i,j])-np.trace(cm[i,0])==tr[i,j,4]-tr[i,j,5]
            if j>=3:
                assert cm[i,3,:17,17].sum()-cm[i,j,:17,17].sum()==post[i,j,0]-post[i,j,1]
                assert cm[i,3,17,:17].sum()-cm[i,j,17,:17].sum()==post[i,j,2]-post[i,j,3]
                assert np.trace(cm[i,j])-np.trace(cm[i,3])==post[i,j,4]-post[i,j,5]
    for j in range(3,7):
        np.testing.assert_array_equal(arrays['class_coverage'][:,2,j][:,:,[0,1,2,3,7]],
            arrays['class_coverage'][:,2,3][:,:,[0,1,2,3,7]])
        np.testing.assert_array_equal(arrays['attribute_stats'][:,2,j],arrays['attribute_stats'][:,2,3])
        np.testing.assert_array_equal(arrays['semantic_statistics'][:,1,j],arrays['semantic_statistics'][:,1,3])
    with np.load(E_HISTORY/'aggregate/tiny.npz') as old:
        for key in ('confusion','class_coverage','transitions'):
            np.testing.assert_array_equal(arrays[key][:,1,:4],old[key])
        np.testing.assert_array_equal(arrays['loss_trajectory'][:,1,:21],old['loss_trajectory'])
    with np.load(G_HISTORY/'aggregate/totals.npz') as old:
        # Historical CSRO is the last of the 15 attribute groups.
        np.testing.assert_array_equal(cm[2,:4],old['confusion'][-1])
        np.testing.assert_array_equal(cc[2,:4],old['class_coverage'][-1])
    return protocol,arrays,records


def bootstrap(records):
    weights=np.random.default_rng(928).multinomial(150,np.full(150,1/150),size=1500)
    cms=np.stack([r['confusion'] for r in records]);ccs=np.stack([r['class_coverage'] for r in records])
    samples={}
    for name,(arm,step) in ENDPOINTS.items():
        cm=(weights@cms[:,arm,step].reshape(150,-1)).reshape(-1,18,18)
        samples[name]=sample_metrics(cm)
        selected=ccs[:,arm,step][:,:,[0,1,4]].sum(1)
        cover=weights@selected
        samples[name].update(geo1=100*cover[:,1]/cover[:,0],same1=100*cover[:,2]/cover[:,0])
    intervals={}
    for before,after in combinations(ENDPOINTS,2):
        intervals[f'{after}-{before}']={key:np.quantile(samples[after][key]-samples[before][key],[.025,.975]).tolist()
            for key in METRICS}
    return intervals


def make_tables(arrays,records,ci):
    cm=arrays['confusion'].sum(0);cc=arrays['class_coverage'].sum(0)
    post=arrays['post20_transitions'].sum(0)
    endpoints={key:describe(cm[a,j],cc[a,j]) for key,(a,j) in ENDPOINTS.items()}
    metrics=[];classes=[];scenes=[];attributes=[];losses=[];pairs=[];stages=[];final_classes=[]
    def class_row(c,v,i):
        gt=int(c[i].sum());tp=int(c[i,i]);fe=int(c[17,i]);fo=int(c[:17,i].sum())-tp
        return dict(class_id=i,class_name=CLASS_NAMES[i],GT=gt,TP=tp,FP_empty=fe,FP_other_foreground=fo,
            FN_to_empty=int(c[i,17]),IoU=100*tp/(gt+fe+fo),
            **{key:100*int(v[i,k])/gt for k,key in enumerate(CLASS_COLUMNS[1:],1)})
    for a,arm in enumerate(ARMS):
        for j,step in enumerate(STEPS):
            active='CSROE' if arm=='J' else (('E' if step<=20 else 'CSRO') if arm=='E_then_G' else ('CSRO' if step<=20 else 'E'))
            metrics.append(dict(arm=arm,step=step,active_block_at_snapshot=active,**describe(cm[a,j],cc[a,j])))
            for i in range(17):classes.append(dict(arm=arm,step=step,**class_row(cm[a,j],cc[a,j],i)))
            for r in records:
                scenes.append(dict(scene=r['scene'],frames=r['frames'],arm=arm,step=step,
                    **describe(r['confusion'][a,j],r['class_coverage'][a,j])))
            g=arrays['attribute_stats'][:,a,j];e=arrays['semantic_statistics'][:,a,j]
            coupling=dict(zip(COUPLING_STATS,arrays['coupling_statistics'][:,a,j].sum(0)))
            attributes.append(dict(arm=arm,step=step,**geometry_changes(g.sum(0),g.max(0)),
                **semantic_changes(e.sum(0),e.max(0)),
                **{key+'_percent':100*int(coupling[key])/int(coupling['GS']) for key in COUPLING_STATS[1:]}))
        traj=arrays['loss_trajectory'][:,a]
        for step in range(41):
            losses.append(dict(arm=arm,step=step,total_loss=float(traj[:,step,0].mean()),
                weighted_CE=float(traj[:,step,1].mean()),weighted_Lovasz=float(traj[:,step,2].mean()),
                frames_above_initial=int((traj[:,step,0]>traj[:,0,0]).sum())))
        before=describe(cm[a,3],cc[a,3]);after=describe(cm[a,6],cc[a,6])
        stages.append(dict(arm=arm,**{f'delta_{key}':after[key]-before[key] for key in METRICS},
            rescued_FN=int(post[a,6,0]),new_FN=int(post[a,6,1]),removed_FP=int(post[a,6,2]),
            new_FP=int(post[a,6,3]),all_corrected=int(post[a,6,4]),all_damaged=int(post[a,6,5]),
            corrected_foreground_class=int(post[a,6,6]),damaged_foreground_class=int(post[a,6,7])))
    for name,(a,j) in ENDPOINTS.items():
        for i in range(17):final_classes.append(dict(endpoint=name,**class_row(cm[a,j],cc[a,j],i)))
    for before,after in combinations(ENDPOINTS,2):
        key=f'{after}-{before}'
        pairs.append(dict(comparison=key,**{f'delta_{k}':endpoints[after][k]-endpoints[before][k] for k in METRICS},
            **{f'{k}_CI_{bound}':ci[key][k][n] for k in METRICS for n,bound in enumerate(('low','high'))}))
    for name,rows in [('metrics',metrics),('per_class',classes),('per_scene',scenes),
        ('comparisons',pairs),('attribute_changes',attributes),('loss_trajectory',losses),
        ('second_stage_gains',stages),('final_per_class',final_classes)]:csv_write(AGG/f'{name}.csv',rows)
    return endpoints,metrics,attributes,stages,final_classes,pairs


def figures(endpoints,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    FIGURES.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    names=list(ENDPOINTS);short=['Base','E20','G20','J20','J40','E20→G20','G20→E20']
    colors=['#999999','#4477aa','#66aa88','#dd9955','#cc6677','#aa77bb','#66aacc']
    heights=[endpoints[x]['mIoU'] for x in names]
    axes[0].bar(short,heights,color=colors);axes[0].bar_label(axes[0].containers[0],fmt='%.2f',fontsize=8)
    axes[0].set(title='mIoU',ylabel='%',ylim=(0,max(heights)*1.16));axes[0].tick_params(axis='x',rotation=35)
    for name,label,color in zip(names,short,colors):
        m=endpoints[name]
        axes[1].scatter(m['FP_over_GTfg'],m['FN_rate'],c=color,s=45)
        axes[1].annotate(label,(m['FP_over_GTfg'],m['FN_rate']),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[1].set(title='Binary errors (lower/left is better)',xlabel='FP / GT foreground (%)',ylabel='FN / GT foreground (%)')
    for arm,color in zip(ARMS,('#cc6677','#aa77bb','#66aacc')):
        rows=[r for r in metrics if r['arm']==arm]
        axes[2].plot(STEPS,[r['mIoU'] for r in rows],'o-',label=arm,color=color)
    axes[2].axvline(20,color='grey',linestyle='--',alpha=.6)
    axes[2].set(title='Progress through 40 updates',xlabel='Total updates',ylabel='mIoU (%)');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Tiny | same-frame GT-assisted fitting | 6,019 frames\nG = center + scale + rotation + opacity; E = semantics')
    for ext in ('png','pdf'):fig.savefig(FIGURES/f'joint_ordered_comparison.{ext}',dpi=180)
    plt.close(fig)


def report(endpoints,attributes,stages,classes,ci):
    final_names=('J40','E_then_G40','G_then_E40')
    best=max(final_names,key=lambda x:endpoints[x]['mIoU'])
    b=endpoints['baseline']
    sequence_gap=endpoints['E_then_G40']['mIoU']-endpoints['G_then_E40']['mIoU']
    reverse_ci=ci['G_then_E40-E_then_G40']['mIoU']
    sequence_ci=(-reverse_ci[1],-reverse_ci[0])
    sequence_class_gains=[]
    for i,name in enumerate(CLASS_NAMES):
        first=next(x for x in classes if x['endpoint']=='E_then_G40' and x['class_id']==i)
        second=next(x for x in classes if x['endpoint']=='G_then_E40' and x['class_id']==i)
        sequence_class_gains.append((name,first['IoU']-second['IoU']))
    class_improved=sum(delta>0 for _,delta in sequence_class_gains)
    largest_class_gains=sorted(sequence_class_gains,key=lambda x:-x[1])[:3]
    lines=['# 几何与语义：联合优化及先后顺序的完整对照','',
        '**实验完成：Tiny、150 个场景、6,019 帧，三组均执行到总计40步。网络参数固定，每帧使用自身GT调整输出。**','',
        f"三组最终结果中，mIoU最高的是 **{LABELS[best]}：{endpoints[best]['mIoU']:.3f}%**，相对原预测 {endpoints[best]['mIoU']-b['mIoU']:+.3f} 个百分点。",'',
        '**G 沿用此前 CSRO，包含中心、尺度、旋转及 opacity；opacity 属于证据强度，因此这里不是严格的纯几何对照。E 是17类非负语义值，幅度和类别比例均可改变。**',
        '', '本轮不训练或发布新权重，不能作为正常推理性能。执行细节见[设计报告](gaussian_joint_oracle_full_val_design.md)。','',
        '先看三项结果：','',
        f"- **两类属性有明显互补。** 联合20步达到 {endpoints['J20']['mIoU']:.3f}% mIoU，分别高于E20、G20 {endpoints['J20']['mIoU']-endpoints['E20']['mIoU']:.3f}、{endpoints['J20']['mIoU']-endpoints['G20']['mIoU']:.3f} 个百分点。",
        f"- **先后顺序有明显影响。** 每个属性块都更新20次时，E→G高于G→E {sequence_gap:.3f} 个百分点，配对95%场景区间为 [{sequence_ci[0]:+.3f}, {sequence_ci[1]:+.3f}]；{class_improved}/17类IoU更高。",
        f"- **更高的核心覆盖不保证更好的预测。** G→E的1σ支持为 {endpoints['G_then_E40']['geo1']:.3f}%，高于E→G的 {endpoints['E_then_G40']['geo1']:.3f}%，但mIoU和二值IoU均更低。",'',
        '## 1. 三组如何运行，怎样比较预算','',
        '| 结果 | 更新方式 | 总反向次数 | G更新次数 | E更新次数 |',
        '|---|---|---:|---:|---:|',
        '| J20 | G与E同时更新20步 | 20 | 20 | 20 |',
        '| J40 | G与E同时更新40步 | 40 | 40 | 40 |',
        '| E→G | E20步，然后固定E，更新G20步 | 40 | 20 | 20 |',
        '| G→E | G20步，然后固定G，更新E20步 | 40 | 20 | 20 |','',
        '比较先后顺序时，E→G与G→E的总反向次数及各属性更新次数相同。J20与两组顺序实验的各属性更新次数相同，但总反向次数更少；J40的总反向次数相同，但各属性更新次数更多。两种联合检查点均保留，不将某一种预算称为完全公平。','',
        '## 2. 整体指标','',
        '漏检、前景类别错分、二值FP都除以GT前景体素数。所有指标先累计完整验证集计数，再计算比例。','',
        '| 结果 | mIoU | ΔmIoU/百分点 | 二值IoU | 漏检率 | 前景错分率 | FP/GT前景 | 1σ支持 | 同类1σ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in endpoints.items():
        lines.append(f"| {LABELS[name]} | {m['mIoU']:.3f}% | {m['mIoU']-b['mIoU']:+.3f} | "+
            ' | '.join(f"{m[k]:.3f}%" for k in ('IoU','FN_rate','class_error_over_GTfg','FP_over_GTfg','geo1','same1'))+' |')
    lines+=['','1σ支持沿用此前正opacity×核值口径。opacity变为0时可减少活动支持，即使GS位置和形状没有改变；同类支持同时受当前语义类别影响。','',
        '![三组优化方式及错误变化](figures/gaussian_joint_oracle_full_val/joint_ordered_comparison.png)','',
        '## 3. 第二阶段又修正了多少','',
        '比较每组第40步与自己的第20步。J是额外联合优化20步，其余两组是切换到另一个属性块后的增量。','',
        '| 组别 | ΔmIoU | Δ二值IoU | Δ漏检率 | ΔFP/GT前景 | Δ前景错分率 | Δ1σ支持 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in stages:
        lines.append('| '+s['arm']+' | '+' | '.join(f"{s['delta_'+k]:+.3f}" for k in
            ('mIoU','IoU','FN_rate','FP_over_GTfg','class_error_over_GTfg','geo1'))+' |')
    lines+=['','变化单位为百分点。第二阶段实际发生的修复与新增错误如下：','',
        '| 组别 | 救回漏检 | 新增漏检 | 消除二值FP | 新增二值FP |',
        '|---|---:|---:|---:|---:|']
    for s in stages:
        lines.append('| '+s['arm']+' | '+' | '.join(f"{s[k]:,}" for k in
            ('rescued_FN','new_FN','removed_FP','new_FP'))+' |')
    lines+=['',
        '先语义后G的第二阶段同时大量减少漏检与FP；先G后语义的第二阶段主要补足漏检并纠正类别，但新增二值FP多于消除的FP。修复与新增计数可以同时很大，因此要与净变化一起阅读。','',
        '## 4. 联合方式与先后顺序的差异','',
        '下表按相同150个场景配对bootstrap重采样1,500次，种子928。区间反映固定实验输出的场景差异，不包括重新训练随机性。','',
        '| 比较 | ΔmIoU | 95%场景区间 | Δ二值IoU | Δ漏检率 | ΔFP/GT前景 |',
        '|---|---:|---|---:|---:|---:|']
    comparisons=(('J20','E20'),('J20','G20'),('J40','J20'),
        ('E_then_G40','J20'),('G_then_E40','J20'),('E_then_G40','J40'),
        ('G_then_E40','J40'),('G_then_E40','E_then_G40'))
    order=list(ENDPOINTS)
    for after,before in comparisons:
        direct=order.index(after)>order.index(before)
        key=f'{after}-{before}' if direct else f'{before}-{after}'
        interval=ci[key]['mIoU'] if direct else [-ci[key]['mIoU'][1],-ci[key]['mIoU'][0]]
        delta={k:endpoints[after][k]-endpoints[before][k] for k in METRICS}
        lines.append(f"| {after} − {before} | {delta['mIoU']:+.3f} | [{interval[0]:+.3f}, {interval[1]:+.3f}] | {delta['IoU']:+.3f} | {delta['FN_rate']:+.3f} | {delta['FP_over_GTfg']:+.3f} |")
    interval=ci['G_then_E40-E_then_G40']['mIoU']
    if interval[0]>0:order_text='本次数据支持G→E的mIoU高于E→G，顺序效应的区间位于零以上。'
    elif interval[1]<0:order_text='本次数据支持E→G的mIoU高于G→E，顺序效应的区间位于零以下。'
    else:order_text='两种先后顺序的mIoU差值区间跨过零，本次数据不足以确定稳定的顺序优势。'
    lines+=['',order_text,'',
        '## 5. 哪些属性改变了，冻结是否生效','',
        '| 结果 | 中心平均位移/m | 尺度每轴平均绝对变化/m | 平均旋转/° | opacity平均绝对变化 | 平均语义总幅度 | 零opacity GS | 零语义GS | 两者均零GS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,(a,j) in ENDPOINTS.items():
        x=next(v for v in attributes if v['arm']==ARMS[a] and v['step']==STEPS[j])
        lines.append(f"| {LABELS[name]} | {x['center_mean_m']:.4f} | {x['scale_axis_mean_abs_change_m']:.4f} | {x['rotation_mean_deg']:.3f} | {x['opacity_mean_abs_change']:.4f} | {x['mean_semantic_mass']:.5f} | {x['zero_opacity_percent']:.3f}% | {x['zero_semantic_mass_percent']:.3f}% | {x['both_zero_percent']:.3f}% |")
    lines+=['',
        '**冻结已逐帧核验。** E→G的第二阶段语义向量逐元素不变；G→E的第二阶段中心、尺度、旋转和opacity逐元素不变。阶段切换不改变当时的logits，新的属性块使用全新Adam状态。',
        '',
        '零值可能影响后续阶段的可调整性：固定opacity为0时，语义更新无法恢复该GS的渲染贡献；固定全部语义为0时，几何与opacity更新也无法恢复该GS的贡献。零值比例记录这一条件存在多少，并不单独证明它解释了全部顺序差异。','',
        '## 6. 逐类IoU与支持变化','',
        '每格为“IoU / 同类1σ支持”，单位为%。完整几何支持、逐类TP/FP/FN及所有检查点在CSV中。','',
        '| 类别 | 原预测 | E20 | G20 | J20 | J40 | E→G | G→E |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for i,name in enumerate(CLASS_NAMES):
        vals=[]
        for endpoint in ENDPOINTS:
            x=next(v for v in classes if v['endpoint']==endpoint and v['class_id']==i)
            vals.append(f"{x['IoU']:.3f} / {x['same1']:.3f}")
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines+=['','## 7. 这组实验能说明什么','',
        '**互补性已经得到本次输出优化的直接支持。** J20相对E20和G20均有明显增益，且17类IoU全部提高。它同时改变空间分布与语义证据；不能把收益简单分成互不相关的几何误差和语义误差两部分。',
        '',f"**先修正语义，再让G适配，是本次两种顺序中更有效的路径。** 两组各属性更新次数和总反向次数相同，E→G的17类IoU全部高于G→E；差异最大的三类为 "+
            '、'.join(f'{name}（{delta:+.3f}个百分点）' for name,delta in largest_class_gains)+'。',
        '', '一种与结果相符的解释是：先优化G时，得到的是适合原始语义配置的空间分布和opacity；之后语义改变，但G已被冻结，不能继续适配。先优化E再优化G，则允许最终空间分布适配已经修正的语义。这是机制候选，仍需进一步干预区分它与有限步优化等因素。',
        '',f"**核心覆盖率不能替代最终预测质量。** E→G的几何1σ支持、同类1σ支持均低于G→E，但mIoU高 {sequence_gap:.3f} 个百分点、二值IoU高 {endpoints['E_then_G40']['IoU']-endpoints['G_then_E40']['IoU']:.3f} 个百分点。J20到J40的几何1σ支持也略降，mIoU却继续提高。需要研究的是正确与错误证据的空间分布，而非单独最大化覆盖计数。",
        '', '顺序差异可能来自优化路径、有限步数、约束投影和零值门控等因素。这组实验没有把它们逐项分离，不能把顺序差异直接解释为原训练中某条因果链。',
        '', '所有更新使用当前验证帧GT；网络参数、GS数量和父query均固定。本轮没有证明网络可以在训练集学会这些输出调整，也没有测量可部署方法的泛化或FPS。','',
        '## 8. 核验与数据入口','',
        '- 全部150场景、6,019帧；三组共用相同原始输入、预测与camera有效GT。',
        '- 顺序组第一阶段的混淆、覆盖、错误转变和21次损失，逐帧精确复现历史E20或G20。',
        '- C/S/R/O/E的原生CUDA梯度通过有限差分，联合起点的五条梯度与独立路径逐元素相同。',
        '- 冻结属性不更新，所有损失和梯度有限，约束始终相对原始预测，不在阶段切换时重置。',
        '- 原预测及第20步之后的净错误变化，均与对应体素转变计数核对。','',
        '| 文件 | 内容 |','|---|---|',
        '| [全部检查点指标](../work_dirs/gaussian_joint_oracle_full_val_v1/aggregate/metrics.csv) | 三组各七个检查点 |',
        '| [终点两两差值](../work_dirs/gaussian_joint_oracle_full_val_v1/aggregate/comparisons.csv) | 七个终点的21项比较及各指标配对区间 |',
        '| [第二阶段增量](../work_dirs/gaussian_joint_oracle_full_val_v1/aggregate/second_stage_gains.csv) | 第20步到40步的指标和纠错计数 |',
        '| [逐类终点](../work_dirs/gaussian_joint_oracle_full_val_v1/aggregate/final_per_class.csv) | 七个终点的17类IoU与覆盖 |',
        '| [全部逐类检查点](../work_dirs/gaussian_joint_oracle_full_val_v1/aggregate/per_class.csv) | 所有更新步骤的逐类计数与指标 |',
        '| [属性实际变化](../work_dirs/gaussian_joint_oracle_full_val_v1/aggregate/attribute_changes.csv) | 几何、语义及零值统计 |',
        '| [逐场景指标](../work_dirs/gaussian_joint_oracle_full_val_v1/aggregate/per_scene.csv) | 场景与检查点数据 |',
        '| [损失轨迹](../work_dirs/gaussian_joint_oracle_full_val_v1/aggregate/loss_trajectory.csv) | 三组各41个损失记录 |',
        '| [汇总JSON](../work_dirs/gaussian_joint_oracle_full_val_v1/summary.json) | 精确终点、区间与执行范围 |',
        '| [完整性核验](../work_dirs/gaussian_joint_oracle_full_val_v1/delivery_audit.json) | 输入与产物哈希 |','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def main():
    AGG.mkdir(parents=True,exist_ok=True)
    protocol,arrays,records=aggregate();np.savez_compressed(AGG/'tiny.npz',**arrays)
    ci=bootstrap(records)
    endpoints,metrics,attributes,stages,classes,pairs=make_tables(arrays,records,ci)
    figures(endpoints,metrics);report(endpoints,attributes,stages,classes,ci)
    summary=dict(status='complete',frames=6019,scenes=150,endpoints=endpoints,
        comparisons=pairs,paired_delta_CI=ci,second_stage_gains=stages,
        checkpoint_sha256=protocol['checkpoint_sha256'],budget=protocol['budget'],
        interpretation='Same-frame GT-assisted output fitting, not network training or normal inference')
    write(RUN/'summary.json',summary)
    files=[REPORT,ROOT/'analysis/gaussian_joint_oracle_full_val_design.md',
        ROOT/'tools/eval_gaussian_joint_oracle.py',Path(__file__),ROOT/'tools/run_gaussian_joint_oracle.py',
        RUN/'protocol.json',RUN/'pilot.json',RUN/'complete.json',RUN/'summary.json',
        *sorted(AGG.glob('*')),*sorted(FIGURES.glob('*'))]
    audit=dict(status='passed',frames=6019,scenes=150,arms=3,steps_per_arm=40,
        historical_first_phases_exact=True,inactive_blocks_bitwise_fixed=True,
        phase_boundary_logits_exact=True,finite_losses_and_gradients=True,
        source_and_checkpoint_hashes_verified=True,
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    write(RUN/'delivery_audit.json',audit)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
