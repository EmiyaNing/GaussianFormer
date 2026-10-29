"""Scene-paired endpoint attribution, operating curves and readable report."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage3 import RUN,read,write_json,file_sha256
from tools.eval_empty_mechanism_stage3 import MODELS,EDGES,NB
from tools.eval_empty_decision_optimization import metrics

NAMES={'B0':'原训练目标','S':'阻断 Set 负类路径','E':'阻断 CE empty 路径',
       'SE':'同时阻断两条路径','original5':'历史原模型 epoch 5','no_final_set5':'历史全程删除末阶段 Set 分类'}


def ratio(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.zeros(a.shape,dtype=float),where=b>0)


def interval(v):
    return dict(value=float(v[0]),ci95=np.percentile(v[1:],[2.5,97.5]).tolist())


def scene_array(records,metadata,scenes,extractor):
    lookup={s:i for i,s in enumerate(scenes)};shape=np.asarray(extractor(records[0])).shape
    result=np.zeros((len(scenes),)+shape,dtype=float)
    assert len(records)==len(metadata)==1024
    for row,meta in zip(records,metadata):
        assert row['index']==meta['index']
        result[lookup[meta['scene']]]+=extractor(row)
    return result


def recomputed_matched_rescue(curves,w):
    fp=(w@curves['empty_hist'].astype(float))
    fp=np.cumsum(fp[:,::-1],axis=-1)[:,::-1][:,1:]
    target=w@curves['baseline_fp']
    admissible=fp<=target[:,None]
    if not admissible.any(1).all():raise RuntimeError('Some scene bootstrap curves cannot reach FPR budget')
    indices=np.argmax(admissible,axis=1)
    del fp,admissible
    rescue=w@curves['rescue_hist'].astype(float)
    rescue=np.cumsum(rescue[:,::-1],axis=-1)[:,::-1][:,1:]
    rescued=rescue[np.arange(len(w)),indices]
    return 100*ratio(rescued,w@curves['cohort_size']),EDGES[indices]


def summarize():
    metadata=read(RUN/'evaluation/B0/validation_frames.json')
    scenes=sorted({r['scene'] for r in metadata});ns=len(scenes)
    rng=np.random.default_rng(928)
    w=np.vstack([np.ones(ns),rng.multinomial(ns,np.full(ns,1/ns),size=1000)])
    vectors={};result=dict(status='complete',scenes=ns,frames=1024,bootstrap=1000,models={},contrasts={},dominance={})
    for name in MODELS:
        folder=RUN/'evaluation'/name;data=read(folder/'statistics.json')
        records=data['records'];cohorts=data['cohort_records']
        np.testing.assert_equal(read(folder/'validation_frames.json'),metadata)
        scene_rank=scene_array(records,metadata,scenes,lambda r:r['rank_correct'])
        scene_group=scene_array(cohorts,metadata,scenes,lambda r:r['cohort'])
        scene_baseline_correct=scene_array(cohorts,metadata,scenes,lambda r:r['baseline_correct'])
        vectors[name]={};record=dict(label=NAMES[name],modes={},
            matched_threshold=data['threshold_matched'],target_empty_fp=data['target_fp'],achieved_empty_fp=data['achieved_fp'],
            calibration=read(folder/'selection.json'))
        for mode in ['native','calibrated','matched']:
            sc=scene_array(records,metadata,scenes,lambda r:np.asarray(r['modes'][mode]['counts']))
            c=(w@sc.reshape(ns,-1)).reshape(-1,18,18)
            m=metrics(c);occ=c[:,:17,:].sum((1,2));tp=np.diagonal(c[:,:17,:17],axis1=-2,axis2=-1).sum(-1)
            v={k:m[key] for k,key in [('miou','mIoU'),('iou','IoU'),('fnr','occupied_FNR'),('fpr','empty_FPR')]}
            v['rank_accuracy']=100*ratio(w@scene_rank,occ)
            v['final_accuracy']=100*ratio(tp,occ)
            rc=scene_array(records,metadata,scenes,lambda r:r['modes'][mode]['recoverable'])
            v['recoverable_rate']=100*ratio(w@rc,occ)
            fields=['rescued','remaining_empty','wrong_foreground','rank_lost','newly_empty','newly_wrong_foreground']
            rawgroups={}
            for field in fields:
                a=scene_array(cohorts,metadata,scenes,lambda r:r['modes'][mode][field]);rawgroups[field]=a
                denominator=scene_baseline_correct if field.startswith('newly_') else scene_group
                v[field+'_fraction']=100*ratio(w@a,w@denominator)
            np.testing.assert_allclose(sum(w@rawgroups[k] for k in fields[:3]),w@scene_group)
            vectors[name][mode]=v
            record['modes'][mode]=dict(metrics={k:interval(x) for k,x in v.items()},
                counts=sc.sum(0).astype(np.int64).tolist(),
                fixed_cohort_size=int(scene_group.sum()),
                cohort_counts={k:int(a.sum()) for k,a in rawgroups.items()})
        curves=np.load(folder/'operating_curves.npz')
        np.testing.assert_array_equal(curves['scenes'],scenes)
        if name=='B0':
            rescued=np.zeros(len(w));thresholds=np.zeros(len(w))
        else:rescued,thresholds=recomputed_matched_rescue(curves,w)
        np.testing.assert_allclose(rescued[0],vectors[name]['matched']['rescued_fraction'][0],atol=1e-10)
        vectors[name]['matched']['rescued_reselected_fraction']=rescued
        record['modes']['matched']['metrics']['rescued_reselected_fraction']=interval(rescued)
        record['matched_threshold_bootstrap']=interval(thresholds)
        result['models'][name]=record
    for name in MODELS:
        result['contrasts'][name]={}
        for mode in ['native','calibrated','matched']:
            ref=vectors['B0'][mode]
            result['contrasts'][name][mode]={k:interval(vectors[name][mode][k]-ref[k]) for k in
                ['miou','iou','fnr','fpr','rank_accuracy','final_accuracy','recoverable_rate','rescued_fraction']}
    for name in ['S','E','SE']:
        result['dominance'][name]={}
        for mode in ['native','matched']:
            rescue=vectors[name][mode]['rescued_fraction' if mode=='native' else 'rescued_reselected_fraction']
            rank=vectors[name][mode]['rank_accuracy']-vectors['B0'][mode]['rank_accuracy']
            accuracy=vectors[name][mode]['final_accuracy']-vectors['B0'][mode]['final_accuracy']
            passed=np.percentile(rescue[1:],2.5)>50 and np.percentile(rank[1:],2.5)>=-.5 and accuracy[0]>=0
            result['dominance'][name][mode]=dict(passed=bool(passed),rescue_fraction=interval(rescue),
                rank_accuracy_delta=interval(rank),final_accuracy_delta=interval(accuracy))
    result['interactions']={}
    for mode in ['native','calibrated','matched']:
        result['interactions'][mode]={k:interval(vectors['SE'][mode][k]-vectors['S'][mode][k]-vectors['E'][mode][k]+vectors['B0'][mode][k])
                                      for k in ['miou','fnr','fpr','rescued_fraction']}
    audits={};p=read(RUN/'protocol.json')
    expected_steps=p.get('total_optimizer_steps',1024)
    expected_frames=p.get('total_frame_visits_per_model',4096)
    for name in ['B0','S','E','SE']:
        r=read(RUN/'train'/name/'complete.json');assert r['steps']==expected_steps and r['frames']==expected_frames
        totals={}
        for source in ['S','E']:
            selected=[t['sources'][source] for t in r['trace'] if source in t['sources']]
            if selected:
                original=sum(a['suppressive_derivative'] for a in selected)
                removed=sum(a['removed_derivative'] for a in selected)
                fraction=removed/original
                assert abs(fraction-1)<1e-8
                totals[source]=dict(steps=len(selected),removed_derivative_fraction=fraction,
                    mean_removed_squared_norm_fraction=float(np.mean([a['removed_fraction'] for a in selected])))
        audits[name]=dict(steps=r['steps'],frames=r['frames'],sources=totals,elapsed_seconds=r['elapsed_seconds'])
    result['training_audit']=audits
    p=read(RUN/'protocol.json')
    for path,sha in p['preexisting_results_sha256'].items():assert file_sha256(ROOT/path)==sha
    assert file_sha256(ROOT/p['checkpoint']['path'])==p['checkpoint']['sha256']
    result['original_artifacts_unchanged']=True
    write_json(RUN/'summary.json',result)
    rows=[]
    for name,r in result['models'].items():
        for mode,m in r['modes'].items():
            row=dict(condition=name,mode=mode)
            for k,v in m['metrics'].items():
                row[k]=v['value'];row[k+'_lo']=v['ci95'][0];row[k+'_hi']=v['ci95'][1]
            rows.append(row)
    with (RUN/'all_conditions.csv').open('w') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return result


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def report(s):
    def value(name,mode,key):return s['models'][name]['modes'][mode]['metrics'][key]['value']
    def ci(v):return f'{v["value"]:.2f} [{v["ci95"][0]:.2f}, {v["ci95"][1]:.2f}]'
    native=table(['条件','mIoU（%）','前景首选正确率（%）','可恢复漏检率（%）','漏检率（%）','误占用率（%）'],
        [[NAMES[n]]+[f'{value(n,"native",k):.4f}' for k in ['miou','rank_accuracy','recoverable_rate','fnr','fpr']]
         for n in ['B0','S','E','SE']])
    rescues=table(['条件','原生正确救回比例 [95% 区间]','同误占用正确救回比例 [95% 区间]'],
        [[NAMES[n],ci(s['dominance'][n]['native']['rescue_fraction']),ci(s['dominance'][n]['matched']['rescue_fraction'])]
         for n in ['S','E','SE']])
    calibration=table(['条件','原生 mIoU（%）','校准后 mIoU（%）','校准后相对 B0，百分点 [95% 区间]'],
        [[NAMES[n],f'{value(n,"native","miou"):.4f}',f'{value(n,"calibrated","miou"):.4f}',
          ci(s['contrasts'][n]['calibrated']['miou'])] for n in MODELS])
    matched=table(['条件','margin 阈值','实际误占用率（%）','漏检率（%）','mIoU（%）'],
        [[NAMES[n],f'{s["models"][n]["matched_threshold"]:.5f}']+
         [f'{value(n,"matched",k):.4f}' for k in ['fpr','fnr','miou']] for n in MODELS])
    consequences=[];harms=[]
    for name in ['S','E','SE']:
        for mode in ['native','matched']:
            c=s['models'][name]['modes'][mode]['cohort_counts']
            prefix=[NAMES[name],'原生' if mode=='native' else '同误占用']
            consequences.append(prefix+[c[k] for k in ['rescued','remaining_empty','wrong_foreground']])
            harms.append(prefix+[c[k] for k in ['rank_lost','newly_empty','newly_wrong_foreground']])
    consequences=table(['条件','决策','正确救回','仍 empty','错前景类'],consequences)
    harms=table(['条件','决策','漏检组首选丢失','原正确→empty','原正确→错类'],harms)
    base=s['models']['B0']['modes']['native']['fixed_cohort_size']
    all_misses=int(np.asarray(s['models']['B0']['modes']['native']['counts'])[:17,17].sum())
    gate=s['dominance']['SE'];npass=gate['native']['passed'];mpass=gate['matched']['passed']
    if mpass:
        conclusion='联合阻断通过了相同误占用率下的操作性主导判据：在本轮完整最后一个 epoch 的条件下，这两条直接路径具有较大的可干预作用。整个训练历史与跨模型的主导成因仍需独立训练复核。'
    elif npass:
        conclusion='联合阻断在原生决策下通过主导判据，但相同误占用率下未通过。当前更支持其显著改变前景与 empty 的决策工作点，尚不支持其主导了相同误占用水平下的残余漏检。'
    else:
        conclusion='在最后一个 epoch 的干预窗口中，联合阻断未通过预先定义的主导判据。两条直接路径存在，并不足以保证移除后能正确救回大部分基线可恢复漏检；本轮没有建立其在这一窗口中具有主导作用的证据。'
    p=read(RUN/'evaluation_protocol.json')
    rescue_native=ci(gate['native']['rescue_fraction'])
    rescue_matched=ci(gate['matched']['rescue_fraction'])
    med=read(RUN/'mediation/summary.json')
    factor_names={'geometry':'空间参数与 opacity','amplitude':'Gaussian 语义总幅度',
                  'composition':'Gaussian 类别组成','empty':'empty 强度与全局 bias'}
    factors=table(['输出因子','对 mIoU 差值的分摊 [95% 区间]','对正确救回比例的分摊 [95% 区间]'],
        [[label,ci(med['effects']['miou'][key]),ci(med['effects']['rescued_fraction'][key])] for key,label in factor_names.items()])
    endpoint_rows=[]
    for name in ['B0','S','E','SE']:
        records=read(RUN/'train'/name/'step_1024.json')['records']
        endpoint_rows.append([NAMES[name]]+[f'{np.mean([row[key] for row in records]):.4f}'
            for key in ['amplitude','empty_scalar','bias']])
    endpoint_states=table(['条件','平均 Gaussian 语义总幅度','empty_scalar','全局 empty bias'],endpoint_rows)
    max_render_error=max(x['max_margin_error'] for row in med['parity'] for x in row)
    render_disagreements=sum(x['prediction_disagreements'] for row in med['parity'] for x in row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11})
    assets=ROOT/'analysis/empty_mechanism/stage3';assets.mkdir(parents=True, exist_ok=True)
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for offset,mode,label,color in [(-.17,'native','原生决策','#4c78a8'),(.17,'matched','相同误占用率','#e09547')]:
        entries=[s['dominance'][n][mode]['rescue_fraction'] for n in ['S','E','SE']]
        y=np.array([v['value'] for v in entries]);limits=np.array([v['ci95'] for v in entries]).T
        ax.bar(np.arange(3)+offset,y,.32,label=label,color=color)
        ax.errorbar(np.arange(3)+offset,y,yerr=np.maximum(np.vstack([y-limits[0],limits[1]-y]),0),fmt='none',ecolor='#333',capsize=4)
    ax.axhline(50,color='#777',ls='--',lw=1,label='超过一半的救回判据')
    ax.set_xticks(np.arange(3),['只阻断 Set 负类','只阻断 CE empty','同时阻断'])
    ax.set_ylabel('基线可恢复漏检中，正确救回的比例（%）')
    ax.set_ylim(0,max(60,max(v['ci95'][1] for n in ['S','E','SE'] for v in
                           [s['dominance'][n]['native']['rescue_fraction'],s['dominance'][n]['matched']['rescue_fraction']])+10))
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(assets/'rescue_fraction.png',dpi=180,bbox_inches='tight');plt.close(fig)
    text=f'''# 最终 empty 漏检的成因：第三阶段实验结果

日期：2026-10-19。状态：**四组完整末期训练、六个模型的完整验证与配对统计均已完成。** [实验设计](empty_mechanism_stage3_design.md) · [全部派生数值](../../../work_dirs/empty_mechanism_stage3/all_conditions.csv)。

## 1. 本轮结论

**{conclusion}**

联合阻断在原生决策下，正确救回基线漏检组的 **{rescue_native}%**；匹配误占用后为 **{rescue_matched}%**。方括号为 95% 场景区间。

本轮量化的是：从相同 epoch 4 权重和优化器状态出发，完整训练最后一个 epoch 后，原训练目标的可恢复漏检有多少被干预模型正确救回。它测量末期干预的因果效果，不是整个五个 epoch 的历史归因百分比。

原生决策保留模型已有的可学习全局 empty bias。相同误占用比较进一步约束报前景的代价；同等类别校准则给所有模型相同的静态阈值优化机会。

## 2. 训练结束后，整体表现如何

四组均训练完整 4096 帧，执行 1024 次优化器更新；全部用 1024 个验证帧评价。B0 为原训练目标，S 只阻断 Set 负类的直接幅度抑制，E 只阻断 CE empty 的直接幅度抑制，SE 同时阻断。各项监督的前向损失值、Set 几何项及其他来源梯度均保留。

{native}

前景首选正确率忽略 empty，只检查 17 个前景类别的排序。可恢复漏检率的分母是全部真实前景，计数对象是“前景首选正确、最终却为空”。该数量下降也可能来自类别首选丢失，因此不能单独作为机制改善的证据。

## 3. 原来的漏检，有多少被真正救回

固定基线组共有 **{base} 个体素**：它们在 B0 最终模型中真实为前景、前景首选正确，但最终输给 empty。所有干预都评价这同一批体素，只有最终语义预测正确才算救回。

B0 的全部前景漏检为 **{all_misses} 个**，固定组占其中 **{100*base/all_misses:.2f}%**。下表的救回比例以固定组为分母，不能直接写成对同样比例的全部漏检的解释。

以下是正确救回比例，单位为百分比；方括号为配对场景 95% 区间。

{rescues}

![固定基线漏检组：原生与相同误占用率下的正确救回比例](rescue_fraction.png)

相同误占用率的主要救回区间，在每次场景重采样中重新计算 B0 误占用预算及干预模型的曲线阈值。没有将新增的错误前景预测计作救回。

“正确救回 + 仍 empty + 错前景类”恰好等于固定漏检组大小。新增漏检和新增错类另外报告，不混入救回分子；完整计数见附录。主导判据还要求前景类别排序没有明显劣化、最终前景语义正确率没有下降，具体检查保存在机器汇总中。

## 4. 相同误占用水平下，还剩多少改善

下表采用完整验证集上的统一 margin 阈值曲线，选择不超过 B0 原生误占用数量的最近网格点。它是描述性 ROC 工作点评价；阈值使用了验证标签，不作为可部署校准参数。

{matched}

若实际误占用率略低于目标，是离散网格造成的保守匹配。曲线阈值会改变前景与 empty 的竞争，但不会改变前景类别排序。除主要救回比例外，其他匹配工作点的区间条件于整体样本选定的阈值，不能当作完整阈值重选的不确定性。

## 5. 同等类别校准后，方法是否有额外收益

每个模型在相同的 **{len(p['fitting_train_indices'])} 个训练场景帧**上拟合／调参：每个训练场景取一个确定帧，{len(p['fit_scenes'])} 个场景用于拟合类别阈值，{len(p['tune_scenes'])} 个场景用于选共同平移量。验证标签不参与这一校准。

下表最后两行是已存在的历史模型，单独用于参考。它们没有参与本轮四组配对末期训练。“历史全程删除末阶段 Set 分类”删去了整项末阶段分类监督，不能等同于本轮只阻断负类幅度分量。

{calibration}

原生、同误占用和类别校准分别回答不同问题：是否改变原决策、是否在控制误占用后仍能救回、是否优于训练后静态校准。机制作用与方法收益不能互相替代。

## 6. 能否认定为主导成因

判据在处理组正式结果产生前固定：救回比例区间下界超过 50%；前景首选正确率差值区间下界不低于 −0.5 个百分点；最终前景语义正确率点估计不下降。若要支持超出工作点移动的主导作用，还需在同误占用比较中通过。

联合阻断的原生判定为 **{'通过' if npass else '未通过'}**，同误占用判定为 **{'通过' if mpass else '未通过'}**。这里的 50% 是操作性判据，不是普遍定律。完整连续效果及单独路径、交互区间保存在[机器汇总](../../../work_dirs/empty_mechanism_stage3/summary.json)。

末期阻断不能追溯消除前四个 epoch 已形成的参数状态。没有通过，只能限制该直接路径在本轮干预窗口中的解释力，不能证明它在训练早期或其他方法中不重要。通过也不意味着整个训练历史的残余错误都由它造成。

## 附录 A：救回与新增错误的体素数量

先看固定漏检组在干预后的三种互斥结果：

{consequences}

再看类别排序损失和新增错误。“原正确”指 B0 最终语义预测正确的真实前景；“漏检组首选丢失”在固定漏检组内统计，与上表“仍 empty”等列可能重叠。

{harms}

## 附录 B：实现、校验与复现

本轮主训练共 **4 × 4096 = 16384 个训练帧次、4096 次优化器更新**。中间评价在第 64、256、1024 步进行；完整终点评价为六模型各 1024 帧。

Set 干预覆盖六个分类阶段；CE 干预覆盖实际 occupancy-supervised 阶段。记录表明指定来源的正幅度导数分量被完整去除；其他来源与输出空间几何／empty 直接梯度保留。该检验约束直接路径，不保证共享参数更新后的总幅度恒定。

梯度测试包括负类 focal 的实际系数与分母、只去除抑制方向、保持几何与 empty 直接梯度；试运行通过后才启动正式训练。原训练 seed 为 928，本轮使用相同起点和配对重建的数据流，不是不同训练 seed 的重复。

场景不确定性使用 150 个验证场景、1000 次配对重采样；没有做多重比较校正。历史原权重与前两阶段结果的校验值保持一致。

实现：[训练与路径干预](../../../tools/empty_mechanism_stage3.py) · [完整验证与校准](../../../tools/eval_empty_mechanism_stage3.py) · [统计汇总](../../../tools/report_empty_mechanism_stage3.py)。协议：[训练协议](../../../work_dirs/empty_mechanism_stage3/protocol.json) · [执行补充](../../../work_dirs/empty_mechanism_stage3/execution_addendum.json) · [评价协议](../../../work_dirs/empty_mechanism_stage3/evaluation_protocol.json)。

## 附录 C：输出因子交叉替换——检查训练中的补偿

先看固定 32 个验证场景中实际记录的终点状态。语义总幅度是每个 Gaussian 的 17 类 softplus 值之和，再对 Gaussian 和场景取平均；它不是体素前景正确率或概率。empty_scalar 是全局 empty Gaussian 的语义强度，empty bias 是最终从 empty 分数减去的可学习全局值。

{endpoint_states}

幅度增加并不自动保证正确前景体素获胜：空间权重、opacity、类别组成与 empty 参数仍会共同改变渲染后的竞争。因此下面继续检查这些输出因素的补偿。

额外在固定 32 个验证场景中，将 B0 与 SE 最终输出的空间参数、语义幅度、类别组成和 empty 参数交叉组合，完成 16 种替换。再对所有替换顺序取对称分摊，检查各输出因子如何影响最终差异。

下表的参照是同一批 32 场景中的 SE − B0，不是完整 1024 帧；mIoU 列单位为百分点，救回列单位为基线漏检组比例的百分点。

{factors}

这些行分摊的是两个模型的输出差异，不是训练损失的自然因果贡献。空间、幅度和 empty 的变化可能相互补偿；交换不同模型相同 query 槽位的输出，也不等于一个可直接训练的中间模型。因此，这一探索性分析不用于替代主救回实验或主导判据。

重渲染终点的最大 margin 数值差为 {max_render_error:.2e}，合计类别预测差异为 {render_disagreements} 个体素。完整组合与校验见[输出因子分解](../../../work_dirs/empty_mechanism_stage3/mediation/summary.json)。从训练起点开始的更具体共享渲染层对照，另见[全程实验设计](../history/empty_mechanism_history_design.md)。
'''
    (ROOT/'analysis/empty_mechanism/stage3/empty_mechanism_stage3_results.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-only',action='store_true')
    a=parser.parse_args();s=read(RUN/'summary.json') if a.report_only else summarize();report(s)
    print(ROOT/'analysis/empty_mechanism/stage3/empty_mechanism_stage3_results.md')


if __name__=='__main__':main()
