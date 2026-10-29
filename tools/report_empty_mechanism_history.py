"""Full-history causal contribution report; method utility is reported separately."""
from pathlib import Path
from datetime import datetime
import json
import csv
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_history import RUN,MODELS,read,write_json,file_sha256,setup_evaluator


def main():
    setup_evaluator()
    import tools.report_empty_mechanism_stage3 as r
    r.RUN=RUN;r.MODELS=MODELS
    s=r.summarize()
    s['gross_rescue_quality_gate']=s.pop('dominance')
    meta=read(RUN/'evaluation/B0/validation_frames.json');scenes=sorted({m['scene'] for m in meta})
    rng=np.random.default_rng(928)
    w=np.vstack([np.ones(len(scenes)),rng.multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),size=1000)])
    s['net_rescue']={};net_vectors={}
    for name in MODELS:
        rows=read(RUN/'evaluation'/name/'statistics.json')['cohort_records']
        denominator=r.scene_array(rows,meta,scenes,lambda x:x['cohort'])
        s['net_rescue'][name]={}
        net_vectors[name]={}
        for mode in ['native','matched','calibrated']:
            numerator=r.scene_array(rows,meta,scenes,lambda x:x['modes'][mode]['rescued']-x['modes'][mode]['newly_empty'])
            values=100*r.ratio(w@numerator,w@denominator)
            if mode=='matched' and name!='B0':
                curves=np.load(RUN/'evaluation'/name/'operating_curves.npz')
                gross,thresholds=r.recomputed_matched_rescue(curves,w)
                selected=np.searchsorted(r.EDGES,thresholds)
                correct_hist=w@curves['baseline_correct_hist'].astype(float)
                total=correct_hist.sum(-1)
                correct_tail=np.cumsum(correct_hist[:,::-1],axis=-1)[:,::-1][:,1:]
                newly_empty=total-correct_tail[np.arange(len(w)),selected]
                revised=gross-100*r.ratio(newly_empty,w@denominator)
                np.testing.assert_allclose(revised[0],values[0],atol=1e-10)
                values=revised
            s['net_rescue'][name][mode]=r.interval(values)
            net_vectors[name][mode]=values
    s['net_rescue_comparisons']={mode:{
        'E_minus_S':r.interval(net_vectors['E'][mode]-net_vectors['S'][mode]),
        'interaction_SE_minus_S_minus_E_plus_B0':r.interval(net_vectors['SE'][mode]-net_vectors['S'][mode]-net_vectors['E'][mode]+net_vectors['B0'][mode])}
        for mode in ['native','matched','calibrated']}
    s['dominance']={name:{mode:dict(passed=bool(s['net_rescue'][name][mode]['ci95'][0]>50),
        net_rescue_fraction=s['net_rescue'][name][mode],criterion='Lower95 net semantic-correct rescue >50%; fixed recoverable-miss cohort')
        for mode in ['native','matched']} for name in ['S','E','SE']}
    with (RUN/'all_conditions.csv').open() as f:csv_rows=list(csv.DictReader(f))
    for row in csv_rows:
        net_result=s['net_rescue'][row['condition']][row['mode']]
        row.update(net_rescue_percent=net_result['value'],net_rescue_percent_lo=net_result['ci95'][0],net_rescue_percent_hi=net_result['ci95'][1])
    with (RUN/'all_conditions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    write_json(RUN/'summary.json',s)
    def ci(v):return f'{v["value"]:.2f} [{v["ci95"][0]:.2f}, {v["ci95"][1]:.2f}]'
    def v(name,mode,key):return s['models'][name]['modes'][mode]['metrics'][key]['value']
    names=r.NAMES
    native=r.table(['训练条件','mIoU（%）','漏检率（%）','误占用率（%）','前景首选正确率（%）'],
        [[names[n]]+[f'{v(n,"native",k):.4f}' for k in ['miou','fnr','fpr','rank_accuracy']] for n in MODELS])
    rescue=r.table(['条件','原生正确救回比例 [95% 区间]','同误占用正确救回比例 [95% 区间]'],
        [[names[n],ci(s['gross_rescue_quality_gate'][n]['native']['rescue_fraction']),ci(s['gross_rescue_quality_gate'][n]['matched']['rescue_fraction'])] for n in ['S','E','SE']])
    net=r.table(['条件','原生净救回比例 [95% 区间]','同误占用净救回比例 [95% 区间]'],
        [[names[n],ci(s['net_rescue'][n]['native']),ci(s['net_rescue'][n]['matched'])] for n in ['S','E','SE']])
    accounting=r.table(['条件','决策','正确救回（体素）','原正确→empty（体素）','原正确→错前景类（体素）'],
        [[names[n],'原生' if mode=='native' else '同误占用']+
         [s['models'][n]['modes'][mode]['cohort_counts'][key] for key in ['rescued','newly_empty','newly_wrong_foreground']]
         for n in ['S','E','SE'] for mode in ['native','matched']])
    calibrated=r.table(['条件','校准后 mIoU（%）','相对校准后 B0，百分点 [95% 区间]'],
        [[names[n],f'{v(n,"calibrated","miou"):.4f}',ci(s['contrasts'][n]['calibrated']['miou'])] for n in MODELS])
    matched=r.table(['条件','匹配阈值','实际误占用率（%）','漏检率（%）','mIoU（%）'],
        [[names[n],f'{s["models"][n]["matched_threshold"]:.5f}']+
         [f'{v(n,"matched",k):.4f}' for k in ['fpr','fnr','miou']] for n in MODELS])
    comparisons=r.table(['比较','原生净救回差值 [95% 区间]','同误占用净救回差值 [95% 区间]'],
        [[label,ci(s['net_rescue_comparisons']['native'][key]),ci(s['net_rescue_comparisons']['matched'][key])]
         for key,label in [('E_minus_S','CE 路径 − Set 路径'),('interaction_SE_minus_S_minus_E_plus_B0','两条路径的交互量')]])
    checks=[];initsha=None;initial_forward=None;paired_inputs=None
    for n in MODELS:
        c=read(RUN/'train'/n/'complete.json')
        assert c['epochs']==5 and c['steps']==5120 and c['frames']==20480
        rows=[json.loads(line) for line in (RUN/'train'/n/'microbatch_trace.jsonl').read_text().splitlines()]
        assert len(rows)==20480 and [row['global_iter'] for row in rows]==list(range(20480))
        fingerprints=[row['data_fingerprint'] for row in rows]
        if paired_inputs is None:paired_inputs=fingerprints
        assert fingerprints==paired_inputs, (n,'Actual training inputs differ from B0')
        for source in ['S','E']:
            if source in n:assert s['training_audit'][n]['sources'][source]['steps']==5120
        if initsha is None:initsha=c['initial_raw_sha256']
        current=np.load(RUN/'train'/n/'initial_forward.npz')
        if initial_forward is None:initial_forward=dict(current)
        np.testing.assert_array_equal(current['labels'],initial_forward['labels'])
        np.testing.assert_allclose(current['raw'],initial_forward['raw'],rtol=1e-5,atol=1e-6)
        checks.append(dict(condition=n,epochs=5,optimizer_steps=5120,train_frame_visits=20480,
            initial_raw_sha256=c['initial_raw_sha256'],initial_forward_max_error=float(np.abs(current['raw']-initial_forward['raw']).max()),
            all_20480_input_fingerprints_paired=True))
    se=s['net_rescue']['SE'];native_pass=se['native']['ci95'][0]>50;matched_pass=se['matched']['ci95'][0]>50
    if matched_pass:
        conclusion='联合阻断在控制误占用后仍通过了本轮数量主导判据，支持这两条直接训练路径在当前配置中净纠正了大部分“前景首选正确、却输给 empty”的基线漏检。'
    elif native_pass:
        conclusion='联合阻断在原生决策下通过数量主导判据，但同误占用水平下未通过。它们对原生决策工作点具有较大因果影响，但不能将这一结果直接解释为同等误占用预算下的判别能力改善。'
    else:
        conclusion='联合阻断未通过数量主导判据；从训练第一步开始切断这两条直接路径，也没有建立其净纠正了大部分“前景首选正确、却输给 empty”的基线漏检的证据。'
    cohort=s['models']['B0']['modes']['native']['fixed_cohort_size']
    baseline_cm=np.asarray(s['models']['B0']['modes']['native']['counts'])
    all_misses=int(baseline_cm[:17,17].sum())
    s['baseline_cohort_scope']=dict(recoverable_misses=cohort,all_foreground_misses=all_misses,
        recoverable_fraction_of_all_misses=100*cohort/all_misses)
    write_json(RUN/'summary.json',s)
    date=datetime.now().astimezone().strftime('%Y-%m-%d')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11})
    assets=ROOT/'analysis/empty_mechanism/history';assets.mkdir(parents=True, exist_ok=True)
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for offset,mode,label,color in [(-.17,'native','原生决策','#4c78a8'),(.17,'matched','相同误占用率','#e09547')]:
        entries=[s['net_rescue'][n][mode] for n in ['S','E','SE']]
        y=np.array([x['value'] for x in entries]);limits=np.array([x['ci95'] for x in entries]).T
        ax.bar(np.arange(3)+offset,y,.32,label=label,color=color)
        ax.errorbar(np.arange(3)+offset,y,yerr=np.maximum(np.vstack([y-limits[0],limits[1]-y]),0),fmt='none',ecolor='#333',capsize=4)
    ax.axhline(0,color='#777',lw=1);ax.axhline(50,color='#777',ls='--',lw=1,label='数量主导判据')
    ax.set_xticks(np.arange(3),['只阻断 Set 负类','只阻断 CE empty','同时阻断'])
    ax.set_ylabel('净救回数量 / 基线可恢复漏检数量（%）')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(assets/'net_rescue.png',dpi=180,bbox_inches='tight');plt.close(fig)
    text=f'''# empty 偏置的全程因果对照：实验结果

更新：{date}。状态：**四组完整五个 epoch 的训练及完整验证已完成。** [实验设计](empty_mechanism_history_design.md) · [全部指标与区间](../../../work_dirs/empty_mechanism_history/all_conditions.csv)。

## 1. 最重要的结果

**{conclusion}**

联合阻断在原生决策下的净救回比例为 **{ci(se['native'])}%**，同误占用水平下为 **{ci(se['matched'])}%**。它们是预先定义的反事实计数贡献，不是唯一的自然归因比例。

本轮四组都从相同图像预训练权重与 seed 928 的网络初始化开始，训练完整五个 epoch。它补上[末期干预](../stage3/empty_mechanism_stage3_results.md)不能消除早期训练历史的限制，并将处理限定在实际与 occupancy 共享语义值的最后渲染阶段。末期检查处理全部六个 Set 分类阶段，两轮窗口、范围与基线组不同，贡献数值不直接相减。

## 2. 最终模型的整体表现

以下均为完整 1024 个验证帧、默认 camera mask 下的原生结果。原生仍包含原有可学习全局 empty bias。

{native}

S 只阻断最后渲染阶段 Set 负类直接压低语义幅度的分量，E 只阻断同一渲染阶段 CE empty 的对应分量，SE 同时阻断。其他监督和更早阶段分类保持原样；模型的共享参数仍可发生训练补偿。

## 3. 原来的漏检有多少被正确救回

以全程 B0 最终模型定义固定基线组，共 **{cohort} 个体素**：真实前景、前景首选正确，但最终预测为空。所有处理组都评价这同一批体素。

B0 的全部前景漏检为 **{all_misses} 个**；本轮固定组占其中 **{100*cohort/all_misses:.2f}%**。下文贡献比例以固定组为分母，不能写成“解释了同样比例的全部漏检”。前景首选本身出错的其他漏检不在这一主指标内。

只有最终语义预测正确才计入救回。下表单位为百分比，方括号为配对场景 95% 区间。

{rescue}

同误占用主救回区间在每次场景重采样中重新计算预算和阈值。它不会把错误前景预测计作救回，也不会因前景首选变错而表面改善。

匹配使用离散 margin 阈值，取误占用数量不超过 B0 原生预算的最近点。下表列出实际达到的工作点；它是验证集曲线上的描述性比较，阈值不作为部署参数。只有主救回与净救回区间在重采样中重新选择阈值，其他匹配指标的区间条件于原先选定的阈值。

{matched}

## 4. 计入新增漏检后，净贡献是多少

净救回数量等于“固定组中的正确救回”减去“B0 原本语义正确的前景，在处理后变为空”。再除以固定组大小，得到下表。该量可以为负。

这是预先定义的语义正确救回记账，不等于全部前景漏检数的净变化。B0 原先就分错前景类别、随后变为空等变化不在该扣减组中，全部漏检变化请结合第二节整体漏检率阅读。

{net}

下面列出计数，便于核对比例来自哪里。新增错前景类别单独列出，不混入“新增漏检”。

{accounting}

![完整五个 epoch 后：原生与同误占用水平下的净救回比例](net_rescue.png)

同误占用的正确救回与净救回区间，均在每次场景重采样中重新计算基线预算和曲线阈值。50% 是本轮事先固定的数量判据，不是普遍定律。

前景首选质量及新增错类是另外的代价，不作为本节计数贡献不存在的证明。完整体素结果保存在机器汇总与逐帧统计中。

两条路径的直接比较和交互如下，单位为净救回比例的百分点。交互量定义为 SE − S − E + B0；不能仅把两个单独贡献相加，替代联合结果。

{comparisons}

## 5. 校准后 mIoU：单独判断方法价值

每个模型都用相同 698 个训练场景代表帧拟合／调参，并在完整 1024 验证帧上评价。前景排序保持不变；验证标签不参与类别阈值选择。

{calibrated}

校准后性能回答“是否超出了静态阈值参照”，不用于代替机制的因果贡献判断。即使训练路径影响了原生 empty 偏置，它的阻断也可能带来语义或误占用代价；反过来，校准后没有额外收益，也不自动否定其训练作用。

## 6. 可以怎样表述机制结论

本轮原生数量主导判定为 **{'通过' if native_pass else '未通过'}**，同误占用数量主导判定为 **{'通过' if matched_pass else '未通过'}**。应同时引用连续贡献值、正确救回与新增错误，而不是只引用某个损失存在局部抑制方向。

全程受控训练可以检验这两条直接路径对最终模型的因果影响，但不能把多个相互补偿的训练来源解释成唯一可加的历史成因百分比。四组仍只有一个原始训练 seed；场景区间不涵盖重新训练的不确定性，也不能代替其他 Semantic Gaussian 方法上的验证。

## 附录：完成量与核查

四组共完成 **20 个训练 epoch、81920 个训练帧次、20480 次优化器更新**。每组最终评价完整 1024 个验证帧。首个样本标签一致，分类输出通过数值一致性检查；全部 20480 个训练微批次的投影矩阵与标签指纹逐一配对一致，其校验值及误差、完整微批次编号和源梯度去除记录保存可查。

使用原 train.py 加载预训练、建立数据流并优化；新增包装只调用已验证的路径阻断。小规模原生试运行验证了四个累积反向和一次优化器更新，真实网络图检查验证了 CE 校正对共享原始 logit 父节点的作用，以及空间与 empty 输出直接梯度保持。

核查结果：[全程完整性审计](../../../work_dirs/empty_mechanism_history/completion_audit.json) · [机器汇总](../../../work_dirs/empty_mechanism_history/summary.json) · [全程协议](../../../work_dirs/empty_mechanism_history/protocol.json)。实现：[原生训练包装](../../../loss/empty_history_path_loss.py) · [训练与评价](../../../tools/empty_mechanism_history.py) · [报告生成](../../../tools/report_empty_mechanism_history.py)。
'''
    (ROOT/'analysis/empty_mechanism/history/empty_mechanism_history_results.md').write_text(text)
    write_json(RUN/'completion_audit.json',dict(status='passed',models=checks,reference_initial_raw_sha256=initsha,
        all_initial_forwards_within_tolerance=True,
        validation_frames_per_model=1024,scenes=s['scenes'],original_artifacts_unchanged=True,
        whole_history_runs=True,native_net_dominance=native_pass,matched_net_dominance=matched_pass,
        summary_sha256=file_sha256(RUN/'summary.json'),source_sha256=file_sha256(__file__)))
    design=ROOT/'analysis/empty_mechanism/history/empty_mechanism_history_design.md'
    if design.exists():
        design.write_text(design.read_text().replace('状态：原生训练接口试运行已通过，正式全程对照已排队。',
            '状态：**四组完整五个 epoch 的训练、完整验证和配对统计已完成。**').replace(
            '状态：**正式全程对照执行中，当前完成量见[实时进度](../empty_mechanism_progress.md)。**',
            '状态：**四组完整五个 epoch 的训练、完整验证和配对统计已完成。**'))
    print(ROOT/'analysis/empty_mechanism/history/empty_mechanism_history_results.md')


if __name__=='__main__':main()
