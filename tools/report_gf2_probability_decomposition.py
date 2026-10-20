"""Paired scene statistics and readable report for probability rendering study."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('OMP_NUM_THREADS','8')
from pathlib import Path
import csv
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.gf2_probability_experiment import RUN,read,write,sha,initialize
from tools.eval_gf2_probability_decomposition import MODELS,RULES
from tools.calibrate_empty_mechanism_stage2 import EDGES,NB
from tools.eval_empty_decision_optimization import metrics

NBOOT=1000
LABELS=dict(joint='联合（原生）',hierarchical='分层（未校准）',joint_global='联合＋全局校准',
    joint_class='联合＋类别校准',hierarchical_class='分层＋类别校准',risk_class='风险族＋类别校准')


def interval(values):return np.quantile(values,[.025,.975]).tolist()


def metric_values(counts):
    result=metrics(counts)
    c=np.asarray(counts,dtype=np.float64)
    fg=c[...,:17,:].sum((-2,-1))
    result['foreground_final_accuracy']=np.diagonal(c,axis1=-2,axis2=-1)[...,:17].sum(-1)/np.maximum(fg,1)*100
    return result


def matched_budget(curves,weights):
    """Reselect a conservative scalar threshold for EACH scene resample.

    Return actual false positives, gross/net correct rescue, and threshold.
    +infinite scores remain foreground for every finite registered threshold.
    A failed budget is exposed rather than silently counted as matched.
    """
    empty=np.asarray(curves['empty_hist'],dtype=np.float64)
    rescue=np.asarray(curves['rescue_hist'],dtype=np.float64)
    correct=np.asarray(curves['baseline_correct_hist'],dtype=np.float64)
    budget=weights@curves['baseline_fp'];cohort=weights@curves['cohort_size']
    output={k:[] for k in ['fp','gross','lost','net','threshold','feasible','cohort']}
    for begin in range(0,len(weights),25):
        w=weights[begin:begin+25]
        tails=[np.cumsum((w@a)[:,::-1],axis=-1)[:,::-1] for a in [empty,rescue,correct]]
        feasible=tails[0][:,1:]<=budget[begin:begin+len(w),None]
        valid=feasible.any(-1);index=feasible.argmax(-1)
        index=np.where(valid,index,len(EDGES)-1)
        row=np.arange(len(w));bucket=index+1
        fp=tails[0][row,bucket];gross=tails[1][row,bucket]
        lost=tails[2][:,0]-tails[2][row,bucket]
        values=dict(fp=fp,gross=gross,lost=lost,net=gross-lost,
            threshold=EDGES[index],feasible=valid,cohort=cohort[begin:begin+len(w)])
        for k,v in values.items():output[k].extend(v.tolist())
    return {k:np.asarray(v) for k,v in output.items()}


def summarize_model(name,weights,scenes):
    dest=RUN/'evaluation'/name
    assert read(dest/'statistics_complete.json')['status']=='complete'
    stats=read(dest/'statistics.json');records=stats['records'];meta=read(dest/'validation_frames.json')
    assert len(records)==len(meta)==1024 and stats['scenes']==scenes
    lookup={s:i for i,s in enumerate(scenes)}
    def aggregate(values):
        array=np.asarray(values)
        grouped=np.zeros((len(scenes),)+array.shape[1:],dtype=array.dtype)
        for m,value in zip(meta,array):grouped[lookup[m['scene']]]+=value
        return grouped
    rules={};bootstrap={}
    for rule in RULES:
        c=aggregate([r['rules'][rule]['counts'] for r in records])
        point=metric_values(c.sum(0)[None]);boot=metric_values((weights@c.reshape(len(scenes),-1)).reshape(NBOOT,18,18))
        rules[rule]={k:dict(value=float(v[0]),ci=interval(boot[k])) for k,v in point.items() if k!='per_class_IoU'}
        rules[rule]['per_class_IoU']=point['per_class_IoU'][0].tolist()
        rules[rule]['transitions']={k:sum(r['rules'][rule][k] for r in records) for k in ['rescued','newly_empty','newly_wrong']}
        bootstrap[rule]=boot
    comparisons={}
    for first,second in [('hierarchical','joint'),('joint_class','joint'),('risk_class','joint_class'),
        ('risk_class','hierarchical_class'),('hierarchical_class','joint_class')]:
        delta=bootstrap[first]['mIoU']-bootstrap[second]['mIoU']
        comparisons[first+'__'+second]=dict(delta_pp=rules[first]['mIoU']['value']-rules[second]['mIoU']['value'],ci=interval(delta))
    temperature={}
    for t in ['0.5','0.75','1.0','1.5','2.0']:
        c=aggregate([r['temperatures'][t]['counts'] for r in records])
        point=metric_values(c.sum(0)[None]);boot=metric_values((weights@c.reshape(len(scenes),-1)).reshape(NBOOT,18,18))
        temperature[t]={k:float(v[0]) for k,v in point.items() if k!='per_class_IoU'}
        temperature[t]['mIoU_ci']=interval(boot['mIoU'])
        temperature[t]['delta_mIoU_ci']=interval(boot['mIoU']-bootstrap['joint']['mIoU'])
        temperature[t]['events']={k:sum(r['temperatures'][t][k] for r in records) for k in
            ['correct_to_empty','missed_to_correct','empty_to_foreground','foreground_to_empty',
             'rank_changes','empty_probability_changes','hierarchical_changes','numerical_argmax_ties']}
    totals={k:sum(r[k] for r in records) for k in ['foreground','empty','native_missed','cohort','rank_correct','saturated_zero','saturated_one']}
    gap={k:sum(r['gap'][k] for r in records) for k in records[0]['gap']}
    gap['fraction_all_percent']=100*gap['gap']/(totals['foreground']+totals['empty'])
    gap['fraction_true_foreground_percent']=100*gap['gap_true_foreground']/totals['foreground']
    gap['recoverable_miss_fraction_percent']=100*gap['gap_correct_winner']/max(totals['cohort'],1)
    matched={}
    for rule in ['joint','hierarchical','joint_class','hierarchical_class','risk_class']:
        curves=np.load(dest/f'curves_{rule}.npz')
        np.testing.assert_array_equal(curves['scenes'],scenes)
        point=matched_budget(curves,np.ones((1,len(scenes))))
        boot=matched_budget(curves,weights)
        result={k:float(v[0]) for k,v in point.items()}
        for k in ['gross','net']:
            denom=np.maximum(boot['cohort'],1)
            result[k+'_percent']=100*point[k][0]/max(point['cohort'][0],1)
            result[k+'_percent_ci']=interval(100*boot[k]/denom)
        result['false_positive_rate']=100*point['fp'][0]/totals['empty']
        result['bootstrap_infeasible']=int((~boot['feasible']).sum())
        matched[rule]=result
    grouped_rank=aggregate([[r['rank_correct'],r['foreground']] for r in records])
    bootrank=weights@grouped_rank
    rank=dict(value=100*totals['rank_correct']/totals['foreground'],ci=interval(100*bootrank[:,0]/bootrank[:,1]))
    return dict(rules=rules,comparisons=comparisons,temperatures=temperature,totals=totals,gap=gap,
        matched=matched,rank_accuracy=rank,selection=read(dest/'selection.json'),timing=read(dest/'timing.json')),bootstrap


def audit():
    p=initialize();checks={}
    checks['baseline_training_complete']=read(ROOT/p['baseline_complete'])['status']=='complete'
    training=read(RUN/'train/probability/complete.json')
    checks['probability_training_complete']=training['status']=='complete' and training['epochs']==5 and training['frames']==20480 and training['paired_inputs']
    checks['registered_protocol_before_training']=p['registered_before_formal_probability_training']
    checks['backend_forward_gradient_check']=read(RUN/'verification/backend_check.json')['status']=='passed'
    checks['native_pilot']=read(RUN/'native_pilot/pilot_success.json')['status']=='passed'
    unit=read(RUN/'verification/unit_tests.json')
    checks['decision_and_statistical_unit_tests']=unit['passed']==10 and unit['source_sha256']==sha(ROOT/'tests/test_gf2_probability_decisions.py')
    reporting=read(RUN/'verification/reporting_tests.json')
    checks['isolated_report_pipeline_test']=reporting['passed']==1 and reporting['source_sha256']==sha(ROOT/'tests/test_gf2_probability_reporting.py')
    checks['finite_losses_and_native_gradient_guard']=training['finite_losses'] and training['native_nonfinite_gradient_guard']
    for path,value in training['sources'].items():checks['source:'+path]=sha(ROOT/path)==value
    for path,value in p['old_results_sha256'].items():checks['old_result:'+path]=sha(ROOT/path)==value
    manifest=read(RUN/'vendor_manifest.json')
    # The manifest explicitly stores the unchanged vendored source files.
    for path,value in manifest['sha256'].items():checks['vendor:'+path]=sha(RUN/'vendor/localagg_prob'/path)==value
    metadata=[]
    for name in MODELS:
        dest=RUN/'evaluation'/name
        record=read(dest/'extraction_complete.json')
        checks[name+':full_extraction']=record['status']=='complete' and record['validation_frames']==1024 and record['calibration_frames']==698
        checks[name+':checkpoint']=sha(ROOT/read(dest/'checkpoint.json')['path'])==record['checkpoint_sha256']
        records=read(dest/'statistics.json')['records']
        checks[name+':full_statistics']=len(records)==1024 and [r['index'] for r in records]==list(range(1024))
        checks[name+':negative_controls']=all(
            all(r['temperatures'][t][k]==0 for k in ['rank_changes','empty_probability_changes','hierarchical_changes'])
            for r in records for t in r['temperatures'])
        checks[name+':temperature_identity']=all(r['temperatures']['1.0']['counts']==r['rules']['joint']['counts'] for r in records)
        metadata.append(read(dest/'validation_frames.json'))
    checks['paired_validation_metadata']=metadata[0]==metadata[1]
    write(RUN/'completion_audit.json',dict(status='passed' if all(checks.values()) else 'failed',checks=checks))
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    return checks


def number(value):return f'{value:.3f}'


def ci(value):return f"[{number(value[0])}, {number(value[1])}]"


def main():
    checks=audit()
    meta=read(RUN/'evaluation/baseline/validation_frames.json');scenes=sorted({r['scene'] for r in meta})
    rng=np.random.default_rng(9282027)
    draws=rng.integers(0,len(scenes),size=(NBOOT,len(scenes)))
    weights=np.asarray([np.bincount(a,minlength=len(scenes)) for a in draws],dtype=np.float64)
    summary={};boots={}
    for name in MODELS:
        summary[name],boots[name]=summarize_model(name,weights,scenes)
        assert all(r['feasible'] and r['bootstrap_infeasible']==0 for r in summary[name]['matched'].values()),'Registered matched-FPR budget infeasible'
        print('Summarized',name,flush=True)
    summary['cross_model']={}
    for rule in ['joint','joint_global','joint_class','hierarchical_class','risk_class']:
        delta=boots['probability'][rule]['mIoU']-boots['baseline'][rule]['mIoU']
        summary['cross_model'][rule]=dict(delta_pp=summary['probability']['rules'][rule]['mIoU']['value']-summary['baseline']['rules'][rule]['mIoU']['value'],ci=interval(delta))
    summary['protocol']=dict(bootstrap=NBOOT,scene_count=len(scenes),validation_frames=1024,seed=9282027)
    write(RUN/'summary.json',summary)
    rows=[]
    for name in MODELS:
        for rule,r in summary[name]['rules'].items():
            rows.append(dict(model=name,rule=rule,**{k:v['value'] for k,v in r.items() if isinstance(v,dict) and 'value' in v}))
    with (RUN/'performance.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot(summary)
    lines=['# GaussianFormer-2 式概率渲染替换：实验结果','',
        '**状态：训练、验证和三项诊断均已完成。** 本轮使用原 4096 帧训练／1024 帧验证划分，默认 camera mask；两组各训练五轮，每组完成 20480 个训练帧次、5120 次优化器更新。',
        '', '本报告比较 OPUS 上的渲染替换，不能作为 GaussianFormer-2 整模型复现结果。实验条件和符号见[实验设计](gf2_probability_decomposition_design.md)。',
        '', '## 1. 先看替换后的性能','',
        '表中指标单位均为 %；差值用百分点（pp）。mIoU 为 17 个前景类别 IoU 的平均，二值 IoU 合并所有前景。漏检率的分母是真实前景，误占用率的分母是真实 empty。“最终正确率”要求既预测占有又预测对类别。',
        '', '| 模型 | 决策 | mIoU | 95% 场景区间 | 二值 IoU | 漏检率 | 误占用率 | 最终正确率 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name in MODELS:
        for rule in ['joint','joint_global','joint_class','hierarchical','hierarchical_class','risk_class']:
            r=summary[name]['rules'][rule]
            vals=[number(r[k]['value']) for k in ['IoU','occupied_FNR','empty_FPR','foreground_final_accuracy']]
            lines.append(f"| {'B0 加性' if name=='baseline' else 'P 概率'} | {LABELS[rule]} | {number(r['mIoU']['value'])} | {ci(r['mIoU']['ci'])} | "+' | '.join(vals)+' |')
    lines+=['','B0 保留原先可学习的全局 empty bias；P 按概率分解移除了该 bias 和 empty Gaussian。因此原生差异包含参数化变化。全局和逐类别校准均只使用训练场景，每个模型获得同等机会。',
        '', '| 同等决策下 P − B0 | mIoU 差值（pp） | 配对 95% 场景区间 |','| --- | --- | --- |']
    for rule,r in summary['cross_model'].items():lines.append(f"| {LABELS[rule]} | {number(r['delta_pp'])} | {ci(r['ci'])} |")
    native=summary['cross_model']['joint'];calibrated=summary['cross_model']['joint_class']
    calibration_verdict='同等类别校准后，概率替换仍有正向优势，且本轮场景区间不跨 0。' if calibrated['ci'][0]>0 else '同等类别校准后的场景区间跨 0，本轮没有建立稳健的额外优势。' if calibrated['ci'][1]>=0 else '同等类别校准后概率替换表现更低，本轮场景区间不跨 0。'
    lines+=['',f"替换的原生 mIoU 差值为 **{number(native['delta_pp'])} pp**；同等类别校准后差值为 **{number(calibrated['delta_pp'])} pp**。这两个结果需要同时阅读：前者反映直接使用效果，后者检查工作点校正后是否仍有优势。",'',
        calibration_verdict+'这是当前 mini 划分和一个训练 seed 下的结果。','',
        '![性能与温度干预](../work_dirs/gf2_probability_decomposition/figures/overview.png)','',
        '## 2. 验证一：自然预测中的联合／分层差异','',
        '决策差异组指“分层预测前景、联合预测 empty”的体素。分层会把这一组全部改成前景，但只有“真实前景且首选正确”的部分能正确救回。',
        '', '| 统计量 | B0 加性 | P 概率 |','| --- | --- | --- |']
    for key,label in [('gap','差异组体素数'),('gap_correct_winner','真实前景、首选正确'),('gap_wrong_winner','真实前景、首选错误'),('gap_true_empty','真实 empty：新增误占用')]:
        lines.append(f"| {label} | {summary['baseline']['gap'][key]:,} | {summary['probability']['gap'][key]:,} |")
    for key,label in [('fraction_all_percent','差异组占全部有效体素（%）'),('fraction_true_foreground_percent','差异组真实前景占全部真实前景（%）'),('recoverable_miss_fraction_percent','差异组正确首选占原生可恢复漏检（%）')]:
        lines.append(f"| {label} | {number(summary['baseline']['gap'][key])} | {number(summary['probability']['gap'][key])} |")
    lines+=['','这里的“可恢复漏检”固定为原生联合预测 empty、但前景首选正确的真实前景体素。它不包含前景首选错误的体素。',
        '', '| 模型 | 前景首选正确率（忽略 empty 门） | 占有概率等于 0 的体素 | 等于 1 的体素 |','| --- | --- | --- | --- |']
    for name in MODELS:
        a=summary[name];lines.append(f"| {name} | {number(a['rank_accuracy']['value'])}% | {a['totals']['saturated_zero']:,} | {a['totals']['saturated_one']:,} |")
    lines+=['','B0 的占有量来自最终 softmax 的代数分组；它并非独立学习的几何概率。P 的占有量才直接来自官方 noisy-OR 渲染。概率饱和与无支持体素会限制任何有限阈值校准的效果。',
        '', '## 3. 验证二：empty 概率不变，类别置信度能否改变漏检','',
        '以下只改变条件类别分布的温度，固定每个体素的占有概率、empty 概率和前景首选类别。T<1 使分布更尖锐；T>1 使分布更平坦。转换数量都相对该模型 T=1 的原生联合预测。',
        '', '| 模型 | T | mIoU | 漏检率 | 误占用率 | 正确前景→empty | 漏检→正确前景 | 真实 empty→前景 |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name in MODELS:
        for t,r in summary[name]['temperatures'].items():
            e=r['events'];lines.append(f"| {name} | {t} | {number(r['mIoU'])} | {number(r['occupied_FNR'])} | {number(r['empty_FPR'])} | {e['correct_to_empty']:,} | {e['missed_to_correct']:,} | {e['empty_to_foreground']:,} |")
    p=summary['probability']['temperatures']['2.0']['events'];b=summary['probability']['temperatures']['0.5']['events']
    numerical_ties={name:sum(v['events']['numerical_argmax_ties'] for v in summary[name]['temperatures'].values()) for name in MODELS}
    lines+=['',f"在 P 中，仅将 T 调到 2，就有 **{p['correct_to_empty']:,}** 个原本正确前景变为 empty；调到 0.5 可正确救回 **{b['missed_to_correct']:,}** 个原生漏检。所有温度下，前景首选变化、empty 概率变化和分层决策变化均为 **0**；T=1 精确重现原决策。",'',
        f"数值处理：温度变换可能把相差一个双精度单位的概率舍入为完全并列的最大值。本轮跨五个温度共记录 B0 **{numerical_ties['baseline']:,}**、P **{numerical_ties['probability']:,}** 个此类“体素×温度”事件；均保留原首选类别，并严格核验它仍为变换后的最大值。没有放宽严格排序反转的检查，也没有改动训练或验证缓存。修订经过 11 项检查，见[统计修订记录](../work_dirs/gf2_probability_decomposition/statistical_repair.json)。",'',
        '这个干预检验的是联合决策路径：即使 empty 分数不升高，前景概率份额也能改变 empty 是否获胜。它不能单独证明自然训练中的主要成因，也不意味着联合 MAP 对平坦类别错误代价是错误的。',
        '', '## 4. 验证三：显式任务代价是否超过公平校准参照','',
        '风险族固定误占用和漏检代价为 1，用 γ 控制前景错类代价。γ=1 为联合决策，γ=0 为分层决策。每个候选先在 523 个训练场景拟合类别阈值，再用另外 175 个训练场景选择公共平移和 γ。验证标签不参与此选择。',
        '', '| 模型 | 训练集选中 γ | 风险族 − 联合类别校准（pp） | 95% 区间 | 风险族 − 分层类别校准（pp） | 95% 区间 |','| --- | --- | --- | --- | --- | --- |']
    for name in MODELS:
        a=summary[name];x=a['comparisons']['risk_class__joint_class'];y=a['comparisons']['risk_class__hierarchical_class']
        lines.append(f"| {name} | {a['selection']['risk']['ratio']} | {number(x['delta_pp'])} | {ci(x['ci'])} | {number(y['delta_pp'])} | {ci(y['ci'])} |")
    if all(summary[name]['selection']['risk']['ratio']==1 for name in MODELS):
        lines+=['','**两组都选择了 γ=1，即联合决策端点。** 因而本轮风险族没有超过“联合＋类别校准”这一参照。固定占有时存在置信度影响路径，与显式降低错类代价能提高实际性能，是两个需要分开判断的命题。']
    lines+=['','风险族与类别阈值是一个决策校准诊断，不是新的 GaussianFormer-2 渲染机制。区间若跨 0，应写成未获得稳健优势；不能用最优验证工作点代替训练集选出的结果。风险代价也不等于 mIoU 最优性证明。',
        '', '## 5. 保持误占用预算时，还能正确救回多少','',
        '本节仅作验证曲线诊断：以各模型原生联合预测的误占用数量为上限，在登记网格上选择满足预算的最低公共阈值。“净救回”=正确救回−原本正确但新增漏检；分母固定为原生可恢复漏检数量。每次场景重采样重新选择阈值。这里的阈值不作为可部署参数，也不声称精确相同的 FPR。',
        '', '| 模型 | 决策 | 实际误占用率（%） | 正确救回／可恢复漏检（%） | 95% 区间 | 净救回（%） | 95% 区间 |','| --- | --- | --- | --- | --- | --- | --- |']
    for name in MODELS:
        for rule,r in summary[name]['matched'].items():
            lines.append(f"| {name} | {LABELS[rule]} | {number(r['false_positive_rate'])} | {number(r['gross_percent'])} | {ci(r['gross_percent_ci'])} | {number(r['net_percent'])} | {ci(r['net_percent_ci'])} |")
    lines+=['','完整阈值、预算可行性、计数及区间保存在机器可读统计中。该指标针对固定漏检组，不等于 mIoU，也不能跨模型把不同的可恢复组直接当作相同因果分母。',
        '', '## 6. 效率、完整性和结论边界','',
        '| 模型 | 预加载输入的平均单帧时间（ms） | FPS | 额外风险门耗时（ms） |','| --- | --- | --- | --- |']
    for name in MODELS:
        t=summary[name]['timing'];extra='—' if t['additional_risk_gate_seconds'] is None else number(t['additional_risk_gate_seconds']*1000)
        lines.append(f"| {name} | {number(t['mean_seconds']*1000)} | {number(t['fps'])} | {extra} |")
    lines+=['','使用同一 RTX 4090、batch=1、FP32、256×704 相机输入，10 次预热＋30 次计时；不包含图像读取和预处理。风险门单独计时。这里只代表本机该实现的速度，不能直接与其他论文公布的 FPS 横比。',
        '', f"完整性审计 **{sum(checks.values())}/{len(checks)}** 项通过，包括官方后端前向／梯度核对、原生累积更新预检、两组首次输出及标签和投影矩阵的全部 20480 条指纹配对、完整 1024 帧验证、温度阴性对照以及既有结果文件未改变。",'',
        f"区间来自 {len(scenes)} 个验证场景的 1000 次配对重采样。这里只有一个训练 seed；区间没有覆盖训练随机性的变化。概率渲染的 empty NLL 对固定几何下的语义输出直接梯度为零，不表示共享网络参数与几何训练完全独立。",'',
        '这轮需要区分三个层次：概率渲染替换的实际性能；固定输出条件下可以验证的决策耦合；需要额外自然训练干预才能判断的主导成因。分解与分层决策本身已有 GaussianFormer-2 先例，创新性应建立在新增的可检验机制和超过公平校准参照的方法收益上。',
        '', '数据入口：[完整统计](../work_dirs/gf2_probability_decomposition/summary.json) · [性能 CSV](../work_dirs/gf2_probability_decomposition/performance.csv) · [完整性审计](../work_dirs/gf2_probability_decomposition/completion_audit.json) · [运行协议](../work_dirs/gf2_probability_decomposition/protocol.json)。','']
    (ROOT/'analysis/gf2_probability_decomposition_results.md').write_text('\n'.join(lines))
    design=ROOT/'analysis/gf2_probability_decomposition_design.md'
    text=design.read_text().replace('协议已登记，官方 CUDA 后端已编译，正在完成正式训练前校验。',
        '按登记协议完成训练、验证和三项诊断；实测结果见配套结果报告。')
    text=text.replace('协议已登记，官方后端及完整模型预检通过，正式五轮训练已启动。',
        '按登记协议完成训练、验证和三项诊断；实测结果见配套结果报告。')
    text=text.replace('正在运行的全程对照中、同一 seed 从头训练的原模型',
        '全程对照中、同一 seed 从相同预训练初始化训练的原模型')
    design.write_text(text)
    print('Result report written.',flush=True)


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=RUN/'figures';dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    x=np.arange(3);width=.34
    for index,name in enumerate(MODELS):
        vals=[summary[name]['rules'][r]['mIoU']['value'] for r in ['joint','joint_class','risk_class']]
        axes[0].bar(x+(index-.5)*width,vals,width,label=name)
        t=summary[name]['temperatures'];axes[1].plot([float(k) for k in t],[v['occupied_FNR'] for v in t.values()],marker='o',label=name)
    axes[0].set_xticks(x,['Native joint','Class calibrated','Risk + class']);axes[0].set_ylabel('mIoU (%)')
    height=max(summary[name]['rules'][rule]['mIoU']['value'] for name in MODELS for rule in ['joint','joint_class','risk_class'])
    axes[0].set_ylim(0,height*1.23);axes[0].legend()
    axes[1].set_xlabel('Conditional semantic temperature');axes[1].set_ylabel('Foreground miss rate (%)');axes[1].legend()
    fig.tight_layout();fig.savefig(dest/'overview.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
