"""Evidence tables, paired uncertainty, calibration and readable experiment report."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from tools.empty_decision_optimization import RUN, REPORT, SEEDS, PRIMARY_BUDGET, read, write_json, file_sha256
from tools.eval_empty_decision_optimization import metrics
from tools.diagnose_opus_v1_gaussian import CLASSES

LABELS = dict(B0='现有 learnable bias',B1='全局 bias',B2='类别 bias',B3='普通局部网络',
    B3_class='普通局部网络＋类别阈值',P1='三结果概率＋统一阈值',P1_q05='三结果概率，q≥0.5',
    P2='概率＋IoU 决策',P3='概率＋IoU＋风险约束',P4='P3 去掉 Gaussian 支持',P3_oneway='P3 仅允许降低 empty')
ENGLISH = dict(B0='Existing bias',B1='Global bias',B2='Class bias',B3='Local NLL',B3_class='Local NLL + class thresholds',
    P1='Three-outcome probability',P1_q05='Probability q >= 0.5',P2='Probability + IoU',P3='IoU + risk constraint',
    P4='Without Gaussian support',P3_oneway='Lower empty only')
ORDER = ['B0','B1','B2','B3','B3_class','P1_q05','P1','P2','P3','P4','P3_oneway']

METHOD_DETAILS = r'''### 1.3 类别 bias B2：先预测前景类别，再查阈值

B2 在 Gaussian 聚合后的体素分数上运行。先忽略 empty，在 17 个前景类别中找出首选 $c^*$，再查该类别的阈值 $t_c$。margin $m$ 是前景首选分数减去**已经包含全局 learnable bias** 的 empty 分数。

$$
s_{\mathrm{B2}}(x)=m(x)-t_{c^*(x)},\qquad b_c=-t_c.
$$

s≥0 输出前景首选，否则输出 empty。这等价于在已有全局 bias 上，再将 empty 分数减去类别 bias $b_c$。负阈值更容易放出前景，正阈值更保守。

| 前景首选类别 | 主工作点阈值 | 示例 |
|---|---:|---|
| trailer | −1.75000 | 明显放宽前景决策 |
| car | −0.46875 | margin=−0.3 时，由 empty 改为 car |
| driveable_surface | +1.03125 | margin=+0.5 时，由路面改为 empty |

查表依据是模型预测，推理不使用真值类别或 camera mask。即使原生最终输出 empty，仍可在前景中找出首选。B2 不重排前景类别，也不是给每个 Gaussian 单独修改语义分数。

17 个阈值在全部 994 个训练内调参帧上校准，范围 −2 到 +2、步长 1/32；风险权重搜索产生 109 组不同组合，在满足预算的组合中选 camera mIoU 最高者。没有反向传播训练这 17 个阈值，也未重新训练主模型。选择在评价验证集之前冻结；有限搜索不保证所有组合中的全局最优。

### 1.4 q/e/o 在已实现的 P3 中究竟怎么用？

**q/e/o 是决策网络的辅助输入，网络最终输出一个额外 bias。当前没有把三个概率分别换成三个固定修正项，也没有直接执行设计文档中的概率收益公式。**

```text
主模型分数（已减去全局 bias）
    → 提取特征 h → 概率头输出 q/e/o
    → 拼接 h＋q/e/o → 决策网络输出单个 Δb
    → 比较 margin＋Δb 与统一阈值 t
    → 输出当前前景首选或 empty
```

实际计算为：

$$
\Delta b_\theta=2\tanh\bigl(g_\theta([h,q,e,o])\bigr),\qquad
s=m+\Delta b_\theta.
$$

在 margin ±2 的候选内，s≥t 时输出前景，否则输出 empty；t 为训练内选定的固定工作点。P3 使用全局 bias 加局部修正，**没有再叠加 B2 的 17 个类别阈值**。代码见[决策拟合与 Predictor.__call__](../../../tools/fit_empty_decision_optimization.py)及[最终阈值判断](../../../tools/eval_empty_decision_optimization.py)。

| 概率 | 拟合时的事件标签 | 希望为决策提供的信息 |
|---|---|---|
| q | 当前前景首选正确 | 正确语义收益 |
| e | 真值为 empty | 新增误占用风险 |
| o | 真值为其他前景类别 | 恢复占用但类别仍错的风险 |

概率头由三分类 NLL 训练后冻结。决策层再从 h＋q/e/o 学习修正；其 soft mIoU 和风险均用真实标签计算，不用预测 e 替代真实风险。没有强制 q 增大时 bias 必须增大，或 e 增大时 bias 必须减小；o 也不能指出应该改成哪个其他类别。

因此，给概率赋予事件含义，并不等于已证明决策网络有效使用了该含义。网络还可直接依赖 h 中的类别、位置和分数。P3 优于 P2 支持风险约束的作用；概率的独立贡献仍需同目标、相近容量的“仅 h”与“h＋q/e/o”对照。P1 是直接按 q 阈值判断的另一诊断版本，不经过上述局部决策网络。

**后续两种“类别 bias＋残差”不属于本报告的首轮结果。** 它们分别从普通特征、普通特征加 q/e/o 预测一个残差，加入 B2 的类别决策分数。定义与算术示例见[复盘文档第 7.1 节](empty_decision_optimization_failure_analysis.md)；截至 2026-10-18 已完成后续独立验证，见[进一步实验结果](../followup/empty_decision_followup_results.md)。
'''


def primary_points(validation):
    return {p['family']:p for p in validation['operating_points'].values() if p['id'] == 'B0' or
        (p['budget'] == PRIMARY_BUDGET and p['seed'] in [None,928]) or p['family'] == 'P1_q05'}


def probability_ap(validation):
    dest = RUN/'probability_metrics.json'
    if dest.exists():
        return read(dest)
    arrays = {k:[] for k in ['probability_sample_qe','probability_sample_outcome','probability_sample_region','probability_sample_weight']}
    names = None
    for frame in validation['frames']:
        stem = ROOT/frame['path']
        frame_names = read(stem.with_suffix('.json'))['probability_sample_models']
        if names is None:
            names = frame_names
        assert names == frame_names
        with np.load(stem.with_suffix('.npz')) as z:
            for key in arrays:
                arrays[key].append(z[key])
    data = {k:np.concatenate(v) for k,v in arrays.items()}
    probability,target,region,w = [data[k] for k in arrays]
    result = dict(status='complete',rows=len(target),sampling='256 per camera/complement x native margin sign per frame; inverse inclusion weights',
        scope='abs(native margin)<=2; bidirectional candidate population, different from previous b=+1 flip-only AP',models={})
    for i,name in enumerate(names):
        result['models'][name] = {}
        for r,label in [(0,'camera'),(1,'camera_complement'),(None,'all_valid')]:
            keep = np.ones(len(target),bool) if r is None else region == r
            ww = w[keep]
            row = {}
            for event,key in [(0,'q'),(1,'e')]:
                truth = target[keep] == event
                row[key+'_AP'] = float(average_precision_score(truth,probability[keep,i,event],sample_weight=ww))
                row[key+'_prevalence'] = float(np.average(truth,weights=ww))
            result['models'][name][label] = row
    write_json(dest,result)
    return result


def scene_bootstrap(validation,primary):
    with np.load(RUN/'validation/per_frame_counts.npz') as z:
        counts = z['confusion']
    points = list(validation['operating_points'])
    scene = np.asarray([f['scene'] for f in validation['frames']])
    unique,inverse = np.unique(scene,return_inverse=True)
    families = [f for f in ORDER if f in primary]
    ids = [points.index(primary[f]['id']) for f in families]
    byscene = np.zeros((len(unique),len(ids),2,18,18),np.float64)
    np.add.at(byscene,inverse,counts[:,ids])
    rng = np.random.default_rng(928)
    multiplicities = rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),500)
    with threadpool_limits(8):
        summed = (multiplicities@byscene.reshape(len(unique),-1)).reshape(500,len(ids),2,18,18)
    camera = metrics(summed[:,:,0])['mIoU']
    risk = metrics(summed[:,:,1])['empty_FPR']
    result = dict(repetitions=500,scenes=len(unique),conditional='fixed checkpoint and fitted seed928 modules; no retraining or operating-point reselection',comparisons={})
    pairs = [('P3',b) for b in ['B0','B1','B2','B3','B3_class','P2','P4','P3_oneway']]
    pairs += [('P2','B3'),('B2','B1')]
    for a,b in pairs:
        ia,ib = families.index(a),families.index(b)
        result['comparisons'][a+'__minus__'+b] = dict(
            camera_mIoU_delta_ci95=np.quantile(camera[:,ia]-camera[:,ib],[.025,.975]).tolist(),
            out_FPR_delta_ci95=np.quantile(risk[:,ia]-risk[:,ib],[.025,.975]).tolist())
    write_json(RUN/'scene_bootstrap.json',result)
    return result


def ray_scene_bootstrap(validation,primary):
    families = [f for f in ORDER if f in primary]
    scenes = sorted({f['scene'] for f in validation['frames']})
    mapping = {s:i for i,s in enumerate(scenes)}
    values = np.zeros((len(scenes),len(families),90),np.float64)
    for frame in validation['frames']:
        data = read(RUN/'ray'/f'frame_{frame["index"]:04d}.json')['counts']
        for j,family in enumerate(families):
            row = data[primary[family]['id']]
            values[mapping[frame['scene']],j] += np.r_[row['gt'],row['pred'],np.asarray(row['tp']).reshape(-1)]
    rng = np.random.default_rng(928)
    multiplicity = rng.multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),500)
    with threadpool_limits(8):
        summed = (multiplicity@values.reshape(len(scenes),-1)).reshape(500,len(families),90)
    gt,pred,tp = summed[:,:,:18],summed[:,:,18:36],summed[:,:,36:].reshape(500,len(families),3,18)
    union = gt[:,:,None,:]+pred[:,:,None,:]-tp
    iou = np.divide(tp,union,out=np.full_like(tp,np.nan),where=union>0)
    score = np.nanmean(iou[:,:,:,:17],axis=(-1,-2))*100
    pairs = [('P3','B2'),('P3','P4'),('P3','P3_oneway'),('B2','B0')]
    result = dict(repetitions=500,scenes=len(scenes),comparisons={})
    for a,b in pairs:
        delta = score[:,families.index(a)]-score[:,families.index(b)]
        result['comparisons'][a+'__minus__'+b] = np.quantile(delta,[.025,.975]).tolist()
    write_json(RUN/'ray_scene_bootstrap.json',result)
    return result


def export_per_class(validation,primary):
    with np.load(RUN/'validation/per_frame_counts.npz') as z:
        total = z['confusion'].sum(0)
    names = list(validation['operating_points'])
    base = total[names.index('B0')]
    fields = ['method','region','class','TP','FP','FN','IoU','delta_IoU_vs_B0']
    with (REPORT/'per_class_metrics.csv').open('w',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        for family in ORDER:
            cc = total[names.index(primary[family]['id'])]
            for r,label in [(0,'camera'),(1,'camera_complement'),(None,'all_valid')]:
                c = cc.sum(0) if r is None else cc[r]
                b = base.sum(0) if r is None else base[r]
                iou,base_iou = metrics(c)['per_class_IoU'],metrics(b)['per_class_IoU']
                for k,name in enumerate(CLASSES):
                    writer.writerow(dict(method=family,region=label,**{'class':name},TP=c[k,k],
                        FP=c[:,k].sum()-c[k,k],FN=c[k].sum()-c[k,k],IoU=iou[k],delta_IoU_vs_B0=iou[k]-base_iou[k]))


def savefig(fig,name):
    fig.tight_layout()
    fig.savefig(REPORT/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(REPORT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def plots(validation,primary,benchmark):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(12,5.1))
    ax,zoom = axes
    for family in ['B1','B2','B3','B3_class','P2','P3','P4','P3_oneway']:
        points = [p for p in validation['operating_points'].values() if p['family'] == family and p['seed'] in [None,928]]
        points.sort(key=lambda p:p['validation']['camera_complement']['empty_FPR'])
        x = [p['validation']['camera_complement']['empty_FPR'] for p in points]
        y = [p['validation']['camera']['mIoU'] for p in points]
        line, = ax.plot(x,y,'o-',label=ENGLISH[family],lw=1.5,ms=4,alpha=.9)
        point = primary[family]['validation']
        px,py = point['camera_complement']['empty_FPR'],point['camera']['mIoU']
        zoom.scatter(px,py,s=42,color=line.get_color())
        if family in ['B1','B2','P3','P4']:
            dy = -14 if family == 'P3' else 8
            zoom.annotate(family,(px,py),xytext=(5,dy),textcoords='offset points',fontsize=10)
    base = primary['B0']['validation']
    ax.scatter(base['camera_complement']['empty_FPR'],base['camera']['mIoU'],marker='*',s=180,c='black',label='Existing bias',zorder=10)
    zoom.scatter(base['camera_complement']['empty_FPR'],base['camera']['mIoU'],marker='*',s=160,c='black',zorder=10)
    zoom.annotate('B0',(base['camera_complement']['empty_FPR'],base['camera']['mIoU']),xytext=(7,-13),textcoords='offset points')
    ax.set(xlabel='Empty false occupancy in camera-mask complement (%)',ylabel='Camera mIoU (%)',
        title='All training-selected operating points')
    ax.grid(alpha=.2)
    zoom.set(xlim=(10.5,11.75),ylim=(24.4,25.62),xlabel='Complement false occupancy (%)',ylabel='Camera mIoU (%)',title='Primary +0.5 pp budget: actual validation risk')
    zoom.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),fontsize=8,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.16))
    savefig(fig,'risk_tradeoff')
    fig,axes = plt.subplots(1,2,figsize=(11,4.3))
    for i,family in enumerate(['P2','P3','P4','P3_oneway']):
        points = [p for p in validation['operating_points'].values() if p['family'] == family and p['budget'] == PRIMARY_BUDGET]
        points.sort(key=lambda p:p['seed'])
        for ax,region,metric in [(axes[0],'camera','mIoU'),(axes[1],'camera_complement','empty_FPR')]:
            values = [p['validation'][region][metric] for p in points]
            ax.scatter(np.full(len(values),i)+np.linspace(-.07,.07,len(values)),values,s=35)
            ax.plot([i-.15,i+.15],[np.mean(values)]*2,color='black',lw=2)
            ax.set_xticks(range(4),['IoU','IoU + risk','Without support','One-way'],rotation=15)
            ax.grid(axis='y',alpha=.2)
    axes[0].set(ylabel='Camera mIoU (%)',title='Three module seeds; shared frozen backbone')
    axes[1].set(ylabel='Complement false occupancy (%)',title='Same training-selected +0.5 pp risk budget')
    savefig(fig,'module_seeds')
    fig,axes = plt.subplots(2,2,figsize=(9,8))
    raw = validation['raw_calibration']
    for r,region in enumerate(['camera','camera_complement']):
        for event,label in enumerate(['q: correct class','e: true empty']):
            ax = axes[r,event]
            for key,title in [('prob_support_seed928','With support'),('prob_no_support_seed928','Without support')]:
                values = np.asarray(raw[key])[r,3+event*45:3+(event+1)*45].reshape(15,3)
                use = values[:,0] >= max(1000,values[:,0].sum()*.0001)
                confidence = values[use,1]/values[use,0]
                frequency = values[use,2]/values[use,0]
                line, = ax.plot(confidence,frequency,'-',label=title)
                ax.scatter(confidence,frequency,s=12+200*values[use,0]/max(values[:,0].sum(),1),color=line.get_color())
            ax.plot([0,1],[0,1],'--',c='gray',lw=1)
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted probability',ylabel='Observed frequency',title=region+' / '+label)
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    savefig(fig,'probability_calibration')


def make_report(validation,primary,ray,benchmark,ap,bootstrap,ray_bootstrap):
    REPORT.mkdir(parents=True, exist_ok=True)
    protocol = read(RUN/'protocol.json')
    sample = read(RUN/'sample_manifest.json')
    selection = read(RUN/'operating_points.json')
    fit = read(RUN/'fit_results.json')
    p3_parameters = fit['models']['P3_seed928']['gate_parameters']+fit['models']['P3_seed928']['probability_parameters']
    b3_parameters = fit['models']['B3_seed928']['parameters']
    ref = primary['B0']['validation']['camera_complement']['empty_FPR']
    p3,p4,b2 = [primary[k]['validation'] for k in ['P3','P4','B2']]
    intro = (f'本次完成冻结 OPUS Semantic Gaussian 的验证：4096 个训练帧提取双向候选，'
        f'523 个场景拟合、175 个场景调参，在独立的 150 个场景、1024 个验证帧上评估。'
        '前景类别排序和主模型权重均固定，只调整 empty/前景决策。')
    lines = ['# 实验报告：Semantic Gaussian 的 empty 决策优化验证','',
        '日期：2026-10-17。状态：**冻结模型阶段已完成；未进行主模型联合训练。**','',intro,'',
        '对应[方法设计](../../../plans/semantic_gaussian_empty_decision_optimization.md)。所有模型、阈值和工作点均在训练数据内部确定；验证结果未用于回调方法。','',
        '## 先看本次结果','',
        f'- 完整 P3：camera mIoU **{p3["camera"]["mIoU"]:.4f}**，补集误占用率 **{p3["camera_complement"]["empty_FPR"]:.4f}%**。',
        f'- 类别 bias B2：camera mIoU **{b2["camera"]["mIoU"]:.4f}**，补集误占用率 **{b2["camera_complement"]["empty_FPR"]:.4f}%**。',
        f'- 去掉 Gaussian 支持的 P4：camera mIoU **{p4["camera"]["mIoU"]:.4f}**，补集误占用率 **{p4["camera_complement"]["empty_FPR"]:.4f}%**。','',
        '**这些数字来自同一个训练风险预算，并不自动构成验证集上的相同风险比较。** 下文同时报告实际风险、几何指标、随机种子和配对区间。','',
        ]
    if (RUN/'interpretation.md').exists():
        lines += [(RUN/'interpretation.md').read_text().strip(),'']
    lines += ['## 1. 本次究竟验证了什么？','',
        '| 项目 | 本次设置 |','|---|---|',
        '| 主模型 | 既有 5-epoch、seed 928 checkpoint；已含全局 learnable bias |',
        '| 训练候选 | 全部 4096 帧，原生 margin 绝对值不超过 2，覆盖 empty 和前景两侧 |',
        f'| 拟合与调参 | {sample["cohorts"]["fit"]["frames"]} 帧拟合；{sample["cohorts"]["tune"]["frames"]} 帧调参，按 scene 分组互斥 |',
        '| 拟合行数 | 400,000；按区域与 margin 符号采样，再做与标签无关的行数上限；逆概率加权 |',
        '| 概率/决策调参 | 200,000 抽样行选择网络；全部 994 个调参帧确定最终离散工作点 |',
        '| 网络随机种子 | 928、929、930；共享同一冻结主模型 |',
        f'| 局部网络参数 | B3 为 {b3_parameters:,}；P3 的概率头＋决策层合计 {p3_parameters:,}，规模相近 |',
        '| 验证 | 全部 1024 帧的完整体素网格；150 个独立验证场景 |',
        '| 推理输入 | margin、分数统计、前景首选类别、位置/距离、可选真实 Gaussian 支持 |',
        '| 训练监督 | 所有方法均可使用两区域标签；camera mask 只用于训练和评估分组 |',
        '| RayIoU | 11 个主要工作点，全部 1024 帧，官方 OPUS 实现与完整 scene origins，无 camera mask |','',
        '训练内拟合/调参场景参与过主模型训练，因此不是主模型完全未见的独立校准集。camera-mask 补集也不等于纯粹不可见空间。','',
        '### 1.1 如何避免把“减去更多 empty”误当作方法改进？','',
        '| 编号 | 含义 |','|---|---|']
    for family in ORDER:
        explanation = LABELS[family]
        if family == 'B3_class':
            explanation += '；额外强对照，使用 B3 的同一网络'
        lines.append(f'| {family} | {explanation} |')
    lines += ['',
        '概率头输出 $q$（当前前景类别正确）、$e$（真实 empty）和 $o$（其他前景类别），三者之和为 1。它是两层宽度 32 的 MLP；决策层为两层宽度 16 的 MLP。普通局部对照 B3 为两层宽度 40 的 MLP。','',
        'B2 按当前前景首选类别设置 17 个独立 margin 阈值，保留前景类别排序；这些阈值在训练内校准，未重新训练主模型分类器。B3_class 同样按前景首选类别设置阈值，但作用于 B3 的局部正确率分数。','',
        '本次 Gaussian 特有输入仅为实际前景核总支持的 log(1+S)，不包含全局 empty Gaussian，也未加入可选的有效贡献数量、opacity 或尺度统计。因此支持消融的结论限定于本次单一支持特征。','',
        'B3 使用二分类 NLL，目标为“当前前景首选是否正确”；P1 使用三分类 NLL，区分正确类别、真实 empty、其他前景。P2 的决策层优化 camera soft mIoU；P3/P4 再加入补集误占用约束。',
        'P2/P3/P4 在同一候选范围内都可以双向调整；P3_oneway 保留原本的全部前景预测，只允许新增前景。','',
        '### 1.2 工作点如何选择？','',
        '预先设置补集误占用允许增量为 **0、0.5、2、5 个百分点**，另设不加风险上限的参照；主比较采用 **+0.5 个百分点**。',
        'P3/P4/单向版本的约束训练预算固定为 +0.5 点；其他预算用同一网络在训练内重新校准阈值，未为每个预算另训一个网络。',
        f'调参集 B0 的补集误占用率为 **{selection["reference_out_fpr"]*100:.4f}%**。每种方法在相同上限内选择 camera mIoU 最高的工作点。全局 bias 使用 1/32 的步长搜索额外修正 −2 到 +2；类别 bias 使用密集风险乘子搜索。',
        '主预算下，B1 选出的额外全局 bias 为 **0.09375**。不限制风险时，训练内选择的 bias 为 **0.90625**，验证 camera mIoU 为 **25.8123**，但补集误占用率达到 **15.2751%**。因此主表比较的是同一风险预算下的表现，并非各方法只追求最高 camera 指标的排名。',
        '验证风险匹配容差在评价前约定为 **0.25 个百分点**。超出容差时，只能称为同一训练预算下的比较，不能声称严格相同风险。','',
        METHOD_DETAILS.strip(),'',
        '## 2. 主要结果：实际风险与语义收益一起看','',
        '除 P1_q05 的固定 0.5 诊断门槛外，下表均为训练内选定的 +0.5 个百分点工作点；神经模块此表使用 seed 928。mIoU/IoU 使用百分数表示，差值使用百分点。','',
        '| 方法 | camera mIoU | 全空间 mIoU | 补集 mIoU | 补集误占用率 | 相对 B0 的误占用变化 |',
        '|---|---:|---:|---:|---:|---:|']
    for family in ORDER:
        p = primary[family]
        v = p['validation']
        lines.append(f'| {family} | {v["camera"]["mIoU"]:.4f} | {v["all_valid"]["mIoU"]:.4f} | {v["camera_complement"]["mIoU"]:.4f} | {v["camera_complement"]["empty_FPR"]:.4f}% | {v["camera_complement"]["empty_FPR"]-ref:+.4f} |')
    lines += ['',
        '误占用率的分母是该区域全部真实 empty 体素，不是翻转体素数量。若验证增量大于 +0.5 个百分点，表示训练预算未直接迁移为验证约束。','',
        '![训练内固定工作点的验证收益与误占用关系](risk_tradeoff.png)','',
        '图中每个点都在查看验证结果之前选定；连线只帮助观察，不是使用验证集搜索出的最优曲线。','',
        '### 2.1 主要对照的配对不确定性','',
        '采用 500 次场景级配对 bootstrap。区间条件于既有主模型、已拟合模块及固定工作点，未重新拟合；不包含主模型训练不确定性。','',
        '| 比较（前者减后者） | camera mIoU 差 | 差值 95% 区间 | 补集误占用差 | 风险是否在 0.25 点容差内 |',
        '|---|---:|---:|---:|---|']
    for other in ['B1','B2','B3','B3_class','P2','P4','P3_oneway']:
        base = primary[other]['validation']
        delta = p3['camera']['mIoU']-base['camera']['mIoU']
        risk_delta = p3['camera_complement']['empty_FPR']-base['camera_complement']['empty_FPR']
        interval = bootstrap['comparisons']['P3__minus__'+other]['camera_mIoU_delta_ci95']
        lines.append(f'| P3 − {other} | {delta:+.4f} | [{interval[0]:+.4f}, {interval[1]:+.4f}] | {risk_delta:+.4f} | {"是" if abs(risk_delta)<=.25 else "否"} |')
    lines += ['',
        '## 3. 几何指标与官方 RayIoU','',
        '| 方法 | camera 二值 IoU | 全空间二值 IoU | RayIoU | @1m | @2m | @4m |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for family in ORDER:
        p = primary[family]
        r = ray['models'][p['id']]
        lines.append(f'| {family} | {p["validation"]["camera"]["IoU"]:.4f} | {p["validation"]["all_valid"]["IoU"]:.4f} | {r["RayIoU"]:.4f} | {r["RayIoU@1"]:.4f} | {r["RayIoU@2"]:.4f} | {r["RayIoU@4"]:.4f} |')
    lines += ['',
        'RayIoU 使用完整验证 scene 的 ego origins 和官方语义/深度匹配规则，不使用 camera mask。逐项计数与官方聚合核验；B0 的完整 RayIoU 计数还与此前独立评估完全一致。RayIoU 含语义匹配，不能单独解释为几何指标。','',
        '场景级配对 RayIoU 差值（500 次 bootstrap，固定工作点）：','']
    for a,b in [('P3','B2'),('P3','P4'),('P3','P3_oneway'),('B2','B0')]:
        delta = ray['models'][primary[a]['id']]['RayIoU']-ray['models'][primary[b]['id']]['RayIoU']
        interval = ray_bootstrap['comparisons'][a+'__minus__'+b]
        lines.append(f'- {a} − {b}：{delta:+.4f} 个百分点，95% 区间 [{interval[0]:+.4f}, {interval[1]:+.4f}]。')
    lines += ['',
        '## 4. 三个模块随机种子：支持信息和双向纠错是否稳定有用？','',
        '下表为 928、929、930 三个模块种子的均值 ± 样本标准差；主模型只有一个 seed，不能视为三个主模型独立训练。','',
        '| 方法 | camera mIoU | 补集误占用率 | 全空间 mIoU |',
        '|---|---:|---:|---:|']
    for family in ['B3','P1','P2','P3','P4','P3_oneway']:
        points = [p for p in validation['operating_points'].values() if p['family'] == family and p['budget'] == PRIMARY_BUDGET]
        values = [[p['validation'][r][m] for p in points] for r,m in [('camera','mIoU'),('camera_complement','empty_FPR'),('all_valid','mIoU')]]
        strings = [f'{np.mean(v):.4f} ± {np.std(v,ddof=1):.4f}' for v in values]
        lines.append('| '+family+' | '+' | '.join(strings)+' |')
    lines += ['',
        '![冻结主模型下的三个模块种子](module_seeds.png)','',
        'P3 与 P4 分别重新拟合了概率头和决策层，因此该消融衡量“可用支持信息的整体方法增量”，不是对同一已训练网络遮蔽一个输入的因果干预。','',
        '## 5. 概率是否可靠？','',
        'NLL、三分类 Brier 和 ECE 在完整候选体素上统计；下表为 seed 928。AP 使用与标签无关的候选抽样及逆纳入概率权重，点估计按原始分数排序，未分箱近似。','',
        'NLL 和 Brier 越低越好；ECE 使用 15 个等宽概率分箱，衡量预测概率与实际频率的差距，越低越好；AP 衡量正确事件的排序质量，越高越好。AP 提高不等于 mIoU 提高。','',
        '| 输入 | 区域 | NLL | Brier | q 的 ECE | e 的 ECE | 正确前景 AP | 该事件自然比例 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,label in [('prob_support_seed928','含支持'),('prob_no_support_seed928','不含支持')]:
        for region,region_label in [('camera','camera'),('camera_complement','补集')]:
            c = validation['calibration'][name][region]
            a = ap['models'][name][region]
            lines.append(f'| {label} | {region_label} | {c["NLL"]:.4f} | {c["Brier"]:.4f} | {c["q_ECE"]*100:.2f}% | {c["e_ECE"]*100:.2f}% | {a["q_AP"]*100:.2f}% | {a["q_prevalence"]*100:.2f}% |')
    lines += ['',
        f'AP 共使用 **{ap["rows"]:,} 个候选样本**。本次候选是 margin 绝对值不超过 2 的双向邻域，与旧报告仅额外减去 1 会翻转的候选不同，AP 数值不能直接作前后提升比较。低 ECE 也不等于具有足够排序能力或能提高 mIoU。','',
        '![三结果概率的分区域校准图](probability_calibration.png)','',
        '图中圆点面积随分箱体素比例增加；为避免极少数样本形成尖峰，隐藏少于该区域候选 0.01% 或少于 1000 个体素的分箱。上表的 NLL、Brier 与 ECE 仍使用全部候选体素。','',
        '## 6. 修改后的体素究竟发生了什么？','',
        '以下展示 seed 928 的主要方法。正确前景新增与撤回均按原生 B0 比较；“误占用撤回”只统计真实 empty。完整六类变化计数见结构化结果。','',
        '| 方法 | 区域 | 正确前景新增 | 真实 empty 新增误占用 | 误占用撤回 | 正确前景被撤回 |',
        '|---|---|---:|---:|---:|---:|']
    for family in ['B1','B2','B3_class','P2','P3','P4','P3_oneway']:
        events = np.asarray(primary[family]['transition_counts'])
        for r,label in [(0,'camera'),(1,'补集')]:
            e = events[r]
            lines.append(f'| {family} | {label} | {e[0]:,} | {e[2]:,} | {e[5]:,} | {e[3]:,} |')
    lines += ['',
        '即使新增误占用多于正确救回，某类别 IoU 也可能提高。以只新增前景的操作为例：若原 IoU 为 $J=T/U$，新增正确预测为 $r$、错误预测为 $f$，则：','',
        '$$',
        "J'=\\frac{T+r}{U+f},\\qquad J'>J\\Longleftrightarrow\\frac{r}{f}>J\\quad(f>0).",
        '$$','',
        '这里 $f$ 还包括新增的其他前景错类，不能只统计真实 empty。双向纠错还包含撤回事件，实际结果均从完整混淆矩阵计算。','',
        '## 7. 完整推理开销','',
        '在同一张 RTX 4090、batch size 1 上，使用 8 个输入，每输入每方法预热 2 次、计时 5 次，随机化方法顺序。所有方法均计入主模型前向、候选特征和自身决策；统一排除数据读取和 H2D。','',
        '**含 Gaussian 支持的原型使用一次额外原生聚合；尚未融合 CUDA。** 下表是当前可运行原型的实际开销，不能直接当作融合后的最终论文 FPS。','',
        '| 方法 | 平均延迟（ms） | FPS | 峰值 allocated（MiB） |',
        '|---|---:|---:|---:|']
    for family in ORDER:
        if family == 'P1_q05':
            continue
        b = benchmark['models'][primary[family]['id']]
        lines.append(f'| {family} | {b["mean_latency_ms"]:.3f} | {b["FPS"]:.2f} | {b["peak_allocated_MiB"]:.1f} |')
    lines += ['',
        'P1_q05 与 P1 共享网络与计算流程，仅阈值不同，未重复计时。小样本计时受桌面 GPU 负载及频率影响，完整逐次延迟保存在原始结果中。','',
        '## 8. 本次证据的适用范围','',
        '- 结论限定于同一 5-epoch mini checkpoint、固定前景排序和 margin ±2 邻域；不能推广为所有 semantic Gaussian 模型的规律。',
        '- 概率训练先做 160-epoch 训练内试跑，因调参 NLL 仍下降，在查看验证结果前统一扩展到 800 epoch；决策训练上限为 600 步。原试跑和最终训练配置均已保存。',
        '- 本次所有概率头选中了 800-epoch 边界；没有证明其已达到最充分拟合，也没有证明所用网络/优化器构成理论方法的能力上限。',
        '- 补集标签用于训练风险建模及所有公平对照。这是相同监督下的分组风险控制实验，不是仅 camera 监督下的无额外标签泛化证明。',
        '- 工作点使用离散阈值网格和密集风险乘子搜索。类别约束优化不保证获得所有离散组合的全局最优解。',
        '- mIoU 局部收益推导是期望计数代理与固定前景下的分析，不是多类别模型的全局最优性定理。',
        '- 尚未执行主模型联合训练；是否值得推进，应以强对照下的实际收益、Gaussian 支持增量和效率代价共同判断。','',
        '## 9. 复现与原始产物','',
        '完成性检查已通过：全部工作点保持真值总体计数；六类新增/撤回事件与混淆矩阵变化一致；单向策略的撤回数严格为零。另在 2 帧中将真值全部替换为 empty、反转可用真值 mask，渲染分数、支持、特征与 18 组决策输出均逐值不变。新增与既有相关测试共 11 项通过。','',
        '依次运行以下入口；已有阶段产物可复用，重新选择工作点应作为新实验，不能在查看验证结果后覆盖本次冻结选择。','',
        '```bash',
        'python3.10 tools/empty_decision_optimization.py extract',
        'python3.10 tools/fit_empty_decision_optimization.py',
        'python3.10 tools/eval_empty_decision_optimization.py tune',
        'python3.10 tools/select_empty_decision_optimization.py',
        'python3.10 tools/eval_empty_decision_optimization.py validate',
        'python3.10 tools/audit_empty_decision_optimization.py ray',
        'python3.10 tools/audit_empty_decision_optimization.py benchmark',
        'python3.10 tools/report_empty_decision_optimization.py',
        '```','',
        '- [冻结工作点与训练内选择结果](../../../work_dirs/empty_decision_optimization/operating_points.json)。',
        '- [全部 59 个验证工作点、校准统计及逐帧索引](../../../work_dirs/empty_decision_optimization/validation/summary.json)。',
        '- [官方 RayIoU 及计数](../../../work_dirs/empty_decision_optimization/ray/summary.json)。',
        '- [完整推理计时](../../../work_dirs/empty_decision_optimization/benchmark.json)。',
        '- [场景级配对区间](../../../work_dirs/empty_decision_optimization/scene_bootstrap.json)。',
        '- [全部工作点的 CSV 汇总](all_operating_points.csv)。',
        '- [主要方法逐类别 TP/FP/FN 与 IoU](per_class_metrics.csv)。',
        '- [完成性与来源审计](../../../work_dirs/empty_decision_optimization/completion_audit.json)。','',
        f'Checkpoint SHA256：`{protocol["checkpoint_sha256"]}`。','']
    path = ROOT/'analysis/empty_decision/optimization/empty_decision_optimization_results.md'
    path.write_text('\n'.join(lines))
    fields = ['id','family','seed','budget','camera_mIoU','complement_mIoU','all_mIoU','complement_FPR','camera_IoU','all_IoU','tuning_camera_mIoU','tuning_complement_FPR']
    with (REPORT/'all_operating_points.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for p in validation['operating_points'].values():
            v,t = p['validation'],p['tuning']
            writer.writerow(dict(id=p['id'],family=p['family'],seed=p['seed'],budget=p['budget'],
                camera_mIoU=v['camera']['mIoU'],complement_mIoU=v['camera_complement']['mIoU'],all_mIoU=v['all_valid']['mIoU'],
                complement_FPR=v['camera_complement']['empty_FPR'],camera_IoU=v['camera']['IoU'],all_IoU=v['all_valid']['IoU'],
                tuning_camera_mIoU=t['camera']['mIoU'],tuning_complement_FPR=t['camera_complement']['empty_FPR']))
    return path


def main():
    validation = read(RUN/'validation/summary.json')
    ray = read(RUN/'ray/summary.json')
    benchmark = read(RUN/'benchmark.json')
    assert validation['samples'] == ray['samples'] == 1024
    REPORT.mkdir(parents=True, exist_ok=True)
    primary = primary_points(validation)
    ap = probability_ap(validation)
    bootstrap = scene_bootstrap(validation,primary)
    ray_bootstrap = ray_scene_bootstrap(validation,primary)
    export_per_class(validation,primary)
    plots(validation,primary,benchmark)
    path = make_report(validation,primary,ray,benchmark,ap,bootstrap,ray_bootstrap)
    print('Wrote',path,flush=True)


if __name__ == '__main__':
    main()
