"""Readable class-bias results, paired intervals, plots and CSV exports."""
from __future__ import annotations
import csv
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.learnable_class_bias import RUN,OUT,read,write_json,load_samples
from tools.eval_empty_decision_optimization import metrics
from tools.diagnose_opus_v1_gaussian import CLASSES

MAIN=['C0','B2_shift_historical']+[f'C{k}_eps0.005' for k in range(1,6)]
NAMES={'C0':'原生全局 bias','B2_shift_historical':'历史 B2＋再校准',
       'C1_eps0.005':'同数据 B2＋再校准','C2_eps0.005':'交叉熵学习＋再校准',
       'C3_eps0.005':'交叉熵＋收缩＋再校准','C4_eps0.005':'语义指标优化',
       'C5_eps0.005':'再加入全空间保护'}
SHORT={name:(name.split('_')[0] if name!='B2_shift_historical' else 'Historical B2') for name in MAIN}
COMPARISONS=[('C1_eps0.005','B2_shift_historical'),('C2_eps0.005','C1_eps0.005'),
    ('C3_eps0.005','C2_eps0.005'),('C4_eps0.005','C1_eps0.005'),
    ('C5_eps0.005','C1_eps0.005'),('C5_eps0.005','B2_shift_historical'),('C5_eps0.005','C4_eps0.005')]
COLORS={'C1':'#4477aa','C2':'#ee7733','C3':'#aa4499','C4':'#aa7733','C5':'#228833'}


def proxy_diagnostics(v):
    """Reproduce the numerical proxy check directly from the fitting samples."""
    from mmengine import Config
    from tools.eval_empty_decision_optimization import region_metrics
    data=load_samples('fit');results={}
    binned=-2+(np.minimum(((data['X'][:,0]+2)*256).astype(int),1023)+.5)/256
    code=data['region'].astype(np.int64)*306+data['gt'].astype(np.int64)*17+data['fg']
    for name in ['C1_eps0.005','C4_eps0.005','C5_eps0.005']:
        a=np.asarray(v['operating_points'][name]['a']);results[name]={}
        for label,margin in [('exact',data['X'][:,0]),('binned',binned)]:
            gate=expit((margin+a[data['fg']])/.05);weight=data['weight']
            occupied=np.bincount(code,weights=weight*gate,minlength=612).reshape(2,18,17)
            counts=data['fixed_confusion'].astype(float).copy();counts[:,:,:17]+=occupied
            counts[:,:,17]+=np.bincount(data['region'].astype(int)*18+data['gt'],weights=weight*(1-gate),minlength=36).reshape(2,18)
            results[name][label]=region_metrics(counts)
    write_json(RUN/'proxy_diagnostics.json',results)
    config=Config.fromfile(read(RUN/'protocol.json')['config'])
    write_json(RUN/'main_model_loss_context.json',dict(resolved_loss=list(map(dict,config.loss.loss_cfgs)),
        loss_input_mapping=dict(config.loss_input_convertion),
        interpretation='original voxel loss: camera mask, weighted CE + Lovasz; C2: IPW unweighted full18 CE on both-region candidates'))


def paired(v):
    names=list(v['operating_points']);scenes=np.asarray([f['scene'] for f in v['frames']])
    unique,inverse=np.unique(scenes,return_inverse=True)
    with np.load(RUN/'validation/per_frame_counts.npz') as z:c=z['confusion'][:,[names.index(k) for k in MAIN]]
    summed=np.zeros((len(unique),len(MAIN),2,18,18),np.float64);np.add.at(summed,inverse,c)
    rng=np.random.default_rng(928);multiplicity=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),500)
    with threadpool_limits(8):sampled=(multiplicity@summed.reshape(len(unique),-1)).reshape(500,len(MAIN),2,18,18)
    values=dict(camera=metrics(sampled[:,:,0])['mIoU'],all=metrics(sampled.sum(2))['mIoU'],risk=metrics(sampled[:,:,1])['empty_FPR'])
    result=dict(repetitions=500,scenes=len(unique),scope='fixed fitted parameters and calibration; no refitting',comparisons={})
    for a,b in COMPARISONS:
        ia,ib=MAIN.index(a),MAIN.index(b)
        result['comparisons'][a+'__minus__'+b]={k:np.quantile(x[:,ia]-x[:,ib],[.025,.975]).tolist() for k,x in values.items()}
    rays=np.zeros((len(unique),len(MAIN),90),np.float64);lookup={s:i for i,s in enumerate(unique)}
    for f in v['frames']:
        counts=read(RUN/'ray'/f'frame_{f["index"]:04d}.json')['counts']
        for i,name in enumerate(MAIN):
            row=counts[name];rays[lookup[f['scene']],i]+=np.r_[row['gt'],row['pred'],np.asarray(row['tp']).reshape(-1)]
    with threadpool_limits(8):sampled=(multiplicity@rays.reshape(len(unique),-1)).reshape(500,len(MAIN),90)
    gt,pred,tp=sampled[:,:,:18],sampled[:,:,18:36],sampled[:,:,36:].reshape(500,len(MAIN),3,18)
    union=gt[:,:,None,:]+pred[:,:,None,:]-tp
    ray=np.nanmean(np.divide(tp,union,out=np.full_like(tp,np.nan),where=union>0)[:,:,:,:17],axis=(-1,-2))*100
    for a,b in COMPARISONS:
        result['comparisons'][a+'__minus__'+b]['ray']=np.quantile(ray[:,MAIN.index(a)]-ray[:,MAIN.index(b)],[.025,.975]).tolist()
    write_json(RUN/'paired_scene_intervals.json',result);return result


def stability(v):
    frozen=read(RUN/'stability/selection.json');result={}
    for family in ['C2','C3','C5']:
        points=[p for p in v['operating_points'].values() if p['family']=='bootstrap_'+family]
        raw=np.array([m['a'] for m in frozen['models'] if m['family']==family]);adjusted=np.array([p['a'] for p in points])
        entry=dict(replicates=len(points),fallbacks=sum(p['fallback'] for p in points),
            raw_mean_class_std=float(raw.std(0,ddof=1).mean()),
            raw_centered_mean_class_std=float((raw-raw.mean(-1,keepdims=True)).std(0,ddof=1).mean()),
            calibrated_mean_class_std=float(adjusted.std(0,ddof=1).mean()))
        for key,region,metric in [('camera','camera','mIoU'),('all','all_valid','mIoU'),('risk','camera_complement','empty_FPR')]:
            values=np.array([p['validation'][region][metric] for p in points])
            entry[key]=dict(mean=float(values.mean()),std=float(values.std(ddof=1)),values=values.tolist())
        result[family]=entry
    write_json(RUN/'stability/results.json',result);return result


def exports(v):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'operating_points.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','family','source','shift','fallback','budget','camera_mIoU','all_mIoU','complement_FPR']);writer.writeheader()
        for p in v['operating_points'].values():
            writer.writerow({**{k:p.get(k) for k in ['id','family','source','shift','fallback','budget']},
                'camera_mIoU':p['validation']['camera']['mIoU'],'all_mIoU':p['validation']['all_valid']['mIoU'],
                'complement_FPR':p['validation']['camera_complement']['empty_FPR']})
    with np.load(RUN/'validation/per_frame_counts.npz') as z:cc=z['confusion'].sum(0)
    names=list(v['operating_points'])
    with (OUT/'per_class.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','region','class','TP','FP','FN','IoU']);writer.writeheader()
        for name in MAIN:
            c=cc[names.index(name)]
            for region,counts in [('camera',c[0]),('complement',c[1]),('all',c.sum(0))]:
                for k,label in enumerate(CLASSES):writer.writerow(dict(method=name,region=region,**{'class':label},TP=int(counts[k,k]),
                    FP=int(counts[:,k].sum()-counts[k,k]),FN=int(counts[k].sum()-counts[k,k]),IoU=float(metrics(counts)['per_class_IoU'][k])))
    with (OUT/'class_biases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','class','total_bias','additional_bias','class_deviation']);writer.writeheader()
        for name in MAIN[2:]:
            a=np.asarray(v['operating_points'][name]['a']);b0=read(RUN/'protocol.json')['original_bias']
            for k,label in enumerate(CLASSES):writer.writerow(dict(method=name,**{'class':label},total_bias=b0+a[k],additional_bias=a[k],class_deviation=a[k]-a.mean()))


def figures(v,stable):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(13,5))
    x=np.arange(17)
    for name in ['C1_eps0.005','C2_eps0.005','C3_eps0.005','C5_eps0.005']:
        ax.plot(x,v['operating_points'][name]['a'],'o--' if name.startswith('C3') else 'o-',label=SHORT[name],ms=4,color=COLORS[SHORT[name]])
    ax.axhline(0,color='black',lw=1,alpha=.5);ax.set_xticks(x,CLASSES,rotation=40,ha='right',fontsize=8)
    ax.set(ylabel='Additional empty bias relative to native model',title='Training-selected primary rules; same fixed foreground winner');ax.legend(ncol=4);ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/('class_biases.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4));colors=COLORS
    for family in ['C1','C4','C5']:
        points=[p for p in v['operating_points'].values() if p['family']==family]
        points.sort(key=lambda p:p['validation']['camera_complement']['empty_FPR'])
        risk=[p['validation']['camera_complement']['empty_FPR'] for p in points]
        for ax,region in [(axes[0],'camera'),(axes[1],'all_valid')]:
            ax.plot(risk,[p['validation'][region]['mIoU'] for p in points],'o-',label=family,color=colors[family])
            primary=v['operating_points'][family+'_eps0.005'];ax.scatter(primary['validation']['camera_complement']['empty_FPR'],primary['validation'][region]['mIoU'],s=100,facecolors='none',edgecolors=colors[family])
    axes[0].set(ylabel='Camera mIoU (%)',title='Camera benefit');axes[1].set(ylabel='All-space mIoU (%)',title='All-space semantic quality')
    for ax in axes:ax.set_xlabel('Complement empty false occupancy (%)');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/('risk_tradeoff.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,key,label in [(axes[0],'camera','Camera mIoU (%)'),(axes[1],'all','All-space mIoU (%)')]:
        for i,family in enumerate(['C2','C3','C5']):
            values=np.array(stable[family][key]['values']);ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=23,color=COLORS[family])
            ax.plot([i-.18,i+.18],[values.mean()]*2,color='black')
        ax.set_xticks(range(3),['C2','C3','C5']);ax.set_ylabel(label);ax.grid(alpha=.2)
        ax.set_title('10 fitting-scene resamples; shared backbone')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/('scene_stability.'+ext),dpi=180)
    plt.close(fig)


def interval(pair,key):
    lo,hi=pair[key];return f'[{lo:+.4f}, {hi:+.4f}]'


def report(v,ci,stable):
    p=read(RUN/'protocol.json');selection=read(RUN/'operating_points.json');fit=read(RUN/'fit_summary.json')
    ce=read(RUN/'ce_shrink_selection.json');rays=read(RUN/'ray/summary.json')['models'];bench=read(RUN/'benchmark.json')['models']
    points=v['operating_points'];c1,c2,c3,c4,c5=[points[f'C{k}_eps0.005'] for k in range(1,6)]
    def value(q,region='camera',key='mIoU'):return q['validation'][region][key]
    def delta(a,b,region='camera',key='mIoU'):return value(a,region,key)-value(b,region,key)
    pair=ci['comparisons']['C5_eps0.005__minus__C1_eps0.005']
    lines=['# 可学习类别 Empty Bias：实验结果与解释','',
        '日期：2026-10-18。状态：**C0—C5 冻结模型阶段完成；未进行主模型联合训练。**','',
        '设计见[实验设计](learnable_class_bias_experiment_design.md)，数学定义见[方法设计](../../../plans/semantic_gaussian_learnable_class_bias_design.md)。本轮使用同一个 5-epoch 主模型、4096 个训练帧与 1024 个验证帧。','',
        '## 先看结论','',
        '**本轮没有得到 camera mIoU 稳定超过类别阈值搜索的学习方案；得到的是不同的空间质量折中。** 17 个参数能够通过梯度正常学习，但参数是否可学习，与学习目标是否适合当前评价指标是两个问题。','',
        f'- **交叉熵版本过于保守。** C2 再校准后的 camera mIoU 为 {value(c2):.4f}，同数据 C1 为 {value(c1):.4f}；补集 FPR 分别为 {value(c2,"camera_complement","empty_FPR"):.4f}% 和 {value(c1,"camera_complement","empty_FPR"):.4f}%，这不是等风险比较。',
        f'- **改用指标目标后显著接近 C1，但未超过它的 camera 指标。** 训练内预选的 C5 相对 C1：camera {delta(c5,c1):+.4f} 点，95% 区间 {interval(pair,"camera")}；全空间 {delta(c5,c1,"all_valid"):+.4f} 点，区间 {interval(pair,"all")}；补集 FPR {delta(c5,c1,"camera_complement","empty_FPR"):+.4f} 点。',
        f'- **本轮未观察到收缩带来的实用增益。** 训练内交叉熵选择了最小收缩系数 {fit["shrink"]:g}，C3 与 C2 的性能及重采样波动接近。它只说明本次选择方式下的结果，不能排除其他任务目标下的正则化价值。',
        f'- **额外全空间项的独立增量很小。** C5 相对 C4 的 camera 变化 {delta(c5,c4):+.4f} 点、全空间变化 {delta(c5,c4,"all_valid"):+.4f} 点。主要空间折中已在 C4 出现，不能把它全部归功于 C5。','',
        '**读表规则：** 指标用百分数，差值用百分点。C1 与学习方法共享拟合数据、区域标签和最终统一平移机会；历史 B2 单独列出。所有工作点在本轮方法评价前冻结，C5 主方案没有按验证结果改选。','',
        '## 1. 主工作点：收益和代价一起看','',
        '| 方法 | camera mIoU | 全空间 mIoU | 补集 FPR | camera 二值 IoU | RayIoU | FPS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in MAIN:
        q=points[name]
        lines.append(f'| {SHORT[name]}：{NAMES[name]} | {value(q):.4f} | {value(q,"all_valid"):.4f} | {value(q,"camera_complement","empty_FPR"):.4f}% | {value(q,"camera","IoU"):.4f} | {rays[name]["RayIoU"]:.4f} | {bench[name]["FPS"]:.2f} |')
    lines += ['', 'C2／C3 的表中结果已包含训练内选择的统一再校准。C1—C5 采用相同的类别查表推理实现，没有额外特征网络或 Gaussian 支持聚合。FPS 是本轮完整前向实测；这些方法之间的小幅计时差不构成结构性速度差异。','',
        '### 1.1 配对区间','',
        '150 个验证场景，500 次场景级配对 bootstrap。区间条件于固定参数和工作点，不包含重新训练、重新选择或主模型种子变化。','',
        '| 比较 | camera 差值／95% 区间 | 全空间差值／95% 区间 | 补集 FPR 差值 |',
        '|---|---|---|---:|']
    for a,b in COMPARISONS:
        row=ci['comparisons'][a+'__minus__'+b]
        lines.append(f'| {SHORT[a]} − {SHORT[b]} | {delta(points[a],points[b]):+.4f} / {interval(row,"camera")} | {delta(points[a],points[b],"all_valid"):+.4f} / {interval(row,"all")} | {delta(points[a],points[b],"camera_complement","empty_FPR"):+.4f} |')
    lines += ['', 'RayIoU 同时涉及语义和距离，不能当作纯几何指标。主要比较如下：','',
        '| 比较 | RayIoU 差 | 95% 区间 |','|---|---:|---|']
    for a,b in [('C5_eps0.005','C1_eps0.005'),('C5_eps0.005','B2_shift_historical'),('C5_eps0.005','C4_eps0.005')]:
        row=ci['comparisons'][a+'__minus__'+b];lines.append(f'| {SHORT[a]} − {SHORT[b]} | {rays[a]["RayIoU"]-rays[b]["RayIoU"]:+.4f} | {interval(row,"ray")} |')
    lines += ['', '## 2. 为什么交叉熵学习明显退化？','',
        '### 2.1 先看没有再校准的输出','',
        '| 版本 | camera mIoU | 全空间 mIoU | 补集 FPR | 训练内选中的统一平移 |',
        '|---|---:|---:|---:|---:|']
    for name in ['C2_raw','C2_eps0.005','C3_raw','C3_eps0.005']:
        label=name[:2]+('，未再校准' if name.endswith('_raw') else '，已再校准')
        q=points[name];lines.append(f'| {label} | {value(q):.4f} | {value(q,"all_valid"):.4f} | {value(q,"camera_complement","empty_FPR"):.4f}% | {q["shift"]:+.5f} |')
    c2model=read(RUN/'models/C2.json');a=np.asarray(c2model['a'])
    lines += ['',f'C2 的 {int(np.sum(a<=-2+1e-7))}/17 个类别到达拟合下界 −2，额外偏置范围为 [{a.min():.4f}, {a.max():.4f}]。负额外偏置表示将原模型的 empty 分数提高，撤回更多前景。最终平移选择到达预设 +1 边界；本轮未在看到验证结果后扩展范围。','',
        '因此，不能仅凭表中 camera 差距断言“梯度学习表达能力更差”。其风险工作点差异很大，并且统一再校准范围有限；可以直接判断的是：当前普通交叉熵目标没有自动找到高 camera mIoU 的类别策略。','',
        '### 2.2 这是损失与训练区域的变化，不是原模型训练的复刻','',
        '本轮 C2 是两个区域候选上的逆概率加权、无额外类别权重的完整 18 类交叉熵。原 checkpoint 的体素训练使用 camera mask、类别加权交叉熵与 Lovász 项。两者的区域分布和目标均不同。C2 是最简学习基线，不能把它的退化归因于反向传播本身，更不能推断 OPUS set 分类损失或 Gaussian 表征必然有害。','',
        '仅调整 empty、固定前景分数时，单个候选体素的交叉熵梯度为（此处不含采样权重和正则）：','',
        '$$',r'\frac{\partial\mathcal L_{\mathrm{CE}}(x)}{\partial a_{c^*(x)}}',
        r'=\mathbb I[y(x)=E]-\widetilde p_E(x).','$$','',
        '真实类别是前景时，无论它是否等于当前首选类别，这个直接偏置梯度都倾向于降低 empty。交叉熵比较的是 empty 与所有前景概率质量，而硬预测比较的是 empty 与单个前景首选。因此，它没有直接表达“救回正确首选”与“释放其他前景错类”在 mIoU 上的不同代价；混合两个区域的训练又改变了目标分布。这里解释的是本轮冻结偏置学习的目标差异，不是主模型损失的普遍结论。','',
        '本轮确实补提取了 4096 个训练帧的分类充分统计量，并逐帧核对直接 18 类交叉熵。所有六个凸 CE 拟合的投影梯度最大值低于 0.00001，支持“预设边界内优化已接近受约束驻点”，而非简单的未训练完成。','',
        '## 3. 收缩与全空间保护分别贡献了什么？','',
        '### 3.1 收缩：训练内选择几乎关闭了正则','',
        '| 收缩系数 | 调参候选交叉熵 | 最大投影梯度 |','|---:|---:|---:|']
    for row in ce['candidates']:lines.append(f'| {row["shrink"]:g} | {row["tune_candidate_ce"]:.8f} | {row["projected_gradient_max"]:.3g} |')
    config=read(RUN/'models'/f'{c5["source"]}.json')
    lines += ['', '更强收缩并未赢得本轮的调参交叉熵选择。C4／C5 沿用了同一系数，因此当前结果也不能证明“收缩＋指标优化”已得到充分有效的正则化收益。若继续研究，应将正则强度与目标选择的作用分开。','',
        '![主工作点的类别偏置](class_biases.png)','',
        '纵轴为相对原生全局 bias 的额外修正。正值更容易释放前景，负值更容易保留 empty；C2 与 C3 基本重合。完整 17 类数值保留在 CSV。','',
        '### 3.2 全空间保护：不能把 C4 已有的折中算作新增贡献','',
        f'完整调参集上，C4 的全空间 mIoU 为 {c4["tuning"]["all_valid"]["mIoU"]:.4f}，C1 为 {c1["tuning"]["all_valid"]["mIoU"]:.4f}。C4 主工作点本身已满足 C5 的调参全空间下限，因此额外约束在该工作点没有强烈的筛选需求。',
        f'C5 训练内选中配置为：补集惩罚 {config["rho_out"]:g}、全空间惩罚 {config["rho_all"]:g}，从 C3 参数初始化；统一平移为 {c5["shift"]:+.5f}，**{"发生回退" if c5["fallback"] else "未回退"}**。验证的微小差异不足以支持独立的强贡献。','',
        '![固定工作点的空间折中](risk_tradeoff.png)','',
        '各点均由训练内规则确定，连线只用于观察折中，不是验证集上重新搜索的最优曲线；空心圈表示 +0.5 点主预算。C2／C3 的风险远低于这些点，主表单独保留。','',
        '## 4. 场景重采样：有没有更稳定？','',
        '从 523 个拟合场景中有放回抽取 10 组数据，重新拟合；主模型、调参场景及验证场景不变。C3 固定主实验选中的收缩系数，C5 固定训练内选中的惩罚配置。以下为十次拟合后的验证指标均值与样本标准差。','',
        '| 方法 | camera mIoU | 全空间 mIoU | 补集 FPR | 原始类别偏置平均标准差 | 回退次数 |',
        '|---|---:|---:|---:|---:|---:|']
    for family,s in stable.items():
        lines.append(f'| {family} | {s["camera"]["mean"]:.4f} ± {s["camera"]["std"]:.4f} | {s["all"]["mean"]:.4f} ± {s["all"]["std"]:.4f} | {s["risk"]["mean"]:.4f} ± {s["risk"]["std"]:.4f}% | {s["raw_mean_class_std"]:.6f} | {s["fallbacks"]}/10 |')
    lines += ['', '偏置标准差在统一再校准前计算，并对 17 类取平均。C2／C3 多类达到参数边界，其波动也受边界截断影响，不能把低波动自动归功于收缩。',
        '这些结果衡量拟合场景变化，不是十个独立主模型，也不是贝叶斯后验不确定性；本轮未重采样拟合 C1，不能据此宣称学习方案比 B2 更稳定。','',
        '![拟合场景变化下的指标](scene_stability.png)','',
        '## 5. 类别变化与数值复核','',
        '### 5.1 C5 相对 C1 的 camera 损失集中在哪里？','',
        '| 类别 | camera IoU 变化 | 全空间 IoU 变化 | C1 额外 bias | C5 额外 bias |','|---|---:|---:|---:|---:|']
    ca=np.asarray(c5['validation']['camera']['per_class_IoU'])-np.asarray(c1['validation']['camera']['per_class_IoU'])
    ga=np.asarray(c5['validation']['all_valid']['per_class_IoU'])-np.asarray(c1['validation']['all_valid']['per_class_IoU'])
    for k in np.argsort(ca)[:5]:lines.append(f'| {CLASSES[k]} | {ca[k]:+.4f} | {ga[k]:+.4f} | {c1["a"][k]:+.4f} | {c5["a"][k]:+.4f} |')
    lines += ['', '类别偏置只能决定该首选前景是否击败 empty，不能修复其他前景之间的排序。全空间宏平均 IoU 也不是 camera 与补集 mIoU 的简单平均；完整 TP/FP/FN 见逐类别 CSV。','',
        '### 5.2 分桶近似有没有造成明显计数误差？','',
        '以选定规则回到同一 40 万拟合样本，分别使用原始 margin 与 1024 桶中心计算软指标，差值如下。它检查选定点的数值近似，不构成训练轨迹相同的证明。','',
        '| 规则 | camera soft mIoU 差 | 全空间 soft mIoU 差 |','|---|---:|---:|']
    for name,row in read(RUN/'proxy_diagnostics.json').items():
        lines.append(f'| {SHORT[name]} | {row["exact"]["camera"]["mIoU"]-row["binned"]["camera"]["mIoU"]:+.6f} | {row["exact"]["all_valid"]["mIoU"]-row["binned"]["all_valid"]["mIoU"]:+.6f} |')
    lines += ['', '最终工作点选择与全部验证使用原始 margin 的精确硬决策，没有使用上述分桶代理。候选外体素也被计入统计。','',
        '## 6. 对下一阶段的含义','',
        '1. **保留类别 bias 作为强基线。** “变成可学习参数”本身没有扩大固定模型的决策能力；本轮尚无证据支持仅靠这种改变获得更高 camera mIoU。',
        '2. **优先匹配训练域与指标。** 普通混合区域交叉熵与当前任务目标差异很大。若继续，先比较 camera 条件下的类别损失、原加权体素目标与指标代理，避免把目标变化误归因于参数形式。',
        '3. **把空间折中作为显式选择，而非全面提升。** C4／C5 的全空间表现值得保留为工作点选项；它们没有在当前主预算下同时超越 C1 的 camera 指标。',
        '4. **联合训练仍需单独立项。** 冻结阶段尚未提供强到足以直接扩大训练的证据；本轮不外推为完整模型或其他 Gaussian 方法的结论。','',
        '## 7. 执行范围与产物','',
        f'- 4096 帧分类信息补提取；40 万拟合与 20 万调参候选；完整 994 帧调参、1024 帧验证。',
        f'- {len(fit["models"])} 个主拟合模型＋30 个场景重采样模型，共 {len(fit["models"])+30} 个；{len(selection["points"])} 个冻结工作点。',
        '- 官方 RayIoU 覆盖七个主要方法和全部 1024 帧；完整前向计时覆盖相同七个方法。',
        '- 三项数值／行为测试通过；两个完整输入上的 GT／mask 不变性与缓存预测一致性通过。',
        '- 原生预测、历史类别 bias 计数、模型与工作点哈希及六事件恒等式通过独立审计。','',
        '相关文件：','',
        '- [实验协议](../../../work_dirs/learnable_class_bias/protocol.json)、[冻结工作点](../../../work_dirs/learnable_class_bias/operating_points.json)、[完整验证结果](../../../work_dirs/learnable_class_bias/validation/summary.json)。',
        '- [全部工作点 CSV](operating_points.csv)、[逐类别 TP/FP/FN](per_class.csv)、[17 类偏置参数](class_biases.csv)。',
        '- [场景配对区间](../../../work_dirs/learnable_class_bias/paired_scene_intervals.json)、[重采样统计](../../../work_dirs/learnable_class_bias/stability/results.json)、[分桶复核](../../../work_dirs/learnable_class_bias/proxy_diagnostics.json)。',
        '- [原主模型损失口径](../../../work_dirs/learnable_class_bias/main_model_loss_context.json)、[官方 RayIoU](../../../work_dirs/learnable_class_bias/ray/summary.json)、[完整前向计时](../../../work_dirs/learnable_class_bias/benchmark.json)。',
        '- [执行审计](../../../work_dirs/learnable_class_bias/completion_audit.json)、[测试日志](../../../work_dirs/learnable_class_bias/tests.log)。','',
        '复现顺序如下；选择阶段在验证之前执行：','',
        '```bash','python3.10 tools/learnable_class_bias.py initialize','python3.10 tools/learnable_class_bias.py extract',
        'python3.10 tools/learnable_class_bias.py tuning','python3.10 tools/fit_learnable_class_bias.py fit',
        'python3.10 tools/eval_learnable_class_bias.py calibrate','python3.10 tools/eval_learnable_class_bias.py stability',
        'python3.10 -m pytest -q tests/test_learnable_class_bias.py > work_dirs/learnable_class_bias/tests.log 2>&1','python3.10 tools/eval_learnable_class_bias.py freeze',
        'python3.10 tools/eval_learnable_class_bias.py validate','python3.10 tools/audit_learnable_class_bias.py ray',
        'python3.10 tools/audit_learnable_class_bias.py benchmark','python3.10 tools/audit_learnable_class_bias.py invariance',
        'python3.10 tools/audit_learnable_class_bias.py completion','python3.10 tools/report_learnable_class_bias.py','```','']
    (ROOT/'analysis/empty_decision/class_bias/learnable_class_bias_experiment_results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    assert read(RUN/'completion_audit.json')['status']=='complete'
    v=read(RUN/'validation/summary.json');OUT.mkdir(parents=True, exist_ok=True)
    ci=paired(v);stable=stability(v);proxy_diagnostics(v);exports(v);figures(v,stable);report(v,ci,stable)
    write_json(RUN/'status.json',dict(status='complete',report='analysis/empty_decision/class_bias/learnable_class_bias_experiment_results.md'))
    print('Wrote analysis/empty_decision/class_bias/learnable_class_bias_experiment_results.md')
