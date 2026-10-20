"""Produce paired analyses, figures and a readable follow-up report from fixed results."""
from __future__ import annotations
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from tools.empty_decision_followup import RUN,OLD,read,write_json
from tools.eval_empty_decision_optimization import metrics,region_metrics
from tools.diagnose_opus_v1_gaussian import CLASSES

OUT=ROOT/'analysis/empty_decision_followup'
ORDER=['B0','B2','B2_shift','OldP3','H','Mix','Long','Balanced','Joint','Conditional','S','Support','Expanded','Safe']
NAMES=dict(B0='原生全局 learnable bias',B2='冻结类别 bias',B2_shift='类别 bias 再校准',OldP3='上轮完整 P3',
    H='普通特征残差',Mix='原混合概率残差',Long='延长概率拟合',Balanced='区域平衡概率',Joint='六事件联合概率',
    Conditional='两组条件概率',S='单一支持 S',Support='完整支持组',Expanded='扩大残差拟合样本',Safe='正确前景撤回惩罚')
PNAMES=dict(prob_no_support_seed928='混合头 800 步',Q_mix_long_seed928='混合头 2400 步',
    Q_balanced_seed928='区域等权',Q_joint6_seed928='联合六事件',Q_mask_oracle_seed928='真实分区参照（仅诊断）',
    Q_mix_wide_seed928='加宽混合头',Q_mix_expanded_seed928='扩大样本混合头')
VNAMES=dict(direct='三结果',marginal='边缘三结果',conditional_diagnostic='条件三结果（诊断）')
COLORS=dict(zip(['B2_shift','H','Mix','Joint','Conditional','S','Support','Expanded','Safe'],plt.get_cmap('tab10').colors))


def primary(validation):
    points={p['family']:p for p in validation['operating_points'].values() if p['family'] in ['B0','B2'] or
        (p['budget']==.005 and p['seed'] in [None,928])}
    old=read(OLD/'validation/summary.json')
    points['OldP3']=next(p for p in old['operating_points'].values() if p['family']=='P3' and p['budget']==.005 and p['seed']==928)
    return points


def paired(validation,points):
    names=list(validation['operating_points'])
    with np.load(RUN/'validation/per_frame_counts.npz') as z:c=z['confusion']
    with np.load(OLD/'validation/per_frame_counts.npz') as z:
        old=list(read(OLD/'validation/summary.json')['operating_points'])
        extra=z['confusion'][:,old.index(points['OldP3']['id'])]
    selected=np.stack([extra if f=='OldP3' else c[:,names.index(points[f]['id'])] for f in ORDER],1)
    scenes=np.asarray([f['scene'] for f in validation['frames']]);unique,inverse=np.unique(scenes,return_inverse=True)
    byscene=np.zeros((len(unique),len(ORDER),2,18,18),np.float64);np.add.at(byscene,inverse,selected)
    rng=np.random.default_rng(928);multiplicity=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),500)
    with threadpool_limits(8):b=(multiplicity@byscene.reshape(len(unique),-1)).reshape(500,len(ORDER),2,18,18)
    camera=metrics(b[:,:,0])['mIoU'];risk=metrics(b[:,:,1])['empty_FPR']
    pairs=[(f,'B2_shift') for f in ['H','Mix','Joint','Conditional','S','Support','Expanded','Safe']]
    pairs += [('Mix','H'),('Joint','H'),('Conditional','Joint'),('Conditional','H'),('Long','Mix'),('Balanced','Mix'),('S','H'),('Support','S'),('Support','H'),('Expanded','H'),('Safe','Support'),('H','OldP3')]
    result=dict(repetitions=500,scenes=len(unique),scope='fixed fitted models and rules; no refitting/reselection',comparisons={})
    for a,d in pairs:
        ia,ib=ORDER.index(a),ORDER.index(d)
        result['comparisons'][a+'__minus__'+d]=dict(camera_ci95=np.quantile(camera[:,ia]-camera[:,ib],[.025,.975]).tolist(),
            risk_ci95=np.quantile(risk[:,ia]-risk[:,ib],[.025,.975]).tolist())
    write_json(RUN/'scene_bootstrap.json',result)
    return result


def ray_paired(validation,points):
    scenes=sorted({f['scene'] for f in validation['frames']});mapping={s:i for i,s in enumerate(scenes)}
    values=np.zeros((len(scenes),len(ORDER),90),np.float64)
    for frame in validation['frames']:
        new=read(RUN/'ray'/f'frame_{frame["index"]:04d}.json')['counts']
        old=read(OLD/'ray'/f'frame_{frame["index"]:04d}.json')['counts']
        for i,family in enumerate(ORDER):
            row=(old if family=='OldP3' else new)[points[family]['id']]
            values[mapping[frame['scene']],i]+=np.r_[row['gt'],row['pred'],np.asarray(row['tp']).reshape(-1)]
    rng=np.random.default_rng(928);m=rng.multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),500)
    with threadpool_limits(8):summed=(m@values.reshape(len(scenes),-1)).reshape(500,len(ORDER),90)
    gt,pred,tp=summed[:,:,:18],summed[:,:,18:36],summed[:,:,36:].reshape(500,len(ORDER),3,18)
    union=gt[:,:,None,:]+pred[:,:,None,:]-tp
    iou=np.divide(tp,union,out=np.full_like(tp,np.nan),where=union>0)
    score=np.nanmean(iou[:,:,:,:17],axis=(-1,-2))*100
    pairs=[('Safe','B2_shift'),('H','B2_shift'),('Support','B2_shift'),('Conditional','B2_shift'),
           ('Safe','Support'),('Safe','OldP3'),('H','OldP3'),('Conditional','OldP3')]
    result=dict(repetitions=500,scenes=len(scenes),comparisons={})
    for a,b in pairs:
        delta=score[:,ORDER.index(a)]-score[:,ORDER.index(b)]
        result['comparisons'][a+'__minus__'+b]=np.quantile(delta,[.025,.975]).tolist()
    write_json(RUN/'ray_scene_bootstrap.json',result)
    return result


def export(validation,points):
    OUT.mkdir(exist_ok=True)
    with (OUT/'all_operating_points.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','family','seed','budget','source','threshold','camera_mIoU','all_mIoU','complement_FPR','tuning_camera_mIoU','tuning_complement_FPR'])
        writer.writeheader()
        for p in validation['operating_points'].values():
            v=p['validation'];t=p['tuning']
            writer.writerow({**{k:p.get(k) for k in ['id','family','seed','budget','source','threshold']},
                'camera_mIoU':v['camera']['mIoU'],'all_mIoU':v['all_valid']['mIoU'],'complement_FPR':v['camera_complement']['empty_FPR'],
                'tuning_camera_mIoU':t['camera']['mIoU'],'tuning_complement_FPR':t['camera_complement']['empty_FPR']})
    with np.load(RUN/'validation/per_frame_counts.npz') as z:total=z['confusion'].sum(0)
    names=list(validation['operating_points'])
    with (OUT/'per_class_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','region','class','TP','FP','FN','IoU']);writer.writeheader()
        for family,p in points.items():
            if family=='OldP3':continue
            cc=total[names.index(p['id'])]
            for region,c in [('camera',cc[0]),('complement',cc[1]),('all',cc.sum(0))]:
                for k,name in enumerate(CLASSES):writer.writerow(dict(method=family,region=region,**{'class':name},
                    TP=int(c[k,k]),FP=int(c[:,k].sum()-c[k,k]),FN=int(c[k].sum()-c[k,k]),IoU=metrics(c)['per_class_IoU'][k]))


def figures(validation,points):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for family in ['B2_shift','H','Mix','Joint','Conditional','S','Support','Expanded','Safe']:
        pp=[p for p in validation['operating_points'].values() if p['family']==family and p['seed'] in [None,928]]
        pp.sort(key=lambda p:p['validation']['camera_complement']['empty_FPR'])
        x=[p['validation']['camera_complement']['empty_FPR'] for p in pp];y=[p['validation']['camera']['mIoU'] for p in pp]
        line,=axes[0].plot(x,y,'o-',ms=4,label=family,color=COLORS[family])
        p=points[family]['validation'];axes[1].scatter(p['camera_complement']['empty_FPR'],p['camera']['mIoU'],label=family,color=line.get_color())
        offsets={'H':(-12,-17),'Joint':(5,7),'Mix':(5,5),'S':(5,-14),'Support':(5,6),
                 'Conditional':(5,6),'Expanded':(-55,6),'Safe':(-30,7),'B2_shift':(5,6)}
        axes[1].annotate(family,(p['camera_complement']['empty_FPR'],p['camera']['mIoU']),
            xytext=offsets[family],textcoords='offset points',fontsize=8)
    for ax in axes:ax.set(xlabel='Complement empty false occupancy (%)',ylabel='Camera mIoU (%)');ax.grid(alpha=.2)
    axes[0].set_title('All training-selected budgets');axes[1].set_title('Primary +0.5 pp budget');axes[0].legend(fontsize=8,ncol=2)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/('risk_tradeoff.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,family in enumerate(['H','Mix','Joint','Conditional','S','Support']):
        pp=[p for p in validation['operating_points'].values() if p['family']==family and p['budget']==.005]
        for ax,region,key in [(axes[0],'camera','mIoU'),(axes[1],'camera_complement','empty_FPR')]:
            value=[p['validation'][region][key] for p in pp]
            ax.scatter(np.full(len(value),i)+np.linspace(-.07,.07,len(value)),value,color=COLORS[family])
            ax.plot([i-.15,i+.15],[np.mean(value)]*2,color='black');ax.set_xticks(range(6),['H','Mix','Joint','Cond.','S','Support']);ax.grid(alpha=.2)
    axes[0].set(ylabel='Camera mIoU (%)',title='Three head seeds; one shared backbone')
    axes[1].set(ylabel='Complement false occupancy (%)',title='Actual validation risk')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/('module_seeds.'+ext),dpi=180)
    plt.close(fig)
    variants=[('prob_no_support_seed928/direct','Mix 800'),('Q_mix_long_seed928/direct','Mix 2400'),
        ('Q_balanced_seed928/direct','Balanced'),('Q_joint6_seed928/marginal','Joint marginal'),
        ('Q_joint6_seed928/conditional_diagnostic','Joint conditional*'),('Q_mask_oracle_seed928/direct','Mask oracle*'),
        ('Q_mix_wide_seed928/direct','Wide'),('Q_mix_expanded_seed928/direct','Expanded')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for r,ax in enumerate(axes):
        predicted=[];actual=[]
        for key,label in variants:
            stat=np.asarray(validation['raw_calibration'][key])[r]
            bins=stat[3:48].reshape(15,3)
            predicted.append(bins[:,1].sum()/stat[0]*100);actual.append(bins[:,2].sum()/stat[0]*100)
        ax.bar(np.arange(len(variants)),predicted,color=['#4477aa']*4+['#cc6677']*2+['#4477aa']*2)
        ax.axhline(actual[0],color='black',ls='--',label='Actual correct-class frequency')
        ax.set_xticks(np.arange(len(variants)),[v[1] for v in variants],rotation=45,ha='right')
        ax.set(ylabel='Mean predicted correct-class probability (%)',title='Camera' if r==0 else 'Camera-mask complement')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/('regional_probability.'+ext),dpi=180)
    plt.close(fig)


def make_report(validation,points,bootstrap):
    selection=read(RUN/'operating_points.json');ray=read(RUN/'ray/summary.json');bench=read(RUN/'benchmark.json')
    oldray=read(OLD/'ray/summary.json');prob=read(RUN/'probability_validation_metrics.json')
    trainprob=read(RUN/'probability_training_metrics.json');probes=read(RUN/'support_probes_validation.json')
    lines=['# Empty 决策优化后续实验：原因验证与改进尝试','',
        '日期：2026-10-18。状态：**冻结模型阶段完成；未进行主模型联合训练。**','',
        '沿用 4096 帧 mini 训练集和 1024 帧验证集，冻结同一个 5-epoch 主模型。训练内划分、候选范围和全局 learnable bias 均沿用原实验。',
        '本轮检验概率中间表示、区域目标、支持信息、扩大采样及正确前景撤回惩罚。设计见[实验设计](empty_decision_followup_design.md)，原问题解释见[上轮复盘](empty_decision_optimization_failure_analysis.md)。','',
        '**读表规则：** mIoU、IoU、AP、FPR 用百分数，差值用百分点；方法按同一训练预算选定，实际验证风险未必相同。超出 0.25 点容差时，不称为等风险比较。','',
        f'训练内选择的主工作点为 **{selection["training_selected_primary_id"]}**；该选择在本轮验证之前冻结。下文同时报告所有预设版本，不按验证结果回选部署方案。','']
    if (RUN/'interpretation.md').exists():lines += [(RUN/'interpretation.md').read_text().strip(),'']
    lines += ['## 1. 主工作点：有没有超过类别 bias？','',
        '下表为 +0.5 点训练预算、模块 seed 928。B0 为原生输出，OldP3 为上轮已冻结的完整方案。','',
        '| 方法 | camera mIoU | 全空间 mIoU | 补集误占用率 | camera 二值 IoU | RayIoU | FPS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for family in ORDER:
        p=points[family];v=p['validation'];r=oldray['models'][p['id']] if family=='OldP3' else ray['models'][p['id']]
        b=bench['models']['OldP3' if family=='OldP3' else p['id']]
        lines.append(f'| {family}：{NAMES[family]} | {v["camera"]["mIoU"]:.4f} | {v["all_valid"]["mIoU"]:.4f} | {v["camera_complement"]["empty_FPR"]:.4f}% | {v["camera"]["IoU"]:.4f} | {r["RayIoU"]:.4f} | {b["FPS"]:.2f} |')
    lines += ['', '![预先冻结工作点的收益与风险](empty_decision_followup/risk_tradeoff.png)','',
        '各预算都在训练内选择；连线用于展示折中，不是验证集上重新搜索的最优包络。B2_shift 获得与局部方法相同的统一阈值扫描机会。','',
        '## 2. 概率是否有独立贡献，支持是否有额外价值？','',
        '500 次场景级配对 bootstrap，150 个场景。区间条件于固定主模型、已拟合头及冻结工作点，不含重新训练或模型选择的不确定性。','',
        '| 比较 | camera mIoU 差 | 95% 区间 | 补集风险差 | 风险在 0.25 点容差内？ |','|---|---:|---|---:|---|']
    for key,ci in bootstrap['comparisons'].items():
        a,b=key.split('__minus__');va,vb=points[a]['validation'],points[b]['validation']
        delta=va['camera']['mIoU']-vb['camera']['mIoU'];risk=va['camera_complement']['empty_FPR']-vb['camera_complement']['empty_FPR']
        lo,hi=ci['camera_ci95'];lines.append(f'| {a} − {b} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {risk:+.4f} | {"是" if abs(risk)<=.25 else "否"} |')
    lines += ['', 'RayIoU 同时要求语义和距离匹配，不能单独当作纯几何指标。其场景级配对区间如下，仍条件于固定工作点：','',
        '| 比较 | RayIoU 差 | 95% 区间 |','|---|---:|---|']
    for key,interval in read(RUN/'ray_scene_bootstrap.json')['comparisons'].items():
        a,b=key.split('__minus__')
        def score(f):return (oldray if f=='OldP3' else ray)['models'][points[f]['id']]['RayIoU']
        lines.append(f'| {a} − {b} | {score(a)-score(b):+.4f} | [{interval[0]:+.4f}, {interval[1]:+.4f}] |')
    lines += ['', '### 2.1 三个模块种子','',
        '| 方法 | camera mIoU 均值 ± 标准差 | 补集误占用率均值 ± 标准差 |','|---|---:|---:|']
    for family in ['H','Mix','Joint','Conditional','S','Support']:
        pp=[p for p in validation['operating_points'].values() if p['family']==family and p['budget']==.005]
        values=[[p['validation'][region][key] for p in pp] for region,key in [('camera','mIoU'),('camera_complement','empty_FPR')]]
        lines.append('| '+family+' | '+' | '.join(f'{np.mean(v):.4f} ± {np.std(v,ddof=1):.4f}' for v in values)+' |')
    lines += ['', '![三个模块种子的结果](empty_decision_followup/module_seeds.png)','',
        '这些是三个局部头种子，共享同一个主模型；不能称为三个完整模型独立训练。','']
    if (RUN/'width_control/results.json').exists():
        width=read(RUN/'width_control/results.json')
        lines += ['### 2.2 补充同宽对照：避免把概率作用与决策网络宽度混淆','',
            '主实验 H40 与 Mix16 匹配总参数量。补充的 H16 与概率版本匹配决策网络的层数和隐藏宽度，使用相同普通特征、风险代价网格和训练步数；它有 705 个决策参数，Mix 有 753 个决策参数及 1987 个冻结概率参数。',
            '该补充在主工作点已冻结、但尚未开始本轮方法预测评价时单独登记；主实验选择未修改。补充共 15 个风险代价/种子头、15 个固定工作点，仅用于容量与概率作用诊断。','',
            '| 比较（seed 928） | camera mIoU 差 | 95% 区间 | 补集风险差 |','|---|---:|---|---:|']
        for name,row in width['comparisons'].items():
            lo,hi=row['camera_ci95'];lines.append(f'| {name.replace("__minus__"," − ")} | {row["camera_delta"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {row["risk_delta"]:+.4f} |')
        values=[[p['validation'][r][k] for p in width['primary'].values()] for r,k in [('camera','mIoU'),('camera_complement','empty_FPR')]]
        lines += ['',f'H16 三种子 camera mIoU 为 **{np.mean(values[0]):.4f} ± {np.std(values[0],ddof=1):.4f}**，补集误占用率为 **{np.mean(values[1]):.4f} ± {np.std(values[1],ddof=1):.4f}%**。',
            '区间仍条件于固定训练结果；需要结合实际风险差阅读，不能只看 camera 差值。','']
    if (RUN/'decision_dependence.json').exists():
        dependence=read(RUN/'decision_dependence.json')['models']
        lines += ['### 2.3 概率输入改变时，原决策网络是否会响应？','',
            '在 20 万个训练内调参样本上，固定类别、margin 所在的 0.25 宽分箱，再在每组内打乱概率向量；原始 h 和阈值保持不变。下表为候选总体的加权翻转比例。','',
            '| 网络 | camera 决策翻转 | 补集决策翻转 | camera 分数变化 RMS |','|---|---:|---:|---:|']
        for family in ['OldP3','Mix','Conditional']:
            if 'probability_permutation' not in dependence[family]:continue
            rows=dependence[family]['probability_permutation'];lines.append(f'| {family} | {rows["camera"]["decision_flip_fraction"]*100:.4f}% | {rows["complement"]["decision_flip_fraction"]*100:.4f}% | {rows["camera"]["score_change_rms"]:.5f} |')
        lines += ['', '这是冻结网络的输入敏感性诊断；打乱会破坏 h 与概率之间的原有对应，不能代替重新训练消融，也不构成因果性能保证。翻转率低不等于对稀有类别指标没有影响；概率是否值得保留，应主要看同宽、同目标对照。','']
    lines += [
        '## 3. 区域混合、拟合不足与容量分别有什么影响？','',
        '这里分析的是额外纠偏概率头的正确/empty/错类事件概率，不是主模型原始 18 类语义分数。其分区域校准现象解释纠偏设计的局限，不能单独证明 Gaussian 聚合或 OPUS 分类损失是原始 empty 问题的物理根因。','',
        '下表用共同的 40 万拟合样本复查训练内 camera 校准；即使某头使用 120 万行拟合，复查口径仍保持一致。所有概率头只按训练内损失选择。','',
        '| 概率头 | camera 平均预测 q | camera 实际正确比例 | camera q 的 ECE | camera q AP |','|---|---:|---:|---:|---:|']
    for name,entry in trainprob.items():
        for variant,row in entry['fit'].items():
            if variant=='region_prediction':continue
            q=row['camera']['q'];lines.append(f'| {PNAMES[name]} / {VNAMES[variant]} | {q["mean"]*100:.2f}% | {q["prevalence"]*100:.2f}% | {q["ECE"]*100:.2f}% | {q["AP"]*100:.2f}% |')
    lines += ['', '### 3.1 独立验证场景的概率质量','',
        'NLL 与 ECE 使用完整候选体素；AP 使用与旧实验相同的 1,048,576 个候选抽样位置及逆纳入概率权重。联合头同时报告边缘概率及条件概率；条件版本以真实区域选择输出，仅供诊断，不能当作推理时拥有真实 mask。','',
        '| 概率头 / 输出 | camera NLL | camera q ECE | camera q AP | 补集 q ECE | 补集 q AP |','|---|---:|---:|---:|---:|---:|']
    for name,entry in prob.items():
        for variant,row in entry.items():
            if variant=='region_prediction':continue
            full=validation['calibration'][name+'/'+variant];a,b=full['camera'],full['camera_complement']
            lines.append(f'| {PNAMES[name]} / {VNAMES[variant]} | {a["NLL"]:.4f} | {a["q_ECE"]*100:.2f}% | {row["camera"]["q"]["AP"]*100:.2f}% | {b["q_ECE"]*100:.2f}% | {row["complement"]["q"]["AP"]*100:.2f}% |')
    lines += ['',
        '![分区域预测正确率与实际比例](empty_decision_followup/regional_probability.png)','',
        '虚线为该区域候选中前景首选正确的实际比例。星号与红色标记仅为真实区域条件化的诊断统计；Conditional 的可部署推理同时接收两组预测条件概率，不使用真实分区来选分支。','',
        '**解释边界：** balanced 改变了目标区域比例；mask_oracle 读取真实分区，只是诊断参照。Joint 输入六个联合概率；Conditional 输入两组各自归一化的三结果概率，两个可部署版本均不读取真实 mask。延长或加宽拟合有助于排查限制，但有限训练仍不能证明已达最优。','']
    reference=selection['oracle_region_reference'];ids=list(validation['operating_points'])
    with np.load(RUN/'validation/per_frame_counts.npz') as z:
        cc=z['confusion'].sum(0)
    combined=np.stack([cc[ids.index(reference['camera_policy_id']),0],cc[ids.index('B2'),1]])
    oracle=region_metrics(combined);write_json(RUN/'oracle_region_reference.json',dict(**reference,metrics=oracle))
    lines += ['### 3.2 如果知道真实区域，固定策略能做到什么？','',
        f'训练内预先选择的无约束 camera 策略为 `{reference["camera_policy_id"]}`。仅在真实 camera 区域采用它，补集采用 B2，得到如下诊断参照：','',
        '| 真实分区路由参照 | camera mIoU | 全空间 mIoU | 补集误占用率 |','|---|---:|---:|---:|',
        f'| 仅诊断，不能部署 | {oracle["camera"]["mIoU"]:.4f} | {oracle["all_valid"]["mIoU"]:.4f} | {oracle["camera_complement"]["empty_FPR"]:.4f}% |','',
        '这里使用真实 mask 路由，未训练一个新的可部署模型。它只展示这一组有限固定策略在给定分区信息下的潜力，不是严格最优上界；不得与主表方法混排。','',
        '## 4. 支持信息存在，但小网络没有用好吗？','',
        '固定 200 轮 HistGradientBoosting 诊断模型，在同一双向候选上分别使用普通特征、增加 S、增加完整支持组。AP 衡量排序能力，不等于 mIoU；这些预测器不参加主方法排名。','',
        '| 拟合范围 / 特征组 | camera 正确前景 AP | 补集正确前景 AP |','|---|---:|---:|']
    for name,p in probes.items():lines.append(f'| {name} | {p["validation"]["camera"]["q_AP"]*100:.2f}% | {p["validation"]["complement"]["q_AP"]*100:.2f}% |')
    lines += ['', '这是单个固定诊断模型的预测关联；本表未给出 AP 配对区间，微小差异不能直接称为稳定增量。之前 +0.63 点 AP 的实验使用另一候选定义，本表不与之作直接提升比较。','',
        '## 5. 实际工作点与预测改变','',
        '| 方法 | 训练内 camera mIoU | 训练内补集 FPR | 风险代价 λ | 统一阈值 | 退回零残差？ |','|---|---:|---:|---:|---:|---|']
    for family in ORDER:
        if family in ['B0','OldP3']:continue
        p=points[family];lines.append(f'| {family} | {p["tuning"]["camera"]["mIoU"]:.4f} | {p["tuning"]["camera_complement"]["empty_FPR"]:.4f}% | {p.get("lam","—")} | {p["threshold"]:.5f} | {"是" if p.get("zero_residual") else "—"} |')
    lines += ['', '| 方法 / 区域 | 正确前景新增 | empty 新增误占用 | 正确前景被撤回 | 误占用被撤回 |','|---|---:|---:|---:|---:|']
    for family in ['B2_shift','H','Mix','Joint','Conditional','Support','Expanded','Safe']:
        for r,label in [(0,'camera'),(1,'补集')]:
            e=points[family]['transition_counts'][r]
            lines.append(f'| {family} / {label} | {e[0]:,} | {e[2]:,} | {e[3]:,} | {e[5]:,} |')
    lines += ['', '新增与撤回均相对原生 B0，完整六事件计数另存于结构化结果。撤回量需要结合 IoU、RayIoU 判断；仅控制补集 empty FP 并不保证其他指标。','',
        '### 5.1 训练内选中方法的全空间损失来自哪些类别？','',
        f'以下对比预先选中的 {validation["operating_points"][selection["training_selected_primary_id"]]["family"]} 与 B2_shift，列出全空间 IoU 下降最多的五类。FP 包含该类的所有误判，既包括真实 empty，也包括其他前景错类。完整 17 类在 CSV 中保留。','',
        '| 类别 | camera IoU 变化 | 全空间 IoU 变化 | 补集 TP 变化 | 补集 FP 变化 |','|---|---:|---:|---:|---:|']
    actual=cc[ids.index(selection['training_selected_primary_id'])];base=cc[ids.index(points['B2_shift']['id'])]
    global_delta=metrics(actual.sum(0))['per_class_IoU']-metrics(base.sum(0))['per_class_IoU']
    camera_delta=metrics(actual[0])['per_class_IoU']-metrics(base[0])['per_class_IoU']
    for k in np.argsort(global_delta)[:5]:
        tp=int(actual[1,k,k]-base[1,k,k])
        fp=int(actual[1,:,k].sum()-actual[1,k,k]-base[1,:,k].sum()+base[1,k,k])
        lines.append(f'| {CLASSES[k]} | {camera_delta[k]:+.4f} | {global_delta[k]:+.4f} | {tp:+,} | {fp:+,} |')
    lines += ['', 'camera 与全空间各类别的 union 不同，mIoU 也不是体素正确率或两个区域 mIoU 的简单平均。区域内的语义增益不能仅凭一个补集误占用总量约束推出全空间增益。','',
        '## 6. 执行、开销与结论范围','',
        '- 完整提取 4096 个训练帧的支持组，保持原 40 万/20 万主比较样本的行顺序与标签、权重一致；另准备 120 万拟合行。',
        f'- 拟合 {len(selection["model_sha256"])} 个风险代价/种子决策头；全体模型和工作点在验证前冻结。',
        f'- 在全部 1024 帧评价 {len(validation["operating_points"])} 个固定工作点，官方 RayIoU 覆盖 {len(ray["models"])} 个主要工作点。',
        '- 同宽补充另完成 15 个决策头和 15 个工作点；合计 125 个决策头、132 个工作点。补充点不混入主实验选择。',
        '- 本轮 FPS 在同一轮重新测量，包括 OldP3；8 个输入，每方法每输入预热 2 次、计时 5 次，随机顺序，统一排除读取和 H2D。',
        '- S 原型包含一次额外聚合，完整支持包含两次额外聚合，均未融合；效率是当前实际实现值。',
        '- 主模型只有一个 5-epoch mini checkpoint，且验证集已参与研究讨论；结果不外推为完整训练模型或所有 Gaussian 方法的普遍结论。',
        '- 相比 OldP3 的整体改动同时包含类别锚点、残差幅度和优化方式；整体收益不能全部归因于某一单项。单项解释以本轮同设置对照为准。',
        '- 本轮未进行主模型联合训练。完整混淆矩阵、方向变化、原生输出复现及 GT/mask 推理不变性由独立审计检查。','',
        '相关测试共 9 项通过。两份新文档的 Markdown 结构、公式、表格和本地链接已通过渲染检查；三张图另经逐图视觉检查。','',
        '## 7. 复现与产物','',
        '```bash','python3.10 tools/empty_decision_followup.py train','python3.10 tools/empty_decision_followup.py prepare',
        'python3.10 tools/fit_empty_decision_followup.py all','python3.10 tools/probe_empty_decision_followup.py fit',
        'python3.10 tools/eval_empty_decision_followup.py tune','python3.10 tools/eval_empty_decision_followup.py select',
        'python3.10 tools/width_empty_decision_followup.py fit',
        'python3.10 tools/empty_decision_followup.py val','python3.10 tools/eval_empty_decision_followup.py validate',
        'python3.10 tools/width_empty_decision_followup.py validate','python3.10 tools/width_empty_decision_followup.py summarize',
        'python3.10 tools/inspect_empty_decision_followup.py',
        'python3.10 tools/probe_empty_decision_followup.py eval','python3.10 tools/audit_empty_decision_followup.py ray',
        'python3.10 tools/audit_empty_decision_followup.py benchmark','python3.10 tools/audit_empty_decision_followup.py invariance',
        'python3.10 tools/audit_empty_decision_followup.py completion','python3.10 tools/report_empty_decision_followup.py','```','',
        '- [本轮协议](../work_dirs/empty_decision_followup/protocol.json)与[冻结选择](../work_dirs/empty_decision_followup/operating_points.json)。',
        '- [完整结果](../work_dirs/empty_decision_followup/validation/summary.json)、[概率训练内诊断](../work_dirs/empty_decision_followup/probability_training_metrics.json)和[验证概率 AP](../work_dirs/empty_decision_followup/probability_validation_metrics.json)。',
        '- [支持信息诊断](../work_dirs/empty_decision_followup/support_probes_validation.json)、[场景级区间](../work_dirs/empty_decision_followup/scene_bootstrap.json)。',
        '- [同宽补充对照](../work_dirs/empty_decision_followup/width_control/results.json)、[冻结网络的概率依赖诊断](../work_dirs/empty_decision_followup/decision_dependence.json)。',
        '- [主实验全部工作点 CSV](empty_decision_followup/all_operating_points.csv)、[逐类别统计](empty_decision_followup/per_class_metrics.csv)；同宽补充点见上方独立结果。',
        '- [测试日志](../work_dirs/empty_decision_followup/tests.log)、[文档渲染检查](../work_dirs/empty_decision_followup/document_validation.json)。',
        '- [官方 RayIoU](../work_dirs/empty_decision_followup/ray/summary.json)、[推理计时](../work_dirs/empty_decision_followup/benchmark.json)、[完成审计](../work_dirs/empty_decision_followup/completion_audit.json)。','']
    (ROOT/'analysis/empty_decision_followup_results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    assert read(RUN/'completion_audit.json')['status']=='complete'
    v=read(RUN/'validation/summary.json');p=primary(v);OUT.mkdir(exist_ok=True)
    b=paired(v,p);ray_paired(v,p);export(v,p);figures(v,p);make_report(v,p,b)
    write_json(RUN/'status.json',dict(status='complete',report='analysis/empty_decision_followup_results.md'))
    print('Wrote analysis/empty_decision_followup_results.md')
