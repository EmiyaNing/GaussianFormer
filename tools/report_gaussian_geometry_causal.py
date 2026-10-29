"""Aggregate exact scene counters, paired scene intervals, and readable tables."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.gaussian_geometry_causal import RUN,ARMS,write,sha
from tools.stat_gaussian_voxel_coverage import FIELDS,FLAGS
from tools.eval_gaussian_geometry_causal import EDGES,G_NAMES,RAW_NAMES

EXTRA='no_empty_center_boost'
TREATMENTS=[*ARMS[:4],EXTRA]
COMPARISONS=[*ARMS[1:4],EXTRA]
NAMES=['common',*TREATMENTS]
LABELS=dict(common='共同起点',standard='标准训练',no_amplitude='阻断幅度',
            no_geometry='阻断几何',no_both='同时阻断',no_empty_center_boost='仅阻断空中心增强')
CLASS_NAMES=('others','barrier','bicycle','bus','car','construction_vehicle','motorcycle',
    'pedestrian','traffic_cone','trailer','truck','driveable_surface','other_flat',
    'sidewalk','terrain','manmade','vegetation')


def class_rows(name,split,cm,source='new_evaluation'):
    rows=[]
    for c,label in enumerate(CLASS_NAMES):
        gt=int(cm[c].sum());pred=int(cm[:,c].sum());tp=int(cm[c,c]);union=gt+pred-tp
        rows.append(dict(model=name,split=split,source=source,class_id=c,class_name=label,
            GT=gt,predicted=pred,TP=tp,FP_from_empty=int(cm[17,c]),
            wrong_fg_into_class=int(cm[:17,c].sum()-tp),FN_to_empty=int(cm[c,17]),
            GT_to_wrong_fg=int(cm[c,:17].sum()-tp),IoU_percent=100*tp/max(union,1),
            semantic_recall_percent=100*tp/max(gt,1),
            occupancy_recall_percent=100*int(cm[c,:17].sum())/max(gt,1)))
    return rows


def cm_metrics(cm):
    cm=np.asarray(cm,dtype=np.float64)
    tp=np.diagonal(cm,axis1=-2,axis2=-1)[...,:17]
    rows=cm.sum(-1);cols=cm.sum(-2)
    union=rows[...,:17]+cols[...,:17]-tp
    miou=np.mean(np.divide(tp,union,out=np.zeros_like(tp),where=union>0),axis=-1)*100
    fg=rows[...,:17].sum(-1);empty=rows[...,17]
    detected=cm[...,:17,:17].sum((-2,-1));fn=cm[...,:17,17].sum(-1);fp=cm[...,17,:17].sum(-1)
    div=lambda x,y:np.divide(x,y,out=np.zeros_like(np.asarray(x,dtype=float)),where=y>0)
    return dict(mIoU=miou,IoU=100*div(detected,fg+fp),FN_rate=100*div(fn,fg),
        FPR=100*div(fp,empty),FP_over_GTfg=100*div(fp,fg),
        FP_over_predfg=100*div(fp,detected+fp),semantic_error=100*div(detected-tp.sum(-1),fg),
        detected_semantic_accuracy=100*div(tp.sum(-1),detected),
        GTfg=fg,GTempty=empty,FP=fp,FN=fn)


def metrics(a):
    m={k:float(v) for k,v in cm_metrics(a['confusion']).items()}
    ns=a['field_n'].sum(0);sums=a['field_sum'].sum(0)
    for key in ('n_1sigma','n_2sigma','n_3sigma','kernel_mass','opacity_mass','dominant_mahalanobis',
                'semantic_amplitude','class_concentration'):
        # Coverage field names are imported from the frozen, audited utility.
        field=key.replace('n_', 'within_', 1) if key.startswith('n_') else key
        if field in FIELDS:
            j=FIELDS.index(field);m[key]=float(sums[j]/max(ns[j],1))
    for name in ('no_native_support','no_1sigma','no_2sigma','no_3sigma'):
        m[name]=float(100*a['flags'][:,FLAGS.index(name)].sum()/max(a['voxels'].sum(),1))
    gs=a['gaussian_total'];n=gs[0]
    for key in G_NAMES[1:]:m['GS_'+key]=float(gs[G_NAMES.index(key)]/n)
    for key in ('maxscale_ge_03999','allaxes_ge_03999','center_camera','center_fg',
                'center_camera_empty','camera_empty_and_high_evidence'):
        m['GS_'+key]*=100
    m['CE_weighted']=float(10*a['loss'][0]/a['loss'][1])
    raw=a['raw_attribute'];rn=max(raw[0],1)
    for k in ('all_scale_axes_zero','largest_scale_axis_zero','opacity_jacobian_zero'):
        m['raw_'+k]=float(100*raw[RAW_NAMES.index(k)]/rn)
    m['raw_largest_axis_jacobian_relative']=float(raw[4]/rn)
    m['raw_opacity_jacobian_relative']=float(raw[6]/rn)
    hist=a['margin_hist'].sum(1)
    tp=hist[:17].sum(0)[::-1].cumsum();fp=hist[17,::-1].cumsum()
    recall=tp/max(m['GTfg'],1);precision=tp/np.maximum(tp+fp,1)
    m['binary_AP_hist']=float(100*np.sum(np.diff(np.r_[0,recall])*precision))
    return m


def aggregate(name,split,expected):
    folder=RUN/'eval'/name/split
    complete=json.loads((folder.parent/'complete.json').read_text())
    paths=sorted(folder.glob('*.npz'))
    assert len(paths)==expected['scenes'],(name,split,len(paths))
    combined=None;scenes=[];allindices=[];rows=[]
    for path in paths:
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['status']=='complete'
        assert meta['checkpoint_sha256']==complete['checkpoint_sha256']
        a=dict(np.load(path))
        if combined is None:combined={k:v.copy() for k,v in a.items()}
        else:
            for k,v in a.items():
                if k=='count_max':combined[k]=np.maximum(combined[k],v)
                else:combined[k]+=v
        # Bootstrap needs only these tiny arrays. Do not retain a multi-MB
        # margin histogram for every scene/model in RAM.
        scenes.append((path.stem,{k:a[k] for k in ('confusion','field_sum','field_n')}))
        allindices.extend(meta['indices'])
        rows.append(dict(name=name,split=split,scene=path.stem,frames=meta['frames'],**metrics(a)))
    assert sorted(allindices)==list(range(expected['frames']))
    np.savez_compressed(RUN/'aggregate'/f'{name}_{split}.npz',**combined)
    return combined,scenes,rows


def calibrated(train,val,target_fp):
    h=train['margin_hist'];suffix=h[...,::-1].cumsum(-1)[...,::-1]
    fp=suffix[17].sum(0)[1:]
    k=int(np.argmin(np.abs(fp-target_fp)))
    def matrix(data):
        h=data['margin_hist'];predicted=h[...,k+1:].sum(-1)
        total=h.sum((1,2));cm=np.zeros((18,18),dtype=np.int64)
        cm[:,:17]=predicted;cm[:,17]=total-predicted.sum(-1)
        return cm
    return dict(threshold=float(EDGES[k]),target_train_FP=int(target_fp),
        train={k:float(v) for k,v in cm_metrics(matrix(train)).items()},
        val={k:float(v) for k,v in cm_metrics(matrix(val)).items()},
        train_FP_error=int(fp[k]-target_fp))


def bootstrap(base,other,seed=928,repetitions=1000):
    assert [n for n,a in base]==[n for n,a in other]
    n=len(base);rng=np.random.default_rng(seed)
    weights=rng.multinomial(n,np.full(n,1/n),size=repetitions)
    def resample(rows):
        cms=np.stack([a['confusion'].reshape(-1) for _,a in rows])
        values=cm_metrics((weights@cms).reshape(-1,18,18))
        # Fixed GT targets are shared across arms; no conditioning on new FNs.
        j=FIELDS.index('within_2sigma')
        total=np.array([a['field_sum'][:,j].sum() for _,a in rows])
        count=np.array([a['field_n'][:,j].sum() for _,a in rows])
        values['n_2sigma']=(weights@total)/np.maximum(weights@count,1)
        return values
    b=resample(base);o=resample(other)
    keys=('mIoU','IoU','FN_rate','FPR','n_2sigma')
    return {k:np.quantile(o[k]-b[k],[.025,.975]).tolist() for k in keys}


def report():
    protocol=json.loads((RUN/'protocol.json').read_text())
    (RUN/'aggregate').mkdir(exist_ok=True)
    arrays={};scenes={};rows=[];summary={}
    for name in NAMES:
        complete=json.loads((RUN/'eval'/name/'complete.json').read_text())
        assert complete['status']=='complete' and set(complete['splits'])=={'train','val'}
        summary[name]={}
        for split in ('train','val'):
            a,s,r=aggregate(name,split,protocol['expected'][split])
            arrays[name,split]=a;scenes[name,split]=s;rows.extend(r)
            summary[name][split]=metrics(a)
    # The added control must match the same complete data/RNG sequence too.
    traces={n:[json.loads(x) for x in (RUN/'train'/f'{n}_s928/trace.jsonl').read_text().splitlines()]
            for n in TREATMENTS}
    reference=[(r['index'],r['fingerprint']) for r in traces['standard']]
    for name,trace in traces.items():
        assert len(trace)==1024 and [(r['index'],r['fingerprint']) for r in trace]==reference
    base=np.load(RUN/'train/standard_s928/initial_forward.npz')
    other=np.load(RUN/'train'/f'{EXTRA}_s928/initial_forward.npz')
    initial_error={k:float(np.abs(base[k]-other[k]).max()) for k in base.files}
    for k in base.files:assert np.allclose(base[k],other[k],rtol=1e-5,atol=2e-5)
    write(RUN/'selective_pairing_audit.json',dict(status='passed',microbatches=1024,
        all_data_fingerprints_equal=True,initial_max_abs_errors=initial_error))
    optimizer_diagnostics={}
    for name,trace in traces.items():
        norms=np.array([r['grad_norm'] for r in trace if 'grad_norm' in r])
        assert len(norms)==128 and np.isfinite(norms).all()
        optimizer_diagnostics[name]=dict(updates=128,median_preclip_norm=float(np.median(norms)),
            maximum_preclip_norm=float(norms.max()),clipped_fraction=float(np.mean(norms>35)))
    targeted=traces[EXTRA]
    optimizer_diagnostics[EXTRA].update(mean_eligible_fraction=float(np.mean([
        r['losses']['selective_eligible_fraction'] for r in targeted])),
        mean_blocked_fraction=float(np.mean([r['losses']['selective_blocked_fraction'] for r in targeted])),
        max_forward_error=max(r['losses']['selective_forward_error'] for r in targeted))
    write(RUN/'optimizer_diagnostics.json',optimizer_diagnostics)
    # All models must have exactly the same GT denominators, not merely the
    # same number of scenes.
    for split in ('train','val'):
        reference=arrays['common',split]['confusion'].sum(1)
        for name in NAMES:
            assert np.array_equal(reference,arrays[name,split]['confusion'].sum(1))
    with (RUN/'scene_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    intervals={name:{split:bootstrap(scenes['standard',split],scenes[name,split])
        for split in ('train','val')} for name in COMPARISONS}
    calibration={name:calibrated(arrays[name,'train'],arrays[name,'val'],
        summary['standard']['train']['FP']) for name in TREATMENTS}
    write(RUN/'summary.json',summary);write(RUN/'scene_bootstrap.json',intervals)
    write(RUN/'train_only_calibration.json',calibration)
    with (RUN/'global_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['name','split',*summary['common']['train']]);w.writeheader()
        for name in NAMES:
            for split in ('train','val'):w.writerow(dict(name=name,split=split,**summary[name][split]))
    cls=[r for name in NAMES for split in ('train','val')
         for r in class_rows(name,split,arrays[name,split]['confusion'])]
    with (RUN/'class_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(cls[0]));w.writeheader();w.writerows(cls)
    write(RUN/'integrity_audit.json',dict(status='passed',models=NAMES,
        frames_per_model=34149,scenes_per_model=850,total_evaluation_frame_visits=34149*len(NAMES),
        all_frame_indices_exact=True,all_GT_class_denominators_equal=True,
        formal_pairing=json.loads((RUN/'train/pairing_audit.json').read_text()),
        protocol_sha256=sha(RUN/'protocol.json')))
    make_document(summary,intervals,calibration)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def make_document(s,intervals,calibration):
    def table(columns,entries):
        return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',
            *['| '+' | '.join(map(str,row))+' |' for row in entries]])
    perf=table(['模型','集合','mIoU ↑','占有 IoU ↑','漏检率 ↓','假阳性率 ↓','语义错分率 ↓'],
        [[LABELS[n],{'train':'训练','val':'验证'}[sp],*[f'{s[n][sp][k]:.3f}%' for k in
          ('mIoU','IoU','FN_rate','FPR','semantic_error')]] for n in NAMES for sp in ('train','val')])
    geometry=table(['模型','集合','1σ GS 数','2σ GS 数','无 1σ','无 2σ','主要 GS 距离'],
        [[LABELS[n],{'train':'训练','val':'验证'}[sp],f'{s[n][sp]["n_1sigma"]:.3f}',
          f'{s[n][sp]["n_2sigma"]:.3f}',f'{s[n][sp]["no_1sigma"]:.2f}%',
          f'{s[n][sp]["no_2sigma"]:.2f}%',f'{s[n][sp]["dominant_mahalanobis"]:.3f}σ']
          for n in NAMES for sp in ('train','val')])
    gs=table(['模型','集合','语义幅度 A','不透明度 α','总证据 b','平均 scale／米','最大轴近上限'],
        [[LABELS[n],{'train':'训练','val':'验证'}[sp],*[f'{s[n][sp][k]:.4f}' for k in
          ('GS_sum_A','GS_sum_alpha','GS_sum_b','GS_sum_sigma')],f'{s[n][sp]["GS_maxscale_ge_03999"]:.2f}%']
          for n in NAMES for sp in ('train','val')])
    raw=table(['模型（验证集）','最大轴导数严格为零','全部尺度轴导数为零','最大轴相对导数均值','opacity 相对导数均值'],
        [[LABELS[n],f'{s[n]["val"]["raw_largest_scale_axis_zero"]:.3f}%',
          f'{s[n]["val"]["raw_all_scale_axes_zero"]:.3f}%',
          f'{s[n]["val"]["raw_largest_axis_jacobian_relative"]:.3g}',
          f'{s[n]["val"]["raw_opacity_jacobian_relative"]:.3g}'] for n in NAMES])
    uncertainty=table(['相对标准训练','验证 ΔmIoU／百分点，95% 区间','验证 Δ2σ 覆盖数，95% 区间'],
        [[LABELS[n],f'[{intervals[n]["val"]["mIoU"][0]:+.3f}, {intervals[n]["val"]["mIoU"][1]:+.3f}]',
          f'[{intervals[n]["val"]["n_2sigma"][0]:+.3f}, {intervals[n]["val"]["n_2sigma"][1]:+.3f}]'] for n in COMPARISONS])
    cal=table(['模型','训练选择的阈值','训练 FP 匹配误差','验证 mIoU','验证漏检率','验证假阳性率'],
        [[LABELS[n],f'{calibration[n]["threshold"]:+.2f}',f'{calibration[n]["train_FP_error"]:+,}',
          *[f'{calibration[n]["val"][k]:.3f}%' for k in ('mIoU','FN_rate','FPR')]] for n in TREATMENTS])
    fit=table(['模型','完整训练集加权 CE ↓','验证二值 AP（直方图近似）↑','验证 FP／GT 前景','验证 FP／预测前景'],
        [[LABELS[n],f'{s[n]["train"]["CE_weighted"]:.4f}',*[f'{s[n]["val"][k]:.3f}%' for k in
          ('binary_AP_hist','FP_over_GTfg','FP_over_predfg')]] for n in NAMES])
    text=f'''# 几何覆盖与语义幅度补偿：训练因果验证结果

> 执行状态：四组基本分叉与一组针对性控制的训练、六个检查点的完整数据评估已完成。每个检查点覆盖 700 个训练场景、150 个验证场景，共 34,149 帧。
>
> 结论解读与起点诊断将在人工复核本表及完整性审计后补充。此处不自动把某一指标的上升写成机制成立。

## 1. 这次比较的是什么

全部实验使用当前 **8 帧 tiny + 加性 softplus 渲染**。先用原图像预训练权重建立共同起点：2,048 个 microbatch、256 次 AdamW 更新。五个分叉再从相同模型与优化器状态出发，各训练 1,024 个 microbatch、128 次更新。

分叉阶段共同固定 empty bias 和 empty scalar；原 Set loss、类别区分路径及其他训练条件一致。梯度累积为 8。训练使用完整训练池中的固定随机排列，**不是每组训练一整轮，更不是 30 epoch 收敛训练**。设计细节见[设计报告](gaussian_geometry_causal_design.md)。

| 组别 | 唯一的直接梯度干预 |
|---|---|
| 标准训练 | 保留全部原始路径 |
| 阻断幅度 | Occupancy 不再直接调整语义总幅度及不透明度；仍调整类别份额与几何 |
| 阻断几何 | Occupancy 不再直接调整中心、尺度和旋转；Set 几何监督保留 |
| 同时阻断 | 同时阻断以上两种直接路径；Set loss 与类别份额学习保留 |
| 仅阻断空中心增强 | 只对 camera 内中心为 empty、1σ 核心无可见前景、且 Occupancy 梯度正要增大幅度的 GS，移除这个增强方向；保留降低幅度、类别份额、几何及全部 Set 梯度 |

最后一组在共同起点仍在训练时登记，并在正式针对性训练开始前增加“1σ 核心无前景”条件；原登记与修订均保留。见 `selective_protocol.json`。它补充控制了两个问题：全局阻断幅度会移除 empty CE 正常降低错误证据的路径；中心落在 empty 但核心覆盖前景的 GS 也可能有用。针对性控制保留这些路径与表示。筛选仅用于训练干预，推理与全量评估不使用此类 GT 筛选。

停止某条直接梯度不等于变量不再变化：Set loss、共享特征和 AdamW 状态仍可改变输出。下面同时报告实际的几何与证据变化。

## 2. 原生性能与错误

{perf}

漏检率以 GT 前景为分母；假阳性率以 GT empty 为分母；语义错分率以 GT 前景为分母，只计预测成其他前景类别。三种率不能直接相加解释 mIoU。全部原生性能都使用模型自己的预测，不额外搜索 empty bias。

## 3. 几何支持是否真的发生变化

{geometry}

本表以**全部 camera-valid GT 前景体素**为目标，在模型之间保持同一组 GT，不随新预测的漏检／检出身份改变。1σ、2σ 是马氏距离椭球范围；“主要 GS”是最高前景得分中贡献最大的高斯。没有正贡献的体素不参与距离均值，零覆盖比例单独保留。

## 4. 幅度、opacity 与尺度的实际变化

{gs}

这里 $A$ 是 17 维 softplus 语义之和，$b=\\alpha A$；表中的 $b$ 是先逐 GS 相乘再平均，不等于两个均值的乘积。固定追加的 empty Gaussian 不计入 GS 统计。最大轴近上限表示最大 scale 至少 0.3999 米，**不等同于反向梯度严格为零**。

### 共同起点的一个限制：训练早期已经接近属性上限

为避免由舍入后的物理尺度反推梯度，本轮还直接读取 sigmoid 前的原始输出，统计真实导数：

{raw}

尺度导数除以理论峰值 0.08，不透明度导数除以峰值 0.25，因此相对值越接近零，表示局部转换导数越小。小导数本身不等同于 AdamW 无法更新，必须结合实际输出变化判断；严格为零则会阻断这条局部链式导数。

本次共同起点虽然只有 256 次优化器更新，仍已高度接近属性上限。因此这轮不是“几何各自由度都充分可学习”的理想比较，不能把位置路径和尺度路径的作用混为一谈。新训练轨迹也不能直接代替既有 30 epoch 权重的历史。

## 5. 拟合程度及阈值解释

{fit}

加权 CE 是完整训练集 camera-valid 体素的加权负对数似然，乘原 CE 系数 10。它衡量 Occupancy 的 CE 拟合，不是包含 Lovasz 和 Set 的完整训练目标。二值 AP 使用 logit margin 的 0.02 分箱近似，不评价前景语义分类。

另用完整训练集选择阈值，使各组的训练 FP 数接近标准组的原生 FP 数，再原样用于验证集：

{cal}

阈值作用于“最高前景得分减 empty 得分”，不是新的模型训练。匹配误差因 0.02 logit 分箱而存在；验证集的 FP 数不保证相同。该表辅助辨认 operating point 的作用，不替代原生结果。

## 6. 场景层面的不确定性

{uncertainty}

区间来自同一 split 内完整场景的配对 bootstrap，重复 1,000 次。它描述固定训练结果下的场景变化，**不表示多训练种子重复已经完成**。

## 7. 本轮证据的边界

这轮可以估计：在共同早期起点、固定 empty 参数和当前训练预算下，移除一条直接梯度路径会带来什么变化。它没有单独估计“最终漏检中多少比例由幅度补偿导致”，也没有完成对 30 epoch 成熟模型的全部训练因果归因。

若某组的训练拟合明显下降，不能只凭它的泛化差距缩小，就称它改善了泛化。反之，干预未改善性能也不否定此前完整数据中观测到的几何覆盖差异。需要结合起点是否充分学习、尺度约束、实际幅度变化及核心覆盖一起判断。

## 8. 复核与原始数据

- 梯度投影的独立公式校验，以及真实帧上的梯度与恒等前向校验均通过。
- 四组基本分叉逐 microbatch 的数据指纹与初始前向核对见 `train/pairing_audit.json`；针对性控制另作同样的配对核对。
- 每个检查点均评估 34,149 帧；全部帧索引、场景数和 GT 类别分母一致，见 `integrity_audit.json`。
- 全局数值：`work_dirs/gaussian_geometry_causal/summary.json`、`global_metrics.csv`。
- 每场景数值：`scene_metrics.csv`；覆盖直方图、类别距离分层及计数保存在各场景 `.npz`。
- 阈值诊断：`train_only_calibration.json`；场景区间：`scene_bootstrap.json`。

本轮未执行的预留组 `no_semantic_amplitude` 不列入性能表。两个更新的配对 pilot 用于实现校验，没有混入正式训练结果。
'''
    (ROOT/'analysis/gaussian_geometry_causal_results.md').write_text(text)


if __name__=='__main__':report()
