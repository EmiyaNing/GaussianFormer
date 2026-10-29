"""Aggregate support-region census, retain per-scene denominators and factors."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.boundary_gaussian_stats import rates

RUN=ROOT/'work_dirs/gaussian_support_regions_full'
SCOPES=('native_camera','core3sigma_camera')
GROUPS=('foreground_mixed','one_foreground_plus_empty','pure_foreground','pure_empty','unsupported')
ZH=('跨前景类别混合','一种前景 + empty','纯前景','纯 empty','无有效支持／未归属')
FIELDS=('log_geometry','log_effective_opacity','log_amplitude','log_concentration','log_empty',
        'log_margin','log_opacity_mass','log_max_foreground','log_opacity_mass_squared',
        'log_amplitude_squared','log_opacity_mass_times_log_amplitude',
        'geometry_mass','opacity_mass','amplitude','concentration','max_foreground','empty')


def read(path):return json.loads(path.read_text())


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))


def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def selected(array,s,groups,outcomes,gt_empty=False):
    result=np.take(np.take(array[s],groups,axis=0),outcomes,axis=1)
    return (result[:,:,17:18] if gt_empty else result[:,:,:17]).sum(axis=(0,1))


def moments(n,sums):
    count=int(n.sum())
    if not count:return dict(voxels=0)
    mean=sums.sum(axis=(0,1))/count
    result=dict(voxels=count,means={k:float(v) for k,v in zip(FIELDS,mean)})
    for name in FIELDS[:8]:
        result['geometric_mean_'+name.removeprefix('log_')]=float(np.exp(mean[FIELDS.index(name)]))
    lo,la=mean[6],mean[2]
    variance_o=max(0,mean[8]-lo*lo);variance_a=max(0,mean[9]-la*la)
    result['log_coverage_amplitude_correlation']=float((mean[10]-lo*la)/np.sqrt(variance_o*variance_a)) if variance_o*variance_a>0 else None
    cell=sums/n.clip(min=1)[...,None]
    variance_o_within=float((n*(cell[...,8]-cell[...,6]**2)).sum())
    variance_a_within=float((n*(cell[...,9]-cell[...,2]**2)).sum())
    covariance_within=float((n*(cell[...,10]-cell[...,6]*cell[...,2])).sum())
    result['within_class_distance_log_coverage_amplitude_correlation']=float(
        covariance_within/np.sqrt(variance_o_within*variance_a_within)) if variance_o_within*variance_a_within>0 else None
    return result


def compare_factors(totals,s,groups,outcomes,gt_empty=False):
    counts={};sums={};base={};result={}
    for split in ('train','val'):
        a=totals[split]
        counts[split]=selected(a['field_counts'],s,groups,outcomes,gt_empty)
        sums[split]=selected(a['field_sums'],s,groups,outcomes,gt_empty)
        base[split]=int(selected(a['strata_counts'],s,groups,outcomes,gt_empty).sum())
        result[split]=moments(counts[split],sums[split])
        result[split]['all_gt_empty_voxels' if gt_empty else 'all_foreground_voxels']=base[split]
        result[split]['valid_field_fraction']=result[split]['voxels']/base[split] if base[split] else None
    common=(counts['train']>0)&(counts['val']>0)
    if not common.any():return result
    weights=np.where(common,counts['train'],0).astype(float);weights/=weights.sum()
    matched={}
    for split in ('train','val'):
        means=sums[split]/counts[split].clip(min=1)[...,None]
        average=(means*weights[...,None]).sum(axis=(0,1))
        matched[split]=dict(log_means={k:float(average[i]) for i,k in enumerate(FIELDS[:8])})
        matched[split]['covered_gt_empty_fraction' if gt_empty else 'covered_foreground_fraction']=float(
            counts[split][common].sum()/base[split]) if base[split] else None
    delta=np.array(list(matched['val']['log_means'].values()))-np.array(list(matched['train']['log_means'].values()))
    matched['val_minus_train_log_factors']={k:float(delta[i]) for i,k in enumerate(FIELDS[:8])}
    matched['val_over_train_factor_ratios']={k.removeprefix('log_'):float(np.exp(delta[i])) for i,k in enumerate(FIELDS[:8])}
    matched['identity_residual']=float(delta[0]+delta[1]+delta[2]+delta[3]-delta[4]-delta[5])
    matched['weighting']='Shared GT class x horizontal-distance cells; training valid-field counts as weights'
    result['matched']=matched
    return result


def compensation_quadrants(totals,s,groups):
    """Training-derived histogram edges; descriptive, never refit predictions."""
    train=np.take(totals['train']['joint_hist'][s],groups,axis=0)[:,:3].sum(axis=(0,1))
    if not train.sum():return {}
    obin=int(np.searchsorted(np.cumsum(train.sum(1)),train.sum()/2))
    abin=int(np.searchsorted(np.cumsum(train.sum(0)),train.sum()/2))
    # Low O is below the median-containing bin; high S is above it.
    # This avoids claiming the coarse histogram identifies an exact median.
    result=dict(low_O_upper_exclusive=2.**(obin-16),high_S_lower_inclusive=2.**(abin+1-16),
                threshold_rule='Training foreground histogram: below O median bin; above S median bin',splits={})
    for split in ('train','val'):
        joint=np.take(totals[split]['joint_hist'][s],groups,axis=0)[:,:3].sum(0)
        selections={'low_O_high_S':(slice(None,obin),slice(abin+1,None)),
                    'low_O_other_S':(slice(None,obin),slice(None,abin+1)),
                    'other_O_high_S':(slice(obin,None),slice(abin+1,None)),
                    'other_O_other_S':(slice(obin,None),slice(None,abin+1))}
        rows={}
        for name,(oi,ai) in selections.items():
            counts=joint[:,oi,ai].sum(axis=(1,2));n=int(counts.sum())
            rows[name]=dict(gt_foreground=n,fraction=float(n/joint.sum()),
                correct_rate=float(counts[0]/n) if n else None,
                wrong_class_rate=float(counts[1]/n) if n else None,
                miss_rate=float(counts[2]/n) if n else None)
        result['splits'][split]=rows
    return result


def aggregate(allow_partial=False):
    protocol=read(RUN/'protocol.json');totals={};meta={};grows=[];erows=[];scene_data={}
    expected=protocol['expected']
    for split in ('train','val'):
        totals[split]=None;meta[split]=dict(scenes=0,frames=0);scene_data[split]=[]
        for path in sorted((RUN/'scenes'/split).glob('*.json')):
            row=read(path)
            assert row['checkpoint_sha256']==protocol['checkpoint_sha256']
            with np.load(path.with_suffix('.npz')) as archive:a={k:archive[k] for k in archive.files}
            if totals[split] is None:totals[split]={k:np.zeros_like(v) for k,v in a.items()}
            for key,value in a.items():totals[split][key]+=value
            gc=a['gaussian_counts'];ec=a['strata_counts'].sum(axis=(-2,-1))
            assert (gc.sum(1)==row['frames']*76800).all()
            np.testing.assert_array_equal(gc,np.sum([r['gaussian_counts'] for r in row['records']],axis=0))
            np.testing.assert_array_equal(ec,np.sum([r['error_counts'] for r in row['records']],axis=0))
            original=read(ROOT/'work_dirs/boundary_gaussian_full/scenes'/split/path.name)
            for s,old_scope in enumerate((1,3)):
                old=original['gaussian_counts'][old_scope]
                assert gc[s,0]==old['boundary_foreground']
                assert gc[s,1]==old['boundary_any']-old['boundary_foreground']
                assert gc[s,2]+gc[s,3]==old['supported']-old['boundary_any']
                assert gc[s,4]==old['predicted']-old['supported']
            meta[split]['scenes']+=1;meta[split]['frames']+=row['frames']
            scene_data[split].append(dict(name=row['scene_name'],g=gc,e=ec))
            for s,scope in enumerate(SCOPES):
                for g,name in enumerate(GROUPS):
                    base=dict(split=split,scene=row['scene_name'],scene_token=row['scene_token'],
                              frames=row['frames'],scope=scope,group=name)
                    grows.append(dict(**base,predicted=int(gc[s].sum()),gaussians=int(gc[s,g]),
                                      gaussian_fraction=float(gc[s,g]/gc[s].sum())))
                    r=rates(ec[s,g]);erows.append(dict(**base,voxel_fraction=float(ec[s,g].sum()/ec[s].sum()),**r))
        if not allow_partial:assert meta[split]==expected[split],(split,meta[split],expected[split])
    if any(v is None for v in totals.values()):
        write(RUN/'partial_summary.json',dict(status='partial',completed=meta))
        return None
    totals['total']={k:totals['train'][k]+totals['val'][k] for k in totals['train']}
    summary=dict(status='complete' if meta==expected else 'partial',completed=meta,
                 checkpoint_sha256=protocol['checkpoint_sha256'],groups=GROUPS,statistics={})
    for split,a in totals.items():
        summary['statistics'][split]={}
        for s,scope in enumerate(SCOPES):
            gc=a['gaussian_counts'][s];ec=a['strata_counts'][s].sum(axis=(-2,-1))
            groups={}
            for g,name in enumerate(GROUPS):
                groups[name]=dict(gaussians=int(gc[g]),gaussian_fraction=float(gc[g]/gc.sum()),
                    gaussian_mean_semantic_sum=float(a['gaussian_sums'][s,g,0]/gc[g]) if gc[g] else None,
                    gaussian_mean_max_semantic=float(a['gaussian_sums'][s,g,1]/gc[g]) if gc[g] else None,
                    gaussian_mean_opacity=float(a['gaussian_sums'][s,g,2]/gc[g]) if gc[g] else None,
                    gaussian_mean_max_scale=float(a['gaussian_sums'][s,g,4]/gc[g]) if gc[g] else None,
                    voxel_fraction=float(ec[g].sum()/ec.sum()),**rates(ec[g]))
            mixed_count=gc[:2].sum();mixed_error=ec[:2].sum(0)
            groups['mixed_total']=dict(gaussians=int(mixed_count),gaussian_fraction=float(mixed_count/gc.sum()),
                voxel_fraction=float(mixed_error.sum()/ec.sum()),**rates(mixed_error))
            summary['statistics'][split][scope]=dict(predicted_gaussians=int(gc.sum()),
                valid_camera_voxels=int(ec.sum()),groups=groups,all=rates(ec.sum(0)))
    summary['factors']={}
    selections={'all_foreground':[0,1,2,3,4],'foreground_mixed':[0],
                'one_foreground_plus_empty':[1],'pure_foreground':[2]}
    for s,scope in enumerate(SCOPES):
        summary['factors'][scope]={}
        for name,groups in selections.items():
            summary['factors'][scope][name]={
                'all_gt_foreground':compare_factors(totals,s,groups,[0,1,2]),
                'correct_foreground_only':compare_factors(totals,s,groups,[0]),
                'miss_only':compare_factors(totals,s,groups,[2]),
                'coverage_amplitude_quadrants':compensation_quadrants(totals,s,groups)}
        summary['factors'][scope]['empty_regions']={name:compare_factors(totals,s,groups,[3,4],True)
            for name,groups in [('all_gt_empty',[0,1,2,3,4]),('foreground_mixed',[0]),
                                ('one_foreground_plus_empty',[1]),('pure_empty',[3])]}
    target='summary.json' if summary['status']=='complete' else 'partial_summary.json'
    write(RUN/target,summary)
    if summary['status']=='complete':
        csv_write(RUN/'scene_gaussian_groups.csv',grows)
        csv_write(RUN/'scene_voxel_errors.csv',erows)
        global_g=[];global_e=[];factor_rows=[]
        for split,scopes in summary['statistics'].items():
            for scope,data in scopes.items():
                for group,r in data['groups'].items():
                    base=dict(split=split,scope=scope,group=group)
                    global_g.append(dict(**base,predicted=data['predicted_gaussians'],gaussians=r['gaussians'],
                        gaussian_fraction=r['gaussian_fraction']))
                    global_e.append(dict(**base,voxel_fraction=r['voxel_fraction'],
                        **{k:v for k,v in r.items() if not k.startswith('gaussian') and k!='voxel_fraction'}))
        for scope,groups in summary['factors'].items():
            for group,stats in groups.items():
                if group=='empty_regions':selections=stats
                else:selections={'all_gt_foreground':stats['all_gt_foreground']}
                for selection,stats in selections.items():
                    if 'matched' not in stats:continue
                    for factor in FIELDS[:8]:
                        matched=stats['matched']
                        factor_rows.append(dict(scope=scope,region=group,selection=selection,factor=factor,
                            raw_train_geometric_mean=float(np.exp(stats['train']['means'][factor])),
                            raw_val_geometric_mean=float(np.exp(stats['val']['means'][factor])),
                            matched_train_geometric_mean=float(np.exp(matched['train']['log_means'][factor])),
                            matched_val_geometric_mean=float(np.exp(matched['val']['log_means'][factor])),
                            matched_val_train_ratio=matched['val_over_train_factor_ratios'][factor.removeprefix('log_')]))
        csv_write(RUN/'global_gaussian_groups.csv',global_g)
        csv_write(RUN/'global_voxel_errors.csv',global_e)
        csv_write(RUN/'matched_factor_details.csv',factor_rows)
        write(RUN/'integrity_audit.json',dict(status='passed',scenes=meta,
            framewise_original_census_exact=True,group_sums_and_stored_frames_exact=True,
            matched_factor_identity_max_residual=max(abs(v['matched']['identity_residual'])
                for scope in summary['factors'].values() for group in scope.values() for v in group.values() if 'matched' in v)))
    return summary


def pct(value):return '—' if value is None else f'{100*value:.3f}%'


def tables(summary,scope):
    stats=summary['statistics'];lines=[]
    lines+=['| GT 支持类型 | 训练高斯占比 | 验证高斯占比 | 训练体素区域占比 | 验证体素区域占比 |',
            '|---|---:|---:|---:|---:|']
    for key,label in [('mixed_total','混合合计'),*zip(GROUPS,ZH)]:
        tr=stats['train'][scope]['groups'][key];va=stats['val'][scope]['groups'][key]
        lines.append(f"| {label} | {pct(tr['gaussian_fraction'])} | {pct(va['gaussian_fraction'])} | {pct(tr['voxel_fraction'])} | {pct(va['voxel_fraction'])} |")
    lines+=['','| 区域 | 训练错分类率 | 验证错分类率 | 训练漏检率 | 验证漏检率 | 训练假阳性率 | 验证假阳性率 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for key,label in [('mixed_total','混合合计'),*zip(GROUPS,ZH)]:
        tr=stats['train'][scope]['groups'][key];va=stats['val'][scope]['groups'][key]
        columns=[]
        for rate in ('wrong_class_rate_gt_foreground','miss_rate_gt_foreground','false_positive_rate_gt_empty'):
            columns.extend([pct(tr[rate]),pct(va[rate])])
        lines.append('| '+label+' | '+' | '.join(columns)+' |')
    return '\n'.join(lines)


def render_report(summary):
    assert summary['status']=='complete'
    core='core3sigma_camera';native='native_camera'
    lines=['# 高斯覆盖了什么：训练／验证的区域比例、错误与补偿机制核查','',
        '> 完整数据统计：700 个训练场景、150 个验证场景，共 34,149 帧。仅统计 camera mask 内有效 GT；冻结同一份 8 帧 tiny 权重，不训练、不调参。','',
        '**结论先行：区域比例和错误退化可以直接确认；“训练通过强语义补偿弱几何，并因此导致验证失败”尚未被证实。** 本次补充的因子统计用于判断哪些现象与该解释相符，哪些需要改写。','',
        '## 1. 阅读结论前，先对齐三个概念','',
        '**高斯占比**的分母是全部预测高斯，每帧 76,800 个，不含固定 empty 高斯。**体素区域占比**的分母是全部有效 camera 体素；按主要贡献高斯归属，每个体素只计一次。两种比例不能互换。','',
        '“混合”分成跨前景类别混合、以及一种前景与 empty 混合。“纯前景”只含一种前景 GT；“纯 empty”只含 empty GT；mask 内没有有效支持的高斯另列。**纯 empty 支持不是错误标签，只有最终预测为前景的空体素才是假阳性。**','',
        '正文先展示 3σ 有效支持；同时附上原生支持结果。实际算子使用整数候选盒，没有马氏距离截断，因此两种口径不完全等同。两者下的最终预测始终是原模型的完整原生渲染结果。','',
        '## 2. 3σ 支持：高斯数量、覆盖区域与错误','',
        '![高斯分组与错误率](figures/gaussian_support_regions_full/support_and_errors.png)','',tables(summary,core),'',
        '**错误率分母：**错分类、漏检除以各组 GT 前景数；假阳性除以各组 GT empty 数。“—”表示该组分母为零，不代表错误率为 0。','',
        '纯前景组的错分类或漏检不矛盾：GT 支持纯净不代表高斯的语义预测正确或足够强；最终结果也由多个高斯共同决定。未归属组在 3σ 下只表示没有正的核心区贡献，原生渲染仍可能有外围贡献。','',
        '### 使用共同分母，观察各组内部错误构成','',
        '下表三项都除以该组全部体素，可直接相加；与上表条件错误率回答的问题不同。','',
        '| 区域 | 集合 | 错分类 / 组内体素 | 漏检 / 组内体素 | 假阳性 / 组内体素 | 总错误 / 组内体素 |',
        '|---|---|---:|---:|---:|---:|']
    for g,label in enumerate(ZH):
        for split,split_zh in [('train','训练'),('val','验证')]:
            r=summary['statistics'][split][core]['groups'][GROUPS[g]]
            vals=[r[k] for k in ('wrong_class_fraction','miss_fraction','false_positive_fraction','total_error_fraction')]
            lines.append('| '+label+' | '+split_zh+' | '+' | '.join(map(pct,vals))+' |')
    bootstrap_path=RUN/'error_scene_bootstrap.json'
    if bootstrap_path.exists():
        b=read(bootstrap_path)
        low,high=b['core3sigma_pure_fg_miss']['val_minus_train_pp_95pct_interval']
        lo_fp,hi_fp=b['core3sigma_pure_empty_fp']['val_minus_train_pp_95pct_interval']
        lines+=['',f'按完整场景分别在训练与验证内重采样 5,000 次（种子 928），纯前景漏检率的“验证 − 训练”差距 95% 区间为 **[{low:.3f}, {high:.3f}] 个百分点**；纯 empty 假阳性差距为 **[{lo_fp:.3f}, {hi_fp:.3f}] 个百分点**。每次先累加分子、分母再计算比例。这些区间只描述固定模型的场景抽样不确定性，不包含训练种子变化。','']
    lines+=['','### 两个集合合并的全局高斯比例','',
        '| 支持类型 | 3σ 支持 | 原生支持 |','|---|---:|---:|']
    for g,label in enumerate(ZH):
        lines.append(f"| {label} | {pct(summary['statistics']['total'][core]['groups'][GROUPS[g]]['gaussian_fraction'])} | {pct(summary['statistics']['total'][native]['groups'][GROUPS[g]]['gaussian_fraction'])} |")
    lines+=['','### 纯 empty 支持高斯本身输出了什么','',
        '下表按高斯等权平均，使用 3σ 支持类型；这里的语义值只包含 17 个前景类别。它与上文按体素贡献加权的幅度 $S$ 是不同统计量。','',
        '| 集合 | 前景语义值之和 | 最高前景语义值 | 不透明度 | 最大尺度，米 |',
        '|---|---:|---:|---:|---:|']
    for split,label in [('train','训练'),('val','验证')]:
        r=summary['statistics'][split][core]['groups']['pure_empty']
        columns=[r[k] for k in ('gaussian_mean_semantic_sum','gaussian_mean_max_semantic','gaussian_mean_opacity','gaussian_mean_max_scale')]
        lines.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in columns)+' |')
    lines+=['','## 3. 原生支持：同一预测下的敏感性对照','',tables(summary,native),'',
        '扩大支持会让更多高斯被归为混合，也会改变体素的主要贡献归属。两表差异是分组变化，不是模型性能变化。','',
        '## 4. “弱几何由强语义补偿”是否有实测依据','',
        '只靠区域错误率无法回答这个问题。此次额外测量了下列因子，全部来自冻结模型：','',
        '| 符号 | 测量定义 | 含义 |','|---|---|---|',
        '| $C$ | 正 opacity 高斯的几何核值之和 | 几何覆盖量 |',
        '| $O$ | opacity × 几何核值之和 | 带不透明度的覆盖权重和 |',
        '| $D$ | 17 个前景渲染得分之和 | 总前景得分 |',
        '| $S=D/O$ | 贡献加权的总语义幅度 | 语义幅度，不是概率 |',
        '| $M$ | 最高前景得分 | 与 empty 竞争的分数 |',
        '| $T=M/D$ | 最高类别占前景得分的比例 | 类别集中度，不是校准概率 |',
        '| $E$ | 校准后的 empty 得分 | 同一位置跨场景固定，但随位置变化 |','',
        '因此存在恒等式：','',r'$$',r'\frac{M}{E}=C\times\frac{O}{C}\times S\times T\times\frac{1}{E}.',r'$$','',
        '$C$ 与 $O$ 直接使用生产渲染算子的辅助通道累积，不替换原预测。下表每项为几何平均数；“对齐”使用共同的 GT 类别 × 水平距离格子，按训练格子权重平均其对数。','',
        '### 全部真实前景：先看未对齐的实测量','',
        '| 因子 | 训练 | 验证 |','|---|---:|---:|']
    f=summary['factors'][core]['all_foreground']['all_gt_foreground']
    factors=[('geometry','几何覆盖 $C$'),('effective_opacity','有效不透明度 $O/C$'),
             ('amplitude','语义幅度 $S$'),('concentration','类别集中度 $T$'),('margin','得分比 $M/E$')]
    for key,label in factors:
        lines.append(f"| {label} | {f['train']['geometric_mean_'+key]:.4f} | {f['val']['geometric_mean_'+key]:.4f} |")
    lines+=['','### 对齐 GT 类别与距离后，验证相对训练的变化','',
        '下表数值为“验证 / 训练”。小于 1 表示验证更低；各列采用同一权重。前四项相乘并除以 empty 比值，等于得分比的变化。','',
        '| 区域 | 几何覆盖 | 有效不透明度 | 语义幅度 | 类别集中度 | empty 得分 | 最高前景 / empty |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in [('all_foreground','全部真实前景'),('foreground_mixed',ZH[0]),
                       ('one_foreground_plus_empty',ZH[1]),('pure_foreground',ZH[2])]:
        m=summary['factors'][core][name]['all_gt_foreground']['matched']
        r=m['val_over_train_factor_ratios']
        lines.append('| '+label+' | '+' | '.join(f"{r[k]:.4f}" for k in
            ('geometry','effective_opacity','amplitude','concentration','empty','margin'))+' |')
    lines+=['','对齐比较保留的样本比例（分母为相应区域全部 GT 前景）：','',
            '| 区域 | 训练保留 | 验证保留 |','|---|---:|---:|']
    for name,label in [('all_foreground','全部真实前景'),('foreground_mixed',ZH[0]),
                       ('one_foreground_plus_empty',ZH[1]),('pure_foreground',ZH[2])]:
        m=summary['factors'][core][name]['all_gt_foreground']['matched']
        lines.append(f"| {label} | {pct(m['train']['covered_foreground_fraction'])} | {pct(m['val']['covered_foreground_fraction'])} |")
    lines+=['','### 真实空区域是否呈现相反变化','',
        '对“纯 empty 支持”的体素区域，使用相同距离对齐方式，检查前景得分是否在验证集反而变强。以下仍是验证 / 训练比值，统计对象是 GT empty；不是前景漏检子集。','',
        '| 区域 | 几何覆盖 | 有效不透明度 | 语义幅度 | 类别集中度 | empty 得分 | 最高前景 / empty |',
        '|---|---:|---:|---:|---:|---:|---:|']
    m=summary['factors'][core]['empty_regions']['pure_empty']['matched'];r=m['val_over_train_factor_ratios']
    lines.append('| 纯 empty 区域 | '+' | '.join(f'{r[k]:.4f}' for k in
        ('geometry','effective_opacity','amplitude','concentration','empty','margin'))+' |')
    lines+=['',f"该比较保留了训练 {pct(m['train']['covered_gt_empty_fraction'])}、验证 {pct(m['val']['covered_gt_empty_fraction'])} 的纯 empty 区域 GT 空体素。",'']
    lines+=['','这里是对数得分差距的描述性分解，不是各项的因果贡献。类别与距离对齐也没有消除遮挡、细粒度位置、边界距离等差异。','',
        '### 训练内部的覆盖—幅度关联','',
        r'比较 $\log O$ 与 $\log S$ 的相关系数；括号内是扣除 GT 类别与距离格子均值后的格内相关。负值表示覆盖较低时幅度往往较高，但不说明该关系一定由训练补偿形成。','',
        '| 样本 | 训练相关系数 | 验证相关系数 |','|---|---:|---:|']
    for selection,label in [('all_gt_foreground','全部真实前景'),('correct_foreground_only','只看正确前景')]:
        d=summary['factors'][core]['all_foreground'][selection]
        def corr(split):
            a=d[split]['log_coverage_amplitude_correlation'];b=d[split]['within_class_distance_log_coverage_amplitude_correlation']
            return f'{a:.4f}（{b:.4f}）' if a is not None and b is not None else '—'
        lines.append(f'| {label} | {corr("train")} | {corr("val")} |')
    q=summary['factors'][core]['all_foreground']['coverage_amplitude_quadrants']
    lines+=['',f"还可直接看低覆盖样本：采用训练分布预先确定的直方图边界，定义低覆盖为 $O<{q['low_O_upper_exclusive']:g}$，高幅度为 $S\\geq {q['high_S_lower_inclusive']:g}$。这些阈值只用于描述统计，不改变模型预测。",'',
        '| 真实前景子集 | 训练样本占比 | 验证样本占比 | 训练分类正确率 | 验证分类正确率 | 训练漏检率 | 验证漏检率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,label in [('low_O_high_S','低覆盖、高幅度'),('low_O_other_S','低覆盖、其余幅度')]:
        tr=q['splits']['train'][key];va=q['splits']['val'][key]
        columns=[tr['fraction'],va['fraction'],tr['correct_rate'],va['correct_rate'],tr['miss_rate'],va['miss_rate']]
        lines.append('| '+label+' | '+' | '.join(map(pct,columns))+' |')
    lines+=['','子集占比除以可计算因子的全部 GT 前景；正确率与漏检率除以该子集 GT 前景。这个粗分箱没有固定每个体素的精确覆盖量、类别集中度或空间位置，因此也不是“只改语义幅度”的干预比较。','']
    positive=summary['factors'][core]['all_foreground']['all_gt_foreground']['matched']['val_over_train_factor_ratios']
    negative=summary['factors'][core]['empty_regions']['pure_empty']['matched']['val_over_train_factor_ratios']
    lines+=['','尤其是“只看正确预测”会自带选择效应：低覆盖样本需要其他因子足够高才会被选入。另外，$S=D/O$ 与 $O$ 共享分母，可能产生数学耦合；类别与距离对齐不能消除这一点。不能把这些负相关直接当作学习到补偿机制的证据。','',
        '## 5. 对原机制表述的核查','',
        '**可以成立的部分是：较低覆盖与较高幅度下仍正确的样本确实存在。** 低覆盖、高幅度组明显优于低覆盖、其余幅度组，而且验证集也保留这种模式。这与乘性得分结构相符，但不是训练集独有的现象。','',
        f"**不能直接成立的是：训练集整体依赖更弱几何、更强语义，因此验证失败。** 对齐类别与距离后，验证真实前景的几何覆盖为训练的 {positive['geometry']:.3f} 倍，语义幅度为 {positive['amplitude']:.3f} 倍，类别集中度为 {positive['concentration']:.3f} 倍。应先看这些实测量，而不是从漏检反推训练依赖某种策略。",'',
        f"**还必须解释空区域中的相反变化。** 在纯 empty 体素区域，验证的几何覆盖为训练的 {negative['geometry']:.3f} 倍、语义幅度为 {negative['amplitude']:.3f} 倍，最高前景 / empty 得分比为 {negative['margin']:.3f} 倍。因此不能用前景证据统一下降解释全部错误。",'',
        '本次更稳妥的描述是：**该加性模型在验证集出现真实前景覆盖减弱与真实空区域前景证据增多并存；覆盖、幅度与类别分配的联合关系发生退化。部分训练样本具有补偿型得分结构，但其是否由训练学得、是否导致泛化失败，尚待受控验证。**','',
        '这仍不是“中心错位已经被证明”：核覆盖变化可能涉及中心、数量、尺度以及条件场景组成。当前分组也由预测支持决定，不是外生固定的难度分层。','',
        '## 6. 核查范围与可复查数据','',
        '- 850 场景全部完成；每帧错误计数精确重现旧统计，五组高斯和五组体素均通过总量检查。',
        '- 原生与 3σ 只改变统计分组，不改变预测。高斯类型在 mask 内由 GT 定义，因此不是模型自己预测的“边界概率”。',
        '- 这次可判断区域比例、错误分布、因子差异和补偿关联；不能仅靠冻结前向证明“训练学会了补偿，而且补偿导致泛化失败”。','',
        '[统计设计](gaussian_support_regions_design.md) · [完整数值与匹配结果](../work_dirs/gaussian_support_regions_full/summary.json) · '
        '[逐场景高斯分组](../work_dirs/gaussian_support_regions_full/scene_gaussian_groups.csv) · '
        '[逐场景体素错误与全部分母](../work_dirs/gaussian_support_regions_full/scene_voxel_errors.csv) · '
        '[完整性检查](../work_dirs/gaussian_support_regions_full/integrity_audit.json)','']
    (ROOT/'analysis/gaussian_support_regions_results.md').write_text('\n'.join(lines))


def plot_summary(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=ROOT/'analysis/figures/gaussian_support_regions_full';directory.mkdir(parents=True,exist_ok=True)
    scope='core3sigma_camera';stats=summary['statistics'];colors=['#3177b7','#e58b32']
    labels=['FG class mix','FG + empty','Pure FG','Pure empty','Unsupported']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    x=np.arange(5)
    for i,(split,label) in enumerate([('train','Train'),('val','Validation')]):
        groups=stats[split][scope]['groups']
        axes[0,0].bar(x+(i-.5)*.36,[100*groups[g]['gaussian_fraction'] for g in GROUPS],.36,color=colors[i],label=label)
    axes[0,0].set(title='Predicted Gaussian support types',ylabel='All predicted Gaussians (%)',xticks=x,xticklabels=labels)
    axes[0,0].legend(frameon=False)
    for ax,metric,title,keep in [(axes[0,1],'wrong_class_rate_gt_foreground','Wrong foreground class',[0,1,2]),
                                (axes[1,0],'miss_rate_gt_foreground','Missed foreground',[0,1,2]),
                                (axes[1,1],'false_positive_rate_gt_empty','False positive',[0,1,3])]:
        for i,split in enumerate(('train','val')):
            values=[stats[split][scope]['groups'][GROUPS[g]][metric] for g in keep]
            ax.bar(np.arange(len(keep))+(i-.5)*.36,np.array(values)*100,.36,color=colors[i])
        ax.set(title=title,ylabel='Conditional error rate (%)',xticks=np.arange(len(keep)),xticklabels=[labels[g] for g in keep])
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen additive renderer | camera mask | 3-sigma support groups')
    for ext in ('png','pdf'):fig.savefig(directory/f'support_and_errors.{ext}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--write-report',action='store_true')
    args=parser.parse_args();summary=aggregate(args.allow_partial)
    if summary:
        if args.write_report:render_report(summary);plot_summary(summary)
        print(json.dumps(dict(status=summary['status'],completed=summary['completed']),ensure_ascii=False))
        for scope in ('core3sigma_camera','native_camera'):
            print(scope);print(tables(summary,scope))
            print('MATCHED FACTOR RATIOS',json.dumps({name:g['all_gt_foreground'].get('matched',{}).get('val_over_train_factor_ratios')
                for name,g in summary['factors'][scope].items() if 'all_gt_foreground' in g},ensure_ascii=False))
