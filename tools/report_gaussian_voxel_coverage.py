"""Aggregate complete scene coverage statistics, export tables and readable report."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.stat_gaussian_voxel_coverage import RUN, OLD, GROUPS, COUNT_NAMES, FIELDS, FLAGS, HISTOGRAMS, COUNT_CAP
from tools.stat_boundary_gaussians import read, write


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def quantile(hist, q, edges=None):
    if hist.sum() == 0:
        return None
    index = int(np.searchsorted(np.cumsum(hist), q*hist.sum(), side='left'))
    if edges is None:
        return index if index <= COUNT_CAP else None
    if index == 0:
        return [None, float(edges[0])]
    if index >= len(edges):
        return [float(edges[-1]), None]
    return [float(edges[index-1]), float(edges[index])]


def summarize(a):
    result = {}
    for i, group in enumerate(GROUPS):
        n = int(a['voxels'][i])
        result[group] = dict(voxels=n,
            means={k:float(a['field_sum'][i,j]/a['field_n'][i,j]) if a['field_n'][i,j] else None for j,k in enumerate(FIELDS)},
            valid_counts={k:int(a['field_n'][i,j]) for j,k in enumerate(FIELDS)},
            flags={k:dict(count=int(a['flags'][i,j]), fraction=float(a['flags'][i,j]/n) if n else None) for j,k in enumerate(FLAGS)},
            counts={k:dict(mean=float(a['field_sum'][i,j]/n) if n else None,
                maximum=int(a['count_max'][i,j]), overflow=int(a['count_hist'][i,j,-1]),
                **{f'q{int(100*q)}':quantile(a['count_hist'][i,j], q) for q in (.25,.5,.75,.9,.95)})
                for j,k in enumerate(COUNT_NAMES)},
            positions={k:{f'q{int(100*q)}_interval':quantile(a['hist_'+k][i], q, edges)
                           for q in (.25,.5,.75,.9,.95)} for k,edges in HISTOGRAMS.items()})
    return result


def aggregate(partial=False):
    protocol = read(RUN/'protocol.json')
    totals, completed, scene_rows, count_rows, histogram_rows = {}, {}, [], [], []
    scene_arrays = {}
    identities = set()
    max_error = 0.
    for split in ('train','val'):
        total = None
        completed[split] = dict(scenes=0, frames=0)
        scene_arrays[split] = []
        for path in sorted((RUN/'scenes'/split).glob('*.json')):
            meta = read(path)
            with np.load(path.with_suffix('.npz')) as z:
                a = {k:z[k] for k in z.files}
            assert meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
            assert meta['frames'] == len(meta['records']) == len(meta['indices'])
            assert meta['indices'] == [r['index'] for r in meta['records']]
            old = read(OLD/'scenes'/split/path.name)
            assert meta['indices'] == old['indices']
            for r, previous in zip(meta['records'], old['records']):
                assert r['sample_token'] == previous['sample_token']
                key = (split, r['sample_token'])
                assert key not in identities
                identities.add(key)
                cm = np.asarray(previous['confusion'][0])
                assert r['voxels'] == [int(cm[:17,17].sum()), int(cm[:17,:17].sum())]
                max_error = max(max_error, r['maximum_winner_mass_relative_error'])
            np.testing.assert_array_equal(a['voxels'], np.asarray([r['voxels'] for r in meta['records']]).sum(0))
            np.testing.assert_array_equal(a['count_hist'].sum(2), np.broadcast_to(a['voxels'][:,None], (2,len(COUNT_NAMES))))
            np.testing.assert_array_equal(a['class_range_voxels'].sum((1,2)), a['voxels'])
            np.testing.assert_array_equal(a['class_range_n'].sum((1,2)), a['field_n'])
            np.testing.assert_allclose(a['class_range_sum'].sum((1,2)), a['field_sum'], rtol=1e-11, atol=1e-8)
            np.testing.assert_array_equal(a['count_distance_joint'].sum((1,2)), a['voxels'])
            assert (a['flags'] <= a['voxels'][:,None]).all()
            assert (a['field_n'] <= a['voxels'][:,None]).all()
            for key,value in a.items():
                assert np.isfinite(value).all(), (path,key)
            for key in HISTOGRAMS:
                np.testing.assert_array_equal(a['hist_'+key].sum(1), a['field_n'][:,FIELDS.index(key)])
            if total is None:
                total = {k:np.zeros_like(v) for k,v in a.items()}
            for key in a:
                if key == 'count_max':
                    total[key] = np.maximum(total[key], a[key])
                else:
                    total[key] += a[key]
            completed[split]['scenes'] += 1
            completed[split]['frames'] += meta['frames']
            scene_arrays[split].append((a['field_sum'],a['field_n']))
            for group,g in summarize(a).items():
                scene_rows.append(dict(split=split,scene=meta['scene_name'],group=group,voxels=g['voxels'],
                    **g['means'],**{'fraction_'+k:v['fraction'] for k,v in g['flags'].items()}))
        totals[split] = total
    complete = completed == protocol['expected']
    if not partial:
        assert complete, completed
    if any(x is None for x in totals.values()):
        result = dict(status='partial', completed=completed,
            statistics={split:summarize(a) for split,a in totals.items() if a is not None},
            maximum_winner_mass_relative_error=max_error)
        write(RUN/'partial_summary.json',result)
        return result
    totals['total'] = {k:(np.maximum(totals['train'][k],totals['val'][k]) if k=='count_max'
                         else totals['train'][k]+totals['val'][k]) for k in totals['train']}
    summary = dict(status='complete' if complete else 'partial', completed=completed,
        checkpoint_sha256=protocol['checkpoint_sha256'],
        statistics={s:summarize(a) for s,a in totals.items()},
        maximum_winner_mass_relative_error=max_error)
    write(RUN/('summary.json' if complete else 'partial_summary.json'), summary)
    if not complete:
        return summary
    for split,a in totals.items():
        for i,group in enumerate(GROUPS):
            g=summary['statistics'][split][group]
            for key,row in g['counts'].items():
                count_rows.append(dict(split=split,group=group,metric=key,**row))
            for j,key in enumerate(COUNT_NAMES):
                for b,value in enumerate(a['count_hist'][i,j]):
                    histogram_rows.append(dict(split=split,group=group,metric=key,bin=b,
                        interpretation='>=1025' if b==COUNT_CAP+1 else str(b),voxels=int(value)))
    csv_write(RUN/'scene_metrics.csv',scene_rows)
    csv_write(RUN/'count_quantiles.csv',count_rows)
    csv_write(RUN/'count_histograms.csv',histogram_rows)
    posrows=[];conditional=[];strata_rows=[]
    for split,a in totals.items():
        for i,group in enumerate(GROUPS):
            for key,edges in HISTOGRAMS.items():
                for j,value in enumerate(a['hist_'+key][i]):
                    posrows.append(dict(split=split,group=group,metric=key,
                        lower='' if j==0 else edges[j-1],
                        upper='' if j>=len(edges) else edges[j],voxels=int(value)))
            for c in range(17):
                for r in range(4):
                    nn=a['class_range_n'][i,c,r]
                    strata_rows.append(dict(split=split,group=group,gt_class=c,range_bin=r,
                        voxels=int(a['class_range_voxels'][i,c,r]),
                        **{k:float(a['class_range_sum'][i,c,r,j]/nn[j]) if nn[j] else None for j,k in enumerate(FIELDS)}))
        labels=('0','1','2-3','4-7','8-15','16-31','32-63','>=64')
        for b,label in enumerate(labels):
            miss,detected=a['count_distance_joint'][:,b].sum(1)
            conditional.append(dict(split=split,within_2sigma_count=label,miss=int(miss),detected=int(detected),
                                    miss_fraction=float(miss/(miss+detected)) if miss+detected else None))
    csv_write(RUN/'position_histograms.csv',posrows)
    csv_write(RUN/'conditional_miss_rates.csv',conditional)
    csv_write(RUN/'class_range_metrics.csv',strata_rows)
    region_rows=[]
    for split,a in totals.items():
        for i,group in enumerate(GROUPS):
            sums=a['field_sum'][i,:4]
            pairs=np.array([sums[1],sums[2]-sums[1],sums[3]-sums[2],sums[0]-sums[3]])
            dominant=a['count_distance_joint'][i].sum(0)
            for j,label in enumerate(('0to1','1to2','2to3','over3')):
                region_rows.append(dict(split=split,group=group,region=label,
                    voxel_gaussian_pairs=int(round(pairs[j])),
                    pair_fraction=float(pairs[j]/sums[0]),
                    dominant_voxels=int(dominant[j]),
                    dominant_fraction_among_owned=float(dominant[j]/dominant[:4].sum())))
    csv_write(RUN/'position_regions.csv',region_rows)
    with (RUN/'global_arrays.npz').open('wb') as f:
        np.savez_compressed(f,**{split+'__'+k:v for split,a in totals.items() for k,v in a.items()})
    bootstrap={}
    rng=np.random.default_rng(928)
    for split,items in scene_arrays.items():
        sums=np.stack([x[0] for x in items]);ns=np.stack([x[1] for x in items])
        weights=rng.multinomial(len(items),np.full(len(items),1/len(items)),size=1000).astype(float)
        draws_s=(weights@sums.reshape(len(items),-1)).reshape(1000,2,len(FIELDS))
        draws_n=(weights@ns.reshape(len(items),-1)).reshape(1000,2,len(FIELDS))
        means=draws_s/np.maximum(draws_n,1)
        bootstrap[split]={k:dict(detected_minus_miss_interval95=np.quantile(means[:,1,j]-means[:,0,j],[.025,.975]).tolist())
                          for j,k in enumerate(FIELDS)}
    write(RUN/'scene_bootstrap.json',dict(repetitions=1000,seed=928,unit='paired complete scenes; pooled conditional means',statistics=bootstrap))
    matched={}
    train_weights=totals['train']['class_range_voxels'].sum(0).astype(float)
    for key in ('native','within_1sigma','within_2sigma','within_3sigma','dominant_mahalanobis',
                'contribution_weighted_mahalanobis','effective_contributors','share_over3'):
        j=FIELDS.index(key)
        common=np.ones((17,4),dtype=bool)
        for split in ('train','val'):
            common &= (totals[split]['class_range_n'][:,:,:,j]>0).all(0)
        weights=train_weights*common
        weights/=weights.sum()
        matched[key]={}
        for split in ('train','val'):
            matched[key][split]={group:dict(mean=float(np.sum(weights*np.divide(
                totals[split]['class_range_sum'][i,:,:,j],totals[split]['class_range_n'][i,:,:,j],
                out=np.zeros((17,4)),where=totals[split]['class_range_n'][i,:,:,j]>0))),
                retained_valid_fraction=float(totals[split]['class_range_n'][i,:,:,j][common].sum()/
                                             totals[split]['class_range_n'][i,:,:,j].sum()))
                for i,group in enumerate(GROUPS)}
    write(RUN/'matched_class_range.json',dict(reference='train GT class x horizontal range distribution; common valid strata in all four groups',statistics=matched))
    write(RUN/'integrity_audit.json',dict(status='passed',completed=completed,unique_frame_identities=len(identities),
        previous_frame_outcomes_exact=True,histogram_and_stratum_totals_exact=True,
        maximum_winner_mass_relative_error=max_error,
        total_voxels={s:a['voxels'].tolist() for s,a in totals.items()}))
    return summary


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|---|'+'---:|'*(len(headers)-1),
                     *['| '+' | '.join(map(str,r))+' |' for r in rows]])


def pct(x):
    return '—' if x is None else f'{100*x:.2f}%'


def approx(interval):
    if interval is None:
        return '—'
    lo,hi=interval
    if hi is None:
        return f'≥{lo:g}'
    if lo is None:
        return f'<{hi:g}'
    return f'≈{(lo+hi)/2:.2f}'


def write_report(s):
    assert s['status']=='complete'
    st=s['statistics']
    labels={'train':'训练集','val':'验证集'}
    group_labels={'miss':'漏检','detected_occupied':'检出'}
    va,vb=[st['val'][g] for g in GROUPS]
    lines=['# 漏检与检出体素受到多少个高斯影响：全数据结果','',
        f"**验证集的直接答案：漏检体素平均受到 {va['counts']['native']['mean']:.2f} 个 GS 的原生正贡献影响，"
        f"检出体素为 {vb['counts']['native']['mean']:.2f} 个；但在 1σ 内分别只有 {va['counts']['within_1sigma']['mean']:.2f} 个与 "
        f"{vb['counts']['within_1sigma']['mean']:.2f} 个。主要贡献高斯到体素的平均距离分别为 "
        f"{va['means']['dominant_mahalanobis']:.2f}σ 与 {vb['means']['dominant_mahalanobis']:.2f}σ。**",'',
        '**完整统计已完成：700 个训练场景、150 个验证场景，共 34,149 帧。** 对象是 camera mask 内全部 GT 前景体素；检出组包括类别分错但仍预测为占有的体素。模型、预处理与此前一致，逐帧漏检／检出计数精确复现此前全量结果。','',
        '本报告分别回答“有几个 GS 到达”“有几个 GS 在较强的核心范围内”“体素处于主要贡献 GS 的什么位置”。统计定义见[设计文档](gaussian_voxel_coverage_full_design.md)。','',
        '## 1. 最直接的数量比较','',
        '原生数量包含 FP32 下仍有正贡献的极弱尾部；1σ、2σ、3σ 数量按高斯自身的尺度和旋转计算，均限制在原生渲染范围内。表内均值按体素等权。','']
    for split in ('val','train'):
        rows=[]
        for key,label in [('native','原生正贡献范围'),('within_1sigma','1σ 内'),('within_2sigma','2σ 内'),('within_3sigma','3σ 内'),
                          ('at_least_1pct','单个贡献至少占 1%'),('at_least_10pct','单个贡献至少占 10%')]:
            a,b=[st[split][g]['counts'][key] for g in GROUPS]
            rows.append([label,f"{a['mean']:.2f}",a['q50'],f"{b['mean']:.2f}",b['q50']])
        lines += ['### '+labels[split],'',table(['范围／贡献门槛','漏检：平均 GS 数','漏检：中位数','检出：平均 GS 数','检出：中位数'],rows),'']
    v=st['val'];a=v['miss']['counts'];b=v['detected_occupied']['counts']
    lines += [f"验证集漏检体素平均受到 **{a['native']['mean']:.2f} 个**原生正贡献 GS 影响，检出体素为 **{b['native']['mean']:.2f} 个**；改看 2σ 内的覆盖，分别为 **{a['within_2sigma']['mean']:.2f} 个**和 **{b['within_2sigma']['mean']:.2f} 个**。",'',
        '![不同覆盖口径下的高斯数量分布](figures/gaussian_voxel_coverage_full/count_cdf.png)','',
        '图中为数量累积分布：同一横坐标下，曲线越高，表示更多体素的覆盖数量不超过该值。均值之外的完整四分位数与 90% 分位数见附录 A。','',
        '## 2. 漏检是完全没有覆盖，还是只有外围覆盖','',
        '下面各比例的分母均为所在组全部体素，不删除零覆盖样本。“没有 2σ 覆盖”表示没有任何 GS 的 2σ 范围到达该体素，并不表示原生尾部完全到达不了。','']
    rows=[]
    for split in ('val','train'):
        for group in GROUPS:
            g=st[split][group]
            rows.append([labels[split]+'：'+group_labels[group],f"{g['voxels']:,}",
                *[pct(g['flags'][k]['fraction']) for k in ('no_native_support','no_1sigma','no_2sigma','no_3sigma')]])
    lines += [table(['集合与组别','体素数','无原生覆盖','无 1σ 覆盖','无 2σ 覆盖','无 3σ 覆盖'],rows),'',
        '## 3. 这些体素位于主要贡献高斯的哪里','',
        '位置使用马氏距离：1σ、2σ、3σ 处的核值分别约为峰值的 0.607、0.135、0.011。距离超过 3σ 的正贡献仍计入统计。主要贡献者按最高前景类别选择；漏检体素也采用同样规则，不用 GT 类别选择贡献者。','']
    rows=[]
    for split in ('val','train'):
        for group in GROUPS:
            g=st[split][group];m=g['means'];p=g['positions']
            rows.append([labels[split]+'：'+group_labels[group],
                f"{m['nearest_mahalanobis']:.2f}",f"{m['dominant_mahalanobis']:.2f}",
                approx(p['dominant_mahalanobis']['q50_interval']),
                f"{m['contribution_weighted_mahalanobis']:.2f}",f"{m['dominant_euclidean']:.3f}"])
    lines += [table(['集合与组别','最近 GS 距离（σ）','主要 GS 距离（σ）','主要 GS 距离中位数','贡献加权距离（σ）','主要 GS 欧氏距离（米）'],rows),'',
        '无对应 GS 的样本无法定义距离，在距离均值中排除；其数量已在上一节完整报告。距离中位数来自宽度 0.1σ 的直方图，使用“≈”标注分箱近似。','',
        '### 所有影响关系：多数 GS 是近处覆盖，还是远处尾部','',
        '将每个体素与影响它的每个 GS 配成一对，下表按这些“体素—GS 对”计数。一个体素受到多个 GS 影响，就对应多对关系；覆盖数量多的体素会贡献更多对数。这张表描述全部影响关系，不按语义得分加权。','']
    with (RUN/'position_regions.csv').open(encoding='utf-8-sig') as f:
        regions=list(csv.DictReader(f))
    rows=[]
    for split in ('val','train'):
        for group in GROUPS:
            selected=[r for r in regions if r['split']==split and r['group']==group]
            rows.append([labels[split]+'：'+group_labels[group],*[pct(float(r['pair_fraction'])) for r in selected]])
    lines += [table(['集合与组别','0～1σ','1～2σ','2～3σ','超过 3σ'],rows),'',
        '### 主要贡献者：真正起主要作用的 GS 位于哪里','',
        '下面只看每个体素的主要贡献高斯，分母为有正前景贡献的体素数。每个体素只计一次，不会因覆盖数量多而获得更高权重。','']
    rows=[]
    for split in ('val','train'):
        for group in GROUPS:
            selected=[r for r in regions if r['split']==split and r['group']==group]
            rows.append([labels[split]+'：'+group_labels[group],*[pct(float(r['dominant_fraction_among_owned'])) for r in selected]])
    lines += [table(['集合与组别','0～1σ','1～2σ','2～3σ','超过 3σ'],rows),'',
        '### 证据份额：每个部位实际贡献了多少得分','',
        '下表给出每个位置区间实际提供的前景证据份额。先在每个体素内归一化，再按有正前景贡献的体素等权平均；每行合计约为 100%。','']
    rows=[]
    for split in ('val','train'):
        for group in GROUPS:
            m=st[split][group]['means']
            rows.append([labels[split]+'：'+group_labels[group],*[pct(m[k]) for k in ('share_0to1','share_1to2','share_2to3','share_over3')]])
    lines += [table(['集合与组别','0～1σ','1～2σ','2～3σ','超过 3σ'],rows),'',
        '![不同位置区间提供的前景证据份额](figures/gaussian_voxel_coverage_full/radial_evidence.png)','',
        '## 4. 很多个高斯，是否等于很强的有效覆盖','',
        '有效贡献者数量衡量当前证据的集中程度：一个 GS 独占全部证据时为 1，十个 GS 等量贡献时为 10。它不直接衡量得分强弱，必须和核权重和、前景／empty 得分比一起看。','']
    rows=[]
    for split in ('val','train'):
        for group in GROUPS:
            m=st[split][group]['means']
            rows.append([labels[split]+'：'+group_labels[group],
                f"{m['kernel_mass']:.2f}",f"{m['opacity_mass']:.2f}",
                f"{m['effective_contributors']:.2f}",pct(m['dominant_share']),
                f"{m['foreground_empty_ratio']:.3f}"])
    lines += [table(['集合与组别','几何核权重和','带不透明度的核权重和','有效贡献者数量','最大单个贡献份额','最高前景／empty'],rows),'',
        '这些是各指标的算术平均，不能把几个均值相乘来还原平均前景得分。此前支持区域报告的因子表使用几何平均，两种口径不能直接对照数值高低。有效数量、单个贡献份额在存在正前景贡献时定义；核权重和及得分比保留零覆盖体素。','',
        '“至少贡献 10% 的 GS 数量”也不能单独表示覆盖好坏：证据集中在少数 GS 时，该数量可能较高；证据由许多 GS 分担时，单个份额反而更小。它应结合有效数量与总得分解释。','',
        '![2σ 覆盖数量与漏检率的关系](figures/gaussian_voxel_coverage_full/conditional_miss.png)','',
        '图中每个覆盖数量区间的分母是该区间全部 GT 前景体素；它描述相关性，不是给同一体素增加 GS 的因果实验。','',
        '## 5. 类别和距离组成是否影响比较','',
        '下表统一使用训练集 GT 类别 × 水平距离的分层权重，仅保留训练／验证、漏检／检出四组均有有效样本的格子。它减少类别和距离组成差异，但不控制遮挡、物体大小和边界位置等因素。','']
    matched=read(RUN/'matched_class_range.json')['statistics']
    rows=[]
    for key,label in [('native','原生 GS 数'),('within_2sigma','2σ 内 GS 数'),('dominant_mahalanobis','主要贡献者距离（σ）'),
                      ('effective_contributors','有效贡献者数量'),('share_over3','超过 3σ 的证据份额')]:
        fmt=pct if key=='share_over3' else lambda x:f'{x:.2f}'
        rows.append([label,*[fmt(matched[key][sp][g]['mean']) for sp in ('train','val') for g in GROUPS]])
    lines += [table(['对齐后的指标','训练漏检','训练检出','验证漏检','验证检出'],rows),'',
        '各指标保留比例及全部分层均值见数据文件，不把分层对齐视为已经消除全部混杂。','',
        '## 6. 如何据此解释机制','',
        f"验证集中，{pct(va['flags']['no_1sigma']['fraction'])} 的漏检体素没有任何 1σ 覆盖，"
        f"检出组为 {pct(vb['flags']['no_1sigma']['fraction'])}。但漏检体素在 2σ 以内仍获得了平均 "
        f"{pct(va['means']['share_0to1']+va['means']['share_1to2'])} 的前景证据，不能将全部漏检概括为“只受到 3σ 外的尾部影响”。",'',
        '在当前验证集上，漏检组的原生覆盖数量、核心覆盖数量和核权重和都更低，主要贡献者也更偏向高斯外围。这支持把关注点从“有没有高斯到达”进一步转向“到达的覆盖有多强、处于哪个部位”。','',
        '这轮统计可以定位“有多少 GS 到达”和“到达的是中心还是尾部”，并区分原生数量与实际贡献数量。它仍不能单独证明中心优化不足由语义幅度补偿造成，也不能证明减少某类 GS、重新分配预算后必然降低漏检。','',
        f"同时，验证集仍有 {pct(1-va['flags']['no_2sigma']['fraction'])} 的漏检体素至少受到一个 GS 的 2σ 覆盖。"
        '这类样本还需要继续区分核心覆盖偏少和覆盖较充分但语义／类别得分不足；不能只凭“存在一个核心 GS”认定几何已经足够。漏检与检出由最终得分定义，分组后的因子差异也不能直接当作各因子的训练因果贡献。','',
        '## 附录 A：完整数量分布','',
        '数量分位数为整数直方图的精确分位数；若某个极端分位数进入大于 1,024 的溢出桶，会单独标记。','']
    rows=[]
    for split in ('val','train'):
        for group in GROUPS:
            for key,label in [('native','原生'),('within_1sigma','1σ'),('within_2sigma','2σ'),('within_3sigma','3σ')]:
                r=st[split][group]['counts'][key]
                rows.append([labels[split]+'：'+group_labels[group],label,r['q25'],r['q50'],r['q75'],r['q90'],r['maximum']])
    lines += [table(['集合与组别','范围','25% 分位','中位数','75% 分位','90% 分位','最大值'],rows),'',
        '## 附录 B：场景不确定性与完整性','',
        '按完整场景配对重采样 1,000 次，种子 928；下表是“检出组均值减去漏检组均值”的 95% 区间，不包含不同训练种子的变化。','']
    boot=read(RUN/'scene_bootstrap.json')['statistics']
    rows=[]
    for key,label in [('native','原生 GS 数'),('within_2sigma','2σ 内 GS 数'),('dominant_mahalanobis','主要贡献者距离（σ）'),
                      ('effective_contributors','有效贡献者数量')]:
        rows.append([label,*['['+', '.join(f'{x:.3f}' for x in boot[sp][key]['detected_minus_miss_interval95'])+']' for sp in ('train','val')]])
    lines += [table(['指标','训练集区间','验证集区间'],rows),'',
        f"- 全部帧身份、逐帧漏检／检出数量和此前结果精确一致；场景总数与直方图、类别距离分层分母核对通过。",
        f"- 全量最高前景得分重建的最大相对误差为 {s['maximum_winner_mass_relative_error']:.2e}（原始得分大于 1e-8 的体素）。",
        '- 独立合成枚举及四个真实帧校验通过；真实帧还与生产渲染核权重和及前景得分核对。数值校验未替代全量统计。','',
        '| 想查看的内容 | 数据入口 |','|---|---|',
        '| 完整精确结果 | [summary.json](../work_dirs/gaussian_voxel_coverage_full/summary.json) |',
        '| 逐场景均值与比例 | [scene_metrics.csv](../work_dirs/gaussian_voxel_coverage_full/scene_metrics.csv) |',
        '| 数量分位数 | [count_quantiles.csv](../work_dirs/gaussian_voxel_coverage_full/count_quantiles.csv) |',
        '| 完整位置直方图 | [position_histograms.csv](../work_dirs/gaussian_voxel_coverage_full/position_histograms.csv) |',
        '| 全部影响关系与主要贡献者的位置分布 | [position_regions.csv](../work_dirs/gaussian_voxel_coverage_full/position_regions.csv) |',
        '| 类别 × 距离统计 | [class_range_metrics.csv](../work_dirs/gaussian_voxel_coverage_full/class_range_metrics.csv) |',
        '| 分层对齐结果 | [matched_class_range.json](../work_dirs/gaussian_voxel_coverage_full/matched_class_range.json) |',
        '| 覆盖数量条件下的漏检率 | [conditional_miss_rates.csv](../work_dirs/gaussian_voxel_coverage_full/conditional_miss_rates.csv) |',
        '| 数值校验 | [pilot.json](../work_dirs/gaussian_voxel_coverage_full/pilot.json) |','']
    (ROOT/'analysis/gaussian_voxel_coverage_full_results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--write-report',action='store_true')
    args=p.parse_args()
    summary=aggregate(args.allow_partial)
    if args.write_report:
        write_report(summary)
    print(json.dumps(dict(status=summary['status'],completed=summary['completed'])))
