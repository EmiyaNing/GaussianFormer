"""Full-scene audit and readable 1/2/3-sigma boundary comparison."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.extend_small_sigma_boundaries import RUN,COUNT_NAMES,base
from tools.stat_boundary_gaussians import read,write,sha
from tools.report_query_budget import csv_write

REPORT=ROOT/'analysis/small_sigma_boundary_results.md'
FIGURES=ROOT/'analysis/figures/small_sigma_boundaries'


def ratio(a,b):return float(a/b) if b else None
def pct(x):return '—' if x is None else f'{x*100:.3f}%'


def aggregate(partial=False):
    protocol=read(RUN/'protocol.json')
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(base.RUN/'protocol.json')==protocol['primary_protocol_sha256']
    stats,scene_rows,query_rows={},[],[]
    tokens=set()
    for split in ('train','val'):
        counts=np.zeros((3,2,5),np.int64)
        hist=np.zeros((3,2,19),np.int64)
        transitions=np.zeros((2,3,8),np.int64)
        allocation=np.zeros((3,3,2,3),np.int64)
        budgets=np.zeros(3,np.int64)
        frames=scenes=0
        for path in sorted((RUN/'scenes'/split).glob('*.json')):
            meta=read(path)
            original_path=base.RUN/'scenes'/split/path.name
            if not original_path.exists():
                assert partial
                continue
            primary=read(original_path)
            assert meta['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert meta['indices']==primary['indices']
            assert [(r['index'],r['scene_frame']) for r in meta['records']]==[(r['index'],r['scene_frame']) for r in primary['records']]
            for r in primary['records']:
                assert r['sample_token'] not in tokens
                tokens.add(r['sample_token'])
            with np.load(path.with_suffix('.npz')) as z:
                c,h,t,q=z['counts'],z['histogram'],z['transitions'],z['query_counts']
            with np.load(original_path.with_suffix('.npz')) as z:
                qb,metrics=z['boundary_queries'],z['metrics']
            assert q.shape==(meta['frames'],base.Q,3,2,3)
            np.testing.assert_array_equal(q[:,:,2,1,:],qb[:,:,1,:])
            assert (c[...,:2]>=0).all() and (c[...,1]<=c[...,0]).all()
            assert (c[...,:1]>=c[...,2:]).all() and (c[...,1:2]>=c[...,2:]).all()
            assert (c[:-1,:,1:]<=c[1:,:,1:]).all()
            assert (t[...,1:4]==0).all() and (t[...,5]==0).all()
            np.testing.assert_array_equal(t.sum(-1),np.full((2,3),76800*meta['frames']))
            np.testing.assert_array_equal(c[0,:,2:],t[...,7])
            np.testing.assert_array_equal(c[1,:,2:],t[...,6]+t[...,7])
            np.testing.assert_array_equal(c[2,:,2:],t[...,4]+t[...,6]+t[...,7])
            np.testing.assert_array_equal(q.sum((0,1)),c[...,2:])
            np.testing.assert_array_equal(h.sum(-1),c[...,0])
            np.testing.assert_array_equal(h[...,0],c[...,0]-c[...,1])
            for scope,p_scope in enumerate((2,3)):
                for j,key in enumerate(COUNT_NAMES):assert int(c[2,scope,j])==primary['gaussian_counts'][p_scope][key]
                np.testing.assert_array_equal(h[2,scope],primary['gaussian_counts'][p_scope]['class_count_histogram'])
            counts+=c;hist+=h;transitions+=t
            frames+=meta['frames'];scenes+=1
            for r in range(3):
                for s,scope in enumerate(('all_valid_gt','camera_mask')):
                    values={name:int(c[r,s,j]) for j,name in enumerate(COUNT_NAMES)}
                    values.update({name+'_fraction_all':ratio(c[r,s,j],c[r,s,0]) for j,name in enumerate(COUNT_NAMES[2:],2)})
                    values['boundary_any_fraction_supported']=ratio(c[r,s,2],c[r,s,1])
                    scene_rows.append(dict(split=split,scene=primary['scene_name'],frames=meta['frames'],sigma=r+1,scope=scope,**values))
            load=metrics[...,base.query.FIELDS.index('gt_foreground')]
            order=np.argsort(load,axis=1,kind='stable')
            low=np.zeros_like(load,dtype=bool);high=low.copy()
            np.put_along_axis(low,order[:,:base.Q//5],True,axis=1)
            np.put_along_axis(high,order[:,-base.Q//5:],True,axis=1)
            for j,mask in enumerate((low,high,np.ones_like(low))):
                allocation[j]+=q[mask].sum(0,dtype=np.int64)
                budgets[j]+=int(mask.sum())*base.CHILDREN
        if not scenes:continue
        for g,name in enumerate(('low20','high20','all')):
            for r in range(3):
                for s,scope in enumerate(('all_valid_gt','camera_mask')):
                    for k,kind in enumerate(('any','foreground','fg_empty')):
                        query_rows.append(dict(split=split,group=name,sigma=r+1,scope=scope,kind=kind,
                            gaussians=int(budgets[g]),boundary_gaussians=int(allocation[g,r,s,k]),
                            boundary_fraction_group=ratio(allocation[g,r,s,k],budgets[g]),
                            boundary_share_split=ratio(allocation[g,r,s,k],allocation[2,r,s,k])))
        stats[split]=dict(frames=frames,scenes=scenes,counts=counts.tolist(),histogram=hist.tolist(),transitions=transitions.tolist())
    complete=all(s in stats and {k:stats[s][k] for k in ('frames','scenes')}==protocol['expected'][s] for s in ('train','val'))
    if not partial:
        assert complete
        assert len(tokens)==34149
    if stats:
        stats['total']=dict(frames=sum(v['frames'] for v in stats.values()),scenes=sum(v['scenes'] for v in stats.values()),
            **{key:sum(np.array(v[key],dtype=np.int64) for v in stats.values()).tolist() for key in ('counts','histogram','transitions')})
    result=dict(status='complete' if complete else 'partial',statistics=stats,unique_sample_tokens=len(tokens),
        checkpoint_sha256=protocol['checkpoint_sha256'],count_names=COUNT_NAMES,
        audits=['source hashes','scene and sample identity','query 3sigma exact primary parity','class histogram 3sigma parity',
            'all GS denominator','nested boundary counts','four disjoint radial transition groups','query/global counts'],
        reporting_source_sha256=sha(Path(__file__)))
    csv_write(RUN/'scene_sigma_statistics.csv',scene_rows)
    csv_write(RUN/'query_load_sigma_statistics.csv',query_rows)
    write(RUN/'summary.json',result)
    write(RUN/'audit.json',dict(status='passed' if complete else 'partial_passed',checks=result['audits'],
        unique_sample_tokens=len(tokens),scenes=sum(v['scenes'] for k,v in stats.items() if k!='total')))
    return result


def render(result):
    expected=read(RUN/'protocol.json')['expected']
    complete=result['status']=='complete'
    stats={s:a for s,a in result['statistics'].items() if s=='total' and complete or s in expected and {k:a[k] for k in ('frames','scenes')}==expected[s]}
    drawable={s:a for s,a in stats.items() if s in ('train','val')}
    if drawable:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        FIGURES.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for ax,column,title in zip(axes,(2,3),('At least two GT labels (including empty)','At least two foreground classes')):
            for split,a in drawable.items():
                c=np.asarray(a['counts'])[:,1]
                ax.plot([1,2,3],100*c[:,column]/c[:,0],'o-',label=split.title())
            ax.set(xticks=[1,2,3],xticklabels=['1 sigma','2 sigma','3 sigma'],ylim=(0,100),
                xlabel='Strict Mahalanobis support radius',ylabel='All predicted GS (%)',title=title)
            ax.spines[['top','right']].set_visible(False)
            ax.legend()
        fig.suptitle('Small checkpoint | camera mask | same 76,800 GS per frame')
        fig.savefig(FIGURES/'sigma_comparison.png',dpi=180)
        plt.close(fig)
    names=dict(train='训练',val='验证',total='合计')
    lines=['# small：1σ、2σ、3σ 范围内的跨类别高斯','',
        '**状态：完整统计已完成，覆盖 700 个训练场景、150 个验证场景，共 34,149 帧。**' if complete else '**状态：全量运行尚未结束，正文只展示已完整完成的集合。**','',
        '冻结 `opus_gaussian_8f_small.pth`，每帧 1,200 query×64 GS。仅改变 GT 类别混合的判断范围，'
        '模型的渲染和最终 Occupancy 预测保持原状。**主表使用 camera mask，并始终以全部预测 GS 为分母。**','',
        '## 1. 三个范围分别有多少跨类别 GS','',
        '“任意两类”可以包含 empty；“跨前景类别”要求至少两种前景；“前景/empty 接口”要求两者同时出现。'
        '后两项可能重叠，不能直接相加。','',
        '| 集合 | 范围 | 全部 GS 实例 | 有有效支持 | 任意两类/全部 GS | 跨前景类别/全部 GS | 前景与 empty/全部 GS | 任意两类/有支持 GS |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    if 'val' in stats:
        c=np.asarray(stats['val']['counts'])[:,1]
        core_share=ratio(c[0,3],c[2,3])
        headline=['**完整验证集的直接结果：**','',
            '仅统计前景类别之间的混合时，'+ '，'.join(f'{r+1}σ 为 {pct(ratio(c[r,3],c[r,0]))}' for r in range(3))+'。',
            f'3σ 跨前景类别的 GS 中，{pct(core_share)} 在 1σ 内已经跨类，其余在扩展到 2σ 或 3σ 后才出现跨类。'
            '这里判断的是离散 GT 标签集合，不是错误贡献大小。','']
        at=lines.index('## 1. 三个范围分别有多少跨类别 GS')
        lines[at:at]=headline
    for split,a in stats.items():
        c=np.asarray(a['counts'])
        for r in range(3):
            v=c[r,1]
            lines.append(f'| {names[split]} | {r+1}σ | {v[0]:,} | {pct(ratio(v[1],v[0]))} | {pct(ratio(v[2],v[0]))} | {pct(ratio(v[3],v[0]))} | {pct(ratio(v[4],v[0]))} | {pct(ratio(v[2],v[1]))} |')
    if drawable:
        lines+=['','![不同半径下的跨类别 GS 比例](figures/small_sigma_boundaries/sigma_comparison.png)']
    lines+=['','1σ 为高斯核心椭球，2σ、3σ 将三轴半径分别扩大到两倍、三倍；使用实际旋转和尺度的马氏距离，'
        '同时保留生产候选盒与正 FP32 贡献约束。没有有效 GT 支持的 GS 仍计入“全部 GS”，但不计入“有支持 GS”。'
        '判断对象是 0.4 米 GT 网格的体素中心，不是体素立方体与椭球是否相交；1σ 内采样点较少也会减少观察到的类别混合。','',
        '## 2. 类别混合主要发生在核心还是外围','',
        '以下四类互斥且穷尽，以全部预测 GS 为分母。“3σ 仍不跨类”包括范围内没有有效 GT 的情况。','',
        '| 集合 | 跨类定义 | 1σ 内已跨类 | 到 2σ 才跨类 | 到 3σ 才跨类 | 3σ 仍不跨类 | 3σ 跨类者中，1σ 已跨类 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for split,a in stats.items():
        t=np.asarray(a['transitions'])[1]
        for k,label in enumerate(('任意两类（含 empty）','仅前景类别','前景/empty 接口')):
            v=t[k];n=int(v.sum());mixed=int(v[4]+v[6]+v[7])
            lines.append(f'| {names[split]} | {label} | {pct(ratio(v[7],n))} | {pct(ratio(v[6],n))} | {pct(ratio(v[4],n))} | {pct(ratio(v[0],n))} | {pct(ratio(v[7],mixed))} |')
    lines+=['','这些是 GT 支持集合的变化，不是三种模型的性能变化。外围跨类也可能提供有用覆盖；'
        '比例不能单独证明类别干扰，更不能说明截掉外围就会提高精度。','',
        '## 3. 全部有效 GT 的附加口径','',
        '| 集合 | 范围 | 任意两类/全部 GS | 跨前景类别/全部 GS | 前景与 empty/全部 GS |',
        '|---|---|---:|---:|---:|']
    for split,a in stats.items():
        c=np.asarray(a['counts'])
        for r in range(3):
            v=c[r,0]
            lines.append(f'| {names[split]} | {r+1}σ | {pct(ratio(v[2],v[0]))} | {pct(ratio(v[3],v[0]))} | {pct(ratio(v[4],v[0]))} |')
    lines+=['','主结论使用第1、2节的 camera 口径；本节仅提供同一支持定义下的补充数据。','',
        '## 4. 校验与原始数据','',
        '- [统计定义](small_gaussian_diagnosis_design.md)',
        '- [三个范围的完整计数与区间分组](../work_dirs/small_gaussian_diagnosis_full/sigma_boundaries/summary.json)',
        '- [逐场景、逐范围 CSV](../work_dirs/small_gaussian_diagnosis_full/sigma_boundaries/scene_sigma_statistics.csv)',
        '- [query 负载组的边界分配](../work_dirs/small_gaussian_diagnosis_full/sigma_boundaries/query_load_sigma_statistics.csv)',
        '- [独立实现校验](../work_dirs/small_gaussian_diagnosis_full/sigma_boundaries/pilot.json)',
        '- [半径边界、零 opacity 与无效 GT 校验](../work_dirs/small_gaussian_diagnosis_full/sigma_boundaries/toy_boundary_audit.json)',
        '- [全量数据复核](../work_dirs/small_gaussian_diagnosis_full/sigma_boundaries/audit.json)',
        '- [small 主诊断](small_gaussian_diagnosis_results.md)','',
        '同一 GS 的 1σ 标签集合必须包含于 2σ，2σ 包含于 3σ；3σ 与原诊断逐场景／query 精确核对。'
        'CSV 比例为 0～1，正文转换为百分比；全局结果由原始分子、分母累加后计算。']
    REPORT.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial',action='store_true')
    args=p.parse_args()
    result=aggregate(args.partial);render(result)
    print({'status':result['status'],'completed':{s:(a['scenes'],a['frames']) for s,a in result['statistics'].items()}})
