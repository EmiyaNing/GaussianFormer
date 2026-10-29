"""Aggregate small census and compare with the completed tiny census."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnose_small_gaussians import RUN, Q, CHILDREN, GROUPS, ATTRS, FLAGS
from tools.stat_boundary_gaussians import read, write, sha
from tools.boundary_gaussian_stats import SCOPES, COHORTS, rates
from tools import report_query_budget as qr

REPORT = ROOT/'analysis/small_gaussian_diagnosis_results.md'
AGG = RUN/'aggregate'
FIGURES = ROOT/'analysis/figures/small_gaussian_diagnosis'
ZH = {
    'all':'全部 GS', 'center_visible_foreground':'中心位于可见前景',
    'center_visible_empty':'中心位于可见 empty', 'center_other':'中心未获得有效 camera 标签',
    'center_visible_empty_reaches_foreground_native':'中心可见 empty、原生范围仍接触前景',
    'center_visible_empty_pure_empty_native':'中心可见 empty、camera 原生支持纯空',
    'pure_empty_camera_native':'camera 原生支持纯空', 'pure_empty_all_native':'全部有效 GT 原生支持纯空',
    'pure_empty_camera_3sigma':'camera 3σ 支持纯空',
    'center_visible_empty_pure_empty_3sigma':'中心可见 empty、camera 3σ 支持纯空',
    'pure_empty_all_3sigma':'全部有效 GT 3σ 支持纯空',
    'pure_empty_camera_native_fp_owner':'camera 原生支持纯空且主导过 FP',
    'center_visible_empty_pure_empty_native_fp_owner':'中心可见 empty、camera 原生纯空且主导 FP',
    'pure_foreground_camera_3sigma':'camera 3σ 纯单一前景',
    'mixed_foreground_camera_3sigma':'camera 3σ 跨前景类别',
}


def div(a,b):
    return float(a/b) if b else None


def pct(x):
    return '—' if x is None else f'{100*x:.3f}%'


def number(x, digits=3):
    return '—' if x is None else f'{x:.{digits}f}'


def metrics(cm):
    cm = np.asarray(cm,dtype=np.int64)
    correct = np.diag(cm)[:17]
    unions = cm.sum(0)[:17]+cm.sum(1)[:17]-correct
    iou = np.divide(correct,unions,out=np.zeros(17,dtype=float),where=unions>0)
    fg,empty = int(cm[:17].sum()),int(cm[17].sum())
    fn,fp = int(cm[:17,17].sum()),int(cm[17,:17].sum())
    tp = int(cm[:17,:17].sum())
    return dict(miou=float(iou.mean()),iou=div(tp,tp+fn+fp),foreground=fg,empty=empty,
        fn=fn,fp=fp,fn_rate=div(fn,fg),fp_rate_empty=div(fp,empty),fp_rate_fg=div(fp,fg),
        fp_fraction_predictions=div(fp,tp+fp),semantic_error_rate=div(tp-int(correct.sum()),fg),
        class_iou=iou.tolist(),confusion=cm.tolist())


def configure_query_report():
    qr.RUN,qr.AGG,qr.Q,qr.CHILDREN = RUN,AGG,Q,CHILDREN


def extra_aggregate(partial=False):
    protocol = read(RUN/'protocol.json')
    expected = protocol['expected']
    summaries,scene_rows,gaussian_rows,error_rows,boundary_load_rows = {},[],[],[],[]
    identities=set()
    for split in ('train','val'):
        gc=None
        cm=np.zeros((18,18),np.int64)
        errors=np.zeros((4,9,5),np.int64)
        ng=np.zeros(len(GROUPS),np.int64)
        attrs=np.zeros((len(GROUPS),len(ATTRS)),float)
        flags=np.zeros((len(GROUPS),len(FLAGS)),np.int64)
        frames=scenes=0
        allocation=np.zeros((3,2,3),np.int64)
        allocation_gs=np.zeros(3,np.int64)
        for path in sorted((RUN/'scenes'/split).glob('*.json')):
            meta=read(path)
            with np.load(path.with_suffix('.npz')) as z:
                x={key:z[key] for key in ('group_counts','attribute_sums','flag_counts','confusion','boundary_errors','boundary_queries','metrics')}
            assert meta['checkpoint_sha256']==protocol['checkpoint_sha256']
            frames+=meta['frames'];scenes+=1
            for record in meta['records']:
                token=record['sample_token']
                assert token not in identities,token
                identities.add(token)
            for key,target in (('confusion',cm),('boundary_errors',errors),('group_counts',ng),('attribute_sums',attrs),('flag_counts',flags)):
                target+=x[key]
            if gc is None:
                gc=meta['gaussian_counts']
            else:
                for s in range(4):
                    for key,value in meta['gaussian_counts'][s].items():
                        gc[s][key]=(np.asarray(gc[s][key])+value).tolist() if isinstance(value,list) else gc[s][key]+value
            m=metrics(x['confusion'])
            n=int(x['group_counts'][0])
            assert n==76800*meta['frames']
            assert int(x['group_counts'][1:4].sum())==n
            assert np.all(x['flag_counts']<=x['group_counts'][:,None])
            for group_index,group in enumerate(GROUPS):
                count=int(x['group_counts'][group_index])
                row=dict(split=split,scene=meta['scene_name'],group=group,gaussians=count,all_gaussians=n,fraction=div(count,n))
                row.update({key:div(x['attribute_sums'][group_index,j],count) for j,key in enumerate(ATTRS)})
                row.update({key+'_fraction':div(x['flag_counts'][group_index,j],count) for j,key in enumerate(FLAGS)})
                gaussian_rows.append(row)
            for s,scope in enumerate(SCOPES):
                for c,cohort in enumerate(COHORTS):
                    error_rows.append(dict(split=split,scene=meta['scene_name'],scope=scope,cohort=cohort,**rates(x['boundary_errors'][s,c])))
            scene_rows.append(dict(split=split,scene=meta['scene_name'],frames=meta['frames'],miou=m['miou'],iou=m['iou'],
                fn_rate=m['fn_rate'],fp_rate_empty=m['fp_rate_empty'],
                center_visible_empty_fraction=div(x['group_counts'][2],n),
                center_visible_empty_score_opacity_gt09_fraction_all=div(x['flag_counts'][2,4],n),
                boundary_camera_3sigma_fraction=div(meta['gaussian_counts'][3]['boundary_any'],n),
                boundary_foreground_camera_3sigma_fraction=div(meta['gaussian_counts'][3]['boundary_foreground'],n)))
            load=x['metrics'][...,qr.FIELDS.index('gt_foreground')]
            order=np.argsort(load,axis=1,kind='stable')
            high=np.zeros_like(load,dtype=bool);low=high.copy()
            np.put_along_axis(high,order[:,-Q//5:],True,axis=1)
            np.put_along_axis(low,order[:,:Q//5],True,axis=1)
            for k,mask in enumerate((low,high,np.ones_like(high))):
                allocation[k]+=x['boundary_queries'][mask].sum(0,dtype=np.int64)
                allocation_gs[k]+=int(mask.sum())*CHILDREN
        if not scenes:
            continue
        m=metrics(cm)
        groups={}
        for i,group in enumerate(GROUPS):
            count=int(ng[i])
            groups[group]=dict(gaussians=count,fraction=div(count,int(ng[0])),
                means={key:div(attrs[i,j],count) for j,key in enumerate(ATTRS)},
                flags={key:dict(count=int(flags[i,j]),fraction_group=div(flags[i,j],count),fraction_all=div(flags[i,j],int(ng[0]))) for j,key in enumerate(FLAGS)},
                fp_owner_share=div(attrs[i,-1],m['fp']))
        boundary=[]
        for s in range(4):
            g=gc[s]
            boundary.append(dict(**g,**{key+'_fraction':div(g[key],g['predicted']) for key in ('boundary_any','boundary_foreground','boundary_fg_empty')},
                boundary_any_fraction_supported=div(g['boundary_any'],g['supported']),
                errors={cohort:rates(errors[s,j]) for j,cohort in enumerate(COHORTS)}))
        for i,name in enumerate(('all_query_low20','all_query_high20','all_queries')):
            for s,scope in enumerate(('native_camera','core3sigma_camera')):
                for j,kind in enumerate(('any','foreground','fg_empty')):
                    boundary_load_rows.append(dict(split=split,group=name,scope=scope,kind=kind,
                        gaussians=int(allocation_gs[i]),boundary_gaussians=int(allocation[i,s,j]),
                        boundary_fraction_group=div(allocation[i,s,j],allocation_gs[i]),
                        boundary_share_split=div(allocation[i,s,j],allocation[2,s,j])))
        summaries[split]=dict(frames=frames,scenes=scenes,metrics=m,groups=groups,boundary=boundary)
    complete=all(split in summaries and {k:summaries[split][k] for k in ('frames','scenes')}==expected[split] for split in ('train','val'))
    if not partial:
        assert complete
        assert len(identities)==34149
    qr.csv_write(AGG/'scene_model_and_geometry.csv',scene_rows)
    qr.csv_write(AGG/'scene_gaussian_groups.csv',gaussian_rows)
    qr.csv_write(AGG/'scene_boundary_errors.csv',error_rows)
    qr.csv_write(AGG/'boundary_by_load.csv',boundary_load_rows)
    result=dict(status='complete' if complete else 'partial',statistics=summaries,
        checkpoint_sha256=protocol['checkpoint_sha256'],unique_sample_tokens=len(identities),
        measurement_source_sha256=protocol['source_sha256'],
        reporting_source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    write(AGG/'diagnosis_summary.json',result)
    return result


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def plots(result, query_summary, query_groups):
    expected=read(RUN/'protocol.json')['expected']
    if 'val' not in result['statistics'] or result['statistics']['val']['frames']!=expected['val']['frames']:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    FIGURES.mkdir(parents=True,exist_ok=True)
    tiny_query=read(ROOT/'work_dirs/query_budget_full/aggregate/summary.json')['summary']['val']
    tiny_groups=read_csv(ROOT/'work_dirs/query_budget_full/aggregate/query_groups.csv')
    tiny_high=next(r for r in tiny_groups if r['split']=='val' and r['group']=='all_query_top20')
    small_high=next(r for r in query_groups if r['split']=='val' and r['group']=='all_query_top20')
    tiny_boundary=read(ROOT/'work_dirs/boundary_gaussian_full/summary.json')['statistics']['val']['gaussian_counts'][3]
    small_boundary=result['statistics']['val']['boundary'][3]
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    colors=['#4477aa','#cc6677']
    for k,(name,row) in enumerate((('Tiny',tiny_high),('Small',small_high))):
        values=[20,100*float(row['foreground_share']),100*float(row['hole_share'])]
        axes[0].bar(np.arange(3)+(k-.5)*.36,values,width=.36,label=name,color=colors[k])
    axes[0].set(xticks=np.arange(3),xticklabels=['GS budget','GT foreground','Core holes'],ylabel='Share of full validation (%)',title='Highest-load 20% of all queries',ylim=(0,100))
    axes[0].legend()
    bottom=np.zeros(2)
    rows=[tiny_query,query_summary['val']]
    for key,label,color in [('own_core_rate','Own query','#66aa88'),('neighbor_only_rate','Neighbors only','#ddbb66'),('hole_rate','No core','#cc6677')]:
        value=np.array([r[key]*100 for r in rows])
        axes[1].bar([0,1],value,bottom=bottom,color=color,label=label,width=.6)
        bottom+=value
    axes[1].set(xticks=[0,1],xticklabels=['Tiny','Small'],ylabel='All GT foreground (%)',title='Where core coverage comes from',ylim=(0,100))
    axes[1].legend(fontsize=8,loc='lower left')
    for k,(name,row) in enumerate((('Tiny',tiny_boundary),('Small',small_boundary))):
        values=[row[key]*100 for key in ('boundary_any_fraction','boundary_foreground_fraction','boundary_fg_empty_fraction')]
        axes[2].bar(np.arange(3)+(k-.5)*.36,values,width=.36,label=name,color=colors[k])
    axes[2].set(xticks=np.arange(3),xticklabels=['Any classes','Multiple FG','FG + empty'],ylabel='All predicted GS (%)',title='Boundary GS: camera 3-sigma',ylim=(0,100))
    axes[2].legend()
    fig.suptitle('Frozen tiny vs. small | 6,019 validation frames | same 76,800 GS')
    fig.savefig(FIGURES/'query_and_boundary.png',dpi=180)
    plt.close(fig)


def render(result,query_summary,query_groups):
    tiny_query=read(ROOT/'work_dirs/query_budget_full/aggregate/summary.json')['summary']
    tiny_groups=read_csv(ROOT/'work_dirs/query_budget_full/aggregate/query_groups.csv')
    tiny_boundary=read(ROOT/'work_dirs/boundary_gaussian_full/summary.json')['statistics']
    tiny_support=read(ROOT/'work_dirs/gaussian_support_regions_full/summary.json')['statistics']
    expected=read(RUN/'protocol.json')['expected']
    stats={split:a for split,a in result['statistics'].items()
           if {key:a[key] for key in ('frames','scenes')}==expected[split]}
    complete=result['status']=='complete'
    lines=['# small 权重诊断：分配压力、空区前景得分和边界 GS','',
        '**状态：完整统计已完成。**' if complete else '**状态：全量运行尚未结束。以下只展示已经完整完成的集合，不将部分训练场景与完整 tiny 训练集混比。**','',
        '本轮冻结 small 权重，默认在 camera mask 内评价。small 为 **1,200 query×64 GS**，tiny 为 **600×128**；两者最终数量均为 76,800。'
        '下表逐项比较，两份权重独立训练，不能将差值全部归因于 query 数量。','']
    if 'val' in stats:
        val=stats['val'];qs=query_summary['val']
        high=next(r for r in query_groups if r['split']=='val' and r['group']=='all_query_top20')
        empty=val['groups']['center_visible_empty'];score=empty['flags']['score_and_opacity_gt_09']
        lines+=['**先回答三个问题（完整验证集）：**','',
            f'- **query 分配压力：**最高负载 20% query 承担 {pct(float(high["foreground_share"]))} 前景、集中 {pct(float(high["hole_share"]))} 核心缺口，而 GS 份额仍为 20%。'
            f'全部前景零核心率为 {pct(qs["hole_rate"])}，tiny 为 {pct(tiny_query["val"]["hole_rate"])}。',
            f'- **空区高分 GS：**中心位于有效 camera empty 的 GS 占 {pct(empty["fraction"])}；'
            f'其中 {pct(score["fraction_group"])} 同时满足前景 sigmoid 分数>0.9、opacity>0.9，合计 {score["count"]:,} 个高斯实例，平均每帧 {score["count"]/val["frames"]:.1f} 个。',
            f'- **边界分配：**camera 3σ 下，{pct(val["boundary"][3]["boundary_any_fraction"])} 的 GS 跨任意 GT 类别，'
            f'{pct(val["boundary"][3]["boundary_foreground_fraction"])} 跨至少两种前景类别。','']
        lines+=['![tiny 与 small 的预算压力、核心覆盖和边界分配](figures/small_gaussian_diagnosis/query_and_boundary.png)','']
    lines+=['## 1. 数据与总体预测','',
        '| 集合 | 场景／帧 | small mIoU | 二值 IoU | FN/GT 前景 | FP/GT empty | FP/GT 前景 | FP/预测前景 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split,a in stats.items():
        m=a['metrics']
        lines.append(f'| {split} | {a["scenes"]}／{a["frames"]:,} | {pct(m["miou"])} | {pct(m["iou"])} | {pct(m["fn_rate"])} | {pct(m["fp_rate_empty"])} | {pct(m["fp_rate_fg"])} | {pct(m["fp_fraction_predictions"])} |')
    lines+=['','## 2. query 分配压力','',
        '负载由第五级最近中心划分，最高 20% 按每帧全部 query 取 240 个。下表分母为各模型同一集合的全部前景／全部核心缺口；预算始终为 20%。','',
        '| 集合／模型 | 最高负载 20% 的前景份额 | 其核心缺口份额 | 全体前景无核心率 | 无核心但类别仍正确 |',
        '|---|---:|---:|---:|---:|']
    for split in stats:
        for name,s in (('tiny',tiny_query[split]),('small',query_summary[split])):
            g=next(r for r in (tiny_groups if name=='tiny' else query_groups) if r['split']==split and r['group']=='all_query_top20')
            lines.append(f'| {split}／{name} | {pct(float(g["foreground_share"]))} | {pct(float(g["hole_share"]))} | {pct(s["hole_rate"])} | {pct(s["no_core_correct_rate"])} |')
    lines+=['','### 同样类别、距离和粗层定位误差下，高负载组还更缺覆盖吗？','',
        '| 集合／模型 | 对齐后低负载缺口率 | 对齐后高负载缺口率 | 高−低，百分点 | 95% 场景区间，百分点 |',
        '|---|---:|---:|---:|---:|']
    for split in stats:
        for name,s in (('tiny',tiny_query[split]),('small',query_summary[split])):
            st=s['standardized_low_high'];gap=st['high']['hole_rate']-st['low']['hole_rate'];ci=st['hole_gap_interval']
            lines.append(f'| {split}／{name} | {pct(st["low"]["hole_rate"])} | {pct(st["high"]["hole_rate"])} | {gap*100:+.3f} | [{ci[0]*100:+.3f}, {ci[1]*100:+.3f}] |')
    lines+=['','对齐按各模型自己的粗层分区进行，不能将两张表的差当作单独改变 query 数量的因果效果。负载不是越大必然越难，完整分组见 CSV。','',
        '| 集合／模型 | 活跃负载 P10/P50/P90/P99 | 同分位负载/单 query GS | 自身核心覆盖 | 仅邻居补足 |',
        '|---|---:|---:|---:|---:|']
    for split in stats:
        for name,s,children in (('tiny',tiny_query[split],128),('small',query_summary[split],64)):
            values=list(s['load_quantiles_active'].values())
            raw='/'.join(f'{x:g}' for x in values);normalized='/'.join(f'{x/children:.2f}' for x in values)
            lines.append(f'| {split}／{name} | {raw} | {normalized} | {pct(s["own_core_rate"])} | {pct(s["neighbor_only_rate"])} |')
    lines+=['','## 3. 中心在 empty，前景得分是否仍然很高？','',
        '这里的中心 empty 必须位于 camera mask 内且 GT=empty；mask 外中心不计入。'
        '“高分”使用最大原始前景 logit 的 sigmoid >0.9，另要求 opacity>0.9。它不是校准概率。','',
        '| 集合 | 可见 empty 中心 GS 占全部 | 其中高分且高 opacity | 同时满足两者占全部 GS | 平均最高 softplus 值 | 平均 opacity | 该中心组主导的 FP 份额 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split,a in stats.items():
        g=a['groups']['center_visible_empty'];h=g['flags']['score_and_opacity_gt_09'];m=g['means']
        lines.append(f'| {split} | {pct(g["fraction"])} | {pct(h["fraction_group"])} | {pct(h["fraction_all"])} | {number(m["semantic_max"])} | {number(m["opacity"])} | {pct(g["fp_owner_share"])} |')
    lines+=['','当前 GS 只预测 17 个前景类别，empty 由全局高斯与 bias 提供；因此 argmax 为前景本身不能证明异常。上表检查的是实际前景分数与 opacity。'
        'sigmoid>0.9 对应原 logit>log(9)，等价 softplus>log(10)≈2.303。最后一列统计整个可见 empty 中心组的 FP 归属，'
        '不是仅高分且高 opacity 子集的 FP 份额，也不等于删除后的 FP 降幅。','',
        '### 更严格地要求支持内不接触前景','',
        '| 集合／分组 | GS 数量 | 占全部 GS | 高分且高 opacity/组内 | 平均最高 softplus | 平均 opacity | 主导 FP 份额 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    selected=('center_visible_empty_reaches_foreground_native','center_visible_empty_pure_empty_native',
        'pure_empty_camera_3sigma','center_visible_empty_pure_empty_3sigma','pure_empty_all_native')
    for split,a in stats.items():
        for group in selected:
            g=a['groups'][group];m=g['means']
            lines.append(f'| {split}／{ZH[group]} | {g["gaussians"]:,} | {pct(g["fraction"])} | {pct(g["flags"]["score_and_opacity_gt_09"]["fraction_group"])} | {number(m["semantic_max"])} | {number(m["opacity"])} | {pct(g["fp_owner_share"])} |')
    lines+=['','表中 FP 份额统一按原生支持下的主要贡献者归属；包括 3σ 分组也只是用 3σ 来判断高斯的 GT 支持类型。'
        '各行有重叠，不能将它们相加。所有有效 GT 支持纯空不代表未知空间也已被确认是空。'
        '“全部有效 GT 原生支持纯空”一行未要求中心可见，不能把该行的高分比例直接用于判断可见监督下的现象；重点看显式要求中心可见的行。','',
        '### tiny 与 small 的 camera 3σ 纯空高斯','',
        '| 集合／模型 | 占全部 GS | 平均最高 softplus 值 | 平均语义值总和 | 平均 opacity |',
        '|---|---:|---:|---:|---:|']
    for split,a in stats.items():
        t=tiny_support[split]['core3sigma_camera']['groups']['pure_empty'];s=a['groups']['pure_empty_camera_3sigma'];m=s['means']
        lines.append(f'| {split}／tiny | {pct(t["gaussian_fraction"])} | {number(t["gaussian_mean_max_semantic"])} | {number(t["gaussian_mean_semantic_sum"])} | {number(t["gaussian_mean_opacity"])} |')
        lines.append(f'| {split}／small | {pct(s["fraction"])} | {number(m["semantic_max"])} | {number(m["semantic_sum"])} | {number(m["opacity"])} |')
    lines+=['','## 4. 边界 GS 分配','',
        '追加的 1σ、2σ、3σ 并列统计见[不同范围的跨类别 GS 报告](small_sigma_boundary_results.md)；以下保留主诊断的 3σ／原生口径。','',
        '主口径为 camera mask 内 3σ 支持；分母为全部预测 GS。前景/empty 接口与跨前景类别边界可以重叠。','',
        '| 集合／模型 | 任意两类边界 | 跨前景类别边界 | 前景/empty 接口 | 有支持 GS 中的任意边界 |',
        '|---|---:|---:|---:|---:|']
    for split,a in stats.items():
        for name,g in (('tiny',tiny_boundary[split]['gaussian_counts'][3]),('small',a['boundary'][3])):
            lines.append(f'| {split}／{name} | {pct(g["boundary_any_fraction"])} | {pct(g["boundary_foreground_fraction"])} | {pct(g["boundary_fg_empty_fraction"])} | {pct(g["boundary_any_fraction_supported"])} |')
    lines+=['','### 原生支持敏感性','',
        '| 集合／模型 | 任意两类边界 | 跨前景类别边界 | 前景/empty 接口 |',
        '|---|---:|---:|---:|']
    for split,a in stats.items():
        for name,g in (('tiny',tiny_boundary[split]['gaussian_counts'][1]),('small',a['boundary'][1])):
            lines.append(f'| {split}／{name} | {pct(g["boundary_any_fraction"])} | {pct(g["boundary_foreground_fraction"])} | {pct(g["boundary_fg_empty_fraction"])} |')
    lines+=['','### 边界 GS 分给了哪些 query？','',
        '每帧全部 query 的低／高负载 20% 各获得 20% 的 GS。下面报告它们承接了全体边界 GS 的多少，使用 camera 3σ 口径。','',
        '| 集合／query 组 | GS 预算份额 | 任意边界 GS 的份额 | 跨前景类别边界 GS 的份额 | 组内 GS 为任意边界的比例 |',
        '|---|---:|---:|---:|---:|']
    boundary_load=read_csv(AGG/'boundary_by_load.csv')
    for split in stats:
        for group,label in (('all_query_low20','低负载 20%'),('all_query_high20','高负载 20%')):
            rows={r['kind']:r for r in boundary_load if r['split']==split and r['group']==group and r['scope']=='core3sigma_camera'}
            lines.append(f'| {split}／{label} | 20.000% | {pct(float(rows["any"]["boundary_share_split"]))} | {pct(float(rows["foreground"]["boundary_share_split"]))} | {pct(float(rows["any"]["boundary_fraction_group"]))} |')
    lines+=['','### small 边界区域与非边界区域的预测','',
        '主要贡献者为边界、非边界、未归属三类互斥且穷尽；“边界支持并集”另列，可能包含其他类型 GS 的贡献。'
        '预测均使用未修改的完整原生渲染结果。','',
        '| 集合／区域 | GT 前景数 | GT empty 数 | 错分类/GT 前景 | FN/GT 前景 | FP/GT empty |',
        '|---|---:|---:|---:|---:|---:|']
    for split,a in stats.items():
        for cohort,label in (('dominant_any','边界主要贡献者'),('dominant_nonboundary','非边界主要贡献者'),('unattributed','3σ 未归属'),('union_any','边界支持并集'),('union_foreground','跨前景类别支持并集')):
            e=a['boundary'][3]['errors'][cohort]
            lines.append(f'| {split}／{label} | {e["gt_foreground"]:,} | {e["gt_empty"]:,} | {pct(e["wrong_class_rate_gt_foreground"])} | {pct(e["miss_rate_gt_foreground"])} | {pct(e["false_positive_rate_gt_empty"])} |')
    lines+=['','## 5. 怎样理解这些诊断','',
        '分配压力要同时看需求份额、核心空洞和对齐后的高低负载差，不能只凭“small 每个 query 只有 64 个”判断更拥挤。'
        '高分空区 GS 也不能直接叫无效：中心在空区但支持仍覆盖前景时，它可以参与正确预测；支持纯空且主导 FP 是更严格的错误相关子集。','',
        '边界比例更大或更小都不自动意味着性能更好。重点是有限 GS 是否完整覆盖前景，以及边界与非边界区域中 FN、FP 和错类如何变化。'
        '本轮为冻结统计，不证明固定配额或某一类 GS 是退化的主导成因。','',
        '## 6. 复核与逐场景数据','',
        '- [实验设计](small_gaussian_diagnosis_design.md)',
        '- [完整汇总 JSON](../work_dirs/small_gaussian_diagnosis_full/aggregate/diagnosis_summary.json)',
        '- [query 汇总与对齐结果](../work_dirs/small_gaussian_diagnosis_full/aggregate/summary.json)',
        '- [逐场景模型与几何指标](../work_dirs/small_gaussian_diagnosis_full/aggregate/scene_model_and_geometry.csv)',
        '- [逐场景高斯分组与阈值统计](../work_dirs/small_gaussian_diagnosis_full/aggregate/scene_gaussian_groups.csv)',
        '- [逐场景边界区域错误](../work_dirs/small_gaussian_diagnosis_full/aggregate/scene_boundary_errors.csv)',
        '- [query 负载分组](../work_dirs/small_gaussian_diagnosis_full/aggregate/query_groups.csv)',
        '- [边界 GS 按负载分配](../work_dirs/small_gaussian_diagnosis_full/aggregate/boundary_by_load.csv)',
        '- [测量协议](../work_dirs/small_gaussian_diagnosis_full/protocol.json)',
        '- [实现校验](../work_dirs/small_gaussian_diagnosis_full/pilot.json)',
        '', 'CSV 比例为 0～1；本文转换成百分比。所有全局比例先累加分子、分母再计算。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main(partial=False):
    configure_query_report()
    query_summary,data,classes,buckets,deciles,groups,complete=qr.aggregate(partial)
    del data,classes
    result=extra_aggregate(partial)
    plots(result,query_summary,groups)
    render(result,query_summary,groups)
    print(json.dumps(dict(status=result['status'],completed={s:{k:a[k] for k in ('frames','scenes')} for s,a in result['statistics'].items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    main(args.partial)
