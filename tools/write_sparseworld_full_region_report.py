#!/usr/bin/env python3
"""Human-readable full-census research reports and scientific figures."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.sparseworld_full_regions_common import ROOT,REPORT
from tools.sparseworld_region_utils import CLASS_NAMES,STATE_NAMES


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def val(d,index,digits=2,scale=1):
    v=np.asarray(d['value'],dtype=float)[index]*scale;c=np.asarray(d['ci95'],dtype=float)[index]*scale
    if not np.isfinite(v):return '—'
    return f'{v:.{digits}f} [{c[0]:.{digits}f}, {c[1]:.{digits}f}]'


def export_numeric_tables(s,m):
    """Keep readable Markdown compact while exporting all numerical estimates."""
    rows=[]
    for split,item in s['splits'].items():
        for family,d in item['attributes'].items():
            for metric in ('forecast_accuracy','perception_accuracy','advantage'):
                v=np.asarray(d[metric]['value']);c=np.asarray(d[metric]['ci95'])
                for h,level in np.ndindex(v.shape):
                    rows.append(dict(split=split,family=family,group=level,horizon_seconds=h+1,metric=metric,
                        value=v[h,level],ci_low=c[h,level,0],ci_high=c[h,level,1],
                        gt_voxels=np.asarray(d['total'])[h,level],scenes=np.asarray(d['scenes'])[h,level],
                        unit='percentage_points' if metric=='advantage' else 'percent'))
        for model,d in item['models'].items():
            for metric,estimate in d.items():
                v=np.asarray(estimate['value']);c=np.asarray(estimate['ci95'])
                for index in np.ndindex(v.shape):
                    h,mask,*category=index
                    rows.append(dict(split=split,family='model_'+model,group=f'mask={mask};class={category[0] if category else "all"}',
                        horizon_seconds=h+1,metric=metric,value=v[index],ci_low=c[index+(0,)],ci_high=c[index+(1,)],unit='percent'))
    fields=['split','family','group','horizon_seconds','metric','value','ci_low','ci_high','gt_voxels','scenes','unit']
    with (REPORT/'region_statistics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    if not m:return
    rows=[]
    def visit(d,path):
        if isinstance(d,dict) and 'value' in d and 'ci95' in d:
            v=np.asarray(d['value']);c=np.asarray(d['ci95'])
            if v.shape==(3,):
                for h in range(3):
                    rows.append(dict(path='/'.join(path),horizon_seconds=h+1,value=v[h],ci_low=c[h,0],ci_high=c[h,1],
                        support=np.asarray(d['support'])[h] if 'support' in d else '',scenes=np.asarray(d['scenes'])[h] if 'scenes' in d else ''))
        elif isinstance(d,dict):
            for k,value in d.items():visit(value,path+[k])
    visit(m['splits'],[])
    with (REPORT/'motion_statistics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['path','horizon_seconds','value','ci_low','ci_high','support','scenes']);w.writeheader();w.writerows(rows)


def main():
    s=json.loads((ROOT/'summary.json').read_text());assert s['status']=='complete'
    m=json.loads((ROOT/'motion_summary.json').read_text()) if s['fast_gate']['passed'] else None
    REPORT.mkdir(exist_ok=True);(REPORT/'figures').mkdir(exist_ok=True)
    coverage=table(['集合','场景','全部关键帧','1 秒','2 秒','3 秒'],[[name,s['protocol']['splits'][split]['scenes'],s['protocol']['splits'][split]['raw_keyframes'],*s['splits'][split]['samples']] for split,name in [('train','训练集'),('val','验证集')]])
    fast=table(['快速区域：语义准确率（%）','1 秒','2 秒','3 秒'],[[label+'：'+model]+[val(s['splits'][split]['attributes']['speed'][key],(j,2)) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for model,key in [('预测器 F','forecast_accuracy'),('未来感知器 P','perception_accuracy'),('P−F（百分点）','advantage')]])
    broad=table(['全域 mIoU（%）','1 秒','2 秒','3 秒'],[[label+'：'+model]+[val(s['splits'][split]['models'][model]['mIoU'],(j,0),3) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for model in ('F','P')])
    four=table(['真实占据：四种状态（%）','1 秒','2 秒','3 秒'],[[label+'：'+state]+[val(s['splits'][split]['states']['proportions'],(j,0,1,0,k)) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for k,state in enumerate(['双方都错','仅感知器正确','仅预测器正确','双方都对'])])
    attr_rows=[]
    levels={'distance':['0–10 m','10–20 m','20–30 m','≥30 m'],'speed':['<0.5 m/s','0.5–2 m/s','≥2 m/s','速度未知'],
        'boundary':['内部','0.4 m 边界'],'camera':['camera 不可见','camera 可见'],'static_frustum':['两窗口均未覆盖','仅未来窗口','仅历史窗口','两窗口均覆盖','未知']}
    for attr,labels in levels.items():
        for i,label in enumerate(labels):
            attr_rows.append([label]+[val(s['splits']['val']['attributes'][attr]['advantage'],(j,i)) for j in range(3)])
    attr_table=table(['验证集属性：P−F（百分点）','1 秒','2 秒','3 秒'],attr_rows)
    static=np.asarray(s['splits']['val']['attributes']['static_frustum']['advantage']['value'])[:,1]
    static_ci=np.asarray(s['splits']['val']['attributes']['static_frustum']['advantage']['ci95'])[:,1]
    status=['预测器优势且区间低于零' if c[1]<0 else ('感知器优势且区间高于零' if c[0]>0 else '区间跨零，方向未确认') for c in static_ci]
    gate='全量训练和验证在三个时距均支持感知器的快速区域优势，验证集的三个配对区间均高于零。' if s['fast_gate']['passed'] else '预先规定的全量快速区域确认条件未全部满足；机制结论不按先前假设强行成立。'
    common=table(['共同完整 3 秒源帧：快速 P−F（百分点）','源帧数','1 秒','2 秒','3 秒'],[[label,s['splits'][split]['common_six_future_sources']]+[val(s['splits'][split]['common_attributes']['speed']['advantage'],(j,2)) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')]])
    spatial=s['splits']['val']['spatial_groups']
    spatial_table=table(['验证集空间分组：P−F（百分点）','1 秒','2 秒','3 秒'],[[name]+[val(spatial['statistics']['advantage'],(j,k)) for j in range(3)] for k,name in enumerate(spatial['names'])])
    local=[]
    for j in range(3):
        c=np.asarray(s['splits']['val']['spatial_sensitivity_counts'])[j,1]
        local.append([f'{j+1} 秒',f'{int(c[0]):,}']+[f'{100*c[k]/c[0]:.2f}' for k in (1,2,3,4)])
    local_table=table(['验证集 2m 局部块','有效帧×块','F 更准（%）','P 更准（%）','同分（%）','双方都错超过半数（%）'],local)
    text=f'''# 全量实验结果｜预测器与未来感知器的区域差异

[实验设计](design.md) · [快速目标机制报告](motion_results.md) · [完整统计](statistics.md) · [代码与复现](reproduction.md)

**已完成 {s['sources']:,} 个源帧、{s['paired_targets']:,} 个未来时距配对，覆盖全部 700 个训练场景和 150 个验证场景。**使用同架构同权重的原始 SparseWorld；本轮没有训练新模型。

{gate}

## 1. 全量覆盖与边界

{coverage}

各时距列是具备对应未来 GT 的有效源帧数，非随机子集。每个源帧使用五帧历史；因此每场景最初四帧、及末尾缺少对应未来的帧无法评测。详细原因与数值协议见[设计](design.md)。

## 2. 快速区域优势是否复现？

{fast}

![全量快速区域比较](figures/fast_comparison.png)

表中区间为 2000 次场景配对重采样的 95% 区间，差值单位为百分点；不是将数十亿体素视为独立样本。训练集是原模型训练分布内的统计，验证场景此前也参与过小样本研究，不能将扩大帧覆盖标为全新的独立测试集。

{common}

共同源帧对照用于检查时距比较是否受场景末尾人群变化影响。它保留所有具备完整 3 秒未来的源帧。

## 3. 优势区域、共同错误与整体性能

{four}

![真实占据区域的四种状态](figures/four_states.png)

{broad}

![全量验证集空间优势](figures/spatial_advantage.png)

空间图使用对应未来原生 ego 网格；颜色值为 **P准确率−F准确率**。正值表示感知器更准，负值表示预测器更准；绝对值越大，差距越大。每格以实际 GT 占据体素数为分母，不用自由空间数量掩盖物体错误。

{spatial_table}

近车分组以 2m 局部块中心半径小于 10m 定义；距离属性表使用精确体素半径，两者边缘分母略有区别。全象限／近车分组避免仅凭配色判断某个象限是否总体由 F 占优。

{local_table}

每个源帧独立划分 2m×2m 局部块，至少 20 个 GT 占据体素才进入本表。F／P／同分三项互斥；最后一项与前三项可以重叠，用于量化共同失效。局部存在 F 优势不等于该象限的全部体素或全部目标均由 F 更准。

## 4. 原来的解释是否仍成立？

{attr_table}

“仅未来窗口视锥”的静态结构，在全量验证的 1／2／3 秒分别为：{'；'.join(status)}。数值与区间决定本轮的表述，不沿用旧小样本的方向作为既定事实。

![属性与区域优势](figures/attribute_advantage.png)

GT 类别与距离共同支持标准化、训练场景拟合与验证场景评估的多变量解释，见[统计附表](statistics.md)。视锥覆盖不等于无遮挡可见；这些属性是关联解释，不能单凭它们作因果归因。

## 5. 下一步：快速目标的位置、朝向与形状

{gate}对应的机制证据见[快速目标误差分解](motion_results.md)：它将全部快速错误构成、可靠实例匹配下的几何纠正、以及历史 query 的原始几何追踪分开报告，并保留各自的分母。

原有输入干预、点预算消融和 2048／512 蒸馏实验仍属于其原样本范围；本轮完整覆盖的是原始 F／P 区域研究与快速目标机制统计。这些历史实验的数值不冒充全量结果。

原始数据与验收：[`out/sparseworld_region_comparison/20261010_full_nuscenes`](../../../../out/sparseworld_region_comparison/20261010_full_nuscenes/)。
'''
    (REPORT/'results.md').write_text(text)
    appendix='# 全量统计附表\n\n[主报告](results.md) · [机制报告](motion_results.md) · [区域数值 CSV](region_statistics.csv) · [机制数值 CSV](motion_statistics.csv)\n\n单位为 %；优势为 P−F 百分点。CSV 中属性分组编号依照本页各表的行顺序；mask 0／1／2 分别表示无可见性 mask、camera、lidar。\n\n'
    for split,label in [('train','训练集'),('val','验证集')]:
        appendix+=f'## {label}：全域指标\n\n'
        for metric in ('mIoU','OccIoU'):
            for mask,masklabel in enumerate(['原版无可见性 mask','camera 子集','lidar 子集']):
                appendix+=table([metric+'：'+masklabel,'1 秒','2 秒','3 秒'],[[model]+[val(s['splits'][split]['models'][model][metric],(j,mask),3) for j in range(3)] for model in ('F','P')])+'\n'
        appendix+=f'## {label}：逐类别 P−F 优势\n\n'+table(['类别','1 秒','2 秒','3 秒'],[[name]+[val(s['splits'][split]['attributes']['class']['advantage'],(j,k)) for j in range(3)] for k,name in enumerate(CLASS_NAMES[:17])])+'\n'
        speed=s['splits'][split]['attributes']['speed']
        appendix+=f'## {label}：速度分组分母\n\n'+table(['分组','1 秒 GT 体素／场景','2 秒','3 秒'],[[name]+[f"{int(speed['total'][j][k]):,}／{int(speed['scenes'][j][k])}" for j in range(3)] for k,name in enumerate(levels['speed'])])+'\n'
        for attr,labels in levels.items():
            appendix+=f'## {label}：{attr}\n\n'+table(['分组','1 秒 P−F','2 秒','3 秒'],[[name]+[val(s['splits'][split]['attributes'][attr]['advantage'],(j,k)) for j in range(3)] for k,name in enumerate(labels)])+'\n'
        appendix+=f'## {label}：类别 × 距离共同支持标准化\n\n'+table(['属性差异','1 秒','2 秒','3 秒'],[[name]+[val(d,j) for j in range(3)] for name,d in s['splits'][split]['standardized'].items()])+'\n'
    ep=ROOT/'explanation_summary.json'
    if ep.exists():
        e=json.loads(ep.read_text());assert e['status']=='complete'
        appendix+='## 多变量解释：700 个训练场景拟合，150 个验证场景评估\n\n'
        appendix+=table(['模型','验证 log loss','验证多分类 Brier'],[['类别＋时距基线',f"{e['category_time_baseline_holdout']['log_loss']:.5f}",f"{e['category_time_baseline_holdout']['brier']:.5f}"],['完整属性模型',f"{e['holdout']['log_loss']:.5f}",f"{e['holdout']['brier']:.5f}"]])+'\n'
        names={'camera1_vs0':'可见相对不可见','boundary_vs_interior':'边界相对内部','fast_vs_stopped':'快速相对停止／低速','future_only_vs_both':'仅未来窗口相对双窗口'}
        appendix+=table(['验证集多变量调整后的优势差（百分点）','1 秒','2 秒','3 秒'],[[names[k]]+[f'{v:.2f}' if v is not None else '—' for v in d['advantage_difference_pp']] for k,d in e['adjusted_effects_holdout'].items()])+'\n'
        appendix+='正则强度只在训练场景的三折场景交叉验证中选择；系数冻结后才计算验证成绩。上述调整效应只有点估计，没有重新拟合的 bootstrap 区间，属于条件关联，不能解释为因果效应。\n\n'
    (REPORT/'statistics.md').write_text(appendix)
    if m:write_motion(m)
    export_numeric_tables(s,m)
    figures(s,m)
    (REPORT/'README.md').write_text('# 全量 nuScenes｜区域优势与快速目标误差机制\n\n**全量执行已完成；主报告与机制报告均按实际覆盖分母解读。**\n\n'+coverage+'\n'+gate+'\n\n| 阅读目的 | 入口 |\n|---|---|\n| 全量区域优势与原观察的复核 | [主报告](results.md) |\n| 位姿、尺寸、形状残差与原始 query 追踪 | [机制报告](motion_results.md) |\n| 完整数值 | [统计附表](statistics.md) |\n| 实验控制 | [实验设计](design.md) |\n| 运行代码和产物 | [复现说明](reproduction.md) |\n| 研究总览与历史实验 | [返回总览](../README.md) |\n')


def write_motion(m):
    v=m['splits']['val'];share=np.asarray(v['query_lineage']['pose_share']['value']);shareci=np.asarray(v['query_lineage']['pose_share']['ci95'])
    if shareci[2,0]>50:
        severity=f'在历史绑定子集的 3 秒原始几何差异中，已找到的刚体纠正解释 {share[2]:.2f}% [{shareci[2,0]:.2f}, {shareci[2,1]:.2f}]。区间高于 50%，因此该子集支持位姿偏差比剩余几何差异更严重；仍不能将这个排序直接推广到未绑定目标。'
    elif shareci[2,1]<50:
        severity=f'历史绑定子集的 3 秒刚体可恢复份额为 {share[2]:.2f}% [{shareci[2,0]:.2f}, {shareci[2,1]:.2f}]，余下差异更大。但刚体搜索可能留下未恢复位姿误差，因此不能仅凭这个结果认定所有残差都是形状错误。'
    else:
        severity=f'历史绑定子集的 3 秒刚体可恢复份额为 {share[2]:.2f}% [{shareci[2,0]:.2f}, {shareci[2,1]:.2f}]；区间跨 50%，本轮未确认位姿与剩余几何差异的大小排序。'
    rigid=v['2.0']['all_fast']['gap_changes']['rigid']['gap_reduction']
    scale=v['2.0']['all_fast']['gap_changes']['size']['gap_reduction']
    fg=v['2.0']['all_fast']['gap_changes']['rigid']['forecast_gain'];pg=v['2.0']['all_fast']['gap_changes']['rigid']['perception_gain']
    restoration=f'在全部快速 GT 的固定分母上，3 秒位姿纠正使 F 标签正确率变化 {val(fg,2)} 个百分点，P 的对应变化为 {val(pg,2)} 个百分点；双方纠正后的 P−F 差距缩小 {val(rigid,2)} 个百分点，只改尺寸的对应缩小量为 {val(scale,2)} 个百分点。差距缩小量为正，才表示该因素更多解释了 F 的劣势。'
    harmful=v['query_lineage']['shape_comparable']['metrics']['template_advantage_over_aligned_actual']
    h=np.asarray(harmful['value'])[2];hc=np.asarray(harmful['ci95'])[2]
    shape_evidence=(f'在按历史模板对齐实际未来表示的对照中，保持当前预测形状的 GT IoU 高 {h*100:.2f} [{hc[0]*100:.2f}, {hc[1]*100:.2f}] 个百分点。这为形状保持提供候选证据，但有限位姿搜索和模板初始误差仍可能影响该差值，不能将其唯一归因于有害形变。' if hc[0]>0 else
        f'按历史模板对齐后的 GT IoU 对照差为 {h*100:.2f} [{hc[0]*100:.2f}, {hc[1]*100:.2f}] 个百分点；未确认该子集的平均 GT IoU 存在恶化。')
    rows=[]
    for split,label in [('train','训练集'),('val','验证集')]:
        d=m['splits'][split]['2.0']['all_fast']['metrics']
        for model in ('F','P'):
            for name,key in [('正确','native_correct'),('为空闲：漏占据','native_free'),('非空但错类别','native_wrong_nonfree')]:rows.append([label+' '+model+'：'+name]+[val(d[model+'_'+key],j) for j in range(3)])
    error=table(['全部快速 GT：原始结果（%）','1 秒','2 秒','3 秒'],rows)
    coverage=table(['可靠匹配 GT 体素覆盖（%）','1 秒','2 秒','3 秒'],[[label+f'：轨迹邻域 {radius} m']+[val(m['splits'][split][str(float(radius))]['coverage']['match_fraction'],j,scale=100) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for radius in (2,4)])
    corrected=table(['全体快速 GT：对照标签正确率（%）','1 秒','2 秒','3 秒'],[[label+' '+model+'：'+name]+[val(m['splits'][split]['2.0']['all_fast']['metrics'][model+'_'+key+'_correct'],j) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for model in ('F','P') for name,key in [('原始','native'),('只平移','translation'),('只旋转','yaw'),('刚体位姿','rigid'),('只改尺寸','size'),('位姿＋尺寸','rigid_size')]])
    gap=table(['验证集：原始 P−F 差距的缩小','1 秒','2 秒','3 秒'],[[name]+[val(m['splits']['val']['2.0']['all_fast']['gap_changes'][key]['gap_reduction'],j) for j in range(3)] for name,key in [('只平移','translation'),('只旋转','yaw'),('刚体位姿','rigid'),('只改尺寸','size'),('位姿＋尺寸','rigid_size')]])
    forecast_gain=table(['验证集：仅看 F 的标签正确率恢复（百分点）','1 秒','2 秒','3 秒'],[[name]+[val(v['2.0']['all_fast']['gap_changes'][key]['forecast_gain'],j) for j in range(3)] for name,key in [('只平移','translation'),('只旋转','yaw'),('刚体位姿','rigid'),('只改尺寸','size'),('位姿＋尺寸','rigid_size')]])
    geom=table(['可靠匹配子集：GT 体素加权几何 IoU（%）','1 秒','2 秒','3 秒'],[[label+' '+model+'：'+name]+[val(m['splits'][split]['2.0']['geometry']['voxel_weighted'][model+'_'+key+'_iou'],j,scale=100) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for model in ('F','P') for name,key in [('原始','base'),('位姿','rigid'),('位姿＋尺寸','rigid_size')]])
    attribution=table(['验证集可靠匹配：IoU 错误的分解（百分点）','1 秒','2 秒','3 秒'],[[model+'：'+name]+[val(v['2.0']['geometry']['voxel_weighted'][model+'_'+key],j,scale=100) for j in range(3)] for model in ('F','P') for name,key in [('平移贡献','position'),('朝向贡献','orientation'),('位姿贡献','pose'),('尺寸贡献','extent'),('未解释残差','residual')]])
    pose_params=table(['验证集可靠匹配：几何纠正量','1 秒','2 秒','3 秒'],[[model+'：'+name]+[val(v['2.0']['geometry']['voxel_weighted'][model+'_'+key],j) for j in range(3)] for model in ('F','P') for name,key in [('质心搬移距离（m）','centroid_displacement'),('无向 yaw 纠正角（度）','yaw_abs_degrees')]])
    qcoverage=table(['历史 query 绑定 GT 覆盖（%）','1 秒','2 秒','3 秒'],[[label]+[val(m['splits'][split]['query_lineage']['coverage']['bound_fraction'],j,scale=100) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')]])
    qtable=table(['绑定子集：GT 体素加权原始几何差异','1 秒','2 秒','3 秒'],[[label+'：'+name]+[val(m['splits'][split]['query_lineage']['voxel_weighted'][key],j,3) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for name,key in [('总差异（m²）','raw_squared'),('刚体可恢复（m²）','pose_removable_squared'),('刚体之后残差（m²）','nonrigid_residual_squared'),('质心水平偏差（m）','centroid_error_xy'),('内部成对距离变化（m）','pair_distance_change_m')]])
    pose_shares=table(['历史绑定子集：刚体可恢复份额（%）','1 秒','2 秒','3 秒'],[[label]+[val(m['splits'][split]['query_lineage']['pose_share'],j) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')]])
    shape_comparison=table(['GT 可比子集：形状保持对照','1 秒','2 秒','3 秒'],[[label+'：'+name]+[val(m['splits'][split]['query_lineage']['shape_comparable']['metrics'][key],j,scale=100) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for name,key in [('历史模板的未来 GT IoU（%）','template_gt_iou'),('纠正位姿后实际表示的未来 GT IoU（%）','aligned_actual_gt_iou'),('保持形状的 IoU 优势（百分点）','template_advantage_over_aligned_actual')]])
    shape_support=table(['GT 可比子集：GT 体素／场景支持','1 秒','2 秒','3 秒'],[[label]+[f"{int(m['splits'][split]['query_lineage']['shape_comparable']['metrics']['template_gt_iou']['support'][j]):,} / {int(m['splits'][split]['query_lineage']['shape_comparable']['metrics']['template_gt_iou']['scenes'][j])}" for j in range(3)] for split,label in [('train','训练集'),('val','验证集')]])
    drop=table(['绑定子集：原始点渲染原因（%）','1 秒','2 秒','3 秒'],[[label+'：'+name]+[val(m['splits'][split]['query_lineage']['voxel_weighted'][key+'_fraction'],j,scale=100) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for name,key in [('超出组中心 3 m','drop_geometry'),('超出原网格','drop_range'),('置信度不足','drop_confidence'),('保留但类别错','kept_wrong_class'),('保留且类别正确','kept_object_class')]])
    gt=table(['GT 物体坐标系：完整可比较实例','1 秒','2 秒','3 秒'],[[label+'：'+name]+[val(m['splits'][split]['gt_body']['voxel_weighted'][key],j,scale=100) for j in range(3)] for split,label in [('train','训练集'),('val','验证集')] for name,key in [('当前 GT 刚体搬移后的未来 IoU（%）','gt_body_iou'),('框尺寸最大相对变化（%）','box_dimension_change')]])
    per_class=table(['验证集快速类别：3 秒历史绑定诊断','绑定 GT 覆盖（%）','刚体可恢复份额（%）','内部距离变化（m）','GT 可比子集：保持形状的 IoU 优势（百分点）'],[[name,val(d['query_coverage']['bound_fraction'],2,scale=100),val(d['pose_share'],2),val(d['query_geometry']['pair_distance_change_m'],2),val(d['shape_comparable']['template_advantage_over_aligned_actual'],2,scale=100)] for name,d in v['by_class'].items()])
    exclusion_rows=[]
    for radius in (2.,4.):
        for row in v[str(radius)]['exclusions']:
            exclusion_rows.append([f'{radius:g} m',row['exclusion'] or '通过匹配条件',f"{row['instances']:,}",f"{row['voxels']:,}"])
    exclusion_table=table(['验证集轨迹邻域','条件／排除组合','实例条目数','GT 体素'],exclusion_rows)
    text=f'''# 快速目标机制｜位置、朝向、预测形状，还是其他错误？

[全量主报告](results.md) · [预先规定的对照](design.md) · [代码与复现](reproduction.md)

**主效应已经通过全量确认。下面的纠正都是使用 GT 的离线诊断，不是可部署预测器，不作为新方法性能。**

“实例条目”以源帧×时距×GT 实例计数，同一物体可在不同源帧重复出现；场景配对重采样保留这种相关性。比例与区间分别使用实例等权或 GT 体素加权，本页主表注明体素加权。

{severity}

{restoration}

{shape_evidence}

## 1. 先看全部目标，不把漏检当成形变

{error}

这些比例覆盖全部快速 GT 体素。非空错类别与为空闲分开计数；为空闲可能来自没有生成目标、位置偏离、几何过滤或低置信度，单凭最终网格无法区分。

## 2. 位置、朝向纠正后，预测错误恢复多少？

{coverage}

{corrected}

对照搬移预测体素并保留类别，未可靠匹配的目标保留原始结果。GT 分母始终不变，因此不会因删除困难目标而提高准确率。不同类别撞入同一新体素时保守标为空闲。

{gap}

正值表示该纠正缩小了原始 P−F 差距，负值表示差距反而扩大。对照按几何 IoU 选择变换，不按 GT 类别标签选变换，所以几何更重叠未必意味着分类更正确。

只改尺寸时，验证集 3 秒 F 标签正确率变化为 {val(v['2.0']['all_fast']['gap_changes']['size']['forecast_gain'],2)} 个百分点，P 的变化为 {val(v['2.0']['all_fast']['gap_changes']['size']['perception_gain'],2)} 个百分点。此时差距缩小来自双方损失不同，不能据此声称尺寸纠正提高了 F 的任务性能；几何 IoU 优化也可能减少 GT 体素上的正确覆盖。

{forecast_gain}

这一表只比较 F 纠正前后，P 保持原始输出；上一表则对 F／P 使用相同纠正规则后比较双方差距。二者回答“F 自身能恢复多少”与“F 的相对劣势能解释多少”两个问题。

![位姿与尺寸对照](figures/motion_oracles.png)

{geom}

{attribution}

平移贡献＋朝向贡献＝刚体位姿收益；位姿贡献＋尺寸贡献＋未解释残差＝原始 1−IoU。两因素 Shapley 贡献对纠正顺序取平均；位置／朝向分解与位姿／尺寸分解的分组不同，不将五行相加。尺寸仅覆盖轴向跨度，几何 IoU 残差包括缺失表面、额外点、未恢复的位姿及 GT／匹配噪声；类别错误由上文固定 GT 分母的标签统计单独衡量。

{pose_params}

角度是已找到的无向点集旋转纠正量，长方体的前后对称会产生 180° 歧义；小目标或接近圆形的目标朝向更不稳定。它不是模型原生框朝向误差或官方检测 mAOE。

这里只覆盖可靠实例匹配子集。2／4 m 邻域的敏感性结果与完整排除分母保存在 `motion_summary.json`；邻域归属歧义、双方缺失、小目标和边界截断均单列，不能将它们直接解释成形变。

<details>
<summary>展开验证集匹配与排除分母</summary>

{exclusion_table}

`small_GT`：GT 小于 20 体素；`F/P_missing_or_sparse`：对应预测点少于 3 个；`association_ambiguous`：归属歧义占比大于 20%；`future_grid_truncated`：未来框被评测网格截断。表中按排除原因组合计数，每个条目只出现一次，避免重复累加。

</details>

## 3. 最终占据缺失时，内部几何发生了什么？

{qcoverage}

本对照使用历史时刻的 GT 绑定当前 720 个 query，再追踪同一 query 的未来解码坐标。保留全部原始点，不因未来置信度低或坐标超出网格而抹去它们。对 48 个点使用无序点集比较，避免把点通道重排当作形变；未来新增 query 槽位另属未覆盖部分。

{qtable}

{pose_shares}

总差异是实际未来点集与“当前预测点集按真实实例刚体运动搬移”的对称平方 Chamfer。最佳已找到的完整 SE(3) 纠正将总差异分成可恢复位姿部分与剩余部分，三项按定义守恒。内部成对距离分布对刚体运动和点重排不变，其变化提供独立的非刚体结构变化证据。

![历史 query 的位姿与非刚体残差](figures/query_decomposition.png)

这是继承历史表示的滚动退化诊断：不覆盖初始表示就没有绑定成功的目标，也不把剩余几何差异直接叫作 GT 物体的真实形变。位置纠正还可能包含 ego 坐标预测的共同偏差，不能只凭本对照将其全部归于物体运动估计。

query 不是模型原生物体实例，绑定的是主要 GT 支持，其全部原始点未必属于同一个物体。因此这是关联表示的结构变化证据，不能直接当作完整物体表面的形变比例。

{drop}

点的互斥丢弃统计按“组中心几何→网格范围→置信度→类别”的顺序计算；另一个顺序可能改变重叠原因的归属。超过组中心 3 m 的点无法仅靠刚体平移旋转恢复，因为刚体变换不改变这个距离。这里是原始点的统计，原版闭运算产生的新体素不冒充原始点。

## 4. 预测形状变化，是否等于真实物体发生形变？

{gt}

当前 GT 按真实实例平移旋转到未来后仍可能与未来 GT 不同，原因包括可见表面、标注与体素化差异。仅靠这份占据标签不能证明刚性车辆发生真实物理形变；行人等可变形类别需要另行阅读。预测点集的内部变化则能直接从同一 query 追踪获得。

尺寸纠正只检查轴向跨度，不补造缺失表面或删除幻影团块。判断形状变化是否有害，还要比较当前表示刚体搬移后的未来 GT IoU，与实际未来表示纠正位姿之后的 GT IoU。这里额外要求当前／未来 GT 都至少 20 体素，且两时刻框均未被网格截断；这项 GT 可比控制保留 {v['query_lineage']['shape_comparable']['instances']:,} 个验证实例条目，逐时距体素与场景支持见 `motion_summary.json` 的 `shape_comparable`。其余历史绑定目标仍保留在原始几何漂移与点丢弃主统计中。

{shape_comparison}

{shape_support}

历史模板与实际表示的差值在各自相同的 GT 可比子集内配对计算。模板保留当前预测表示及其初始误差；实际未来表示向模板进行有限刚体对齐，而非直接向未来 GT 求全局最优。因此正差值支持进一步研究形状保持，但其大小不等于已被唯一识别的形状因果贡献。

{per_class}

车辆与行人分开阅读；空组记为“—”。这里的形状优势按无阈值原始点集计算，不能直接替换原生 occupancy renderer 的任务准确率。

## 5. 怎样回答“哪个更严重”？

{severity}

{restoration}

用全体快速目标的标签恢复量回答实际错误能恢复多少；用可靠匹配子集的 IoU 和历史绑定子集的几何差异回答位姿与形状结构各有多少贡献。三种分母分别报告，不能互换。未解释残差、未绑定目标与错分类也不会被强制塞入“位置”或“形状”两个桶。

87.88% 的分母是相对历史模板的滚动点集差异，不是全部快速体素的预测错误。在可靠占据匹配子集，3 秒 F 的位姿贡献为 {val(v['2.0']['geometry']['voxel_weighted']['F_pose'],2,scale=100)}、尺寸贡献为 {val(v['2.0']['geometry']['voxel_weighted']['F_extent'],2,scale=100)}、未解释残差为 {val(v['2.0']['geometry']['voxel_weighted']['F_residual'],2,scale=100)} 个 IoU 百分点；残差依然很大，不能整体命名为形状错误。因此，本轮确认的是“可追踪表示的滚动偏差中位姿更大，并且形状结构变化也存在”，尚不能给全部任务错误一个完整的位姿／形状二分比例。

这是单一固定 checkpoint 的全量问题诊断。重采样区间描述场景稳定性；不包含多模型、多种子或独立新场景的不确定性，也不是因果唯一归因。

[具体目标图例](cases.md)提供位姿主导、历史形状保持较好和两种差异并存的条件中位数案例；总体结论由本页全量分母决定。
'''
    (REPORT/'motion_results.md').write_text(text)


def figures(s,m):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');font_manager.fontManager.addfont(font.get_file())
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
    dest=REPORT/'figures'
    def save(fig,name):
        for ext in ('png','svg'):fig.savefig(dest/(name+'.'+ext),dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,split,title in zip(axes,('train','val'),('全量训练集','全量验证集')):
        for key,label,color in [('forecast_accuracy','预测器 F','#a45d47'),('perception_accuracy','未来感知器 P','#36799a')]:
            d=s['splits'][split]['attributes']['speed'][key];v=np.asarray(d['value'])[:,2];ci=np.asarray(d['ci95'])[:,2]
            ax.errorbar([1,2,3],v,yerr=np.stack([v-ci[:,0],ci[:,1]-v]),fmt='o-',color=color,label=label)
        ax.set_title(title);ax.set_xticks([1,2,3]);ax.set_xlabel('未来时距（秒）');ax.set_ylabel('快速区域语义准确率（%）');ax.legend(frameon=False);ax.grid(alpha=.15)
    save(fig,'fast_comparison')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');colors=['#b7bec5','#36799a','#ad694e','#648a72']
    for ax,split,title in zip(axes,('train','val'),('训练集：真实占据','验证集：真实占据')):
        v=np.asarray(s['splits'][split]['states']['proportions']['value'])[:,0,1,0];bottom=np.zeros(3)
        for k,label in enumerate(['双方都错','仅感知器对','仅预测器对','双方都对']):ax.bar([1,2,3],v[:,k],bottom=bottom,label=label,color=colors[k]);bottom+=v[:,k]
        ax.set_xticks([1,2,3]);ax.set_title(title);ax.set_xlabel('未来时距（秒）');ax.set_ylabel('比例（%）')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False);save(fig,'four_states')
    counts=dict(np.load(ROOT/'scene_counts.npz'));xy=counts['xy'][counts['split']=='val'].sum(0).reshape(3,40,40,4)
    advantage=np.divide(xy[...,1]-xy[...,2],xy.sum(-1),out=np.full(xy.shape[:-1],np.nan),where=xy.sum(-1)>0)*100
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    for j,ax in enumerate(axes):
        plot=ax.imshow(advantage[j],origin='lower',extent=[-40,40,-40,40],cmap='RdBu',vmin=-25,vmax=25)
        ax.set_title(f'{j+1} 秒：P − F');ax.set_xlabel('y（左方，m）');ax.set_ylabel('x（前方，m）');ax.plot(0,0,'k+',ms=9)
    fig.colorbar(plot,ax=axes,label='P − F（百分点）：蓝色 P 更准，红色 F 更准',shrink=.85);save(fig,'spatial_advantage')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');labels=[];values=[];intervals=[]
    for attr,names,groups in [('speed',['停止／低速','中速','快速'],[0,1,2]),('distance',['0–10 m','10–20 m','20–30 m','≥30 m'],[0,1,2,3]),('static_frustum',['静态：仅未来窗口','静态：两窗口均覆盖'],[1,3])]:
        d=s['splits']['val']['attributes'][attr]['advantage']
        for label,k in zip(names,groups):labels.append(label);values.append(d['value'][2][k]);intervals.append(d['ci95'][2][k])
    values=np.asarray(values);ci=np.asarray(intervals);ax.errorbar(values,np.arange(len(values)),xerr=np.stack([values-ci[:,0],ci[:,1]-values]),fmt='o',color='#36799a')
    ax.axvline(0,color='gray');ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel('3 秒 P − F（百分点；95% 场景配对区间）');ax.grid(axis='x',alpha=.15);save(fig,'attribute_advantage')
    if not m:return
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,metric,title in [(axes[0],'correct','全体快速 GT：标签正确率'),(axes[1],'iou','可靠匹配子集：几何 IoU')]:
        for model,color in [('F','#a45d47'),('P','#36799a')]:
            names=['native','rigid','size','rigid_size'] if metric=='correct' else ['base','rigid','size','rigid_size']
            v=[]
            for name in names:
                d=m['splits']['val']['2.0']['all_fast']['metrics'][model+'_'+name+'_correct'] if metric=='correct' else m['splits']['val']['2.0']['geometry']['voxel_weighted'][model+'_'+name+'_iou']
                v.append(d['value'][2]*(1 if metric=='correct' else 100))
            ax.plot(range(4),v,'o-',color=color,label=model)
        ax.set_xticks(range(4),['原始','位姿','尺寸','位姿＋尺寸']);ax.set_title(title+'：3 秒');ax.set_ylabel('%');ax.legend(frameon=False);ax.grid(alpha=.15)
    save(fig,'motion_oracles')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,split,title in zip(axes,('train','val'),('训练集：历史绑定子集','验证集：历史绑定子集')):
        q=m['splits'][split]['query_lineage']['voxel_weighted'];a=np.asarray(q['pose_removable_squared']['value']);b=np.asarray(q['nonrigid_residual_squared']['value'])
        ax.bar([1,2,3],a,label='刚体可恢复',color='#648a72');ax.bar([1,2,3],b,bottom=a,label='刚体之后的残差',color='#b7bec5')
        ax.set_xticks([1,2,3]);ax.set_xlabel('未来时距（秒）');ax.set_ylabel('对称平方 Chamfer（m²）');ax.set_title(title);ax.legend(frameon=False)
    save(fig,'query_decomposition')

if __name__=='__main__':main()
