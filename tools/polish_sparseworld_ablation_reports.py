"""Editorial templates over the existing numeric report tables; no experiment changes."""
import re
from pathlib import Path
import numpy as np

TEMPLATES=Path(__file__).parent/'report_templates'/'sparseworld_ablations'
LABELS={
'F':'预测器 F','P':'未来感知 P','H':'历史感知 H','M':'混合感知 M',
'HP_full':'双窗口融合：1440 query','HP_equal_points':'双窗口融合：720 query',
'native':'原版设置','equal_count':'等 query 数','equal_spatial':'等空间配额',
'no_padding':'关闭空间填补','no_group_filter':'关闭组几何过滤',
'threshold_low':'统一低阈值 0.2','threshold_high':'统一高阈值 0.5',
'future_only_static':'未来视锥独有静态','current_near_static':'当前 10 m 内静态',
'F_geometry_P_semantics':'F 的位置 + P 的语义','P_geometry_F_semantics':'P 的位置 + F 的语义',
'P_unit_scale':'P：取消自适应缩放'}
for branch in ('F','P'):
    for code in ('equal_count','equal_spatial','no_padding','no_group_filter','threshold_low','threshold_high'):
        LABELS[f'{branch}_{code}']=f'{branch}：{LABELS[code]}'

def translate_table(s):
    for key,value in LABELS.items():s=s.replace('| '+key+' |','| '+value+' |')
    return s

def tables(s):return re.findall(r'(?:^\|.*\|\n)+',s,re.M)

def render_reports(report,s):
    from tools.write_sparseworld_ablation_report import table,fmt,at
    names=['a1_windows.md','a2_migration.md','a3_sparse_representation.md','a4_input_interventions.md']
    originals={name:(report/name).read_text() for name in names}
    original_stats=(report/'statistics.md').read_text()
    ts={name:tables(text) for name,text in originals.items()}
    m=s['models'];g='static_future_only'
    context={}
    arms=['F','H','P','M','HP_full','HP_equal_points']
    context['A1_SHORT']=table(['3 秒设置','整体 mIoU（%）','局部准确率（%）','局部相对 P 的变化（百分点）'],[
        [LABELS[a],fmt(m[a]['metrics']['mIoU']['value'][2]),fmt(m[a]['groups'][g]['value'][2]),at(m[a]['delta_vs_P'][g],2)] for a in arms])
    rows=[]
    for a in ['H','P','M']:
        ds=[v for k,v in s['adaptation'].items() if k.startswith(a+'_')]
        mi=np.array([d['metrics']['mIoU']['value'][2] for d in ds]);acc=np.array([d['groups'][g]['value'][2] for d in ds])
        rows.append([LABELS[a],f'{mi.mean():.3f} ± {mi.std(ddof=1):.3f}',f'{acc.mean():.2f} ± {acc.std(ddof=1):.2f}'])
    context['A1_ADAPT_SHORT']=table(['适配的输入','3 秒整体 mIoU（%）','3 秒局部准确率（%）'],rows)
    context['A2_NEAR_TABLE']=ts[names[1]][0];context['A2_CROSS_TABLE']=ts[names[1]][1]
    context['A3_SHORT']=table(['3 秒配对设置','P−F（百分点）','怎样理解'],[
        [LABELS[a],at(s['paired_renderer_advantage'][a][g],2),read] for a,read in [
        ('native','原始 F 占优'),('equal_count','该种子下转为 P 占优'),('equal_spatial','该种子下转为 P 占优'),('no_padding','F 仍占优，但差值缩小')]])
    context['A3_COVERAGE_TABLE']=ts[names[2]][1]
    context['A3_SEED_SHORT']=table(['采样种子','双方 720 query：保留填补','双方 720 query：关闭填补'],[
        [seed,at(s['budget_followup'][f'720_native_{seed}'][g]['P_minus_F']),at(s['budget_followup'][f'720_no_padding_{seed}'][g]['P_minus_F'])] for seed in (20261008,20261009,20261010)])
    rows=[];areas=[]
    for roi,data in s['interventions'].items():
        for b in ('F','P'):
            d=data[b+'_all'];rows.append([LABELS[roi],b,d['scenes'],at(d['targeted_minus_random'])])
            areas.append([LABELS[roi],b,fmt(100*d['mask_fraction_mean'])+'%'])
    context['A4_EFFECT_SHORT']=table(['待评估区域','分支','有效场景','定向 − 随机（百分点）'],rows)
    context['A4_AREA_SHORT']=table(['待评估区域','分支','平均遮挡像素比例'],areas)
    for name in names:
        text=(TEMPLATES/name).read_text()
        for key,value in context.items():text=text.replace('{{'+key+'}}',translate_table(value).rstrip())
        assert '{{' not in text,name
        (report/name).write_text(text)
    intro='''# 统计附表｜按问题查找完整数字

[返回总览](README.md) · [执行口径](protocol.md) · [机器可读 CSV](regional_statistics.csv)

这里保留全部主对照、三个时距和随机种子结果。首次阅读建议先看四篇实验正文，再回来核对区间和样本数。

| 想查什么 | 跳转 |
|---|---|
| 输入窗口与输出头适配 | [A1 完整表](#a1) |
| 双坐标距离与深度支持 | [A2 完整表](#a2) |
| 点预算、后处理及多种子复核 | [A3 完整表](#a3) |
| 遮挡面积、原准确率和干预差值 | [A4 完整表](#a4) |
| 各设置的完整域指标、适配逐种子分数、距离全部交叉格 | [逐设置查询](#all-metrics) |

**统一读法：**P−F 为正表示感知器更好，为负表示预测器更好。准确率、mIoU 用百分数；它们的差用百分点（表中 pp）。`数值 [下限, 上限]` 给出估计值与场景级 95% 区间；区间跨 0 表示该方向未稳定排除 0。适配汇总中的 `±` 是三个训练种子的标准差，含义不同。

除 A2 距离表使用完整 512 样本外，GPU 主对照使用 150 样本／150 场景。局部“静态独有视锥”均指只进入未来输入窗口视锥的静态 GT；它没有包含空闲体素上的误报，需结合全域指标阅读。

'''
    titles=[['窗口：三个时距','匹配输出头适配：均值与标准差'],['近距区域','选出的双距离静态子组','深度支持的标准化差值'],['配对表示与后处理','原始点命中率与最终准确率','预算 × 空间填补 × 采样种子','空间配额的多种子复核'],['ROI、像素面积与全部干预差值']]
    for i,name in enumerate(names):
        intro+=f'<a id="a{i+1}"></a>\n\n## A{i+1} 完整表\n\n'
        for title,t in zip(titles[i],ts[name]):intro+=f'### {title}\n\n'+translate_table(t)+'\n'
    intro+='<a id="all-metrics"></a>\n\n## 逐设置查询\n\n'
    intro+=translate_table(original_stats.split('\n',1)[1]).replace('## 输出头适配逐种子结果','### 输出头适配逐种子结果').replace('## 3 秒静态双距离全部交叉格','### 3 秒静态双距离全部交叉格')
    intro+='''
## 对照名称与缓存字段的对应

CSV 和 JSON 保留原始字段，便于脚本复算。`equal_count` 表示等 query 数；`equal_spatial` 表示按 4 m 块匹配数量；`no_padding` 表示关闭空间填补；`no_group_filter` 同时关闭紧凑度修正和中心距离过滤；`unit_scale` 表示初始缩放为 1。`geometry_P_semantics` 表示保留接收方点位置、换入 P 的最近点语义。

图表和区间属于探索性分析，没有做多重比较校正。缺少支持的格子记为“—”或 null；不把它们当成零准确率。
'''
    (report/'statistics.md').write_text(intro)
