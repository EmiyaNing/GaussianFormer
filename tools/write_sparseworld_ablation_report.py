#!/usr/bin/env python3
"""Readable reports from the completed four-ablation run."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.sparseworld_region_ablation_common import OUT,REPORT,GROUPS


def fmt(x):return '—' if x is None else f'{x:.2f}'
def interval(v,c):return f'{fmt(v)} [{fmt(c[0])}, {fmt(c[1])}]'
def at(d,*ix):
    v,c=d['value'],d['ci']
    for i in ix:v,c=v[i],c[i]
    return interval(v,c)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def save(name,text): (REPORT/name).write_text(text,encoding='utf8')

def main():
    s=json.loads((OUT/'summary.json').read_text());assert len(s['adaptation'])==9 and len(s.get('budget_followup',{}))==21
    models=s['models'];group='static_future_only'
    base='> 已执行 · 2026-10-08。除特别说明外，GPU 干预为 150 样本／150 场景，时距 1／2／3 秒；括号为场景级配对 bootstrap 95% 区间，2000 次。区域准确率只统计 GT 占据，完整域 mIoU／OccIoU 包含误报。\n\n'
    rows=[]
    descriptions={'F':'原生历史预测器','H':'历史五帧，目标坐标感知','P':'未来五帧感知','M':'历史／未来混合五帧','HP_full':'双窗口融合，1440 queries','HP_equal_points':'双窗口融合，总720 queries'}
    for a in descriptions:
        d=models[a];rows.append([descriptions[a],*[fmt(v) for v in d['metrics']['mIoU']['value']],at(d['groups'][group],2),at(d['delta_vs_P'][group],2)])
    text='# A1：历史、未来和联合窗口\n\n'+base
    text+='**历史与未来存在可以利用的局部互补，但当前融合方式不能带来全域净收益；联合后的高局部准确率明显依赖点预算。**\n\n'
    text+='H、P、M 使用同一个感知头和 720 个 query，每个 query 输出 48 点。H 的历史图像投影到目标坐标，它不是原生预测器 F。M 从两个窗口并集等间隔取五帧；HP_full 分别解码 H、P 后统一渲染。\n\n'
    text+=table(['输入／输出','1 s mIoU','2 s mIoU','3 s mIoU','3 s 静态独有视锥准确率','该区域相对 P 增量（pp）'],rows)
    text+='\n![窗口对照](figures/a1_windows.png)\n\n'
    text+='混合五帧在目标静态区域比 P 高 **'+at(models['M']['delta_vs_P'][group],2)+' pp**，但 3 秒全域 mIoU 从 '+fmt(models['P']['metrics']['mIoU']['value'][2])+' 降为 '+fmt(models['M']['metrics']['mIoU']['value'][2])+'%。这是局部互补与整体代价同时存在的例子。\n\n'
    text+='HP_full 的局部正确率达到 '+fmt(models['HP_full']['groups'][group]['value'][2])+'%，但减少到与单支相同的点数后降到 '+fmt(models['HP_equal_points']['groups'][group]['value'][2])+'%。H+H、P+P 的重复点融合均与单支逐体素完全相同，因此“多跑一遍”本身不能解释 HP_full 的变化；额外不同位置的点及其渲染交互仍是混杂因素。没有把这个现象归因于纯粹的信息增加。\n\n'
    text+='**匹配输出头适配也已完成。**H／P／M 分别使用同样 256 个训练场景、三个训练种子、每组 768 次更新；只更新原版最后的 cls／reg 分支，输入编码和交互冻结。三时距任务共同训练，训练／验证场景无交集。以下为三个种子的均值 ± 标准差（百分点标度），不是 bootstrap 区间。\n\n'
    rows=[]
    for a in ['H','P','M']:
        dd=[v for k,v in s['adaptation'].items() if k.startswith(a+'_')]
        mi=np.array([v['metrics']['mIoU']['value'] for v in dd]);acc=np.array([v['groups'][group]['value'][2] for v in dd])
        rows.append([a,*[f'{mi[:,j].mean():.3f} ± {mi[:,j].std(ddof=1):.3f}' for j in range(3)],f'{acc.mean():.2f} ± {acc.std(ddof=1):.2f}'])
    text+=table(['匹配适配输入','1 s mIoU','2 s mIoU','3 s mIoU','3 s 静态独有视锥准确率'],rows)
    text+='\n完整逐种子数值见[统计附表](statistics.md)。这检验的是有限输出接口适配，不能排除编码器对窗口变化的分布偏移，也不代表端到端训练已经收敛。因此，本实验支持继续研究历史／未来证据融合，但没有证明“仅仅增加历史输入就能解决问题”。\n'
    save('a1_windows.md',text)
    text='# A2：当前距离、未来距离与观测质量\n\n'+base.replace('GPU 干预为 150 样本／150 场景','距离分析为 512 样本／150 场景，观测质量为 150 样本／150 场景')
    text+='**“靠近当前车就更适合预测器”不成立；更具体的局部条件是：部分原来在 10–30 m、未来落到 ≥30 m 的静态区域偏向 F。**以下距离均为水平距离，同一世界位置由目标 ego 映射到当前 ego；对于动态类别，当前距离是未来目标位置到当前车的距离，不是目标的历史位置。\n\n'
    rows=[]
    for key,name in [('current_near10','距当前车 <10 m'),('target_near10','距目标车 <10 m'),('current_near_target_far20','当前 <10 m 且目标 ≥20 m')]:
        d=s['observation']['distance_regions'][key]
        rows.append([name,at(d['P_minus_F'],2,0),at(d['P_minus_F'],2,1)])
    text+=table(['区域（3 s）','全部占据 P−F（pp）','静态占据 P−F（pp）'],rows)
    text+='\n![双坐标距离交叉表](figures/a2_distance.png)\n\n图横轴是目标时刻距离，纵轴是当前距离。红色 P 更强，蓝色 F 更强；图中数值是百分点。颜色截断于 ±20 pp，少于 20 个场景支持的格子不显示。\n\n'
    d=s['observation']['distance'];rows=[]
    for a,b,name in [(0,0,'当前和目标均 <10 m'),(1,3,'当前 10–20 m，目标 ≥30 m'),(2,3,'当前 20–30 m，目标 ≥30 m'),(3,3,'当前和目标均 ≥30 m')]:
        rows.append([name,at(d['P_minus_F'],2,1,a,b),d['F']['scenes'][2][1][a][b]])
    text+=table(['静态区域（3 s）','P−F（pp）','支持场景'],rows)
    text+='\n这支持“相对观测位置迁移与局部优势有关”的条件关联；它不证明距离是唯一原因，也没有排除点分布随目标坐标变化的影响。交叉表的细分属于本轮探索性分析。\n\n'
    text+='观测质量使用实际输入视角投影，统计视锥次数、投影尺度，并将各输入时刻 GT 占据投影成 4-pixel 深度缓冲（3×3 最小滤波、0.8 m 容差）。目标静态体素与该表面深度一致，才记为几何深度支持；在前方或无深度记录的位置记 unknown，不能直接称无遮挡。\n\n'
    rows=[]
    for d in s['observation']['depth_supported_vs_zero_standardized']:
        rows.append([d['branch'],d['horizon'],interval(d['increment'],d['ci']),d['strata']])
    text+=table(['哪一窗口有深度支持','时距（s）','支持组相对零支持组的 P−F 增量（pp）','共同类别×目标距离层'],rows)
    text+='\n上述标准化在每层双方至少 100 个体素的共同支持域内，以合并体素频数固定加权。F 历史窗口深度支持与更偏向 F 的差值相关；P 窗口的同类指标在 3 秒没有稳定非零增量。不能把几何代理自动解释成真实可见性，也不能以此否定未来图像信息——A4 将直接检验输入依赖。\n'
    save('a2_migration.md',text)
    text='# A3：点预算、空间覆盖与后处理\n\n'+base
    text+='**本轮最强的机制线索来自稀疏表示：F 的静态局部优势对 query 预算与 padding 高度敏感。它不能被简单解释为“历史中蕴含更多正确语义”。**\n\n'
    rows=[]
    for mode,d in s['paired_renderer_advantage'].items():rows.append([mode,*[at(d[group],j) for j in range(3)]])
    text+=table(['双方配对设置','1 s P−F（pp）','2 s P−F（pp）','3 s P−F（pp）'],rows)
    text+='\n![表示与渲染对照](figures/a3_controls.png)\n\n'
    text+='原版 3 秒 F 有 1040 个 query，P 有 720 个。equal_count 将 F 随机保留 720 个完整 query；equal_spatial 在 4 m 水平块内令双方 query 中心数量一致，3 秒平均每支保留 '+fmt(s['diagnostics']['equal_spatial_queries_6']['mean'])+' 个。两种控制均改变了自然输出，不能把删点后的性能当成原生模型的公平排名；但它们确实说明原始优势不对点预算干预保持不变。\n\n'
    text+='**删除 query 同时改变数量、身份和空间支持；本实验不把优势反转解释为点数的唯一因果贡献。**关闭 padding 后两支准确率都大幅降低，不能把差值缩小写成模型被“修复”。group 消融同时关闭紧凑度修正与中心距离过滤，也不单独归因于其中一项。\n\n'
    rows=[]
    for a in ['F','P']:
        rows.append([a,at(s['raw_point_coverage'][a][group],2),at(models[a+'_no_padding']['groups'][group],2),at(models[a]['groups'][group],2)])
    text+=table(['分支','原始点命中体素率（%）','无 padding 语义准确率（%）','原版语义准确率（%）'],rows)
    text+='\n原始点命中率不要求类别正确；与两个语义准确率的分子不同。它用来说明表示覆盖和后处理的作用，不能与准确率直接相减作误差分解。\n\n'
    text+='为排除一次随机删点的偶然性，在初筛后追加三个种子、360／540／720 queries 与 padding 的交叉对照。此追加使用原场景，是事后复核。\n\n'
    rows=[]
    for budget in (360,540,720):
        for mode in ('native','no_padding'):
            dd=[s['budget_followup'][f'{budget}_{mode}_{seed}'][group]['P_minus_F'] for seed in (20261008,20261009,20261010)]
            rows.append([budget,mode,*[at(d) for d in dd]])
    text+=table(['双方 query 数','padding 设置','种子 20261008 P−F','种子 20261009 P−F','种子 20261010 P−F'],rows)
    text+='\n720 query 且保留原版 padding 时，三个种子点估计都偏向 P，但后两个区间跨零。因此可复核的结论是原始 F 优势对预算干预敏感，不能宣称等预算后 P 在所有采样下都显著更强。更低预算时方向也随采样变化，说明 query 身份和分布同样重要。\n\n'
    rows=[]
    for seed in (20261008,20261009,20261010):rows.append([seed,at(s['budget_followup'][f'spatial_{seed}'][group]['P_minus_F'])])
    text+=table(['空间配额匹配种子','3 s P−F（pp）'],rows)
    text+='\n其他接口干预也表明改进不能只看一个区域。P 取消范围自适应缩放后，静态独有视锥准确率提高 '+at(models['P_unit_scale']['delta_vs_P'][group],2)+' pp，但全域 3 秒 mIoU 从 '+fmt(models['P']['metrics']['mIoU']['value'][2])+' 降到 '+fmt(models['P_unit_scale']['metrics']['mIoU']['value'][2])+'%。\n\n'
    text+='在 1.2 m 内用 P 的最近点语义替换 F 语义，保持 F 几何，3 秒 mIoU 为 '+fmt(models['F_geometry_P_semantics']['metrics']['mIoU']['value'][2])+'%，高于原 F 的 '+fmt(models['F']['metrics']['mIoU']['value'][2])+'%；反向把 F 语义放到 P 几何则为 '+fmt(models['P_geometry_F_semantics']['metrics']['mIoU']['value'][2])+'%，低于原 P。两方向平均点匹配率分别为 '+fmt(100*s['diagnostics']['F_geometry_P_semantics_match_fraction_6']['mean'])+'%／'+fmt(100*s['diagnostics']['P_geometry_F_semantics_match_fraction_6']['mean'])+'%。这支持几何与语义贡献应分开分析；最近点交换不是精确的一一对齐，也不是可部署收益。\n'
    save('a3_sparse_representation.md',text)
    text='# A4：历史上下文与未来局部图像干预\n\n'+base
    text+='**F 确实依赖 ROI 周围的历史视觉上下文；P 也确实使用未来局部图像。未来窗口的局部失败不能解释成“模型完全不看该处图像”。**\n\n'
    text+='本实验只评估 3 秒。ROI 按 GT 覆盖数选择，不按谁正确选择：在“未来视锥独有静态”或“当前 10 m 内静态”中取 GT 最多的 2 m 水平块。F 遮挡其 8 m 邻域静态上下文的投影，P 遮挡 ROI 本身的投影；填充值为归一化零（均值颜色）。每个视角使用两种循环平移 mask 保持完全相同的像素面积。\n\n'
    text+='**F/P 遮挡区域的定义和面积不同，因此只能在同一分支内比较定向与随机干预。**主表排除该分支 mask 全零的场景；不将没有实际遮挡当成“不依赖输入”。\n\n'
    rows=[]
    for roi,v in s['interventions'].items():
        for b in ('F','P'):
            d=v[b+'_all'];rows.append([roi,b,d['scenes'],fmt(d['mask_fraction_mean']*100),fmt(d['baseline']['value']),at(d['targeted_minus_baseline']),at(d['targeted_minus_random'])])
    text+=table(['ROI','分支','有效场景','平均像素面积（%）','原准确率（%）','定向−原始（pp）','定向−随机（pp）'],rows)
    text+='\n![局部干预效果](figures/a4_interventions.png)\n\n'
    text+='在未来视锥独有静态 ROI 中，F 的定向上下文移除比同面积随机移除多损失 '+fmt(-s['interventions']['future_only_static']['F_all']['targeted_minus_random']['value'])+' pp。由于 ROI 体素中心本身不在历史视锥中，这提供了上下文依赖的受控证据；仍不能进一步区分邻近物体、结构连续性或更广泛的图像关联。\n\n'
    text+='P 遮挡 ROI 的投影也比随机遮挡造成更明显下降，说明新增图像确实参与判断。结合 A3，更合理的解释是“视觉证据与稀疏输出转换之间存在限制”，而不是“未来图像没有被使用”。\n\n'
    text+='50 个原分析留出场景中，future-only ROI 的 F／P 定向−随机差分别为 '+at(s['interventions']['future_only_static']['F_holdout']['targeted_minus_random'])+'／'+at(s['interventions']['future_only_static']['P_holdout']['targeted_minus_random'])+' pp，有效场景均为 49。该方向在分析留出复现，但这些场景仍不是新的独立测试数据。\n\n'
    text+='![干预像素示例](figures/a4_mask_example.png)\n\n示例按固定样本身份顺序选取第一个双方均有有效 mask 的场景，再显示每支 mask 面积最大的视角；没有按干预效应选择。随机 mask 可能碰到其他相关内容，灰色遮挡也可能引入分布偏移；因此这里识别的是模型对这类干预的相对依赖，不是对真实世界唯一成因的证明。\n'
    save('a4_input_interventions.md',text)
    # Exhaustive main metric table + long-form CSV.
    text='# 统计附表\n\n全域指标为百分数；区域差值为百分点。区间均为场景级配对 bootstrap。\n\n'
    rows=[];csvrows=[]
    for a,d in models.items():
        rows.append([a,*[at(d['metrics']['mIoU'],j) for j in range(3)],at(d['groups'][group],2)])
        for domain in ('models','holdout'):
            dd=s[domain][a]
            for g in GROUPS:
                for j in range(3):
                    v=dd['groups'][g];delta=dd['delta_vs_P'][g]
                    csvrows.append(dict(domain=domain,arm=a,group=g,horizon=j+1,accuracy=v['value'][j],lower=v['ci'][j][0],upper=v['ci'][j][1],
                       voxels=v['n'][j],scenes=v['scenes'][j],delta_vs_P=delta['value'][j],delta_lower=delta['ci'][j][0],delta_upper=delta['ci'][j][1]))
    text+=table(['设置','1 s mIoU','2 s mIoU','3 s mIoU','3 s 静态独有视锥准确率'],rows)
    text+='\n## 输出头适配逐种子结果\n\n'
    rows=[]
    for a,d in s['adaptation'].items():rows.append([a,*[fmt(v) for v in d['metrics']['mIoU']['value']],at(d['groups'][group],2)])
    text+=table(['组别／种子','1 s mIoU','2 s mIoU','3 s mIoU','3 s 静态独有视锥准确率'],rows)
    text+='\n## 3 秒静态双距离全部交叉格\n\n';rows=[];dist=['0–10','10–20','20–30','≥30'];d=s['observation']['distance']
    for i in range(4):
        for j in range(4):rows.append([dist[i],dist[j],at(d['P_minus_F'],2,1,i,j),d['F']['n'][2][1][i][j],d['F']['scenes'][2][1][i][j]])
    text+=table(['当前距离（m）','目标距离（m）','P−F（pp）','占据体素任务','场景'],rows)
    text+='\n全部九个区域、三时距、全体／留出对照见 [regional_statistics.csv](regional_statistics.csv)。更多观测次数、深度支持和投影尺度分箱、OccIoU 与区间保存在 [summary.json](../../../../out/sparseworld_region_ablations/20261008/summary.json)。\n'
    save('statistics.md',text)
    with (REPORT/'regional_statistics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    from tools.polish_sparseworld_ablation_reports import render_reports
    render_reports(REPORT, s)
    print('Wrote illustrated A1-A4 reports and complete statistics',flush=True)
if __name__=='__main__':main()
