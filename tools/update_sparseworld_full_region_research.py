#!/usr/bin/env python3
"""Promote validated full-census research; preserve readable pilot history."""
import json,os,re,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.sparseworld_full_regions_common import ROOT,REPORT
from tools.sparseworld_region_utils import write_json,sha256

BASE=REPORT.parent
DISTILLATION=BASE.parent/'fast_motion_distillation'
UPDATED_DATE='2026-10-11'


def rewrite_links(text,source,destination,mapping=None,fallback_source=None):
    def replace(match):
        link=match.group(1)
        if link.startswith(('http:','https:','#')):return match.group(0)
        body,*anchor=link.split('#',1);target=(source/body).resolve()
        if not target.exists() and fallback_source is not None:
            fallback=(fallback_source/body).resolve()
            if fallback.exists() or (mapping and fallback in mapping):target=fallback
        if mapping and target in mapping:target=mapping[target]
        relative=os.path.relpath(target,destination)
        if anchor:relative+='#'+anchor[0]
        return ']('+relative+')'
    return re.sub(r'\]\(([^)]+)\)',replace,text)


def archive():
    dest=BASE/'pilot512';dest.mkdir(exist_ok=True)
    if (dest/'snapshot.json').exists():return
    names=['README.md','results.md','statistics.md','cases.md','reproduction.md','cvpr2027_research_direction.md']
    files=[BASE/n for n in names if (BASE/n).exists()];mapping={p.resolve():(dest/p.name).resolve() for p in files}
    # The distillation directory is a sibling; repair stale links in the copy
    # while preserving hashes of the original pilot documents.
    mapping.update({(BASE/'fast_motion_distillation'/p.relative_to(DISTILLATION)).resolve():p.resolve()
                    for p in DISTILLATION.rglob('*') if p.is_file()})
    figures=BASE/'figures'
    if figures.exists():
        shutil.copytree(figures,dest/'figures',dirs_exist_ok=True)
        mapping.update({p.resolve():(dest/'figures'/p.relative_to(figures)).resolve() for p in figures.rglob('*') if p.is_file()})
    if (BASE/'attribute_statistics.csv').exists():mapping[(BASE/'attribute_statistics.csv').resolve()]=(dest/'attribute_statistics.csv').resolve()
    hashes={str(p):sha256(p) for p in files}
    for p in files:(dest/p.name).write_text(rewrite_links(p.read_text(),BASE,dest,mapping))
    for name in ('attribute_statistics.csv',):
        if (BASE/name).exists():shutil.copyfile(BASE/name,dest/name)
    write_json(dest/'snapshot.json',dict(scope='Historical 512 validation samples; preserved before full-census promotion',original_sha256=hashes))


def main():
    assert json.loads((ROOT/'final_validation.json').read_text())['status']=='passed'
    s=json.loads((ROOT/'summary.json').read_text());archive()
    historical_mapping={(BASE/name).resolve():(BASE/'pilot512'/name).resolve() for name in
        ('README.md','results.md','statistics.md','cases.md','reproduction.md','cvpr2027_research_direction.md')}
    for section in (BASE/'ablations',DISTILLATION):
        for p in section.rglob('*.md'):
            prior_parent=BASE/'fast_motion_distillation'/p.relative_to(DISTILLATION).parent if section==DISTILLATION else None
            updated=rewrite_links(p.read_text(),p.parent,p.parent,historical_mapping,prior_parent)
            if updated!=p.read_text():p.write_text(updated)
    for name in ('results.md','statistics.md','reproduction.md'):
        (BASE/name).write_text(rewrite_links((REPORT/name).read_text(),REPORT,BASE))
    shutil.copyfile(REPORT/'region_statistics.csv',BASE/'attribute_statistics.csv')
    (BASE/'README.md').write_text('''# 预测器与未来感知器｜全量 nuScenes 区域研究

**2026-10-11 更新：原始同权重 SparseWorld 的区域研究已经扩展到全部 700 个训练场景与 150 个验证场景，逐样本全量验收通过。**29,049 个有效源帧、82,047 个未来时距配对；缺少历史或未来 GT 的场景边界逐时距排除，未随机抽样。

先读[全量主报告](results.md)，再读[快速目标机制报告](full_dataset/motion_results.md)。前者回答优势是否复现，后者区分位姿可恢复错误、预测尺寸／结构变化、分类与漏占据，并保留每种诊断的覆盖分母。

| 想了解什么 | 入口 |
|---|---|
| 全量 train／val、四种区域状态、速度与视锥结论 | [结果报告](results.md) |
| 快速目标位置／朝向／形状以及渲染丢弃机制 | [机制报告](full_dataset/motion_results.md) |
| 全部数值、场景配对区间与标准化关联 | [统计附表](statistics.md) |
| 全量覆盖与对照定义 | [实验设计](full_dataset/design.md) |
| 运行代码、可复现命令与缓存 | [复现说明](reproduction.md) |
| 下一阶段研究与方法设计 | [研究方向更新](cvpr2027_research_direction.md) |
| 旧 512 样本的图册与完整历史报告 | [历史快照](pilot512/README.md) |
| 原输入／点预算等机制干预，仍属原样本规模 | [历史消融](ablations/README.md) |
| 2048／512 蒸馏与修复，仍属原样本规模 | [蒸馏研究](../fast_motion_distillation/README.md) |

快速区域定义为 GT 实例速度 ≥2 m/s。验证集 1／2／3 秒的感知器优势为 17.89／34.08／43.73 个百分点，场景配对区间均高于零。位置与朝向纠正中，位置恢复更大；历史绑定子集 3 秒的刚体可恢复几何差异占 87.88% [85.24, 90.08]，该子集覆盖 64.91% 的快速 GT 体素。同时存在内部结构变化，保持历史形状在 GT 可比子集的 IoU 高 17.83 个百分点。该结果支持优先解决位姿并约束预测形状，不能据此认定真实物体发生物理形变。

完整验收及代码／报告哈希见[验收记录](../../../out/sparseworld_region_comparison/20261010_full_nuscenes/final_validation.json)。旧 512 样本已逐字段精确回放；相关行为测试 10 项通过。

训练集成绩属于已训练分布内统计；验证场景也已参与此前研究。完整帧覆盖增强了数据集内复核，不能自动当作新场景泛化证据。原消融与蒸馏的数值没有被替换或扩大分母。
''')
    direction='''# 研究方向更新｜以全量区域与几何机制为问题消融

**本页以全量诊断为当前证据；原 512 样本提出的方向保留在[历史建议](pilot512/cvpr2027_research_direction.md)。**

核心路线应从“直接对齐快速区域分类分数”推进到“明确对象身份、对齐位姿与对象坐标形状、再学习语义占据”。用[全量主报告](results.md)确认感知器优势，用[机制报告](full_dataset/motion_results.md)决定方法预算首先投向哪个误差来源；不预设位姿和形状一定有相同权重。

| 全量证据 | 方法与下一轮对照 |
|---|---|
| 位姿纠正缩小 F／P 差距 | 物体运动与 ego 运动分开建模；共享刚体位姿更新，避免逐点速度自由漂移；比较真实可预测运动与 GT oracle 上限 |
| 内部非刚体变化且保持历史形状更接近未来 GT | 持久对象形状表示与小残差更新；车辆使用刚性约束，行人保留可变形能力；形状损失在对象坐标系计算 |
| 错分类／置信度过滤仍造成明显错误 | 在可靠几何对应之后监督类别和占据；分别记录几何覆盖与语义证据，避免把无支持体素硬当自由空间 |
| 历史无可靠绑定或未来新增槽位 | 独立处理出生／遮挡／未观测目标，报告可观测性和覆盖率；不使用 GT 身份或速度 mask 作为推理输入 |

全量验证快速区域 P−F 为 17.89／34.08／43.73 个百分点。1／2／3 秒历史绑定子集的刚体可恢复份额为 77.89%／84.48%／87.88%，且相应场景区间均高于 50%；位姿应作为优先问题，其中平移恢复量大于单独 yaw。3 秒历史绑定覆盖 64.91% 的快速 GT 体素，未覆盖目标不沿用该排序。

预测内部结构也随滚动发生变化：验证 3 秒内部成对距离变化为 2.60 m，GT 可比子集保持历史形状的 IoU 高 17.83 个百分点。该证据支持持久形状表示与受限残差，但有限位姿搜索、初始形状误差和表观 GT 变化仍需进一步控制。只改轴向尺寸反而使 F 原生标签正确率下降 0.68 个百分点，因此不能把几何尺寸优化直接当成任务性能改善。

87.88% 衡量可追踪表示相对历史模板的滚动点集差异，不能改写成“解释全部快速预测错误的 87.88%”。可靠占据匹配子集的 3 秒未解释 IoU 残差仍为 58.43 个百分点，且纯位姿纠正后全体快速 GT 标签正确率仅为 26.21%；漏占据、分类与未覆盖目标需要继续单独研究。

```mermaid
flowchart LR
    H[历史观测] --> O[持续对象表示]
    O --> M[对象位姿＋ego 运动]
    O --> S[对象形状＋受限残差]
    M --> G[对应未来几何]
    S --> G
    G --> C[语义占据读出]
    T[训练期未来 teacher] --> P[GT 辅助对齐与监督]
    P --> M
    P --> S
    P --> C
```

GT 辅助的 oracle、query 身份绑定与几何搬移仅用于训练监督或离线诊断；新方法必须使用历史信息推理。对象位姿、形状与语义三种收益需要分开消融，再验证完整组合是否守住静态区域和全域 mIoU。

论文问题消融应引用全量 F／P、可靠匹配覆盖率、历史绑定覆盖率、GT 体坐标控制与两类几何分解；原输入／点预算干预继续标注其原样本规模。单个固定 checkpoint 的全量统计不足以证明跨模型普遍性；下一轮方法结果应增加独立种子与未用于问题分析的评测。
'''
    (BASE/'cvpr2027_research_direction.md').write_text(direction)
    (BASE/'cases.md').write_text('# 图表与具体案例\n\n本轮全量空间与属性图见[主报告](results.md)，位姿／尺寸与原始 query 的机制图见[机制报告](full_dataset/motion_results.md)。具体目标与原始点云图见[全量案例](full_dataset/cases.md)。\n\n此前按四种区域状态选择的 512 样本具体图册已保留为[历史案例](pilot512/cases.md)，不将这些旧案例的样本范围改写成全量。\n')
    for p in (BASE.parent.parent/'sparseworld_prediction_perception_regions_plan.md',BASE.parent.parent/'sparseworld_prediction_perception_regions_protocol.md'):
        text=p.read_text();note=f'> **{UPDATED_DATE} 全量扩展：**原 512 样本研究已扩展至全部 700／150 个 train／val 场景，当前证据见[全量结果](hypotheses/prediction_perception_regions/results.md)与[快速目标机制报告](hypotheses/prediction_perception_regions/full_dataset/motion_results.md)。本页保留原方案／协议的历史约定；全量样本和场景划分以[本轮设计](hypotheses/prediction_perception_regions/full_dataset/design.md)为准。\n\n'
        if '全量扩展：**' not in text:
            first,*rest=text.split('\n',1);p.write_text(first+'\n\n'+note+rest[0].lstrip('\n'))
    # Add a scope note without changing historical intervention/training numbers.
    for p in (BASE/'ablations'/'README.md',DISTILLATION/'README.md'):
        text=p.read_text();note=f'> **{UPDATED_DATE} 范围说明：**原始 F／P 区域研究已完成[全量复核]('+os.path.relpath(REPORT/'results.md',p.parent)+')及[快速目标机制分析]('+os.path.relpath(REPORT/'motion_results.md',p.parent)+')。本目录保留其原实验预算和样本范围，没有把这些历史干预／训练结果改称全量实验。\n\n'
        if '范围说明：**' not in text:
            first,*rest=text.split('\n',1);p.write_text(first+'\n\n'+note+(rest[0].lstrip('\n') if rest else ''))
    docs=sorted(set(BASE.rglob('*.md'))|set(DISTILLATION.rglob('*.md')))
    for p in docs:
        assert p.read_text().count('```')%2==0,p
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not link.startswith(('http:','https:','#')):assert (p.parent/link.split('#')[0]).resolve().exists(),(p,link)
    write_json(ROOT/'research_update_audit.json',dict(status='complete',history_preserved=True,full_reports_promoted=True,
        updated=[str(p) for p in BASE.glob('*.md')],validated_markdown_files=len(docs),
        historical_experiment_scope='kept original',report_sha256={str(p):sha256(p) for p in docs}))
    write_json(ROOT/'finish_status.json',dict(status='complete',reports_written=True,full_validation='passed',
        research_updated=True,sources=29049,paired_targets=82047,completed_date=UPDATED_DATE))

if __name__=='__main__':main()
