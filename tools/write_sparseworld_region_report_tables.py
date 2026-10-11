#!/usr/bin/env python3
"""Generate human-readable tables and case index from completed regional results."""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from tools.sparseworld_region_utils import ROOT,CLASS_NAMES,STATE_NAMES,write_json,sha256
from tools.analyze_sparseworld_region_pairs import summarize, bootstrap_weights

REPORT=Path('analysis/hypotheses/prediction_perception_regions')


def number(x,precision=2):
    return 'NA' if x is None or not np.isfinite(x) else f'{x:.{precision}f}'


def interval(value,ci):
    return f'{number(value)} [{number(ci[0])}, {number(ci[1])}]'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    s=json.loads((ROOT/'summary.json').read_text())
    e=json.loads((ROOT/'extras.json').read_text())
    geo=json.loads((ROOT/'geometry_diagnosis.json').read_text())
    rules=json.loads((ROOT/'holdout_summary.json').read_text())['rules']
    exp=json.loads((ROOT/'explanation_summary.json').read_text())
    all_=s['splits']['all']
    lines=['# 区域优势：统计附表','',
      '本页所有区间均为场景级配对 bootstrap 的 95% 区间，默认 2000 次。数值单位为百分数；优势／优势增量为百分点。NA 表示分母为零或没有支持。体素比例使用原始计数加权，不把体素当成独立重复实验。','',
      '主文：[结果报告](results.md) · 原始汇总：[summary.json](../../../out/sparseworld_region_comparison/20261008_native512/summary.json)。','',
      '## 1. 四状态完整表','',
      '每一行的四状态都以该行统计域为分母。“仅谁正确”指精确语义类别正确；二值版本另列。','']
    rows=[]
    domains=['全部合法 GT','GT 占据','GT 空闲']
    for d,label in enumerate(domains):
        for h in range(3):
            value=all_['states']['proportions']['value'][h][0][d][0]
            ci=all_['states']['proportions']['ci95'][h][0][d][0]
            rows.append([f'{h+1}s · {label}',* [interval(value[k],ci[k]) for k in [3,2,1,0]],
                         f"{int(all_['states']['total'][h][0][d][0]):,}"])
    lines+=[table(['时距／统计域','双方正确 %','仅 F 正确 %','仅 P 正确 %','双方错误 %','体素数'],rows),'']
    rows=[]
    for h in range(3):
        info=all_['states']
        rows.append([f'{h+1}s',interval(info['advantage']['value'][h][0][1][0],info['advantage']['ci95'][h][0][1][0]),
                     number(info['perception_corrects_forecast']['value'][h][0][1][0]),
                     number(info['forecast_corrects_perception']['value'][h][0][1][0])])
    lines+=['### 净优势与纠错率','',table(['时距','GT 占据 P−F','F 错时 P 能纠正 %','P 错时 F 能纠正 %'],rows),'',
            '### 二值占据版本：GT 占据域','']
    rows=[]
    for h in range(3):
        values=all_['states']['proportions']['value'][h][0][1][1]
        rows.append([f'{h+1}s',*[number(values[k]) for k in [3,2,1,0]]])
    lines+=[table(['时距','双方找到 %','仅 F 找到 %','仅 P 找到 %','双方漏检 %'],rows),'',
            '### 17 个非空类别等权的四状态比例','']
    rows=[]
    for h in range(3):
        values=all_['class_equal_states']['value'][h]
        rows.append([f'{h+1}s',*[number(values[k]) for k in [3,2,1,0]],all_['class_equal_states']['classes_present'][h]])
    lines+=[table(['时距','双方正确 %','仅 F 正确 %','仅 P 正确 %','双方错误 %','有效类别数'],rows),'',
            '## 2. 原版指标与 mask 敏感性','',
            'none 是主协议；camera、lidar 只是在同一输出上改变评测域，不改模型。此表不是区域准确率。','']
    rows=[]
    for m,mask in enumerate(['none','camera','lidar']):
        for h in range(3):
            for model in ['F','P']:
                metric=all_['metrics'][model]
                rows.append([mask,f'{h+1}s',model,interval(metric['mIoU'][h][m],metric['mIoU_ci95'][h][m]),
                             interval(metric['OccIoU'][h][m],metric['OccIoU_ci95'][h][m])])
    lines+=[table(['mask','时距','模型','mIoU %','OccIoU %'],rows),'',
            '## 3. 哪些类别、属性偏向谁？','',
            '下面的 P−F 是条件精确分类率差，不是条件 IoU。速度表只覆盖可运动类别；frustum 表只把已知对应纳入相应组，unknown 单列。','']
    labels={'class':CLASS_NAMES,'distance':['0–10 m','10–20 m','20–30 m','≥30 m'],
      'boundary':['内部','0.4 m 边界带'],'boundary08':['带外','0.8 m 边界带'],
      'camera':['mask=0','mask=1'],'lidar':['mask=0','mask=1'],
      'speed':['<0.5 m/s','0.5–2 m/s','≥2 m/s','未知／不适用'],
      'frustum':['都不在','仅未来窗口','仅历史窗口','双方窗口','未知'],
      'static_frustum':['都不在','仅未来窗口','仅历史窗口','双方窗口','未知'],
      'dynamic_frustum':['都不在','仅未来窗口','仅历史窗口','双方窗口','未知'],
      'semantic_group':['可运动类别','静态类别','others']}
    for attr,title in [('class','逐类'),('distance','距离'),('speed','实际速度'),('camera','camera 标注 mask'),
                       ('boundary','边界'),('frustum','实际输入窗口视锥'),
                       ('static_frustum','静态类视锥'),('dynamic_frustum','可运动类别视锥')]:
        info=all_['attributes'][attr]
        rows=[]
        for g,label in enumerate(labels[attr]):
            if attr=='class' and g==17:
                continue
            rows.append([label,*[interval(info['advantage']['value'][h][g],info['advantage']['ci95'][h][g])
                                  for h in range(3)],f"{int(info['total'][2][g]):,}",info['scenes'][2][g]])
        lines += [f'### {title}','',table(['条件','1s P−F','2s P−F','3s P−F','3s 体素数','3s 场景数'],rows),'']
    cov=np.load(ROOT/'region_counts.npz')['covariates']
    conditional=cov[:,:,:17,:,:,:,:,1,:].sum(axis=(3,4,5,6))
    conditional_summary=summarize(conditional,bootstrap_weights(150,np.random.default_rng(20261008)))
    write_json(ROOT/'conditional_frustum_class.json',dict(status='complete',
       definition='GT occupied and in P-window frustum only, split by GT class',
       results=conditional_summary,caveat='Post-hoc subgroup portrait; not independent confirmation'))
    rows=[]
    for g,label in enumerate(CLASS_NAMES[:17]):
        if conditional_summary['total'][:,g].sum()==0:
            continue
        rows.append([label,*[interval(conditional_summary['advantage']['value'][h,g],conditional_summary['advantage']['ci95'][h,g])
                            for h in range(3)],f"{int(conditional_summary['total'][2,g]):,}",int(conditional_summary['scenes'][2,g])])
    lines += ['### 仅未来窗口视锥组的类别构成与反例','',
              '本表是事后细分，完整呈现有支持的类别，不能作为新的独立确认。3 秒主要由 vegetation 和 manmade 构成。','',
              table(['GT 类别','1s P−F','2s P−F','3s P−F','3s 体素数','3s 场景数'],rows),'']
    lines += ['## 4. 控制类别与距离后的优势增量','',
       '在双方条件都有支持的“GT 类别 × 距离”层内比较，再按共同人群权重标准化。结果是条件之间 P−F 的差，不是某一条件本身的模型差。','']
    rows=[]
    for split in ['all','holdout']:
        for key,label in [('camera1_vs0','camera1 − camera0'),('boundary_vs_interior','边界 − 内部'),
                          ('fast_vs_stopped','快速 − 低速'),('future_only_vs_both','仅未来视锥 − 双方视锥')]:
            d=s['splits'][split]['standardized_contrasts'][key]
            rows.append([split,label,*[interval(d['value'][h],d['ci95'][h]) for h in range(3)]])
    lines+=[table(['样本域','对比','1s 增量','2s 增量','3s 增量'],rows),'',
             '## 5. 发现规则与留出复核','',
             '发现集 341 样本／100 场景，留出集 171 样本／50 场景。规则是在发现集按单属性候选筛选，属于探索性发现；没有进行多重检验调整后的显著性声明。','']
    rows=[]
    for r in rules:
        rows.append([r['label'],f"{r['horizon']}s",r['side'],interval(r['discovery_advantage'],r['discovery_ci95']),
           interval(r['holdout_advantage'],r['holdout_ci95']),r['holdout_scenes'],
           number(r['enrichment'])+'×',number(r['coverage_percent'])+'%',
           '方向复现；支持不足' if r['holdout_scenes']<20 else '方向及区间复现'])
    lines+=[table(['规则','时距','偏向','发现集 P−F','留出 P−F','留出场景','独有正确富集','独有正确覆盖','判断'],rows),'',
        '覆盖率的分母为留出集对应时距“仅该侧正确”的全部 GT 占据体素；富集是规则内该状态概率／总体该状态概率。bus 与快速运动规则有重叠，覆盖率不能直接相加。','',
        '## 6. 共同错误的构成','',
        '此表前五项以 GT 占据且双方语义错误为分母，后两项以 GT 空闲且双方错误为分母。','']
    rows=[]
    error_labels=['共同漏检','F 漏检、P 错类','P 漏检、F 错类','双方占据且错成同类','双方占据且错成不同类','共同误报同类','共同误报不同类']
    err=np.asarray(all_['errors']['counts'])
    for k,label in enumerate(error_labels):
        denom=err[:,:5].sum(1) if k<5 else err[:,5:].sum(1)
        rows.append([label,*[number(err[h,k]/denom[h]*100) for h in range(3)]])
    lines+=[table(['错误子类型','1s 占同域共同错误 %','2s %','3s %'],rows),'',
       '## 7. 实例等权、分块尺度与几何容差','',
       f"匹配实例任务共 {e['instance_motion']['instance_frames']:,} 个，涉及 {e['instance_motion']['distinct_instances']:,} 个不同实例。每个实例任务仅计入三维框内、GT 类别一致且无重叠歧义的体素；同实例的多个预测任务保留在同一场景重采样簇内。",'']
    rows=[]
    for g,label in enumerate(labels['speed'][:3]):
        d=e['instance_motion']
        rows.append([label,*[interval(d['advantage'][h][g],d['advantage_ci95'][h][g]) for h in range(3)]])
    lines+=[table(['实际速度','1s 实例等权 P−F','2s P−F','3s P−F'],rows),'',
            '### 3 秒不同分块尺度','',
            '统计“至少 20 个 GT 占据体素”的块；尺度改变后支持块集合也会改变，不能将下表视为同一分母的严格配对效果。','']
    rows=[]
    d=e['block_sensitivity']
    for i,size in enumerate(d['sizes_m']):
        rows.append([f'{size} m',f"{d['counts'][2][i][0]:,}",*[number(v) for v in d['percent'][2][i]]])
    lines+=[table(['块边长','支持块任务数','F 更准 %','P 更准 %','准确率相同 %','双方错误过半 %'],rows),'',
            '### 20 个留出场景的同类几何容差','',
            '每场景按固定哈希选一个样本，三个时距分别评测。Precision 与 recall 是双向同类别匹配的体素数汇总；允许多对一匹配，不能冒充严格 IoU。','']
    rows=[]
    for h in range(3):
        for m,model in enumerate(['F','P']):
            for t,radius in enumerate(geo['tolerances_m']):
                rows.append([f'{h+1}s',model,f'{radius} m',interval(geo['precision'][h][m][t],geo['precision_ci95'][h][m][t]),
                             interval(geo['recall'][h][m][t],geo['recall_ci95'][h][m][t])])
    lines+=[table(['时距','模型','容差','precision %','recall %'],rows),'',
            '## 8. 解释模型与 GT 辅助选择','',
            '多项 logistic 模型用发现集整场三折选择 C∈{0.1,1,10}，随后冻结。计算使用精确的类别组合计数，避免随机体素抽样带来的近似；频率权重与原体素分布一致。这里没有把这些计数当作独立样本推断系数区间。','']
    rows=[]
    for key,label in [('category_time_baseline_holdout','类别＋时距'),('holdout','完整属性模型')]:
        d=exp[key]
        rows.append([label,number(d['log_loss'],4),number(d['brier'],4)])
    lines += [table(['模型（50 留出场景）','log loss ↓','Brier score ↓'],rows),'',
              '完整模型还包含距离、边界、camera mask、速度、视锥，以及速度×时距、视锥×时距。其条件效应只有点诊断，没有重新拟合 bootstrap 区间；原始系数见 explanation_model.json。','']
    rows=[]
    for h in range(3):
        rows.append([f'{h+1}s',number(all_['states']['oracle_accuracy']['value'][h][0][1][0]),
                  number(e['oracle']['metrics']['mIoU'][h][0],3),number(e['oracle']['metrics']['OccIoU'][h][0],3)])
    lines += [table(['时距','GT 占据 oracle 准确率 %','全域合成 mIoU %','全域合成 OccIoU %'],rows),'',
              '选择规则：至少一方精确分类正确就选正确标签，双方都错则保留 F。它读取 GT 和未来 P，不能部署；准确率互补性不等于 mIoU 最优上界。','',
              '## 9. 数据完整性与限制','']
    speed=np.asarray(all_['attributes']['speed']['total'])
    frustum=np.asarray(all_['attributes']['dynamic_frustum']['total'])
    rows=[]
    for h in range(3):
        rows.append([f'{h+1}s',number(speed[h,3]/speed[h].sum()*100),number(frustum[h,4]/frustum[h].sum()*100)])
    lines += [table(['时距','可运动类别速度未知 %','可运动类别视锥对应未知 %'],rows),'',
       '未知轨迹、框外 GT 体素及重叠框不强行分配实例。静态八类的视锥属性全量构建；others 和不完整运动轨迹单列 unknown。遮挡、天气、转向／加速度、点覆盖机制未测。输入干预未执行。','',
       '本轮规则最多只解释独有正确区域的一部分；不构成关于所有场景、所有权重或蒸馏方法的结论。场景区间不覆盖重新训练种子的方差。','']
    (REPORT/'statistics.md').write_text('\n'.join(lines),encoding='utf8')
    # Flat tables are convenient for downstream checks and plots.
    csvrows=[]
    for split,entry in s['splits'].items():
        for attr,info in entry['attributes'].items():
            for h in range(3):
                for g,label in enumerate(labels[attr]):
                    row=dict(split=split,horizon_seconds=h+1,attribute=attr,group=label,
                         advantage_pp=info['advantage']['value'][h][g],
                         advantage_ci_low=info['advantage']['ci95'][h][g][0],
                         advantage_ci_high=info['advantage']['ci95'][h][g][1],
                         voxels=info['total'][h][g],scenes=info['scenes'][h][g])
                    row.update({name:info['proportions']['value'][h][g][k] for k,name in enumerate(STATE_NAMES)})
                    csvrows.append(row)
    pd.DataFrame(csvrows).to_csv(REPORT/'attribute_statistics.csv',index=False)
    cases=json.loads((ROOT/'selected_cases.json').read_text())['cases']
    gallery=['# 案例图册：同一场景位置的四种状态','',
       '每种状态各三个案例，来自不同场景；从满足条件的 2 m 块中选状态比例最接近组内中位的案例，固定哈希处理同分。F/P 主题还要求该侧在块内有净优势。**案例是混合状态的区域，不代表整块每个体素都属于标题状态。**','',
       '黑框标出用于统计的 2 m 块；周围网格提供背景。图中 GT、F、P 在相同的 GT 上表面高度读取；空闲 XY 列使用固定高度。标题和底部比例使用框内全部高度。相机画面是场景上下文，没有精确叠加三维块，也不是遮挡证据。','',
       '正文的四状态颜色约定：灰色双方正确，蓝色仅 F 正确，橙色仅 P 正确，紫色双方错误。类别颜色在每幅图的图例中说明。','']
    for name in STATE_NAMES:
        subset=[c for c in cases if c['state']==name]
        gallery+=[f"## {subset[0]['state_zh']}",'']
        for case in subset:
            gallery += [f"### {case['horizon']} 秒 · 中心 ({case['x']:.0f}, {case['y']:.0f}) m",'',
               f"该状态占块内 GT 占据体素 **{case['fraction']*100:.2f}%**；块内有 {case['occupied']} 个占据体素。",'',
               f"scene：`{case['scene']}`；sample：`{case['token']}`；block：`{case['block_id']}`。",'',
               f"![{case['state_zh']}案例]({case['figure']})",'']
    gallery += ['## 与最初假设不一致的结果','',
      '“仅进入未来窗口视锥”这一组反而偏向 F；同组的 terrain、sidewalk、truck、bus 则仍偏向 P。不能仅凭“有未来图像”或“进入视野”选择监督。总体规律及组内反例的完整数值见统计附表；图册仅展示固定选例，不能用图片替代计数。','']
    (REPORT/'cases.md').write_text('\n'.join(gallery),encoding='utf8')
    write_json(ROOT/'run_completion.json',dict(status='complete',stages=['R0','R1','R2','R3'],
       additional=['instance_equal_weight','block_size_sensitivity','20_scene_geometry_tolerance'],
       not_executed=['R4 input/renderer interventions','training','novel scene evaluation'],
       code_sha256={str(p):sha256(p) for p in sorted(Path('tools').glob('*sparseworld_region*.py'))},
       report_sha256={str(p):sha256(p) for p in REPORT.glob('*.md')},
       regression_converged=json.loads((ROOT/'explanation_model.json').read_text())['n_iter'][0]<5000))
    print('[report tables]',REPORT,flush=True)


if __name__=='__main__':
    main()
