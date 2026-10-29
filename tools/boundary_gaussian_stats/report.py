"""Human-readable report and tidy per-scene/global CSVs, from raw counts only."""
import csv
import json
from pathlib import Path
import numpy as np
from . import SCOPES,COHORTS,rates


def read(path):
    return json.loads(Path(path).read_text())


def ratio(a,b):
    return float(a/b) if b else None


def pct(value):
    return '—' if value is None else f'{value*100:.3f}%'


def write_csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarize(records):
    counts=np.array([r['error_counts'] for r in records],np.int64).sum(0)
    gaussians=[]
    for s in range(4):
        g={key:np.asarray([r['gaussian_counts'][s][key] for r in records]).sum(0).tolist()
           for key in records[0]['gaussian_counts'][s]}
        for kind in ('any','foreground','fg_empty'):
            g['boundary_'+kind+'_fraction']=ratio(g['boundary_'+kind],g['predicted'])
        g['supported_fraction']=ratio(g['supported'],g['predicted'])
        g['boundary_any_fraction_supported']=ratio(g['boundary_any'],g['supported'])
        gaussians.append(g)
    return dict(frames=sum(r['frames'] for r in records),scenes=len(records),
        gaussian_counts=gaussians,error_counts=counts.tolist(),
        errors={s:{cohort:rates(counts[i,j]) for j,cohort in enumerate(COHORTS)} for i,s in enumerate(SCOPES)},
        scene_macro_boundary_fraction={s:float(np.mean([r['gaussian_counts'][i]['boundary_any']/r['gaussian_counts'][i]['predicted'] for r in records])) for i,s in enumerate(SCOPES)})


def generate(folder,destination):
    folder=Path(folder);destination=Path(destination)
    protocol=read(folder/'protocol.json')
    splits={s:[read(p) for p in sorted((folder/'scenes'/s).glob('*.json'))] for s in ('train','val')}
    if not all(splits.values()):raise ValueError('Each split needs at least one completed scene to report')
    groups={**splits,'total':splits['train']+splits['val']}
    summary={s:summarize(records) for s,records in groups.items()}
    complete=all(summary[s]['scenes']==protocol['expected'][s]['scenes'] and
                 summary[s]['frames']==protocol['expected'][s]['frames'] for s in ('train','val'))
    # Full completion additionally requires exactly the metadata's scene/frame census.
    availability=read(folder/'availability.json')
    for split,records in splits.items():
        assert len({r['scene_token'] for r in records})==len(records)
        for r in records:
            assert r['frames']==availability[split]['scene_frame_counts'][r['scene_token']]
            assert r['frames']==len(r['indices'])==len(r['records'])
            np.testing.assert_array_equal(np.asarray([f['error_counts'] for f in r['records']]).sum(0),r['error_counts'])
            assert r['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert r['gaussian_counts'][0]['predicted']==76800*r['frames']
        if complete:
            assert {r['scene_token'] for r in records}==set(availability[split]['scene_frame_counts'])
    for value in summary.values():
        counts=np.array(value['error_counts'])
        np.testing.assert_array_equal(counts[:,0],counts[:,4]+counts[:,7]+counts[:,8])
        assert (counts[:,1]>=counts[:,4]).all()

    detailed_g=[];detailed_e=[];main=[]
    for split,records in groups.items():
        for r in records:
            # Scene CSV is not duplicated into the pooled total group.
            if split=='total':break
            row=dict(split=split,scene_name=r['scene_name'],scene_token=r['scene_token'],
                     frames=r['frames'],predicted_gaussians=r['gaussian_counts'][0]['predicted'])
            for i,scope in enumerate(SCOPES):
                g=r['gaussian_counts'][i]
                grow=dict(split=split,scene_name=r['scene_name'],scene_token=r['scene_token'],frames=r['frames'],scope=scope)
                grow.update({k:v for k,v in g.items() if k!='class_count_histogram'})
                for kind in ('any','foreground','fg_empty'):
                    grow['boundary_'+kind+'_fraction']=ratio(g['boundary_'+kind],g['predicted'])
                grow['boundary_any_fraction_supported']=ratio(g['boundary_any'],g['supported'])
                detailed_g.append(grow)
                for j,cohort in enumerate(COHORTS):
                    erow=dict(split=split,scene_name=r['scene_name'],scene_token=r['scene_token'],frames=r['frames'],scope=scope,cohort=cohort)
                    erow.update(rates(r['error_counts'][i][j]));detailed_e.append(erow)
                if i<2:
                    prefix='all' if i==0 else 'camera'
                    for key in ('supported','boundary_any','boundary_foreground','boundary_fg_empty',
                                'boundary_any_fraction','boundary_foreground_fraction','boundary_fg_empty_fraction'):
                        row[prefix+'_'+key]=grow[key]
                    for key,value in rates(r['error_counts'][i][1]).items():
                        row[prefix+'_union_'+key]=value
            main.append(row)
    write_csv(folder/'scene_statistics.csv',main)
    write_csv(folder/'scene_gaussian_details.csv',detailed_g)
    write_csv(folder/'scene_error_details.csv',detailed_e)
    global_g=[];global_e=[]
    for split,value in summary.items():
        for i,scope in enumerate(SCOPES):
            global_g.append(dict(split=split,scope=scope,**{k:v for k,v in value['gaussian_counts'][i].items() if k!='class_count_histogram'}))
            for cohort in COHORTS:global_e.append(dict(split=split,scope=scope,cohort=cohort,**value['errors'][scope][cohort]))
    write_csv(folder/'global_gaussian_statistics.csv',global_g)
    write_csv(folder/'global_error_statistics.csv',global_e)
    result=dict(status='complete' if complete else 'partial',statistics=summary,
                expected=protocol['expected'],checkpoint_sha256=protocol['checkpoint_sha256'])
    (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

    lines=['# 完整 Occ3D 数据集：边界高斯统计结果','',
        f'**状态：{"完整统计已完成" if complete else "部分完成，以下不是全量结论"}。** '
        f'训练集已统计 {summary["train"]["scenes"]}/700 个场景、{summary["train"]["frames"]:,}/28,130 帧；'
        f'验证集已统计 {summary["val"]["scenes"]}/150 个场景、{summary["val"]["frames"]:,}/6,019 帧。','',
        '本报告统计既有权重的预测，没有重新训练，也没有调整 empty bias。主边界定义包含 empty：一个预测高斯的有效渲染支持内出现至少两种 GT 类别。'
        '主误差结果评估这些高斯覆盖区域内的**原模型最终预测**，重叠体素只计一次。','',
        '## 1. 每个场景的结果在哪里','',
        '- [逐场景主结果](../work_dirs/boundary_gaussian_full/scene_statistics.csv)：每个场景一行，包含全空间及 camera mask 下的高斯比例、三类错误计数与比例。',
        '- [逐场景高斯明细](../work_dirs/boundary_gaussian_full/scene_gaussian_details.csv)：另列前景类别边界、前景/empty 接口和 3σ 敏感性结果。',
        '- [逐场景误差明细](../work_dirs/boundary_gaussian_full/scene_error_details.csv)：覆盖区域、主要贡献归属及非边界参照结果。',
        '- [全局 JSON](../work_dirs/boundary_gaussian_full/summary.json)、[全局高斯 CSV](../work_dirs/boundary_gaussian_full/global_gaussian_statistics.csv)、[全局误差 CSV](../work_dirs/boundary_gaussian_full/global_error_statistics.csv)。','',
        'CSV 中比例均为 0–1 小数；本文表格转换为百分比。训练、验证、总计的全局值均先累加计数再相除，**不是对场景百分比取平均**。同一个高斯在不同帧的预测分别计数。','',
        '## 2. 边界高斯占全部预测高斯的多少','',
        '“全空间”使用所有有效 GT；“camera”仅使用 camera mask 内 GT 判断边界。两者分母都是该组帧的全部预测高斯，包括没有覆盖该统计域的高斯。额外追加的固定 empty 高斯不属于模型预测高斯，予以排除。','',
        '| 数据 | GT 范围 | 预测高斯总数 | 任意两类边界 | 跨前景类别边界 | 前景/empty 接口 |',
        '|---|---|---:|---:|---:|---:|']
    names=dict(train='训练',val='验证',total='合计')
    for split,value in summary.items():
        for i,domain in ((0,'全空间'),(1,'camera')):
            g=value['gaussian_counts'][i]
            lines.append(f'| {names[split]} | {domain} | {g["predicted"]:,} | {pct(g["boundary_any_fraction"])} | {pct(g["boundary_foreground_fraction"])} | {pct(g["boundary_fg_empty_fraction"])} |')
    lines += ['', '后两种边界可能同时出现在同一个高斯上，因此它们的比例不能直接相加。各场景内部同样先累计所有关键帧，再计算比例。','',
        '### 场景之间是否差异很大','',
        '| 数据 | GT 范围 | 场景比例 P10 | 中位数 | P90 | 场景等权平均 |',
        '|---|---|---:|---:|---:|---:|']
    for split,records in groups.items():
        for i,domain in ((0,'全空间'),(1,'camera')):
            a=np.array([r['gaussian_counts'][i]['boundary_any']/r['gaussian_counts'][i]['predicted'] for r in records])
            q=np.quantile(a,[.1,.5,.9])
            lines.append(f'| {names[split]} | {domain} | {pct(q[0])} | {pct(q[1])} | {pct(q[2])} | {pct(float(a.mean()))} |')
    lines += ['', '## 3. 边界高斯覆盖区域的三类错误','',
        '三类错误互斥：**错分类**指 GT 和预测都是前景但类别不同；**漏检**指真实前景被预测为 empty；**假阳性**指真实 empty 被预测为前景。'
        '本节三个比例统一除以该行边界覆盖区域的有效体素数，因此可以相加得到总错误占比。','',
        '| 数据 | 统计范围 | 边界覆盖体素数 | 错分类 / 覆盖体素 | 漏检 / 覆盖体素 | 假阳性 / 覆盖体素 | 总错误 / 覆盖体素 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for split,value in summary.items():
        for scope,domain in (('native_all','全空间'),('native_camera','camera')):
            e=value['errors'][scope]['union_any']
            lines.append(f'| {names[split]} | {domain} | {e["voxels"]:,} | {pct(e["wrong_class_fraction"])} | {pct(e["miss_fraction"])} | {pct(e["false_positive_fraction"])} | {pct(e["total_error_fraction"])} |')
    lines += ['', '体素数是逐帧累计的体素观测数，不是跨时间配准后去重的世界坐标体素数。边界高斯覆盖区域里还叠加了其他高斯的贡献；该表描述相关性，不能把每个错误都因果归责于边界高斯。','',
        '### 用各自 GT 数量作分母','',
        '下面错分类率、漏检率除以该区域真实前景数，假阳性率除以该区域真实 empty 数。这能减少前景/empty 数量不平衡对上一张表的影响。','',
        '| 数据 | 统计范围 | 错分类 / GT 前景 | 漏检 / GT 前景 | 假阳性 / GT empty |',
        '|---|---|---:|---:|---:|']
    for split,value in summary.items():
        for scope,domain in (('native_all','全空间'),('native_camera','camera')):
            e=value['errors'][scope]['union_any']
            lines.append(f'| {names[split]} | {domain} | {pct(e["wrong_class_rate_gt_foreground"])} | {pct(e["miss_rate_gt_foreground"])} | {pct(e["false_positive_rate_gt_empty"])} |')
    lines += ['', '## 4. 用主要贡献高斯作更严格的归属检查','',
        '对每个体素，先从原模型的 17 个前景 logit 中找到得分最高的类别，再找到对该类别贡献最大的单个预测高斯。即使最终 empty 获胜，也使用这一前景类别确定归属；选择过程不使用体素 GT。'
        '按这个高斯是否为边界高斯分组，得到互斥的边界归属与非边界归属；没有正贡献的体素另列为未归属。这里仍然评估原模型预测。','',
        '| 数据 | 范围 | 主要贡献归属 | 体素数 | 错分类 / GT 前景 | 漏检 / GT 前景 | 假阳性 / GT empty |',
        '|---|---|---|---:|---:|---:|---:|']
    for split,value in summary.items():
        for scope,domain in (('native_all','全空间'),('native_camera','camera')):
            for cohort,label in (('dominant_any','边界'),('dominant_nonboundary','非边界'),('unattributed','未归属')):
                e=value['errors'][scope][cohort]
                lines.append(f'| {names[split]} | {domain} | {label} | {e["voxels"]:,} | {pct(e["wrong_class_rate_gt_foreground"])} | {pct(e["miss_rate_gt_foreground"])} | {pct(e["false_positive_rate_gt_empty"])} |')
    lines += ['', '两组的类别组成、距离和遮挡程度仍可能不同，因此这些条件错误率的差异也不能单独证明边界高斯导致了错误。','',
        '## 5. 排除极小高斯尾部后的敏感性检查','',
        '主结果严格遵循算子实际整数盒支持，并排除 FP32 权重已经下溢为零的位置。作为补充，本节只保留同一整数盒内 Mahalanobis 距离不超过 3 的位置；其相对核值至少为 $e^{-4.5}$。'
        '这改变了统计支持的定义，但**没有重新渲染或改变模型预测**。','',
        '| 数据 | GT 范围 | 原生支持：边界高斯比例 | 3σ 支持：边界高斯比例 | 原生：覆盖区域错误占比 | 3σ：覆盖区域错误占比 |',
        '|---|---|---:|---:|---:|---:|']
    for split,value in summary.items():
        for i,domain in ((0,'全空间'),(1,'camera')):
            lines.append(f'| {names[split]} | {domain} | {pct(value["gaussian_counts"][i]["boundary_any_fraction"])} | {pct(value["gaussian_counts"][i+2]["boundary_any_fraction"])} | '
                f'{pct(value["errors"][SCOPES[i]]["union_any"]["total_error_fraction"])} | {pct(value["errors"][SCOPES[i+2]]["union_any"]["total_error_fraction"])} |')
    pilot=read(folder/'pilot.json')
    lines += ['', '## 6. 执行与正确性核查','',
        f'- 权重：`{protocol["checkpoint"]}`；SHA-256：`{protocol["checkpoint_sha256"]}`。',
        f'- 配置：`{protocol["config"]}`。用户给出的目录不存在，使用仓库内同名完整数据 8 帧 tiny 配置；权重按 `strict=True` 完整加载。',
        f'- 权重中的 learned empty bias 为 `{pilot["empty_bias"]:.7f}`，按原模型执行。',
        '- 两个集合都采用确定性测试预处理、当前帧加 7 个历史关键帧；不足 7 帧时按原实现补帧。训练集不是训练增强视角下的统计。',
        '- 缓存与原生 48 图流程在训练/验证各两个检查帧上，图像、投影矩阵、预测高斯及渲染 logits 均逐值一致。',
        '- 独立 NumPy 实现核查类别集合、边界标记、覆盖去重、主要贡献归属和三类误差；另用原渲染核公式检查前景 logits。',
        '- 每帧检查：边界归属、非边界归属和未归属计数之和等于全域计数；边界归属体素属于边界覆盖体素。',
        '- 每场景检查帧数、权重指纹与逐帧计数；全量完成时检查场景 token 集与元数据完全一致。','',
        '复现命令：','', '```bash',
        'python3.10 -m pytest -q tests/test_boundary_gaussian_statistics.py',
        'python3.10 tools/stat_boundary_gaussians.py --action pilot',
        'python3.10 tools/stat_boundary_gaussians.py --action run --workers 3',
        'python3.10 tools/stat_boundary_gaussians.py --action report', '```','',
        '统计脚本首次运行会编译独立 CUDA 统计扩展；它不替换模型算子。已完成场景可以恢复使用，未完成场景重新计算。'
        '实验设计、公式和分母定义见 [统计设计文档](boundary_gaussian_full_design.md)。','']
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(lines))
    if complete:
        (folder/'complete.json').write_text(json.dumps(dict(status='complete',frames=summary['total']['frames'],scenes=summary['total']['scenes'],checkpoint_sha256=protocol['checkpoint_sha256']),indent=2))
    return result
