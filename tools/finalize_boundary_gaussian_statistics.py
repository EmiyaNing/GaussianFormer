"""Audit complete results and export compact Chinese tables and figures."""
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'work_dirs/boundary_gaussian_full'


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def foreground_metrics(g):
    """>=2 foreground labels may coexist with empty; strict no-empty is separate."""
    predicted=int(g['predicted']);boundary=int(g['boundary_any'])
    foreground=int(g['boundary_foreground']);interface=int(g['boundary_fg_empty'])
    pure=boundary-interface
    assert 0<=pure<=foreground<=boundary<=predicted
    return dict(predicted=predicted,foreground_boundary=foreground,
        foreground_boundary_fraction=foreground/predicted if predicted else None,
        strict_no_empty_boundary=pure,
        strict_no_empty_boundary_fraction=pure/predicted if predicted else None,
        foreground_fraction_within_original_boundary=foreground/boundary if boundary else None)


def main():
    from tools.boundary_gaussian_stats.report import generate,pct
    from tools.audit_boundary_gaussian_results import main as audit,sha
    from tools.plot_boundary_gaussian_statistics import main as plots
    report=ROOT/'analysis/boundary_gaussian_full_results.md'
    result=generate(RUN,report)
    assert result['status']=='complete'
    audit();plots()
    with (RUN/'scene_statistics.csv').open() as stream:raw=list(csv.DictReader(stream))
    names={'train':'训练集','val':'验证集','total':'全部'}
    def compact(split,scene,frames,gaussians,values):
        row={'数据集':names[split],'场景':scene,'关键帧数':int(frames),'预测高斯总数':int(gaussians)}
        for domain,label in (('all','全空间'),('camera','camera')):
            row['边界高斯占比_'+label+'_%']=round(float(values[domain]['boundary'])*100,6)
            for key,title in (('wrong_class_fraction','错分类'),('miss_fraction','漏检'),('false_positive_fraction','假阳性')):
                value=values[domain][key]
                row[title+'占覆盖体素比_'+label+'_%']=round(float(value)*100,6) if value is not None else None
        return row
    rows=[]
    for r in raw:
        values={}
        for domain in ('all','camera'):
            values[domain]=dict(boundary=float(r[domain+'_boundary_any_fraction']))
            for key in ('wrong_class_fraction','miss_fraction','false_positive_fraction'):
                value=r[domain+'_union_'+key]
                values[domain][key]=float(value) if value else None
        rows.append(compact(r['split'],r['scene_name'],r['frames'],r['predicted_gaussians'],values))
    assert len(rows)==850
    rows.sort(key=lambda r:({'训练集':0,'验证集':1}[r['数据集']],r['场景']))
    write_csv(RUN/'scene_summary_zh.csv',rows)
    global_rows=[]
    for split,s in result['statistics'].items():
        values={domain:dict(boundary=s['gaussian_counts'][i]['boundary_any_fraction'],
                            **s['errors']['native_'+domain]['union_any']) for i,domain in enumerate(('all','camera'))}
        global_rows.append(compact(split,'全局累计',s['frames'],s['gaussian_counts'][0]['predicted'],values))
    write_csv(RUN/'global_summary_zh.csv',global_rows)

    # Foreground-only refers to boundary membership, not to dropping GT-empty
    # voxels from the error analysis. Thus false-positive counts remain defined.
    with (RUN/'scene_gaussian_details.csv').open() as stream:
        foreground_lookup={(r['split'],r['scene_token'],r['scope']):r for r in csv.DictReader(stream)}
    def foreground_compact(split,scene,frames,gs):
        row={'数据集':names[split],'场景':scene,'关键帧数':int(frames),
             '预测高斯总数':int(gs[0]['predicted'])}
        for i,label in ((0,'全空间'),(1,'camera')):
            g=foreground_metrics(gs[i])
            for field,title in (('foreground_boundary','跨前景类别边界高斯数'),
                    ('strict_no_empty_boundary','严格无empty混合高斯数')):
                row[title+'_'+label]=g[field]
            for field,title in (('foreground_boundary_fraction','跨前景边界占全部高斯比'),
                    ('strict_no_empty_boundary_fraction','严格无empty混合占全部高斯比'),
                    ('foreground_fraction_within_original_boundary','跨前景边界占原边界高斯比')):
                value=g[field]
                row[title+'_'+label+'_%']=round(value*100,6) if value is not None else None
        return row
    foreground_rows=[]
    for r in raw:
        gs=[foreground_lookup[(r['split'],r['scene_token'],scope)] for scope in ('native_all','native_camera')]
        foreground_rows.append(foreground_compact(r['split'],r['scene_name'],r['frames'],gs))
    foreground_rows.sort(key=lambda r:({'训练集':0,'验证集':1}[r['数据集']],r['场景']))
    assert len(foreground_rows)==850
    write_csv(RUN/'scene_foreground_boundary_summary_zh.csv',foreground_rows)
    write_csv(RUN/'global_foreground_boundary_summary_zh.csv',[
        foreground_compact(split,'全局累计',s['frames'],s['gaussian_counts'])
        for split,s in result['statistics'].items()])

    total=result['statistics']['total']
    all_g,camera_g=total['gaussian_counts'][:2]
    camera_error=total['errors']['native_camera']['union_any']
    full_interface_share=all_g['boundary_fg_empty']/all_g['boundary_any']
    overview='\n'.join([
        '### 先读这几个数','',
        f'- 850 个场景合计：全空间边界高斯比例 **{pct(all_g["boundary_any_fraction"])}**，camera mask 内判定的比例 **{pct(camera_g["boundary_any_fraction"])}**。',
        f'- **仅用前景类别判断边界**：全空间为 **{pct(all_g["boundary_foreground_fraction"])}**，camera 为 **{pct(camera_g["boundary_foreground_fraction"])}**；分母同样为全部预测高斯。',
        f'- camera 边界覆盖区域内：错分类 **{pct(camera_error["wrong_class_fraction"])}**、漏检 **{pct(camera_error["miss_fraction"])}**、假阳性 **{pct(camera_error["false_positive_fraction"])}**；三者分母都是边界覆盖体素数。',
        f'- 全空间的“任意两类边界”中，**{full_interface_share*100:.6f}%** 同时跨越前景/empty；跨前景类别边界本身占全部高斯 **{pct(all_g["boundary_foreground_fraction"])}**。这两个概念不能互换。',
        '- 本实验衡量边界覆盖与错误之间的统计关系。是否存在边界并不能单独证明该高斯导致错分类或漏检。','',
        '便于直接阅读的表格：[850 场景中文简表](../work_dirs/boundary_gaussian_full/scene_summary_zh.csv)、[训练/验证/合计中文简表](../work_dirs/boundary_gaussian_full/global_summary_zh.csv)。'
        '仅前景口径另有 [850 场景简表](../work_dirs/boundary_gaussian_full/scene_foreground_boundary_summary_zh.csv) 和 [全局简表](../work_dirs/boundary_gaussian_full/global_foreground_boundary_summary_zh.csv)。'
        '简表列名标明百分比，数值范围为 0–100；原始英文 CSV 则使用 0–1 小数。',''])
    text=report.read_text()
    text=text.replace('python3.10 -m pytest -q tests/test_boundary_gaussian_statistics.py\n',
        'python3.10 -m pytest -q tests/test_boundary_gaussian_statistics.py tests/test_boundary_gaussian_reporting.py tests/test_boundary_foreground_reporting.py\n')
    text=text.replace('python3.10 tools/stat_boundary_gaussians.py --action report\n```',
        'python3.10 tools/stat_boundary_gaussians.py --action report\n'
        'python3.10 tools/audit_boundary_history_images.py\n'
        'python3.10 tools/finalize_boundary_gaussian_statistics.py\n```')
    text=text.replace('- 独立 NumPy 实现核查类别集合、边界标记、覆盖去重、主要贡献归属和三类误差；另用原渲染核公式检查前景 logits。',
        '- 在小规模合成数据、以及每个检查帧的前 64 个高斯上，独立 NumPy 实现核查类别集合、边界标记、覆盖去重、主要贡献归属和三类误差；另用全部预测高斯在 32 个体素处检查前景 logits。')
    text=text.replace('## 1. 每个场景的结果在哪里',overview+'\n## 1. 每个场景的结果在哪里',1)
    foreground_section=['### 仅看前景类别时，比例是多少','',
        '**主要解释：**先从类别集合中排除 empty，再检查是否至少有两种前景 GT 类别。'
        '高斯仍然可以同时覆盖 empty；这不影响它被计为跨前景类别边界。分母保持为全部预测高斯，便于和原始比例比较。','',
        '**更严格的另一种解释：**至少两种前景类别，且统计域内的支持完全不含 empty。'
        '这只是前一种定义的子集。它的数量可以由“任意两类边界数 − 前景/empty 接口数”精确得到。','',
        '| 数据 | GT 范围 | 跨前景类别边界数 | 占全部高斯 | 严格不含 empty 的混合高斯数 | 占全部高斯 |',
        '|---|---|---:|---:|---:|---:|']
    for split,s in result['statistics'].items():
        for i,label in ((0,'全空间'),(1,'camera')):
            g=foreground_metrics(s['gaussian_counts'][i])
            foreground_section.append(f'| {names[split]} | {label} | {g["foreground_boundary"]:,} | {pct(g["foreground_boundary_fraction"])} | '
                f'{g["strict_no_empty_boundary"]:,} | {g["strict_no_empty_boundary_fraction"]*100:.6f}% |')
    foreground_section += ['',
        'camera 的“完全不含 empty”仅指 camera mask 内的支持；它不表示该高斯在 mask 外也不覆盖 empty。'
        '上表两种比例都除以全部预测高斯；“跨前景边界数 / 原始边界数”这一不同分母的比例仅在简表中单列。','']
    foreground_section += [
        '![仅前景类别判定下的场景分布](figures/boundary_gaussian_full/scene_foreground_boundary_distribution.png)','',
        '上图只统计至少包含两种前景 GT 类别的高斯，分母仍为全部预测高斯。'
        '[对应 PDF](figures/boundary_gaussian_full/scene_foreground_boundary_distribution.pdf)。','']
    text=text.replace('## 3. 边界高斯覆盖区域的三类错误',
        '\n'.join(foreground_section)+'\n## 3. 边界高斯覆盖区域的三类错误',1)
    text=text.replace('### 场景之间是否差异很大',
        '![850 个场景的边界高斯比例分布](figures/boundary_gaussian_full/scene_boundary_distribution.png)\n\n'
        '图中采用任意两类、包含 empty 的原始边界定义，每个场景等权；横轴为场景内边界高斯比例，纵轴为累积场景比例。实线为原生支持，虚线为 3σ 支持。'
        '对应 [PDF](figures/boundary_gaussian_full/scene_boundary_distribution.pdf) 可直接导出。\n\n### 场景之间是否差异很大',1)
    text=text.replace('### 用各自 GT 数量作分母',
        '![边界覆盖区域的三类错误占比](figures/boundary_gaussian_full/boundary_union_errors.png)\n\n'
        '图中各错误比例统一除以各自边界覆盖区域的有效体素数；左右子图纵轴范围不同。'
        '对应 [PDF](figures/boundary_gaussian_full/boundary_union_errors.pdf) 可直接导出。\n\n### 用各自 GT 数量作分母',1)
    foreground_errors=['### 跨前景类别边界高斯覆盖区域的错误','',
        '这里使用“至少两种前景类别”的边界定义，对这些高斯的全部有效支持求并集，仍评估原模型最终预测。'
        'empty 只从边界判定的类别计数中排除，不从支持区域的误差统计中删除，因此假阳性仍有意义。'
        '下表三个错误比例的共同分母为该行去重后的覆盖体素数。','',
        '| 数据 | 范围 | 覆盖体素数 | 错分类 / 覆盖体素 | 漏检 / 覆盖体素 | 假阳性 / 覆盖体素 |',
        '|---|---|---:|---:|---:|---:|']
    for split,s in result['statistics'].items():
        for scope,label in (('native_all','全空间'),('native_camera','camera')):
            e=s['errors'][scope]['union_foreground']
            foreground_errors.append(f'| {names[split]} | {label} | {e["voxels"]:,} | {pct(e["wrong_class_fraction"])} | {pct(e["miss_fraction"])} | {pct(e["false_positive_fraction"])} |')
    foreground_errors += ['',
        '以真实前景或真实 empty 为分母的条件错误率，保存在误差明细 CSV 的 `union_foreground` 行中。'
        '“严格不含 empty”的子集此处只报告高斯数量与比例；没有将其他边界组的错误率套用到该子集。','']
    text=text.replace('## 4. 用主要贡献高斯作更严格的归属检查','\n'.join(foreground_errors)+'\n## 4. 用主要贡献高斯作更严格的归属检查',1)
    # Conditioned rates can be unstable when a nonboundary group has very few FG voxels.
    denominators=['\n### 主要贡献分组的条件错误率分母\n',
        '上一表的漏检率与错分类率使用以下真实前景数作为分母，假阳性率使用真实 empty 数。'
        '全空间非边界组的前景支持可能极少，不宜只比较其百分比。\n',
        '| 数据 | 范围 | 归属 | GT 前景数 | GT empty 数 |','|---|---|---|---:|---:|']
    for split,s in result['statistics'].items():
        for scope,label in (('native_all','全空间'),('native_camera','camera')):
            for cohort,title in (('dominant_any','边界'),('dominant_nonboundary','非边界'),('unattributed','未归属')):
                e=s['errors'][scope][cohort]
                denominators.append(f'| {names[split]} | {label} | {title} | {e["gt_foreground"]:,} | {e["gt_empty"]:,} |')
    text=text.replace('## 5. 排除极小高斯尾部后的敏感性检查','\n'.join(denominators)+'\n\n## 5. 排除极小高斯尾部后的敏感性检查',1)
    text+='\n最终完整性审计已通过：34,149 帧与 850 个场景的 token、帧索引、样本 token、'
    text+='逐帧到场景及全局的计数汇总、源代码与权重指纹、CSV 分母均已核对。'
    text+='详见 [完整性审计](../work_dirs/boundary_gaussian_full/integrity_audit.json)。\n'
    assert text.count('$$')%2==0 and text.count('```')%2==0
    report.write_text(text)
    from datetime import datetime
    progress=json.loads((RUN/'progress.json').read_text())
    progress.update(status='complete',completed_frames=34149,completed_scenes=850,
        completed_train_scenes=700,completed_val_scenes=150,updated_at=datetime.now().astimezone().isoformat())
    (RUN/'progress.json').write_text(json.dumps(progress,indent=2))
    scripts=[Path(__file__),ROOT/'tools/audit_boundary_gaussian_results.py',
        ROOT/'tools/plot_boundary_gaussian_statistics.py',ROOT/'tools/audit_boundary_history_images.py',
        ROOT/'tests/test_boundary_gaussian_statistics.py',ROOT/'tests/test_boundary_gaussian_reporting.py',
        ROOT/'tests/test_boundary_foreground_reporting.py']
    hashes={}
    for source in scripts:
        relative=source.relative_to(ROOT);target=RUN/'sources_postprocessing'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        hashes[str(relative)]=sha(source)
    (RUN/'delivery_complete.json').write_text(json.dumps(dict(status='complete',frames=34149,scenes=850,
        source_sha256=hashes,report_sha256=sha(report),integrity_audit_sha256=sha(RUN/'integrity_audit.json')),
        indent=2))
    print(report)


if __name__=='__main__':main()
