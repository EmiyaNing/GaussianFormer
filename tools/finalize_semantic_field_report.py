"""Add interpretation and independently recorded audits to the frozen base report."""
from pathlib import Path
import json,sys,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.semantic_field_experiment_common import RUN,sha,write

def main():
    result=json.loads((RUN/'summary.json').read_text());assert result['status']=='complete'
    m=result['metrics'];b=m['baseline'];g=m['geometry'];s=m['semantic']
    geometry=json.loads((RUN/'geometry_change_audit.json').read_text())
    gradient=json.loads((RUN/'gradient_balance_audit.json').read_text())
    balance=json.loads((RUN/'coverage_class_balance_audit.json').read_text())
    target=ROOT/'analysis/semantic_field_mini_results.md';text=target.read_text()
    if '<!-- independent-interpretation-v1 -->' in text:return
    lead='''<!-- independent-interpretation-v1 -->
**结论：这轮局部同类场监督没有达到预期，尚不支持它能够解决query负载压力。** 几何场G只有很小的覆盖收益；局部同类版S的mIoU基本持平，核心覆盖和固定高负载区域的召回反而下降。

![相对同预算基线的完整验证变化](figures/semantic_field_mini_comparison.png)

图中单位为百分点，误差线仅表示mIoU和几何覆盖的场景配对95%区间；其他面板为点估计。完整数值见下文。

'''
    position=text.index('**几何场版相对同预算基线')
    text=text[:position]+lead+text[position:]
    # Class-balanced coverage supplements the volume-weighted primary results.
    addition=['### 按类别看，结论是否被道路体素主导','',
        '| 模型 | 17类几何覆盖率的均值 | 去除driveable surface后的几何覆盖 | 核心覆盖下降的类别数 |',
        '| --- | ---: | ---: | ---: |']
    rows=list(csv.DictReader((RUN/'aggregate/per_class.csv').open()))
    original={int(x['class_id']):x for x in rows if x['arm']=='baseline'}
    for name,label in [('baseline','B'),('geometry','G'),('semantic','S')]:
        count=sum(float(x['core1'])<float(original[int(x['class_id'])]['core1']) for x in rows if x['arm']==name)
        v=balance[name];addition.append(f"| {label} | {v['macro_core1']:.3f}% | {v['nonroad_core1']:.3f}% | {count}/17 |")
    addition +=['', 'S在15个类别上的1σ覆盖下降，去除道路类别后仍下降，不能解释为只牺牲道路换取小类别。G的体素加权覆盖略升，但类别均值略降，也不是各类普遍改善。', '',
        'mIoU与核心覆盖不等价。例如S的motorcycle IoU增加0.364个百分点，但该类核心覆盖下降0.424个百分点；pedestrian分别为+0.116和−0.500个百分点。不能把这些类别的IoU收益自动归因于几何覆盖改善。完整17类结果保存在逐类CSV。','']
    marker='## 3. Query负载压力'
    text=text.replace(marker,'\n'.join(addition)+'\n'+marker)
    more=['### 接触更多，不等于覆盖更广','',
        '| 模型 | 每GT对应的query–体素接触对数 | 每个已覆盖GT接触的query数 |',
        '| --- | ---: | ---: |']
    for name,label in [('baseline','B'),('geometry','G'),('semantic','S')]:
        v=m[name];pairs=v['query_core_pairs_per_GT'];multiplicity=pairs/(v['geo1']/100)
        more.append(f'| {label} | {pairs:.4f} | {multiplicity:.4f} |')
    more+=['', f"S的低覆盖query比例比B少{b['core_le20']-s['core_le20']:.3f}个百分点，但核心覆盖并集下降{b['geo1']-s['geo1']:.3f}个百分点，最近归属top20%份额上升{s['top20_nearest_share']-b['top20_nearest_share']:.3f}个百分点。更多query轻微接触前景，仍可能同时出现更集中的归属和更多重复接触；这不能记为有效的负载重分配。",'']
    text=text.replace('## 4. 是否通过退缩或改变语义幅度降损','\n'.join(more)+'\n## 4. 是否通过退缩或改变语义幅度降损')
    # Actual completed training time, not isolated kernel timing.
    timing=['### 实际训练开销','', '| 组别 | 4096帧训练耗时 | 峰值已分配CUDA显存 |', '| --- | ---: | ---: |']
    for name,label in [('baseline','B'),('geometry','G'),('semantic','S')]:
        v=json.loads((RUN/'train'/name/'complete.json').read_text())
        timing.append(f"| {label} | {v['seconds']/60:.2f}分钟 | {v['peak_cuda_GB']:.3f}GB |")
    timing+=['', 'GT缓存在CPU上提前准备，4096帧合计约19.9分钟，未计入上述训练循环耗时；G、S复用同一缓存。该机器上的总体训练吞吐接近，不应据此将场项本身的计算成本视为零。推理没有新增模块或渲染步骤，本轮未重新测FPS。','']
    text=text.replace('## 6. 本次实验能回答到哪里','\n'.join(timing)+'\n## 6. 本次实验能回答到哪里')
    start=text.index('## 6. 本次实验能回答到哪里');end=text.index('## 7. 可追溯产物',start)
    diagnosis=['## 6. 结果说明了什么，原因还差哪些证据','',
        '**直接成立的结论是：S减少了一部分误占用，却没有补足真实前景的核心覆盖。** 相对B，FP总数减少165,984，FN总数增加66,477，mIoU差值的95%区间包含零。固定初始高负载区域的1σ覆盖从70.065%降至69.717%，二值召回从74.681%降至74.603%。这组结果与“解决query压力”的预期方向相反。','',
        '### 6.1 不是代码未求导，也不是绝大多数同类目标都回退了','',
        f"S的训练帧平均可靠同类对应比例为62.152%，同类候选回退比例仅5.922%。平均每帧约有{614400*.28549:,.0f}个场内样本位于有效监督区域。距离与梯度检查均通过，因此不能将这次结果简单归因于场项完全未生效，或只剩极少监督样本。",'',
        '补充检查使用预定的4个训练帧、同一测试输入；没有据此选择权重。G、S相对B的同一输出槽中心位移均值分别约1.004、1.127个体素，三个尺度轴的绝对变化均值分别约0.00480、0.00590个体素。中心确实发生了变化，但变化没有转化为更多前景覆盖。这里比较的是输出槽，不能解释为固定物体身份的追踪。','',
        '### 6.2 这套标量目标允许“总损失下降，但覆盖项变差”','',
        '在同4个固定输入上，用完全相同的场采样重新计算S目标。各模型按同一规则、各自当前几何重新建立局部对应；这不是固定匹配后的单步因果实验：','',
        '| 权重来源 | 同类覆盖项 | 局部同类拟合项 | 不确定泄漏项 | 三项之和 | 有效样本比例 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,label in [('baseline','B权重'),('semantic','S权重')]:
        rr=[x['diagnostics'] for x in geometry['paired_field_diagnostics'] if x['arm']==name and x['mode']=='semantic']
        means={k:float(np.mean([x[k] for x in rr])) for k in ('cover','fit','spill','raw','known_sample_fraction')}
        diagnosis.append(f"| {label} | {means['cover']:.5f} | {means['fit']:.5f} | {means['spill']:.5f} | {means['raw']:.5f} | {means['known_sample_fraction']*100:.3f}% |")
    diagnosis+=['', 'S的拟合和泄漏代价下降，足以抵消覆盖代价的增加。这个局部检查直接说明：优化三项的加权和，并不保证覆盖也改善。它与完整验证的覆盖下降方向一致，但4帧结果不能证明全部场景都由同一种训练路径导致。','',
        f"同时，S的中心在有效empty中的比例从{b['center_camera_empty']:.3f}%降至{s['center_camera_empty']:.3f}%，在有效前景中的比例也从{b['center_camera_fg']:.3f}%降至{s['center_camera_fg']:.3f}%；未知区域比例从{b['center_unknown']:.3f}%升至{s['center_unknown']:.3f}%。所以‘空中心减少’不能直接当作‘中心移入前景’。这些是净分布变化，尚不是逐GS跨区域迁移的因果证明。",'',
        '### 6.3 梯度强度和尺度饱和是限制，但尚不能当作主导成因','',
        '在4个预定训练帧、各自最终权重处，检查新增场项和原有总目标对最终几何回归分支参数的梯度：','',
        '| 检查对象 | 场梯度范数/原总梯度范数 | 场与原中心CD梯度的平均余弦 | 场梯度非零的原始尺度轴比例 |',
        '| --- | ---: | ---: | ---: |']
    for arm,mode,label in [('geometry','geometry','G权重＋G场项'),('semantic','semantic','S权重＋S场项')]:
        rr=[x for x in gradient['records'] if x['arm']==arm and x['mode']==mode]
        mean=lambda key:float(np.mean([x[key] for x in rr]))
        diagnosis.append(f"| {label} | {100*mean('field_over_original'):.3f}% | {mean('cosine_with_CD'):.3f} | {100*mean('field_raw_scale_nonzero_fraction'):.3f}% |")
    diagnosis+=['', '新增项的局部梯度相对较弱，原始尺度输出也有较多饱和轴；但场与CD的平均余弦为正，不能把失败直接写成“CD与场项普遍反向冲突”。这些是少量固定训练帧上的未预条件梯度，不是AdamW整条轨迹的因果分解，也没有证明加大权重或修改尺度参数化就会有效。','',
        '### 6.4 对后续方法设计的含义','',
        '本轮不支持直接扩大这一版局部同类loss的训练规模。若继续场监督，应先解决目标之间的取舍：同类拟合和泄漏的改善不能以真实前景覆盖退化为代价；局部片段选择也需要检查是否增加重复支持。query重定位则仍值得单独验证，但应直接以固定高负载区域的覆盖缺口和召回为判据，不能以“这版loss已经解决分配”为前提。','',
        '以上结论只针对当前GT主导的单连通片段对应、2米邻域、固定权重和1epoch微调。原有CD完整保留，已覆盖点的原始CD吸引仍存在；本轮没有完整检验降低或替换CD后的方案，也没有排除一般软类别对应的价值。','']
    text=text[:start]+'\n'.join(diagnosis)+'\n'+text[end:]
    text+='''
补充产物：[几何变化与固定输入场目标](../work_dirs/semantic_field_mini_v1/geometry_change_audit.json)、[梯度强度检查](../work_dirs/semantic_field_mini_v1/gradient_balance_audit.json)、[类别均衡覆盖检查](../work_dirs/semantic_field_mini_v1/coverage_class_balance_audit.json)、[PDF图](figures/semantic_field_mini_comparison.pdf)。

补充检查的mini训练帧测试使用原生48图像入口，避免稀疏mini索引缺少历史帧的问题；主实验的完整验证与专用训练器不受影响。可读性解释由`tools/finalize_semantic_field_report.py`在冻结的基础汇总后生成；补充检查不改变训练配置或权重。
'''
    target.write_text(text)
    sources=[ROOT/'tools/finalize_semantic_field_report.py',ROOT/'tools/plot_semantic_field_mini.py',
        ROOT/'tools/audit_semantic_field_geometry_change.py',ROOT/'tools/audit_semantic_field_gradient_balance.py']
    write(RUN/'supplemental_provenance.json',dict(status='complete',sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        artifacts={p.name:sha(p) for p in [RUN/'geometry_change_audit.json',RUN/'gradient_balance_audit.json',
            RUN/'coverage_class_balance_audit.json',ROOT/'analysis/figures/semantic_field_mini_comparison.png',target]}))

if __name__=='__main__':main()
