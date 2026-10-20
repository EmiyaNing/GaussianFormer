"""Refresh the contribution-1 report from completed artifacts and queue state."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.center_support_common import write_json
from tools.summarize_center_support import summarize, interval
from tools.summarize_mini_occ3d import parse_log

RUN = ROOT / 'work_dirs/center_support_contribution1'
REPORT = ROOT / 'plans/sparse_render_offical_9_28/contribution1_diagnostic_results.md'
MARKER = '\n## 5. 真实实验执行记录（自动更新）'
AUTO_START = '<!-- CONTRIBUTION1_RESULTS_AUTO -->'
AUTO_END = '<!-- CONTRIBUTION1_RESULTS_END -->'
CONTROLS = ('a0_original', 'a1_align_final', 'a1_align_all', 'a2_geometry_quarter',
            'a2_classification_quarter', 'a3_no_final_geometry',
            'a4_no_final_classification', 'a5_voxel_only_final', 'low_budget_19200')


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def number(value, places=4):
    return f'{value:.{places}f}' if value is not None else '—'


def estimate(item):
    if item is None:
        return '—'
    bounds = item['ci95']
    return number(item['mean'], 6) + (f' [{bounds[0]:.6f}, {bounds[1]:.6f}]' if bounds else '（无 CI）')


def local_verdict(item):
    own = item['self_loss']['ci95']
    field = item['same_batch_voxel']['ci95']
    if not own or own[1] >= 0:
        return '自身目标下降未确认'
    if field and field[0] > 0:
        return '自身变好，场变差'
    if field and field[1] < 0:
        return '自身与场同时变好'
    return '场变化方向未确定'


def render_overview(diagnostics, training):
    """Keep scene confidence intervals separate from single-seed training deltas."""
    selected = []
    for name in ('epoch1', 'epoch5', 'epoch10'):
        for branch in ('camera/geometry_only', 'camera/classification_only'):
            item = next((b for b in diagnostics.get(name, {}).get('branches', [])
                         if b['branch'] == branch and b['radius'] == 1e-4), None)
            if not item or not item['same_batch_voxel']['ci95']:
                return False
            selected.append(item['same_batch_voxel'])
    evaluations = [training[name]['final_evaluation'] for name in CONTROLS[:8]]
    if not all(evaluations):
        return False
    directory = REPORT.parent / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'contribution1_diagnostic_overview'
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() +
                                 json.dumps([selected, evaluations], sort_keys=True).encode()).hexdigest()
    cache = target.with_suffix('.sha256')
    if cache.exists() and cache.read_text() == fingerprint and all(target.with_suffix(ext).exists() for ext in ('.png', '.svg')):
        return True
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei')
             if name in available_fonts] + ['DejaVu Sans']
    with plt.rc_context({'font.family': fonts,
                         'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'path'}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(13.6, 6.0),
                                        gridspec_kw={'width_ratios': [1, 1.25]})
        fig.subplots_adjust(left=.07, right=.99, bottom=.24, top=.76, wspace=.38)
        colors = ('#176a90', '#be5c2a')
        for offset, label, color in [(0, '只更新中心几何', colors[0]), (1, '只更新最近邻分类', colors[1])]:
            values = selected[offset::2]
            means = np.array([v['mean'] for v in values])
            bounds = np.array([v['ci95'] for v in values]).T
            y = np.arange(3) + (-.12 if offset == 0 else .12)
            left.errorbar(means, y, xerr=np.vstack((means-bounds[0], bounds[1]-means)),
                          fmt='o', capsize=4, color=color, label=label, markersize=6)
        left.set(yticks=range(3), yticklabels=['epoch 1', 'epoch 5', 'epoch 10'],
                 xlabel='场损失变化 Δ（CE + Lovasz）', title='局部干预：原目标变好后，场是否变差？')
        left.invert_yaxis()
        left.axvline(0, color='#5e6670', linewidth=1)
        left.grid(axis='x', alpha=.2)
        left.set_title(left.get_title(), y=1.20, pad=0)
        left.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=10)
        left.text(.02, -.30, 'ρ = 10⁻⁴；每项 32 对场景；误差线为场景 bootstrap 95% CI\n'
                  '左侧：场改善  ←  0  →  右侧：场变差', transform=left.transAxes, fontsize=9)
        baseline = evaluations[1]
        names = ['原监督域', '所有阶段对齐', '几何权重 1/4', '分类权重 1/4',
                 '移除最终几何', '移除最终分类', '最终仅体素监督']
        variants = [v for i, v in enumerate(evaluations) if i != 1]
        y = np.arange(len(names))
        for offset, (key, container, label) in enumerate([('mIoU', 'camera_metrics', 'camera mIoU'),
                                                        ('RayIoU', 'ray_metrics', '官方 RayIoU')]):
            delta = [v[container][key] - baseline[container][key] for v in variants]
            right.barh(y + (-.16 if offset == 0 else .16), delta, height=.29,
                       label=label, color=colors[offset])
        right.set(yticks=y, yticklabels=names, xlabel='相对对齐基线的变化（百分点）',
                  title='独立训练：降权与删除有相同收益吗？')
        right.invert_yaxis()
        right.axvline(0, color='#5e6670', linewidth=1)
        right.grid(axis='x', alpha=.2)
        right.set_title(right.get_title(), y=1.20, pad=0)
        right.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=10)
        right.text(.02, -.30, 'mini 划分；5 epoch；seed 928；各组独立训练\n'
                   '左侧：指标下降  ←  0  →  右侧：指标提升；无多 seed 误差线',
                   transform=right.transAxes, fontsize=9)
        for ax in (left, right):
            ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(target.with_suffix('.png'), dpi=180)
        fig.savefig(target.with_suffix('.svg'))
        plt.close(fig)
    cache.write_text(fingerprint)
    return True


def reader_summary(state, diagnostics, training, sensitivity, full_evaluations, figure=False):
    completed = sum(task['status'] == 'complete' for task in state.get('tasks', []))
    lines = ['## 1. 先看本轮回答了什么', '',
        f'**首轮任务：{completed}/{len(state.get("tasks", []))} 项完成。** '
        f'产物最后更新于 {state.get("updated_at", "—")}。CPU 玩具另行完成，不计入这 40 项 GPU 任务。', '',
        '| 研究问题 | 当前证据 | 应如何理解 |',
        '| --- | --- | --- |']
    for label, branch in [('几何监督是否损害场拟合（QG）', 'camera/geometry_only'),
                          ('分类监督是否损害场拟合（QS）', 'camera/classification_only')]:
        observed = [item for name in ('epoch1', 'epoch5', 'epoch10')
                    for item in diagnostics.get(name, {}).get('branches', [])
                    if item['branch'] == branch and item['radius'] <= 1e-4]
        harmful = [item for item in observed if local_verdict(item) == '自身变好，场变差']
        helpful = [item for item in observed if local_verdict(item) == '自身与场同时变好']
        if not observed:
            evidence, meaning = '待完成', '等待真实模型证据'
        else:
            evidence = f'{len(observed)} 个 checkpoint/小步长条件中，{len(harmful)} 个支持局部冲突，{len(helpful)} 个支持共同改善'
            meaning = ('本轮未观察到稳定的几何有害效应' if not harmful else '冲突具有条件性，需结合训练消融') if branch.endswith('geometry_only') else '逐 checkpoint 和步长判断；局部冲突不等于删除分类目标有益'
        lines.append(f'| {label} | {evidence} | {meaning} |')
    lines += ['', '这里的“条件”指一个 checkpoint 与一个步长的组合；不是独立训练次数。每个条件使用 32 对场景，置信区间来自场景重采样，未做多重比较校正。', '',
        '**结论边界：** 玩具实验支持特定条件下的机制存在；真实模型需要分别判断几何与分类。标签不同、梯度方向不同或一次 mIoU 提升，都不足以证明普遍的架构缺陷。', '']
    if figure:
        lines += ['![左图展示局部干预的场损失变化及场景置信区间；右图展示训练消融相对基线的 mIoU 与 RayIoU 变化。](figures/contribution1_diagnostic_overview.png)', '',
                  '[下载矢量图 SVG](figures/contribution1_diagnostic_overview.svg)。左图越靠左越好，右图越靠右越好；两图的误差与指标含义不同。', '']
    lines += [
        '## 2. 真实模型：向原目标走一步，场拟合会怎样？', '',
        '每次只更新指定参数，从同一个 checkpoint 重新出发。先检查原目标是否下降，再看渲染场目标的变化。'
        '场目标是训练配置中带实际权重的 **CE + Lovasz**；变化量 Δ < 0 表示改善，Δ > 0 表示变差。', '',
        '### 2.1 完整数据模型：三个训练阶段', '',
        '下表统一展示相对步长 **ρ = 10⁻⁴**，使用相机可见区域对齐监督；完整步长 10⁻⁵/10⁻⁴/10⁻³ 和独立场景结果见 §5。', '',
        '| checkpoint | 只优化哪一项 | 自身损失 Δ ↓ | 场损失 Δ [95% CI] ↓ | 读法 |',
        '| --- | --- | ---: | --- | --- |']
    for name in ('epoch1', 'epoch5', 'epoch10'):
        for branch, label in [('camera/geometry_only', '中心几何'), ('camera/classification_only', '最近邻分类')]:
            item = next((b for b in diagnostics.get(name, {}).get('branches', []) if b['branch'] == branch and b['radius'] == 1e-4), None)
            if item:
                lines.append(f'| {name} | {label} | {item["self_loss"]["mean"]:+.6f} | {estimate(item["same_batch_voxel"])} | {local_verdict(item)} |')
    lines += ['', '读法：自身损失下降且场损失变化的整个置信区间高于 0，才支持该条件下的局部冲突。'
              '区间跨 0 表示场变化方向尚不能确定。结论只适用于表中条件，不外推到所有模型或训练阶段。', '',
        '### 2.2 相同训练协议下，分类冲突是否随 Gaussian 数量变化？', '',
        '以下模型都独立训练于同一 mini 划分；只比较最终 Gaussian 数 76,800 与 19,200。仍取 ρ = 10⁻⁴，每项 8 对场景。', '',
        '| 最终 Gaussian 数 | checkpoint | 分类更新后的场损失 Δ [95% CI] ↓ | 读法 |',
        '| ---: | --- | --- | --- |']
    for budget, count in [('high', '76,800'), ('low', '19,200')]:
        for epoch in (1, 5):
            item = next((b for b in diagnostics.get(f'mini_{budget}_epoch{epoch}', {}).get('branches', [])
                         if b['branch'] == 'camera/classification_only' and b['radius'] == 1e-4), None)
            if item:
                lines.append(f'| {count} | epoch{epoch} | {estimate(item["same_batch_voxel"])} | {local_verdict(item)} |')
    lines += ['', '该对照用于定位预算相关条件。8 对场景和单一训练 seed 仍不足以证明冲突随预算单调变化。', '',
        '## 3. 重新训练：局部冲突是否转化为最终收益？', '',
        'mini 协议统一为 **训练 4,096 帧 / 验证 1,024 帧、5 epoch、seed 928**。'
        '下表比较共同的最后 epoch；所有差值均相对“最终阶段对齐相机域”基线，单位为**百分点**。'
        'mIoU 使用 camera mask，RayIoU 使用官方无 camera mask 的射线协议，二者不能互换。', '',
        '| 训练设置 | 配置简称 | mIoU ↑ | Δ mIoU | RayIoU ↑ | Δ RayIoU |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    labels = ['原监督域', '最终阶段对齐相机域（基线）', '所有阶段对齐相机域', '最终几何权重降至 1/4',
              '最终分类权重降至 1/4', '移除最终几何项', '移除最终分类项', '最终阶段仅体素监督']
    baseline = training['a1_align_final']['final_evaluation']
    for name, label in zip(CONTROLS[:8], labels):
        data = training[name]['final_evaluation']
        if not data or not baseline:
            lines.append(f'| {label} | `{name}` | 待完成 | — | — | — |')
            continue
        miou, ray = data['camera_metrics']['mIoU'], data['ray_metrics']['RayIoU']
        lines.append(f'| {label} | `{name}` | {miou:.2f} | {miou-baseline["camera_metrics"]["mIoU"]:+.2f} | '
                     f'{ray:.2f} | {ray-baseline["ray_metrics"]["RayIoU"]:+.2f} |')
    lines += ['', '**如何读这组结果：** 分类降权与分类删除需要分开判断；mIoU 与 RayIoU 也需要同时看。']
    quarter = training['a2_classification_quarter']['final_evaluation']
    removed = training['a4_no_final_classification']['final_evaluation']
    voxel = training['a5_voxel_only_final']['final_evaluation']
    if baseline and quarter and removed:
        if all(quarter[container][metric] > baseline[container][metric] > removed[container][metric]
               for container, metric in [('camera_metrics', 'mIoU'), ('ray_metrics', 'RayIoU')]):
            lines.append('本轮分类降权在两项指标上均高于基线，而直接删除分类项在两项指标上均低于基线。')
    if baseline and voxel:
        if voxel['camera_metrics']['mIoU'] > baseline['camera_metrics']['mIoU'] and voxel['ray_metrics']['RayIoU'] < baseline['ray_metrics']['RayIoU']:
            lines.append('最终仅用体素监督时，mIoU 略高于基线、RayIoU 较低，不能概括成所有指标追平。')
    lines += ['以上是单 seed 五轮筛选结果，尚无训练随机性区间。', '',
        '### 3.1 Gaussian 数量对精度与训练成本的影响', '',
        '| 最终 Gaussian 数 | mIoU ↑ | RayIoU ↑ | 训练及每轮验证耗时（小时） | CUDA 峰值 allocated（MiB） |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for name, count in [('a1_align_final', '76,800'), ('low_budget_19200', '19,200')]:
        data, cost = training[name]['final_evaluation'], training[name]['training_runtime']
        if data and cost and cost.get('status') == 'complete':
            lines.append(f'| {count} | {data["camera_metrics"]["mIoU"]:.2f} | {data["ray_metrics"]["RayIoU"]:.2f} | '
                         f'{cost["training_and_validation_seconds"]/3600:.2f} | {cost["peak_cuda_allocated_mib"]:.1f} |')
    lines += ['', '两组 query 数均为 600，仅最终 children 从 128 改为 32，早期 decoder 保留。'
              '训练耗时和显存为实际测量；不能按 Gaussian 数量的比例推断整体加速。', '',
        '## 4. 归因可靠性与完整验证集结果', '',
        '### 4.1 最近邻标签与贡献区域多数标签，到底有多少不一致？', '',
        '下表在**相同 8 个 source 场景**上比较两种统计方式。百分比只针对达到支持门槛的 Gaussian。', '',
        '| checkpoint | 抽样 1,024 个 occupied voxel | 精确累积全部可见 occupied voxel |',
        '| --- | ---: | ---: |']
    for name in ('epoch1', 'epoch5', 'epoch10'):
        sampled = sensitivity.get(f'{name}/1024', {}).get('mismatch')
        exact = sensitivity.get(f'{name}/全部 occupied', {}).get('mismatch')
        if sampled and exact:
            lines.append(f'| {name} | {sampled["mean"]*100:.2f}% | {exact["mean"]*100:.2f}% |')
    lines += ['', '增加样本量也会改变达到支持门槛的 Gaussian 集合，因此两列不是同一固定总体的两个误差估计。'
              '精确累积仍不代表全部 Gaussian 的错配率；抽样比例也不能直接作为整体错配比例。'
              '错配组与一致组的渲染误差差异属于关联，不能据此单独推断因果。', '',
        '### 4.2 完整数据模型的官方评估', '',
        '| checkpoint | 验证帧数 | camera mIoU ↑ | 二值占用 IoU ↑ | 官方 RayIoU ↑ |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in ('epoch1', 'epoch5', 'epoch10'):
        data = full_evaluations.get(name)
        if data:
            lines.append(f'| {name} | {data["samples"]:,} | {data["camera_metrics"]["mIoU"]:.2f} | '
                         f'{data["camera_metrics"]["IoU"]:.2f} | {data["ray_metrics"]["RayIoU"]:.2f} |')
    lines += ['', '这些是完整数据模型随训练推进的成绩，用于定位诊断对象。'
              '它们不与 §3 的 mini 模型构成训练消融，也不证明删除某个目标带来的收益。', '']
    return lines


def refresh():
    state = read(RUN / 'execution_status.json') or {'status': 'preparing', 'tasks': []}
    lines = [MARKER.lstrip(), '',
        '<details>', '<summary>展开完整数值表、全部步长、训练成本和 40 项任务清单</summary>', '',
        '本节直接读取实际产物。正文选取关键对照，本节保留完整数值；玩具结果见 §6，修复和审计记录见 §7。', '',
        f'队列状态：`{state["status"]}`；当前任务：`{state.get("current_task") or "无"}`；最后更新时间：{state.get("updated_at", "—")}。',
        '运行环境：RTX 4090，宿主机 NVIDIA 595.99.02，PyTorch 2.12.0+cu126；GPU 任务在沙箱外执行。', '',
        '任务状态与命令：[execution_status.json](../../work_dirs/center_support_contribution1/execution_status.json)。',
        '队列日志：[execution_console.log](../../work_dirs/center_support_contribution1/execution_console.log)。', '',
        '### checkpoint 诊断', '',
        '| checkpoint | 场景对 | 18 类最大绝对误差 | supported Gaussian 均数 | supported 最近邻/多数错配率 |',
        '| --- | ---: | ---: | ---: | ---: |']
    diagnostics, contribution_effects = {}, {}
    for name in ('epoch1', 'epoch5', 'epoch10', 'mini_high_epoch1', 'mini_high_epoch5', 'mini_low_epoch1', 'mini_low_epoch5'):
        options = [RUN / 'real' / f'{name}_intervene_32.json', RUN / 'real' / f'{name}_intervene.json']
        data = next((d for path in options if (d := read(path)) and d.get('status') == 'complete'), None)
        if data is None:
            lines.append(f'| {name} | 待完成 | — | — | — |')
            continue
        stats = [r['contribution'] for r in data['pairs']]
        diagnostics[name] = summarize(data)
        contribution_effects[name] = {
            group: interval([r[group] for r in stats if r.get(group) is not None])
            for group in ('mismatch_responsibility_weighted_error', 'agreement_responsibility_weighted_error')}
        write_json(RUN / 'real' / f'{name}_current_summary.json', diagnostics[name])
        n = len(stats)
        mismatch = [r['majority_mismatch_fraction'] for r in stats if r.get('majority_mismatch_fraction') is not None]
        lines.append(f'| {name} | {n} | {max(r["renderer_max_abs_error"] for r in stats):.3g} | '
                     f'{sum(r["supported_gaussians"] for r in stats)/n:.1f} | {number(sum(mismatch)/len(mismatch) if mismatch else None)} |')
    lines += ['', '错配率只针对达到抽样质量门槛的 Gaussian；1024 个 occupied 抽样可能只覆盖很小的 primitive 子集。不得解释为全部 Gaussian 的错配率。', '',
              '| checkpoint | 错配组责任加权错误均数 [95% CI] | 一致组责任加权错误均数 [95% CI] |',
              '| --- | --- | --- |']
    for name, groups in contribution_effects.items():
        lines.append(f'| {name} | {estimate(groups["mismatch_responsibility_weighted_error"])} | '
                     f'{estimate(groups["agreement_responsibility_weighted_error"])} |')
    lines += ['', '组间误差关系是观察性关联；不能仅据此判断中心标签监督的因果作用。', '',
              '### 相机 mask 对齐后的局部干预', '',
              '场目标为带真实乘数的 CE+Lovasz；表中为同 batch 变化均值及 scene 配对 bootstrap 95% CI。负数表示场目标下降；自身目标下降/场目标上升的计数是局部机制证据。独立场景结果和 all-domain 控制保存在 JSON 中。', '',
              '| checkpoint | 分支 | 相对步长 | 自身目标 Δ | 场目标 Δ [95% CI] | 自身下降/场上升 |',
              '| --- | --- | ---: | ---: | --- | ---: |']
    for name, summary in diagnostics.items():
        for item in summary['branches']:
            if item['branch'] in ('camera/geometry_only', 'camera/classification_only'):
                lines.append(f'| {name} | {item["branch"]} | {item["radius"]:g} | '
                             f'{number(item["self_loss"]["mean"], 6)} | {estimate(item["same_batch_voxel"])} | '
                             f'{item["own_down_field_up_scenes"]}/{item["same_batch_voxel"]["n_scenes"]} |')
    lines += ['', '不从较大步长的 field 上升外推无穷小梯度冲突，不从负余弦外推训练成绩。QG/QS 分别结合 mask 控制、步长敏感性、独立场景和 A4 训练解释。', '',
              '### 1024/4096/16384 抽样敏感性', '',
              '| checkpoint | 每域样本量 | 场景数 | supported Gaussian 均数 | 多数错配率均数 [95% CI] |',
              '| --- | ---: | ---: | ---: | --- |']
    sensitivity = {}
    for name in ('epoch1', 'epoch5', 'epoch10'):
        for count in (1024, 4096, 16384, '全部 occupied'):
            filename = (f'{name}_intervene.json' if count == 1024 else
                        f'{name}_full_occupied.json' if isinstance(count, str) else f'{name}_sampling{count}.json')
            path = RUN / 'real' / filename
            data = read(path)
            if not data or data.get('status') != 'complete':
                lines.append(f'| {name} | {count} | 待完成 | — | — |')
                continue
            stats = [r['contribution'] for r in data['pairs']]
            supported = sum(r['supported_gaussians'] for r in stats) / len(stats)
            values = [r['majority_mismatch_fraction'] for r in stats if r.get('majority_mismatch_fraction') is not None]
            summary = interval(values)
            sensitivity[f'{name}/{count}'] = dict(supported_gaussians_mean=supported, mismatch=summary)
            lines.append(f'| {name} | {count} | {len(stats)} | {supported:.1f} | {estimate(summary)} |')
    lines += ['', '由于增加抽样量会改变达到支持门槛的 primitive 集合，抽样条件的错配率变化不只反映标签估计误差。全部 occupied 条件对相机可见 occupied voxel 精确累积责任目标；empty 仍独立抽样 1024。该条件用于检查并替代不稳定的抽样目标，仍不包含无 occupied 支持的 Gaussian。']
    lines += ['', '### mini 训练控制与官方 RayIoU', '',
              '训练 4096 帧、验证 1024 帧、5 epoch、seed 928、micro-batch 1、梯度累积 4；各组从同一预训练起点独立训练。主要比较共同最后 epoch；best 仅为附加指标。',
              '低预算实际独立训练，600 query × 32 children=19,200；早期 decoder 保留，最后 child 数改变。预算对照只使用相同 mini 训练预算的 a1_align_final，不能与完整数据十轮模型混为预算效应。', '',
              '| 组 | 已完成验证 epoch | 最后 mIoU % | 最后二值 IoU % | precision % | recall % | 官方 RayIoU % | best mIoU % (epoch) |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    training = {}
    for name in CONTROLS:
        directory = RUN / 'train' / f'{name}_seed928'
        try:
            curve = parse_log(sorted(directory.glob('20*.log')))
        except ValueError:
            curve = []
        evaluation = read(directory / 'evaluation_epoch5.json')
        if evaluation and evaluation.get('status') != 'complete':
            evaluation = None
        final = evaluation['camera_metrics'] if evaluation else None
        best = max(curve, key=lambda r: (r['mIoU'], -r['epoch'])) if curve else None
        training[name] = dict(epochs=curve, final_evaluation=evaluation,
                              training_runtime=read(directory / 'training_runtime.json'),
                              checkpoint5_exists=(directory / 'epoch_5.pth').exists())
        last = curve[-1] if curve else {}
        lines.append(f'| {name} | {",".join(str(r["epoch"]) for r in curve) or "待完成"} | '
                     f'{number(final["mIoU"] if final else last.get("mIoU"), 2)} | '
                     f'{number(final["IoU"] if final else last.get("IoU"), 2)} | '
                     f'{number(final["precision"] if final else None, 2)} | '
                     f'{number(final["recall"] if final else None, 2)} | '
                     f'{number(evaluation["ray_metrics"]["RayIoU"] if evaluation else None, 2)} | '
                     f'{number(best["mIoU"], 2) + " (" + str(best["epoch"]) + ")" if best else "—"} |')
    lines += ['', '各组评估 JSON 同时保存逐类 intersection/union、camera confusion、empty 误报、尺度饱和、耗时、峰值显存与 RayIoU@1/2/4。完整 epoch 曲线保存在训练日志和 [execution_summary.json](../../work_dirs/center_support_contribution1/execution_summary.json)。', '',
              '| 组 | 训练+验证耗时（小时） | 训练进程 CUDA 峰值 allocated MiB |',
              '| --- | ---: | ---: |']
    for name in CONTROLS:
        runtime = training[name]['training_runtime']
        if runtime and runtime.get('status') == 'complete':
            lines.append(f'| {name} | {runtime["training_and_validation_seconds"]/3600:.2f} | {runtime["peak_cuda_allocated_mib"]:.1f} |')
        else:
            lines.append(f'| {name} | 待完成 | — |')
    lines += ['', '成本由训练进程直接记录，包含其每轮验证，不用推理或四步预检的显存峰值替代完整训练峰值。', '',
              'RayIoU 直接调用[官方 OPUS](https://github.com/jbwang1997/OPUS/blob/main/loaders/ray_metrics.py)的 raycaster、ray 方向和 EgoPoseDataset；使用完整验证 scene 的最多 8 个 origins，不对 ray 输入加 camera mask。流式计数与官方 `calc_metrics` 数值逐项核对。产物记录官方 commit、文件 SHA256 和现有 PyTorch API 兼容补丁。mini RayIoU 不代表官方完整验证集成绩。', '',
              '### 完整验证集的官方指标', '',
              '| checkpoint | 验证帧数 | camera mIoU % | 二值 IoU % | RayIoU@1 % | RayIoU@2 % | RayIoU@4 % | RayIoU % |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    full_evaluations = {}
    for epoch in (1, 5, 10):
        data = read(RUN / 'real' / f'epoch{epoch}_official_eval.json')
        if not data or data.get('status') != 'complete':
            lines.append(f'| epoch{epoch} | 待完成 | — | — | — | — | — | — |')
            continue
        full_evaluations[f'epoch{epoch}'] = data
        camera, ray = data['camera_metrics'], data['ray_metrics']
        lines.append(f'| epoch{epoch} | {data["samples"]} | {camera["mIoU"]:.2f} | {camera["IoU"]:.2f} | '
                     f'{ray["RayIoU@1"]:.2f} | {ray["RayIoU@2"]:.2f} | {ray["RayIoU@4"]:.2f} | {ray["RayIoU"]:.2f} |')
    lines += ['', '### QG 与 QS 的证据解释', '',
              '下面分别描述已完成的小步证据和单 seed mini 控制，不自动将二者合并成完整 Occ3D 因果结论。']
    for question, branch in (('QG', 'camera/geometry_only'), ('QS', 'camera/classification_only')):
        tested, positive, negative, crosses = [], [], [], []
        for name in ('epoch1', 'epoch5', 'epoch10'):
            summary = diagnostics.get(name)
            if not summary:
                continue
            for item in summary['branches']:
                if item['branch'] != branch or item['radius'] > 1e-4:
                    continue
                tag = f'{name}/rho={item["radius"]:g}'
                own, field = item['self_loss']['ci95'], item['same_batch_voxel']['ci95']
                tested.append(tag)
                if own and own[1] < 0 and field and field[0] > 0:
                    positive.append(tag)
                elif own and own[1] < 0 and field and field[1] < 0:
                    negative.append(tag)
                else:
                    crosses.append(tag)
        if not tested:
            lines.append(f'{question}：真实小步诊断尚待完成。')
        else:
            lines.append(f'{question}：在 rho≤1e−4 的 {len(tested)} 个 checkpoint/步长条件中，'
                         f'{len(positive)} 个条件同时满足自身目标下降 CI<0、场目标上升 CI>0；'
                         f'{len(negative)} 个条件自身和场目标均下降；其余 {len(crosses)} 个条件的相应区间跨零或自身下降未确认。'
                         f'支持条件：{", ".join(positive) or "无"}。这些是逐条件区间，未作多重比较校正，不能解释为整体显著性。')
    baseline = training['a1_align_final']['final_evaluation']
    if baseline:
        base = baseline['camera_metrics']['mIoU']
        lines += ['', f'以相机域对齐基线 a1_align_final 的最后 epoch mIoU={base:.2f}% 为参照：']
        for name in CONTROLS:
            evaluation = training[name]['final_evaluation']
            if evaluation and name != 'a1_align_final':
                delta = evaluation['camera_metrics']['mIoU'] - base
                ray_delta = evaluation['ray_metrics']['RayIoU'] - baseline['ray_metrics']['RayIoU']
                lines.append(f'- {name}：mIoU {delta:+.2f} 个百分点，RayIoU {ray_delta:+.2f} 个百分点。')
        lines += ['', '降权与删除的差值分别用于检查权重效应和最终目标必要性；单 seed、五轮结果只用于筛选，不具有多 seed 训练不确定性。a5 若追平其他组，应承认最终 voxel 监督已经足够支持这轮成绩。']
    else:
        lines += ['', 'A4 控制尚待相机域对齐基线与对应控制的最后 epoch 官方评估完成。QG/QS 的训练效应保持待判断。']
    lines += ['', '### 执行完整性', '',
              '| 任务 | 状态 |', '| --- | --- |']
    for task in state.get('tasks', []):
        lines.append(f'| {task["name"]} | {task["status"]} |')
    lines += ['', '任何尚未完成的任务保持“待完成”；队列只有全部必要产物验证通过后才标记 complete。多种子和完整数据迁移属于筛选后的阶段，不与本轮单 seed mini 结果混写。', '', '</details>', '']
    write_json(RUN / 'execution_summary.json', dict(status=state['status'], diagnostics=diagnostics,
        contribution_error_relationship=contribution_effects,
        sampling_sensitivity=sensitivity, training=training, full_official_evaluations=full_evaluations))
    original = REPORT.read_text()
    if original.count(AUTO_START) != 1 or original.count(AUTO_END) != 1:
        raise ValueError('Report must contain exactly one automatic-results start/end pair')
    prefix, remainder = original.split(AUTO_START)
    _, suffix = remainder.split(AUTO_END)
    figure = render_overview(diagnostics, training)
    overview = reader_summary(state, diagnostics, training, sensitivity, full_evaluations, figure=figure)
    REPORT.write_text(prefix + AUTO_START + '\n\n' + '\n'.join(overview + lines) + '\n' + AUTO_END + suffix)


if __name__ == '__main__':
    refresh()
