"""Readable, data-derived report for the OPUS Gaussian spatial calibration audit."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'plans/sparse_render_offical_9_28/opus_gaussian_empty_scope_results.md'
MASKS = [('camera', '相机可见'), ('camera_invisible', 'camera mask 外'), ('all_valid', '全空间')]


def m(model, mask, calibrated=False):
    shift = model['selected_shift'] if calibrated else 0.
    return model['validation']['masks'][mask]['empty_logit_shift_metrics'][str(shift)]


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ['Noto Sans CJK SC', 'Noto Sans CJK JP'] if f in available] + ['DejaVu Sans']
    with plt.rc_context({'font.family': fonts, 'axes.unicode_minus': False, 'svg.fonttype': 'path'}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
        for row, (name, title) in enumerate([('current', '当前版'), ('original', '原始版')]):
            d = summary['models'][name]
            for ax, (mask, label) in zip(axes[row], MASKS):
                for split, caption, color in [('training', '训练子集 256 帧', '#176a90'), ('validation', '验证集 1,024 帧', '#be5c2a')]:
                    options = d[split]['masks'][mask]['empty_logit_shift_metrics']
                    bs = sorted(map(float, options))
                    ax.plot(bs, [options[str(b)]['mIoU'] for b in bs], marker='.', label=caption, color=color)
                ax.axvline(0, color='gray', linewidth=.8)
                ax.axvline(d['selected_shift'], color='#5c7840', linestyle='--', label=f'训练集 camera 选定 b={d["selected_shift"]:+g}')
                ax.set(title=title+' · '+label, xlabel='额外 empty 偏移 b', ylabel='mIoU（%）')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
        for extension in ['png', 'svg']:
            fig.savefig(REPORT.parent / 'figures' / ('opus_gaussian_empty_scope.' + extension), dpi=180)
        plt.close(fig)


def write_report(summary):
    lines = ['# OPUS Gaussian：可见区域的 empty 校准能否推广到全空间？', '',
        '**实验状态：全部完成。** 两个版本都固定最后 epoch 5 权重；每组 256 帧训练子集选偏移、1,024 帧验证。'
        '偏移 b 定义为在模型已有 learned bias 等操作之后，额外执行 `empty_logit -= b`。', '',
        '## 1. 直接回答：两个版本分别发生了什么？', '',
        '下表中的差值为“额外偏移后 − 默认输出”，单位为百分点。偏移只按训练子集 camera-mask mIoU 选择，所有空间范围固定使用同一个值。', '',
        '| 模型 | 训练集选定 b | 可见 mIoU：默认→校准 | Δ | 全空间 mIoU：默认→校准 | Δ | RayIoU：默认→校准 | Δ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, d in summary['models'].items():
        a, b, f, g = m(d, 'camera'), m(d, 'camera', True), m(d, 'all_valid'), m(d, 'all_valid', True)
        r, s = d['raw_official']['ray_metrics']['RayIoU'], d['calibrated_official']['ray_metrics']['RayIoU']
        lines.append(f'| {d["label"]} | {d["selected_shift"]:+g} | {a["mIoU"]:.2f}→{b["mIoU"]:.2f} | {b["mIoU"]-a["mIoU"]:+.2f} | '
            f'{f["mIoU"]:.2f}→{g["mIoU"]:.2f} | {g["mIoU"]-f["mIoU"]:+.2f} | {r:.2f}→{s:.2f} | {s-r:+.2f} |')
    for name, title in [('current', '当前版'), ('original', '原始版')]:
        d = summary['models'][name]
        camera_gain = m(d, 'camera', True)['mIoU']-m(d, 'camera')['mIoU']
        full_gain = m(d, 'all_valid', True)['mIoU']-m(d, 'all_valid')['mIoU']
        ray_gain = d['calibrated_official']['ray_metrics']['RayIoU']-d['raw_official']['ray_metrics']['RayIoU']
        lines += ['', f'**{title}：** 可见 mIoU {camera_gain:+.2f}，全空间 mIoU {full_gain:+.2f}，RayIoU {ray_gain:+.2f} 个百分点。'
            '三项指标分别衡量不同错误与范围，不能用其中一项替代其他两项。']
    lines += ['', '**综合判断：** 相机可见区域的 mIoU 收益不能保证全空间体素指标和 RayIoU 同时受益。'
        '当前版的全空间 mIoU 下降，但 RayIoU 基本持平；原始版的全空间 mIoU 基本持平，RayIoU 却下降。'
        '因此“降低 empty 必然损害全空间 mIoU”并不成立。更明确的共同现象是：按可见区域选择的偏移，'
        '会同时降低 camera mask 外的 occupied 漏检、增加该区域的 empty 误占用，收益随区域和指标变化。', '',
        '## 2. 漏检减少的代价：误占用在哪里增加？', '',
        '下表比例均为百分数。occupied→empty 以该范围真实 occupied 为分母；empty→occupied 以该范围真实 empty 为分母。', '',
        '| 版本 | 范围 | 决策 | mIoU | 二值 IoU | precision | recall | occupied→empty | empty→occupied | 预测占用率 | GT 占用率 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, title in [('current', '当前版'), ('original', '原始版')]:
        d = summary['models'][name]
        for mask, label in MASKS:
            for calibrated in [False, True]:
                values = m(d, mask, calibrated)
                lines.append(f'| {title} | {label} | '+('校准' if calibrated else '默认')+' | '+' | '.join(f'{values[k]:.2f}' for k in
                    ['mIoU', 'IoU', 'precision', 'recall', 'occupied_missed_as_empty_percent', 'empty_falsely_occupied_percent',
                     'predicted_occupied_fraction', 'gt_occupied_fraction'])+' |')
    current = summary['models']['current']
    outside_raw, outside_cal = m(current, 'camera_invisible'), m(current, 'camera_invisible', True)
    lines += ['', f'当前版在 camera mask 外的 GT 占用率仅为 **{outside_raw["gt_occupied_fraction"]:.2f}%**，'
        f'预测占用率却从默认的 **{outside_raw["predicted_occupied_fraction"]:.2f}%** 进一步升至 '
        f'**{outside_cal["predicted_occupied_fraction"]:.2f}%**。'
        '这表明这些区域同时存在 occupied 漏检和 empty 误占用，不能只按漏检率将它们概括为“empty 分数过高”。', '',
        'camera mask 外是数据集评估 mask 的补集，不自动等同于物理上完全不可观测区域。'
        '每个偏移均核对“camera 内混淆矩阵＋camera 外混淆矩阵＝全空间混淆矩阵”。'
        'mIoU 是非线性统计量，三个范围的 mIoU 差值不能直接相加。', '',
        '## 3. 完整偏移敏感性与固定 +4 参照', '',
        '本次候选为完整整数网格，比历史 {−2, 0, 2, 4, 6, 8} 更细；'
        '原始版因此可能选到不同于历史 +4 的值。它仍完全按训练子集 camera mIoU 选择，'
        '不是观察验证结果后更换偏移。下表保留固定 +4 的结果，方便核对原有结论。', '',
        '![OPUS Gaussian 在三个空间范围的 empty 偏移曲线](figures/opus_gaussian_empty_scope.png)', '',
        '[矢量图 SVG](figures/opus_gaussian_empty_scope.svg)。验证曲线只展示敏感性，未用于选值。', '',
        '| 版本 | 范围 | b=0 mIoU | 固定 b=+4 mIoU |', '| --- | --- | ---: | ---: |']
    for name, title in [('current', '当前版'), ('original', '原始版')]:
        for mask, label in MASKS:
            d = summary['models'][name]
            plus = d['validation']['masks'][mask]['empty_logit_shift_metrics']['4.0']['mIoU']
            lines.append(f'| {title} | {label} | {m(d, mask)["mIoU"]:.2f} | {plus:.2f} |')
    gf = summary['gaussianformer_reference']
    lines += ['', '## 4. 与 GaussianFormer 的参照和结论边界', '',
        f'上次 GaussianFormer 实验选定 b={gf["selected_shift"]:+g}：可见 mIoU '
        f'{m(gf,"camera")["mIoU"]:.2f}→{m(gf,"camera",True)["mIoU"]:.2f}，全空间 mIoU '
        f'{m(gf,"all_valid")["mIoU"]:.2f}→{m(gf,"all_valid",True)["mIoU"]:.2f}。'
        '这里复用同一 mini 划分上的历史产物，不重新训练或评估 GaussianFormer。', '',
        '本实验能检验“按可见区域选定的全局 empty 修正是否损害其他范围”，不能直接证明 empty 偏差来自架构、'
        'OPUS 分类损失或某个独立机制。降低 empty 后语义 mIoU 改善也不等于几何全面改善。', '',
        '当前版与原始版同时存在语义激活、voxel 监督 mask 和 learned bias 等差异，跨版本对比不能单独识别某个改动的因果作用。'
        '当前版的 learned empty bias 始终保留；本次只研究其上的额外偏移。'
        '全部结果限于单 seed、五轮 mini 权重；验证集曾用于历史诊断，不是全新独立测试集。', '',
        '官方 RayIoU 使用无 camera mask 的完整 scene origins 和官方语义/深度匹配，并通过计数一致性核验。'
        '扫描的 camera 指标与默认、校准官方评估结果严格一致；复用的历史产物路径和 SHA256 保存在汇总中。', '',
        '## 5. 复现与产物', '',
        '- [实验设计](opus_gaussian_empty_scope_design.md)',
        '- [汇总 JSON](../../work_dirs/opus_gaussian_empty_scope/summary.json)',
        '- [完成核验](../../work_dirs/opus_gaussian_empty_scope/completion_audit.json)',
        '- [训练子集记录](../../work_dirs/opus_gaussian_empty_scope/training_calibration_manifest.json)',
        '- [GaussianFormer 参照报告](gaussianformer_empty_diagnostic_results.md)', '',
        '运行 `python3.10 tools/run_opus_gaussian_empty_scope.py` 可复用已核验产物或补齐缺失评估。']
    REPORT.write_text('\n'.join(lines)+'\n')
