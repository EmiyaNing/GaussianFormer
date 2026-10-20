"""Run/reuse the frozen GaussianFormer empty audit and produce a readable report."""
from __future__ import annotations

from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.center_support_common import write_json
from tools.diagnose_opus_v1_gaussian import choose_empty_shift, file_sha256

RUN = ROOT / 'work_dirs/gaussianformer_empty_diagnostic'
REPORT = ROOT / 'plans/sparse_render_offical_9_28/gaussianformer_empty_diagnostic_results.md'
CONFIG = 'config/mini_occ3d_4096_1024/gaussianformer.py'
CHECKPOINT = 'work_dirs/mini_occ3d_4096_1024/gaussianformer/epoch_5.pth'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def job(name, command, requirements, checkpoint_hash):
    output = RUN / (name + '.json')
    result = read(output)
    if result and result.get('status') == 'complete':
        assert all(result['protocol'].get(k) == v for k, v in requirements.items()), name
        assert result['checkpoint_sha256'] == checkpoint_hash, name
        return result
    write_json(RUN / 'status.json', dict(status='running', task=name, command=command,
        updated_at=datetime.now().astimezone().isoformat()))
    print('Starting', name, flush=True)
    with (RUN / (name + '.log')).open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = read(output)
    if not result or result.get('status') != 'complete':
        raise RuntimeError(f'{name} exited without a complete result: {output}')
    return job(name, command, requirements, checkpoint_hash)


def sweep(split, shifts, checkpoint_hash):
    limit = 256 if split == 'train' else 1024
    suffix = 'wide' if len(shifts) > 17 else 'initial'
    name = split + '_' + suffix
    cmd = [sys.executable, 'tools/diagnose_gaussianformer_empty.py', '--py-config', CONFIG,
           '--checkpoint', CHECKPOINT, '--split', split, '--limit', str(limit), '--shifts',
           *map(str, shifts), '--out', str((RUN / (name + '.json')).relative_to(ROOT))]
    if split == 'train':
        cmd += ['--selection-manifest', str((RUN / 'training_calibration_manifest.json').relative_to(ROOT))]
    return job(name, cmd, dict(py_config=CONFIG, checkpoint=CHECKPOINT, split=split,
                               limit=limit, shifts=shifts), checkpoint_hash)


def selected_shift(training):
    return choose_empty_shift(dict(split=training['split'],
        empty_logit_shift_metrics=training['masks']['camera']['empty_logit_shift_metrics']))


def plot(training, validation, shift):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = [f for f in ['Noto Sans CJK SC', 'Noto Sans CJK JP'] if f in names] + ['DejaVu Sans']
    with plt.rc_context({'font.family': font, 'axes.unicode_minus': False, 'svg.fonttype': 'path'}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
        for ax, (mask, title) in zip(axes, [('camera', '相机可见区域'), ('all_valid', '全部有效体素（无 camera mask）')]):
            for d, label, color in [(training, '训练子集 256 帧', '#176a90'), (validation, '验证集 1,024 帧', '#be5c2a')]:
                options = d['masks'][mask]['empty_logit_shift_metrics']
                bs = sorted(map(float, options))
                ax.plot(bs, [options[str(b)]['mIoU'] for b in bs], marker='.', label=label, color=color)
            ax.axvline(0, color='gray', linewidth=.8)
            ax.axvline(shift, linestyle='--', color='#5c7840', label=f'训练集 camera mIoU 选定 b={shift:+g}')
            ax.set(title=title, xlabel='额外 empty 偏移 b（正值减少 empty）', ylabel='语义 mIoU（%）')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        folder = REPORT.parent / 'figures'
        for extension in ['png', 'svg']:
            fig.savefig(folder / ('gaussianformer_empty_calibration.' + extension), dpi=180)
        plt.close(fig)


def report(summary):
    train, val, b = summary['training'], summary['validation'], summary['selected_shift']
    raw = val['masks']['camera']['empty_logit_shift_metrics']['0.0']
    cal = val['masks']['camera']['empty_logit_shift_metrics'][str(b)]
    full_raw = val['masks']['all_valid']['empty_logit_shift_metrics']['0.0']
    full_cal = val['masks']['all_valid']['empty_logit_shift_metrics'][str(b)]
    lines = ['# GaussianFormer：是否也存在 empty 过度预测？', '',
        '**状态：全部完成。GaussianFormer2 无可用权重，按用户要求跳过。**', '',
        f'在本仓库 GaussianFormer 五轮 mini checkpoint 上，默认 camera mIoU 为 **{raw["mIoU"]:.2f}**，'
        f'训练子集选定 `empty_logit -= {b:g}` 后为 **{cal["mIoU"]:.2f}**'
        f'（**{cal["mIoU"]-raw["mIoU"]:+.2f} 个百分点**）。'
        f'真实 occupied 被误判为 empty 的比例从 **{raw["occupied_missed_as_empty_percent"]:.2f}%** 变为 '
        f'**{cal["occupied_missed_as_empty_percent"]:.2f}%**；真实 empty 被误判为 occupied 的比例从 '
        f'**{raw["empty_falsely_occupied_percent"]:.2f}%** 变为 **{cal["empty_falsely_occupied_percent"]:.2f}%**。', '',
        '这支持当前 checkpoint 存在可通过降低 empty 分数缓解的保守决策，但减少漏检也增加了误占用。'
        '它不能证明所有 GaussianFormer checkpoint 或 semantic Gaussian 表征都存在相同缺陷。', '',
        f'**评估范围改变时，收益并不保持：** 同一个 b={b:+g} 在全部有效体素上使 mIoU 从 '
        f'**{full_raw["mIoU"]:.2f}** 变为 **{full_cal["mIoU"]:.2f}**'
        f'（{full_cal["mIoU"]-full_raw["mIoU"]:+.2f} 个百分点）。'
        '因此应将“可见 occupied 漏检较多”与“全空间应该统一降低 empty”区分开；前者不能直接推出后者。', '',
        '## 1. 实验如何执行', '',
        '复用已有 R50、25,600 Gaussian、六相机单帧 704×256 的五轮 mini 权重，不重新训练、不修改模型参数。'
        '在训练 split 随机抽取共同的 256 帧，仅按 camera-mask mIoU 选择偏移，再固定用于全部 1,024 帧验证。'
        '相同偏移直接用于无 camera mask 的指标；验证集和无 mask 曲线均未参与选值。', '',
        'b 定义为渲染之后执行 `empty_logit -= b`。b>0 减少 empty，b<0 增加 empty；'
        '全部评估帧都逐帧验证 b=0 与原生 `final_occ` 完全一致。'
        '当前模型没有额外 learned empty bias，使用可学习 empty scalar 生成全局 empty Gaussian。', '',
        f'训练集选定 b={b:+g}；empty scalar={val["empty_scalar"]:.4f}。最后 epoch 5 固定作为评估 checkpoint。', '',
        '## 2. 默认、训练集校准与固定 +4 对照', '',
        '以下数值均为百分数。漏检率以真实 occupied 为分母；误占用率以真实 empty 为分母。', '',
        '| 评估范围 | 决策 | b | mIoU | 二值 IoU | precision | recall | occupied→empty | empty→occupied | 预测占用率 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mask, label in [('camera', '相机可见'), ('all_valid', '全部有效体素')]:
        for mode, shift in [('默认', 0.), ('训练集选定', b), ('固定 +4', 4.)]:
            m = val['masks'][mask]['empty_logit_shift_metrics'][str(shift)]
            lines.append(f'| {label} | {mode} | {shift:+g} | ' + ' | '.join(f'{m[k]:.2f}' for k in
                ['mIoU', 'IoU', 'precision', 'recall', 'occupied_missed_as_empty_percent',
                 'empty_falsely_occupied_percent', 'predicted_occupied_fraction']) + ' |')
    lines += ['', '真实 GT 占用率：' + '；'.join(f'{label} {val["masks"][mask]["empty_logit_shift_metrics"]["0.0"]["gt_occupied_fraction"]:.2f}%'
        for mask, label in [('camera', '相机可见'), ('all_valid', '全部有效体素')]) + '。', '',
        '![GaussianFormer 的 empty 偏移敏感性](figures/gaussianformer_empty_calibration.png)', '',
        '[矢量图 SVG](figures/gaussianformer_empty_calibration.svg)。曲线展示敏感性，不能据验证曲线另选最优值。', '',
        '## 3. 无 camera mask 的官方 RayIoU', '',
        '| 决策 | b | RayIoU | @1 m | @2 m | @4 m |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, title, shift in [('raw_official', '默认', 0.), ('calibrated_official', '训练集选定', b)]:
        ray = summary[mode]['ray_metrics']
        lines.append(f'| {title} | {shift:+g} | ' + ' | '.join(f'{ray[k]:.2f}' for k in
                     ['RayIoU', 'RayIoU@1', 'RayIoU@2', 'RayIoU@4']) + ' |')
    lines += ['', '使用完整验证场景的 ego origins、全部 1,024 帧以及官方射线/语义与深度匹配规则；'
        '两项均通过 streaming 与官方计数一致性核验。RayIoU 包含语义匹配，不能单独解释为几何能力。', '',
        '### 重复前向与数值复核', '',
        '对同一输入连续执行两次 `no_grad` 前向，也观察到少量预测变化；换成 `inference_mode` 同样存在变化。'
        '因此跨次前向不能要求逐位相同。本报告的全部偏移扫描均复用每帧同一个渲染张量，b=0 与该次原生预测严格相等；'
        '官方 RayIoU 使用独立前向，其 camera 指标与扫描结果的差异逐项记录如下，复核容差为 0.02 个百分点。'
        '该容差用于检查独立评估结果是否接近，不代表预测完全可复现。重复前向检查记录在 '
        '`work_dirs/gaussianformer_empty_diagnostic/forward_repeatability.json`。', '',
        '| 决策 | ΔmIoU | Δ二值 IoU | Δprecision | Δrecall |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, label in [('raw_official', '默认'), ('calibrated_official', '训练集选定')]:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{summary["independent_camera_metric_deltas"][name][key]:+.6f}'
            for key in ['mIoU', 'IoU', 'precision', 'recall']) + ' |')
    lines += ['',
        '## 4. 前景语义与逐类 empty 漏检', '',
        '额外 empty 偏移不改变前景通道排序。仅在真实 occupied 上忽略 empty、比较 17 个前景类的准确率为：' +
        '；'.join(f'{label} **{val["masks"][mask]["foreground_accuracy_on_gt_occupied"]:.2f}%**'
        for mask, label in [('camera', '相机可见'), ('all_valid', '全部有效体素')]) + '。', '',
        '| 类别（相机可见） | 默认 IoU | 校准后 IoU | 默认被误判 empty 的比例 |', '| --- | ---: | ---: | ---: |']
    counts = val['masks']['camera']['default_per_class_gt_and_empty_misses']
    for i, category in enumerate(raw['per_class_IoU']):
        fraction = 100 * counts[i][1] / max(counts[i][0], 1)
        lines.append(f'| {category} | {raw["per_class_IoU"][category]:.2f} | {cal["per_class_IoU"][category]:.2f} | {fraction:.2f}% |')
    lines += ['', '## 5. 结论边界与复现', '',
        '本实验验证的是当前仓库的 Occ3D 适配版、一个 seed、五轮 mini 训练后的 checkpoint。'
        '与 OPUS Gaussian + camera mask + learned bias 的训练 mask、参数化及监督不同，跨模型差异不能只归因于架构。'
        '若要判断表征是否固有地导致 empty 问题，还需要匹配训练条件、更多 checkpoint 和完整训练验证。', '',
        'GaussianFormer2 缺少可用权重，已按用户要求跳过；没有将 OPUSv2 或仅切换聚合器的模型当作 GaussianFormer2。'
        '官方实现参考：[GaussianFormer / GaussianFormer2 仓库](https://github.com/huang-yh/GaussianFormer)。', '',
        '- [实验设计](gaussianformer_empty_diagnostic_design.md)',
        '- [汇总 JSON](../../work_dirs/gaussianformer_empty_diagnostic/summary.json)',
        '- [训练子集记录](../../work_dirs/gaussianformer_empty_diagnostic/training_calibration_manifest.json)',
        '- [完成核验](../../work_dirs/gaussianformer_empty_diagnostic/completion_audit.json)', '',
        '复现入口：`python3.10 tools/finish_gaussianformer_empty_diagnostic.py`。'
        '该脚本先复用并核验已完成产物，缺失时再执行评估；不会训练模型。']
    REPORT.write_text('\n'.join(lines) + '\n')


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'work_dirs/center_support_contribution1/execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint_hash = file_sha256(ROOT / CHECKPOINT)
    shifts = list(map(float, range(-8, 9)))
    training = sweep('train', shifts, checkpoint_hash)
    if abs(selected_shift(training)) == 8:
        shifts = list(map(float, range(-16, 17)))
        training = sweep('train', shifts, checkpoint_hash)
    b = selected_shift(training)
    validation = sweep('val', shifts, checkpoint_hash)
    assert training['samples'] == 256 and validation['samples'] == 1024
    assert training['imageset_sha256'] != validation['imageset_sha256']
    assert training['native_zero_offset_parity_passed'] and validation['native_zero_offset_parity_passed']
    official, numerical_deltas = {}, {}
    for name, shift in [('raw_official', 0.), ('calibrated_official', b)]:
        cmd = [sys.executable, 'tools/eval_center_support.py', '--py-config', CONFIG, '--checkpoint', CHECKPOINT,
               '--out', str((RUN / (name + '.json')).relative_to(ROOT)), '--empty-logit-shift', str(shift)]
        e = job(name, cmd, dict(py_config=CONFIG, checkpoint=CHECKPOINT, empty_logit_shift=shift), checkpoint_hash)
        assert e['samples'] == 1024 and e['streaming_official_parity_passed']
        m = validation['masks']['camera']['empty_logit_shift_metrics'][str(shift)]
        differences = {key:e['camera_metrics'][key] - m[key] for key in ['mIoU', 'IoU', 'precision', 'recall']}
        assert all(abs(value) < .02 for value in differences.values()), differences
        numerical_deltas[name] = differences
        official[name] = e
    summary = dict(status='complete', finished_at=datetime.now().astimezone().isoformat(), selected_shift=b,
        training=training, validation=validation, **official,
        independent_camera_metric_deltas=numerical_deltas,
        independent_camera_metric_tolerance_pp=.02,
        gaussianformer2=dict(status='skipped', reason='No available checkpoint; user explicitly requested skipping'),
        source_sha256={p:file_sha256(ROOT / p) for p in [CONFIG, 'model/head/gaussian_head.py',
            'tools/diagnose_gaussianformer_empty.py', 'tools/finish_gaussianformer_empty_diagnostic.py',
            'tools/eval_center_support.py']})
    write_json(RUN / 'summary.json', summary)
    plot(training, validation, b)
    report(summary)
    write_json(RUN / 'completion_audit.json', dict(status='complete', samples_train=256, samples_validation=1024,
        selected_shift_training_camera_only=b, checkpoint_sha256=checkpoint_hash,
        strict_state_dict_loaded=True, zero_offset_native_parity=True,
        camera_sweep_official_exact_parity=all(abs(v)<1e-8 for d in numerical_deltas.values() for v in d.values()),
        camera_sweep_official_within_tolerance=True, camera_metric_tolerance_pp=.02,
        independent_camera_metric_deltas=numerical_deltas,
        official_ray_count_parity=True, gaussianformer2_skipped=True))
    write_json(RUN / 'status.json', dict(status='complete', finished_at=summary['finished_at']))
    print('Completed; training selected shift', b, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(RUN / 'status.json', dict(status='failed', error=repr(error)))
        raise
