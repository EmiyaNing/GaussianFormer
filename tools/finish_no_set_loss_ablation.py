"""Evaluate all-stage set-loss removal with matched train-only empty calibration."""
from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.center_support_common import write_json
from tools.diagnose_opus_v1_gaussian import choose_empty_shift, file_sha256
from tools.summarize_mini_occ3d import parse_log

RUN = ROOT / 'work_dirs/no_set_loss_ablation_mini'
REPORT = ROOT / 'plans/sparse_render_offical_9_28/no_set_loss_mini_results.md'
BASE_CONFIG = 'config/center_support_contribution1/a0_original.py'
NEW_CONFIG = 'config/mini_occ3d_4096_1024/opusv1_gaussian_no_set_loss.py'
BASE_TRAIN = 'work_dirs/center_support_contribution1/train/a0_original_seed928'
NEW_TRAIN = 'work_dirs/no_set_loss_ablation_mini/train/no_set_loss_seed928'
MODELS = {'with_set_loss': (BASE_CONFIG, BASE_TRAIN), 'no_set_loss': (NEW_CONFIG, NEW_TRAIN)}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def run_job(name, command, output, requirements):
    existing = read(output)
    if existing and existing.get('status') == 'complete':
        if any(existing.get('protocol', {}).get(k) != v for k, v in requirements.items()):
            raise ValueError(f'Existing result protocol differs: {output}')
        if existing.get('checkpoint_sha256') != file_sha256(ROOT / requirements['checkpoint']):
            raise ValueError(f'Checkpoint changed since the saved result: {output}')
        return existing
    write_json(RUN / 'evaluation_status.json', dict(status='running', task=name, command=command,
        updated_at=datetime.now().astimezone().isoformat()))
    print('Starting', name, flush=True)
    with (RUN / f'{name}.log').open('a') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{name} failed; inspect {RUN / (name + ".log")}')
    result = read(output)
    if not result or result.get('status') != 'complete':
        raise RuntimeError(f'{name} returned without complete output')
    if any(result.get('protocol', {}).get(k) != v for k, v in requirements.items()):
        raise ValueError(f'Generated result protocol differs: {output}')
    print('Completed', name, flush=True)
    return result


def diagnose(name, split, shifts):
    config, directory = MODELS[name]
    checkpoint = directory + '/epoch_5.pth'
    limit = 256 if split == 'train' else 1024
    suffix = 'wide' if len(shifts) > 17 else 'initial'
    output = RUN / f'{name}_{split}_{suffix}.json'
    command = [sys.executable, 'tools/diagnose_opus_v1_gaussian.py', '--py-config', config,
        '--checkpoint', checkpoint, '--split', split, '--limit', str(limit),
        '--selection-seed', '928', '--out', str(output.relative_to(ROOT)),
        '--shifts', *map(str, shifts)]
    if split == 'train':
        command += ['--selection-manifest', str((RUN / 'training_calibration_manifest.json').relative_to(ROOT))]
    return run_job(f'{name}_{split}_{suffix}', command, output,
        dict(py_config=config, checkpoint=checkpoint, split=split, limit=limit, shifts=shifts))


def official(name, shift, raw=None):
    config, directory = MODELS[name]
    checkpoint = directory + '/epoch_5.pth'
    output = RUN / f'{name}_official_{"raw" if raw is None else "calibrated"}.json'
    if raw is not None and shift == 0:
        result = copy.deepcopy(raw)
        result['protocol'] = dict(result['protocol'], out=str(output.relative_to(ROOT)), empty_logit_shift=0.)
        result['checkpoint_sha256'] = file_sha256(ROOT / checkpoint)
        result['reuse_reason'] = 'Training selected zero extra shift; predictions and metrics are exactly the raw evaluation.'
        result['reused_evaluation_elapsed_seconds'] = result.get('elapsed_seconds')
        result['elapsed_seconds'] = 0.
        write_json(output, result)
        return result
    command = [sys.executable, 'tools/eval_center_support.py', '--py-config', config,
        '--checkpoint', checkpoint, '--out', str(output.relative_to(ROOT)),
        '--empty-logit-shift', str(shift)]
    return run_job(output.stem, command, output,
                   dict(py_config=config, checkpoint=checkpoint, empty_logit_shift=shift))


def write_report(summary):
    baseline = summary['models']['with_set_loss']
    ablated = summary['models']['no_set_loss']
    base_raw = baseline['raw_official']['camera_metrics']['mIoU']
    new_raw = ablated['raw_official']['camera_metrics']['mIoU']
    new_calibrated = ablated['calibrated_official']['camera_metrics']['mIoU']
    plus_four = ablated['validation_sweep']['empty_logit_shift_metrics']['4.0']['mIoU']
    lines = ['# mini 消融：完全移除 OPUS set loss 后的性能与 empty 校准', '',
        '实验状态：**全部完成**。主比较为相同五轮预算的最后 epoch；差值由“无 set loss”减去“保留 set loss”。', '',
        '两组均使用当前 softplus + learned empty bias 模型。额外偏移 b 定义为在 learned bias 之后再做 `empty_logit -= b`；'
        'b > 0 减少 empty 判定，b < 0 增加 empty 判定。它不等于训练得到的 learned bias。', '',
        f'**先看结果：** 默认 mIoU 从基线的 **{base_raw:.2f}** 变为无 set loss 的 **{new_raw:.2f}**。'
        f'无 set loss 模型在训练集选定 b={ablated["selected_shift"]:+.1f} 后，验证 mIoU 为 **{new_calibrated:.2f}**'
        f'（相对默认 {new_calibrated-new_raw:+.2f} 个百分点）；固定 b=+4 时为 **{plus_four:.2f}**。'
        '应将下面的性能差距与校准收益分别理解，额外偏移只能改变 empty 与前景的竞争，不能改变前景通道之间的排序。', '',
        '## 1. 默认输出：是否下降，以及下降多少？', '',
        '| 模型 | camera mIoU ↑ | 二值 IoU ↑ | precision ↑ | recall ↑ | 官方 RayIoU ↑ |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, label in [('with_set_loss', '保留全部 set loss（基线）'), ('no_set_loss', '完全移除六阶段 set loss')]:
        raw = summary['models'][name]['raw_official']
        m = raw['camera_metrics']
        lines.append(f'| {label} | {m["mIoU"]:.2f} | {m["IoU"]:.2f} | {m["precision"]:.2f} | {m["recall"]:.2f} | {raw["ray_metrics"]["RayIoU"]:.2f} |')
    delta = summary['raw_delta_no_set_minus_baseline']
    lines += [f'| 差值（百分点） | {delta["mIoU"]:+.2f} | {delta["IoU"]:+.2f} | {delta["precision"]:+.2f} | {delta["recall"]:+.2f} | {delta["RayIoU"]:+.2f} |', '',
        '## 2. 是否仍受益于额外的 empty 固定偏移？', '',
        '偏移只在同一组训练集随机 256 帧上按 mIoU 选择，随后固定用于全部 1,024 帧验证。验证集未用于选偏移或选 checkpoint。', '',
        '| 模型 | 训练集选定 b | 默认 mIoU | 固定 b 后 mIoU | 增益（百分点） | 默认 RayIoU | 固定 b 后 RayIoU |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, label in [('with_set_loss', '保留 set loss'), ('no_set_loss', '移除 set loss')]:
        d = summary['models'][name]
        a, b = d['raw_official'], d['calibrated_official']
        lines.append(f'| {label} | {d["selected_shift"]:+.1f} | {a["camera_metrics"]["mIoU"]:.2f} | {b["camera_metrics"]["mIoU"]:.2f} | '
            f'{b["camera_metrics"]["mIoU"]-a["camera_metrics"]["mIoU"]:+.2f} | {a["ray_metrics"]["RayIoU"]:.2f} | {b["ray_metrics"]["RayIoU"]:.2f} |')
    delta = summary['calibrated_delta_no_set_minus_baseline']
    lines += ['', f'给予两组相同校准机会后，“无 set loss − 基线”的 mIoU 差值为 **{delta["mIoU"]:+.2f} 个百分点**，'
              f'RayIoU 差值为 **{delta["RayIoU"]:+.2f} 个百分点**。']
    if summary.get('calibration_figure'):
        lines += ['', '![两组训练和验证 mIoU 随额外 empty 偏移的变化，标线为训练集选定偏移。](figures/no_set_loss_mini_calibration.png)', '',
                  '[下载矢量图 SVG](figures/no_set_loss_mini_calibration.svg)。验证曲线用于展示敏感性，未参与选值。']
    lines += ['', '下表给出无额外偏移、预先保留的 +4 偏移、以及训练集选定偏移的完整占用指标。+4 是历史诊断参照，不参与验证集调参。', '',
        '| 模型 | 决策方式 | b | mIoU | 二值 IoU | precision | recall | 预测占用率 | GT 占用率 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, label in [('with_set_loss', '保留 set loss'), ('no_set_loss', '移除 set loss')]:
        d = summary['models'][name]
        for mode, shift in [('默认', 0.), ('固定 +4', 4.), ('训练集选定', d['selected_shift'])]:
            m = d['validation_sweep']['empty_logit_shift_metrics'][str(float(shift))]
            lines.append(f'| {label} | {mode} | {shift:+.1f} | {m["mIoU"]:.2f} | {m["IoU"]:.2f} | {m["precision"]:.2f} | {m["recall"]:.2f} | '
                         f'{m["predicted_occupied_fraction"]:.2f} | {m["gt_occupied_fraction"]:.2f} |')
    lines += ['', '## 3. 训练曲线与参数诊断', '',
        '| 模型 | epoch | camera mIoU | 二值 IoU |', '| --- | ---: | ---: | ---: |']
    for name, label in [('with_set_loss', '保留 set loss'), ('no_set_loss', '移除 set loss')]:
        for row in summary['models'][name]['training_curve']:
            lines.append(f'| {label} | {row["epoch"]} | {row["mIoU"]:.2f} | {row["IoU"]:.2f} |')
    lines += ['', '| 模型 | learned empty bias | empty scalar | occupied GT 上的前景最高分均值 | occupied GT 上的 empty 分数均值 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, label in [('with_set_loss', '保留 set loss'), ('no_set_loss', '移除 set loss')]:
        d = summary['models'][name]['validation_sweep']
        v = d['render_logit_means']
        lines.append(f'| {label} | {d["learned_empty_bias"]:.4f} | {d["empty_scalar"]:.4f} | '
                     f'{v["foreground_max_gt_occupied"]:.4f} | {v["empty_logit_gt_occupied"]:.4f} |')
    lines += ['', '上表是未叠加额外偏移的分数均值。下面仅在真实 occupied 体素上比较 17 个前景通道的 argmax，忽略 empty 是否获胜；'
              '因此该准确率不会被额外 empty 偏移改变，也不能替代完整 mIoU。', '',
              '| 模型 | GT occupied 上的前景类别准确率 |', '| --- | ---: |']
    for name, label in [('with_set_loss', '保留 set loss'), ('no_set_loss', '移除 set loss')]:
        accuracy = summary['models'][name]['validation_sweep']['foreground_class_accuracy_on_gt_occupied']
        lines.append(f'| {label} | {accuracy:.2f}% |')
    lines += ['', '<details>', '<summary>展开默认及校准输出的逐类 IoU</summary>', '',
              '| 类别 | 基线默认 | 无 set 默认 | 基线校准 | 无 set 校准 |',
              '| --- | ---: | ---: | ---: | ---: |']
    base = summary['models']['with_set_loss']
    new = summary['models']['no_set_loss']
    for category in base['raw_official']['camera_metrics']['per_class_IoU']:
        scores = [model[mode]['camera_metrics']['per_class_IoU'][category]
                  for mode, model in [('raw_official', base), ('raw_official', new),
                                      ('calibrated_official', base), ('calibrated_official', new)]]
        lines.append('| ' + category + ' | ' + ' | '.join(f'{v:.2f}' for v in scores) + ' |')
    lines += ['', '</details>']
    lines += ['', '## 4. 如何解释这组消融', '',
        '完全移除 set loss 同时取消六阶段的几何与分类监督。它回答整套辅助监督的净作用，不能单独把变化归因于分类项。'
        '原 OPUS decoder 保留 stage 间点坐标 detach；移除早期 set loss 后，早期回归分支不再获得原有的直接几何监督。模型结构和 detach 均未改变。'
        '剩余 voxel 监督范围不变，但基线 set loss 提供的相机可见域外 occupied 监督也被取消。', '',
        '此前 a5 只取消最终阶段的 set 项、保留早期 set 监督，与本次完全去除的设置不同。'
        '两组对照可用于理解早期监督的作用，不能都称作“无 OPUS set loss”。'
        '原始记录见[贡献一结果 §3](contribution1_diagnostic_results.md#3-重新训练局部冲突是否转化为最终收益)。', '',
        '额外偏移的最优值和收益用于衡量剩余决策敏感性。即使偏移能改善成绩，也不证明 Gaussian 几何正确；即使无需额外偏移，也不证明所有校准问题消失。'
        '本实验只有一个训练 seed、五轮 mini 预算，不能据此得出完整数据或收敛后的普遍结论。', '',
        '## 5. 复现与产物', '',
        '- [实验设计](no_set_loss_mini_design.md)',
        '- [执行状态](../../work_dirs/no_set_loss_ablation_mini/status.json)',
        '- [机器可读汇总](../../work_dirs/no_set_loss_ablation_mini/summary.json)',
        '- [共同训练校准子集](../../work_dirs/no_set_loss_ablation_mini/training_calibration_manifest.json)',
        '- [配置一致性检查](../../work_dirs/no_set_loss_ablation_mini/config_check.json)',
        '- [新模型训练日志](../../work_dirs/no_set_loss_ablation_mini/train/no_set_loss_seed928/console.log)', '',
        '所有原始 sweep、官方评估及日志位于 `work_dirs/no_set_loss_ablation_mini/`。官方 RayIoU 使用完整 scene origins、无 camera mask，并核对官方计数。']
    REPORT.write_text('\n'.join(lines) + '\n')


def calibration_figure(models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = [f for f in ('Noto Sans CJK SC', 'Noto Sans CJK JP') if f in fonts] + ['DejaVu Sans']
    with plt.rc_context({'font.family': family, 'font.size': 11, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path'}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for ax, (name, label) in zip(axes, [('with_set_loss', '保留六阶段 set loss'), ('no_set_loss', '完全移除 set loss')]):
            d = models[name]
            for split, title, color in [('calibration', '训练子集 256 帧', '#176a90'),
                                         ('validation_sweep', '验证集 1,024 帧', '#be5c2a')]:
                options = d[split]['empty_logit_shift_metrics']
                shifts = sorted(map(float, options))
                ax.plot(shifts, [options[str(v)]['mIoU'] for v in shifts], marker='.',
                        color=color, label=title)
            ax.axvline(0, color='#666666', linewidth=.8)
            ax.axvline(d['selected_shift'], color='#5c7840', linestyle='--',
                       label=f'训练集选定 b={d["selected_shift"]:+.0f}')
            ax.set(title=label, xlabel='额外 empty 偏移 b（正值减少 empty）', ylabel='camera mIoU（%）')
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
        folder = REPORT.parent / 'figures'
        folder.mkdir(exist_ok=True)
        fig.savefig(folder / 'no_set_loss_mini_calibration.png', dpi=180)
        fig.savefig(folder / 'no_set_loss_mini_calibration.svg')
        plt.close(fig)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    shifts = [float(i) for i in range(-8, 9)]
    calibration = {name: diagnose(name, 'train', shifts) for name in MODELS}
    if any(abs(choose_empty_shift(data)) == 8 for data in calibration.values()):
        shifts = [float(i) for i in range(-16, 17)]
        calibration = {name: diagnose(name, 'train', shifts) for name in MODELS}
    models = {}
    baseline_raw_path = ROOT / BASE_TRAIN / 'evaluation_epoch5.json'
    for name, (config, directory) in MODELS.items():
        selected = choose_empty_shift(calibration[name])
        validation = diagnose(name, 'val', shifts)
        assert calibration[name]['samples'] == 256 and validation['samples'] == 1024
        assert calibration[name]['imageset_sha256'] != validation['imageset_sha256']
        assert calibration[name]['checkpoint_sha256'] == validation['checkpoint_sha256']
        raw = read(baseline_raw_path) if name == 'with_set_loss' else None
        if raw:
            assert raw['status'] == 'complete' and raw['samples'] == 1024
            assert raw['protocol']['py_config'] == config and raw['protocol']['checkpoint'] == directory + '/epoch_5.pth'
            assert raw['streaming_official_parity_passed']
        else:
            raw = official(name, 0.)
        calibrated = official(name, selected, raw=raw)
        for shift, evaluation in [(0., raw), (selected, calibrated)]:
            assert evaluation['samples'] == 1024 and evaluation['streaming_official_parity_passed']
            reference = validation['empty_logit_shift_metrics'][str(shift)]
            for key in ('mIoU', 'IoU', 'precision', 'recall'):
                if abs(evaluation['camera_metrics'][key] - reference[key]) > 1e-8:
                    raise ValueError(f'Sweep and official entry disagree for {name}/{shift}/{key}')
        models[name] = dict(selected_shift=selected, calibration=calibration[name], validation_sweep=validation,
            raw_official=raw, calibrated_official=calibrated,
            training_curve=parse_log(sorted((ROOT / directory).glob('20*.log'))),
            training_runtime=read(ROOT / directory / 'training_runtime.json'))
        write_json(RUN / 'partial_summary.json', dict(status='running', models=models))
    delta = {key: models['no_set_loss']['raw_official']['camera_metrics'][key] - models['with_set_loss']['raw_official']['camera_metrics'][key]
             for key in ('mIoU', 'IoU', 'precision', 'recall')}
    delta['RayIoU'] = models['no_set_loss']['raw_official']['ray_metrics']['RayIoU'] - models['with_set_loss']['raw_official']['ray_metrics']['RayIoU']
    calibrated_delta = {key: models['no_set_loss']['calibrated_official']['camera_metrics'][key] - models['with_set_loss']['calibrated_official']['camera_metrics'][key]
                        for key in ('mIoU', 'IoU', 'precision', 'recall')}
    calibrated_delta['RayIoU'] = models['no_set_loss']['calibrated_official']['ray_metrics']['RayIoU'] - models['with_set_loss']['calibrated_official']['ray_metrics']['RayIoU']
    summary = dict(status='complete', finished_at=datetime.now().astimezone().isoformat(), models=models,
        protocol=dict(epochs=5, seed=928, accumulation=4, calibration_frames=256, validation_frames=1024,
            extra_shift_candidates=shifts, calibration_selection='Training-only mIoU, then smallest absolute shift',
            training_selection_manifest='work_dirs/no_set_loss_ablation_mini/training_calibration_manifest.json'),
        raw_delta_no_set_minus_baseline=delta,
        calibrated_delta_no_set_minus_baseline=calibrated_delta,
        calibration_figure=True,
        source_sha256={f: file_sha256(ROOT / f) for f in ['tools/finish_no_set_loss_ablation.py',
            'tools/diagnose_opus_v1_gaussian.py','tools/eval_center_support.py',NEW_CONFIG]})
    calibration_figure(models)
    write_report(summary)
    write_json(RUN / 'summary.json', summary)
    write_json(RUN / 'evaluation_status.json', dict(status='complete', finished_at=summary['finished_at']))
    print(json.dumps({'delta': delta, 'selected_shifts':{k:v['selected_shift'] for k,v in models.items()}}, indent=2))


if __name__ == '__main__':
    main()
