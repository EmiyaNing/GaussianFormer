"""Report the complete bounded center-only GT-assisted diagnostic."""
from pathlib import Path
import csv
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.eval_gaussian_center_oracle import (
    RUN, HISTORY, MODELS, STEPS, TRANSITIONS, MOVEMENT, CLASS_NAMES, CLASS_COLUMNS)
from tools.report_gaussian_sizes import performance
from tools.stat_boundary_gaussians import read, write, sha

AGG = RUN / 'aggregate'
REPORT = ROOT / 'analysis/gaussian_center_oracle_full_val_results.md'
FIGURES = ROOT / 'analysis/figures/gaussian_center_oracle_full_val'


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(cm, cc):
    result = performance(cm)
    for j, name in enumerate(CLASS_COLUMNS[1:], 1):
        result[name] = float(100 * cc[:, j].sum() / cc[:, 0].sum())
    return result


def aggregate(size):
    folder = RUN / size
    protocol = read(folder / 'protocol.json')
    complete = read(folder / 'complete.json')
    assert complete['status'] == 'complete' and complete['frames'] == 6019 and complete['scenes'] == 150
    assert complete['checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert sha(ROOT / protocol['checkpoint']) == protocol['checkpoint_sha256']
    assert read(folder / 'pilot.json')['status'] == 'passed'
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    arrays, indices, tokens, records = {}, [], [], []
    for path in sorted((folder / 'scenes').glob('*.json')):
        meta = read(path)
        assert meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
        indices.extend(meta['indices'])
        tokens.extend(meta['tokens'])
        with np.load(path.with_suffix('.npz')) as z:
            assert z['confusion'].shape == (len(meta['indices']), 4, 18, 18)
            assert z['class_coverage'].shape == (len(meta['indices']), 4, 17, 8)
            assert np.isfinite(z['loss_trajectory']).all() and np.isfinite(z['movement']).all()
            assert (z['audit'][:, 0] == 0).all()
            assert (z['movement'][..., 3] <= .40001).all()
            assert (z['movement'][:, 0, 1:] == 0).all()
            np.testing.assert_array_equal(z['transitions'][:, 0], 0)
            for name in z.files:
                arrays.setdefault(name, []).append(z[name])
            records.append(dict(scene=meta['scene'], frames=meta['frames'],
                confusion=z['confusion'].sum(0), class_coverage=z['class_coverage'].sum(0),
                transitions=z['transitions'].sum(0)))
    assert len(records) == 150 and sorted(indices) == list(range(6019)) and len(set(tokens)) == 6019
    order = np.argsort(indices)
    result = {name:np.concatenate(values)[order] for name, values in arrays.items()}
    result['frame_indices'] = np.array(indices)[order]
    cm = result['confusion'].sum(0)
    cc = result['class_coverage'].sum(0)
    trans = result['transitions'].sum(0)
    assert np.all(cc[:, :, 1] <= cc[:, :, 2]) and np.all(cc[:, :, 2] <= cc[:, :, 3])
    assert np.all(cc[:, :, 4:7] <= cc[:, :, 1:4])
    for j in range(4):
        np.testing.assert_array_equal(cm[j, :17].sum(1), cc[j, :, 0])
        np.testing.assert_array_equal(cc[j, :, 0], cc[0, :, 0])
        assert cm[0, :17, 17].sum() - cm[j, :17, 17].sum() == trans[j, 0] - trans[j, 1]
        assert cm[0, 17, :17].sum() - cm[j, 17, :17].sum() == trans[j, 2] - trans[j, 3]
        assert np.trace(cm[j]) - np.trace(cm[0]) == trans[j, 4] - trans[j, 5]
    with np.load(HISTORY / 'aggregate' / f'{size}.npz') as old:
        np.testing.assert_array_equal(cm[0], old['confusion'])
        np.testing.assert_array_equal(cc[0], old['class_coverage'])
    rows = []
    for j, step in enumerate(STEPS):
        move = result['movement'][:, j]
        metrics = describe(cm[j], cc[j])
        metrics.update(model=size, step=step,
            mean_displacement_m=float(move[:, 1].sum() / move[:, 0].sum()),
            rms_displacement_m=float(np.sqrt(move[:, 2].sum() / move[:, 0].sum())),
            max_displacement_m=float(move[:, 3].max()),
            moving_GS_percent=float(100 * move[:, 4].sum() / move[:, 0].sum()),
            radius_GS_percent=float(100 * move[:, 5].sum() / move[:, 0].sum()),
            mean_total_loss=float(result['loss_trajectory'][:, step, 0].mean()),
            mean_weighted_CE=float(result['loss_trajectory'][:, step, 1].mean()),
            mean_weighted_Lovasz=float(result['loss_trajectory'][:, step, 2].mean()),
            **{name:int(trans[j, k]) for k, name in enumerate(TRANSITIONS)})
        rows.append(metrics)
    return protocol, result, rows, records, np.asarray(tokens)[order].tolist()


def intervals(records):
    weights = np.random.default_rng(928).multinomial(150, np.full(150, 1/150), size=1500)
    cm = np.stack([r['confusion'] for r in records])
    cc = np.stack([r['class_coverage'] for r in records])
    samples = []
    for j in range(4):
        c = (weights @ cm[:, j].reshape(150, -1)).reshape(-1, 18, 18)
        p = np.diagonal(c, axis1=1, axis2=2)[:, :17]
        gt = c[:, :17].sum(2)
        union = gt + c.sum(1)[:, :17] - p
        ciou = np.divide(100*p, union, out=np.zeros_like(p, float), where=union > 0)
        btp = c[:, :17, :17].sum((1, 2))
        cover = weights @ cc[:, j][:, :, [0, 1]].sum(1)
        samples.append(dict(mIoU=ciou.mean(1),
            IoU=100 * btp / (btp+c[:, :17, 17].sum(1)+c[:, 17, :17].sum(1)),
            geo1=100 * cover[:, 1] / cover[:, 0],
            FN_rate=100 * c[:, :17, 17].sum(1) / gt.sum(1),
            FP_over_GTfg=100 * c[:, 17, :17].sum(1) / gt.sum(1)))
    return {str(STEPS[j]):{k:np.quantile(samples[j][k]-samples[0][k], [.025, .975]).tolist()
                          for k in samples[j]} for j in range(1, 4)}


def class_rows(size, arrays):
    cm = arrays['confusion'].sum(0)
    cc = arrays['class_coverage'].sum(0)
    result = []
    for j, step in enumerate(STEPS):
        for i, name in enumerate(CLASS_NAMES):
            gt = int(cm[j, i].sum())
            tp = int(cm[j, i, i])
            fp_empty = int(cm[j, 17, i])
            fp_other = int(cm[j, :17, i].sum()) - tp
            result.append(dict(model=size, step=step, class_id=i, class_name=name,
                GT=gt, TP=tp, FP_empty=fp_empty, FP_other_foreground=fp_other,
                FN_to_empty=int(cm[j, i, 17]), IoU=100*tp/(gt+fp_empty+fp_other),
                **{k:100*int(cc[j, i, n])/gt for n,k in enumerate(CLASS_COLUMNS[1:], 1)}))
    return result


def figures(metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'font.family':'DejaVu Sans'})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout='constrained')
    for ax, key, title in zip(axes, ['mIoU', 'IoU', 'geo1', 'mean_total_loss'],
                            ['mIoU', 'Binary IoU', '1-sigma coverage', 'Mean objective / initial']):
        for size, color in zip(MODELS, ['#4477aa', '#dd9955']):
            values = np.array([row[key] for row in metrics[size]])
            if key == 'mean_total_loss':
                values /= values[0]
            ax.plot(STEPS, values, 'o-', label=size, color=color)
        ax.set(title=title, xlabel='Center updates', xticks=STEPS)
        ax.set_ylabel('%' if key != 'mean_total_loss' else 'Ratio')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Same-frame GT-assisted center refinement | 6,019 frames | fixed radius 0.4 m')
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES/f'center_refinement.{ext}', dpi=180)
    plt.close(fig)


def report(metrics, classes, ci, arrays):
    lines = ['# 固定语义，只移动高斯中心：完整验证结果', '',
        '**Tiny 与 Small 均完成完整验证：各 150 个场景、6,019 帧。** 每帧使用自己的 GT，只调整最终 GS 中心，最多移动 0.4 米；语义、不透明度、尺度、旋转和 empty 参数固定。', '',
        '第 20 步的 mIoU：'+'；'.join(f"**{s.capitalize()} {metrics[s][0]['mIoU']:.3f}% → {metrics[s][-1]['mIoU']:.3f}%（{metrics[s][-1]['mIoU']-metrics[s][0]['mIoU']:+.3f} 个百分点）**" for s in MODELS)+'。', '',
        '**这是 GT 辅助的几何诊断，不能作为正常推理成绩。** 它判断现有输出附近是否存在可利用的位置调整，不检验网络是否能预测这些调整，也不是几何优化的全局上限。执行约定见[设计文档](gaussian_center_oracle_full_val_design.md)。', '',
        '## 1. 中心位置调整带来了多少收益', '',
        '第 20 步为预先规定的主要结果。表内数值为百分比，变化为百分点；各步均从整个验证集汇总混淆计数后计算。', '',
        '| 模型 | 更新步数 | mIoU | ΔmIoU | 二值IoU | Δ二值IoU | 1σ覆盖 | Δ1σ覆盖 | 同类1σ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for size in MODELS:
        base = metrics[size][0]
        for row in metrics[size]:
            lines.append(f"| {size} | {row['step']} | {row['mIoU']:.3f}% | {row['mIoU']-base['mIoU']:+.3f} | {row['IoU']:.3f}% | {row['IoU']-base['IoU']:+.3f} | {row['geo1']:.3f}% | {row['geo1']-base['geo1']:+.3f} | {row['same1']:.3f}% |")
    lines += ['', '![中心调整过程中的性能、覆盖与损失](figures/gaussian_center_oracle_full_val/center_refinement.png)', '',
        '| 第20步减第0步 | ΔmIoU的95%场景区间 | Δ二值IoU的95%场景区间 | Δ1σ覆盖的95%场景区间 |',
        '|---|---|---|---|']
    for size in MODELS:
        fmt = lambda k:'['+', '.join(f'{v:+.3f}' for v in ci[size]['20'][k])+']'
        lines.append(f"| {size} | {fmt('mIoU')} | {fmt('IoU')} | {fmt('geo1')} |")
    lines += ['', '区间来自相同 150 个场景的配对 bootstrap，重采样 1,500 次、种子 928；不表示训练随机性。', '',
        '## 2. 漏检、假阳性与类别区分', '',
        '漏检和二值FP都以 GT 前景体素数为分母。类别准确率仅在 GT 前景且预测为前景的体素中计算；类别错分仍算作二值检出。', '',
        '| 模型 | 步数 | 漏检/GT前景 | 二值FP/GT前景 | 二值FP/预测前景 | 已检出前景类别准确率 |',
        '|---|---:|---:|---:|---:|---:|']
    for size in MODELS:
        for row in (metrics[size][0], metrics[size][-1]):
            lines.append(f"| {size} | {row['step']} | {row['FN_rate']:.3f}% | {row['FP_over_GTfg']:.3f}% | {row['FP_over_predfg']:.3f}% | {row['detected_semantic_accuracy']:.3f}% |")
    lines += ['', '下面统计同一批体素在第 0 步与第 20 步之间的转变，而不是只比较净变化。救回漏检只要求从 empty 变为任意前景，不保证类别正确。', '',
        '| 模型 | 救回漏检 | 新增漏检 | 消除二值FP | 新增二值FP | 纠正前景类别错分 | 新增前景类别错分 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for size in MODELS:
        r = metrics[size][-1]
        lines.append('| '+size+' | '+' | '.join(f"{r[k]:,}" for k in
            ('rescued_FN','new_FN','removed_binary_FP','new_binary_FP',
             'corrected_foreground_class','damaged_foreground_class'))+' |')
    lines += ['', '最后两列仅统计调整前后都预测为前景的 GT 前景体素：错误前景类别变成正确类别，或正确类别变成错误前景类别。', '',
        '## 3. 逐类IoU与核心覆盖', '',
        '每格为“IoU / 1σ几何覆盖率”，单位为%；变化为百分点，先使用原始值作差再取三位小数。同类覆盖及 2σ、3σ 和原生覆盖见逐类 CSV。', '']
    for size in MODELS:
        lines += [f'### {size.capitalize()}', '',
            '| 类别 | 第0步 IoU / 覆盖 | 第20步 IoU / 覆盖 | ΔIoU / Δ覆盖 |',
            '|---|---:|---:|---:|']
        data = {step:{r['class_id']:r for r in classes if r['model']==size and r['step']==step} for step in (0,20)}
        for i, name in enumerate(CLASS_NAMES):
            b, r = data[0][i], data[20][i]
            lines.append(f"| {name} | {b['IoU']:.3f} / {b['geo1']:.3f} | {r['IoU']:.3f} / {r['geo1']:.3f} | {r['IoU']-b['IoU']:+.3f} / {r['geo1']-b['geo1']:+.3f} |")
        both = sum(data[20][i]['IoU']>data[0][i]['IoU'] and data[20][i]['geo1']>data[0][i]['geo1'] for i in range(17))
        iou_up = sum(data[20][i]['IoU']>data[0][i]['IoU'] for i in range(17))
        cover_up = sum(data[20][i]['geo1']>data[0][i]['geo1'] for i in range(17))
        lines += ['', f'17 类中，IoU 提高 {iou_up} 类，覆盖提高 {cover_up} 类，两者同时提高 {both} 类。', '']
    lines += ['## 4. 实际移动了多少，优化是否正常', '',
        '| 模型 | 第20步平均位移 | 最大位移 | 移动GS比例 | 到达0.4米上限 | 第0→20步平均总损失 |',
        '|---|---:|---:|---:|---:|---:|']
    for size in MODELS:
        b, r = metrics[size][0], metrics[size][-1]
        lines.append(f"| {size} | {r['mean_displacement_m']:.4f} m | {r['max_displacement_m']:.4f} m | {r['moving_GS_percent']:.3f}% | {r['radius_GS_percent']:.3f}% | {b['mean_total_loss']:.5f} → {r['mean_total_loss']:.5f} |")
    lines += ['', '位移比例以每帧全部 76,800 个预测 GS 为分母；移动指位移大于 0.00001 米。平均损失按帧等权，不是汇总混淆矩阵导出的指标。达到上限使用 0.00001 米数值容差。', '',
        '## 5. 这组结果能说明什么', '']
    for size in MODELS:
        b, r = metrics[size][0], metrics[size][-1]
        lines.append(f"{size.capitalize()} 的 mIoU 变化为 **{r['mIoU']-b['mIoU']:+.3f}** 个百分点，二值 IoU 为 **{r['IoU']-b['IoU']:+.3f}**，1σ覆盖为 **{r['geo1']-b['geo1']:+.3f}**。已检出前景的类别准确率从 **{b['detected_semantic_accuracy']:.3f}%** 变为 **{r['detected_semantic_accuracy']:.3f}%**。这些变化只来自中心移动；GS 的类别得分虽然固定，空间叠加关系变化仍会改变最终体素类别。")
        lines.append('')
    if all(ci[s]['20']['mIoU'][0] > 0 for s in MODELS):
        lines += ['**两模型的中心调整均带来正向 mIoU 收益，配对场景区间均高于零。** 因此，在固定现有语义与形状的条件下，当前位置附近仍存在可以改善预测的位置调整。这个实验直接改变了中心，结论比“漏检与低覆盖相关”更进一步。', '']
    else:
        lines += ['需结合上表的实际变化与场景区间判断中心调整是否稳定有效，不能把损失下降自动等同于 mIoU 收益。', '']
    paired = {(r['model'], r['step'], r['class_id']):r for r in classes}
    tally = []
    for size in MODELS:
        up_iou = sum(paired[size,20,i]['IoU']>paired[size,0,i]['IoU'] for i in range(17))
        up_core = sum(paired[size,20,i]['geo1']>paired[size,0,i]['geo1'] for i in range(17))
        tally.append(f'{size.capitalize()}：{up_iou}/17 类 IoU 提高，{up_core}/17 类核心覆盖提高')
    b, r = paired['tiny',0,2], paired['tiny',20,2]
    lines += ['；'.join(tally)+'。'+f"例如 Tiny 的 bicycle，IoU 从 **{b['IoU']:.3f}%** 提高到 **{r['IoU']:.3f}%**，1σ覆盖反而从 **{b['geo1']:.3f}%** 降到 **{r['geo1']:.3f}%**。这表明，即使固定语义、只改变位置，核心覆盖率也不能单独判断某类预测是否改善。", '']
    lines += ['这与“不同模型的整体覆盖率不能直接预测性能”并不矛盾。此前跨模型对比同时改变了语义、形状和网络能力；本实验固定这些输出，只改变位置。位置调整会同时改变正确证据、错误类别贡献、empty 竞争与相邻体素上的扩散，因此不能把本轮收益全部归因于覆盖率变化。', '',
        '**目前仍不能据此认定几何是原训练的主导瓶颈，也不能认定固定每query的GS数量造成了本轮错误。** 实验没有测量语义独立优化的收益，也没有改变任何query的GS配额；它确认的是当前输出的位置改进空间。', '',
        '本实验使用每帧自己的 GT、允许每个 GS 独立移动，条件比可部署方法更宽松。它不证明某个中心监督、Set loss 或 query 重定位结构能够学会这些调整。', '',
        '## 6. 完整性与复现入口', '',
        '- Tiny 与 Small 分别覆盖完整 150 个场景、6,019 帧，帧索引与 token 完整且一致。',
        '- 每帧第 0 步渲染与原模型 camera 筛选后的 logits 逐元素一致；混淆计数和覆盖计数逐帧复现此前全量统计。',
        '- 原生 CUDA 中心梯度通过有限差分检查，Occupancy 目标与原实现一致；全部损失、梯度有限，中心位移没有超过约定范围。',
        '- 固定 empty 输出的缓存与完整原生反传得到逐元素一致的中心梯度；前 80 帧的全部损失轨迹及四次记录的指标，与未缓存版本逐元素一致。',
        '- 语义、尺度、旋转、不透明度保持原值；网络参数没有接收梯度，优化器仅持有位移参数。原权重与历史统计文件没有被覆盖。',
        '- 每个记录步数的漏检、FP与正确预测净变化，均和对应体素转变计数精确一致。', '',
        '| 文件 | 用途 |', '|---|---|',
        '| [整体指标](../work_dirs/gaussian_center_oracle_full_val_v1/aggregate/metrics.csv) | 两模型各四个记录步数的完整性能、覆盖、位移与误差转变 |',
        '| [逐类指标](../work_dirs/gaussian_center_oracle_full_val_v1/aggregate/per_class.csv) | 各类别IoU、覆盖和TP/FP/FN |',
        '| [逐类变化](../work_dirs/gaussian_center_oracle_full_val_v1/aggregate/per_class_differences.csv) | 第5、10、20步相对第0步的变化 |',
        '| [逐场景指标](../work_dirs/gaussian_center_oracle_full_val_v1/aggregate/per_scene.csv) | 完整场景上的配对结果 |',
        '| [优化轨迹](../work_dirs/gaussian_center_oracle_full_val_v1/aggregate/loss_trajectory.csv) | 第0–20步的CE、Lovasz和总损失 |',
        '| [汇总JSON](../work_dirs/gaussian_center_oracle_full_val_v1/summary.json) | 精确结果和场景置信区间 |',
        '| [完整性检查](../work_dirs/gaussian_center_oracle_full_val_v1/delivery_audit.json) | 覆盖范围、原生复现与文件校验 |', '',
        '```bash', 'python3.10 tools/eval_gaussian_center_oracle.py --action pilot',
        'python3.10 tools/eval_gaussian_center_oracle.py --action run --workers 3',
        'python3.10 tools/report_gaussian_center_oracle.py', '```', '']
    REPORT.write_text('\n'.join(lines))


def main():
    AGG.mkdir(parents=True, exist_ok=True)
    parity = read(RUN/'native_renderer_parity.json')
    assert parity['status'] == 'passed' and parity['frames'] == 80 and parity['all_arrays_bitwise_exact']
    metrics, arrays, all_classes, intervals_all = {}, {}, [], {}
    scene_rows, trajectories, differences = [], [], []
    tokens0 = None
    protocols = {}
    for size in MODELS:
        protocol, a, rows, records, tokens = aggregate(size)
        protocols[size] = protocol
        if tokens0 is None:
            tokens0 = tokens
        assert tokens0 == tokens
        arrays[size], metrics[size] = a, rows
        with (AGG/f'{size}.npz').open('wb') as stream:
            np.savez_compressed(stream, **a)
        classes = class_rows(size, a)
        all_classes += classes
        base = {r['class_id']:r for r in classes if r['step']==0}
        for r in classes:
            if r['step']:
                differences.append(dict(model=size, step=r['step'], class_name=r['class_name'],
                    **{f'delta_{k}':r[k]-base[r['class_id']][k] for k in ('IoU', *CLASS_COLUMNS[1:])}))
        intervals_all[size] = intervals(records)
        for record in records:
            for j, step in enumerate(STEPS):
                scene_rows.append(dict(model=size, scene=record['scene'], frames=record['frames'], step=step,
                    **describe(record['confusion'][j], record['class_coverage'][j])))
        for step in range(21):
            loss = a['loss_trajectory'][:, step]
            trajectories.append(dict(model=size, step=step, total_loss_mean=float(loss[:, 0].mean()),
                weighted_CE_mean=float(loss[:, 1].mean()), weighted_Lovasz_mean=float(loss[:, 2].mean()),
                frames_loss_above_initial=int((loss[:, 0]>a['loss_trajectory'][:, 0, 0]).sum())))
    csv_write(AGG/'metrics.csv', [r for size in MODELS for r in metrics[size]])
    csv_write(AGG/'per_class.csv', all_classes)
    csv_write(AGG/'per_class_differences.csv', differences)
    csv_write(AGG/'per_scene.csv', scene_rows)
    csv_write(AGG/'loss_trajectory.csv', trajectories)
    summary = dict(status='complete', frames_per_model=6019, scenes_per_model=150,
        metrics=metrics, paired_scene_intervals=intervals_all,
        checkpoint_sha256={s:protocols[s]['checkpoint_sha256'] for s in MODELS})
    write(RUN/'summary.json', summary)
    figures(metrics)
    report(metrics, all_classes, intervals_all, arrays)
    paths = [REPORT, ROOT/'analysis/gaussian_center_oracle_full_val_design.md',
             Path(__file__), ROOT/'tools/eval_gaussian_center_oracle.py',
             RUN/'summary.json', RUN/'native_renderer_parity.json',
             *[RUN/s/n for s in MODELS for n in ('protocol.json','pilot.json','complete.json')],
             *sorted(AGG.glob('*')), *sorted(FIGURES.glob('*'))]
    write(RUN/'delivery_audit.json', dict(status='passed', frames_per_model=6019, scenes_per_model=150,
        exact_baseline_per_frame=True, no_network_gradients=True,
        finite_losses_and_gradients=True, transitions_reconcile=True,
        input_and_source_hashes_verified=True, models_frame_tokens_exact=True,
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print(json.dumps(summary, indent=2), flush=True)
    print('Full report and artifact checks passed.', flush=True)


if __name__ == '__main__':
    main()
