"""Aggregate the matched 4090-frame intervention and full-validation diagnosis."""
from datetime import datetime
from pathlib import Path
import csv
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.all_layer_render_experiment import RUN, ARMS, assert_protocol, write, sha
from tools.decoder_layer_stats.common import DISTANCES, COUNTS, GROUPS, SUPPORTS, quintiles
from tools.report_decoder_layers import ratios, save_csv, table
from tools.report_gaussian_sizes import performance
from tools.eval_gaussian_sizes import CLASS_NAMES

MODELS = ('initial', *ARMS)
REPORT = ROOT / 'analysis/all_layer_render_small_mini_results.md'
AGG = RUN / 'aggregate'
FIG = ROOT / 'analysis/figures/all_layer_render_small_mini'


def aggregate(arm):
    folder = RUN / 'eval' / arm
    complete = json.loads((folder / 'complete.json').read_text())
    assert complete['frames'] == 6019 and complete['scenes'] == 150
    assert json.loads((folder / 'pilot.json').read_text())['status'] == 'passed'
    totals, per_scene, identities, scene_cm, scene_fg, scene_unions = {}, [], {}, [], [], []
    cm = np.zeros((6, 18, 18), np.int64)
    cc = np.zeros((6, 17, 5), np.int64)
    attrs = np.zeros((6, 6), np.float64)
    files = sorted((folder / 'scenes').glob('*.json'))
    assert len(files) == 150
    for path in files:
        meta = json.loads(path.read_text())
        assert meta['checkpoint_sha256'] == complete['checkpoint_sha256']
        assert sha(path.with_suffix('.npz')) == meta['npz_sha256']
        for index, token in zip(meta['indices'], meta['tokens']):
            assert index not in identities
            identities[index] = token
        with np.load(path.with_suffix('.npz')) as z:
            d, c, u, fg, load, groups = [z[key] for key in
                ('query_distances', 'query_GS_counts', 'group_unions', 'foreground', 'initial_load', 'load_groups')]
            n = len(fg)
            assert d.shape == c.shape == (n, 6, 1200, 4)
            assert np.isfinite(d).all() and np.all(d >= 0)
            np.testing.assert_array_equal(groups, np.stack([quintiles(row) for row in load]))
            np.testing.assert_array_equal(load.sum(1), fg)
            this_cm = z['confusion'].sum(0)
            this_cc = z['class_coverage'].sum(0)
            np.testing.assert_array_equal(this_cm[:, :17].sum(2), this_cc[:, :, 0])
            np.testing.assert_array_equal(this_cc[:, :, 1:].sum(1), u[:, :, 0].sum(0))
            assert np.all(u[:, :, 1:].max(2) <= u[:, :, 0])
            assert np.all(u[:, :, 1:].sum(2) >= u[:, :, 0])
            cm += this_cm
            cc += this_cc
            attrs += z['gaussian_attributes'].sum(0)
            scene_cm.append(this_cm)
            scene_fg.append(int(fg.sum()))
            scene_unions.append(u[:, :, 0].sum(0))
            for layer, k in enumerate((1, 4, 8, 16, 32, 64)):
                assert np.all(c[:, layer, :, 1] <= c[:, layer, :, 0]) and np.all(c[:, layer, :, :2] <= k)
                for g, group in enumerate(GROUPS):
                    keep = np.ones((n, 1200), bool) if g == 0 else groups == g-1
                    assert np.all(keep.sum(1) == (1200 if g == 0 else 240))
                    row = dict(model=arm, layer=layer+1, group=group, children_per_query=k,
                        frames=n, queries=int(keep.sum()), GS=int(keep.sum())*k,
                        foreground=int(fg.sum()), assigned_foreground=int(load[keep].sum()))
                    for j, key in enumerate(DISTANCES):
                        row[key+'_sum'] = float(d[:, layer, :, j][keep].sum(dtype=np.float64))
                    for j, key in enumerate(COUNTS):
                        row[key] = int(c[:, layer, :, j][keep].sum(dtype=np.int64))
                    for j, key in enumerate(SUPPORTS):
                        row[key+'_union'] = int(u[:, layer, g, j].sum(dtype=np.int64))
                    key = (layer+1, group)
                    if key not in totals:
                        totals[key] = dict(row)
                    else:
                        entry = totals[key]
                        for key, value in row.items():
                            if key not in ('model', 'layer', 'group', 'children_per_query'):
                                entry[key] += value
                    per_scene.append(dict(scene=meta['scene'], **ratios(row)))
    assert sorted(identities) == list(range(6019)) and len(set(identities.values())) == 6019
    rows = [ratios(totals[(i, group)]) for i in range(1, 7) for group in GROUPS]
    for layer in range(1, 7):
        these = [r for r in rows if r['layer'] == layer]
        for key in ('queries', 'GS', 'assigned_foreground', *COUNTS, *[k+'_sum' for k in DISTANCES]):
            np.testing.assert_allclose(these[0][key], sum(r[key] for r in these[1:]), rtol=1e-12, atol=1e-8)
        assert these[0]['frames'] == 6019
    performances, classes = [], []
    for layer in range(6):
        performances.append(dict(model=arm, layer=layer+1, **performance(cm[layer]),
            mean_axis_sigma_m=attrs[layer, 1]/(3*attrs[layer, 0]),
            mean_major_sigma_m=attrs[layer, 2]/attrs[layer, 0],
            mean_opacity=attrs[layer, 3]/attrs[layer, 0],
            major_sigma_saturated_pct=100*attrs[layer, 5]/attrs[layer, 0]))
        for j, name in enumerate(CLASS_NAMES):
            gt, tp = int(cm[layer, j].sum()), int(cm[layer, j, j])
            fp = int(cm[layer, :, j].sum()) - tp
            classes.append(dict(model=arm, layer=layer+1, class_id=j, class_name=name,
                GT=gt, TP=tp, FP=fp, FN_to_empty=int(cm[layer, j, 17]),
                IoU=100*tp/(gt+fp) if gt+fp else 0.,
                **{key+'_pct': 100*int(cc[layer, j, s+1])/gt if gt else 0. for s, key in enumerate(SUPPORTS)}))
    return dict(rows=rows, performance=performances, classes=classes, per_scene=per_scene,
        identities=identities, scene_cm=np.stack(scene_cm), scene_fg=np.array(scene_fg),
        scene_unions=np.stack(scene_unions), confusion=cm)


def bootstrap(a, b):
    rng = np.random.default_rng(928)
    values = []
    for _ in range(1000):
        ids = rng.integers(0, 150, 150)
        delta = performance(b['scene_cm'][ids, -1].sum(0))['mIoU'] - performance(a['scene_cm'][ids, -1].sum(0))['mIoU']
        coverage = 100*(b['scene_unions'][ids, -1, 0].sum() - a['scene_unions'][ids, -1, 0].sum())/a['scene_fg'][ids].sum()
        values.append((delta, coverage))
    limits = np.quantile(values, [.025, .975], axis=0)
    return dict(mIoU_delta_pp_95=limits[:, 0].tolist(), core_coverage_delta_pp_95=limits[:, 1].tolist())


def figures(rows, performances):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = ('mIoU', DISTANCES[1], 'core_1sigma_pct_foreground', 'same_class_core_hit_GS_pct_all_GS')
    labels = ('mIoU (%)', 'Query reference distance (voxels)', 'Foreground 1sigma coverage (%)', 'Correct core-hit GS / all GS (%)')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, label in zip(axes.flat, metrics, labels):
        for arm in MODELS:
            source = performances if key == 'mIoU' else rows
            data = [r for r in source if r['model'] == arm and r.get('group', 'all') == 'all']
            ax.plot(range(1, 7), [r[key] for r in data], marker='o', label=arm)
        ax.set(xlabel='Decoder layer', ylabel=label, xticks=range(1, 7))
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / ('layers.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(axes, metrics[1:], labels[1:]):
        for arm in MODELS:
            data = [r for r in rows if r['model'] == arm and r['layer'] == 6 and r['group'] != 'all']
            ax.plot(range(5), [r[key] for r in data], marker='o', label=arm)
        ax.set(xticks=range(5), xticklabels=GROUPS[1:], ylabel=label, xlabel='Initial final-load rank, highest first')
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / ('load_groups.'+ext), dpi=180)
    plt.close(fig)


def document(rows, performances, training, uncertainty):
    final = {r['model']: r for r in performances if r['layer'] == 6}
    geom = {r['model']: r for r in rows if r['layer'] == 6 and r['group'] == 'all'}
    delta = final['all_layers']['mIoU'] - final['baseline']['mIoU']
    ci = uncertainty['all_layers_minus_baseline']['mIoU_delta_pp_95']
    lines = ['# Small：六层完整GS与等权Occupancy监督结果', '',
        '**已完成：两组各4,090帧、1epoch、2,045次更新；原始权重与两组最终权重均完成全部6,019帧、150场景的六层验证。**', '',
        f"与同预算原结构对照相比，六层方案的最终mIoU变化为 **{delta:+.3f}个百分点**，场景配对重采样95%区间为[{ci[0]:+.3f}, {ci[1]:+.3f}]。", '',
        '[实验设计](all_layer_render_small_mini_design.md)解释初始化、监督强度和训练范围。六层OPUS Set Loss各权重1；六层Occupancy各使用原CE×10＋Lovasz×1，直接相加，不除以6。对照组仅末层学习GS属性并接受Occupancy监督。', '',
        '新增第1–5层属性头按原构造初始化；第六层及其他预训练参数精确保留。两组使用新的同配置AdamW、相同4,090帧和增强，全部训练帧输入指纹逐项一致。验证不选模、不校准。', '',
        '| 模型 | 最终mIoU | 二值IoU | 参考中心距离（格） | 1σ前景覆盖 | 同类1σ前景覆盖 | 正确GS/全部GS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in MODELS:
        p, g = final[arm], geom[arm]
        lines.append(f"| {arm} | {p['mIoU']:.3f}% | {p['IoU']:.3f}% | {g[DISTANCES[1]]:.3f} | {g['core_1sigma_pct_foreground']:.3f}% | {g['same_class_core_1sigma_pct_foreground']:.3f}% | {g['same_class_core_hit_GS_pct_all_GS']:.3f}% |")
    lines += ['', '![六层总体变化](figures/all_layer_render_small_mini/layers.png)', '',
        '![固定初始负载五组](figures/all_layer_render_small_mini/load_groups.png)', '',
        '距离为到camera有效GT前景体素立方体并集的精确欧氏距离，1格=0.4米。参考中心由原始未clamp的query点解码后平均；子GS距离是分别求每个GS中心距离后的均值。', '',
        '前景覆盖为GS和query去重后的并集/全部可见GT前景。正确GS表示核心内至少命中一个预测同类别的GT前景体素，分母包含该层全部输出GS；另报同类核心体素覆盖，不能将两者混用。固定原始Small的最终层负载排名，将query从高到低五等分，包含零负载；各组在模型和层之间保持相同query ID。组覆盖可以重叠，不能相加。', '',
        '原始模型和对照模型前五层仍是固定尺度0.2米、单位旋转、opacity0.1的诊断输出；六层方案则每层均学习这些属性。这里的差值反映新增属性头与新增渲染监督的共同效果。', '']
    for arm in MODELS:
        lines += [f'## {arm}：各层性能与属性', '',
            '| 层 | mIoU | 二值IoU | 平均轴σ（米） | 平均opacity | 主轴尺度饱和GS |', '|---|---:|---:|---:|---:|---:|']
        for p in performances:
            if p['model'] == arm:
                lines.append(f"| {p['layer']} | {p['mIoU']:.3f}% | {p['IoU']:.3f}% | {p['mean_axis_sigma_m']:.4f} | {p['mean_opacity']:.4f} | {p['major_sigma_saturated_pct']:.3f}% |")
        lines += ['', table([r for r in rows if r['model'] == arm and r['group'] == 'all'], True), '']
        for group in GROUPS[1:]:
            lines += [f'### {arm}：初始负载排名{group}', '',
                table([r for r in rows if r['model'] == arm and r['group'] == group]), '']
    lines += ['## 训练与核验', '', '| 组别 | 帧数 | 更新次数 | 训练时间（分钟） | 峰值显存（GB） |', '|---|---:|---:|---:|---:|']
    for arm, t in training.items():
        lines.append(f"| {arm} | {t['frames']} | {t['optimizer_updates']} | {t['seconds_this_run']/60:.2f} | {t['peak_cuda_GB']:.3f} |")
    lines += ['', '3项损失/梯度/配置检查通过；真实首帧六层Set和末层Occupancy与原结构一致，六个属性头的尺度、旋转、opacity均有非零Occupancy梯度。训练损失及裁剪前梯度全程有限，六层Occupancy求和检查全程通过。每组原生48图与缓存的六层验证logit精确一致，原始权重末层覆盖和混淆矩阵逐帧复现历史全量结果。', '',
        '- [18行逐层性能及属性](../work_dirs/all_layer_render_small_mini_v1/aggregate/performance_layers.csv)、[108行逐层覆盖与负载分组](../work_dirs/all_layer_render_small_mini_v1/aggregate/layer_groups.csv)。',
        '- [306行逐类IoU和覆盖](../work_dirs/all_layer_render_small_mini_v1/aggregate/per_class.csv)、[逐场景计数](../work_dirs/all_layer_render_small_mini_v1/aggregate/per_scene.csv)。',
        '- [结构化汇总](../work_dirs/all_layer_render_small_mini_v1/summary.json)、[冻结协议](../work_dirs/all_layer_render_small_mini_v1/protocol.json)、[真实帧预检](../work_dirs/all_layer_render_small_mini_v1/pilot.json)、[最终复核](../work_dirs/all_layer_render_small_mini_v1/delivery_audit.json)。',
        '- [原结构权重](../work_dirs/all_layer_render_small_mini_v1/train/baseline/epoch_1.pth)、[六层方案权重](../work_dirs/all_layer_render_small_mini_v1/train/all_layers/epoch_1.pth)。', '',
        '场景配对区间来自seed928的1,000次场景重采样；本轮只有一个训练种子、一个epoch。', '',
        '```bash', 'python3.10 tools/report_all_layer_render_mini.py', '```', '']
    REPORT.write_text('\n'.join(lines))


def main():
    protocol = assert_protocol()
    training, traces = {}, {}
    for arm in ARMS:
        folder = RUN / 'train' / arm
        t = json.loads((folder / 'complete.json').read_text())
        assert t['frames'] == 4090 and t['optimizer_updates'] == 2045
        assert t['checkpoint_sha256'] == sha(folder / 'epoch_1.pth')
        training[arm] = t
        traces[arm] = [json.loads(s) for s in (folder / 'trace.jsonl').read_text().splitlines()]
        assert len(traces[arm]) == 4090
        assert np.isfinite([r['loss'] for r in traces[arm]]).all()
        assert np.isfinite([r['grad_norm'] for r in traces[arm] if 'grad_norm' in r]).all()
    assert [(r['index'], r['input_sha256']) for r in traces['baseline']] == [(r['index'], r['input_sha256']) for r in traces['all_layers']]
    arrays = {arm: aggregate(arm) for arm in MODELS}
    assert arrays['initial']['identities'] == arrays['baseline']['identities'] == arrays['all_layers']['identities']
    for arm in ARMS:
        np.testing.assert_array_equal(arrays['initial']['confusion'][:, :17].sum(2), arrays[arm]['confusion'][:, :17].sum(2))
    rows = [r for arm in MODELS for r in arrays[arm]['rows']]
    performances = [r for arm in MODELS for r in arrays[arm]['performance']]
    classes = [r for arm in MODELS for r in arrays[arm]['classes']]
    scene_rows = [r for arm in MODELS for r in arrays[arm]['per_scene']]
    assert len(rows) == 108 and len(performances) == 18 and len(classes) == 306 and len(scene_rows) == 16200
    uncertainty = dict(all_layers_minus_baseline=bootstrap(arrays['baseline'], arrays['all_layers']),
                       baseline_minus_initial=bootstrap(arrays['initial'], arrays['baseline']))
    AGG.mkdir(parents=True, exist_ok=True)
    for name, data in [('layer_groups', rows), ('performance_layers', performances), ('per_class', classes), ('per_scene', scene_rows)]:
        save_csv(AGG / (name+'.csv'), data)
    summary = dict(status='complete', train_frames=4090, validation_frames=6019, epochs=1,
        training=training, performances=performances, layer_groups=rows, scene_bootstrap_95=uncertainty)
    write(RUN / 'summary.json', summary)
    figures(rows, performances)
    document(rows, performances, training, uncertainty)
    artifacts = [REPORT, Path(__file__), ROOT / 'analysis/all_layer_render_small_mini_design.md',
                 *sorted(AGG.glob('*.csv')), *sorted(FIG.glob('*')), RUN / 'summary.json',
                 RUN / 'unit_test_result.json', RUN / 'evaluation_smoke.json', RUN / 'pilot.json']
    unit = json.loads((RUN / 'unit_test_result.json').read_text())
    smoke = json.loads((RUN / 'evaluation_smoke.json').read_text())
    assert unit['status'] == smoke['status'] == 'passed'
    assert unit['test_sha256'] == sha(ROOT / 'tests/test_all_layer_occupancy.py')
    write(RUN / 'delivery_audit.json', dict(status='passed', paired_all_4090_inputs_exact=True,
        same_6019_validation_samples=True, six_layer_rows=18, grouped_rows=108, per_class_rows=306,
        unit_tests=unit, evaluation_smoke=smoke,
        source_and_checkpoint_hashes_checked=True,
        artifacts={str(p.relative_to(ROOT)): sha(p) for p in artifacts}))
    write(RUN / 'status.json', dict(status='complete', report=str(REPORT.relative_to(ROOT)),
        updated_at=datetime.now().astimezone().isoformat()))
    print('Complete:', REPORT, flush=True)


if __name__ == '__main__':
    main()
