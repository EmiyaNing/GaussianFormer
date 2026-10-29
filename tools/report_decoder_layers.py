"""Recompute all decoder-layer tables from scene artifacts without inference."""
from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.stat_decoder_layers import RUN, HISTORY, MODELS
from tools.stat_boundary_gaussians import read, write, sha
from tools.decoder_layer_stats.common import SUPPORTS, GROUPS, DISTANCES, COUNTS, quintiles, summarize_frame

REPORT = ROOT / 'analysis/decoder_layers_full_val_results.md'
AGG = RUN / 'aggregate'
FIG = ROOT / 'analysis/figures/decoder_layers_full_val'


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ratios(row):
    row = dict(row)
    for name in DISTANCES:
        row[name] = row[name + '_sum'] / row['queries']
    for name in SUPPORTS:
        row[name + '_pct_foreground'] = 100 * row[name + '_union'] / row['foreground']
    row['assigned_foreground_pct'] = 100 * row['assigned_foreground'] / row['foreground']
    row['same_class_core_hit_GS_pct_all_GS'] = 100 * row['same_class_core_hit_GS'] / row['GS']
    row['foreground_core_hit_GS_pct_all_GS'] = 100 * row['foreground_core_hit_GS'] / row['GS']
    active = row['foreground_core_hit_GS']
    pairs = row['foreground_core_GS_voxel_pairs']
    row['same_class_core_hit_GS_pct_contacting_GS'] = 100 * row['same_class_core_hit_GS'] / active if active else None
    row['same_class_core_GS_voxel_pair_accuracy_pct'] = 100 * row['same_class_core_GS_voxel_pairs'] / pairs if pairs else None
    row['query_reference_distance_meters'] = row[DISTANCES[1]] * .4
    row['mean_child_GS_distance_meters'] = row[DISTANCES[3]] * .4
    return row


def aggregate(size):
    folder = RUN / size
    protocol = read(folder / 'protocol.json')
    complete = read(folder / 'complete.json')
    assert complete['status'] == 'complete' and complete['frames'] == 6019 and complete['scenes'] == 150
    assert read(folder / 'pilot.json')['status'] == 'passed'
    for rel, digest in protocol['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    q = protocol['queries']
    total = {}
    per_scene = []
    indices, tokens = [], []
    history_gt = np.zeros(8, np.int64)
    scene_files = sorted((folder / 'scenes').glob('*.json'))
    assert len(scene_files) == 150
    for scene_number, path in enumerate(scene_files):
        meta = read(path)
        assert meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
        assert meta['npz_sha256'] == sha(path.with_suffix('.npz'))
        old_meta = read(HISTORY / size / 'scenes' / path.name)
        assert meta['indices'] == old_meta['indices'] and meta['tokens'] == old_meta['tokens']
        indices.extend(meta['indices'])
        tokens.extend(meta['tokens'])
        with np.load(HISTORY / size / 'scenes' / path.with_suffix('.npz').name) as old:
            old_load = old['load_sixth']
            old_core = old['query_core']
            old_cc = old['class_coverage'].sum((0, 1))
            history_gt += old_cc
        with np.load(path.with_suffix('.npz')) as z:
            d, c, u, coverage, fg, load, groups = [z[key] for key in
                ('query_distances', 'query_GS_counts', 'group_unions', 'query_coverage', 'foreground', 'load_sixth', 'load_groups')]
            n = len(meta['indices'])
            assert d.shape == c.shape == coverage.shape == (n, 6, q, 4)
            assert u.shape == (n, 6, 6, 4) and fg.shape == (n,)
            assert np.isfinite(d).all() and np.all(d >= 0)
            np.testing.assert_array_equal(load, old_load)
            np.testing.assert_array_equal(load.sum(1), fg)
            np.testing.assert_array_equal(coverage[:, -1, :, 0], old_core)
            np.testing.assert_array_equal(groups, np.stack([quintiles(x) for x in load]))
            np.testing.assert_array_equal(u[:, -1, 0].sum(0), old_cc[[1, 3, 7, 4]])
            assert np.all(u[:, :, :, 3] <= u[:, :, :, 0])
            assert np.all(u[:, :, :, 0] <= u[:, :, :, 1]) and np.all(u[:, :, :, 1] <= u[:, :, :, 2])
            assert np.all(u[:, :, 1:].max(2) <= u[:, :, 0])
            assert np.all(u[:, :, 1:].sum(2) >= u[:, :, 0])
            for layer, k in enumerate(protocol['children']):
                assert np.all(c[:, layer, :, 0] <= k) and np.all(c[:, layer, :, 1] <= c[:, layer, :, 0])
                assert np.all(c[:, layer, :, 3] <= c[:, layer, :, 2])
                independent_rows = [summarize_frame(d[f, layer], c[f, layer], load[f], groups[f],
                    u[f, layer], fg[f], k) for f in range(n)] if scene_number == 0 else None
                for g, group in enumerate(GROUPS):
                    keep = np.ones((n, q), bool) if g == 0 else groups == g - 1
                    assert np.all(keep.sum(1) == (q if g == 0 else q//5))
                    row = dict(model=size, layer=layer+1, group=group, children_per_query=k,
                        frames=n, queries=int(keep.sum()), GS=int(keep.sum())*k,
                        foreground=int(fg.sum()), assigned_foreground=int(load[keep].sum()))
                    for j, name in enumerate(DISTANCES):
                        row[name + '_sum'] = float(d[:, layer, :, j][keep].sum(dtype=np.float64))
                    for j, name in enumerate(COUNTS):
                        row[name] = int(c[:, layer, :, j][keep].sum(dtype=np.int64))
                    for j, name in enumerate(SUPPORTS):
                        row[name + '_union'] = int(u[:, layer, g, j].sum(dtype=np.int64))
                    if independent_rows is not None:
                        for name in independent_rows[0][g]:
                            if name != 'group':
                                np.testing.assert_allclose(row[name], sum(r[g][name] for r in independent_rows), rtol=1e-12, atol=1e-8)
                    key = (layer+1, group)
                    if key not in total:
                        total[key] = dict(row)
                    else:
                        for name in row:
                            if name not in ('model', 'layer', 'group', 'children_per_query'):
                                total[key][name] += row[name]
                    per_scene.append(dict(scene=meta['scene'], **ratios(row)))
    assert sorted(indices) == list(range(6019)) and len(set(tokens)) == 6019
    rows = [ratios(total[(layer, group)]) for layer in range(1, 7) for group in GROUPS]
    for layer in range(1, 7):
        rr = [r for r in rows if r['layer'] == layer]
        for name in ('queries', 'GS', 'assigned_foreground', *COUNTS, *[x+'_sum' for x in DISTANCES]):
            np.testing.assert_allclose(rr[0][name], sum(x[name] for x in rr[1:]), rtol=1e-12, atol=1e-8)
        assert rr[0]['assigned_foreground'] == rr[0]['foreground'] == int(history_gt[0])
    with (HISTORY / 'aggregate/query_quintiles.csv').open(encoding='utf-8-sig') as stream:
        prior_groups = [r for r in csv.DictReader(stream) if r['model'] == size]
    assert len(prior_groups) == 5
    for group, old in zip(GROUPS[1:], prior_groups):
        assert old['group'] == group
        for row in rows:
            if row['group'] == group:
                np.testing.assert_allclose(row['assigned_foreground_pct'], float(old['GT_share']), rtol=1e-13, atol=1e-12)
    return protocol, rows, per_scene, tokens


def plots(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    keys = (DISTANCES[1], 'core_1sigma_pct_foreground', 'same_class_core_hit_GS_pct_all_GS')
    labels = ('Query reference distance (voxels)', 'Foreground core coverage (%)', 'Correct core-hit GS / all GS (%)')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, label in zip(axes, keys, labels):
        for size in MODELS:
            rr = [r for r in rows if r['model'] == size and r['group'] == 'all']
            ax.plot([r['layer'] for r in rr], [r[key] for r in rr], marker='o', label=size)
        ax.set(xlabel='Decoder layer', ylabel=label, xticks=range(1, 7))
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / ('overall.'+ext), dpi=180)
    plt.close(fig)
    colors = plt.cm.viridis(np.linspace(.05, .9, 5))
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    for i, size in enumerate(MODELS):
        for j, (key, label) in enumerate(zip(keys, labels)):
            ax = axes[i, j]
            for color, group in zip(colors, GROUPS[1:]):
                rr = [r for r in rows if r['model'] == size and r['group'] == group]
                ax.plot(range(1, 7), [r[key] for r in rr], marker='o', color=color, label=group)
            ax.set(title=f'{size}: {label}', xlabel='Decoder layer', xticks=range(1, 7))
            ax.grid(alpha=.25)
            if j == 0:
                ax.legend(title='Final load rank', fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / ('load_groups.'+ext), dpi=180)
    plt.close(fig)


def table(rows, include_gs_count=False):
    head = '| 层 | GS/query | 参考中心距前景（格） | 子GS平均距离（格） | 1σ前景覆盖 | 原生前景覆盖 | 同类1σ前景覆盖 | 正确GS/全部GS |'
    lines = [head, '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        gs = f"{r['same_class_core_hit_GS_pct_all_GS']:.3f}%"
        if include_gs_count:
            gs += f"（{r['same_class_core_hit_GS']:,}/{r['GS']:,}）"
        lines.append(f"| {r['layer']} | {r['children_per_query']} | {r[DISTANCES[1]]:.3f} | {r[DISTANCES[3]]:.3f} | {r['core_1sigma_pct_foreground']:.3f}% | {r['native_pct_foreground']:.3f}% | {r['same_class_core_1sigma_pct_foreground']:.3f}% | {gs} |")
    return '\n'.join(lines)


def write_report(rows, protocols):
    lines = ['# Tiny / Small / Large：六层Decoder的距离、覆盖和正确GS', '',
        '**状态：三个模型均完成完整验证集150场景、6,019帧，共18,057次模型–帧推理。** 每个模型统计全部六层，无训练、无校准、无帧抽样。', '',
        '前景为camera mask内GT类别0–16。距离以体素格为单位，1格=0.4米；表中主距离是到前景体素立方体并集的精确欧氏距离。', '',
        '“参考中心”从该层原始、未clamp的query_points解码到物理坐标，再对子点求均值；它是第1–5层输出送往下一层采样的参考中心。第六层只称输出参考中心，不声称它被第七层使用。“子GS平均距离”则分别求这一query全部子GS中心到前景的距离，再等权平均；两者不同。CSV同时给出到GT体素中心点的距离以及米制距离。', '',
        '**前五层GS的尺度固定为0.2米、旋转为单位旋转、opacity为0.1；只有第六层这些属性经过学习。** 本轮直接测量各层已有输出，不新增中间层属性头。因此覆盖随层变化同时受到中心、GS数量和末层属性变化影响。', '',
        '1σ前景覆盖率=该层所选query的GS核心覆盖到的不同GT前景体素数/全部可见GT前景体素数；原生覆盖沿用生产候选盒与正贡献条件。同类1σ覆盖另要求GS预测类别与GT相同。所有并集在GS间和query间去重。', '',
        '“正确GS/全部GS”=核心内至少包含一个与该GS预测类别相同的可见GT前景体素的GS数/该组该层全部输出GS数。GS类别取17类语义得分argmax；未接触前景的GS也在分母中。一个GS命中同类体素即可计入，不要求其核心中的所有体素都同类；核心纯度另见CSV的GS–体素配对正确率。这些几何和语义接触指标不等于最终渲染预测正确率。', '',
        '负载组沿用[此前三模型实验](gaussian_sizes_full_val_design.md)：每帧按最终第六层最近GS归属前景数从高到低，将全部query五等分，含零负载；并列时query ID较大者靠前，与历史top20约定一致。固定这些query ID追踪六层，不在每层重新排序。0–20%是最高负载，80–100%是最低负载，百分位不是绝对负载范围。', '',
        '组覆盖的分母仍是整个前景；五组覆盖空间可重叠，因此五组覆盖率不能相加为总体覆盖率。距离先对query等权，正确GS率累加GS分子/分母，覆盖率累加体素分子/分母。跨模型的query编号没有对应关系。', '',
        '![整体逐层变化](figures/decoder_layers_full_val/overall.png)', '',
        '![固定负载五组逐层变化](figures/decoder_layers_full_val/load_groups.png)', '']
    for size in MODELS:
        p = protocols[size]
        lines += [f'## {size.capitalize()}：全部query', '',
                  f"每帧{p['queries']:,}个query；各层GS/query为{p['children']}。正确GS数量按整个验证集的GS–帧实例累计。", '',
                  table([r for r in rows if r['model'] == size and r['group'] == 'all'], True), '']
        for group in GROUPS[1:]:
            lines += [f'### {size.capitalize()}：负载排名{group}', '',
                      table([r for r in rows if r['model'] == size and r['group'] == group]), '']
    lines += ['## 数据与复核', '',
        '- [完整108行逐层分组CSV](../work_dirs/decoder_layers_full_val/aggregate/layer_groups.csv)：每一行保留正确GS原始计数、全部GS分母、并集体素数、全部GT前景分母、两种中心的两类距离、1σ/3σ/原生/同类核心覆盖。',
        '- [18行总体CSV](../work_dirs/decoder_layers_full_val/aggregate/overall.csv)与[90行负载分组CSV](../work_dirs/decoder_layers_full_val/aggregate/load_quintiles.csv)。',
        '- [逐场景统计](../work_dirs/decoder_layers_full_val/aggregate/per_scene.csv)、[结构化汇总](../work_dirs/decoder_layers_full_val/summary.json)、[复核记录](../work_dirs/decoder_layers_full_val/delivery_audit.json)。',
        '- 每场景NPZ保存逐帧、逐层、逐query的参考中心、两种中心的两类距离、四种query去重覆盖数、GS命中数与配对数、固定负载组以及组覆盖并集。汇总可脱离模型推理复算。',
        '- [逐帧NPZ字段与分母说明](../work_dirs/decoder_layers_full_val/artifact_schema.md)。',
        '- [Tiny协议](../work_dirs/decoder_layers_full_val/tiny/protocol.json)、[Small协议](../work_dirs/decoder_layers_full_val/small/protocol.json)、[Large协议](../work_dirs/decoder_layers_full_val/large/protocol.json)冻结权重、配置、标注和源码哈希。',
        '- 每模型真实帧原生48图与缓存推理的全部六层GS、原始点和最终logit逐元素一致；每层与既有独立CUDA覆盖位集一致；距离使用CPU FP64穷举核验，另逐层直接枚举32个GS的全部可见前景体素检查核心同类接触。',
        '- 全部18,057模型–帧的末层逐query核心覆盖、整体四种覆盖与混淆矩阵精确复现此前结果。五组负载份额在全部六层均复现历史五等分表。核对所有场景、索引、sample token唯一性与跨模型样本一致性；各组的距离总和、GS计数、归属负载通过守恒检查。', '',
        '```bash', 'python3.10 tools/report_decoder_layers.py', '```', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


def main():
    AGG.mkdir(parents=True, exist_ok=True)
    rows, scene_rows, protocols, token_sets = [], [], {}, []
    for size in MODELS:
        protocol, data, scenes, tokens = aggregate(size)
        protocols[size] = protocol
        rows.extend(data)
        scene_rows.extend(scenes)
        token_sets.append(set(tokens))
    assert token_sets[0] == token_sets[1] == token_sets[2]
    assert len(rows) == 108 and len(scene_rows) == 16200
    save_csv(AGG / 'layer_groups.csv', rows)
    save_csv(AGG / 'overall.csv', [r for r in rows if r['group'] == 'all'])
    save_csv(AGG / 'load_quintiles.csv', [r for r in rows if r['group'] != 'all'])
    save_csv(AGG / 'per_scene.csv', scene_rows)
    write(RUN / 'summary.json', dict(status='complete', models=list(MODELS),
        frames_per_model=6019, scenes_per_model=150, rows=rows))
    plots(rows)
    write_report(rows, protocols)
    test_path = RUN / 'unit_test_result.json'
    tests = read(test_path) if test_path.exists() else dict(status='not_recorded')
    if test_path.exists():
        assert tests['status'] == 'passed' and tests['passed'] == 5 and tests['skipped'] == 0
        assert sha(ROOT / 'tests/test_decoder_layer_stats.py') == tests['test_source_sha256']
    artifacts = [REPORT, Path(__file__), ROOT / 'analysis/decoder_layers_full_val_design.md',
                 ROOT / 'tests/test_decoder_layer_stats.py', RUN / 'artifact_schema.md',
                 *sorted(AGG.glob('*.csv')), *sorted(FIG.glob('*')), RUN / 'summary.json']
    if test_path.exists():
        artifacts.append(test_path)
    write(RUN / 'delivery_audit.json', dict(status='passed', aggregate_rows=108, overall_rows=18,
        load_group_rows=90, scene_rows=16200, samples_matched_across_models=True,
        independent_frame_and_vectorized_scene_aggregation_exact=True, unit_tests=tests,
        historical_final_layer_counts_exact=True, source_and_scene_sha256_checked=True,
        artifacts={str(p.relative_to(ROOT)): sha(p) for p in artifacts}))
    write(RUN / 'status.json', dict(status='complete', models=list(MODELS), frames_per_model=6019,
        report=str(REPORT.relative_to(ROOT)), updated_at=datetime.now().astimezone().isoformat()))
    print(f'Complete: {REPORT}', flush=True)


if __name__ == '__main__':
    main()
