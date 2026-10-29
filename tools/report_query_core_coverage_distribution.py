"""Full-validation distribution of visible foreground in final query cores.

CPU only: reuses all saved final-level query coverage and historical camera
contact counts. No inference, training, or frame subsampling.
"""
from __future__ import annotations
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.stat_boundary_gaussians import read, write, sha
from tools.stat_query_budget import FIELDS

SOURCE = ROOT / 'work_dirs/small_final_query_coverage_full'
OLD = ROOT / 'work_dirs/small_gaussian_diagnosis_full'
OUT = SOURCE / 'query_core_distribution_val'
REPORT = ROOT / 'analysis/query_foreground_core_coverage_val_results.md'
Q, CHILDREN = 1200, 64
BINS = ((0, 0), (1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, None))
THRESHOLDS = (0, 1, 3, 5, 10, 20, 50, 100)


def csv_write(path, rows):
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def fraction(a, b):
    return float(a / b) if b else None


def quantiles(values):
    levels = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)
    return {f'{100 * p:g}%': float(x) for p, x in zip(levels, np.quantile(values, levels))}


def main():
    protocol = read(SOURCE / 'protocol.json')
    marker = read(SOURCE / 'val_complete.json')
    assert marker == dict(status='complete', frames=6019, scenes=150)
    paths = sorted((SOURCE / 'scenes/val').glob('*.json'))
    assert len(paths) == 150
    chunks, scene_rows, records, seen, tokens, input_hashes = [], [], [], set(), set(), {}
    for path in paths:
        meta = read(path)
        assert meta['status'] == 'complete' and meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
        ids = meta['indices']
        assert [r['index'] for r in meta['records']] == ids and not seen.intersection(ids)
        seen.update(ids)
        with np.load(path.with_suffix('.npz')) as z:
            core = z['core_1sigma_query_coverage']
            strict = z['strict_3sigma_query_coverage']
            native = z['native_support_query_coverage']
            nearest5, nearest6 = z['load_fifth'], z['load_sixth']
        old_meta = read(OLD / 'scenes/val' / path.name)
        with np.load(OLD / 'scenes/val' / path.with_suffix('.npz').name) as z:
            previous = z['metrics']
        assert ids == old_meta['indices']
        assert [r['token'] for r in meta['records']] == [r['sample_token'] for r in old_meta['records']]
        for row in meta['records']:
            assert row['token'] not in tokens
            tokens.add(row['token'])
        assert core.shape == (len(ids), Q)
        np.testing.assert_array_equal(core, previous[..., FIELDS.index('core_coverage_all_fg')])
        np.testing.assert_array_equal(nearest5, previous[..., FIELDS.index('gt_foreground')])
        camera = previous[..., FIELDS.index('gs_camera_core')] > 0
        camera_native = previous[..., FIELDS.index('gs_camera_native')] > 0
        core_gs = previous[..., FIELDS.index('core_gs_touches_fg')].astype(np.int32)
        np.testing.assert_array_equal(core > 0, core_gs > 0)
        np.testing.assert_array_equal(native > 0, previous[..., FIELDS.index('native_gs_touches_fg')] > 0)
        assert np.all(core <= strict) and np.all(strict <= native)
        assert not np.any((core > 0) & ~camera)
        chunks.append((core, strict, native, camera, camera_native, core_gs, nearest5, nearest6))
        for index, row in zip(ids, meta['records']):
            records.append(dict(index=index, scene=meta['scene'], token=row['token']))
        scene_row = dict(scene_token=meta['scene'], scene_name=old_meta['scene_name'], frames=len(ids),
            queries=core.size, zero_queries=int((core == 0).sum()), zero_fraction=float((core == 0).mean()),
            mean_foreground_voxels=float(core.mean()), median_foreground_voxels=float(np.median(core)),
            camera_contact_queries=int(camera.sum()))
        for t in (1, 5, 10, 20):
            scene_row[f'le_{t}_queries'] = int((core <= t).sum())
            scene_row[f'le_{t}_fraction'] = float((core <= t).mean())
        scene_rows.append(scene_row)
        input_hashes[str(path.relative_to(ROOT))] = sha(path)
        input_hashes[str(path.with_suffix('.npz').relative_to(ROOT))] = sha(path.with_suffix('.npz'))
    assert seen == set(range(6019)) and len(tokens) == 6019
    core, strict, native, camera, camera_native, core_gs, near5, near6 = [np.concatenate([c[i] for c in chunks]) for i in range(8)]
    assert core.shape == (6019, Q)
    total, contact = core.size, int(camera.sum())
    bin_rows, threshold_rows = [], []
    for low, high in BINS:
        mask = (core >= low) & ((core <= high) if high is not None else True)
        n = int(mask.sum())
        label = '0' if low == high == 0 else (f'{low}–{high}' if high is not None else f'>{low - 1}')
        bin_rows.append(dict(bin=label,
            lower_inclusive=low, upper_inclusive=high, queries=n, fraction=n / total,
            mean_queries_per_frame=n / 6019, gaussian_budget=n * CHILDREN,
            camera_contact_queries=int((mask & camera).sum()),
            fraction_within_camera_contact=fraction((mask & camera).sum(), contact),
            query_voxel_pairs=int(core[mask].sum()),
            query_voxel_pair_share=fraction(core[mask].sum(), core.sum(dtype=np.int64))))
    assert sum(r['queries'] for r in bin_rows) == total
    for t in THRESHOLDS:
        mask = core <= t
        per_frame = mask.sum(1)
        threshold_rows.append(dict(max_foreground_voxels=t, queries=int(mask.sum()), fraction=float(mask.mean()),
            mean_queries_per_frame=float(per_frame.mean()),
            frame_queries_p10=float(np.quantile(per_frame, .1)),
            frame_queries_p50=float(np.quantile(per_frame, .5)),
            frame_queries_p90=float(np.quantile(per_frame, .9)),
            camera_contact_queries=int((mask & camera).sum()),
            fraction_within_camera_contact=fraction((mask & camera).sum(), contact)))
    zero = core == 0
    groups = {
        'no_core_foreground': zero,
        'no_core_foreground_no_known_camera_core': zero & ~camera,
        'no_core_foreground_camera_core_empty_only': zero & camera,
        'no_core_foreground_but_strict3sigma_foreground': zero & (strict > 0),
        'no_core_foreground_but_native_foreground': zero & (native > 0),
        'no_native_foreground': native == 0,
        'no_core_foreground_but_fifth_nearest_load': zero & (near5 > 0),
        'no_core_foreground_but_sixth_nearest_load': zero & (near6 > 0),
    }
    group_rows = {name: dict(queries=int(mask.sum()), fraction_all=float(mask.mean()),
                           mean_queries_per_frame=float(mask.sum() / 6019)) for name, mask in groups.items()}
    assert group_rows['no_core_foreground_no_known_camera_core']['queries'] + group_rows['no_core_foreground_camera_core_empty_only']['queries'] == int(zero.sum())
    # Cross-check the previously published all-validation zero-core count.
    old_zero = read(OLD / 'aggregate/query_zero_load_followup.json')['groups']['all_queries']['queries_without_core_fg_contact']
    assert int(zero.sum()) == old_zero == 443790
    by_id = []
    for q in range(Q):
        a = core[:, q]
        by_id.append(dict(query_slot=q, frames=6019, mean_foreground_voxels=float(a.mean()),
            median_foreground_voxels=float(np.median(a)), zero_frames=int((a == 0).sum()),
            zero_frame_fraction=float((a == 0).mean()), le5_frame_fraction=float((a <= 5).mean()),
            le10_frame_fraction=float((a <= 10).mean()), p90_foreground_voxels=float(np.quantile(a, .9))))
    assert sum(r['zero_frames'] for r in by_id) == int(zero.sum())
    result = dict(status='complete', generated_at=datetime.now().astimezone().isoformat(),
        checkpoint_sha256=protocol['checkpoint_sha256'], scenes=150, frames=6019,
        queries_per_frame=Q, gaussians_per_query=CHILDREN, query_frame_instances=total,
        definition='Distinct camera-valid GT foreground voxels covered by union of each query final 64 GS 1sigma cores; class agnostic',
        camera_contact_queries=contact, mean_foreground_voxels=float(core.mean()),
        mean_foreground_voxels_positive_only=float(core[core > 0].mean()),
        coverage_quantiles=quantiles(core), contacting_gs_quantiles=quantiles(core_gs),
        bins=bin_rows, thresholds=threshold_rows, zero_groups=group_rows,
        provenance=dict(final_protocol_sha256=sha(SOURCE / 'protocol.json'),
            historical_protocol_sha256=sha(OLD / 'protocol.json'),
            input_sha256=input_hashes, reporting_script_sha256=sha(Path(__file__))),
        audit=dict(full_unique_frame_partition=True, unique_sample_tokens=6019,
            historical_per_query_core_coverage_exact=True, historical_zero_count_exact=True,
            zero_camera_partition_conserved=True, bin_counts_conserved=True,
            query_slot_counts_conserved=True))
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'summary.json', result)
    csv_write(OUT / 'coverage_bins.csv', bin_rows)
    csv_write(OUT / 'coverage_thresholds.csv', threshold_rows)
    csv_write(OUT / 'per_scene.csv', scene_rows)
    csv_write(OUT / 'per_query_slot.csv', by_id)
    # Save integer distribution for rebinning without reading historical files.
    hist = np.bincount(core.flatten())
    with (OUT / 'histogram.npz.tmp').open('wb') as f:
        np.savez_compressed(f, foreground_voxel_count=np.arange(len(hist)), query_instances=hist)
    (OUT / 'histogram.npz.tmp').replace(OUT / 'histogram.npz')
    report(result)
    print(json.dumps({k: result[k] for k in ('status', 'frames', 'scenes', 'query_frame_instances', 'mean_foreground_voxels', 'coverage_quantiles', 'zero_groups')}, ensure_ascii=False))


def report(r):
    pct = lambda value: f'{100 * value:.3f}%'
    groups = r['zero_groups']
    lines = ['# 完整验证集：有多少Query的1σ核心几乎不覆盖前景', '',
        '**状态：150个场景、6,019帧全部完成；本轮仅复算已保存数据。**', '',
        '统计对象是原始small权重的最终第六级：每帧1,200个query，每query输出64个GS。'
        '将一个query的64个GS的1σ核心取并集，计算其中有多少个不同的camera有效GT前景体素。'
        '类别暂不区分；同一query的GS重复覆盖同一体素只计一次。不同query之间可以覆盖同一体素。', '',
        f"共有 **{r['query_frame_instances']:,}个query–帧实例**。这里不是只有1,200个query，也不是把跨帧位置当成同一物理对象。", '',
        '## 覆盖数量分布', '',
        '| 每query覆盖的不同前景体素数 | Query–帧数量 | 占全部query | 平均每帧query数 |', '|---|---:|---:|---:|']
    for b in r['bins']:
        lines.append(f"| {b['bin']} | {b['queries']:,} | {pct(b['fraction'])} | {b['mean_queries_per_frame']:.2f} |")
    lines += ['', '“特别少”没有预先唯一的阈值，下面同时给出几个累计阈值，便于判断。', '',
        '| 覆盖上限（包含零覆盖） | Query–帧数量 | 占全部query | 平均每帧query数 |', '|---|---:|---:|---:|']
    for b in r['thresholds']:
        if b['max_foreground_voxels'] in (0, 1, 5, 10, 20):
            lines.append(f"| ≤{b['max_foreground_voxels']}个前景体素 | {b['queries']:,} | {pct(b['fraction'])} | {b['mean_queries_per_frame']:.2f} |")
    q = r['coverage_quantiles']
    lines += ['', f"平均每query覆盖 **{r['mean_foreground_voxels']:.3f}** 个不同前景体素；中位数 **{q['50%']:.3f}**。"
        f"P10／P25／P75／P90分别为 **{q['10%']:.3f}／{q['25%']:.3f}／{q['75%']:.3f}／{q['90%']:.3f}**。", '',
        '## 零前景核心覆盖发生在哪里', '',
        '| 零覆盖分组 | Query–帧数量 | 占全部query | 平均每帧query数 |', '|---|---:|---:|---:|']
    for key, label in (
        ('no_core_foreground', '1σ核心没有任何可见GT前景'),
        ('no_core_foreground_no_known_camera_core', '其中：核心不接触任何camera有效GT体素'),
        ('no_core_foreground_camera_core_empty_only', '其中：核心接触camera有效体素，但只接触empty')):
        g = groups[key]
        lines.append(f"| {label} | {g['queries']:,} | {pct(g['fraction_all'])} | {g['mean_queries_per_frame']:.2f} |")
    lines += ['', '后两行互不重叠，相加等于第一行。这里的“只接触empty”仅指camera有效GT范围，不能外推到未观测空间。', '',
        f"若只保留至少一个核心接触camera有效GT的query，分母为 **{r['camera_contact_queries']:,}个query–帧实例**，其低覆盖比例如下：", '',
        '| 累计覆盖上限 | 占核心接触camera有效GT的query |', '|---|---:|']
    for b in r['thresholds']:
        if b['max_foreground_voxels'] in (0, 5, 10, 20):
            lines.append(f"| ≤{b['max_foreground_voxels']}个前景体素 | {pct(b['fraction_within_camera_contact'])} |")
    lines += ['',
        '## 没有核心覆盖，外围是否仍接触前景', '',
        '| 条件 | Query–帧数量 | 占全部query |', '|---|---:|---:|']
    for key, label in (
        ('no_core_foreground_but_strict3sigma_foreground', '1σ无前景，但严格3σ支持接触前景'),
        ('no_core_foreground_but_native_foreground', '1σ无前景，但原生渲染支持接触前景'),
        ('no_native_foreground', '连原生渲染支持也不接触可见GT前景')):
        g = groups[key]
        lines.append(f"| {label} | {g['queries']:,} | {pct(g['fraction_all'])} |")
    lines += ['', '这些行不是互斥分组；严格3σ正支持包含于原生正支持。'
        '一个query没有1σ前景，并不意味着其全部GS没有渲染贡献；外围仍可能参与预测。', '',
        '## 解释这份统计时需要保留的口径', '',
        '- 这里数的是实际最终核心接触到的前景，不使用第五级或第六级最近邻来强制分配体素。',
        '- 每个query的GS数量相同，因此query比例也等于这组query占用的GS输出预算比例；这不等于可无损删除的预算比例。',
        '- 体素数不是预测正确率，也不要求GS类别与GT一致。低覆盖query可能仍承担外围支持，空区影响也未由本表衡量。',
        '- 零可见覆盖中的一部分来自核心没有进入camera有效区域；不能把这一部分写成“处于可见前景附近却完全失效”。', '',
        '另外保留了至少一个核心接触camera有效体素的query子集，阈值CSV同时报告该子集内的低前景覆盖率。'
        '逐输出槽位CSV仅按query索引汇总出现频率，不把槽位视为跨帧追踪的同一对象。', '',
        '## 数据文件与复核', '',
        '- [完整统计与源数据哈希](../work_dirs/small_final_query_coverage_full/query_core_distribution_val/summary.json)',
        '- [互斥覆盖分档](../work_dirs/small_final_query_coverage_full/query_core_distribution_val/coverage_bins.csv)',
        '- [累计阈值](../work_dirs/small_final_query_coverage_full/query_core_distribution_val/coverage_thresholds.csv)',
        '- [逐场景统计](../work_dirs/small_final_query_coverage_full/query_core_distribution_val/per_scene.csv)',
        '- [逐query输出槽位](../work_dirs/small_final_query_coverage_full/query_core_distribution_val/per_query_slot.csv)',
        '- [可重新分档的整数直方图](../work_dirs/small_final_query_coverage_full/query_core_distribution_val/histogram.npz)',
        '- [复算脚本](../tools/report_query_core_coverage_distribution.py)', '',
        '复算逐query覆盖与历史统计完全一致，零覆盖总数也与历史完整统计的443,790一致；'
        '场景、帧、sample token、分档总量及零覆盖分区均通过守恒检查。', '',
        '```bash', 'python3.10 tools/report_query_core_coverage_distribution.py', '```', '']
    REPORT.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
