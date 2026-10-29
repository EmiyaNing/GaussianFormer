"""Full/partial CPU summaries of exact GS-to-foreground distances."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.stat_boundary_gaussians import read, write, sha
from tools.stat_query_foreground_distance import RUN, Q, K

REPORT = ROOT / 'analysis/query_foreground_distance_val_results.md'
BINS = ((0, 0), (1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, None))


def qstats(x):
    levels = np.quantile(x, (.1, .5, .9, .99))
    return dict(mean=float(np.mean(x, dtype=np.float64)),
        p10=float(levels[0]), p50=float(levels[1]),
        p90=float(levels[2]), p99=float(levels[3]), maximum=float(np.max(x)))


def summarize(group, selected, d, bounds, contact):
    n = int(selected.sum())
    if not n:
        return dict(group=group, queries=0)
    values = d[selected]
    center, volume = values[..., 0], values[..., 1]
    gap = bounds[selected]
    row = dict(group=group, queries=n, gaussians=n * K,
        center_min=qstats(center.min(1)), center_query_mean=qstats(center.mean(1, dtype=np.float64)),
        volume_min=qstats(volume.min(1)), volume_query_mean=qstats(volume.mean(1, dtype=np.float64)),
        volume_query_median=qstats(np.median(volume, axis=1)), volume_query_max=qstats(volume.max(1)),
        volume_all_gaussians=qstats(volume),
        zero_core_gs_per_query_mean=float((K - contact[selected]).mean()),
        query_at_least44_zero_core_gs_fraction=float((contact[selected] <= 20).mean()),
        query_some_center_in_foreground_fraction=float((volume.min(1) == 0).mean()),
        query_all_centers_in_foreground_fraction=float((volume.max(1) == 0).mean()),
        core_gap_lower=qstats(gap[:, 0]), core_gap_upper=qstats(gap[:, 1]),
        core_gap_lower_positive_fraction=float((gap[:, 0] > 1e-4).mean()),
        core_gap_upper_zero_fraction=float((gap[:, 1] <= 1e-6).mean()))
    for t in (.5, 1, 2, 3, 5, 10):
        row[f'gs_center_volume_distance_gt_{t:g}_fraction'] = float((volume > t).mean())
        row[f'query_all_centers_distance_gt_{t:g}_fraction'] = float((volume.min(1) > t).mean())
    return row


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        key = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, key + '_'))
        else:
            result[key] = item
    return result


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def markdown(r):
    pct = lambda x: f'{100 * x:.3f}%'
    lines = ['# 低前景覆盖Query离前景有多远', '',
        f"**状态：{'完整验证集已完成' if r['status'] == 'complete' else '仅部分场景，不能作全量结论'}。** "
        f"已统计 {r['scenes']}/150 场景、{r['frames']}/6,019 帧，共 {r['queries']:,} 个query–帧实例。", '',
        '本轮沿用原始small权重与前一报告的覆盖分档。一个query有64个最终GS；分档依据是这64个GS的1σ核心并集内，'
        '不同camera有效GT前景体素的数量。', '',
        '**距离统一用体素网格单位：1格=0.4米。** 采用三维欧氏距离，允许小数；对角距离不按整数步数计算。'
        '前景目标限定为camera有效GT，不按预测类别筛选。', '',
        '## 先说明“一个Query的距离”是什么意思', '',
        '每个GS中心分别计算到前景的距离，然后汇总query的64个中心。'
        '“最近中心距离”取64个距离中的最小值；“64个中心平均距离”取平均值。'
        '这两项衡量最终子GS的布局，未测量query用于取图像特征的参考中心；距离也不按opacity或语义幅度加权。', '',
        '**当前OPUS确实有query空间参考位置。** 第六级采样先把第五级32个原始参考点还原到三维坐标，'
        '取均值作为采样中心、逐轴标准差作为采样展开尺度，再加预测偏移并投影到FPN。'
        '这是网络实际使用的中心，不能称作“虚构中心”。'
        '它位于最终64个GS产生之前，与本表按子GS距离取最小值或平均值是不同统计。'
        '严格复算该中心时应使用原始参考点，不能直接用经过渲染边界clamp后的GS中心替代。', '',
        '主表测量的是**到前景体素实体的距离**：GT前景由0.4米立方体组成，中心落在其中时距离为0。'
        '另行报告到GT体素中心点的距离，便于与原Set几何监督对照。这两个最小值都遍历全部前景体素独立计算，'
        '不会假定最近体素中心对应的立方体一定也是最近的实体。', '',
        '## 各覆盖分档的实际距离', '',
        '| 核心内不同前景体素数 | Query–帧数 | 最近GS中心距离的中位数（格） | 64个中心平均距离的组均值（格） | 全部GS中心距前景>1格 | >3格 |',
        '|---|---:|---:|---:|---:|---:|']
    for name in r['bin_order']:
        x = r['groups'][name]
        if not x['queries']:
            continue
        lines.append(f"| {name} | {x['queries']:,} | {x['volume_min']['p50']:.3f} | {x['volume_query_mean']['mean']:.3f} | "
            f"{pct(x['gs_center_volume_distance_gt_1_fraction'])} | {pct(x['gs_center_volume_distance_gt_3_fraction'])} |")
    lines += ['', '“>1格”表示中心到最近可见GT前景实体的距离大于0.4米；“>3格”表示大于1.2米。'
        '这两列按组内全部GS计算，不能读成整组query都远离前景。最近中心距离很小也只说明至少一个GS靠近，'
        '不能代表其他63个GS。', '',
        '## 与前表一致的累计低覆盖组', '',
        '| 累计覆盖上限 | Query–帧数 | 最近中心距离P50 / P90（格） | 64中心平均距离：均值 / P50 / P90（格） |', '|---|---:|---:|---:|']
    for name in ('le_5', 'le_10', 'le_20'):
        x = r['groups'][name]
        if not x['queries']:
            continue
        a, b = x['volume_min'], x['volume_query_mean']
        lines.append(f"| ≤{name[3:]}个前景体素 | {x['queries']:,} | {a['p50']:.3f} / {a['p90']:.3f} | {b['mean']:.3f} / {b['p50']:.3f} / {b['p90']:.3f} |")
    lines += ['', '这三行包含零覆盖，互相嵌套，不能相加。', '',
        '## 零核心覆盖：区分camera范围', '',
        '| 零覆盖组 | Query–帧数 | 最近中心距离P50 / P90（格） | 64中心平均距离的组均值（格） | 至少一个中心在GT前景实体内 |',
        '|---|---:|---:|---:|---:|']
    labels = dict(zero_all='全部零核心覆盖', zero_no_camera_core='核心不接触camera有效GT',
                  zero_camera_empty_only='核心接触camera，但只接触empty')
    for name, label in labels.items():
        x = r['groups'][name]
        if not x['queries']:
            continue
        a = x['volume_min']
        lines.append(f"| {label} | {x['queries']:,} | {a['p50']:.3f} / {a['p90']:.3f} | {x['volume_query_mean']['mean']:.3f} | {pct(x['query_some_center_in_foreground_fraction'])} |")
    lines += ['', '核心未进入camera有效区域时，距离仍以可见GT前景为目标；不能据此断言它远离全部真实前景。'
        '“零核心覆盖”按GT体素中心是否进入1σ判断，而实体距离按整个体素立方体计算。'
        '因此中心可以位于前景体素内部，但较小或偏心的核心没有包含该体素的中心；这两种统计出现不同结果并不矛盾。', '',
        '## 对照：到GT体素中心点的距离', '',
        '| 覆盖分档 | 最近中心到GT中心距离P50（格） | 64中心到GT中心平均距离（格） |', '|---|---:|---:|']
    for name in r['bin_order']:
        x = r['groups'][name]
        if x['queries']:
            lines.append(f"| {name} | {x['center_min']['p50']:.3f} | {x['center_query_mean']['mean']:.3f} |")
    lines += ['', '## 1σ场到前景的间隔：给出界，而非伪装成精确距离', '',
        '中心距离与场距离不同。每个GS的1σ椭球包含半径为最小尺度的球，并被最大尺度的球包住。'
        '设E为该1σ椭球，F为可见GT前景立方体的并集，体素边长h=0.4米。中心到F的距离除以h记为d，'
        '两个球的物理半径除以h后记为r_min、r_max，均以格为单位，则：', '',
        '$$', r'\max(0,d-r_{\max})\ \le\ \frac{\operatorname{dist}(E,\mathcal F)}{h}\ \le\ \max(0,d-r_{\min}).', '$$', '',
        '对query的64个GS分别计算，再分别取下界和上界的最小值。以下为未截断几何椭球的保守界，'
        '不是逐椭球与体素并集求投影后的精确场距离。', '',
        '| 零覆盖组 | 间隔下界P50（格） | 间隔上界P50（格） | 下界>0.0001格：确定存在间隔 | 上界≤0.000001格：数值精度内接触 |', '|---|---:|---:|---:|---:|']
    for name, label in labels.items():
        x = r['groups'][name]
        if x['queries']:
            lines.append(f"| {label} | {x['core_gap_lower']['p50']:.3f} | {x['core_gap_upper']['p50']:.3f} | {pct(x['core_gap_lower_positive_fraction'])} | {pct(x['core_gap_upper_zero_fraction'])} |")
    lines += ['', '下界为零但上界为正的情况无法仅凭这个界判定是否接触。中心距离主表是精确计算，两种精度口径分开报告。', '',
        '## 附加：直接数有多少GS没有前景核心覆盖', '',
        '| 覆盖分档 | 平均每query无前景核心覆盖的GS数 / 64 | 至少44个GS无前景核心覆盖的query比例 |', '|---|---:|---:|']
    for name in r['bin_order']:
        x = r['groups'][name]
        if x['queries']:
            lines.append(f"| {name} | {x['zero_core_gs_per_query_mean']:.3f} | {pct(x['query_at_least44_zero_core_gs_fraction'])} |")
    lines += ['', '此表直接读取逐GS接触统计，未使用“64减去体素数”的推断。', '',
        '## 数据与复核', '',
        '- [汇总JSON](../work_dirs/small_query_foreground_distance_val/aggregate/summary.json)',
        '- [全部分档和累计组CSV](../work_dirs/small_query_foreground_distance_val/aggregate/coverage_distance_groups.csv)',
        '- [逐场景统计](../work_dirs/small_query_foreground_distance_val/aggregate/per_scene.csv)',
        '- [冻结协议](../work_dirs/small_query_foreground_distance_val/protocol.json)',
        '- [FP64穷举复核](../work_dirs/small_query_foreground_distance_val/pilot.json)',
        '- [距离计算实现](../tools/query_foreground_distance/kernels.cu)', '',
        '- [Query采样参考中心与采样点生成](../model/encoder/opus_encoder.py)', '',
        '每帧的逐query1σ覆盖与历史结果一致；分档数量复现此前完整验证统计。'
        '首次运行对128个真实GS进行独立FP64穷举复核。距离以FP32保存，汇总均值采用FP64；分位数由原始数值计算，未用直方图近似。'
        '逐GS两种距离保存在各场景NPZ，后续调整距离阈值无需再次运行模型。', '',
        '```bash', 'python3.10 tools/report_query_foreground_distance.py', '```', '']
    return '\n'.join(lines)


def report(partial=False):
    protocol = read(RUN / 'protocol.json')
    chunks, gaps, cores, cameras, contacts, seen, tokens, rows, hashes = [], [], [], [], [], set(), set(), [], {}
    paths = sorted((RUN / 'scenes').glob('*.json'))
    for path in paths:
        meta = read(path)
        assert meta['status'] == 'complete' and meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
        assert [r['index'] for r in meta['records']] == meta['indices'] and not seen.intersection(meta['indices'])
        seen.update(meta['indices'])
        for row in meta['records']:
            assert row['token'] not in tokens
            tokens.add(row['token'])
        with np.load(path.with_suffix('.npz')) as z:
            d, b, c = z['distances'], z['core_support_gap_bounds'], z['core_foreground_voxels']
            cam, touch = z['camera_core_contact'], z['core_contact_gs']
        assert d.shape == (len(meta['indices']), Q, K, 2) and c.shape == d.shape[:2]
        assert np.isfinite(d).all() and np.all(d >= 0) and np.all(b >= 0)
        delta = d[..., 0] - d[..., 1]
        assert np.all(delta >= -1e-5) and np.all(delta <= np.sqrt(3) / 2 + 2e-5)
        assert np.all(b[..., 0] <= b[..., 1] + 1e-5)
        zero = c == 0
        region = d[..., 1]
        rows.append(dict(scene=meta['scene'], frames=len(meta['indices']), queries=c.size,
            center_volume_mean_grid=float(region.mean(dtype=np.float64)),
            query_min_volume_p50_grid=float(np.median(region.min(-1))),
            zero_queries=int(zero.sum()),
            zero_query_center_volume_mean_grid=float(region[zero].mean(dtype=np.float64)) if zero.any() else None))
        chunks.append(d.reshape(-1, K, 2)); gaps.append(b.reshape(-1, 2)); cores.append(c.flatten())
        cameras.append(cam.flatten()); contacts.append(touch.flatten())
        hashes[str(path.relative_to(ROOT))] = sha(path)
        hashes[str(path.with_suffix('.npz').relative_to(ROOT))] = sha(path.with_suffix('.npz'))
    complete = (RUN / 'complete.json').exists() and len(paths) == 150 and seen == set(range(6019))
    if not partial:
        assert complete, f'Incomplete: {len(paths)} scenes / {len(seen)} frames'
    assert len(tokens) == len(seen)
    d, b, c, cam, touch = np.concatenate(chunks), np.concatenate(gaps), np.concatenate(cores), np.concatenate(cameras), np.concatenate(contacts)
    del chunks, gaps, cores, cameras, contacts
    groups, order = {}, []
    for low, high in BINS:
        name = '0' if low == high else (f'{low}–{high}' if high is not None else f'>{low - 1}')
        selected = (c >= low) & ((c <= high) if high is not None else True)
        groups[name] = summarize(name, selected, d, b, touch)
        order.append(name)
    assert sum(groups[n]['queries'] for n in order) == len(c)
    for t in (5, 10, 20):
        name = f'le_{t}'
        groups[name] = summarize(name, c <= t, d, b, touch)
    for name, mask in dict(zero_all=c == 0, zero_no_camera_core=(c == 0) & ~cam,
                           zero_camera_empty_only=(c == 0) & cam, all_queries=np.ones(len(c), bool)).items():
        groups[name] = summarize(name, mask, d, b, touch)
    if complete:
        old = read(ROOT / 'work_dirs/small_final_query_coverage_full/query_core_distribution_val/summary.json')
        assert len(c) == old['query_frame_instances']
        for a, name in zip(old['bins'], order):
            assert a['queries'] == groups[name]['queries']
    result = dict(status='complete' if complete else 'partial', generated_at=datetime.now().astimezone().isoformat(),
        frames=len(seen), scenes=len(paths), queries=len(c), grid_meters=.4,
        checkpoint_sha256=protocol['checkpoint_sha256'], bin_order=order, groups=groups,
        source_hashes=hashes, reporter_sha256=sha(Path(__file__)),
        audit=dict(unique_tokens=len(tokens), exact_full_frame_partition=complete,
            coverage_bins_match_prior_complete=complete, center_vs_cube_distance_bounds_pass=True))
    write(RUN / 'aggregate/summary.json', result)
    csv_write(RUN / 'aggregate/coverage_distance_groups.csv', [flatten(x) for x in groups.values()])
    csv_write(RUN / 'aggregate/per_scene.csv', rows)
    temp = REPORT.with_suffix('.md.tmp')
    temp.write_text(markdown(result))
    temp.replace(REPORT)
    print(dict(status=result['status'], frames=len(seen), scenes=len(paths)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    report(parser.parse_args().partial)
