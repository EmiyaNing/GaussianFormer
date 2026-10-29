"""Count-audited report of actual final-level GS coverage by parent query."""
from __future__ import annotations
import argparse
from datetime import datetime
import csv
import fcntl
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.stat_boundary_gaussians import read, write
from tools.stat_query_final_coverage import RUN, RANGES, top20

REPORT = ROOT / 'analysis/query_final_level_coverage_results.md'
NAMES = dict(core_1sigma='1σ 核心', strict_3sigma='严格 3σ 椭球', native_support='原生正贡献范围')


def add(target, source):
    for k, v in source.items():
        if isinstance(v, dict):
            add(target.setdefault(k, {}), v)
        else:
            target[k] = target.get(k, 0) + v


def p(n, d):
    return f'{100 * n / d:.3f}%' if d else '—'


def scene(path, protocol):
    meta = read(path)
    assert meta['status'] == 'complete' and meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert [r['index'] for r in meta['records']] == meta['indices']
    with np.load(path.with_suffix('.npz')) as z:
        arrays = {k: z[k] for k in z.files}
    assert len(meta['records']) == len(arrays['load_sixth'])
    totals = {}
    for i, row in enumerate(meta['records']):
        fg = row['foreground']
        for name in ('fifth', 'sixth'):
            load = arrays['load_' + name][i]
            assert int(load.sum()) == fg == row['nearest'][name]['foreground']
            assert int(load[top20(load)].sum()) == row['nearest'][name]['top20_foreground']
        dom = arrays['dominant_foreground'][i]
        assert int(dom.sum()) == row['dominant_foreground']['attributed']
        assert int(dom[top20(dom)].sum()) == row['dominant_foreground']['top20_attributed']
        for name in RANGES:
            r = row['supports'][name]
            coverage = arrays[name + '_query_coverage'][i]
            assert int(coverage.sum()) == r['query_voxel_pairs']
            assert 0 <= r['all_covered'] <= fg
            for group, counts in r['groups'].items():
                load_name = {'fifth_top20': 'load_fifth', 'sixth_top20': 'load_sixth'}.get(group)
                values = arrays[load_name][i] if load_name else coverage
                ids = top20(values)
                assert int(coverage[ids].sum()) == counts['top_query_voxel_pairs']
                assert counts['top_union'] + counts['remaining_union'] - counts['both_groups'] == r['all_covered']
                assert counts['top_union'] == counts['top_only'] + counts['both_groups']
                assert counts['remaining_union'] == counts['remaining_only'] + counts['both_groups']
                assert counts['top_union'] <= counts['top_query_voxel_pairs']
        add(totals, {k: v for k, v in row.items() if k not in ('index', 'token')})
    return meta, totals


def markdown(result):
    lines = ['# 最后一级 GS：Query 归属与实际覆盖的区别', '',
        f"**状态：{'全量统计完成' if result['status'] == 'complete' else '部分数据，不能作为全量结论'}。**", '',
        '本轮使用产生原始“20% query 对应45.071%前景”结果的 small 权重。模型共六级，最终级为1,200个query×64个GS。'
        '权重冻结，默认统计camera mask内的GT前景，前景类别之间暂不作区分。', '',
        '比较两个问题：**将最近中心从第五级换成第六级，前景归属会怎样变化；按第六级GS实际覆盖排序，20% query又覆盖多少前景。**', '',
        '| 划分 | 场景 | 帧 | 状态 |', '|---|---:|---:|---|']
    for split, data in result['splits'].items():
        lines.append(f"| {'验证集' if split == 'val' else '训练集'} | {data['scenes']} / {data['expected']['scenes']} | {data['frames']} / {data['expected']['frames']} | {'完成' if data['status'] == 'complete' else '进行中'} |")
    lines.append('')
    val = result['splits']['val']
    if val['status'] == 'complete':
        x = val['counts']
        s = x['supports']['core_1sigma']
        g = s['groups']['coverage_top20']
        lines += ['**完整验证集直接答案：**', '',
            f"- 最近中心从第五级换成第六级后，最高负载20%的query对应前景由 {p(x['nearest']['fifth']['top20_foreground'], x['foreground'])} 变为 **{p(x['nearest']['sixth']['top20_foreground'], x['foreground'])}**。",
            f"- 改成按最终GS的1σ实际覆盖数排序，前20% query的覆盖并集为全部GT前景的 **{p(g['top_union'], x['foreground'])}**，占全部已获1σ覆盖前景的 **{p(g['top_union'], s['all_covered'])}**。",
            '- 层级变化没有改变模型：最近邻归属比例与实际覆盖比例是不同统计。实际覆盖有重叠，不能用100%减去前20%的覆盖得到其余80%的覆盖。', '']
    for split, data in result['splits'].items():
        if not data['frames']:
            continue
        x = data['counts']
        fg = x['foreground']
        lines += ['', f"## {'验证集' if split == 'val' else '训练集'}", '',
            '### 先只改变最近中心所在的层级', '',
            '| 最近邻归属依据 | 最高负载20% query对应的GT前景 | 归属query自身的最终1σ核心覆盖 / 全部GT前景 |', '|---|---:|---:|']
        for name, label in (('fifth', '第五级中心'), ('sixth', '第六级中心')):
            row = x['nearest'][name]
            lines.append(f"| {label} | {p(row['top20_foreground'], fg)} | {p(row['own_core_foreground'], fg)} |")
        lines += ['', '两行仍是最近邻分区：即使一个体素没有被任何GS覆盖，也会分给最近中心所属的query。'
            '第二列各自重新选择最高负载的240个query，不是固定同一组query。', '',
            '### 改用第六级的实际支持范围', '',
            '对每个query，合并其64个GS的支持，统计接触到多少个不同GT前景体素，再选覆盖数最高的240个query。'
            '然后对这240个query再次取覆盖并集，得到下表的去重覆盖。', '',
            '| 第六级支持定义 | 全部query覆盖 / 全部GT前景 | 覆盖最高20% query的并集 / 全部GT前景 | 该并集 / 全部query已覆盖前景 | 其余80% query的并集 / 全部GT前景 | 两组共同覆盖 / 全部GT前景 |',
            '|---|---:|---:|---:|---:|---:|']
        for name in RANGES:
            s = x['supports'][name]
            g = s['groups']['coverage_top20']
            lines.append(f"| {NAMES[name]} | {p(s['all_covered'], fg)} | {p(g['top_union'], fg)} | {p(g['top_union'], s['all_covered'])} | {p(g['remaining_union'], fg)} | {p(g['both_groups'], fg)} |")
        lines += ['', '**两组覆盖有交集，因此20%的覆盖和其余80%的覆盖不能相加当作全体覆盖。**'
            '上表已经在每组内去重；不存在简单的“这一组45%，另一组必然55%”关系。'
            '各支持定义分别排序，选中的20% query也可能不同。'
            '这里按单query覆盖数取前240个，并未求解覆盖并集最大的240个query；'
            '单query覆盖多的候选可能互相重叠，另一组query的并集有时反而更大。', '',
            '1σ与严格3σ使用旋转、各轴尺度定义的Mahalanobis距离，并限定在生产算子的候选范围内。'
            '原生范围采用实际候选盒与FP32正贡献条件，可能包含椭球外很弱的尾部。'
            '这些数字是几何支持，不代表类别正确、足够强的前景证据或最终预测正确。', '',
            '### 固定原第五级高负载组，再看它的实际覆盖', '',
            '| 支持定义 | 原高负载20%的并集 / 全部GT前景 | 其余80%的并集 / 全部GT前景 | 两组共同覆盖 / 全部GT前景 |', '|---|---:|---:|---:|']
        for name in RANGES:
            g = x['supports'][name]['groups']['fifth_top20']
            lines.append(f"| {NAMES[name]} | {p(g['top_union'], fg)} | {p(g['remaining_union'], fg)} | {p(g['both_groups'], fg)} |")
        lines += ['', '这里保留原来的分组，说明“原先被分给45%左右前景的query”实际提供了多少支持；'
            '它与上一表重新按覆盖选组的结果回答不同问题。', '',
            '### 如果按每个Query的覆盖次数求份额', '',
            '| 支持定义 | 覆盖最高20% query占全部query–体素对应次数 |', '|---|---:|']
        for name in RANGES:
            s = x['supports'][name]
            g = s['groups']['coverage_top20']
            lines.append(f"| {NAMES[name]} | {p(g['top_query_voxel_pairs'], s['query_voxel_pairs'])} |")
        lines += ['', '这一表允许同一体素分别计入多个query，因此不能写成“占全部前景体素的比例”。'
            '它是覆盖工作量的集中度补充，主结论优先看前面的去重并集。', '']
        dom = x['dominant_foreground']
        lines += ['### 按最大单个前景分数贡献追溯Query', '',
            f"在每个GT前景体素，找到对该位置最高前景类别分数贡献最大的单个GS，再追溯其query。"
            f"按此计数排序的前20% query，对应全部GT前景的 **{p(dom['top20_attributed'], fg)}**，"
            f"占存在正前景贡献的GT体素的 **{p(dom['top20_attributed'], dom['attributed'])}**。", '',
            '这是原生渲染分数的主导来源，不要求该前景类别等于GT，最终预测为empty的GT前景也参与。'
            '其余GS仍可能有贡献，不能把最大单项归属解释为该query独立完成了预测。', '']
    lines += ['## 能够得出的结论与边界', '',
        '第五级或第六级最近邻负载，衡量的是空间分区中的前景数量。实际支持并集衡量几何接触，'
        '主导贡献归属衡量某个前景分数由谁贡献最多。这三种口径不能互相替代。'
        '本次没有用GT类别筛选候选，也没有通过删GS测量各query的必要性。', '',
        '统计对每一帧复现旧第五级负载、旧最终1σ逐query覆盖和总核心缺口；'
        '逐场景保留索引和sample token，检查所有query支持并集的集合恒等式。'
        '运行检查还与旧CUDA位集逐位比较，并在三种范围中直接复核GT位置的全部query成员关系。', '',
        '- [计数汇总](../work_dirs/small_final_query_coverage_full/aggregate/summary.json)',
        '- [逐场景表](../work_dirs/small_final_query_coverage_full/aggregate/per_scene.csv)',
        '- [源文件与权重协议](../work_dirs/small_final_query_coverage_full/protocol.json)',
        '- [统计实现](../tools/stat_query_final_coverage.py)', '',
        '```bash', 'python3.10 tools/report_query_final_coverage.py', '```', '']
    return '\n'.join(lines)


def report(partial=False):
    protocol = read(RUN / 'protocol.json')
    result = dict(status='partial', generated_at=datetime.now().astimezone().isoformat(),
        checkpoint_sha256=protocol['checkpoint_sha256'], splits={})
    rows = []
    tokens = set()
    for split in ('val', 'train'):
        counts, indices, scenes = {}, set(), 0
        for path in sorted((RUN / 'scenes' / split).glob('*.json')):
            meta, total = scene(path, protocol)
            assert meta['split'] == split and not indices.intersection(meta['indices'])
            for r in meta['records']:
                assert r['token'] not in tokens
                tokens.add(r['token'])
            indices.update(meta['indices'])
            scenes += 1
            add(counts, total)
            fg = total['foreground']
            row = dict(split=split, scene=meta['scene'], frames=len(meta['indices']), foreground=fg,
                fifth_top20_foreground=total['nearest']['fifth']['top20_foreground'],
                sixth_top20_foreground=total['nearest']['sixth']['top20_foreground'])
            for name in RANGES:
                s = total['supports'][name]
                row[name + '_all_covered'] = s['all_covered']
                for group, g in s['groups'].items():
                    for k, value in g.items():
                        row[name + '_' + group + '_' + k] = value
            rows.append(row)
        expected = protocol['expected'][split]
        marker_path = RUN / (split + '_complete.json')
        marker = read(marker_path) if marker_path.exists() else None
        if marker is not None:
            assert marker['status'] == 'complete'
            assert all(marker[k] == expected[k] for k in ('frames', 'scenes'))
        complete = (marker is not None and len(indices) == expected['frames'] and scenes == expected['scenes'])
        if complete:
            assert indices == set(range(expected['frames']))
        if not partial:
            assert complete, f'{split} incomplete'
        result['splits'][split] = dict(status='complete' if complete else 'partial', scenes=scenes,
            frames=len(indices), expected=expected, counts=counts)
    if all(v['status'] == 'complete' for v in result['splits'].values()):
        result['status'] = 'complete'
    write(RUN / 'aggregate/summary.json', result)
    path = RUN / 'aggregate/per_scene.csv'
    with path.with_suffix('.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['split', 'scene'])
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix('.csv.tmp').replace(path)
    tmp = REPORT.with_suffix('.md.tmp')
    tmp.write_text(markdown(result))
    tmp.replace(REPORT)
    print({s: (r['status'], r['scenes'], r['frames']) for s, r in result['splits'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    with (RUN / 'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report(args.partial)
