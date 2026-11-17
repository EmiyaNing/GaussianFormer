"""Render the P0.2 cost matrix (JSONL) as Markdown tables for the P0 report.

    python3.10 benchmarks/summarize_p0.py benchmarks/results/p0_matrix.jsonl
    python3.10 benchmarks/summarize_p0.py benchmarks/results/p0_matrix.jsonl --write
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value, digits=3):
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def core_g2q(rows):
    lines = [
        '| 分布 | G | 前向 p50 (ms) | 前向 p95 (ms) | 反向 p50 (ms) | 峰值分配 (MiB) | 候选 pair 数 | 局部度数 p95 | 局部度数 max |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        if row.get('group') != 'core' or row.get('op') != 'g2q':
            continue
        forward = row['forward']
        pairs = row.get('pairs', {})
        lines.append(
            f"| {row['dist']} | {row['gaussians']:,} | "
            f"{fmt(forward['p50_ms'])} | {fmt(forward['p95_ms'])} | "
            f"{fmt(row['backward_derived']['p50_ms'])} | "
            f"{fmt(forward['peak_alloc_mib'], 1)} | "
            f"{pairs.get('candidate_pairs', 0):,} | "
            f"{fmt(pairs.get('degree_p95'), 0)} | "
            f"{fmt(pairs.get('degree_max'), 0)} |")
    return '\n'.join(lines)


def core_g2v(rows):
    lines = [
        '| 分布 | G | 前向 p50 (ms) | 前向 p95 (ms) | 反向 p50 (ms) | 峰值分配 (MiB) | L (support∩grid) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        if row.get('group') != 'core' or row.get('op') != 'g2v':
            continue
        forward = row['forward']
        pairs = row.get('pairs', {})
        lines.append(
            f"| {row['dist']} | {row['gaussians']:,} | "
            f"{fmt(forward['p50_ms'])} | {fmt(forward['p95_ms'])} | "
            f"{fmt(row['backward_derived']['p50_ms'])} | "
            f"{fmt(forward['peak_alloc_mib'], 1)} | "
            f"{pairs.get('L_pairs', 0):,} |")
    return '\n'.join(lines)


def ratio_table(rows):
    """Per-distribution G2Q vs G2V memory/latency ratio at the largest G."""
    g2q = {}
    g2v = {}
    for row in rows:
        if row.get('group') != 'core':
            continue
        key = (row['dist'], row['gaussians'])
        if row.get('op') == 'g2q':
            g2q[key] = row
        elif row.get('op') == 'g2v':
            g2v[key] = row
    lines = [
        '| 分布 | G | G2Q 峰值 (MiB) | G2V 峰值 (MiB) | 显存比 (G2Q/G2V) | G2Q 前向 (ms) | G2V 前向 (ms) | 延迟比 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for key in sorted(set(g2q) & set(g2v), key=lambda k: (k[0], k[1])):
        a, b = g2q[key], g2v[key]
        memory_ratio = a['forward']['peak_alloc_mib'] / max(b['forward']['peak_alloc_mib'], 1e-9)
        time_ratio = a['forward']['p50_ms'] / max(b['forward']['p50_ms'], 1e-9)
        lines.append(
            f"| {key[0]} | {key[1]:,} | "
            f"{fmt(a['forward']['peak_alloc_mib'], 1)} | "
            f"{fmt(b['forward']['peak_alloc_mib'], 1)} | "
            f"{memory_ratio:.3f} | "
            f"{fmt(a['forward']['p50_ms'])} | {fmt(b['forward']['p50_ms'])} | "
            f"{time_ratio:.3f} |")
    return '\n'.join(lines)


def rasterizer_table(rows):
    lines = [
        '| 分布 | G | padding off (ms) | off 峰值 (MiB) | closing on (ms) | on 峰值 (MiB) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        if row.get('group') != 'core' or row.get('op') != 'rasterizer':
            continue
        off, on = row['padding_off'], row['padding_on']
        lines.append(
            f"| {row['dist']} | {row['gaussians']:,} | "
            f"{fmt(off['p50_ms'])} | {fmt(off['peak_alloc_mib'], 1)} | "
            f"{fmt(on['p50_ms'])} | {fmt(on['peak_alloc_mib'], 1)} |")
    return '\n'.join(lines)


def knn_table(rows):
    lines = [
        '| 分布 | G | 前向 p50 (ms) | 反向 p50 (ms) | 峰值分配 (MiB) |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        if row.get('group') != 'core' or row.get('op') != 'knn_gather':
            continue
        forward = row['forward']
        lines.append(
            f"| {row['dist']} | {row['gaussians']:,} | "
            f"{fmt(forward['p50_ms'])} | {fmt(row['backward_derived']['p50_ms'])} | "
            f"{fmt(forward['peak_alloc_mib'], 1)} |")
    return '\n'.join(lines)


def sweep_table(rows, group, column):
    header = {
        'cell': '| 分布 | cell (m) | 前向 p50 (ms) | 前向 p95 (ms) | 峰值 (MiB) |',
        'scale': '| 分布 | scale (m) | 前向 p50 (ms) | 前向 p95 (ms) | 峰值 (MiB) |',
        'support': '| 分布 | support (σ) | 前向 p50 (ms) | 前向 p95 (ms) | 峰值 (MiB) |',
        'batch': '| 分布 | batch | 前向 p50 (ms) | 前向 p95 (ms) | 峰值 (MiB) |',
    }[group]
    lines = [header, '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row.get('group') != group or row.get('op') != 'g2q':
            continue
        forward = row['forward']
        lines.append(
            f"| {row['dist']} | {row[column] if column in row else row['params'][column]} | "
            f"{fmt(forward['p50_ms'])} | {fmt(forward['p95_ms'])} | "
            f"{fmt(forward['peak_alloc_mib'], 1)} |")
    return '\n'.join(lines)


def pairs_table(rows):
    lines = [
        '| 分布 | G | 候选 pair 数 | 度数 mean | 度数 p50 | 度数 p95 | 度数 p99 | 度数 max | L (G2V) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        if row.get('group') != 'core' or row.get('op') != 'pairs':
            continue
        g2q, g2v = row['g2q'], row['g2v']
        lines.append(
            f"| {row['dist']} | {row['gaussians']:,} | "
            f"{g2q['candidate_pairs']:,} | {fmt(g2q['degree_mean'], 1)} | "
            f"{fmt(g2q['degree_p50'], 0)} | {fmt(g2q['degree_p95'], 0)} | "
            f"{fmt(g2q['degree_p99'], 0)} | {fmt(g2q['degree_max'], 0)} | "
            f"{g2v['L_pairs']:,} |")
    return '\n'.join(lines)


SECTIONS = (
    ('### G2Q 核心表（分布 × G）', core_g2q),
    ('### G2V 核心表（分布 × G）', core_g2v),
    ('### G2Q vs G2V 显存与延迟比', ratio_table),
    ('### 硬体素化 OPUSRasterizer', rasterizer_table),
    ('### KNN-8 gather 基线', knn_table),
    ('### cell 尺寸扫描（G=76,800）', lambda r: sweep_table(r, 'cell', 'cell_size')),
    ('### scale 扫描（G=76,800）', lambda r: sweep_table(r, 'scale', 'scale')),
    ('### support σ 扫描（G=76,800）', lambda r: sweep_table(r, 'support', 'support_sigma')),
    ('### batch 扫描（G=34,560）', lambda r: sweep_table(r, 'batch', 'batch')),
    ('### pair / 局部度数统计', pairs_table),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jsonl')
    parser.add_argument('--write', action='store_true',
                        help='write to <jsonl>.tables.md instead of stdout')
    args = parser.parse_args(argv)

    rows = load(args.jsonl)
    errors = [row for row in rows if 'error' in row]
    chunks = [f'共 {len(rows)} 条记录，其中失败 {len(errors)} 条。']
    if errors:
        chunks.append('\n失败记录：')
        for row in errors:
            chunks.append(f"- `{row.get('op')}` {row.get('params')}: {row['error']}")
    for title, builder in SECTIONS:
        table = builder(rows)
        chunks.append(f'\n{title}\n\n{table}')
    text = '\n'.join(chunks) + '\n'

    if args.write:
        target = Path(str(args.jsonl) + '.tables.md')
        target.write_text(text, encoding='utf-8')
        print(f'wrote {target}')
    else:
        print(text)


if __name__ == '__main__':
    main()
