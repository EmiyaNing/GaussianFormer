"""Render ``p1_eval.json`` as Markdown tables for the P1 result report.

    python3.10 benchmarks/p1/summarize_p1.py benchmarks/results/p1_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VARIANT_LABELS = {
    'a0_base': 'A0 无 adapter（基线）',
    'a2_knn_k215': 'A2′ KNN-215 均值（名义感受野匹配）',
    'a2_knn_k59': 'A2″ KNN-59 均值（有效感受野匹配）',
    'a4_fixed_gaussian': 'A4 高斯，固定各向同性几何',
    'a5_cap0': 'A5 高斯，cap=0（无截断）',
    'a5_cap8': 'A5 高斯，cap=8',
    'a5_cap32': 'A5 高斯，cap=32',
    'a5_cap64': 'A5 高斯，cap=64',
    'a5_cap128': 'A5 高斯，cap=128',
    'a1_pointwise': 'A1 逐点 MLP（无邻域）',
    'a3_graph_attention': 'A3 kNN 图注意力',
    'a2_knn_mean': 'A2 KNN-8 均值',
    'a2_knn_gaussian8': 'A2-gauss KNN-8 + 高斯权重（固定尺度）',
    'a2_knn_max': 'A2 KNN-8 最大值',
    'a5_s43': 'A5 高斯 seed43',
    'a5_s44': 'A5 高斯 seed44',
    'a2_s43': 'A2 KNN 均值 seed43',
    'a2_s44': 'A2 KNN 均值 seed44',
}


def value(rows, variant, key):
    for row in rows:
        if row.get('variant') == variant:
            return (row.get('metrics') or {}).get(key)
    return None


def fmt(number):
    if number is None:
        return '-'
    if isinstance(number, float):
        return f'{number:.2f}'
    return str(number)


def horizon_cell(recorded, comparable, iou):
    """Render one ``mIoU / IoU`` cell, flagging a divisor mismatch.

    When the recorded and fixed-divisor per-horizon mIoU differ by more than
    0.05 the recorded number is arithmetic rather than model behaviour, so both
    are shown as ``recorded⚠️fixed``.  Without this flag the per-horizon table is
    exactly where the artifact hides: A5 seed 44's 4 s reads 13.36 recorded but
    14.25 fixed, and A2 seed 43's 4 s reads 13.38 but 14.27.
    """
    if (recorded is not None and comparable is not None
            and abs(recorded - comparable) > 0.05):
        miou = f'{recorded:.2f}⚠️{comparable:.2f}'
    else:
        miou = fmt(recorded)
    return f'{miou} / {fmt(iou)}'


def seed_aggregation(rows, fixed, groups):
    """Per-group ``(label, n, mean, spread, values)``, fixed divisor preferred.

    ``rows`` carries the recorded metrics and ``fixed`` the recomputed
    fixed-divisor mIoU (``{variant: avg}``).  The recorded value's divisor changes
    with the finite-class set, which inflated the apparent A5 seed spread from the
    real 0.05 to 0.38 (P1-5, third instance), so the aggregate must prefer
    ``fixed`` and only fall back to the recorded value when a log is unavailable.
    """
    aggregated = []
    for label, members in groups.items():
        values = []
        for name in members:
            item = fixed.get(name)
            if item is None:
                item = value(rows, name, 'avg/mIoU')
            if item is not None:
                values.append(item)
        if len(values) >= 2:
            aggregated.append((label, len(values), sum(values) / len(values),
                               max(values) - min(values), values))
    return aggregated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_path')
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.json_path).read_text(encoding='utf-8'))
    present = [row['variant'] for row in rows
               if (row.get('metrics') or {})]

    # Fixed-divisor mIoU, recomputed from the per-class IoU in each eval log.
    # The recorded ``avg/mIoU`` is ``nanmean`` over 17 classes, so its divisor
    # depends on which classes happen to be finite in that run -- a run with one
    # extra finite class scores lower for purely arithmetic reasons.  See
    # ``P1-5_metric_divisor_artifact.md``: the two A5 seeds differ by 0.16 in the
    # recorded value and by -0.05 once the divisor is fixed.  Reported side by
    # side so neither number can be quoted without the other.
    fixed = {}
    fixed_horizon = {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from benchmarks.p1.common_class_miou import comparable_for, load
        logs = load(present)
        if logs:
            table, _ = comparable_for(logs, present)
            fixed = {name: values['avg'] for name, values in table.items()}
            fixed_horizon = {name: values.get('per_horizon')
                             for name, values in table.items()}
    except Exception as error:  # pragma: no cover - reported, never silent
        print(f'# 警告：固定分母 mIoU 重算失败（{error}）；此列留空')

    lines = ['| 变体 | avg/mIoU（记录） | avg/mIoU（固定分母） | avg/IoU | '
             'avg/boundaryIoU | 占用 P / R | num_samples | 相对 A0（固定分母） |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    base = fixed.get('a0_base', value(rows, 'a0_base', 'avg/mIoU'))
    for name in present:
        miou = value(rows, name, 'avg/mIoU')
        iou = value(rows, name, 'avg/IoU')
        samples = value(rows, name, 'num_samples')
        comparable = fixed.get(name)
        delta = '-' if (base is None or comparable is None or name == 'a0_base') \
            else f'{comparable - base:+.2f}'
        boundary = value(rows, name, 'avg/boundaryIoU')
        # Occupied precision/recall: the pair that separates "misses occupied
        # space" from "hallucinates occupied space", which both collapse to the
        # same IoU.  Definition-gated by ``parse_metrics`` (a run without the
        # pooled flag contributes no average), so a missing value prints '-'
        # rather than a stale number.
        precision = value(rows, name, 'avg/occupiedPrecision')
        recall = value(rows, name, 'avg/occupiedRecall')
        occupancy = ('-' if precision is None or recall is None
                     else f'{precision:.2f} / {recall:.2f}')
        flag = ''
        if (comparable is not None and miou is not None
                and abs(comparable - miou) > 0.05):
            flag = ' ⚠️'
        lines.append(f'| {VARIANT_LABELS.get(name, name)}{flag} | {fmt(miou)} | '
                     f'{fmt(comparable)} | {fmt(iou)} | {fmt(boundary)} | '
                     f'{occupancy} | {fmt(samples)} | {delta} |')
    if any(v is not None for v in fixed.values()):
        lines.append('')
        lines.append('> ⚠️ **两列的含义**：`记录` 是官方协议的 `nanmean(semantic[:17])`，'
                     '其分母随运行变化；`固定分母` 是把 mIoU 重算在所有变体共有的类别集上。'
                     '带 ⚠️ 的行说明两者相差 > 0.05，**跨运行比较必须用固定分母列**。'
                     '详见 [P1-5](P1-5_metric_divisor_artifact.md)。')
        lines.append('> `占用 P / R` 是**合并计数**的占用 precision/recall，'
                     '把"漏掉占用空间"与"凭空多占"分开——两者在 IoU 上是同一个数。'
                     '它是**定义门控**的：解析器只接受带 `avg/occupiedPrecision_pooled` '
                     '标志的日志，所以尚未重评估的变体显示 `-`，而不是一个旧值。')

    lines.append('')
    lines.append('### 每时域明细（mIoU / IoU；mIoU 在分母不一致时标出固定分母值）')
    lines.append('')
    lines.append('| 变体 | 0s | 2s | 4s | 6s |')
    lines.append('| --- | ---: | ---: | ---: | ---: |')
    for name in present:
        cells = []
        per_horizon_fixed = fixed_horizon.get(name) or [None] * 4
        for position, horizon in enumerate((0, 2, 4, 6)):
            cells.append(horizon_cell(
                value(rows, name, f'{horizon}s/mIoU'),
                per_horizon_fixed[position],
                value(rows, name, f'{horizon}s/IoU')))
        lines.append(f'| {VARIANT_LABELS.get(name, name)} | ' +
                     ' | '.join(cells) + ' |')
    if any(v is not None for v in fixed_horizon.values()):
        lines.append('')
        lines.append('> ⚠️ 的读法：`记录⚠️固定分母`。mIoU 的**记录值**用 '
                     '`nanmean(semantic[:17])`，分母随运行的有限类别数变化；'
                     '标出的第二个数是同一 horizon 在所有变体共有的类别集上重算的值。'
                     '**跨运行比较只能用第二个数**（[P1-5](P1-5_metric_divisor_artifact.md)）。'
                     'IoU 不受影响（占用二值矩阵的类别集固定）。')

    # Seed aggregation for the decisive pair, when the repeats exist.
    #
    # Uses the **fixed-divisor** value: the recorded mIoU's divisor changes with
    # the finite-class set, which inflated the apparent A5 seed spread from the
    # real 0.05 to 0.38 (P1-5, third instance).  Quoting the recorded spread here
    # would contradict the fixed-divisor column three lines above and overstate
    # the training-side noise by ~7x.
    groups = {'a5': ['a5_cap0', 'a5_s43', 'a5_s44'],
              'a2': ['a2_knn_mean', 'a2_s43', 'a2_s44']}
    aggregated = seed_aggregation(rows, fixed, groups)
    if aggregated:
        lines.append('')
        lines.append('### 决定性对比的 seed 聚合（A5 高斯 vs A2 KNN 均值，'
                     '固定分母）')
        lines.append('')
        lines.append('| 变体 | n seeds | 平均 mIoU（固定分母） | 极差 (max-min) '
                     '| 各 seed（固定分母） |')
        lines.append('| --- | ---: | ---: | ---: | --- |')
        for label, count, mean, spread, values in aggregated:
            rendered = ', '.join(f'{item:.2f}' for item in values)
            lines.append(f'| {label} | {count} | {mean:.2f} | {spread:.2f} | {rendered} |')
        if len(aggregated) == 2:
            gap = aggregated[0][2] - aggregated[1][2]
            noise = max(aggregated[0][3], aggregated[1][3])
            verdict = ('高斯领先超过 seed 极差' if gap > noise else
                       '差异落在 seed 极差之内，无法区分')
            lines.append('')
            lines.append(f'A5 - A2 = {gap:+.2f} mIoU；最大 seed 极差 = {noise:.2f} '
                         f'-> **{verdict}**')

    failures = [row for row in rows if not (row.get('metrics') or {})]
    if failures:
        lines.append('')
        lines.append('### 失败的变体')
        lines.append('')
        for row in failures:
            lines.append(f"- `{row['variant']}`: {row.get('error', 'unknown')[:300]}")

    print('\n'.join(lines))


if __name__ == '__main__':
    main()
