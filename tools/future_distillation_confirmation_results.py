"""CPU native occupancy statistics for the approved development replication."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_future2current_regions import grouped_metrics
from tools.summarize_future_distillation_causal import native_tables, write

HORIZONS = ('1s', '2s', '3s')
REPO = Path(__file__).resolve().parents[1]
DOCS = REPO/'analysis/hypotheses/future_distillation_causes'
LABELS = {'task': '学生：仅任务', 'teacher': '学生：正确教师', 'shuffled': '学生：打乱教师'}


def load_counts(path, count):
    with np.load(path) as data:
        result = {key: data[key].copy() for key in ('hist', 'intersection', 'union')}
    expected = {'hist': (count, 3, 18, 18), 'intersection': (count, 3), 'union': (count, 3)}
    for key, shape in expected.items():
        if result[key].shape != shape or not np.isfinite(result[key]).all() or (result[key] < 0).any():
            raise ValueError(f'Invalid native counts {path}/{key}')
    if (result['intersection'] > result['union']).any():
        raise ValueError(f'Intersection exceeds union: {path}')
    return result


def metrics_from_counts(counts):
    hist = counts['hist'].sum(0)
    groups = grouped_metrics(hist)
    class_iou = groups['class_iou']
    occ = 100*counts['intersection'].sum(0)/np.maximum(counts['union'].sum(0), 1)
    return dict(metrics={h: dict(mIoU=float(np.nanmean(class_iou[i, :17])), OccIoU=float(occ[i]))
                         for i, h in enumerate(HORIZONS)},
                groups={h: {g: dict(mIoU=float(groups[g]['macro_iou'][i]),
                                    IoU=float(groups[g]['group_iou'][i]))
                            for g in ('dynamic', 'static')} for i, h in enumerate(HORIZONS)})


def mean_metrics(rows):
    # Average the metrics of independent trained seeds, NOT their predictions
    # or confusion matrices; the latter is a different ensemble experiment.
    return dict(metrics={h: {m: float(np.mean([r['metrics'][h][m] for r in rows]))
                             for m in ('mIoU', 'OccIoU')} for h in HORIZONS},
                groups={h: {g: {m: float(np.mean([r['groups'][h][g][m] for r in rows]))
                                 for m in ('mIoU', 'IoU')} for g in ('dynamic', 'static')}
                        for h in HORIZONS})


def final_rows(report, manifest):
    result = {}
    for supervision in manifest['supervisions']:
        rows = []
        for seed in manifest['seeds']:
            row = report.get('arms', {}).get(f'{supervision}_seed{seed}', {})
            value = row.get('epochs', {}).get(str(manifest['epochs']))
            if row.get('status') == 'complete' and value is not None:
                rows.append((seed, value))
        result[supervision] = rows
    return result


def paired_scene_bootstrap(left, right, scenes, repetitions=1000, seed=20261007):
    """Pair the same seed and scene, reporting fixed-seed and nested intervals."""
    if set(left) != set(right) or not left:
        raise ValueError('Paired seed identities differ or are empty')
    seeds = sorted(left)
    unique = sorted(set(scenes))
    lookup = {s: i for i, s in enumerate(unique)}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(unique), np.ones(len(unique))/len(unique), size=repetitions)
    sampled_seeds = rng.integers(len(seeds), size=(repetitions, len(seeds)))

    def scores(counts):
        hist = np.zeros((len(unique), 3, 18, 18), dtype=np.float64)
        for value, scene in zip(counts['hist'], scenes):
            hist[lookup[scene]] += value
        if len(counts['hist']) != len(scenes):
            raise ValueError('Scene/sample identity length differs')
        sampled = np.einsum('bs,shij->bhij', weights, hist, optimize=True)
        iou = grouped_metrics(sampled)['class_iou']
        return np.nanmean(iou[..., :17], axis=-1)

    differences = np.stack([scores(left[s])-scores(right[s]) for s in seeds])
    conditional = differences.mean(0)
    nested = differences[sampled_seeds, np.arange(repetitions)[:, None]].mean(1)
    point = np.stack([[metrics_from_counts(left[s])['metrics'][h]['mIoU']-
                       metrics_from_counts(right[s])['metrics'][h]['mIoU'] for h in HORIZONS]
                      for s in seeds])

    def intervals(values):
        result = {h: np.quantile(values[:, i], [.025, .975]).tolist() for i, h in enumerate(HORIZONS)}
        result['mean_1_2_3s'] = np.quantile(values.mean(-1), [.025, .975]).tolist()
        return result

    return dict(seeds=seeds, scenes=len(unique), repetitions=repetitions,
                delta_pp={h: float(point[:, i].mean()) for i, h in enumerate(HORIZONS)},
                mean_delta_pp=float(point.mean()),
                per_seed_mean_delta_pp={str(s): float(point[i].mean()) for i, s in enumerate(seeds)},
                conditional_scene_ci95=intervals(conditional),
                nested_seed_scene_ci95=intervals(nested),
                caveat='Exploratory intervals on reused development scenes; three seeds give limited seed-variance resolution. Not independent confirmation or equivalence.')


def write_report(root):
    root = Path(root)
    report_path, manifest_path = root/'results.json', root/'manifest.json'
    if not report_path.exists() or not manifest_path.exists():
        return
    report, manifest = json.loads(report_path.read_text()), json.loads(manifest_path.read_text())
    docs = root/'reports' if manifest['smoke'] else DOCS/'s3_confirmation'
    arms = report.get('arms', {})
    rows = [
        '# S3 开发集长程复验：实际进度与原生头结果', '',
        '**这次获得授权的是旧开发集复验，不是独立场景确认。** '
        '它能检查扩大样本和训练预算后的稳定性，但不能补足新的确认场景。', '',
        f'状态：`{report["status"]}`；phase：`{report.get("phase", "—")}`。', '',
        '## 1．预先固定的设置', '',
        '| 项目 | 设置 |', '|---|---|',
        f'| 训练／验证样本 | {len(manifest["train"]["indices"])}／{len(manifest["val"]["indices"])} |',
        f'| 验证场景 | {len(set(manifest["val"]["scenes"]))}；均为开发场景 |',
        f'| 种子 | {manifest["seeds"]} |',
        f'| 固定最终 epoch | {manifest["epochs"]} |',
        '| 学生输入 | 当前／历史图像；不读未来图像 |',
        '| 学生参数 | 全部允许训练；固定 BN 运行统计，affine 仍可训练 |',
        '| 输出 | 原生未来语义／位置／速度分支，保留 future-born queries |',
        '| 结构 | 与 S0 续实验相同的独立封装和已有残差 MLP；没有新探针 |', '',
        '## 2．逐组进度', '',
        '| 监督 | 种子 | 更新次数 | 状态 | 峰值 torch 分配 GiB |', '|---|---:|---:|---|---:|',
    ]
    for seed in manifest['seeds']:
        for supervision in manifest['supervisions']:
            row = arms.get(f'{supervision}_seed{seed}', {})
            rows.append(f'| {LABELS[supervision]} | {seed} | {row.get("progress", 0)}/{manifest["updates"]} | '
                        f'{row.get("status", "not_started")} | {row.get("peak_allocated_gib", "—")} |')
    if report.get('error'):
        rows.extend(['', '运行错误已记录；没有将失败状态标为完成：', '', f'`{report["error"]}`'])
    if 'baseline' in report:
        records = [('原版 SparseWorld', report['baseline'])]
        if 'teacher' in report:
            records.append(('真实未来感知教师', report['teacher']))
        rows.extend(['', '## 3．同一验证集合的参照', '', *native_tables(records, report['baseline']), '',
                     '教师读取目标时刻图像作感知，其分数不属于可部署预测成绩。'
                     '所有括号正负差值只相对本表原版 SparseWorld。'])
        complete = final_rows(report, manifest)
        rows.extend(['', '## 4．固定最终 epoch 的结果', ''])
        for supervision, completed in complete.items():
            if not completed:
                rows.append(f'{LABELS[supervision]}：尚无完成最终 epoch 的种子。\n')
                continue
            records = [('原版 SparseWorld', report['baseline'])]
            records.extend((f'{LABELS[supervision]}／seed{seed}', result) for seed, result in completed)
            if len(completed) == len(manifest['seeds']):
                records.append((f'{LABELS[supervision]}／全部种子指标均值', mean_metrics([v for _, v in completed])))
            rows.extend(['', f'### {LABELS[supervision]}', '', *native_tables(records, report['baseline'])])
        rows.extend(['', '## 5．逐条回答假设', '',
                     '| 问题 | 现阶段可说的结论 |', '|---|---|',
                     '| 长预算下教师是否有净作用？ | 九组完成前不作最终判断；不挑最高中间 epoch |',
                     '| R1：同条件未来是否不唯一？ | 本组不是多未来分支实验，不能验证 |',
                     '| R2：相似是否等于正确？ | 需要结合实际特征误差和任务分数，不由损失下降单独定论 |',
                     '| R3：冲突是否是主要原因？ | 本组三监督对照不单独识别梯度方向原因 |',
                     '| S5 是否能正式组合？ | 需有两项独立确认有效的修复，本组不能自动满足 |'])
        analysis_path = root/'paired_analysis.json'
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text())
            rows.extend(['', '### 已完成种子的配对比较（开发集，不是独立确认）', '',
                         '| 比较 | 平均 mIoU 差值 | 嵌套种子／场景 95% 区间 | 完成配对种子数 |',
                         '|---|---:|---|---:|'])
            for name, value in analysis['comparisons'].items():
                ci = value['nested_seed_scene_ci95']['mean_1_2_3s']
                rows.append(f'| {name} | {value["mean_delta_pp"]:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {len(value["seeds"])} |')
            rows.extend(['', '此处差值是单列的因果比较，不混入仅相对基线的主性能表。'
                         '区间跨零不代表等效；三种子仍不能精确排除很小效应。'])
            if analysis.get('interpretation'):
                rows.extend(['', analysis['interpretation']])
    alignment_path=root/'alignment_audit.json'
    alignment=json.loads(alignment_path.read_text()) if alignment_path.exists() else None
    rows.extend(['', '## 6．机制审计与未完成项', '',
                 '固定特征目标审计使用预先选定的验证子集；固定 query 槽位不保证训练后物理位置完全相同。'
                 '原版头占据成绩不由该特征审计代替。', ''])
    if alignment:
        rows.extend([f'特征审计状态：`{alignment["status"]}`。', '',
                     '| 模型 | 固定目标特征损失 | 移动匹配特征损失 | 样本数 |', '|---|---:|---:|---:|'])
        for label,value in alignment['cases'].items():
            rows.append(f'| {label} | {value["fixed_feature_loss"]:.6f} | {value["moving_feature_loss"]:.6f} | {len(value["rows"])} |')
        rows.extend(['', '两种特征损失都越低越相似；它们不是 mIoU。固定目标便于比较，但训练后位置变化仍可能导致语义对应变化。'
                     '逐样本覆盖、最近教师距离的均值／中位数／90 分位见 `alignment_audit.json`。'])
    else:
        rows.append('机制审计尚未执行，不能声称学生已被拉近到教师。')
    rows.extend(['',
                 f'原始目录：`{root}`。复现入口和限制见 [设计](design.md)。' if not manifest['smoke'] else f'这是烟雾测试：`{root}`，不计入正式结果。'])
    write(docs/'results.md', rows)


def analyze(root, repetitions=1000):
    root = Path(root)
    report, manifest = [json.loads((root/name).read_text()) for name in ('results.json', 'manifest.json')]
    complete = final_rows(report, manifest)
    counts = {name: {seed: load_counts(value['rows'], len(manifest['val']['scenes']))
                     for seed, value in rows} for name, rows in complete.items()}
    comparisons = {}
    for a, b in [('teacher', 'task'), ('teacher', 'shuffled'), ('shuffled', 'task')]:
        seeds = sorted(set(counts[a]) & set(counts[b]))
        if seeds:
            comparisons[a+' vs '+b] = paired_scene_bootstrap(
                {s: counts[a][s] for s in seeds}, {s: counts[b][s] for s in seeds},
                manifest['val']['scenes'], repetitions)
    full = all(len(rows) == len(manifest['seeds']) for rows in complete.values())
    interpretation = '训练尚未全部完成；配对比较仅对应已完成的种子，不作最终判断。'
    if full:
        key = comparisons['teacher vs task']
        ci = key['nested_seed_scene_ci95']['mean_1_2_3s']
        if ci[0] > 0:
            interpretation = '开发集长预算复验支持正确教师相对仅任务的正作用；还需核对打乱组、时距一致性和独立确认，不能直接提升到独立因果确认等级。'
        elif ci[1] < 0:
            interpretation = '开发集长预算复验观察到正确教师的额外损害；本组三组对照不能单独确定 R1、R2、R3 哪项造成它。'
        else:
            interpretation = '开发集长预算复验尚未建立教师净效应方向；区间跨零不证明等效或绝对无效。'
    value = dict(status='complete_development_only' if full else 'partial', comparisons=comparisons,
                 interpretation=interpretation, independent_confirmation=False)
    temporary = root/'paired_analysis.json.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(root/'paired_analysis.json')
    return value
