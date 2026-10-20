"""Scene-held-out evidence probes and readable reports for frozen mini diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits
import joblib
from tools.center_support_common import write_json

RUN = ROOT / 'work_dirs/empty_decision_mini'
REPORT = ROOT / 'analysis/empty_decision_mini'
REGIONS = ['camera', 'camera_complement', 'all_valid']
LABELS = dict(camera='camera 区域', camera_complement='camera mask 补集', all_valid='全空间')
GROUPS = dict(margin=[0], position=[1, 2, 3, 4], support=[5, 6, 7, 8],
              semantics=[9, 10, 11, 12, 13, 14], image=[15, 16, 17, 18, 19],
              predicted_class=list(range(20, 37)))
MODELS = dict(margin=['margin'], margin_position=['margin', 'position'],
    margin_support=['margin', 'support'], margin_semantics=['margin', 'semantics'],
    margin_image=['margin', 'image'], position_support=['margin', 'position', 'support'],
    position_semantics=['margin', 'position', 'semantics'],
    position_image=['margin', 'position', 'image'],
    all=['margin', 'position', 'support', 'semantics', 'image'],
    all_without_support=['margin', 'position', 'semantics', 'image'],
    all_without_semantics=['margin', 'position', 'support', 'image'],
    all_without_image=['margin', 'position', 'support', 'semantics'],
    margin_position_class=['margin', 'position', 'predicted_class'],
    position_class_support=['margin', 'position', 'predicted_class', 'support'],
    position_class_semantics=['margin', 'position', 'predicted_class', 'semantics'],
    position_class_image=['margin', 'position', 'predicted_class', 'image'],
    all_class=['margin', 'position', 'predicted_class', 'support', 'semantics', 'image'],
    all_class_without_support=['margin', 'position', 'predicted_class', 'semantics', 'image'],
    all_class_without_semantics=['margin', 'position', 'predicted_class', 'support', 'image'],
    all_class_without_image=['margin', 'position', 'predicted_class', 'support', 'semantics'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def precision_at_fraction(y, p, w, fraction=.2):
    order = np.argsort(-p, kind='stable')
    # Allocate a partial score-tie group proportionally, independent of row order.
    boundaries = np.r_[0, np.where(np.diff(p[order]) != 0)[0]+1]
    ww = np.add.reduceat(w[order], boundaries)
    positive = np.add.reduceat(w[order]*y[order], boundaries)
    budget = w.sum()*fraction
    before = np.cumsum(ww)-ww
    selected = np.minimum(ww, np.maximum(budget-before, 0))
    return float(np.dot(selected, positive/ww)/max(selected.sum(), 1e-20))


def scores(y, p, w):
    if not len(y) or len(np.unique(y)) < 2:
        raise ValueError('Both target classes required')
    return dict(rows=len(y), population=float(w.sum()), prevalence=float(np.average(y, weights=w)),
        auroc=float(roc_auc_score(y, p, sample_weight=w)),
        auprc=float(average_precision_score(y, p, sample_weight=w)),
        brier=float(brier_score_loss(y, p, sample_weight=w)),
        precision_top20=precision_at_fraction(y, p, w),
        precision_top50=precision_at_fraction(y, p, w, .5))


def histogram_scores(positive, negative):
    """Ascending bins, with ties treated as half a concordance."""
    a, b = positive.sum(-1), negative.sum(-1)
    below = np.cumsum(negative, -1)-negative
    auc = (positive*(below+negative*.5)).sum(-1)/(a*b).clip(1e-20)
    tp = np.cumsum(positive[..., ::-1], -1)
    total = np.cumsum((positive+negative)[..., ::-1], -1)
    ap = (positive[..., ::-1]/a[..., None].clip(1e-20)*tp/total.clip(1e-20)).sum(-1)
    return auc, ap


def bootstrap_delta(y, baseline, proposed, w, scene, repetitions=300):
    """Paired scene bootstrap; 1024 score bins bound CPU/memory use."""
    scenes, inverse = np.unique(scene, return_inverse=True)
    bins = 1024
    hist = []
    for p in [baseline, proposed]:
        index = np.minimum((p*bins).astype(int), bins-1)
        pos = np.bincount(inverse*bins+index, weights=w*y, minlength=len(scenes)*bins).reshape(-1, bins)
        neg = np.bincount(inverse*bins+index, weights=w*(1-y), minlength=len(scenes)*bins).reshape(-1, bins)
        hist.append((pos, neg))
    rng = np.random.default_rng(928)
    multiplicity = rng.multinomial(len(scenes), np.ones(len(scenes))/len(scenes), repetitions)
    stats = [histogram_scores(multiplicity@pos, multiplicity@neg) for pos, neg in hist]
    result = dict(scene_count=len(scenes), repetitions=repetitions, score_bins=bins,
                  caveat='CI uses fixed-width probability bins; point estimates use exact weighted metrics')
    for i, metric in enumerate(['auroc', 'auprc']):
        delta = stats[1][i]-stats[0][i]
        result[metric+'_delta_ci95'] = np.quantile(delta, [.025, .975]).tolist()
        points = [histogram_scores(pos.sum(0), neg.sum(0))[i] for pos, neg in hist]
        result[metric+'_binned_point_delta'] = float(points[1]-points[0])
    return result


def load_data(split, kind):
    summary = json.loads((RUN/split/'summary.json').read_text())
    arrays = {k: [] for k in ['X', 'gt', 'fg', 'region', 'weight', 'scene', 'frame']}
    scenes = summary['scene_ids']
    mapping = {s: i for i, s in enumerate(scenes)}
    for frame in summary['frames']:
        with np.load(ROOT/(frame['path']+'.npz')) as z:
            keep = z['kind'] == kind
            for k in ['X', 'gt', 'fg', 'region', 'weight']:
                arrays[k].append(z[k][keep])
            arrays['scene'].append(np.full(int(keep.sum()), mapping[frame['scene']], np.int16))
            arrays['frame'].append(np.full(int(keep.sum()), frame['index'], np.int16))
    result = {k: np.concatenate(v) for k, v in arrays.items()}
    if kind == 1:
        result['X'] = np.concatenate([result['X'], np.eye(17, dtype=np.float32)[result['fg']]], axis=1)
    result['scenes'] = scenes
    return result, summary


def metric_from_confusion(c):
    c = np.asarray(c, dtype=np.float64)
    union = c.sum(0)+c.sum(1)-np.diag(c)
    iou = np.divide(np.diag(c), union, out=np.zeros(18), where=union > 0)
    tp = c[:17, :17].sum()
    fn, fp = c[:17, 17].sum(), c[17, :17].sum()
    return dict(mIoU=100*float(iou[:17].mean()), IoU=100*float(tp/max(tp+fn+fp, 1)))


def run_probes():
    train, tr_summary = load_data('train', 1)
    val, va_summary = load_data('val', 1)
    assert set(train['scenes']).isdisjoint(val['scenes'])
    assert tr_summary['previous_sweep_exact_parity'] and va_summary['previous_sweep_exact_parity']
    for data, summary in [(train, tr_summary), (val, va_summary)]:
        for ri, region in enumerate(REGIONS[:2]):
            population = data['weight'][data['region'] == ri].sum()
            assert np.isclose(population, summary['counts'][region]['flips'], rtol=1e-12)
    rng = np.random.default_rng(928)
    unique = np.unique(train['scene'])
    rng.shuffle(unique)
    fit_scenes = unique[:int(len(unique)*.75)]
    tune_scenes = unique[int(len(unique)*.75):]
    fit = np.where((train['region'] == 0) & np.isin(train['scene'], fit_scenes))[0]
    tune = np.where((train['region'] == 0) & np.isin(train['scene'], tune_scenes))[0]
    # A fixed, label-independent row cap controls CPU cost. Inclusion is uniform.
    if len(fit) > 200000:
        fit = rng.choice(fit, 200000, replace=False)
    if len(tune) > 100000:
        tune = rng.choice(tune, 100000, replace=False)
    write_json(RUN/'probe_split_manifest.json', dict(seed=928,
        fit_scenes=[train['scenes'][s] for s in fit_scenes],
        tune_scenes=[train['scenes'][s] for s in tune_scenes],
        train_frames=tr_summary['frames'], validation_scenes=val['scenes'],
        fit_rows=len(fit), tune_rows=len(tune), training_regions=['camera'],
        target_definition=dict(occupied='GT<17', correct_rescue='GT<17 AND FG argmax=GT'),
        iterations_candidates=[40, 100], model='HistGradientBoostingClassifier',
        max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=1., learning_rate=.07,
        fitting='Fit scene subset only; select iterations using training tuning-scene weighted AP; no refit'))
    result = dict(status='running', models={}, features=tr_summary['protocol']['features']+
                  [f'predicted_fg_is_{i:02d}' for i in range(17)],
                  source_sha256=sha(__file__), extraction_source_sha256=tr_summary['protocol']['source_sha256'])
    predictions = {}
    for task in ['correct_rescue', 'occupied']:
        yt = ((train['gt'] < 17) & (train['gt'] == train['fg'])) if task == 'correct_rescue' else train['gt'] < 17
        yv = ((val['gt'] < 17) & (val['gt'] == val['fg'])) if task == 'correct_rescue' else val['gt'] < 17
        result['models'][task] = {}
        predictions[task] = {}
        for name, groups in MODELS.items():
            cols = sum([GROUPS[g] for g in groups], [])
            xfit = train['X'][fit][:, cols]
            xtune = train['X'][tune][:, cols]
            wfit = train['weight'][fit]
            choices = []
            for iterations in [40, 100]:
                est = HistGradientBoostingClassifier(max_iter=iterations, learning_rate=.07,
                    max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=1., early_stopping=False,
                    random_state=928)
                est.fit(xfit, yt[fit], sample_weight=wfit/wfit.mean())
                p = est.predict_proba(xtune)[:, 1]
                ap = average_precision_score(yt[tune], p, sample_weight=train['weight'][tune])
                choices.append((ap, -iterations, est))
            chosen = max(choices, key=lambda x: (x[0], x[1]))
            est = chosen[2]
            p = est.predict_proba(val['X'][:, cols])[:, 1]
            predictions[task][name] = p
            joblib.dump(est, RUN/f'probe_{task}_{name}.joblib')
            regions = {}
            for ri, region in enumerate(REGIONS):
                keep = val['region'] == ri if ri < 2 else np.ones(len(yv), bool)
                regions[region] = scores(yv[keep], p[keep], val['weight'][keep])
            result['models'][task][name] = dict(features=[result['features'][c] for c in cols],
                selected_iterations=-chosen[1], tuning_weighted_ap=chosen[0], validation=regions)
            write_json(RUN/'probes_partial.json', result)
            print(task, name, 'camera AP', regions['camera']['auprc'],
                  'complement AP', regions['camera_complement']['auprc'], flush=True)
        result['models'][task]['paired_comparisons'] = {}
        comparisons = [('margin_position', n) for n in ['position_support', 'position_semantics', 'position_image', 'all']]
        comparisons += [(n, 'all') for n in ['all_without_support', 'all_without_semantics', 'all_without_image']]
        comparisons += [('margin_position_class', n) for n in
            ['position_class_support', 'position_class_semantics', 'position_class_image', 'all_class']]
        comparisons += [(n, 'all_class') for n in
            ['all_class_without_support', 'all_class_without_semantics', 'all_class_without_image']]
        for base, proposed in comparisons:
            pair = base+'__to__'+proposed
            result['models'][task]['paired_comparisons'][pair] = {}
            for ri, region in enumerate(REGIONS):
                keep = val['region'] == ri if ri < 2 else np.ones(len(yv), bool)
                ci = bootstrap_delta(yv[keep].astype(float), predictions[task][base][keep],
                    predictions[task][proposed][keep], val['weight'][keep], val['scene'][keep])
                result['models'][task]['paired_comparisons'][pair][region] = ci
    # Per-frame predictions remain auditable, no GT used in any input feature.
    np.savez_compressed(RUN/'validation_probe_predictions.npz',
        **{task+'__'+name: p.astype(np.float32) for task, pred in predictions.items() for name, p in pred.items()},
        gt=val['gt'], fg=val['fg'], region=val['region'], weight=val['weight'], scene=val['scene'], frame=val['frame'])
    result.update(status='complete', finished_at=datetime.now().astimezone().isoformat(),
        validation_candidate_rows=len(val['gt']), validation_candidate_population=float(val['weight'].sum()),
        training_candidate_rows=len(train['gt']), fit_rows=len(fit), tune_rows=len(tune),
        training_scenes=len(train['scenes']), validation_scenes=len(val['scenes']))
    write_json(RUN/'probes.json', result)
    return result


def evidence_distributions():
    val, summary = load_data('val', 0)
    result = {}
    for ri, region in enumerate(REGIONS):
        keep = val['region'] == ri if ri < 2 else np.ones(len(val['gt']), bool)
        margin = val['X'][:, 0]
        events = dict(native_fn_fg_correct=(val['gt'] < 17) & (val['fg'] == val['gt']) & (margin < 0),
            native_fn_fg_wrong=(val['gt'] < 17) & (val['fg'] != val['gt']) & (margin < 0),
            correct_rescue=(val['gt'] < 17) & (val['fg'] == val['gt']) & (margin < 0) & (margin >= -1),
            wrong_class_exposed=(val['gt'] < 17) & (val['fg'] != val['gt']) & (margin < 0) & (margin >= -1),
            new_false_occupied=(val['gt'] == 17) & (margin < 0) & (margin >= -1))
        result[region] = {}
        for event, mask in events.items():
            use = keep & mask
            w = val['weight'][use]
            x = val['X'][use]
            result[region][event] = dict(rows=int(use.sum()), population_estimate=float(w.sum()),
                weighted_mean_features=dict(zip(summary['protocol']['features'],
                    np.average(x, weights=w, axis=0).astype(float).tolist())))
    write_json(RUN/'evidence_distributions.json', result)
    return result


def plots(summary, probes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    names = ['Camera mask', 'Mask complement', 'All valid']
    for ax, region, name in zip(axes, REGIONS, names):
        d = summary['counts'][region]
        vals = np.array([d[k] for k in ['correct_rescue', 'wrong_class_exposed', 'new_false_occupied']])/d['flips']*100
        ax.bar(['Correct\nrescue', 'Wrong class\nexposed', 'New false\noccupied'], vals,
               color=['#218c74', '#f2b134', '#d65a59'])
        for i, v in enumerate(vals):
            ax.text(i, v+1, f'{v:.1f}%', ha='center')
        ax.set(ylim=(0, 100), title=name, ylabel='Share of actual +1 flips (%)')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('What does an extra empty shift actually change? Frozen mini checkpoint')
    fig.tight_layout()
    fig.savefig(REPORT/'error_decomposition.png', dpi=180)
    fig.savefig(REPORT/'error_decomposition.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    chosen = ['margin', 'margin_position_class', 'position_class_support', 'position_class_semantics', 'position_class_image', 'all_class']
    labels = ['Margin', '+Position/class', '+Support', '+Semantics', '+Image', 'All features']
    for ax, region, name in zip(axes, REGIONS, names):
        values = [100*probes['models']['correct_rescue'][n]['validation'][region]['auprc'] for n in chosen]
        prevalence = 100*probes['models']['correct_rescue']['margin']['validation'][region]['prevalence']
        ax.barh(labels, values, color=['#8994a0', '#557b9e', '#a8c5a0', '#49a58b', '#d8af67', '#325f88'])
        ax.axvline(prevalence, linestyle='--', color='#d65a59', label='Natural prevalence')
        for i, v in enumerate(values):
            ax.text(v+.3, i, f'{v:.1f}', va='center', fontsize=9)
        ax.invert_yaxis()
        ax.set(xlim=(0, 100), title=name, xlabel='Correct-rescue weighted AUPRC (%)')
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Predicting beneficial corrections; probes trained only on camera-region training labels')
    fig.tight_layout()
    fig.savefig(REPORT/'local_evidence_probes.png', dpi=180)
    fig.savefig(REPORT/'local_evidence_probes.svg')
    plt.close(fig)


def report():
    REPORT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RUN/'val/summary.json').read_text())
    train = json.loads((RUN/'train/summary.json').read_text())
    probes = json.loads((RUN/'probes.json').read_text())
    distribution = json.loads((RUN/'evidence_distributions.json').read_text())
    def pct(n, d):
        return 100*n/max(d, 1)
    lines = ['# Mini set：empty 修正来源与局部证据预测——实验结果', '',
        '本报告对应 [实验设计](experiment_design.md)。所有主模型预测均来自同一个已完成 4096 样本训练、seed 928、5 epoch 的冻结 checkpoint；验证覆盖全部 1024 帧。没有重训主模型，也没有使用另一台四卡机器的结果。', '',
        '> 本报告区分三种证据强度：输出层错误分解可以精确确认；局部特征的预测能力属于独立场景上的统计关联；特征是否造成错误、以及新方法能否改善端到端性能，尚未验证。', '',
        '## 1. empty 修正具体改变了什么', '',
        '下表只统计额外 b=+1 实际使预测从 empty 变为前景的体素。b=+1 沿用此前训练子集选择，不在验证集重新选取。已有 learnable bias 仍然生效。', '',
        '|区域|翻转体素数|正确语义被救回|二值恢复但类别仍错|真实 empty 被误占用|',
        '|---|---:|---:|---:|---:|']
    for region in REGIONS:
        d = summary['counts'][region]
        lines.append(f"|{LABELS[region]}|{d['flips']:,}|{pct(d['correct_rescue'],d['flips']):.2f}%|{pct(d['wrong_class_exposed'],d['flips']):.2f}%|{pct(d['new_false_occupied'],d['flips']):.2f}%|")
    lines += ['', '“正确救回”要求前景 argmax 等于 GT；“类别仍错”仅恢复 occupied 状态。三项互斥且穷尽全部翻转。它们是体素事件数，不能直接解释为 mIoU 的可加贡献。', '',
        '![实际翻转的错误构成](error_decomposition.png)', '',
        '### 默认漏检中，有多少已经具备正确前景类别', '',
        '|区域|occupied→empty 总数|其中前景 argmax 正确|前景正确占全部漏检|b=+1 救回这些正确类别的比例|',
        '|---|---:|---:|---:|---:|']
    for region in REGIONS:
        d = summary['counts'][region]
        lines.append(f"|{LABELS[region]}|{d['native_fn']:,}|{d['native_fn_fg_correct']:,}|{pct(d['native_fn_fg_correct'],d['native_fn']):.2f}%|{pct(d['correct_rescue'],d['native_fn_fg_correct']):.2f}%|")
    lines += ['', '这一分解能检验“已有正确类别被 empty 压制”是否存在及其规模；不能据此判断几何完全正确。前景 argmax 在所有前景分数均很弱时也可能偶然等于 GT。', '',
        '### 必须结合实际 Gaussian 支持阅读', '',
        'S=Σ αᵢKᵢ(x)，由生产 CUDA 聚合器按原始候选体素盒、协方差和截断计算；排除额外 empty Gaussian。下面的分母均为“默认漏检且前景 argmax 正确”的体素数。', '',
        '|区域|S>10⁻⁶|S>10⁻³|S>10⁻²|S>10⁻¹|', '|---|---:|---:|---:|---:|']
    for region in REGIONS:
        den = summary['counts'][region]['native_fn_fg_correct']
        vals = [pct(summary['support_counts'][region][str(t)][1], den) for t in [1e-6, 1e-3, 1e-2, .1]]
        lines.append('|'+LABELS[region]+'|'+'|'.join(f'{v:.2f}%' for v in vals)+'|')
    lines += ['', '这些阈值没有“几何合格”的物理含义。核尾部贡献大于零不等于可靠几何支持；阈值敏感性用于防止把“存在贡献”包装成“几何没有问题”。', '',
        '### 全量指标与强修正参照', '',
        '|区域|默认 mIoU|额外 b=+1|额外 b=+4|固定权重移除已学 bias|', '|---|---:|---:|---:|---:|']
    bias = summary['protocol']['learned_bias']
    for region in REGIONS:
        vals = [metric_from_confusion(summary['confusion'][region][str(b)])['mIoU'] for b in [0., 1., 4., -bias]]
        lines.append('|'+LABELS[region]+'|'+'|'.join(f'{v:.4f}' for v in vals)+'|')
    lines += ['', f'learnable bias={bias:.6f}。最后一列只在当前权重上取消 bias，并非重新训练无 bias 模型；不能用于估计该模块的完整训练收益。此次未重复执行 RayIoU，沿用之前报告时须明确是既有结果。', '',
        '## 2. 哪些局部信号能预测修正是否有益', '',
        '以下只研究 −1≤foreground−empty margin<0 的候选带，即 b=+1 会翻转的位置。主目标为“修正后类别完全正确”；负例同时包含真实 empty 和错误前景类别。辅助目标仅区分 GT occupied/empty。', '',
        f"训练候选采样覆盖 {train['samples']} 个 mini 训练帧；{probes['training_scenes']} 个训练场景按 scene 分为拟合与调参组，拟合 {probes['fit_rows']:,} 行、调参 {probes['tune_rows']:,} 行。预测器只使用 camera 区域训练标签。验证包含 {probes['validation_scenes']} 个不重叠场景、{probes['validation_candidate_rows']:,} 行候选样本；按抽样概率倒数恢复自然体素先验。", '',
        '相同容量浅层树，迭代次数只在训练调参场景的加权 AP 上二选一。AP 为 AUPRC，ROC 为 AUROC；P@20% 是各模型在相同的 20% 候选体素修正预算下，选中体素的正确救回率。它是离线排序诊断，不是部署阈值或新方法性能。', '',
        '|特征|camera AP / ROC|补集 AP / ROC|全空间 AP / ROC|camera P@20%|补集 P@20%|',
        '|---|---:|---:|---:|---:|---:|']
    names = dict(margin='margin', margin_position='margin＋位置', margin_support='margin＋支持',
        margin_semantics='margin＋语义', margin_image='margin＋图像', position_support='margin＋位置＋支持',
        position_semantics='margin＋位置＋语义', position_image='margin＋位置＋图像', all='全部特征',
        all_without_support='全部−支持', all_without_semantics='全部−语义', all_without_image='全部−图像',
        margin_position_class='margin＋位置＋预测类别', position_class_support='上述＋支持',
        position_class_semantics='上述＋语义', position_class_image='上述＋图像', all_class='全部＋预测类别',
        all_class_without_support='全部＋类别−支持', all_class_without_semantics='全部＋类别−语义',
        all_class_without_image='全部＋类别−图像')
    main = probes['models']['correct_rescue']
    for name in MODELS:
        v = main[name]['validation']
        pairs = [' / '.join(f"{100*v[r][m]:.2f}" for m in ['auprc', 'auroc']) for r in REGIONS]
        lines.append('|'+names[name]+'|'+'|'.join(pairs)+f"|{100*v['camera']['precision_top20']:.2f}%|{100*v['camera_complement']['precision_top20']:.2f}%|")
    lines += ['', '自然正例比例（也即无信息排序的 AP 参照）：'+ '；'.join(f"{LABELS[r]} {100*main['margin']['validation'][r]['prevalence']:.2f}%" for r in REGIONS)+'。不同区域的 AP 不能脱离这个先验直接比较。', '',
        '![局部证据预测能力](local_evidence_probes.png)', '',
        '### 相对 margin＋位置的增量是否跨场景成立', '',
        '下面为配对 scene bootstrap 的 AP 差异，单位为百分点；300 次重采样，95% 区间。点估计采用精确加权 AP，区间采用 1024 个固定概率分箱近似，包含有限体素抽样与固定 probe 的局限，不包含重训种子不确定性。多个探索性对照未作多重比较校正。', '',
        '|新增特征|camera ΔAP [95% CI]|补集 ΔAP [95% CI]|全空间 ΔAP [95% CI]|', '|---|---:|---:|---:|']
    for name in ['position_support', 'position_semantics', 'position_image', 'all']:
        values = []
        for region in REGIONS:
            delta = 100*(main[name]['validation'][region]['auprc']-main['margin_position']['validation'][region]['auprc'])
            ci = main['paired_comparisons']['margin_position__to__'+name][region]['auprc_delta_ci95']
            values.append(f'{delta:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]')
        lines.append('|'+names[name]+'|'+'|'.join(values)+'|')
    lines += ['', '进一步控制前景预测类别（17 维 one-hot），检验是否只是不同类别的修正先验。下面相对 margin＋位置＋预测类别：', '',
        '|新增特征|camera ΔAP [95% CI]|补集 ΔAP [95% CI]|全空间 ΔAP [95% CI]|', '|---|---:|---:|---:|']
    for name in ['position_class_support', 'position_class_semantics', 'position_class_image', 'all_class']:
        values = []
        for region in REGIONS:
            delta = 100*(main[name]['validation'][region]['auprc']-main['margin_position_class']['validation'][region]['auprc'])
            ci = main['paired_comparisons']['margin_position_class__to__'+name][region]['auprc_delta_ci95']
            values.append(f'{delta:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]')
        lines.append('|'+names[name]+'|'+'|'.join(values)+'|')
    lines += ['', '### 只判断真实 occupied 是否更容易', '',
        '|特征|camera occupied AP|补集 occupied AP|全空间 occupied AP|', '|---|---:|---:|---:|']
    for name in ['margin', 'margin_position', 'position_support', 'position_semantics', 'position_image', 'all']:
        v = probes['models']['occupied'][name]['validation']
        lines.append('|'+names[name]+'|'+'|'.join(f"{100*v[r]['auprc']:.2f}" for r in REGIONS)+'|')
    lines += ['', '### 对机制的解释限制', '',
        '- 实际核支持与语义投票可预测收益，属于诊断关联；它们也可能编码位置、类别或置信度等混杂信息。加入位置对照和联合模型消融能缩小解释范围，不能证明因果。',
        '- 当前 foreground logits 本来就是核加权语义分数之和；支持特征与渲染分数存在代数联系。支持特征有增益不等于发现了独立物理机制，更不等于证明 Gaussian 独有的缺陷。',
        '- 语义熵基于非负 softplus 分数归一化，不等同于经过校准的类别概率；投票一致性来自核加权 Gaussian 的硬类别。',
        '- 图像特征范数、跨视角余弦与入图相机数只是代理。物理不可见体素也可能投影到图像，不能把这些结果解释为识别了真实遮挡。',
        '- camera mask 仅用于区域分析和训练行选择，不作为预测器输入。补集表现检验跨区域迁移，但该主模型的 set loss 原本使用全域 occupied 标签，因此不是从未获得任何域外监督的设定。',
        '- 当前是单 checkpoint、单种子、5 epoch mini 诊断。不能推导所有 semantic Gaussian 方法普遍存在同一机制，也不能据此声称超过 S2GO 的新贡献已成立。', '',
        '## 3. 完整性与复查入口', '',
        f"- 全量验证：{summary['samples']} 帧；训练诊断：{train['samples']} 帧；原始训练规模：4096。",
        '- b=0 与原生 final_occ 逐帧完全一致；camera＋补集的 confusion 等于全空间；三类翻转互斥且穷尽。',
        '- b=0、+1、+4 的训练与验证全量 confusion 均与此前独立 empty-scope 实验完全一致。',
        '- 额外重渲染的 foreground logits 与原始输出完全一致；抽样核支持与独立 PyTorch 权重参考按容差核验。',
        '- 训练拟合、训练调参、验证场景无交叉；特征中不包含 GT、camera mask、GT depth 或 LiDAR。',
        '- 主模型权重、解析配置、数据 split、诊断源码 SHA256 随结果保存。', '',
        '原始产物：`work_dirs/empty_decision_mini/`。`train/summary.json`、`val/summary.json` 保存全量计数；逐帧 `.npz` 保存抽样特征；`probes.json` 保存全部预测指标与置信区间；`probe_split_manifest.json` 保存场景划分；`validation_probe_predictions.npz` 可复查排序；`evidence_distributions.json` 保存错误组的加权特征均值。', '',
        '复现：', '', '```bash',
        'python3.10 tools/diagnose_empty_decision_mini.py',
        'python3.10 tools/analyze_empty_decision_mini.py', '```', '',
        '诊断提取按逐帧缓存续跑，并校验协议哈希；修改提取源码后应使用新结果目录，不能混合旧缓存。', '']
    report_path = REPORT/'experiment_results.md'
    reviewed = report_path.exists() and '<!-- EDITORIAL_REVIEWED -->' in report_path.read_text()
    if reviewed:
        print('Preserving the reviewed narrative in', report_path, flush=True)
    else:
        report_path.write_text('\n'.join(lines), encoding='utf-8')
    if not reviewed or not all((REPORT/name).exists() for name in
        ['error_decomposition.png', 'error_decomposition.svg',
         'local_evidence_probes.png', 'local_evidence_probes.svg']):
        plots(summary, probes)
    audit = dict(status='complete', train_samples=train['samples'], validation_samples=summary['samples'],
        train_validation_scene_disjoint=set(train['scene_ids']).isdisjoint(summary['scene_ids']),
        previous_sweep_exact_parity=train['previous_sweep_exact_parity'] and summary['previous_sweep_exact_parity'],
        source_sha256={p: sha(ROOT/p) for p in ['tools/diagnose_empty_decision_mini.py', 'tools/analyze_empty_decision_mini.py']},
        report_sha256=sha(REPORT/'experiment_results.md'), finished_at=datetime.now().astimezone().isoformat())
    write_json(RUN/'completion_audit.json', audit)
    write_json(RUN/'status.json', audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=8):
        if not args.report_only:
            run_probes()
            evidence_distributions()
        report()


if __name__ == '__main__':
    main()
