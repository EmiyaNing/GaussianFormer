"""Offline audit of the 1.5x-candidate / 1x-hard-inference query budget run.

Reads only saved artifacts: per-scene routing npz/json, aggregate summaries,
the small/tiny query census strata, and the camera-set baseline per-scene CSV.
No model, CUDA, or training is required.

python3.10 tools/analyze_query_budget_1p5x_failure.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-dsh')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ROUTING = ROOT / 'work_dirs/small_query_budget_camera_full_1ep/routing_diagnosis'
SMALL = ROOT / 'work_dirs/small_gaussian_diagnosis_full'
TINY = ROOT / 'work_dirs/query_budget_full'
BASE = ROOT / 'work_dirs/small_camera_set_full_1ep/diagnostics/aggregate/scene_comparison.csv'
OUT = ROOT / 'work_dirs/query_budget_1p5x_analysis'
FIGS = ROOT / 'analysis/figures/query_budget_1p5x'
NAMES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
MODES = ['soft_all', 'hard_raw', 'hard_gated', 'uniform64', 'opacity_only']
COLORS = dict(soft_all='#2a7f62', hard_raw='#c0392b', hard_gated='#e67e22',
              uniform64='#2c6fbb', opacity_only='#7f8c8d', baseline='#111111')


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=float)


def per_scene_arrays():
    frames = []
    for path in sorted((ROUTING / 'scenes/val').glob('*.npz')):
        z = np.load(path)
        frames.append(dict(
            gt_load=z['gt_load'], selected=z['selected_count'],
            score=z['query_score'], alpha=z['mean_opacity'],
            soft_mass=z['soft_mass'], miss=z['hard_gt_miss'],
            wrong=z['hard_gt_wrong_class'],
            no_core=z['current_hard_gt_no_core'],
            coarse=z['gt_coarse_distance_sum'],
            confusion={m: z['confusion_' + m] for m in
                       ('hard_raw', 'soft_all', 'hard_gated', 'uniform64', 'opacity_only')},
        ))
    keys = ('gt_load', 'selected', 'score', 'alpha', 'soft_mass', 'miss',
            'wrong', 'no_core', 'coarse')
    merged = {k: np.concatenate([f[k].reshape(-1) for f in frames], axis=0)
              for k in keys}
    return frames, merged


def mode_metrics(confusion):
    c = confusion.astype(float)
    gt, pred, diag = c.sum(1), c.sum(0), np.diag(c)
    iou = diag / (gt + pred - diag + 1e-12)
    tp, fn, fp = c[:17, :17].sum(), c[:17, 17].sum(), c[17, :17].sum()
    return dict(miou=float(np.mean(iou[:17]) * 100),
                biou=float(100 * tp / (tp + fn + fp)),
                fn_rate=float(100 * fn / (tp + fn)),
                fp_empty=float(100 * fp / c[17].sum()),
                class_iou=dict(zip(NAMES, (iou[:17] * 100).tolist())),
                class_fn=dict(zip(NAMES, (100 * c[:17, 17] / c[:17].sum()).tolist())),
                class_fn_count=dict(zip(NAMES, c[:17, 17].tolist())),
                confusion=c.tolist())


def fit_allocation(m):
    active = m['gt_load'] > 0
    load, sel, score, alpha = (m[k][active] for k in ('gt_load', 'selected', 'score', 'alpha'))
    X = np.stack([np.log1p(load), np.log(score / (1 - score)), np.log(alpha)], axis=1)
    X = (X - X.mean(0)) / X.std(0)
    y = sel.astype(float)
    y = (y - y.mean()) / y.std()
    design = np.c_[np.ones(len(X)), X]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    r2 = 1 - float(((y - design @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return dict(beta=dict(intercept=beta[0], log_load=beta[1],
                          logit_score=beta[2], log_mean_alpha=beta[3]),
                r2=r2, n=int(active.sum()))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d else float('nan')


def bootstrap_mean(values, seed=928, repeats=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
    return dict(mean=float(values.mean()),
                ci=np.quantile(draws, [.025, .975]).tolist())


def strata(root, split):
    total = None
    for path in sorted((root / 'scenes' / split).glob('*.npz')):
        value = np.load(path)['strata']
        total = value if total is None else total + value
    return total


def standardize(s):
    low, high = s[1:3].sum(0), s[9:11].sum(0)
    all_fg = s[..., 0].sum(0)
    common = (low[..., 0] > 0) & (high[..., 0] > 0)
    weight = all_fg[common].astype(float)
    retained = float(weight.sum() / all_fg.sum())
    weight /= weight.sum()
    rates = {}
    for name, block in (('low', low), ('high', high)):
        rates[name] = float((weight * block[..., 1][common] / block[..., 0][common]).sum())
    raw_low = float(low[..., 1].sum() / low[..., 0].sum())
    raw_high = float(high[..., 1].sum() / high[..., 0].sum())
    return dict(raw_low=raw_low, raw_high=raw_high, raw_gap=raw_high - raw_low,
                aligned_low=rates['low'], aligned_high=rates['high'],
                aligned_gap=rates['high'] - rates['low'],
                composition_share=1 - (rates['high'] - rates['low']) / (raw_high - raw_low),
                common_cells=int(common.sum()), weight_retained=retained)


def scene_table(frames, merged):
    rows = []
    for f in frames:
        load2, sel2, miss2 = f['gt_load'], f['selected'], f['miss']
        load, sel, miss = load2.reshape(-1), sel2.reshape(-1), miss2.reshape(-1)
        fg = float(load.sum())
        top = np.zeros_like(load2, dtype=bool)
        top[np.arange(len(load2))[:, None],
            np.argsort(-load2, axis=1, kind='stable')[:, :max(load2.shape[1] // 5, 1)]] = True
        top = top.reshape(-1)
        rows.append(dict(fg=fg,
                         zero_fg=float(load[sel == 0].sum() / max(fg, 1)),
                         zero_miss=float(miss[sel == 0].sum() / max(miss.sum(), 1)),
                         zero_q=float(((sel == 0) & (load > 0)).sum() / max((load > 0).sum(), 1)),
                         high_sel=float(sel[top].sum() / max(sel.sum(), 1)),
                         high_fg=float(load[top].sum() / max(fg, 1))))
        soft = mode_metrics(f['confusion']['soft_all'])
        hard = mode_metrics(f['confusion']['hard_raw'])
        uni = mode_metrics(f['confusion']['uniform64'])
        rows[-1].update(soft_miou=soft['miou'], hard_miou=hard['miou'], uni_miou=uni['miou'],
                        soft_biou=soft['biou'], hard_biou=hard['biou'], uni_biou=uni['biou'])
    return rows


def baseline_map():
    if not BASE.exists():
        return {}
    rows = list(csv.DictReader(open(BASE, encoding='utf-8-sig')))
    return {r['scene']: r for r in rows if r['model'] == 'after'}


def token_to_name():
    out = {}
    for path in (SMALL / 'scenes/val').glob('*.json'):
        d = read_json(path)
        out[path.stem] = d['scene_name']
    return out


def plot_modes(result):
    labels = ['baseline', 'soft_all', 'hard_raw', 'hard_gated', 'uniform64', 'opacity_only']
    values = [result['baseline'], result['modes']['soft_all'], result['modes']['hard_raw'],
              result['modes']['hard_gated'], result['modes']['uniform64'],
              result['modes']['opacity_only']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, title in ((axes[0], 'miou', '17-class mIoU (%)'),
                           (axes[1], 'biou', 'Binary occupancy IoU (%)')):
        bars = ax.bar(range(len(labels)), [v[key] for v in values],
                      color=[COLORS.get(x, COLORS['baseline']) for x in labels])
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.25)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .6,
                    f'{v[key]:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, max(v[key] for v in values) * 1.15)
    fig.suptitle('1.5x checkpoint under five rendering modes, with the 1x baseline', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / 'mode_metrics.png', dpi=160)
    plt.close(fig)


def plot_class_tradeoff(result):
    focus = ['driveable', 'terrain', 'sidewalk', 'other_flat', 'car', 'pedestrian',
             'vegetation', 'manmade']
    soft = result['modes']['soft_all']['class_iou']
    hard = result['modes']['hard_raw']['class_iou']
    uni = result['modes']['uniform64']['class_iou']
    x = np.arange(len(focus))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = .38
    ax.bar(x - width / 2, [hard[k] - soft[k] for k in focus], width,
           label='global hard - soft', color=COLORS['hard_raw'])
    ax.bar(x + width / 2, [uni[k] - soft[k] for k in focus], width,
           label='uniform 64 - soft', color=COLORS['uniform64'])
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(x, focus, rotation=20, ha='right')
    ax.set_ylabel('class IoU change (pp)')
    ax.set_title('Two hard 1x selections fail on different classes')
    ax.legend()
    ax.grid(axis='y', alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'class_tradeoff.png', dpi=160)
    plt.close(fig)


def plot_selection(result):
    hist = result['selection']['count_bins']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].bar(range(len(hist)), [100 * v for v in hist],
                color=['#7f8c8d'] + ['#2c6fbb'] * (len(hist) - 2) + ['#2a7f62'])
    axes[0].set_xticks(range(len(hist)), ['0', '1-16', '17-32', '33-48', '49-64',
                                          '65-80', '81-95', '96'], rotation=0)
    axes[0].set_ylabel('% of active query-frames')
    axes[0].set_title('Hard selection is nearly query-level on/off')
    axes[0].grid(axis='y', alpha=.25)
    for i, v in enumerate(hist):
        if v > .02:
            axes[0].text(i, 100 * v + .8, f'{100 * v:.1f}', ha='center', fontsize=8)
    d = result['selection']['score_deciles']
    x = np.arange(10)
    axes[1].bar(x, d['mean_selected'], color='#2c6fbb', label='mean selected / query')
    axes[1].set_ylabel('mean selected GS')
    axes[1].set_xticks(x, [f'd{i + 1}' for i in x])
    axes[1].set_xlabel('query-score decile (low to high)')
    ax2 = axes[1].twinx()
    ax2.plot(x, d['fg_share'], color='#c0392b', marker='o', label='foreground share')
    ax2.plot(x, d['zero_fraction'], color='#e67e22', marker='s', label='zero-retained share')
    ax2.axhline(10, color='#999999', ls='--', lw=.8)
    ax2.set_ylabel('%')
    lines = axes[1].get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    axes[1].legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=8)
    axes[1].set_title('Bottom two score deciles: 18.9% foreground, 0.2/11.1 GS per query')
    fig.tight_layout()
    fig.savefig(FIGS / 'selection_shape.png', dpi=160)
    plt.close(fig)


def plot_scene(result):
    rows = result['scene']
    zero = np.array([r['zero_fg'] for r in rows]) * 100
    gap = np.array([r['soft_miou'] - r['hard_miou'] for r in rows])
    gapb = np.array([r['soft_biou'] - r['hard_biou'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, y, title in ((axes[0], gap, 'mIoU gap (soft - hard)'),
                         (axes[1], gapb, 'binary IoU gap (soft - hard)')):
        ax.scatter(zero, y, s=14, alpha=.7, color='#2c6fbb')
        coef = np.polyfit(zero, y, 1)
        xs = np.linspace(zero.min(), zero.max(), 20)
        ax.plot(xs, np.polyval(coef, xs), color='#c0392b', lw=1.2)
        ax.set_xlabel('% foreground assigned to zero-retained queries')
        ax.set_ylabel(title + ' (pp)')
        ax.set_title(f'r = {corr(zero, y):+.3f}')
        ax.grid(alpha=.25)
    fig.suptitle('Scene-level association between query starvation and hard-render loss', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / 'scene_gap.png', dpi=160)
    plt.close(fig)


def plot_alignment(result):
    labels = ['small train', 'small val', 'tiny train', 'tiny val']
    keys = ['small/train', 'small/val', 'tiny/train', 'tiny/val']
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    x = np.arange(len(labels))
    width = .38
    ax.bar(x - width / 2, [100 * result['alignment'][k]['raw_gap'] for k in keys],
           width, label='raw high - low', color='#c0392b')
    ax.bar(x + width / 2, [100 * result['alignment'][k]['aligned_gap'] for k in keys],
           width, label='aligned high - low', color='#2a7f62')
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel('core-hole gap (pp)')
    ax.set_title('Most of the raw load gap is class/range/coarse-error composition')
    ax.legend()
    ax.grid(axis='y', alpha=.25)
    for i, k in enumerate(keys):
        ax.text(i + width / 2, 100 * result['alignment'][k]['aligned_gap'] + .3,
                f"comp. {100 * result['alignment'][k]['composition_share']:.0f}%",
                ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / 'aligned_gap.png', dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    full = read_json(ROUTING / 'aggregate/summary.json')
    summary = full['splits']['val']
    frames, merged = per_scene_arrays()
    modes = {m: mode_metrics(np.asarray(summary['render_modes'][m]['confusion']))
             for m in MODES}
    baseline = dict(miou=40.38775, biou=67.19838)

    active = merged['gt_load'] > 0
    load_arr, sel, score, alpha = (merged[k][active] for k in
                                    ('gt_load', 'selected', 'score', 'alpha'))
    total_fg = float(merged['gt_load'].sum())
    zero_mask = merged['selected'] == 0
    miss_total = float(merged['miss'].sum())
    zero_fg = float(merged['gt_load'][zero_mask].sum() / total_fg)
    zero_miss_share = float(merged['miss'][zero_mask].sum() / miss_total)
    zero_miss_rate = float(merged['miss'][zero_mask].sum() /
                           max(merged['gt_load'][zero_mask].sum(), 1))
    sel_miss_rate = float(merged['miss'][~zero_mask].sum() /
                          max(merged['gt_load'][~zero_mask].sum(), 1))
    soft_rate = modes['soft_all']['fn_rate'] / 100
    excess = float(merged['miss'][~zero_mask].sum() + merged['miss'][zero_mask].sum() -
                   total_fg * soft_rate)
    zero_excess = float(merged['miss'][zero_mask].sum() -
                        merged['gt_load'][zero_mask].sum() * soft_rate)
    bins = np.histogram(sel, bins=[-0.5, 0.5, 16.5, 32.5, 48.5, 64.5, 80.5, 95.5, 96.5])[0]
    bins = (bins / len(sel)).tolist()
    q = np.quantile(score, np.linspace(0, 1, 11))[1:-1]
    dec = np.clip(np.digitize(score, q), 0, 9)
    deciles = dict(mean_selected=[], zero_fraction=[], fg_share=[], mean_alpha=[], mean_coarse=[])
    coarse = merged['coarse'][active] / np.maximum(load_arr, 1)
    for i in range(10):
        m = dec == i
        deciles['mean_selected'].append(float(sel[m].mean()))
        deciles['zero_fraction'].append(float(100 * (sel[m] == 0).mean()))
        deciles['fg_share'].append(float(100 * load_arr[m].sum() / load_arr.sum()))
        deciles['mean_alpha'].append(float(alpha[m].mean()))
        deciles['mean_coarse'].append(float(coarse[m].mean()))

    rows = scene_table(frames, merged)
    for r in rows:
        r['gap_miou'] = r['soft_miou'] - r['hard_miou']
        r['gap_biou'] = r['soft_biou'] - r['hard_biou']
    scene = rows
    partial = {}
    for key in ('zero_fg', 'high_sel'):
        y = np.array([r['gap_miou'] for r in scene])
        x = np.array([r[key] for r in scene])
        ctrl = np.c_[np.log([r['fg'] for r in scene]), [r['soft_miou'] for r in scene]]
        ctrl = np.c_[np.ones(len(ctrl)), ctrl]
        partial[key] = corr(y - ctrl @ np.linalg.lstsq(ctrl, y, rcond=None)[0],
                            x - ctrl @ np.linalg.lstsq(ctrl, x, rcond=None)[0])

    alignment = {}
    for name, root in (('small', SMALL), ('tiny', TINY)):
        for split in ('train', 'val'):
            summary_root = read_json((SMALL if name == 'small' else TINY) / 'aggregate/summary.json')
            published = summary_root['summary'][split]['standardized_low_high']
            item = standardize(strata(root, split))
            item['published_gap_interval'] = published['hole_gap_interval']
            item['published_aligned'] = [published['low']['hole_rate'],
                                         published['high']['hole_rate']]
            alignment[(name, split)] = item

    base = baseline_map()
    token_name = token_to_name()
    paired = {}
    if base:
        for mode in ('soft_all', 'hard_raw', 'uniform64'):
            vals = []
            for path in sorted((ROUTING / 'scenes/val').glob('*.npz')):
                token = path.stem
                vals.append(mode_metrics(np.load(path)['confusion_' + mode]))
            diffs_m = [v['miou'] - float(base[token_name[path.stem]]['mIoU'])
                       for v, path in zip(vals, sorted((ROUTING / 'scenes/val').glob('*.npz')))]
            diffs_b = [v['biou'] - float(base[token_name[path.stem]]['IoU'])
                       for v, path in zip(vals, sorted((ROUTING / 'scenes/val').glob('*.npz')))]
            paired[mode] = dict(miou=bootstrap_mean(diffs_m), biou=bootstrap_mean(diffs_b))

    result = dict(
        checkpoint=full.get('checkpoint_sha256'), completeness=summary['audit'],
        modes=modes, baseline=baseline,
        classes=NAMES, selection=dict(
            count_bins=bins, zero_fraction=float((sel == 0).mean()),
            full_fraction=float((sel == 96).mean()),
            mean=float(sel.mean()), std=float(sel.std()),
            score_deciles=deciles),
        zero=dict(fg_share=zero_fg, miss_share=zero_miss_share,
                  miss_rate=zero_miss_rate, selected_miss_rate=sel_miss_rate,
                  excess_miss=excess, zero_excess=zero_excess,
                  zero_excess_share=zero_excess / excess),
        allocation=fit_allocation(merged),
        high_load=dict(foreground_share=float(summary['query']['high_load_foreground_share']),
                       soft_mass_share=float(summary['query']['high_load_soft_mass_share']),
                       hard_budget_share=float(summary['variants']['current_hard']['high_load_budget_share']),
                       max_budget_share=240 * 96 / 76800,
                       zero_load_budget_share=float(summary['variants']['current_hard']['zero_load_budget_share']),
                       positive_load_zero_fraction=float(summary['variants']['current_hard']['positive_load_zero_selected_query_fraction'])),
        coverage=dict(hard=float(summary['variants']['current_hard']['no_core_rate']),
                      uniform=float(summary['variants']['uniform64_per_query']['no_core_rate']),
                      all_candidates=float(summary['variants']['all_candidates']['no_core_rate']),
                      baseline=0.209099),
        scene=scene, scene_partial=partial, scene_corr=dict(
            zero_miou=corr([r['zero_fg'] for r in scene], [r['gap_miou'] for r in scene]),
            zero_biou=corr([r['zero_fg'] for r in scene], [r['gap_biou'] for r in scene]),
            high_sel_miou=corr([r['high_sel'] for r in scene], [r['gap_miou'] for r in scene]),
            high_sel_biou=corr([r['high_sel'] for r in scene], [r['gap_biou'] for r in scene])),
        alignment={f'{k[0]}/{k[1]}': v for k, v in alignment.items()},
        paired_baseline=paired,
        alpha=dict(mean=float(alpha.mean()), median=float(np.median(alpha)),
                   q90=float(np.quantile(alpha, .9)),
                   fraction_gt_098=float((alpha > .98).mean()),
                   fraction_gt_0999=float((alpha > .999).mean())),
        notes='Offline aggregation of saved artifacts; no new model inference or training.',
    )
    write_json(OUT / 'summary.json', result)
    with open(OUT / 'scene_gaps.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(scene[0]))
        writer.writeheader()
        writer.writerows(scene)

    plot_modes(result)
    plot_class_tradeoff(result)
    plot_selection(result)
    plot_scene(result)
    plot_alignment(result)

    print(json.dumps({k: result[k] for k in ('baseline', 'selection', 'zero',
                                             'allocation', 'high_load', 'coverage',
                                             'scene_corr', 'scene_partial',
                                             'paired_baseline', 'alpha')},
                     ensure_ascii=False, indent=2, default=float))
    print('\nwrote', OUT / 'summary.json')
    print('wrote', FIGS)


if __name__ == '__main__':
    main()
