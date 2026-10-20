"""Attach interpretation, extra-region controls, and final integrity audit."""
from __future__ import annotations

import csv
from datetime import datetime
import json
import pickle
import re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.analyze_empty_decision_mini import RUN, REPORT, REGIONS, LABELS, sha
from tools.center_support_common import write_json


def read(name):
    return json.loads((RUN/name).read_text())


def pct(n, d):
    return 100*n/d


def verified_execution_source(relative_path, expected_sha256):
    """Keep experiment provenance intact after report-only code revisions."""
    current = ROOT/relative_path
    if sha(current) == expected_sha256:
        return str(current.relative_to(ROOT))
    snapshot = RUN/'source_snapshots'/current.name
    assert snapshot.exists() and sha(snapshot) == expected_sha256, relative_path
    return str(snapshot.relative_to(ROOT))


def write_class_csv(val):
    classes = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
    with (REPORT/'per_class_changes.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['region', 'class', 'native_iou', 'shift1_iou', 'delta_iou_pp',
            'correct_rescue', 'wrong_class_incoming', 'new_empty_false_positive',
            'occupied_misses_recovered_any_class'])
        for region in REGIONS:
            a, b = [np.asarray(val['confusion'][region][str(shift)]) for shift in [0., 1.]]
            delta = b-a
            ia = np.diag(a)/(a.sum(0)+a.sum(1)-np.diag(a)).clip(1)
            ib = np.diag(b)/(b.sum(0)+b.sum(1)-np.diag(b)).clip(1)
            for k, name in enumerate(classes):
                writer.writerow([region, name, 100*ia[k], 100*ib[k], 100*(ib[k]-ia[k]),
                    delta[k, k], delta[:17, k].sum()-delta[k, k], delta[17, k], -delta[k, 17]])


def main():
    val, train, probes, extra, refined = [read(n) for n in
        ['val/summary.json', 'train/summary.json', 'probes.json', 'all_region_probes.json', 'refined_probes.json']]
    assert probes['status'] == extra['status'] == 'complete'
    probe_execution_source = verified_execution_source(
        'tools/analyze_empty_decision_mini.py', probes['source_sha256'])
    assert val['protocol']['source_sha256'] == sha(ROOT/'tools/diagnose_empty_decision_mini.py')
    assert extra['source_sha256'] == sha(ROOT/'tools/probe_empty_decision_all_regions.py')
    assert refined['status'] == 'complete'
    assert refined['source_sha256'] == sha(ROOT/'tools/probe_empty_decision_refined.py')
    write_class_csv(val)
    main = probes['models']['correct_rescue']
    for region in REGIONS:
        assert abs(refined['models']['plus_contributor_semantics']['validation'][region]['auprc']-
                   main['position_class_semantics']['validation'][region]['auprc']) < 1e-12
    counts = val['counts']
    cam, out = counts['camera'], counts['camera_complement']
    def ap(model, region):
        return 100*main[model]['validation'][region]['auprc']
    def ci(model, region):
        c = main['paired_comparisons']['margin_position_class__to__'+model][region]['auprc_delta_ci95']
        return f'[{100*c[0]:+.2f}, {100*c[1]:+.2f}]'
    intro = ['## 先读结论', '',
        '**问题一得到局部、明确的支持：已有正确前景类别被 empty 压制，确实是当前模型的重要漏检类型；但降低 empty 并不是一个普遍安全的修复。**', '',
        f"- camera 区域的默认漏检中，**{pct(cam['native_fn_fg_correct'], cam['native_fn']):.2f}%** 在忽略 empty 后能选对类别；这些体素中 **{pct(val['support_counts']['camera']['0.1'][1], cam['native_fn_fg_correct']):.2f}%** 的实际核支持 S>0.1。这个结果支持继续研究输出决策，但不证明几何已经正确。",
        f"- b=+1 的实际翻转中，camera 区域 **{pct(cam['correct_rescue'],cam['flips']):.2f}%** 是正确语义被救回、**{pct(cam['wrong_class_exposed'],cam['flips']):.2f}%** 仍是错误类别、**{pct(cam['new_false_occupied'],cam['flips']):.2f}%** 是新增误占用；补集的新增误占用比例为 **{pct(out['new_false_occupied'],out['flips']):.2f}%**。不能把翻转数量直接当作 mIoU 收益。", '',
        '**问题二得到有限支持：局部信号在 camera 区域有额外预测价值，但收益较小，且 camera-only 训练的规律没有直接迁移到 mask 补集。**', '',
        f"- 控制 margin、位置和预测类别后，camera 正确救回 AP 从 **{ap('margin_position_class','camera'):.2f}%** 到全特征 **{ap('all_class','camera'):.2f}%**，增量 **{ap('all_class','camera')-ap('margin_position_class','camera'):+.2f} pp**，scene bootstrap 95% CI **{ci('all_class','camera')} pp**。支持和语义单独加入的结果分别见下表；图像代理的增量也必须结合区间判断。",
        f"- 单独加入支持、语义、图像代理，相对上述强对照的 camera AP 增量依次为 **{ap('position_class_support','camera')-ap('margin_position_class','camera'):+.2f} / {ap('position_class_semantics','camera')-ap('margin_position_class','camera'):+.2f} / {ap('position_class_image','camera')-ap('margin_position_class','camera'):+.2f} pp**。相同 20% 修正预算下，强对照到全特征的正确率仅从 **{100*main['margin_position_class']['validation']['camera']['precision_top20']:.2f}%** 到 **{100*main['all_class']['validation']['camera']['precision_top20']:.2f}%**，实际筛选增益应谨慎评价。",
        f"- 相同全特征模型在补集的 AP 为 **{ap('all_class','camera_complement'):.2f}%**，该区域自然正例比例为 **{100*main['all_class']['validation']['camera_complement']['prevalence']:.2f}%**。因此，当前实验不支持把 camera 内的有效信号直接当作全空间通用纠偏规则。",
        '- 增加补集训练标签的独立参照可以判断“该区域是否存在可预测信息”；它改变了训练域和行分布，不能包装为无额外监督的泛化提升。', '',
        '**更严格的细分对照进一步缩小了机制结论：组级“语义特征”的收益主要可以由渲染后分数置信度解释，不能直接归功于 Gaussian 贡献者的一致性。**', '',
        f"- 在 margin＋位置＋预测类别上再控制渲染后的分数间隔、归一化熵和分数总量，camera AP 已达 **{100*refined['models']['rendered_control']['validation']['camera']['auprc']:.2f}%**。再加入实际核支持的增量为 **{100*(refined['models']['plus_support']['validation']['camera']['auprc']-refined['models']['rendered_control']['validation']['camera']['auprc']):+.2f} pp**；再加入贡献者一致性的增量为 **{100*(refined['models']['plus_contributor_semantics']['validation']['camera']['auprc']-refined['models']['rendered_control']['validation']['camera']['auprc']):+.2f} pp**。详见细分对照表与置信区间。", '',
        '**对下一阶段的约束：先解决空间条件变化和误占用代价，再决定是否设计局部 empty 校正模块。当前结果尚不足以把“局部 bias 网络”认定为论文所需的新机制或强方法。**', '',
        '本实验完成了错误来源与预测关联验证；未完成因果干预、端到端方法训练或跨架构普遍性证明。', '']
    path = REPORT/'experiment_results.md'
    text = path.read_text()
    text = text.replace('包含有限体素抽样与固定 probe 的局限，不包含重训种子不确定性。',
        '区间条件于本次固定体素抽样和已拟合 probe，只重采样 scene，不包含体素重新抽样或 probe 重训的不确定性。')
    # The base generator is rerunnable; this finalizer replaces its own blocks.
    start, end = '<!-- INTERPRETATION_START -->', '<!-- INTERPRETATION_END -->'
    if start in text:
        text = text[:text.index(start)] + text[text.index(end)+len(end):]
    marker = '## 1. empty 修正具体改变了什么'
    text = text.replace(marker, start+'\n'+'\n'.join(intro)+'\n'+end+'\n\n'+marker, 1)
    a, b = '<!-- EXTRA_CONTROLS_START -->', '<!-- EXTRA_CONTROLS_END -->'
    if a in text:
        text = text[:text.index(a)] + text[text.index(b)+len(b):]
    extension = ['## 4. 增加补集训练标签的诊断参照', '',
        '沿用完全相同的训练/调参 scene 划分、200000/100000 行上限、树容量和 40/100 迭代选择规则。不同之处是候选训练行同时来自 camera 与补集，按各区域自然体素总体权重拟合。该设置增加了补集标签并改变训练行构成，不能与 camera-only probe 的差异简单归因于某一个因素。', '',
        '|目标与特征|camera AP|补集 AP|全空间 AP|', '|---|---:|---:|---:|']
    for task, label in [('correct_rescue','正确语义救回'), ('occupied','仅恢复 occupied')]:
        for name, nlabel in [('margin_position_class','margin＋位置＋类别'), ('all_class','全部特征＋类别')]:
            m = extra['models'][task][name]['validation']
            extension.append('|'+label+'：'+nlabel+'|'+'|'.join(f"{100*m[r]['auprc']:.2f}" for r in REGIONS)+'|')
    extension += ['', '正确语义救回目标下，相对 margin＋位置＋类别的全特征增量：', '',
        '|区域|ΔAP [95% scene bootstrap CI]|', '|---|---:|']
    for region in REGIONS:
        e = extra['models']['correct_rescue']
        delta = 100*(e['all_class']['validation'][region]['auprc']-e['margin_position_class']['validation'][region]['auprc'])
        lo, hi = e['paired_comparison'][region]['auprc_delta_ci95']
        extension.append(f'|{LABELS[region]}|{delta:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]|')
    extension += ['', '该参照允许判断局部特征是否在有相应区域标签时有信息；如果主要收益已由位置和预测类别解释，方法设计就必须与这些简单先验形成直接对照。camera 内外同时可靠仍需新的训练与方法实验，当前结果不构成解决方案。', '',
        '### 联合模型中，各信号是否仍有不可替代的贡献', '',
        '下面固定预测类别对照，逐组移除后重新拟合，报告“完整模型−移除模型”的 camera AP 差异。正值说明该组在其余特征已存在时仍有增量；这依然是预测消融，不是物理因果效应。', '',
        '|特征组|camera ΔAP [95% CI]|', '|---|---:|']
    for name, label in [('support','Gaussian 支持'), ('semantics','语义证据'), ('image','图像代理')]:
        base = 'all_class_without_'+name
        delta = ap('all_class','camera')-ap(base,'camera')
        lo, hi = main['paired_comparisons'][base+'__to__all_class']['camera']['auprc_delta_ci95']
        extension.append(f'|{label}|{delta:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]|')
    extension += ['', '当前选取的图像代理在联合模型中没有显示稳定的额外收益；这不能排除更合适的图像证据建模。', '',
        '## 5. 细分对照：是分数置信度，还是贡献者一致性', '',
        '这是观察组级结果后补充的探索性分析。保持 fit/tune 场景、抽样行、容量和迭代选择规则一致，将语义组拆成两部分：渲染后分数置信度（foreground top-2 间隔、正分数归一化熵、log(1＋分数总量)）与 Gaussian 贡献者统计（核加权硬类别一致性、投票熵、单 Gaussian 平均语义熵）。', '',
        '强对照包含 margin、位置、预测类别及全部上述渲染后分数。每次只增加一个特征组。不能把“语义组有效”直接解释成“贡献者分歧导致 empty”。', '',
        '|输入|camera AP|相对强对照 ΔAP [95% CI]|补集 AP|', '|---|---:|---:|---:|']
    for name, label in [('rendered_control','渲染分数强对照'), ('plus_support','强对照＋Gaussian 支持'),
                        ('plus_contributor_semantics','强对照＋贡献者一致性'), ('plus_image','强对照＋图像代理')]:
        metrics = refined['models'][name]['validation']
        if name == 'rendered_control':
            delta_text = '—'
        else:
            delta = 100*(metrics['camera']['auprc']-refined['models']['rendered_control']['validation']['camera']['auprc'])
            lo, hi = refined['comparisons'][name]['camera']['auprc_delta_ci95']
            delta_text = f'{delta:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]'
        extension.append(f"|{label}|{100*metrics['camera']['auprc']:.2f}|{delta_text}|{100*metrics['camera_complement']['auprc']:.2f}|")
    extension += ['', '因此，证据强度应表述为：实际核支持在普通分数置信度之外仍有小幅预测增量；当前贡献者一致性指标尚未显示稳定的额外增量。后者不支持直接把“多 Gaussian 语义冲突”写成已证实的主因。', '',
        '## 6. 哪些类别贡献了 camera mIoU 的变化', '',
        '以下是 b=+1 后 camera 区域单类 IoU 增加最多的五类。类 IoU 的平均恰好等于 mIoU，因此这里可以定位指标收益；与前面的体素翻转比例是不同统计。', '',
        '|类别|默认 IoU|b=+1 IoU|ΔIoU pp|正确救回体素|新增 empty 误占用体素|', '|---|---:|---:|---:|---:|---:|']
    rows = list(csv.DictReader((REPORT/'per_class_changes.csv').open()))
    rows = sorted([r for r in rows if r['region']=='camera'], key=lambda r: float(r['delta_iou_pp']), reverse=True)
    for row in rows[:5]:
        extension.append(f"|{row['class']}|{float(row['native_iou']):.2f}|{float(row['shift1_iou']):.2f}|{float(row['delta_iou_pp']):+.2f}|{int(row['correct_rescue']):,}|{int(row['new_empty_false_positive']):,}|")
    extension += ['', '完整类别与区域分解见 [per_class_changes.csv](per_class_changes.csv)。不能因为空体素的绝对误占用数多，就断言 mIoU 必然下降；指标还受到各类交并比及类别平均方式影响。', '',
        '额外参照复现：`python3.10 tools/probe_empty_decision_all_regions.py` 和 `python3.10 tools/probe_empty_decision_refined.py`；完成全部分析后运行 `python3.10 tools/finalize_empty_decision_mini.py` 更新解释与审计。', '']
    text += '\n'+a+'\n'+'\n'.join(extension)+'\n'+b+'\n'
    text = re.sub(r'\n{3,}', '\n\n', text).rstrip()+'\n'
    if '<!-- EDITORIAL_REVIEWED -->' in path.read_text():
        print('Preserving the reviewed narrative in', path)
    else:
        path.write_text(text)
    # Check binning error rather than silently treating bootstrap as exact.
    bin_errors = []
    for task in ['correct_rescue', 'occupied']:
        tm = probes['models'][task]
        for pair, regions in tm['paired_comparisons'].items():
            base, prop = pair.split('__to__')
            for region, c in regions.items():
                for metric in ['auroc', 'auprc']:
                    exact = tm[prop]['validation'][region][metric]-tm[base]['validation'][region][metric]
                    bin_errors.append(abs(exact-c[metric+'_binned_point_delta']))
    for task in ['correct_rescue', 'occupied']:
        tm = extra['models'][task]
        for region, c in tm['paired_comparison'].items():
            for metric in ['auroc', 'auprc']:
                exact = tm['all_class']['validation'][region][metric]-tm['margin_position_class']['validation'][region][metric]
                bin_errors.append(abs(exact-c[metric+'_binned_point_delta']))
    for name, regions in refined['comparisons'].items():
        for region, c in regions.items():
            for metric in ['auroc', 'auprc']:
                exact = refined['models'][name]['validation'][region][metric]-refined['models']['rendered_control']['validation'][region][metric]
                bin_errors.append(abs(exact-c[metric+'_binned_point_delta']))
    audit = read('completion_audit.json')
    assert train['samples'] == 256 and val['samples'] == 1024
    assert set(train['scene_ids']).isdisjoint(val['scene_ids'])
    manifest = read('probe_split_manifest.json')
    assert set(manifest['fit_scenes']).isdisjoint(manifest['tune_scenes'])
    assert set(manifest['fit_scenes']+manifest['tune_scenes']).isdisjoint(val['scene_ids'])
    assert extra['scene_manifest_sha256'] == sha(RUN/'probe_split_manifest.json')
    with (ROOT/train['protocol']['imageset']).open('rb') as source:
        full_training = pickle.load(source)['metadata']
    with (ROOT/val['protocol']['imageset']).open('rb') as source:
        full_validation = pickle.load(source)['metadata']
    assert len(full_training) == 4096 and len(full_validation) == 1024
    assert set(x[0] for x in full_training).isdisjoint(x[0] for x in full_validation)
    audit.update(status='complete', main_probes_per_target=20, probe_targets=2,
        additional_region_probes_per_target=2, additional_region_probes_complete=True,
        refined_correct_rescue_probes=4, refined_controls_complete=True,
        refined_shared_model_exact_metric_parity=True,
        probe_execution_source=probe_execution_source,
        reviewed_narrative_preserved='<!-- EDITORIAL_REVIEWED -->' in path.read_text(),
        full_training_samples=4096, full_validation_samples=1024,
        full_train_validation_scene_disjoint=True,
        fit_tune_validation_scene_disjoint=True, transition_counts_exact=True,
        maximum_bootstrap_binning_delta_error_pp=100*max(bin_errors),
        report_sha256=sha(path), report_files={str(p.relative_to(ROOT)):sha(p) for p in REPORT.iterdir() if p.is_file()},
        sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tools/diagnose_empty_decision_mini.py',
            ROOT/'tools/analyze_empty_decision_mini.py', ROOT/'tools/probe_empty_decision_all_regions.py',
            ROOT/'tools/probe_empty_decision_refined.py', Path(__file__).resolve()]},
        finished_at=datetime.now().astimezone().isoformat())
    write_json(RUN/'completion_audit.json', audit)
    write_json(RUN/'status.json', audit)
    print('Final report:', path, '\nMaximum bootstrap binning delta error (pp):', 100*max(bin_errors))


if __name__ == '__main__':
    main()
