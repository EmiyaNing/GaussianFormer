"""Full-census summaries, controls, paired scene intervals and readable report."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.probe_gaussian_scale_supervision import RUN,GROUPS,CONDITIONS,FIELDS,FLAGS,MARGIN_EDGES

ZH=('跨前景类别混合','前景／empty 交界','纯前景','纯 empty、未主导 FP','纯 empty、主导 FP','无有效支持','全部高斯',
    '主导 FP：中心在可见 empty','主导 FP：中心在 mask 外前景','主导 FP：中心在 mask 外 empty','主导 FP：全网格支持纯 empty')
CONDITION_ZH=('原始模型 B0','尺度限制 S','等证据 opacity 对照 O','缩尺度并恢复证据 R')


def read(p):return json.loads(p.read_text())


def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))


def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def safe(x,n):return float(x/n) if n else None


def metrics(c):
    c=np.asarray(c,dtype=np.float64);diagonal=np.diag(c);gt=c.sum(1);pred=c.sum(0)
    denominator=gt[:17]+pred[:17]-diagonal[:17]
    union=denominator>0
    miou=float(np.mean(diagonal[:17][union]/denominator[union])) if union.any() else None
    tp=float(c[:17,:17].sum());fn=float(c[:17,17].sum());fp=float(c[17,:17].sum());foreground=gt[:17].sum()
    return dict(mIoU=miou,binary_IoU=safe(tp,tp+fn+fp),miss_rate=safe(fn,foreground),
        false_positive_rate=safe(fp,gt[17]),wrong_class_rate=safe(tp-diagonal[:17].sum(),foreground),
        final_foreground_correct_rate=safe(diagonal[:17].sum(),foreground),
        gt_foreground=int(foreground),gt_empty=int(gt[17]),false_positives=int(fp),misses=int(fn))


def curve(hist,reference_fp):
    # bucketize(right=True): bins >= j+1 implement margin >= threshold[j].
    cumulative=np.cumsum(hist[:,:,::-1],axis=2)[:,:,::-1]
    total=hist.sum((1,2));rows=[]
    for j,threshold in enumerate(MARGIN_EDGES):
        foreground=cumulative[:,:,j+1]
        c=np.zeros((18,18),dtype=np.int64);c[:,:17]=foreground;c[:,17]=total-foreground.sum(1)
        rows.append(dict(threshold=float(threshold),**metrics(c)))
    feasible=[r for r in rows if r['false_positives']<=reference_fp]
    selected=feasible[0] if feasible else dict(threshold=None,status='No feasible registered finite threshold')
    return rows,selected


def summarize_groups(a):
    result={}
    for i,name in enumerate(GROUPS):
        n=int(a['counts'][i]);mean=a['sums'][i]/max(n,1)
        row=dict(gaussians=n,fraction_of_all=safe(n,int(a['counts'][6])),
            means={f:float(mean[j]) for j,f in enumerate(FIELDS)},
            positive_fraction={f:safe(int(a['positive'][i,j]),n) for j,f in enumerate(FIELDS)},
            negative_fraction={f:safe(int(a['negative'][i,j]),n) for j,f in enumerate(FIELDS)},
            flag_fractions={f:safe(int(a['flags'][i,j]),n) for j,f in enumerate(FLAGS)},
            saturated_axis_count_fraction=(a['saturation_hist'][i]/max(n,1)).tolist())
        fp_owner_count=int(round(a['sums'][i,FIELDS.index('fp_owner_voxels')]))
        row['owned_false_positive_voxels']=fp_owner_count
        row['fraction_of_false_positive_voxels']=safe(fp_owner_count,int(a['confusion'][0,17,:17].sum()))
        raw=sum(a['abs_sums'][i,FIELDS.index('ce_empty_raw_scale_'+axis)] for axis in 'xyz')
        physical=sum(a['abs_sums'][i,FIELDS.index('ce_empty_scale_'+axis)] for axis in 'xyz')
        row['ce_scale_gradient_jacobian_relative_to_peak']=safe(raw,.08*physical)
        row['ce_target_to_set_target_absolute_gradient_ratio']=safe(
            a['abs_sums'][i,FIELDS.index('ce_empty_target')],a['abs_sums'][i,FIELDS.index('set_target')])
        row['ce_opacity_gradient_jacobian_relative_to_peak']=safe(
            a['abs_sums'][i,FIELDS.index('ce_empty_raw_opacity')],.25*a['abs_sums'][i,FIELDS.index('ce_empty_opacity')])
        # Empty CE has nonnegative physical opacity derivative; multiplying by
        # alpha*(1-alpha) cannot create a positive derivative from zero.
        physical_positive=int(a['positive'][i,FIELDS.index('ce_empty_opacity')])
        raw_positive=int(a['positive'][i,FIELDS.index('ce_empty_raw_opacity')])
        assert raw_positive<=physical_positive
        row['ce_opacity_physical_positive_but_raw_zero_fraction']=safe(physical_positive-raw_positive,n)
        row['joint_target_negative_given_strong_camera']=safe(
            int(a['flags'][i,FLAGS.index('joint_target_negative_strong_camera')]),
            int(a['flags'][i,FLAGS.index('camera_fraction_ge_05')]))
        result[name]=row
    return result


def bootstrap(confusions,repetitions=1000):
    rng=np.random.default_rng(928);result={}
    for split,values in confusions.items():
        a=np.array(values,dtype=np.float64);n=len(a)
        weights=rng.multinomial(n,np.full(n,1/n),size=repetitions).astype(np.float64)
        sums=(weights@a.reshape(n,-1)).reshape(repetitions,4,18,18)
        derived=[[] for _ in range(4)]
        for row in sums:
            for k in range(4):derived[k].append(metrics(row[k]))
        estimates={}
        for k,name in enumerate(CONDITIONS):
            estimates[name]={}
            for metric in ('mIoU','binary_IoU','miss_rate','false_positive_rate','wrong_class_rate'):
                vals=np.array([r[metric] for r in derived[k]])
                base=np.array([r[metric] for r in derived[0]])
                estimates[name][metric]=dict(interval95=np.quantile(vals,[.025,.975]).tolist(),
                    difference_from_baseline_pp_interval95=(100*np.quantile(vals-base,[.025,.975])).tolist())
        result[split]=estimates
    result['protocol']=dict(repetitions=repetitions,seed=928,unit='paired full scenes within each split',
        scope='native intervention decisions only; not training seed uncertainty; matched-FPR thresholds not bootstrapped')
    return result


def aggregate(partial=False,with_bootstrap=False):
    protocol=read(RUN/'protocol.json');totals={};metadata={};scene_rows=[];scene_groups=[];confusions={};controls={}
    for split in ('train','val'):
        totals[split]=None;metadata[split]=dict(scenes=0,frames=0);confusions[split]=[]
        controls[split]=dict(maximum_mass_match_relative_error=0.,maximum_mass_restore_relative_error=0.,
            eligible_gaussians=0,saturated_gaussians=0,excluded_zero_or_tiny_mass_gaussians=0,maximum_restore_factor=0.)
        for path in sorted((RUN/'scenes'/split).glob('*.json')):
            row=read(path)
            with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
            assert a['counts'][6]==row['frames']*76800
            assert a['counts'][:6].sum()==a['counts'][6]
            assert a['counts'][7:10].sum()==a['counts'][4]
            assert a['counts'][10]<=a['counts'][4]
            assert round(a['sums'][6,FIELDS.index('fp_owner_voxels')])==a['confusion'][0,17,:17].sum()
            np.testing.assert_array_equal(a['confusion'],np.sum([r['confusion'] for r in row['records']],axis=0))
            np.testing.assert_array_equal(a['counts'][:6],np.sum([r['gaussian_counts'] for r in row['records']],axis=0))
            assert (a['hist'].sum(2)==a['counts'][:,None]).all()
            assert (a['axis_hist'].sum(2)==a['counts'][:,None]).all()
            np.testing.assert_array_equal(a['coverage_hist'].sum(1),a['counts'])
            np.testing.assert_array_equal(a['saturation_hist'].sum(1),a['counts'])
            assert ((a['positive']+a['negative'])<=a['counts'][:,None]).all()
            assert ((a['flags']>=0)&(a['flags']<=a['counts'][:,None])).all()
            for key,value in a.items():
                assert np.isfinite(value).all(),(path,key)
                if key in ('counts','confusion','margin_hist'):continue
                np.testing.assert_allclose(value[:6].sum(0),value[6],rtol=1e-12,atol=1e-12)
            if totals[split] is None:totals[split]={k:np.zeros_like(v) for k,v in a.items()}
            for k,v in a.items():totals[split][k]+=v
            metadata[split]['scenes']+=1;metadata[split]['frames']+=row['frames']
            for frame in row['records']:
                check=frame['control'];control=controls[split]
                control['maximum_mass_match_relative_error']=max(control['maximum_mass_match_relative_error'],check['mass_match_relative_error'])
                control['maximum_mass_restore_relative_error']=max(control['maximum_mass_restore_relative_error'],check['mass_restore_relative_error'])
                control['maximum_restore_factor']=max(control['maximum_restore_factor'],check['maximum_restore_factor'])
                control['eligible_gaussians']+=check['eligible'];control['saturated_gaussians']+=check['saturated']
                control['excluded_zero_or_tiny_mass_gaussians']+=check['unsupported_restore']
            confusions[split].append(a['confusion'])
            for k,name in enumerate(CONDITIONS):
                scene_rows.append(dict(split=split,scene=row['scene_name'],condition=name,frames=row['frames'],**metrics(a['confusion'][k])))
            for name,g in summarize_groups(a).items():
                scene_groups.append(dict(split=split,scene=row['scene_name'],group=name,gaussians=g['gaussians'],
                    camera_mass_fraction=g['means']['camera_mass_fraction'],opacity=g['means']['opacity'],
                    scale_min=g['means']['scale_min'],scale_max=g['means']['scale_max'],
                    all_axes_ge_039=g['flag_fractions']['all_axes_ge_039'],
                    ce_empty_target_positive=g['flag_fractions']['ce_empty_target_positive'],
                    joint_target_negative=g['flag_fractions']['joint_target_negative'],
                    scale_gradient_jacobian=g['ce_scale_gradient_jacobian_relative_to_peak']))
        if not partial:assert metadata[split]==protocol['expected'][split],(split,metadata[split])
    if any(v is None for v in totals.values()):return dict(status='partial',completed=metadata)
    totals['total']={k:totals['train'][k]+totals['val'][k] for k in totals['train']}
    status='complete' if metadata==protocol['expected'] else 'partial'
    summary=dict(status=status,completed=metadata,checkpoint_sha256=protocol['checkpoint_sha256'],statistics={},control_audits=controls)
    curves=[];global_groups=[]
    for split,a in totals.items():
        groups=summarize_groups(a);conditions={};budget=int(a['confusion'][0,17,:17].sum())
        for k,name in enumerate(CONDITIONS):
            native=metrics(a['confusion'][k]);rows,selected=curve(a['margin_hist'][k],budget)
            atzero=rows[int(np.argmin(np.abs(MARGIN_EDGES)))]
            for key in ('false_positives','misses','mIoU'):
                assert abs(atzero[key]-native[key])<1e-12,(split,name,key)
            conditions[name]=dict(native=native,matched_fp_budget=selected)
            curves.extend(dict(split=split,condition=name,**r) for r in rows)
        summary['statistics'][split]=dict(groups=groups,conditions=conditions)
        for name,g in groups.items():
            global_groups.append(dict(split=split,group=name,gaussians=g['gaussians'],fraction_of_all=g['fraction_of_all'],
                **g['means'],**{'flag_'+k:v for k,v in g['flag_fractions'].items()},
                ce_scale_gradient_jacobian=g['ce_scale_gradient_jacobian_relative_to_peak'],
                ce_target_to_set_target_gradient_ratio=g['ce_target_to_set_target_absolute_gradient_ratio']))
    target=RUN/('summary.json' if status=='complete' else 'partial_summary.json');write(target,summary)
    if status=='complete':
        csv_write(RUN/'scene_metrics.csv',scene_rows);csv_write(RUN/'scene_gaussian_groups.csv',scene_groups)
        csv_write(RUN/'global_gaussian_groups.csv',global_groups);csv_write(RUN/'decision_curves.csv',curves)
        native_rows=[]
        for split,data in summary['statistics'].items():
            for name,r in data['conditions'].items():native_rows.append(dict(split=split,condition=name,**r['native']))
        csv_write(RUN/'global_metrics.csv',native_rows)
        write(RUN/'integrity_audit.json',dict(status='passed',completed=metadata,
            gaussian_count=sum(x['frames'] for x in metadata.values())*76800,
            per_frame_original_error_counts_exact=True,group_and_histogram_totals_exact=True,
            margin_zero_reconstructs_confusion_exact=True))
        if with_bootstrap:write(RUN/'scene_bootstrap.json',bootstrap(confusions))
    return summary


def pct(x):return '—' if x is None else f'{x*100:.3f}%'


def report(summary):
    """Render the readable narrative while deriving all result tables from data."""
    import re

    assert summary['status'] == 'complete'
    data = summary['statistics']
    splits = ('train', 'val')
    focus = [data[s]['groups']['pure_empty_fp_owner'] for s in splits]
    names = ('B0：原始模型', 'S：直接缩小', 'O：只降低贡献', 'R：缩小后补回贡献')
    context = {}

    def table(headers, rows):
        lines = ['| ' + ' | '.join(headers) + ' |',
                 '|---|' + '---:|' * (len(headers) - 1)]
        lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        return '\n'.join(lines)

    def comparison(rows, groups=focus, headers=('指标', '训练集', '验证集')):
        return table(headers, [[label, *(fmt(get(g)) for g in groups)]
                               for label, get, fmt in rows])

    def flag(key):
        return lambda g: g['flag_fractions'][key]

    def mean(key):
        return lambda g: g['means'][key]

    count = lambda n: f'{n:,}'
    decimal = lambda n: f'{n:.3f}'
    four = lambda n: f'{n:.4f}'
    focus_table_groups = [
        dict(g, pure_empty_fraction=g['gaussians'] / (
            g['gaussians'] + data[s]['groups']['pure_empty_no_fp_owner']['gaussians']))
        for s, g in zip(splits, focus)
    ]
    context['focus_table'] = comparison([
        ('该组高斯实例数', lambda g: g['gaussians'], count),
        ('该组／全部高斯', lambda g: g['fraction_of_all'], pct),
        ('该组／全部纯空高斯', lambda g: g['pure_empty_fraction'], pct),
        ('归属于该组的 FP／全部 FP', lambda g: g['fraction_of_false_positive_voxels'], pct),
        ('每个该组高斯平均主导的 FP 体素数', mean('fp_owner_voxels'), decimal),
    ], focus_table_groups)
    context['frequency_ratio'] = f"{focus[1]['fraction_of_all'] / focus[0]['fraction_of_all']:.2f}"
    strict_keys = ('pure_empty_fp_owner_visible_empty_center', 'pure_empty_fp_owner_all_support_empty')
    strict_rows = []
    for key, label in zip(strict_keys, ('中心可见且为空', '整个影响范围都为空')):
        groups = [data[s]['groups'][key] for s in splits]
        strict_rows.append([label + '：占重点组', *[
            pct(g['gaussians'] / f['gaussians']) for g, f in zip(groups, focus)]])
        strict_rows.append([label + '：归属 FP／全部 FP', *[
            pct(g['fraction_of_false_positive_voxels']) for g in groups]])
    context['strict_overview'] = table(('检查条件与分母', '训练集', '验证集'), strict_rows)
    visible = data['val']['groups'][strict_keys[0]]
    context['visible_fp_share'] = pct(visible['fraction_of_false_positive_voxels'])
    context['visible_scale_zero'] = pct(visible['flag_fractions']['ce_scale_wants_shrink_but_raw_zero'])
    context['visible_opacity'] = four(visible['means']['opacity'])
    context['supervision_table'] = comparison([
        ('平均可见覆盖比例', mean('camera_mass_fraction'), pct),
        ('可见覆盖比例至少 50% 的高斯', flag('camera_fraction_ge_05'), pct),
        ('平均 camera 核权重和（无量纲）', mean('camera_kernel_mass'), four),
        ('CE 要求压低目标类别', flag('ce_empty_target_positive'), pct),
        ('CE 要求至少一个轴缩小', flag('ce_empty_scale_any_positive'), pct),
    ])
    context['conflict_table'] = comparison([
        ('空 CE 与 Set 对目标类别方向相反', flag('ce_set_target_opposed'), pct),
        ('四项合计仍要求抬高目标类别', flag('joint_target_negative'), pct),
        ('空 CE／Set 的目标类绝对梯度总量比', lambda g: g['ce_target_to_set_target_absolute_gradient_ratio'], decimal),
        ('空 CE 要求降低整体语义幅度', lambda g: g['positive_fraction']['ce_empty_log_amplitude'], pct),
        ('Set 要求提高整体语义幅度', lambda g: g['negative_fraction']['set_log_amplitude'], pct),
        ('四项合计仍要求提高整体语义幅度', lambda g: g['negative_fraction']['joint_log_amplitude'], pct),
    ])
    context['scale_focus_table'] = comparison([
        ('三轴都达到 0.39 米', flag('all_axes_ge_039'), pct),
        ('CE 要求缩小，但三轴原始尺度梯度全为零', flag('ce_scale_wants_shrink_but_raw_zero'), pct),
        ('加权尺度转换系数／最大值', lambda g: g['ce_scale_gradient_jacobian_relative_to_peak'], pct),
    ])
    context['foreground_saturated'] = pct(data['val']['groups']['pure_foreground']['flag_fractions']['all_axes_ge_039'])
    metric_keys = ('mIoU', 'binary_IoU', 'miss_rate', 'false_positive_rate', 'wrong_class_rate')
    for split in splits:
        conditions = data[split]['conditions']
        context[split + '_metrics'] = table(
            ('条件', 'mIoU', '二值 IoU', '漏检率', '假阳性率', '错分类率'),
            [[name, *(pct(conditions[key]['native'][m]) for m in metric_keys)]
             for name, key in zip(names, CONDITIONS)])
        context[split + '_matched'] = table(
            ('条件', '决策阈值', 'mIoU', '漏检率', '实际假阳性率'),
            [[name, f"{conditions[key]['matched_fp_budget']['threshold']:.2f}",
              *(pct(conditions[key]['matched_fp_budget'][m]) for m in ('mIoU', 'miss_rate', 'false_positive_rate'))]
             for name, key in zip(names, CONDITIONS)])
    vc = data['val']['conditions']
    for prefix, key in (('base', 'baseline'), ('shrink', 'scale_cap30')):
        context[prefix + '_fp'] = pct(vc[key]['native']['false_positive_rate'])
        context[prefix + '_miss'] = pct(vc[key]['native']['miss_rate'])
    base = vc['baseline']['native']['mIoU']
    context['matched_gain'] = f"{100 * (vc['opacity_mass_matched']['matched_fp_budget']['mIoU'] - base):.3f}"
    intervals = read(RUN / 'scene_bootstrap.json')['val']
    context['intervals'] = table(('条件', 'mIoU 变化（百分点）', '配对 95% 区间'), [
        [name, f"{100 * (vc[key]['native']['mIoU'] - base):+.3f}",
         '[' + ', '.join(f'{v:+.3f}' for v in intervals[key]['mIoU']['difference_from_baseline_pp_interval95']) + ']']
        for name, key in zip(names[1:], CONDITIONS[1:])])
    context['attributes'] = comparison([
        ('平均不透明度', mean('opacity'), four),
        ('平均最高前景语义值', mean('semantic_max'), four),
        ('平均最小轴尺度（米）', mean('scale_min'), lambda x: f'{x:.5f}'),
        ('平均最大轴尺度（米）', mean('scale_max'), lambda x: f'{x:.5f}'),
    ])
    context['centers'] = table(('中心位置', '训练集', '验证集'), [
        [label, *(pct(data[s]['groups'][key]['gaussians'] / focus[i]['gaussians']) for i, s in enumerate(splits))]
        for key, label in zip(GROUPS[7:10], ('可见空区域', 'mask 外前景', 'mask 外空区域'))])
    for split in splits:
        context['strict_' + split] = comparison([
            ('高斯实例数', lambda g: g['gaussians'], count),
            ('占全部高斯', lambda g: g['fraction_of_all'], pct),
            ('平均不透明度', mean('opacity'), four),
            ('CE 要求压低目标类别', flag('ce_empty_target_positive'), pct),
            ('合计仍要求抬高目标类别', flag('joint_target_negative'), pct),
            ('CE 要求缩小，但三轴原始尺度梯度全零', flag('ce_scale_wants_shrink_but_raw_zero'), pct),
        ], [data[split]['groups'][key] for key in strict_keys],
           ('指标', '中心可见且为空', '整个影响范围都为空'))
    size_rows, expansion_rows = [], []
    for split, label in zip(splits, ('训练', '验证')):
        for key, group_label in (('pure_foreground', '纯前景'), ('pure_empty_no_fp_owner', '纯空、未主导 FP'),
                                 ('pure_empty_fp_owner', '纯空且主导 FP')):
            g = data[split]['groups'][key]
            size_rows.append([label + '：' + group_label, *[
                pct(g['flag_fractions'][k]) for k in ('scale_max_ge_039', 'all_axes_ge_039',
                    'all_scale_sigmoid_gradients_zero', 'ce_scale_wants_shrink_but_raw_zero')],
                pct(g['ce_scale_gradient_jacobian_relative_to_peak'])])
        for key, group_label in (('foreground_mixed', '多前景类别混合'),
                                 ('foreground_empty_interface', '前景与空区域交界'),
                                 ('pure_foreground', '纯前景'), ('pure_empty_fp_owner', '纯空且主导 FP')):
            g = data[split]['groups'][key]
            expansion_rows.append([label + '：' + group_label, *[
                pct(g['negative_fraction']['occupancy_scale_' + axis]) for axis in 'xyz']])
    context['size_comparison'] = table(('集合与组别', '至少一轴 ≥0.39 米', '三轴均 ≥0.39 米',
        '三轴 sigmoid 导数全零', 'CE 要求缩小但原始梯度全零', '转换系数比'), size_rows)
    context['expansion'] = table(('集合与组别', 'x 轴要求扩大', 'y 轴要求扩大', 'z 轴要求扩大'), expansion_rows)
    context['opacity'] = comparison([
        ('要求降低不透明度，但转换后梯度为零的高斯比例',
         lambda g: g['ce_opacity_physical_positive_but_raw_zero_fraction'], pct),
        ('加权不透明度转换系数／最大值', lambda g: g['ce_opacity_gradient_jacobian_relative_to_peak'], pct),
    ])
    context['control_quality'] = comparison([
        ('实际干预高斯实例数', lambda g: g['eligible_gaussians'], count),
        ('尺度达标但核权重和过小、保持原值的实例数', lambda g: g['excluded_zero_or_tiny_mass_gaussians'], count),
        ('S/O 最大相对总量误差', lambda g: g['maximum_mass_match_relative_error'], lambda x: f'{x:.2e}'),
        ('R/B0 最大相对总量误差', lambda g: g['maximum_mass_restore_relative_error'], lambda x: f'{x:.2e}'),
        ('最大语义恢复倍数', lambda g: g['maximum_restore_factor'], lambda x: f'{x:.3g}'),
    ], [summary['control_audits'][s] for s in splits])
    template = (ROOT / 'tools/report_templates/gaussian_scale_supervision_full_results.md').read_text()
    rendered = re.sub(r'\{\{([a-z_]+)\}\}', lambda m: context[m.group(1)], template)
    assert '{{' not in rendered and '}}' not in rendered
    (ROOT / 'analysis/gaussian_scale_supervision_full_results.md').write_text(rendered)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--bootstrap',action='store_true');p.add_argument('--write-report',action='store_true');a=p.parse_args()
    s=aggregate(a.allow_partial,a.bootstrap)
    if a.write_report:report(s)
    print(json.dumps(dict(status=s['status'],completed=s['completed'])))
