"""Reconstruct weighted final-stage Set/field gradient relations from saved Gram data.

No model execution or new training. `all` denotes the occupied-GT domain,
not all decoder stages. Missing gradients are zero only in concatenated
parameter spaces; undefined directional cosines remain null.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work_dirs/center_support_contribution1'
PARAMETER_GROUPS = ('shared_parameters', 'classification_parameters',
                    'geometry_parameters', 'attribute_parameters', 'empty_parameters')


def pair(scene, a, b, group):
    key = f'{a} vs {b}'
    if key in scene['gradients']:
        return scene['gradients'][key][group]
    raw = scene['gradients'][f'{b} vs {a}'][group]
    return {**raw, 'norm_a': raw['norm_b'], 'norm_b': raw['norm_a']}


def gram(scene, a, b, group):
    if group == 'recorded_parameters':
        values = [gram(scene, a, b, g) for g in PARAMETER_GROUPS]
        return tuple(sum(v[i] for v in values) for i in range(3))
    raw = pair(scene, a, b, group)
    na, nb = raw['norm_a'], raw['norm_b']
    cosine = raw['cosine']
    if cosine is None:
        if min(na, nb) > 1e-12:
            raise ValueError('Missing cosine with two nonzero gradients')
        dot = 0.
    else:
        dot = na * nb * cosine
    return na * na, nb * nb, dot


def cosine(dot, aa, bb):
    return dot / math.sqrt(aa * bb) if min(aa, bb) > 1e-24 else None


def summarize(rows):
    result = {'scenes': len(rows)}
    for key in ('cosine', 'norm_ratio'):
        values = np.asarray([r[key] for r in rows if r[key] is not None])
        if not len(values):
            result[key] = None
            continue
        rng = np.random.default_rng(928)
        means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
        result[key] = dict(mean=float(values.mean()),
                           ci95=np.quantile(means, [.025, .975]).tolist(),
                           negative_scenes=int((values < 0).sum()),
                           valid_scenes=len(values))
    return result


def main():
    records, summaries, sources = [], [], []
    max_joint_norm_error = 0.
    for epoch in (1, 5, 10):
        source = RUN / f'real/epoch{epoch}_intervene_32.json'
        data = json.loads(source.read_text())
        assert data['status'] == 'complete' and len(data['pairs']) == 32
        sources.append(dict(path=str(source.relative_to(ROOT)),
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        for domain in ('all', 'camera'):
            c, g, v = f'{domain}/classification', f'{domain}/geometry', 'voxel'
            for group in ('shared_query_features', *PARAMETER_GROUPS, 'recorded_parameters'):
                local = []
                for scene in data['pairs']:
                    cc, gg, cg = gram(scene, c, g, group)
                    cc2, vv, cv = gram(scene, c, v, group)
                    gg2, vv2, gv = gram(scene, g, v, group)
                    assert math.isclose(cc, cc2, rel_tol=1e-10, abs_tol=1e-20)
                    assert math.isclose(gg, gg2, rel_tol=1e-10, abs_tol=1e-20)
                    assert math.isclose(vv, vv2, rel_tol=1e-10, abs_tol=1e-20)
                    ss = cc + gg + 2 * cg
                    assert ss >= -1e-18
                    ss = max(ss, 0.)
                    base = dict(epoch=epoch, domain=domain, group=group,
                                source_scene=scene['source_scene'])
                    for relation, dot, aa, bb in (
                            ('classification_vs_geometry', cg, cc, gg),
                            ('set_vs_field', cv + gv, ss, vv)):
                        row = dict(base, relation=relation, cosine=cosine(dot, aa, bb),
                                   norm_a=math.sqrt(aa), norm_b=math.sqrt(bb), dot=dot,
                                   norm_ratio=math.sqrt(aa / bb) if bb > 1e-24 else None)
                        records.append(row)
                        local.append(row)
                    if group == 'recorded_parameters':
                        reconstructed = math.sqrt(max(ss + vv + 2 * (cv + gv), 0.))
                        saved = next(x['gradient_norm'] for x in scene['interventions']
                                     if x['branch'] == domain + '/joint')
                        error = abs(reconstructed - saved) / max(saved, 1e-30)
                        max_joint_norm_error = max(max_joint_norm_error, error)
                        assert error < 1e-6, (epoch, domain, error)
                for relation in ('classification_vs_geometry', 'set_vs_field'):
                    summaries.append(dict(epoch=epoch, domain=domain, group=group,
                        relation=relation,
                        **summarize([r for r in local if r['relation'] == relation])))
    result = dict(status='complete', scope='Weighted final-stage losses; final decoder '
        'parameter groups only; eval-mode validation scenes; no new model runs.',
        domain_definition={'all': 'all valid occupied GT, original Set domain',
                           'camera': 'camera-visible occupied GT, aligned control'},
        reconstruction='s=c+g; dot(s,v)=dot(c,v)+dot(g,v); '
                       'norm(s)^2=norm(c)^2+norm(g)^2+2dot(c,g). '
                       'Original component weights are already included.',
        uncertainty='10,000 scene bootstrap draws, seed 928; no training-seed uncertainty.',
        max_joint_norm_relative_error=max_joint_norm_error,
        sources=sources, summaries=summaries, records=records)
    destination = RUN / 'set_geometry_gradient_relations_audit.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Set 内部与 Set—体素场梯度关系复核', '',
        '本次仅离线重算已有记录，没有运行模型或新增训练。Set 几何为双向中心损失之和；'
        'Set 总项为分类加几何；体素场为 CE＋Lovasz，全部保留原始权重。', '',
        '**范围：最终阶段、最终 decoder 相关参数组，不能代表六阶段与全网络。** '
        '`all` 指全部有效 occupied GT，而不是全部阶段；`camera` 为对齐监督域的控制。'
        '分类和几何在各自独立输出头上存在缺失梯度，因此内部冲突应重点看共享参数。', '',
        '下表余弦为逐场景余弦的均值；区间为场景 bootstrap 95% 区间。'
        '范数比为逐场景比值的均值，比较项顺序为分子／分母。', '']
    for domain in ('all', 'camera'):
        lines += [f'## 监督域：{domain}', '',
            '| 比较 | 参数范围 | epoch | 余弦均值 [95% CI] | 负余弦场景 | 范数比 |',
            '|---|---|---:|---|---:|---:|']
        for relation, label, groups in (
                ('classification_vs_geometry', '分类／几何', ('shared_query_features', 'shared_parameters')),
                ('set_vs_field', 'Set／场', ('shared_parameters', 'recorded_parameters'))):
            for group in groups:
                for epoch in (1, 5, 10):
                    row = next(r for r in summaries if (r['domain'], r['relation'], r['group'], r['epoch'])
                               == (domain, relation, group, epoch))
                    stat, ratio = row['cosine'], row['norm_ratio']
                    lo, hi = stat['ci95']
                    lines.append(f"| {label} | {group} | {epoch} | {stat['mean']:+.5f} "
                        f"[{lo:+.5f}, {hi:+.5f}] | {stat['negative_scenes']}/{stat['valid_scenes']} "
                        f"| {ratio['mean']:.4f} |")
        lines.append('')
    lines += ['## 几何子项检查', '',
        '双向几何项相加可能掩盖某一个方向的竞争。以下保留原始 all 域共享参数比较：', '',
        '| 分类对比项 | epoch | 平均余弦 [95% CI] | 负余弦场景 |', '|---|---:|---|---:|']
    for epoch in (1, 5, 10):
        data = json.loads((RUN / f'real/epoch{epoch}_intervene_32.json').read_text())
        for name in ('pred_to_gt', 'gt_to_pred'):
            values = [pair(s, 'all/classification', f'all/{name}', 'shared_parameters')['cosine']
                      for s in data['pairs']]
            stat = summarize([dict(cosine=x, norm_ratio=None) for x in values])['cosine']
            lo, hi = stat['ci95']
            lines.append(f'| {name} | {epoch} | {np.mean(values):+.5f} '
                         f'[{lo:+.5f}, {hi:+.5f}] | '
                         f'{sum(x < 0 for x in values)}/{len(values)} |')
    lines += ['', '## 重建校验与限制', '',
        f'- 重建 Set＋场总梯度范数，与已保存的联合干预范数比较，最大相对误差 '
        f'{max_joint_norm_error:.3g}；192 个 epoch／监督域／场景条件全部通过。',
        '- 正平均余弦不排除个别场景、参数或几何子项的冲突；近零余弦也不意味着强度相同。',
        '- 这是欧氏、未预条件梯度；未测 AdamW 更新夹角、全网络或六阶段总 Set。',
        '- 不能用现有分项余弦的简单平均代表 Set 总项；此处由逐场景范数与内积重建。',
        '- 分类／几何比较中独立参数头的缺失梯度，不解释为有意义的“零余弦”。', '',
        '原始数据及逐场景重建结果：'
        '[audit.json](../work_dirs/center_support_contribution1/set_geometry_gradient_relations_audit.json)。',
        '复核脚本：[audit_set_geometry_gradient_relations.py](../tools/audit_set_geometry_gradient_relations.py)。', '']
    report = ROOT / 'analysis/set_geometry_gradient_relations_audit.md'
    report.write_text('\n'.join(lines))
    print(str(destination))
    print(str(report))
    print(f'max_joint_norm_relative_error={max_joint_norm_error:.3g}')


if __name__ == '__main__':
    main()
