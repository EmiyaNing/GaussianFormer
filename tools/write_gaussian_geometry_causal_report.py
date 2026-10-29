"""Write the reviewed causal interpretation from completed full-scene results.

This is the human-reviewed narrative for this registered run, separate from the
automatic table collectors. It refuses incomplete inputs or changed main results.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work_dirs/gaussian_geometry_causal'
REPORT = ROOT / 'analysis/gaussian_geometry_causal_results.md'


def read(name):
    return json.loads((RUN / name).read_text())


def table(columns, rows):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(columns)) + ' |',
                      *['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]])


def interval(x, digits=4):
    return '[' + ', '.join(f'{v:+.{digits}f}' for v in x) + ']'


def main():
    assert (RUN / 'all_execution_complete.json').exists()
    s = read('summary.json')
    m = read('mature_summary.json')
    f = read('factorial_interaction.json')['splits']['val']
    opt = read('optimizer_diagnostics.json')
    base, control = 'mature_standard', 'mature_no_empty_center_boost'
    st, ct = m['trained'][base], m['trained'][control]
    # The interpretation below was reviewed against these exact full-validation
    # results. Changed experiments require a new review, not stale prose.
    assert abs(st['val']['mIoU'] - 39.82892375194736) < 1e-8
    assert abs(ct['val']['mIoU'] - 39.824307158492026) < 1e-8
    assert m['intervals']['val']['mIoU'][0] < 0 < m['intervals']['val']['mIoU'][1]
    names = ['common', 'standard', 'no_amplitude', 'no_geometry', 'no_both',
             'no_empty_center_boost']
    labels = {'common': '共同起点', 'standard': '标准训练',
              'no_amplitude': '阻断幅度', 'no_geometry': '阻断几何',
              'no_both': '同时阻断', 'no_empty_center_boost': '针对性控制'}
    mature_rows = []
    for label, values in [('原 30 epoch 权重', m['reference']),
                          ('标准继续训练', st), ('针对性控制', ct)]:
        for split, split_label in [('train', '训练'), ('val', '验证')]:
            mature_rows.append([label, split_label,
                               *[f'{values[split][k]:.4f}%' for k in
                                 ('mIoU', 'IoU', 'FN_rate', 'FPR', 'semantic_error')]])
    mature_table = table(['模型', '集合', 'mIoU ↑', '占有 IoU ↑', '漏检率 ↓',
                          '假阳性率 ↓', '前景语义错分率 ↓'], mature_rows)
    delta_table = table(['验证集指标', '控制减标准／百分点', '场景 bootstrap 95% 区间'],
                        [[label, f'{ct["val"][key] - st["val"][key]:+.4f}',
                          interval(m['intervals']['val'][key])]
                         for key, label in [('mIoU', 'mIoU'), ('IoU', '占有 IoU'),
                                            ('FN_rate', '漏检率'), ('FPR', '假阳性率')]])
    geometry = table(['成熟权重验证集', '标准继续训练', '针对性控制'],
                     [[label, f'{st["val"][key]:.{digits}f}{suffix}',
                       f'{ct["val"][key]:.{digits}f}{suffix}']
                      for key, label, digits, suffix in [
                          ('n_1sigma', '每个 GT 前景体素的 1σ GS 数', 5, ''),
                          ('n_2sigma', '每个 GT 前景体素的 2σ GS 数', 5, ''),
                          ('no_1sigma', '没有 1σ 核心覆盖的前景体素', 4, '%'),
                          ('no_2sigma', '没有 2σ 覆盖的前景体素', 4, '%'),
                          ('no_native_support', '没有任何原生支持的前景体素', 4, '%'),
                          ('GS_center_camera_empty', '中心在可见 empty 的 GS／全部 GS', 4, '%'),
                          ('GS_sum_A', 'GS 平均语义幅度 A', 5, ''),
                          ('GS_sum_alpha', 'GS 平均不透明度 α', 5, ''),
                          ('GS_sum_b', 'GS 平均证据系数 b', 5, '')]])
    early = table(['早期组别', '验证 mIoU', '验证占有 IoU', '验证漏检率',
                    '验证假阳性率', '完整训练集加权 CE'],
                   [[labels[n], *[f'{s[n]["val"][k]:.3f}%' for k in
                                  ('mIoU', 'IoU', 'FN_rate', 'FPR')],
                     f'{s[n]["train"]["CE_weighted"]:.4f}'] for n in names])
    conditional = table(['保留 Occupancy 几何路径的作用', '验证 mIoU 收益／百分点',
                          '场景 bootstrap 95% 区间'],
                         [[label, f'{f[key]["estimate"]:+.4f}',
                           interval(f[key]['scene_bootstrap_95'])]
                          for key, label in [
                              ('geometry_effect_amplitude_available', '幅度路径保留时：标准 − 阻断几何'),
                              ('geometry_effect_amplitude_blocked', '幅度路径阻断时：阻断幅度 − 同时阻断')]])
    early_geometry = table(['早期验证组别', '2σ GS 数', '没有 1σ 核心覆盖',
                             '没有 2σ 覆盖', 'GS 平均 A', 'GS 平均最大类别份额'],
                            [[labels[n], f'{s[n]["val"]["n_2sigma"]:.3f}',
                              f'{s[n]["val"]["no_1sigma"]:.3f}%',
                              f'{s[n]["val"]["no_2sigma"]:.3f}%',
                              f'{s[n]["val"]["GS_sum_A"]:.4f}',
                              f'{s[n]["val"]["GS_sum_max_p"]:.4f}'] for n in names[1:]])
    cal_rows = []
    for stage, values, selected in [
        ('早期', read('train_only_calibration.json'), names[1:]),
        ('成熟', m['calibration'], [base, control])]:
        for name in selected:
            value = values[name]
            label = labels.get(name, '标准继续训练' if name == base else '针对性控制')
            cal_rows.append([stage + '：' + label, f'{value["threshold"]:+.2f}',
                             f'{value["train_FP_error"]:+,}',
                             *[f'{value["val"][k]:.4f}%' for k in
                               ('mIoU', 'FN_rate', 'FPR')]])
    calibration = table(['组别', '训练选择的阈值', '训练 FP 匹配误差／体素',
                          '验证 mIoU', '验证漏检率', '验证假阳性率'], cal_rows)
    early_dose = opt['no_empty_center_boost']
    dose_table = table(['针对性控制所在阶段', '满足几何条件的 GS', '实际阻断增强的 GS',
                         '平均每帧实际阻断 GS 数'],
                        [[label, f'{100*d["mean_eligible_fraction"]:.4f}%',
                          f'{100*d["mean_blocked_fraction"]:.4f}%',
                          f'{76800*d["mean_blocked_fraction"]:.1f}']
                         for label, d in [('早期共同起点', early_dose),
                                          ('成熟预训练权重', m['dose'])]])
    fp_removed = int(st['val']['FP'] - ct['val']['FP'])
    fn_added = int(ct['val']['FN'] - st['val']['FN'])
    narrative = f'''# 几何覆盖与语义幅度补偿：训练因果验证结果

> **本轮结论：有证据支持几何与幅度路径在训练中存在条件性替代；尚未证明这种替代有害，更未证明它主导最终的泛化错误。**
>
> 对实际成熟权重进行针对性干预后，验证 mIoU 为 **{ct['val']['mIoU']:.4f}%**，标准继续训练为 **{st['val']['mIoU']:.4f}%**。干预减少了假阳性，同时增加了漏检，没有得到稳定的 mIoU 收益或一致的几何覆盖改善。
>
> **执行已完成。** 新增 8 个检查点的全量评估；每个覆盖训练 700 场景／28,130 帧、验证 150 场景／6,019 帧，共 273,192 次 keyframe 评估。所有评价默认使用 camera mask，未筛选评估子集。

## 1. 这次实验究竟改变了什么

检验对象是 `opus_gaussian_8f_tiny.pth` 对应的 **8 帧 tiny、加性 softplus 渲染模型**。结论不自动延伸到后续概率分解渲染版本。

下文 GS 指一个 semantic Gaussian（语义高斯）。一个 GS 的非负类别证据可分为两部分：**总幅度 A**，以及各类别占总幅度的**份额 p**。再乘不透明度 α，得到证据系数 b。提高 b 可以增强这个 GS 在外围前景体素上的得分，却不需要移动中心；这在数学上可行，但不等于它在训练中必然有害。

本轮保留原来的 Occupancy CE＋Lovasz 和 OPUS Set loss，通过反向梯度干预区分路径。五组的推理流程相同，均不使用 GT 筛选 GS：

| 组别 | 训练时唯一的直接路径干预 |
|---|---|
| 标准训练 | 保留原有路径 |
| 阻断幅度 | 移除 Occupancy 对总语义幅度和不透明度的直接更新，保留类别份额与几何路径 |
| 阻断几何 | 移除 Occupancy 对中心、尺度和旋转的直接更新，保留 Set 的几何监督 |
| 同时阻断 | 同时移除以上两类直接路径 |
| 针对性控制 | 只阻断“中心在可见 empty、1σ 核心无可见前景，且当前梯度要求增强证据”的 GS 的幅度增强方向 |

针对性控制保留全部降低幅度的方向，避免把正常纠正错误前景证据的路径一起删除。**核心无前景不表示整个渲染范围无前景**：这些 GS 仍可能通过外围区域支持前景。

所有配对组共同固定 empty bias 与 empty scalar，并恢复相同的模型、AdamW 状态、数据顺序和随机数。这里的“阻断”约束直接反向路径，不保证实际输出 A 或几何完全不变；保留的类别份额路径、Set loss、共享特征和优化器均可能改变它们。公式与实现见[设计报告](gaussian_geometry_causal_design.md)。

## 2. 主要检验：成熟模型是否因此泛化得更好

这组直接从实际 30 epoch 权重出发，对比标准继续训练和针对性控制。每组训练 **1,024 个 microbatch／128 次 AdamW 更新**，累积 8，种子 932。学习率为原保存终止值的 10 倍，约 2 × 10⁻⁶；这是已登记的小步诊断，原优化器矩保留。

### 2.1 原生性能：没有稳定的 mIoU 收益

{mature_table}

原权重的参考值来自此前**同一 SHA256、同一预处理**的完整统计，仅复用原生混淆计数。标准继续训练保持了原来的验证表现，因此这组对比没有经历早期模型那样的明显欠拟合。

下表只比较干预与标准继续训练，正值表示指标增大；mIoU／IoU 越高越好，漏检／假阳性率越低越好。

{delta_table}

验证集上，针对性控制减少 **{fp_removed:,} 个假阳性体素**，同时增加 **{fn_added:,} 个漏检体素**。mIoU 和二值 IoU 的区间都包含零。因此当前证据更符合前景预测趋于保守后的误差权衡，不能写成“去掉有害补偿后泛化变好”。

分母固定为：验证 GT 前景 **121,325,057** 个体素，GT empty **414,904,679** 个体素。漏检率＝FN／GT 前景；假阳性率＝FP／GT empty；前景语义错分率只统计 GT 前景被预测成另一前景类别，并以 GT 前景为分母。它们不能直接相加解释 17 类 mIoU。

若改用 GT 前景作分母，FP 占比为 **{st['val']['FP_over_GTfg']:.4f}% → {ct['val']['FP_over_GTfg']:.4f}%**；若改用全部预测前景作分母，则为 **{st['val']['FP_over_predfg']:.4f}% → {ct['val']['FP_over_predfg']:.4f}%**。箭头均表示标准组到针对性控制组。

### 2.2 几何变化：没有出现一致的覆盖改善

{geometry}

2σ 平均覆盖数仅增加 **{ct['val']['n_2sigma']-st['val']['n_2sigma']:.5f}**，相对于约 19.45 个的原有水平很小。没有 1σ 覆盖的比例略降，但没有 2σ 覆盖的比例反而上升，1σ 平均覆盖数也未提高；中心落在可见 empty 的 GS 比例没有减少。

总幅度与证据系数小幅下降，这说明干预确实改变了输出；但没有形成“减少幅度依赖 → 几何更好 → 漏检和误检同时下降”的完整证据链。

完整训练集加权 CE 从 **{st['train']['CE_weighted']:.6f}** 变为 **{ct['train']['CE_weighted']:.6f}**；验证二值 AP 从 **{st['val']['binary_AP_hist']:.4f}%** 变为 **{ct['val']['binary_AP_hist']:.4f}%**，区分能力的变化同样很小。CE 只衡量 Occupancy 的 CE 拟合，不是包含 Lovasz 和 Set 的总训练目标；AP 为直方图近似。

![成熟权重的原生性能、误差权衡与几何覆盖](figures/gaussian_geometry_causal/mature_effects.png)

图中置信区间来自同场景配对 bootstrap。完整导出：[PDF](figures/gaussian_geometry_causal/mature_effects.pdf)。

## 3. 辅助检验：早期训练是否使用了替代路径

早期共同起点由图像预训练权重建立，训练 2,048 个 microbatch／256 次更新。五个分叉再各训练 1,024 个 microbatch／128 次更新，种子 928。**共同起点验证 mIoU 只有 4.37%，分叉标准组也只有 6 个类别出现正确预测。** 这部分用于分析训练路径，不能代替成熟模型的泛化检验。

### 3.1 全局阻断幅度会破坏前景拟合

{early}

阻断幅度后，假阳性率下降，但漏检率由 **64.274%** 升至 **83.423%**，训练 CE 也明显变差。这证明原训练使用了该路径进行前景拟合，**不能证明被移除的路径本身有害**。

### 3.2 几何路径的收益取决于幅度路径是否存在

{conditional}

幅度可以更新时，移除 Occupancy 几何路径对 mIoU 的影响很小；幅度路径被阻断后，保留几何路径的收益明显增大。后者的收益比前者多 **{-f['factorial_interaction']['estimate']:.4f} 个百分点**，95% 场景区间为 **{interval([-v for v in f['factorial_interaction']['scene_bootstrap_95'][::-1]])}**。

这是本轮对**条件性路径替代**最直接的训练证据。但几何仍受 Set loss 和共享特征影响，所以它不是“实际几何冻结、仅靠语义也能完全学好”的实验，更没有单独证明优化器普遍偏爱语义捷径。

### 3.3 覆盖数增加，也可能伴随覆盖空洞增加

{early_geometry}

阻断幅度组的平均覆盖数更高，但无 1σ、无 2σ 覆盖的比例也更高。它把更多 GS 集中到部分前景体素附近，却没有均匀补足缺失区域；因此不能只看覆盖均值就宣布几何变好。类别份额更加集中，也不等于更多前景被正确检出。

早期针对性控制的验证 mIoU 由 5.9948% 升至 6.0740%，但假阳性率同时升高。这个小收益没有在成熟配对中转化成稳定收益，不能作为方法有效的主要结论。

![早期四组与针对性控制的配对变化](figures/gaussian_geometry_causal/paired_effects.png)

此图描述明显欠拟合阶段的结果，不能与成熟模型性能混为一谈。[PDF](figures/gaussian_geometry_causal/paired_effects.pdf)。

## 4. 干预有多强，属性上限是否限制了检验

{dose_table}

百分比的分母均为每帧全部 **76,800 个预测 GS**，不包含额外追加的固定 empty Gaussian。条件筛选只用于训练梯度，不筛选评估场景、GT 体素或推理 GS。两阶段的干预前向损失差均严格为零；成熟两组均没有触发梯度裁剪。

命中比例说明这是一项局部干预，不能把接近零的 mIoU 变化当作所有补偿路径均无害的强证明。反过来，阻断了数百个 GS 的增强方向，也不能自动推出它们的几何位置将改善；本轮实际并未观察到这种系统改善。

尺度还有一个实际限制：共同起点已接近 0.4 米上限；成熟标准组验证集中，**{st['val']['GS_maxscale_ge_03999']:.2f}%** 的 GS 最大轴接近上限，**{st['val']['raw_largest_scale_axis_zero']:.2f}%** 的最大轴 sigmoid 局部导数严格为零。后者读取的是 sigmoid 前原始输出，并非用四舍五入后的尺寸推断。它限制了尺度自由度的解释，但不意味着位置完全不能优化。

![共同起点的语义幅度、不透明度与尺度轨迹](figures/gaussian_geometry_causal/warmup_attributes.png)

各点使用当时不同的训练样本，不是固定验证样本的学习曲线。[PDF](figures/gaussian_geometry_causal/warmup_attributes.pdf)。

## 5. 阈值诊断：仅用训练集匹配 FP 预算

额外用完整训练集选择“最高前景得分减 empty 得分”的阈值，使训练 FP 数接近相应标准组；随后不作调整地应用到完整验证集。阈值分箱为 0.02 logit，匹配存在误差，**不保证验证 FP 相同**。

{calibration}

成熟控制组只需采用训练集选择的 **{m['calibration'][control]['threshold']:+.2f}** 阈值，验证 mIoU 就变为 **{m['calibration'][control]['val']['mIoU']:.4f}%**，与标准组 **{st['val']['mIoU']:.4f}%** 接近，漏检和假阳性率也接近标准组。这与本轮主要改变前景决策强弱、没有带来明显空间区分收益的解释一致。

这项诊断用于区分决策阈值与表示能力，不替代前面的原生结果，也不是验证集调参。它不能凭阈值移动证明 GS 几何得到改善，也不是完整的因果中介分解。

## 6. 对原机制命题应作何判断

| 命题 | 本轮判断 | 依据与范围 |
|---|---|---|
| 同一 GS 可以通过增强证据提高外围前景得分，而不移动中心 | 数学上成立 | 加性渲染的直接性质；不等于有害训练成因已证实 |
| 模型训练使用幅度路径进行前景拟合 | 支持 | 全局阻断幅度后，训练拟合和前景召回明显下降 |
| 幅度与几何路径存在条件性替代 | 在早期轨迹中支持 | 四组配对的条件收益差及场景区间 |
| 减少该增强路径，会迫使成熟模型补足几何覆盖 | 未得到支持 | 核心覆盖变化很小且不一致，empty 中心 GS 比例未下降 |
| 该补偿是最终泛化差距或大部分 FP／FN 的主导成因 | 尚不能成立 | 成熟配对没有稳定性能收益；短程局部干预无法回溯原训练历史 |

目前适合采用的机制表述是：**当前加性模型的前景拟合同时依赖几何支持和证据幅度，两条训练路径存在一定替代性；但“这种替代阻碍几何优化并主导泛化下降”的因果后半段尚未闭环。**

这不否定此前完整数据中“漏检体素核心支持较弱、部分 empty 中心 GS 贡献 FP”的观察。它说明从这些现象走向训练成因，还缺少能实际改善几何这一中间环节的干预。下一步更有辨别力的是先检验：在保持语义证据可比时，直接改善几何支持能否同时减少 FP 与 FN，再检验训练目标能否可靠实现这种改变。

## 7. 数据完整性、适用范围与复核入口

- **评估覆盖完整。** 每个新增检查点均遍历 850 场景／34,149 帧；帧索引、GT 类别分母一致。没有混入旧的中断采集或两个固定帧的实现检查。
- **先合并计数，再算全局指标。** 全局 mIoU 从整个 split 的混淆矩阵计算，不取场景 mIoU 的平均。每场景 CSV 的 mIoU 固定平均 17 类，单场景零并集类别记为 0，供同场景模型间对照。
- **配对条件一致。** 早期五组及成熟两组的完整 1,024 步数据指纹和起始输出核对通过；empty 参数在各自配对中保持一致。
- **干预实现正确。** 恒等前向、独立反向公式、真实渲染器梯度及 1σ 核心搜索均通过独立检查。生产模型和损失代码未因本轮实验改写。
- **因果范围有限。** 每阶段只有一个训练种子；1,000 次场景 bootstrap 表示固定模型下的场景变化，不能代替多种子训练。每个分叉只训练有限步数，并未重新训练一整轮或 30 epoch。
- **几何指标口径明确。** 覆盖数按固定 GT 前景体素加权；1σ／2σ 是旋转椭球范围，原生渲染候选范围是截断整数立方体。主要贡献者身份受语义权重影响，其距离不能单独作为纯几何证据。

| 文件 | 用途 |
|---|---|
| [实验设计](gaussian_geometry_causal_design.md) | 符号定义、梯度公式、预算与登记修订 |
| [成熟模型汇总](../work_dirs/gaussian_geometry_causal/mature_summary.json) | 主要性能、场景区间、干预命中及阈值诊断 |
| [早期模型汇总](../work_dirs/gaussian_geometry_causal/summary.json) | 五组与共同起点的完整全局指标 |
| [条件性路径比较](../work_dirs/gaussian_geometry_causal/factorial_interaction.json) | 四组条件收益与配对区间 |
| [成熟每场景 CSV](../work_dirs/gaussian_geometry_causal/mature_scene_metrics.csv)／[早期每场景 CSV](../work_dirs/gaussian_geometry_causal/scene_metrics.csv) | 每个场景的独立结果 |
| [成熟每类别 CSV](../work_dirs/gaussian_geometry_causal/mature_class_metrics.csv)／[早期每类别 CSV](../work_dirs/gaussian_geometry_causal/class_metrics.csv) | 每类 IoU、漏检、误占用及语义混淆计数 |
| [最终完整性审计](../work_dirs/gaussian_geometry_causal/final_integrity_audit.json) | 8 个检查点、GS 数、索引、分母、参数与来源核验 |
| [复现环境](../work_dirs/gaussian_geometry_causal/execution_environment.json)／[文件校验清单](../work_dirs/gaussian_geometry_causal/final_artifact_manifest.json) | 环境版本、源文件快照及 SHA256 |

预留的“仅阻断语义幅度、允许 opacity 替代”组没有执行，不列为已完成实验。本轮没有拆分语义幅度与 opacity 的独立最终因果贡献，也没有重新判定 CE、Lovasz 和 Set 各自的历史责任。GS 数量始终固定，未通过增删或重新分配 GS 检验“预算挤占造成漏检”的解释。
'''
    automatic = RUN / 'results_automatic_tables.md'
    if not automatic.exists():
        automatic.write_text(REPORT.read_text())
    REPORT.write_text(narrative)
    print(f'Wrote reviewed report: {REPORT}')


if __name__ == '__main__':
    main()
