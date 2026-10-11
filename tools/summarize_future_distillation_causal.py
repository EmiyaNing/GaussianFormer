#!/usr/bin/env python3
"""Generate stage/hypothesis reports from observed results, never inferred scores."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.analyze_future2current_regions import grouped_metrics

REPO=Path(__file__).resolve().parents[1]
DOCUMENTS=REPO/'analysis/hypotheses/future_distillation_causes'
HORIZONS=('1s','2s','3s')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path,lines):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n',encoding='utf8');temporary.replace(path)


def average(result,metric='mIoU'):
    return float(np.mean([result['metrics'][h][metric] for h in HORIZONS]))


def teacher_subset(root,count):
    reference=REPO/'out/future2current/predictable_full_finetune_4096x1024'
    selected=read(root/'manifest.json');full=read(reference/'manifest.json')
    if not selected or not full:return None
    for key in ('indices','scenes'):
        if selected['val'][key][:count]!=full['val'][key][:count]:raise RuntimeError('Teacher report sample identities differ')
    if selected['checkpoint_sha256']!=full['checkpoint_sha256']:raise RuntimeError('Teacher report checkpoint differs')
    with np.load(reference/'teacher_native.npz') as saved:
        hist=saved['hist'][:count].sum(0);intersection=saved['intersection'][:count].sum(0);union=saved['union'][:count].sum(0)
    diagonal=np.diagonal(hist,axis1=-2,axis2=-1);denominator=hist.sum(-1)+hist.sum(-2)-diagonal
    with np.errstate(divide='ignore',invalid='ignore'):iou=100*diagonal/denominator
    groups=grouped_metrics(hist)
    return dict(metrics={h:dict(mIoU=float(np.nanmean(iou[i,:17])),OccIoU=float(100*intersection[i]/max(union[i],1))) for i,h in enumerate(HORIZONS)},
                groups={h:{g:dict(mIoU=float(groups[g]['macro_iou'][i]),IoU=float(groups[g]['group_iou'][i])) for g in ('dynamic','static')} for i,h in enumerate(HORIZONS)})


def table(records,base,metric='mIoU'):
    lines=['| 模型／干预 | 1 秒 | 2 秒 | 3 秒 |','|---|---:|---:|---:|']
    for label,row in records:
        cells=[]
        for h in HORIZONS:
            value=row['metrics'][h][metric];reference=base['metrics'][h][metric]
            cells.append(f'{value:.3f}（参照）' if label=='原版 SparseWorld' else f'{value:.3f}（{value-reference:+.3f}）')
        lines.append('| '+' | '.join([label,*cells])+' |')
    return lines


def native_tables(records,base):
    lines=['', '单位为 %，括号为相对本表真实基线的百分点变化。','', '### 总体 mIoU','',*table(records,base),
           '', '### 总体 OccIoU','',*table(records,base,'OccIoU'),
           '', '<details>', '<summary>展开动态／静态类别组的 mIoU 与 IoU</summary>', '']
    for group,label in [('dynamic','动态'),('static','静态')]:
        for metric in ('mIoU','IoU'):
            lines.extend(['',f'### {label}类别组 {metric}','',
                          '| 模型／干预 | 1 秒 | 2 秒 | 3 秒 |','|---|---:|---:|---:|'])
            for name,row in records:
                cells=[]
                for h in HORIZONS:
                    value=row['groups'][h][group][metric];reference=base['groups'][h][group][metric]
                    cells.append(f'{value:.3f}（参照）' if name=='原版 SparseWorld' else f'{value:.3f}（{value-reference:+.3f}）')
                lines.append('| '+' | '.join([name,*cells])+' |')
    lines.extend(['','</details>'])
    return lines


def arm_means(screen):
    arms={}
    for name,row in screen.get('arms',{}).items():
        if row.get('status')=='complete':arms.setdefault(name.split('_seed')[0],[]).append(row)
    means={}
    for name,runs in arms.items():
        result={'metrics':{h:{m:float(np.mean([r['metrics'][h][m] for r in runs])) for m in ('mIoU','OccIoU')} for h in HORIZONS},
                'groups':{h:{g:{m:float(np.mean([r['groups'][h][g][m] for r in runs])) for m in ('mIoU','IoU')} for g in ('dynamic','static')} for h in HORIZONS}}
        result['seed_count']=len(runs)
        result['feature_error']=float(np.mean([r['feature_validation_loss'] for r in runs]))
        means[name]=result
    return means


def s0_metric_reading(samples=64):
    """Shared metric and reference definitions for the two S0 reports."""
    return [
        '', '### 表中的指标与参照怎样读？', '',
        '- **总体 mIoU**：17 个非空语义类别的平均 IoU；**OccIoU**：把非空类别合并后计算空／非空占据的 IoU。'
        '两者回答不同问题，不应把一个指标的改善写成所有方面都恢复。',
        '- **动态／静态 mIoU**：各类别组内部的平均类别 IoU；**类别组 IoU**：先合并组内类别再计算。'
        '例如把汽车错认成卡车仍属于动态组，会损害语义 mIoU，却未必损害动态组 IoU。'
        '这不是只裁剪 GT 动态区域后计算整体成绩；分组按语义类别，不按实际运动速度。',
        '- **1／2／3 秒**：第 2／4／6 个未来关键帧；三时距平均是三个 mIoU 的算术平均。'
        f'本表使用相同 {samples} 验证样本，不与其他样本清单的绝对分数混算差值。',
        '- **括号中的 ±**：只相对本表未微调的原版 SparseWorld，单位为百分点。'
        '要判断教师净作用，另比较同预算学生组的原始分数，而非仅看是否超过预训练基线。',
        '- **真实未来感知教师**：在目标时刻读取真实图像，用原版感知头识别场景；'
        '该行不是只看历史的未来预测成绩，也不是学生可达到的数学上界。', '',
        '动态类别包括自行车、公交车、汽车、工程车辆、摩托车、行人、拖车、卡车；'
        '静态类别包括 barrier、traffic cone、可行驶地面、其他平面、人行道、地形、建筑、植被。'
        '`others` 不归入这两个组，空类别不进入总体非空 mIoU。',
    ]


def s0_case_explanation(samples=64):
    """Persist checkpoint-intervention definitions in regenerated S0 reports."""
    return [
        '', '## 1．先分清：这里没有重新训练', '',
        '本轮取既有 4096 训练／1024 验证、5 epoch 全参数微调的仅任务检查点，'
        f'在其中相同的 {samples} 个验证样本上复评。{samples} 是本轮诊断样本数，不是原训练集大小。'
        '每个干预都独立从同一末次检查点出发，只在内存中恢复指定模块，不更新优化器，也不覆盖磁盘权重。', '',
        '| 结果中的名称 | 相对仅任务末次检查点改变什么？ | 不是在做什么？ |',
        '|---|---|---|',
        '| `task_epoch5` | 什么也不改，直接评估仅任务第 5 epoch 检查点 | 不是另训一个 64 样本模型 |',
        '| `reset_bn` | 恢复预训练 BN 运行均值、方差、累计批次数；保留训练后的可学习参数 | 不是重置 BN 缩放／偏移或重新训练 |',
        '| `disable_adapter` | 临时将新增残差 MLP 换成恒等映射 | 不是其他权重也回到预训练状态 |',
        '| `reset_heads` | 恢复未来 `semantic`／`reg`／`velocity` 分支 | 不包括当前 OPUS 感知头或轨迹头 |',
        '| `reset_perception` | 恢复 backbone、neck、lifter、encoder、plan_head、points_scale 的参数与缓冲状态 | 不是只恢复图像 backbone，也不是只恢复 BN |',
        '| `reset_bn_disable_adapter` | 同时恢复 BN 状态、将 MLP 换成恒等映射 | 不是在前面其他恢复组上继续叠加干预 |', '',
        '原版基线是未微调预训练 SparseWorld；实验类中 MLP 的初始输出为零，前向与原版一致。'
        '正常学生只看当前／历史图像，保留 future-born queries，最终输出由原版未来头及体素化产生。'
        '恢复后的模型混合了预训练和微调状态，因此某组不恢复成绩也不能证明该模块没有问题，'
        '还可能是新旧模块缺少共同适配。',
    ]


def r1_condition_explanation():
    """Keep experiment-name definitions in every regenerated R1 report."""
    return [
        '', '## 0．先读懂六组名称：三种输入条件 × 两种预测方式', '',
        '这六组都是额外训练的小型特征预测模型。SparseWorld 和教师保持冻结；'
        '小模型不更新或替代 Student，不经过 occupancy 预测头，也不是之前的共享语义读出器。'
        '共同任务是：根据指定输入，预测目标未来时刻按空间近邻合成的教师特征。', '',
        '| 名称 | 小模型能看到什么 | 预测内容／训练目标 |',
        '|---|---|---|',
        '| `position_time_deterministic` | 学生预测的 query 中心位置＋目标时距 | 一个 256 维特征向量；用 MSE 训练 |',
        '| `position_time_gaussian` | 同上 | 特征均值＋各维度方差；用高斯 NLL 训练 |',
        '| `history_deterministic` | 位置＋时距＋学生从当前／历史输入预测出的未来 query 特征 | 一个 256 维特征向量；用 MSE 训练 |',
        '| `history_gaussian` | 同上 | 特征均值＋各维度方差；用高斯 NLL 训练 |',
        '| `privileged_ego_deterministic` | history 组的全部输入＋真实未来自车轨迹 | 一个 256 维特征向量；用 MSE 训练 |',
        '| `privileged_ego_gaussian` | 同上 | 特征均值＋各维度方差；用高斯 NLL 训练 |', '',
        '### 输入词语的准确含义', '',
        '- **位置**：该学生未来 query 预测的 48 个点的算术平均中心，不是真值位置。'
        '它来自冻结 SparseWorld 的历史预测，因此 position_time 组仍含历史的派生信息，不是完全无历史的空间先验。',
        '- **目标时距**：需要预测教师特征的未来时刻，1／2／3 秒对应第 2／4／6 个未来关键帧。',
        '- **history 特征**：冻结 SparseWorld 从当前／历史输入完成未来推演后得到的 query 特征，'
        '已经融合自车上下文、位置／时间等信息；不是原始历史图像，也不是仅当前感知特征。',
        '- **privileged ego**：未来六步真实自车二维轨迹，共 12 个数；不是只提供目标时刻一个位置。'
        '它是特权 GT，不是部署时已知的行为计划，也不告诉模型其他交通参与者的真实未来。',
        '- **教师目标**：在同一未来 ego 坐标系内，按学生 query 中心合成的四近邻教师特征，'
        '再作 LayerNorm 归一化；不是按 query 编号直接取教师向量。', '',
        '所有组使用同一份 query 清单和相同小模型结构。输入统一为 272 维：'
        '256 维学生特征＋3 维中心＋1 维时距＋12 维真实未来自车轨迹；'
        '不允许使用的通道置零，而不是给某一组额外扩大模型容量。', '',
        '### deterministic 与 gaussian 的区别', '',
        '`deterministic` 用 MSE 训练一个预测向量，不学习每个 query 的条件方差。'
        '为报告 NLL，另外按训练集残差估计每个特征维度的固定方差；'
        '这一方差对所有 query 共用，不用验证数据拟合。', '',
        '`gaussian` 同时预测 256 维均值和 256 维方差，用高斯负对数似然 NLL 训练。'
        '“对角 Gaussian”表示各维度分别预测方差，不预测维度之间的协方差。'
        '报告中的 MSE 和余弦误差只比较预测均值与教师目标，NLL 同时评价均值和方差。', '',
        '确定性预测不证明真实未来只有一种；Gaussian 的方差也不证明存在历史无法消除的不确定性。'
        '这里预测的是教师特征分布，不是多条真实未来轨迹或 occupancy 分布。', '',
        '### 三类比较分别回答什么？', '',
        '| 对照 | 想检验的问题 | 不能直接推出什么 |',
        '|---|---|---|',
        '| position_time → history | 学生已有的历史预测特征，是否提供位置／时距之外的教师信息？ | 特征更可预测不等于原版 occupancy 更准 |',
        '| history → privileged_ego | 额外知道真实未来自车轨迹，是否帮助拟合教师特征？ | 不能当成可部署行为计划的收益，也不保证其他参与者未来唯一 |',
        '| deterministic → gaussian | 同一输入条件下，学习均值／方差是否更适合描述教师特征目标？ | NLL 改善不证明概率蒸馏能提高 occupancy |', '',
        '代码位置：[`experiment_future_distillation_conditions.py`](../../../../tools/experiment_future_distillation_conditions.py)，'
        '其中 `feature_dataset` 构造输入／教师目标，`ConditionalFeatureModel` 预测均值／方差，'
        '`main` 按条件屏蔽输入通道并选择 MSE 或 NLL。',
    ]


def r2_intervention_explanation():
    """Explain mixing and distinguish hypothesis, experiment, and stage IDs."""
    return [
        '', '## 0．先读懂 mix 名称与 R／T／S 编号', '',
        'T1 中的十组 mix 是冻结模型下的即时推理干预，不是重新训练的十个模型。'
        '它们临时给学生未来预测头加入真实未来教师特征，以检查接口能否利用这份信息；'
        '不能当成仅看历史的学生已经学会了这些收益。T2／T3 则另有训练筛选。', '',
        '### mix 数字：混合的是特征向量，不是 query 数量', '',
        '`混合特征 = (1−α) × 学生特征 + α × 教师目标特征`', '',
        '| 两种解码下的组名 | 学生特征系数 | 教师特征系数 |',
        '|---|---:|---:|',
        '| `fixed_mix0`／`native_joint_mix0` | 100% | 0% |',
        '| `fixed_mix0.1`／`native_joint_mix0.1` | 90% | 10% |',
        '| `fixed_mix0.25`／`native_joint_mix0.25` | 75% | 25% |',
        '| `fixed_mix0.5`／`native_joint_mix0.5` | 50% | 50% |',
        '| `fixed_mix1`／`native_joint_mix1` | 0% | 100% |', '',
        '该系数用于每个有效匹配 query，不表示只替换相应百分比的 query。'
        '教师目标在同一未来 ego 中，以学生 query 的 48 个预测点的算术平均中心为基准，'
        '对四个近邻教师 query 特征逆距离加权合成；最近中心距离不超过 4 米才干预。'
        '无效匹配的 query 保持原特征，future-born queries 保留。'
        '`mix1` 只完全替换有效 query 的特征，不复制整份教师 query、教师点位置或教师 occupancy。', '',
        '### fixed 与 native_joint：点位置是否重新预测？', '',
        '| 前缀 | 混合后如何解码 | 想回答什么 |',
        '|---|---|---|',
        '| `fixed_mix…` | 用学生原版语义头重新预测类别，保留学生原先预测的未来点位置 | 排除位置变化后，教师特征能否改善类别判断？ |',
        '| `native_joint_mix…` | 从相同的学生交互位置出发，用原版语义／位置／速度分支重新解码，类别和点位置都允许变化 | 教师特征能否被完整未来预测头正确利用？ |', '',
        '例如，`fixed_mix0.5` 和 `native_joint_mix0.5` 的混合向量相同；'
        '前者只采用重新解码的类别，后者同时采用重新解码的类别和点位置。'
        '`native` 指使用学生原版预测分支，`joint` 指联合解码语义和几何，不是联合训练。'
        '联合版本也不是把学生点与教师点按比例插值。各目标时距独立干预，混合后的状态不反馈给下一时距。', '',
        '### R、T、S 是三套不同的编号', '',
        '| 编号 | 表示什么 | 本文中如何使用 |',
        '|---|---|---|',
        '| R1／R2／R3 | 研究假设 | R2 检查特征目标与任务是否兼容；R3 检查教师与任务的优化强度／方向是否冲突 |',
        '| T1／T2／T3 | R2 内部的具体实验 | T1 直接混合特征；T2 检验预测头能否适配；T3 使用教师原生语义输出监督 |',
        '| S0／S1／S2 等 | 整个方案的执行阶段 | T1 在 S1 做即时干预；T2／T3 在 S2 做训练筛选 |', '',
        '**T2 中的 S2**是第二阶段训练筛选，不是 2 秒或模型某一层：'
        '256 个训练样本、128 个验证样本、512 次更新、三个种子，冻结历史感知编码。'
        'T2 比较原版未来语义／位置／速度头可训练或冻结，以及仅任务或任务＋完整特征监督；'
        '每一列内部保持相同可训练参数和预算，不能把跨列差别全部归给教师。', '',
        '**T3 中的 R3**是优化冲突假设，不是第三阶段或 3 秒。'
        '相同辅助损失系数不保证相同梯度强度，所以特征监督与语义输出监督的比较，'
        '要结合 R3 的减弱强度、去除冲突方向等对照；不能把优化强度差别全部解释为目标类型优劣。', '',
        '实现见[`特征干预代码`](../../../../tools/experiment_future_distillation_mechanisms.py)的 `feature_intervention`；'
        '阶段预算见[统一协议](../stage_protocol.md)，优化对照见[R3 设计](../r3_gradient_conflict/design.md)。',
    ]


def update_reports(root,documents=DOCUMENTS):
    root,documents=Path(root),Path(documents)
    s0=read(root/'s0_training_audit.json');mix=read(root/'s1_feature_intervention.json')
    grads=read(root/'s1_gradients.json');screen=read(root/'s2_screens.json');conditions=read(root/'s4_conditions.json')
    pipeline=read(root/'pipeline_state.json') or {}
    protocol_root=REPO/'out/future_distillation_causal/20261001_training_protocol'
    protocol=read(protocol_root/'results.json') if root.resolve()==(REPO/'out/future_distillation_causal/20260930').resolve() else None
    alignment=read(root/'s2_alignment_audit.json')
    bootstrap=read(root/'s1_intervention_bootstrap.json');geometry=read(root/'s1_geometry.json')
    screen_boot=read(root/'s2_bootstrap.json')
    screen_status=screen['status'] if screen else ('正在准备教师原版 occupancy 输出' if pipeline.get('phase')=='S2' else '待执行')
    levels={'R1':'未验证核心不可唯一确定命题','R2':'尚无正式结果','R3':'尚无正式结果'}
    lines=['# S0 实际结果：共同训练退化','',f'状态：{s0["status"] if s0 else "待执行"}。所有结果来自原版任务头，不使用探针。']
    if s0 and all(name in s0['cases'] for name in ('baseline','task_epoch5','reset_perception')):
        base,task,reset=[s0['cases'][name] for name in ('baseline','task_epoch5','reset_perception')]
        lines.extend(['','## 一页结论','',
                      f'三时距平均 mIoU：原版基线 {average(base):.3f}，仅任务 {average(task):.3f}，'
                      f'恢复感知组合 {average(reset):.3f}。', '',
                      '没有教师的微调已经下降，不能把全部降点归给教师。恢复上游组合能改善该检查点，'
                      '但这一操作同时改变多个模块，只能定位组合影响，不能直接认定 BN 或某一损失是唯一原因。'])
    samples=len(s0['validation']['indices']) if s0 else 64
    lines.extend(s0_case_explanation(samples))
    lines.extend(s0_metric_reading(samples))
    if s0:
        base=s0['cases']['baseline'];records=[('原版 SparseWorld',base)]
        teacher=teacher_subset(root,len(s0['validation']['indices']))
        if teacher:records.append(('真实未来感知教师',teacher))
        records.extend((name,row) for name,row in s0['cases'].items() if name!='baseline')
        lines.extend(['','## 2．原版头的实际结果','',f'验证样本：{len(s0["validation"]["indices"])}；所有组是相同样本。',*native_tables(records,base)])
        if 'task_epoch5' in s0['cases'] and 'reset_perception' in s0['cases']:
            task=s0['cases']['task_epoch5'];reset=s0['cases']['reset_perception'];loss=average(base)-average(task)
            ratio=(average(reset)-average(task))/loss if loss>.2 else None
            lines.extend(['','## 3．恢复比例与结论边界','',
                          f'原版三时距平均 mIoU {average(base):.3f}，仅任务组 {average(task):.3f}，恢复感知部分 {average(reset):.3f}。',
                          f'感知恢复追回基线到仅任务组下降的 {ratio*100:.1f}%。' if ratio is not None else '损失不足以稳定计算恢复比例。',
                          '', '`恢复比例 = (干预后平均 mIoU − 仅任务平均 mIoU) / (基线平均 mIoU − 仅任务平均 mIoU)`', '',
                          f'这份比例只描述本检查点、这 {samples} 个验证样本的损失追回量，不是参数责任占比，'
                          '也不是续实验从相同初始化重新训练后的 BN 恢复比例。', '',
                          '这是验证时权重干预，定位感知部分改变与共同退化的关系；不是教师效应，也没有证明某个损失是唯一原因。'
                          '本轮全参数实验的严重下降不能全部归因于教师。'])
            lines.extend(['','### 分开读各指标，避免把“恢复”说成全面恢复','',
                          '恢复感知组合改善了该末次模型的总体语义表现，但三个时距仍低于原版基线；'
                          '它不是已经恢复原版性能的完成证明。', '',
                          '恢复 BN 后总体 mIoU 改善，不等于所有类别与所有 IoU 都改善；例如动态类别组的合并 IoU '
                          '并未随之整体上升。请对照同一指标的原始分数，不把语义分类改善混称为几何占据全面恢复。', '',
                          '关闭 MLP、恢复未来头没有出现与恢复感知组合同等量级的改善；'
                          '这只评价末次检查点中的即时干预，不证明无 MLP 训练或其他解码器适配一定无用。', '',
                          '下一步见[训练协议续实验](../s0_training_protocol/results.md)：它从相同初始化重新训练，'
                          '拆开 BN 模式、任务组合、几何梯度，再复验教师净作用。两份报告的恢复对象和预算不同，不混用恢复比例。'])
    lines.extend(['',f'原始数据：`{root/"s0_training_audit.json"}`。设计见[设计文档](design.md)。'])
    write(documents/'s0_protocol/results.md',lines)
    r2=['# R2 实际结果：特征目标与任务兼容','',f'T1 状态：{mix["status"] if mix else "待执行"}。T2／T3 训练筛选状态：{screen_status}。']
    r2.extend(r2_intervention_explanation())
    if mix:
        base=mix['baseline'];records=[('原版 SparseWorld',base)]
        teacher=teacher_subset(root,len(mix['validation']['indices']))
        if teacher:records.append(('真实未来感知教师',teacher))
        records.extend(mix['cases'].items())
        r2.extend(['','## 先读结论','',
                   '这里的“教师特征”是同一未来 ego 坐标系中，按学生 query 的 48 点平均中心、对四个近邻教师 query 逆距离加权得到的目标。'
                   '完整混合只指有效匹配 query 的混合比例为 1；不是按编号替换整份教师 query，也不是严格的物体身份匹配。',
                   '完整替换后的特征距离为零，但原版头并未因此更准；联合几何尤其退化。'
                   '同时，中等比例的固定几何混合在远时距存在正结果，因此不能说所有拉近都无用。'
                   '这里检验的是未来特权特征的即时利用，不是学生从历史学会了这些收益。',
                   '同架构不等于同一读出：Teacher 感知输出依赖 OPUS encoder 的分类／点细化分支，'
                   'Student 未来输出依赖额外语义／位置／速度分支。此结果支持当前接口不能直接互换，'
                   '不证明预测器与感知器的表示在理论上必须不同。'])
        r2.extend(['','## 1．T1：原版头即时干预',*native_tables(records,base),'','### 实际特征距离','',
                   '| 干预 | 师生余弦误差 |','|---|---:|'])
        for name,row in mix['cases'].items():r2.append(f'| {name} | {row["feature_cosine_error"]:.6f} |')
        match=mix['matching'];coverage=sum(r['valid'] for r in match)/max(sum(r['queries'] for r in match),1)
        r2.extend(['',f'空间监督覆盖 {coverage*100:.2f}%；最近教师中心平均距离 {np.mean([r["mean_nearest_m"] for r in match]):.3f} 米。',
                   '身份初始化：六个时刻的原版点／logits 一致性检查通过。无效 query 不替换，future-born queries 保留。'])
        full=mix['cases']['fixed_mix1']
        if average(full)<average(base) and full['feature_cosine_error']<mix['cases']['fixed_mix0']['feature_cosine_error']:
            levels['R2']='Ⅱ（局部原版头特征干预）；长期适配主因尚未确认'
            r2.extend(['','### 判读','',
                       '完整混合使匹配处特征更接近教师，但固定几何的原版任务表现更差，支持当前接口下“更像教师不等于更准”。'
                       '证据等级为Ⅱ：直接改变特征后的局部干预；它不证明长期原版头适配也必然失败，且仍受近邻对应精度影响。'])
        half=mix['cases'].get('fixed_mix0.5')
        if half:
            r2.extend(['',f'必须同时保留局部正结果：固定几何、50% 混合的三时距平均 mIoU 为 {average(half):.3f}，'
                       f'原版为 {average(base):.3f}。其 2／3 秒原始分数为 '
                       f'{half["metrics"]["2s"]["mIoU"]:.3f}／{half["metrics"]["3s"]["mIoU"]:.3f}。'
                       '不能把完整替换的失败改写成“所有特征拉近都无用”；强度、几何和时距存在差异。'
                       '这是未来特权特征的即时干预，不是只看历史的学生已经学会获得该收益。'])
        if geometry:
            r2.extend(['','### 几何接口发生了什么？','',
                       '| 联合解码混合比例 | 1 秒平均点位移（米） | 2 秒 | 3 秒 |','|---|---:|---:|---:|'])
            for alpha in ('0','0.1','0.25','0.5','1'):
                values=[geometry['cases'][f'mix{alpha}/{h}']['mean_shift_m'] for h in HORIZONS]
                r2.append('| '+alpha+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
            r2.extend(['','点位移比较同一学生、同一交互位置上的两种原版分支解码结果，仅统计有效匹配 query。'
                       '完整替换可导致数米位移，说明仅让特征靠近教师并没有保持学生几何解码意义。'
                       '这不是坐标系变换量，也不是与教师点云的距离；不能把它解释为物体真实移动。'])
        if bootstrap:
            r2.extend(['','### 配对场景重采样区间','',
                       '下表是相对真实基线的 mIoU 变化的 95% 区间（百分点）。同一批 64 场景、1000 次配对重采样；'
                       '未校正多重比较，不覆盖重训练随机性，也不属于独立确认。','',
                       '| 干预 | 1 秒区间 | 2 秒区间 | 3 秒区间 | 三时距平均区间 |',
                       '|---|---:|---:|---:|---:|'])
            for name,values in bootstrap['cases'].items():
                r2.append('| '+name+' | '+' | '.join(f'[{values[h][0]:+.3f}, {values[h][1]:+.3f}]' for h in (*HORIZONS,'mean_1_2_3s'))+' |')
            r2.extend(['','50% 固定几何混合的 3 秒区间支持局部正效应，但三时距平均区间仍跨零；'
                       '不能由点估计超过项目门槛就宣布整体稳定提升。'])
    means=arm_means(screen) if screen else {}
    if screen and 'baseline' in screen:
        screen_records=[('原版 SparseWorld',screen['baseline'])]
        teacher=teacher_subset(root,int(screen['baseline']['metrics']['1s']['num_samples']))
        if teacher:screen_records.append(('真实未来感知教师',teacher))
        screen_records.extend((f'{name}（{row["seed_count"]} 种子均值）',row) for name,row in means.items())
        r2.extend(['','## 2．T2／T3：原版头训练筛选','',
                   '256／128 是开发筛选，冻结历史编码；512 更新。不是全参数 4096／1024 确认。',
                   '组名说明：task＝仅任务；feature＝加完整特征监督；damped＝减弱教师强度；project＝去除冲突方向；'
                   'response＝加原生硬语义输出监督。heads_train／heads_frozen 分别表示原版未来语义／位置／速度头可训练／冻结。',
                   *native_tables(screen_records,screen['baseline']),
                   '','### 特征监督是否真正起作用','',
                   '移动匹配随训练后的点位置重新选教师目标；固定匹配始终使用未微调基线的目标与有效 query 清单。'
                   '后者避免用“换了目标”冒充“更接近同一目标”，但训练后的点仍可移动，不能称为完全相同物理位置采样。','',
                   '| 组 | 完成种子数 | 移动匹配特征损失 | 固定匹配特征损失 |','|---|---:|---:|---:|'])
        if alignment:
            r2.append(f'| 原版 SparseWorld | 不训练 | — | {alignment["baseline_fixed_feature_loss"]:.6f} |')
        for name,row in means.items():
            audited=[v['fixed_feature_loss'] for k,v in (alignment or {}).get('arms',{}).items() if k.startswith(name+'_seed')]
            fixed=f'{np.mean(audited):.6f}（{len(audited)} 种子）' if audited else '待核对'
            r2.append(f'| {name} | {row["seed_count"]} | {row["feature_error"]:.6f} | {fixed} |')
        r2.extend(['','### 三个种子分别是多少？','',
                   '本表为原始 mIoU，不增加相对其他组的 ± 列；可直接比较同一种子的配对组。'
                   '种子变化不替代独立验证场景，样本数量也不是 query 数量。','',
                   '| 组 | 种子 | 1 秒 | 2 秒 | 3 秒 | 三时距平均 |','|---|---:|---:|---:|---:|---:|'])
        for name,row in screen['arms'].items():
            if row.get('status')!='complete':continue
            values=[row['metrics'][h]['mIoU'] for h in HORIZONS]
            r2.append('| '+name.split('_seed')[0]+' | '+str(row['seed'])+' | '+' | '.join(f'{v:.3f}' for v in (*values,np.mean(values)))+' |')
        r2.extend(['','### 按配对组判读，而不是按有没有超过预训练基线判读','',
                   '| 组 | 三时距均值 | 跨种子标准差（样本标准差） |','|---|---:|---:|'])
        for name,row in means.items():
            values=[average(v) for k,v in screen['arms'].items() if k.startswith(name+'_seed') and v.get('status')=='complete']
            sd=f'{np.std(values,ddof=1):.4f}' if len(values)>1 else '种子不足'
            r2.append(f'| {name} | {average(row):.4f} | {sd} |')
        for suffix,description in [('train','头参与训练'),('frozen','头冻结')]:
            task=means.get('task_heads_'+suffix);feature=means.get('feature_heads_'+suffix)
            if task and feature and task['seed_count']==feature['seed_count']==3:
                r2.append(f'\n{description}：仅任务 {average(task):.4f}，完整特征监督 {average(feature):.4f}。'
                          '差距应与种子波动和验证采样误差一起看，不能把很小的点估计写成严重损害或确定提升。')
        if screen_boot:
            r2.extend(['','### 场景与种子的联合重采样检查','',
                       '以下区间只相对原版 SparseWorld，单位为 mIoU 百分点。'
                       '同一开发集合、配对场景及种子重采样，1000 次；三个种子有限，区间未校正多重比较。'
                       '这不是全数据评估，也不是独立确认。','',
                       '| 组 | 1 秒 95% 区间 | 2 秒 | 3 秒 | 三时距平均 |','|---|---:|---:|---:|---:|'])
            for name,row in screen_boot['cases'].items():
                r2.append('| '+name+' | '+' | '.join(f'[{row["baseline_change_ci"][h][0]:+.3f}, {row["baseline_change_ci"][h][1]:+.3f}]' for h in (*HORIZONS,'mean_1_2_3s'))+' |')
            for name in ('feature_heads_train','feature_heads_frozen','response_heads_train'):
                if name not in screen_boot['cases']:continue
                low,high=screen_boot['cases'][name]['paired_task_change_ci']['mean_1_2_3s']
                verdict='区间跨零，尚不支持稳定净变化' if low<=0<=high else '有方向性证据，但仍须看实际幅度并独立确认'
                r2.append(f'\n另外对同参数、同预算任务组做配对诊断：{name} 的三时距平均净变化{verdict}。'
                          '该诊断的原始区间保存在 JSON，不与主表相对基线的差值混用。')
        r2.extend(['','“response” 是原生硬 occupancy 语义监督，不是软 KL 或位置蒸馏。以上各组的差值只对基线；'
                   '教师效应需比较 task 与 feature 的原始分数，不能由是否超过预训练基线单独判断。'
                   '输出监督和特征监督的系数虽同为 0.1，却不代表梯度强度相同；本轮不能排除强度差异。'])
    r2.extend(['',f'原始数据：`{root/"s1_feature_intervention.json"}`、`{root/"s2_screens.json"}`。设计见[设计文档](design.md)。'])
    if protocol:
        runs={r['supervision']:r['steps']['256'] for r in protocol.get('arms',{}).values()
              if r['arm']=='full_bn_fixed' and r.get('status')=='complete' and '256' in r.get('steps',{})}
        if all(k in runs for k in ('task','teacher','shuffled')):
            r2.extend(['','## 3．新增：固定 BN 后的全参数教师复验','',
                       f'同预算任务组的三时距平均 mIoU 为 {average(runs["task"]):.4f}，'
                       f'正确教师为 {average(runs["teacher"]):.4f}，打乱教师为 {average(runs["shuffled"]):.4f}。',
                       '128 训练／64 验证、256 更新，一个配对种子；Student 各组均只读当前／历史图像，'
                       '全部参数允许训练，保留原版输出分支和共同残差 MLP。'
                       '这一次先排除了本设置中 BN 更新造成的严重共同退化，再检查教师目标，不能与旧探针成绩混用。',
                       '是否确实更像同一教师目标、时距／类别组结果及配对区间见[续实验结果](../s0_training_protocol/results.md)。'
                       '极小的均值差别不证明长期适配一定失败，也不证明目标在理论上必须不同；R2 的原有局部干预证据等级不因单种子复验自动升级。'])
    write(documents/'r2_task_compatibility/results.md',r2)
    r3=['# R3 实际结果：梯度强度与方向','',f'完整模型观测状态：{grads["status"] if grads else "待执行"}；训练干预状态：{screen_status}。']
    if grads:
        rows=[s['total_groups']['shared_global'] for s in grads['samples']]
        r3.extend(['','## 先读结论','',f'总任务共有参数上，{sum(r["cosine"]<0 for r in rows)}/{len(rows)} 样本出现负余弦；'
                   f'平均夹角余弦 {np.mean([r["cosine"] for r in rows]):.5f}，加权教师／任务梯度比平均 '
                   f'{np.mean([r["weighted_teacher_to_task_norm"] for r in rows])*100:.3f}%。'
                   '支持局部冲突存在，但强度和平均方向本身不足以解释严重降点。'
                   '本次在预训练初始化测量，尚未覆盖训练过程的时间演化。'])
        r3.extend(['','## 1．完整模型梯度观察','',f'训练样本 {len(grads["samples"])}，eval BN／dropout，原版总任务和未来任务分别记录。','',
                   '| 对应任务 | 参数模块 | 有效样本数 | 平均余弦 | 负余弦比例 | 加权教师／任务范数比均值 |',
                   '|---|---|---:|---:|---:|---:|'])
        for kind in ('total_groups','future_groups'):
            modules=sorted({m for sample in grads['samples'] for m in sample[kind]})
            for module in modules:
                rows=[s[kind][module] for s in grads['samples'] if s[kind].get(module,{}).get('cosine') is not None]
                if rows:
                    r3.append(f'| {kind} | {module} | {len(rows)} | {np.mean([r["cosine"] for r in rows]):.5f} | '
                              f'{np.mean([r["cosine"]<0 for r in rows])*100:.1f}% | {np.mean([r["weighted_teacher_to_task_norm"] for r in rows]):.6f} |')
        globals=[s['total_groups']['shared_global']['cosine'] for s in grads['samples']]
        if any(v is not None and v<0 for v in globals):levels['R3']='Ⅰ：已观察到局部冲突；长期因果归因待干预判读'
        r3.extend(['','负余弦只证明局部冲突，不说明它一定造成严重降点。只有共有梯度参数进入夹角与范数比；'
                   '原始 JSON 保留只参与单一损失的参数数量。该观测不能覆盖训练态全部随机性。'])
    if means:
        selected={k:v for k,v in means.items() if k in ('task_heads_train','feature_heads_train','damped_heads_train','project_heads_train')}
        r3.extend(['','## 2．真实 AdamW 训练干预','',
                   'task＝仅任务；feature＝直接特征对齐；damped＝减弱强度；project＝去除冲突方向。'
                   '四组的原版未来语义／位置／速度头都可训练，历史编码冻结。',
                   *native_tables([('原版 SparseWorld',screen['baseline'])]+list(selected.items()),screen['baseline'])])
        r3.extend(['','减弱组与投影组同为 0.01 教师系数；投影有最多 2 倍残量恢复限制，不能自动宣称梯度范数完全匹配。',
                   '','| 组 | 有效梯度记录 | 平均余弦 | 加权辅助／任务范数比均值 |','|---|---:|---:|---:|'])
        for arm in ('feature_heads_train','damped_heads_train','project_heads_train'):
            stats=[g for name,row in screen['arms'].items() if name.startswith(arm+'_seed') and row.get('status')=='complete' for g in row.get('gradients',[])]
            if stats:r3.append(f'| {arm} | {len(stats)} | {np.mean([g["cosine"] for g in stats]):.5f} | '
                              f'{np.mean([g["effective_aux_norm"]/max(g["task_norm"],1e-30) for g in stats]):.6f} |')
        if all(k in selected and selected[k]['seed_count']==3 for k in ('task_heads_train','feature_heads_train','project_heads_train','damped_heads_train')):
            a,b,c,d=[selected[k] for k in ('task_heads_train','feature_heads_train','project_heads_train','damped_heads_train')]
            gap=average(a)-average(b)
            r3.extend(['','### 干预判读','',f'三时距平均 mIoU：任务 {average(a):.3f}；直接对齐 {average(b):.3f}；减弱 {average(d):.3f}；投影 {average(c):.3f}。'])
            if gap>.2:
                recovery=(average(c)-average(b))/gap
                r3.append(f'投影追回蒸馏额外下降的 {recovery*100:.1f}%；这不是相对预训练基线的全部恢复。')
                if average(c)>average(d) and recovery>0:
                    levels['R3']='Ⅱ：有筛选干预支持；缺独立场景全参数确认'
            else:
                r3.append('筛选中没有大于 0.2 点的三时距平均额外蒸馏损害，不稳定计算恢复比例；这不证明没有冲突。')
                levels['R3']='Ⅰ：局部冲突已观察；筛选未支持其解释主要退化'
            r3.append('本轮没有独立确认场景，且 S1／S2 参数范围不同，不能标为Ⅲ或Ⅳ。')
    r3.extend(['',f'原始数据：`{root/"s1_gradients.json"}`、`{root/"s2_screens.json"}`。设计见[设计文档](design.md)。'])
    if protocol:
        r3.extend(['','## 新增证据怎样影响冲突解释？','',
                   'BN-only 无优化器、无教师梯度也会明显下降；固定 BN 的配对训练能显著恢复本轮完整任务组。'
                   '这是训练协议干预，不是梯度投影，也不证明教师／任务冲突不存在。'
                   '它说明至少不能把采用可更新 BN 的全参数实验的大幅共同退化直接归给教师冲突。',
                   '固定 BN 后的正确／打乱教师复验及当前完成范围见[续实验结果](../s0_training_protocol/results.md)。'
                   '完整训练过程的梯度演化与同强度有限步干预仍未完成；当前 R3 仍停留在局部观察和机制筛选。'])
    write(documents/'r3_gradient_conflict/results.md',r3)
    r1=['# R1 实际结果：条件信息与可学性','',f'U1 状态：{conditions["status"] if conditions else "待执行"}。',
        '', '本节只报告辅助条件特征预测，不输出 occupancy，不将 NLL 当成原版 Student 性能。']
    r1.extend(r1_condition_explanation())
    if conditions:
        grouped={}
        for name,row in conditions['arms'].items():grouped.setdefault(name.split('_seed')[0],[]).append(row)
        r1.extend(['','## 1．先看总体特征误差','',
                   'MSE 越小，说明辅助模型越接近匹配教师向量；不是 occupancy mIoU。'
                   '每一项为完成种子的均值，同一份 query 清单。','',
                   '| 条件／目标 | 完成种子数 | 1 秒 MSE | 2 秒 MSE | 3 秒 MSE |','|---|---:|---:|---:|---:|'])
        for name,runs in grouped.items():
            values=[np.mean([row['metrics'][h+'/all']['mse'] for row in runs]) for h in HORIZONS]
            r1.append('| '+name+' | '+str(len(runs))+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
        r1.extend(['','### 概率质量：NLL（越小越好）','',
                   '确定性组的方差只从训练残差估计；概率组同时预测均值和对角方差。'
                   '这里评估未见场景上的匹配特征密度，不是 occupancy 概率，也不是多个真实未来分支。','',
                   '| 条件／目标 | 1 秒 NLL | 2 秒 NLL | 3 秒 NLL |','|---|---:|---:|---:|'])
        for name,runs in grouped.items():
            values=[np.mean([row['metrics'][h+'/all']['nll'] for row in runs]) for h in HORIZONS]
            r1.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
        r1.extend(['','### 分类别诊断明细','',
                   '类别由原版学生 query 的 48 点均值中心所在 GT 体素定义；不是整份 query 的多数类别。'
                   '动态语义类别不等于当时正在运动，空／其他 query 单列。','',
                   '<details>','<summary>展开各时距、动态／静态／空与其他 query 的 MSE、余弦与 NLL</summary>','',
                   '| 组 | 时距／类别组 | query 数 | MSE | 余弦误差 | NLL |', '|---|---|---:|---:|---:|---:|'])
        for name,runs in grouped.items():
            for key in sorted(set.intersection(*[set(row['metrics']) for row in runs])):
                values=[row['metrics'][key] for row in runs]
                r1.append(f'| {name}（{len(runs)} 种子） | {key} | {values[0]["count"]} | '
                          f'{np.mean([v["mse"] for v in values]):.6f} | {np.mean([v["cosine_error"] for v in values]):.6f} | '
                          f'{np.mean([v["nll"] for v in values]):.6f} |')
        r1.extend(['','</details>'])
        r1.extend(['','### 简单参照：降低误差是否仅需均值／空间先验？','',
                   '| 参照 | 时距 | MSE | 余弦误差 | NLL |','|---|---|---:|---:|---:|'])
        for name,values in conditions.get('auxiliary_baselines',{}).items():
            for h in HORIZONS:
                row=values[h+'/all'];r1.append(f'| {name} | {h} | {row["mse"]:.6f} | {row["cosine_error"]:.6f} | {row["nll"]:.6f} |')
        r1.append('\n若只用位置／时距的模型也有相近误差下降，不得把全部改善归因于历史语义信息；位置本身仍是历史预测的派生量。')
        completed={name:runs for name,runs in grouped.items() if len(runs)==3}
        required=('position_time_deterministic','history_deterministic','privileged_ego_deterministic')
        if all(name in completed for name in required):
            errors={name:float(np.mean([row['metrics'][h+'/all']['mse'] for row in completed[name] for h in HORIZONS])) for name in required}
            r1.extend(['','### 这些误差具体回答了什么？','',
                       f'三时距平均归一化特征 MSE：位置／时距 {errors[required[0]]:.6f}，'
                       f'加入历史预测特征 {errors[required[1]]:.6f}，再加入特权未来自车轨迹 {errors[required[2]]:.6f}。'])
            if errors[required[1]]<errors[required[0]]:
                r1.append('历史预测特征提供了位置／时距之外可用于拟合教师目标的线索，支持“未来感知表示至少有一部分可以由当前历史表示推测”。'
                          '这不是原版任务收益，也不证明剩余部分一定无法推测。')
            else:r1.append('本预算下没有看到历史特征稳定优于位置／时距参照；不能把优化失败解释为历史信息缺失。')
            if errors[required[2]]<errors[required[1]]*.99:
                r1.append('真实未来自车轨迹进一步降低了当前条件模型的误差；这支持增加该特权条件对拟合有帮助。'
                          '它不等于可用行为计划，也不证明完整历史在理论上无法推断相应线索。')
            else:r1.append('特权自车轨迹与历史组的总体 MSE 很接近；即使均值略低，也不能仅凭极小的点估计就认定有稳定帮助。'
                           '这里的 1% 只用于避免文字夸大微小幅度，不是统计显著性门槛。'
                           '本轮未做该辅助条件模型的场景级配对区间，也不排除其他条件、容量或长时距局部帮助。')
            r1.append('上述比较是在开发场景上的三种子均值；尚未提供独立场景确认或严格多未来分支，不宣称不可约信息边界。')
        r1.extend(['','## 2．假设被证明到什么程度','',
                   'U1 完成最多提供当前冻结表示、条件模型与预算下的可学性现象。真实未来自车轨迹是特权 GT，不等于可用行为计划。'
                   '方差或概率质量改善都不能证明原始历史存在不可约信息不足，也不能证明概率监督会提高 occupancy。',
                   '目标不唯一也不等于该监督必然有害：未来 GT 同样只有一次实现，学生仍可能学习条件平均。'
                   '还需原版任务下的干预，才能把不确定性归因为蒸馏失败原因。',
                   '', '## 3．U2 未运行','',str(conditions['u2']['reason']),
                   '没有严格相同历史的多未来场景分支，核心不可唯一确定命题仍未验证。不能用模拟向量实验代替真实场景证据。'])
    else:r1.extend(['','U2 尚无已验证的多分支环境／数据；不能将单轨迹 nuScenes 的误差当作不确定性证明。'])
    r1.extend(['',f'原始数据：`{root/"s4_conditions.json"}`。设计见[设计文档](design.md)。'])
    write(documents/'r1_conditional_information/results.md',r1)
    summary=['# 分阶段实验：逐条证据汇总','', '## 先看现在证明了什么','',
             '| 假设 | 当前证据强度 | 仍不能推出什么 |','|---|---|---|',
             f'| R1：历史条件不足 | {levels["R1"]} | 不能由单轨迹误差或预测方差证明不可约下界 |',
             f'| R2：特征相似不等于任务正确 | {levels["R2"]} | 不能推出所有教师任务信息都无用 |',
             f'| R3：教师／任务优化冲突 | {levels["R3"]} | 不能把观测到冲突直接解释为全部降点原因 |',
             '', '等级Ⅰ为现象、Ⅱ为受控干预、Ⅲ为多种子及独立确认的重要因素、Ⅳ为足以解释主要失败。等级不是成立概率。',
             '', '## 三条最重要的判读','',
             '1. **先分清共同退化。** 仅任务全参数训练也明显下降；恢复感知部分可追回大部分损失。'
             '这不是教师造成全部降点的证据。',
             '2. **当前特征接口并不能直接互换。** 完整教师特征替换降低相似度误差，却损害原版解码；'
             '固定几何的部分混合在 3 秒有局部正效应，不能说未来信息一概无用。',
             '3. **其余两项不能抢先定论。** 负梯度余弦只证明局部冲突；条件模型的残差只反映有限模型下的可学性。'
             '两者都还不足以解释主要失败，更没有证明理论上必须使用不同特征。',
             '', '## 分层结果入口','',
             '- [S0：共同训练退化](s0_protocol/results.md)',
             '- [S0 续实验：BN、损失组合与几何梯度](s0_training_protocol/results.md)',
             '- [R3：梯度与更新方向](r3_gradient_conflict/results.md)',
             '- [R2：原版头特征干预与适配](r2_task_compatibility/results.md)',
             '- [R1：条件信息与可学性](r1_conditional_information/results.md)',
             '', '## 实际执行状态','',
             '| 阶段 | 状态 |','|---|---|',
             f'| S0 | {"已完成" if s0 and s0["status"]=="complete" else "待执行／进行中"} |',
             f'| S1-R2 | {"已完成" if mix and mix["status"]=="complete" else "待执行／进行中"} |',
             f'| S1-R3 | {"已完成" if grads and grads["status"]=="complete" else "待执行／进行中"} |',
             f'| S2 | {"已完成：7 组 × 3 种子" if screen_status=="complete" else screen_status} |',
             f'| S4-U1 | {"已完成：6 组 × 3 种子" if conditions and conditions["status"]=="complete" else "待执行／进行中"} |',
             f'| S0 训练协议续实验 | {"本轮已完成" if protocol and protocol["status"]=="complete" else "进行中" if protocol else "尚无正式结果"} |',
             '| S3 全参数／独立场景确认 | 未完成：短程协议对照不替代长预算训练和独立场景确认 |',
             '| S5 组合确认 | 未进入：需先有可复验的独立有效干预 |',
             '| U2 严格相同历史、多未来 | 未运行：尚无验证过的环境／分支数据 |',
             '', '## 下一阶段为什么不能直接扩大蒸馏？','',
             '| 优先级 | 下一项工作 | 进入下一步的条件 |','|---|---|---|',
             '| 先修共同训练 | 用仅任务对照定位感知权重／BN 变化及损失配置；验证稳定的原版头训练 | 仅任务组不再发生远大于教师净影响的共同退化 |',
             '| 再确认接口 | 在新的确认场景上复验固定几何、有限强度干预；分别检验语义与几何 | 局部收益可复验，且不是借助未来输入的部署成绩 |',
             '| 再查长期原因 | 同参数、同预算训练；复核特征确实变化，联合看任务分数与梯度 | 有超过开发噪声的蒸馏额外损害或收益，才谈恢复比例和重要原因 |',
             '| 最后验证信息边界 | 提供严格相同历史／动作的多未来分支，或可执行受控环境 | 同时观察教师目标和任务真值的分支差异，不能只拿回归误差代替 |',
             '', '这轮不是三个假设都被证明了。它已经缩小了可解释范围，但仍没有证明未来感知特征在理论上不适合预测，'
             '也没有证明所有蒸馏方式都无效。',
             '', '## 判读边界','',
             '共同训练问题与教师额外作用必须分开。S2 冻结历史编码，只检验未来模块机制；U1 是辅助条件模型，不是占据读出器。'
             '未完成、门槛未满足及无环境的实验都如实保留，不把它们标为假设失败。',
             '', f'自动流程状态：{pipeline.get("status","未启动")}；阶段：{pipeline.get("phase","—")}。',
             f'运行错误：{pipeline.get("error","无已记录错误")}。',
             '', f'原始结果目录：`{root}`。复现设置见[阶段协议](stage_protocol.md)。']
    if conditions and conditions.get('status')=='complete':
        position=[r for k,r in conditions['arms'].items() if k.startswith('position_time_deterministic_seed')]
        history=[r for k,r in conditions['arms'].items() if k.startswith('history_deterministic_seed')]
        if len(position)==len(history)==3:
            p=float(np.mean([r['metrics'][h+'/all']['mse'] for r in position for h in HORIZONS]))
            q=float(np.mean([r['metrics'][h+'/all']['mse'] for r in history for h in HORIZONS]))
            index=summary.index('## 分层结果入口')
            summary[index:index]=[
                '4. **教师表示并非全部不可学。** '
                f'辅助条件模型的三时距平均归一化特征 MSE：仅位置／时距 {p:.6f}，加入历史预测特征 {q:.6f}。'
                '这支持当前历史表示可拟合其中一部分，但不证明原版 occupancy 受益；剩余误差是否不可约仍未验证。','']
            summary[summary.index('## 三条最重要的判读')]='## 四条最重要的判读'
    if protocol:
        bn=protocol.get('bn_only',{}).get('steps',{})
        if bn:
            last=max(bn,key=int);row=bn[last]
            index=summary.index('## 实际执行状态')
            scores='／'.join(f'{row["metrics"][h]["mIoU"]:.3f}' for h in HORIZONS)
            section=['## 新增证据：先排除训练协议的共同退化','',
                f'在同一 64 样本上，仅进行 {last} 次训练图像前向、更新 BN 统计，原版头的 1／2／3 秒 mIoU 为 {scores}。'
                '该组没有优化器或教师监督；这是 BN 统计变化的单因素干预，不是教师失败。'
                '它证明此设置的 BN 统计变化本身足以导致退化，不证明全部旧实验都由此造成。','']
            arms=protocol.get('arms',{})
            controlled={}
            for label in ('full_bn_train','full_bn_fixed','future_bn_train','future_bn_fixed',
                          'full_bn_fixed_detached_geom','future_bn_fixed_detached_geom'):
                runs=[r['steps']['256'] for r in arms.values() if r['arm']==label
                      and r['supervision']=='task' and r.get('status')=='complete' and '256' in r.get('steps',{})]
                if len(runs)==1:controlled[label]=runs[0]
            if all(name in controlled for name in ('full_bn_train','full_bn_fixed')):
                train,fixed=controlled['full_bn_train'],controlled['full_bn_fixed']
                loss=average(protocol['baseline'])-average(train)
                section.extend(['### 从相同初始化重新训练，固定 BN 有没有用？','',
                    '| 完整任务组 | 1 秒 mIoU | 2 秒 | 3 秒 |','|---|---:|---:|---:|'])
                for name,result in [('BN 更新',train),('BN 固定',fixed)]:
                    section.append('| '+name+' | '+' | '.join(f'{result["metrics"][h]["mIoU"]:.3f}' for h in HORIZONS)+' |')
                if loss>.2:
                    section.extend(['',f'固定 BN 追回**本轮 128 训练／64 验证、256 更新**完整任务组下降的 '
                                    f'{100*(average(fixed)-average(train))/loss:.1f}%。'
                                    '它是这轮共同退化的重要训练协议因素；不等于解释旧 5 epoch 的全部下降，'
                                    '更不适用于已经冻结感知或保持原版固定统计的所有旧实验。'])
                section.extend(['','固定 BN 下，删去当前／temporal 辅助组合和分离几何梯度没有带来同等量级恢复。'
                                '按预先规定的门槛和优先级，后续采用完整任务＋固定 BN，不按最高验证分数挑选。',''])
            paired={r['supervision']:r['steps']['256'] for r in arms.values()
                    if r['arm']=='full_bn_fixed' and r.get('status')=='complete' and '256' in r.get('steps',{})}
            if all(name in paired for name in ('task','teacher','shuffled')):
                section.extend(['### 固定训练协议后，再判断教师的净作用','',
                    '| 监督 | 1 秒 mIoU | 2 秒 | 3 秒 | 三时距平均 |','|---|---:|---:|---:|---:|'])
                for key,name in [('task','仅任务'),('teacher','正确教师'),('shuffled','打乱教师')]:
                    result=paired[key]
                    section.append('| '+name+' | '+' | '.join([*(f'{result["metrics"][h]["mIoU"]:.3f}' for h in HORIZONS),f'{average(result):.4f}'])+' |')
                section.extend(['','以上三组使用同一开发集、同一种子、相同 256 更新。单种子微小差别不能写成稳定收益或严重损害；'
                                '特征误差是否降低、配对区间和动态／静态统计见下层报告。'
                                '这一复验不验证 R1 的严格多未来命题，也不是 S3 独立场景确认。',''])
                checks=read(protocol_root/'bootstrap.json') or {}
                key='full_bn_fixed_teacher_seed20260930 vs full_bn_fixed_task_seed20260930'
                interval=checks.get('paired_changes',{}).get(key,{}).get('mean_1_2_3s')
                if interval and interval[0]<=0<=interval[1]:
                    section.extend(['正确教师与任务组的三时距平均配对场景区间跨零，**尚未建立稳定蒸馏净收益，也未建立稳定额外损害**。'
                                    '这不是证明二者等效；单种子、64 个开发场景不能排除小效应或更长训练后的变化。',''])
            else:
                section.extend(['### 同协议教师复验','',
                                '正确教师和打乱教师的最终配对结果尚未齐全，不能据当前训练协议结果判断教师的净收益。',''])
            section.extend(['设计、原版头指标与特征审计见[S0 续实验](s0_training_protocol/results.md)。',''])
            summary[index:index]=section
            if all(name in paired for name in ('task','teacher','shuffled')):
                index=summary.index('## 先看现在证明了什么')
                summary[index:index]=['这次新增实验首先定位了一个训练协议因素：**BN 模式改变可造成明显共同退化，固定它能大幅恢复本轮短程结果。**'
                                     '恢复后，正确教师仍未显示稳定原版预测头净收益；因此，既不能把之前全部降点归给蒸馏，'
                                     '也不能据此证明未来教师信息无用。下表分别列出三个研究假设真正被验证到的层次。','']
    entry=read(root.parent/'20261007_confirmation_entry/entry_audit.json')
    if entry and entry.get('status')=='prerequisite_audit_complete_not_training_complete':
        index=summary.index('## 下一阶段为什么不能直接扩大蒸馏？')
        coverage=entry['scene_coverage']
        summary[index:index]=[
            '## S3、S5、U2：本次补充了什么？','',
            '**本次完成三项设计与进入审计，没有完成对应正式实验，也没有新增模型性能结论。**','',
            '| 阶段 | 审计发现 | 设计／执行状态 |','|---|---|---|',
            f'| S3 | 本地验证场景 {coverage["val"]["available_scenes"]} 个，未涉及场景 '
            f'{coverage["val"]["unused_count"]} 个；不能把重新抽样叫独立确认 | '
            '[设计](s3_confirmation/design.md)／[状态](s3_confirmation/results.md) |',
            '| S5 | 尚无两项经过独立多种子确认的有效修复，组合门槛不满足 | '
            '[设计](s5_combination/design.md)／[状态](s5_combination/results.md) |',
            '| U2 | 没有已核验的同条件多未来数据；本地 Python／PATH 检查也未找到直接可用 CARLA | '
            '[设计](u2_multiple_futures/design.md)／[状态](u2_multiple_futures/results.md) |','',
            '审计未新增训练、未生成未来分支、未改变 R1–R3 证据等级。'
            '需要提供确认数据／分支环境，或明确选择先进行旧开发集扩大预算复验；'
            '后者不能替代独立确认。','']
    development=read(root.parent/'20261008_development_confirmation/results.json')
    if development:
        index=summary.index('## 下一阶段为什么不能直接扩大蒸馏？')
        finished=sum(row.get('status')=='complete' for row in development.get('arms',{}).values())
        summary[index:index]=[
            '## 已获授权的长程开发集复验','',
            '**用户已同意先在旧开发集实施 4096／1024、5 epoch、三种子、三监督复验。** '
            '该组保留所有学生可训练参数、固定 BN 统计，直接评测原生头；它不是独立确认。','',
            f'本次汇总快照：`{development["status"]}`，阶段 `'
            f'{development.get("phase","—")}`，已完成训练 {finished}/9。'
            '后台训练的实时进度与末次成绩以 [S3 自动报告](s3_confirmation/results.md) 为准；'
            '不把尚未完成的组计入三种子最终均值。','',
            '新增入口：`tools/experiment_future_distillation_followup.py`；配置：'
            '`config/opus/future_distillation_development_confirmation.py`。'
            'S5 仍按有效干预门槛执行，U2 仍等待真实分支来源，不能因批准训练而标记三阶段全部完成。','']
    write(documents/'summary.md',summary)
    if mix:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,3.2))
        for geometry,label in [('fixed','Fixed geometry'),('native_joint','Native joint geometry')]:
            cases=sorted([(float(k.split('mix')[1]),v) for k,v in mix['cases'].items() if k.startswith(geometry+'_')])
            axes[0].plot([a for a,_ in cases],[average(v) for _,v in cases],marker='o',label=label)
            axes[1].plot([a for a,_ in cases],[v['feature_cosine_error'] for _,v in cases],marker='o',label=label)
        for axis in axes:axis.set_xlabel('Teacher feature mixing ratio');axis.grid(alpha=.25)
        axes[0].set_ylabel('Native mean mIoU (1/2/3s, %)');axes[1].set_ylabel('Feature cosine error')
        axes[0].legend(fontsize=8);fig.tight_layout()
        fig.savefig(documents/'r2_task_compatibility/feature_intervention.svg');plt.close(fig)
        path=documents/'r2_task_compatibility/results.md'
        with path.open('a') as handle:handle.write('\n## 图示：更像教师与任务性能\n\n![原版头性能与特征距离](feature_intervention.svg)\n')
    if conditions and conditions.get('status')=='complete':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axis=plt.subplots(figsize=(6.8,3.5))
        for name,label in [('position_time_deterministic','Position/time'),('history_deterministic','History features'),('privileged_ego_deterministic','History + privileged ego GT')]:
            runs=[row for key,row in conditions['arms'].items() if key.startswith(name+'_seed')]
            if runs:axis.plot([1,2,3],[np.mean([row['metrics'][h+'/all']['mse'] for row in runs]) for h in HORIZONS],marker='o',label=label)
        for name,label in [('train_mean','Train target mean'),('uncalibrated_student','Uncalibrated student vector')]:
            values=conditions.get('auxiliary_baselines',{}).get(name)
            if values:axis.plot([1,2,3],[values[h+'/all']['mse'] for h in HORIZONS],linestyle='--',label=label)
        axis.set_xlabel('Horizon (seconds)');axis.set_ylabel('Auxiliary normalized feature MSE (not occupancy)')
        axis.grid(alpha=.25);axis.legend(fontsize=8);fig.tight_layout()
        fig.savefig(documents/'r1_conditional_information/auxiliary_feature_mse.svg');plt.close(fig)
        with (documents/'r1_conditional_information/results.md').open('a') as handle:
            handle.write('\n## 图示：哪些条件改善了有限模型的特征拟合？\n\n![辅助条件模型的特征误差，不是 occupancy 指标](auxiliary_feature_mse.svg)\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'out/future_distillation_causal/20260930')
    parser.add_argument('--documents',type=Path,default=DOCUMENTS)
    args=parser.parse_args();update_reports(args.root,args.documents)
