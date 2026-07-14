# AdaptiveAllocation 模块设计文档

## 1. 总体概述

### 1.1 设计目标

基于现有 [`DensifyOnly`](model/encoder/gaussian_encoder/topk_module/topk_densify_v2.py:12) 模块，构建一个新的 `AdaptiveAllocation` 模块。核心区别：**不再使用 Top-K 选取 + 统一 densify**，而是对**每一个输入高斯**通过网络学习一个三分类决策因子 `[keep, clone, split]`，自适应地决定每个高斯的命运。

### 1.2 输入输出规范

| 项目 | 符号 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `instance_feature` | `B × N × E` | 实例特征，E = feat_embed_dim |
| 输入 | `anchor` | `B × N × A` | 锚点表示 |
| 输入 | `gaussian` | `GaussianPrediction` | means, scales, rotations, opacities, semantics |
| 输出 | `result_anchor` | `B × M × A` | M = N + N_clone + N_split (动态) |
| 输出 | `result_gaussian` | `GaussianPrediction` | 形状 (B, M, *) |
| 输出 | `result_features` | `B × M × E` | 拼接后的实例特征 |

> 输出数量 M = N_keep + 2×N_clone + 2×N_split = N + N_clone + N_split，是动态值，由决策因子决定。

### 1.3 与 DensifyOnly 的核心差异

```
DensifyOnly:
  按opacity选Top-K → 统一Densify网络 → 残差优化 → Concat[原始N, 新K]

AdaptiveAllocation:
  对每个高斯学习决策因子 → 三分类路由 → ┬─ keep:  原样保留
                                         ├─ clone: 保留原高斯 + 生成1个新副本
                                         └─ split: 舍弃原高斯 + 生成2个子高斯
                                                    (split_mode控制约束/自由)
```

| 特性 | DensifyOnly | AdaptiveAllocation |
|------|------------|-------------------|
| 选择机制 | Top-K by opacity | 网络学习三分类决策 |
| 操作粒度 | K个固定数量 | 所有N个高斯，动态数量 |
| Clone | 无（统一操作） | 保留原高斯 + 1个新副本 |
| Split | 无 | 舍弃原高斯 + 2个子高斯 |
| 输出数量 | N+K (固定) | N + N_clone + N_split (动态) |

---

## 2. 任务分解

| 子任务 | 名称 | 职责 |
|--------|------|------|
| T1 | 决策因子学习 | 从 instance_feature 学习每个高斯的 keep/clone/split 概率 |
| T2 | 分类与路由 | 根据决策因子将高斯分为三类，分别路由到对应处理分支 |
| T3 | Clone 分支 | 对 clone 高斯执行残差优化，生成新副本 |
| T4 | Split 分支 | 对 split 高斯生成两个子高斯，支持 constrained/free 模式 |
| T5 | 结果装配 | 将 keep/clone/split 的输出组装为最终张量 |

---

## 3. 子任务详细设计

### 3.1 T1: 决策因子学习

#### 3.1.1 功能

输入 `instance_feature` (B, N, E)，输出 `decision_probs` (B, N, 3)，三个通道分别表示该高斯的 `[P_keep, P_clone, P_split]`。

#### 3.1.2 数据流图

```
┌──────────────────────────────────────────────────────┐
│                T1: DecisionFactorLearning             │
│                                                      │
│  instance_feature (B, N, E)                           │
│      │                                               │
│      ▼                                               │
│  ┌──────────────────────────┐                        │
│  │  decision_net            │                        │
│  │  Linear → LN → GELU →   │                        │
│  │  Linear → 3             │                        │
│  └──────────────────────────┘                        │
│      │                                               │
│      ▼                                               │
│  decision_logits (B, N, 3)                           │
│      │                                               │
│      ├── [Training]  GumbelSoftmax(hard=True)        │
│      └── [Inference] Softmax → Argmax                │
│      │                                               │
│      ▼                                               │
│  decision_probs (B, N, 3)                            │
│  [P_keep, P_clone, P_split]                           │
│                                                      │
└──────────────────────────────────────────────────────┘
```

#### 3.1.3 逻辑描述

```
输入: instance_feature (所有N个高斯的特征)

1. 通过一个轻量MLP将每个高斯的特征映射为3维logits
2. 训练时用Gumbel-Softmax得到近似one-hot的可微决策
3. 推理时用Softmax后取Argmax得到硬分类标签

输出: 每个高斯的 {keep, clone, split} 决策标签及概率
```

---

### 3.2 T2: 分类与路由

#### 3.2.1 功能

根据 T1 的决策结果，将 N 个高斯划分到三个集合中，并组织对应的特征、锚点和高斯属性。

#### 3.2.2 数据流图

```
┌──────────────────────────────────────────────────────────┐
│                  T2: ClassificationAndRouting             │
│                                                          │
│  decision_label (B, N)  ∈ {0:keep, 1:clone, 2:split}     │
│      │                                                   │
│      ▼                                                   │
│  ┌────────────────────────────────────────────┐          │
│  │         按标签分组                           │          │
│  │                                            │          │
│  │  keep_mask  = (label == 0)                 │          │
│  │  clone_mask = (label == 1)                 │          │
│  │  split_mask = (label == 2)                 │          │
│  └────────────────────────────────────────────┘          │
│      │                                                   │
│      ├──► keep 组:   instance_feature[keep]              │
│      │                anchor[keep]                       │
│      │                gaussian[keep]                     │
│      │                → 直接传递给 T5 装配                │
│      │                                                   │
│      ├──► clone 组:  instance_feature[clone]             │
│      │                anchor[clone]                      │
│      │                gaussian[clone]                    │
│      │                → 输入 T3 Clone分支                │
│      │                                                   │
│      └──► split 组:  instance_feature[split]             │
│                       anchor[split]                      │
│                       gaussian[split]                    │
│                       → 输入 T4 Split分支                │
│                                                          │
│  注意: 同时记录每个高斯在原始序列中的位置索引，          │
│        供 T5 装配时进行原位替换使用                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### 3.2.3 逻辑描述

```
输入: decision_labels (每个高斯的决策标签)

1. 根据标签将instance_feature, anchor, gaussian分别划分到三个组
2. keep组直接保留，不经过任何变换
3. clone组送入Clone分支网络
4. split组送入Split分支网络
5. 同时记录split组在原始序列中的位置索引，供装配时原位替换
```

---

### 3.3 T3: Clone 分支

#### 3.3.1 功能

对 clone 组的高斯执行与 [`DensifyOnly`](model/encoder/gaussian_encoder/topk_module/topk_densify_v2.py:12) 一致的操作：浅层特征投影 → 残差偏移预测 → 几何属性更新，生成新高斯。**原高斯保留不变**。

#### 3.3.2 数据流图

```
┌────────────────────────────────────────────────────────────┐
│                      T3: CloneBranch                       │
│                                                            │
│  clone_feats (来自被标记为clone的高斯)                       │
│  clone_gaussian                                            │
│  clone_anchor                                              │
│      │                                                     │
│      ▼                                                     │
│  ┌──────────────────────────────┐                          │
│  │  clone_feature_proj          │                          │
│  │  Linear → LN → GELU          │                          │
│  └──────────────────────────────┘                          │
│      │                                                     │
│      ▼                                                     │
│  projected_feats                                           │
│      │                                                     │
│      ├──► location_shift  ──► means 残差优化               │
│      ├──► scale_shift     ──► scales 残差优化              │
│      ├──► rotation_shift  ──► rots 残差优化                │
│      ├──► semantic_shift  ──► sems 残差优化                │
│      └──► opacity_shift   ──► opas 残差优化                │
│      │                                                     │
│      ▼                                                     │
│  ┌──────────────────────────────────────────┐              │
│  │  残差优化 (与DensifyOnly完全一致)          │              │
│  │                                          │              │
│  │  means:  new = old + Δ×scales            │              │
│  │          → Clamp(pc_range)               │              │
│  │  scales: new = old × Sigmoid(Δ)          │              │
│  │          → Clamp(scale_range)            │              │
│  │  rots:   new = old/2 + Norm(Δ)/2         │              │
│  │  sems:   new = old/2 + Sigmoid(Δ)/2      │              │
│  │  opas:   new = old/2 + Sigmoid(Δ)/2      │              │
│  └──────────────────────────────────────────┘              │
│      │                                                     │
│      ▼                                                     │
│  输出:                                                      │
│  ├── clone_new_features    (新高斯的特征)                   │
│  ├── clone_new_anchor      (新高斯的锚点)                   │
│  └── clone_new_gaussian    (新高斯的几何属性)               │
│                                                            │
│  注意: 原高斯保持不变，新高斯追加到序列末尾                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 3.3.3 逻辑描述

```
输入: clone组的 instance_feature, anchor, gaussian

1. 特征投影: 浅层MLP处理特征
2. 并行预测: 5个独立Linear头分别预测 means/scale/rot/sem/opa 的偏移量
3. 残差更新:
   - 位置: 以旧scale为单位做相对偏移，限制在pc_range内
   - 缩放: 乘法残差，限制在scale_range内
   - 旋转: 平均混合 (old/2 + new/2)
   - 语义/不透明度: 平均混合
4. 锚点同步更新: 将新高斯的属性反算为anchor格式

输出: 新高斯的 feature, anchor, gaussian
```

---

### 3.4 T4: Split 分支

#### 3.4.1 功能

对 split 组的高斯执行真正的"分裂"操作：**舍弃原父高斯**，通过网络生成**两个子高斯**。子高斯的行为由 `split_mode` 参数控制。

```
操作语义:
  Split:  ● (父高斯)  ──►  ◎₁ (子1, 替换原位)
                          ◎₂ (子2, 追加末尾)
          原高斯被舍弃
```

#### 3.4.2 数据流图

```
┌──────────────────────────────────────────────────────────────────┐
│                         T4: SplitBranch                          │
│                                                                  │
│  split_feats (来自被标记为split的高斯)                             │
│  split_gaussian  ← 父高斯 (将被舍弃)                              │
│  split_anchor                                                    │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────────────────────────┐                        │
│  │  split_feature_proj (深层MLP)         │                        │
│  │  Linear→LN→GELU → Linear→LN→GELU    │                        │
│  └──────────────────────────────────────┘                        │
│      │                                                           │
│      ├─────────────────────────────────┐                         │
│      ▼                                 ▼                         │
│  ┌───────────────────┐     ┌───────────────────┐                 │
│  │  子高斯1 预测头    │     │  子高斯2 预测头    │                 │
│  │  (5个几何+1个特征) │     │  (5个几何+1个特征) │                 │
│  └────────┬──────────┘     └────────┬──────────┘                 │
│           │                         │                             │
│           ▼                         ▼                             │
│  ┌─────────────────────────────────────────────────────┐         │
│  │              根据 split_mode 生成子高斯               │         │
│  │                                                     │         │
│  │  ┌─ constrained ──────────────────────────────────┐ │         │
│  │  │  沿父高斯最大scale轴方向对称分裂                  │ │         │
│  │  │  child1_pos = parent + direction × offset       │ │         │
│  │  │  child2_pos = parent - direction × offset       │ │         │
│  │  │  child_scale = parent_scale × [0.3, 0.7]       │ │         │
│  │  │  child_rot/sem/opa = parent + 微调残差           │ │         │
│  │  └────────────────────────────────────────────────┘ │         │
│  │                                                     │         │
│  │  ┌─ free ────────────────────────────────────────┐ │         │
│  │  │  子高斯位置/属性完全由网络自由学习               │ │         │
│  │  │  child_pos = parent + Tanh(δ) × 半场景范围     │ │         │
│  │  │  child_scale/rot/sem/opa = 独立预测            │ │         │
│  │  └────────────────────────────────────────────────┘ │         │
│  └─────────────────────────────────────────────────────┘         │
│           │                                                      │
│      ┌────┴────┐                                                 │
│      ▼         ▼                                                 │
│  ┌────────┐ ┌────────┐                                          │
│  │★Clamp  │ │★Clamp  │  ← means→pc_range, scales→scale_range    │
│  │全部属性 │ │全部属性 │    opas/sems→[0,1]                       │
│  └───┬────┘ └───┬────┘                                          │
│      │          │                                                 │
│      ▼          ▼                                                 │
│  输出:                                                             │
│  ├── child1: (feature, anchor, gaussian) → 替换父高斯原位          │
│  └── child2: (feature, anchor, gaussian) → 追加到序列末尾          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### 3.4.3 两种模式对比

| 维度 | constrained | free |
|------|------------|------|
| 子高斯位置 | 沿父高斯最大scale轴，对称偏移 | 完全自由，Tanh偏移 |
| 缩放 | 继承父scale × [0.3, 0.7] | 独立 Softplus 预测 |
| 旋转 | 父旋转 + 微调残差 | 完全独立预测 |
| 语义/不透明度 | 父属性 + Tanh残差 | 完全独立预测 |
| 适用场景 | 保持空间连续性，父高斯过大概要分裂 | 探索新的空间位置 |

#### 3.4.4 边界安全

由于 Split 使用 Tanh 残差和自由偏移，子高斯属性容易越界。**所有几何属性在构造输出前必须 Clamp**：

- `means` → Clamp 到 `pc_range`（最关键，否则 `InverseSigmoid` → NaN）
- `scales` → Clamp 到 `scale_range`
- `opas`/`sems` → Clamp 到 `[0, 1]`

#### 3.4.5 逻辑描述

```
输入: split组的 instance_feature, anchor, gaussian (父高斯)

1. 深层特征投影: 2层MLP提取更丰富的特征表示
2. 双子预测: 两个独立的预测头组，每组包含:
   - 5个几何偏移预测头 (means/scale/rot/sem/opa)
   - 1个特征变换头 (生成子高斯的instance_feature)
3. 根据split_mode计算子高斯属性:
   - constrained: 沿父高斯最大scale轴方向对称分裂
   - free: 网络自由学习子高斯位置和属性
4. 边界裁剪: 所有属性Clamp到合法范围
5. 锚点回算: 子高斯属性转anchor格式

输出: child1(替换原位) 和 child2(追加末尾) 的 feature, anchor, gaussian
```

---

### 3.5 T5: 结果装配

#### 3.5.1 功能

将 keep/clone/split 三组的输出组装为最终输出张量。

#### 3.5.2 装配规则

```
操作           输入消耗         输出贡献              放置位置
────────────────────────────────────────────────────────────
keep    N_keep个原高斯   →  N_keep个原高斯        原位置不变
clone   N_clone个原高斯  →  N_clone个原高斯        原位置不变
                           N_clone个新副本         序列末尾追加
split   N_split个原高斯  →  N_split个child_1      替换原位置
                           N_split个child_2       序列末尾追加
────────────────────────────────────────────────────────────
总计:   输入N个          输出N + N_clone + N_split
```

#### 3.5.3 数据流图

```
┌──────────────────────────────────────────────────────────────────┐
│                       T5: ResultAssembly                         │
│                                                                  │
│  原始 instance_feature, anchor, gaussian (B, N, *)               │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────────────────────────────────────┐            │
│  │  Step 1: 复制原始张量                              │            │
│  │  mod_* = Clone(original_*)                        │            │
│  └──────────────────────────────────────────────────┘            │
│      │                                                           │
│      │  split位置索引                                            │
│      ▼                                                           │
│  ┌──────────────────────────────────────────────────┐            │
│  │  Step 2: Split原位替换                             │            │
│  │                                                  │            │
│  │  对于每个split位置i:                               │            │
│  │    mod_feature[i]  ← split_child1_feature        │            │
│  │    mod_anchor[i]   ← split_child1_anchor          │            │
│  │    mod_gaussian[i] ← split_child1_gaussian        │            │
│  └──────────────────────────────────────────────────┘            │
│      │                                                           │
│      ▼                                                           │
│  mod_* (B, N, *)  ← split位置已被child_1替换                     │
│      │                                                           │
│      │  clone_new_* (N_clone个)                                  │
│      │  split_child2_* (N_split个)                               │
│      ▼                                                           │
│  ┌──────────────────────────────────────────────────┐            │
│  │  Step 3: 构造追加部分                              │            │
│  │                                                  │            │
│  │  new_section = [所有clone新副本, 所有split_child2] │            │
│  │  形状: (B, N_clone + N_split, *)                  │            │
│  └──────────────────────────────────────────────────┘            │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────────────────────────────────────┐            │
│  │  Step 4: 拼接                                     │            │
│  │                                                  │            │
│  │  result = Concat([mod_*(N), new_section], dim=1) │            │
│  │  形状: (B, N + N_clone + N_split, *)              │            │
│  └──────────────────────────────────────────────────┘            │
│      │                                                           │
│      ▼                                                           │
│  最终输出:                                                        │
│  ├── result_anchor   (B, M, A)  where M = N+N_clone+N_split      │
│  ├── result_gaussian (B, M, *)                                   │
│  └── result_features (B, M, E)                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### 3.5.4 逻辑描述

```
输入: 原始张量 + clone/split输出 + 分类掩码

1. 复制原始张量得到 mod_* (用于原位修改)
2. 对于每个split位置，用child_1替换mod_*中对应位置的值
   (clone和keep位置保持原样)
3. 收集所有clone新副本和split_child2，构成追加部分
4. 将修改后的原始部分和追加部分沿高斯维度拼接

输出: 最终的结果张量
```

---

## 4. 总体数据流图

```
┌──────────────────────────────────────────────────────────────────┐
│                  AdaptiveAllocation.forward()                     │
│                                                                  │
│  Input: instance_feature, anchor, gaussian  (B, N, *)            │
│      │                                                           │
│      ▼                                                           │
│  ╔══════════════════════════════════╗                             │
│  ║  T1: DecisionFactorLearning     ║                             │
│  ║  每个高斯 → [P_keep,P_clone,P_split]                         │
│  ╚══════════════════════════════════╝                             │
│      │                                                           │
│      ▼                                                           │
│  ╔══════════════════════════════════╗                             │
│  ║  T2: ClassificationAndRouting   ║                             │
│  ║  按决策标签分三组:               ║                             │
│  ║  keep → 直接传递                ║                             │
│  ║  clone → T3, split → T4        ║                             │
│  ╚══════════════════════════════════╝                             │
│      │                                                           │
│      ├──────────────┬──────────────────┐                          │
│      ▼              ▼                  ▼                          │
│  ╔══════════╗  ╔══════════╗  ╔══════════════╗                    │
│  ║   keep   ║  ║T3: Clone ║  ║ T4: Split    ║                    │
│  ║  原样传递 ║  ║          ║  ║              ║                    │
│  ║          ║  ║ 浅层投影  ║  ║ 深层投影     ║                    │
│  ║          ║  ║ 5头预测   ║  ║ 双子预测头   ║                    │
│  ║          ║  ║ 残差优化  ║  ║ split_mode   ║                    │
│  ║          ║  ║ → 1个副本 ║  ║ → child1,2   ║                    │
│  ╚══════════╝  ╚══════════╝  ╚══════════════╝                    │
│      │              │                  │                          │
│      └──────────────┼──────────────────┘                          │
│                     ▼                                             │
│  ╔══════════════════════════════════╗                             │
│  ║  T5: ResultAssembly             ║                             │
│  ║                                  ║                             │
│  ║  split child_1 → 原位替换       ║                             │
│  ║  clone新副本 + child_2 → 追加   ║                             │
│  ║  Concat → (B, M, *)             ║                             │
│  ╚══════════════════════════════════╝                             │
│      │                                                           │
│      ▼                                                           │
│  Output:                                                          │
│  result_anchor (B, M, A)    where M = N + N_clone + N_split       │
│  result_gaussian (B, M, *)                                        │
│  result_features (B, M, E)                                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. 模块初始化参数

```
AdaptiveAllocation.__init__() 参数:

┌──────────────────┬──────────┬──────────────────────────────────┐
│ 参数名            │ 类型      │ 说明                              │
├──────────────────┼──────────┼──────────────────────────────────┤
│ feat_embed_dim   │ int      │ 特征维度，默认128                  │
│ semantic_dim     │ int      │ 语义类别数，默认17                 │
│ pc_range         │ List[6]  │ 点云范围                           │
│ scale_range      │ List[2]  │ 缩放范围 [min, max]               │
│ unit_xyz         │ List[3]  │ Clone分支单位位置偏移量             │
│ split_mode       │ str      │ "constrained"(默认) 或 "free"     │
│ gumbel_tau       │ float    │ Gumbel-Softmax温度，默认1.0        │
└──────────────────┴──────────┴──────────────────────────────────┘
```

---

## 6. 网络结构总览

```
AdaptiveAllocation 模块内部网络:

┌─ T1: Decision Network ──────────────────────────────┐
│  decision_net: Linear→LN→GELU→Linear→3               │
└──────────────────────────────────────────────────────┘

┌─ T3: Clone Branch ──────────────────────────────────┐
│  clone_feature_proj: Linear→LN→GELU                 │
│  clone_location_shift:  Linear(E→3)                 │
│  clone_scale_shift:     Linear(E→3)                 │
│  clone_rotation_shift:  Linear(E→4)                 │
│  clone_semantic_shift:  Linear(E→S)                 │
│  clone_opacities_shift: Linear(E→1)                 │
└──────────────────────────────────────────────────────┘

┌─ T4: Split Branch ──────────────────────────────────┐
│  split_feature_proj:                                │
│    Linear→LN→GELU → Linear→LN→GELU                 │
│                                                     │
│  # 子高斯1 (5几何 + 1特征)                           │
│  child1_location_shift:  Linear(E→3)                │
│  child1_scale_shift:     Linear(E→3)                │
│  child1_rotation_shift:  Linear(E→4)                │
│  child1_semantic_shift:  Linear(E→S)                │
│  child1_opacities_shift: Linear(E→1)                │
│  child1_feature_transform: Linear→LN→GELU          │
│                                                     │
│  # 子高斯2 (5几何 + 1特征)                           │
│  child2_location_shift:  Linear(E→3)                │
│  child2_scale_shift:     Linear(E→3)                │
│  child2_rotation_shift:  Linear(E→4)                │
│  child2_semantic_shift:  Linear(E→S)                │
│  child2_opacities_shift: Linear(E→1)                │
│  child2_feature_transform: Linear→LN→GELU          │
└──────────────────────────────────────────────────────┘
```

---

## 7. 训练与推理策略

### 7.1 训练阶段

```
1. 决策因子使用 Gumbel-Softmax (hard=True) 得到近似one-hot标签，保证梯度可传播
2. 所有三个分支 (keep/clone/split) 的前向都计算
3. 最终输出根据决策标签软加权组合:
   - keep部分:    原始高斯 × P_keep
   - clone部分:   原始高斯 × P_clone + 新副本 × P_clone
   - split部分:   child_1 × P_split + child_2 × P_split
4. 损失函数中可加入决策因子的熵正则化，防止过早收敛到单一类别
```

### 7.2 推理阶段

```
1. 决策因子 Softmax + Argmax 得到硬分类标签
2. 只计算对应分支 (keep不计算, clone/split分别计算)
3. 按 T5 装配规则组装最终输出
```

---

## 8. 与现有模块的集成

在 [`GaussianOccEncoder`](model/encoder/gaussian_encoder/gaussian_encoder.py:10) 中直接替换 `densify_layer`：

```yaml
densify_layer:
  type: "AdaptiveAllocation"
  feat_embed_dim: 128
  semantic_dim: 17
  pc_range: [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
  scale_range: [0.1, 10.0]
  unit_xyz: [0.5, 0.5, 0.5]
  split_mode: "constrained"
```

编码器中 `"densify" in op` 分支的调用签名 `(instance_feature, anchor, gaussian) → (anchor, gaussian, instance_feature)` 与 `AdaptiveAllocation` 完全兼容。

---

## 9. 文件组织

```
model/encoder/gaussian_encoder/topk_module/
├── __init__.py
├── topk_densify_v2.py          # 现有: DensifyOnly
├── topk_semantic_densify.py    # 现有: SemanticDensify
└── adaptive_allocation.py      # 新增: AdaptiveAllocation
```

---

## 10. 设计决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 操作粒度 | 所有N个高斯，不做Top-K | 让网络自主决定每个高斯的命运 |
| 决策因子维度 | 3 (keep/clone/split) | 三分类覆盖所有操作类型 |
| Clone分支逻辑 | 完全复用 DensifyOnly 残差模式 | 需求明确要求 |
| Split操作语义 | 舍弃父高斯 + 2个子高斯 | 真正的split，非clone的变体 |
| Split输出结构 | child_1替换原位, child_2追加 | child_1继承父高斯在序列中的位置 |
| split_mode | constrained (默认) / free | constrained保持空间连续性 |
| 输出数量 | N + N_clone + N_split (动态) | 由决策因子自适应决定 |
| 可微性 | Gumbel-Softmax (训练) / Argmax (推理) | 训练梯度可传播，推理高效 |
| 越界安全 | 所有几何属性 Clamp | 防止 InverseSigmoid → NaN |
