# AdaptiveAllocationV5 学习能力分析

目标文件：`model/encoder/gaussian_encoder/topk_module/adaptive_allocationv5.py`

关注问题：希望 AdaptiveAllocationV5 不只是生成更多 Gaussian candidate，而是真的在预测阶段区分以下区域状态：

- high semantic ambiguity
- geometry anisotropy
- coverage insufficiency
- spatial redundancy

## 1. 结论

当前 V5 的结构已经具备一部分“可被主任务 loss 端到端调节”的能力：TopK 内的 clone / split / attenuation 概率会乘到 candidate opacity 上，因此 `OccupancyLoss -> local_aggregate_react -> Gaussian opacity -> risky_gate` 这条梯度路径是存在的。

但它现在还不能可靠地学成上述四类知识，主要原因有六个：

1. `risky_score_head` 基本没有有效梯度。`risky_score` 只用于 `torch.topk` 产生索引，后续没有作为连续权重进入输出。也就是说 selector 不是一个真正被主 loss 优化的风险评分器。
2. 输入 cue 不足。state encoder 只显式看到 `instance_feature + scale cue + semantic cue`，没有显式看到 `xyz / opacity / neighbor density / Gaussian overlap / local coverage`。coverage insufficiency 和 spatial redundancy 都高度依赖局部空间关系，仅靠单个 Gaussian 的 scale 和 semantic entropy 很难判断。
3. 三个 operation probability 不等于四个诊断概念。当前只有 `p_clone / p_split / p_atten`，但需求是四类可解释状态。没有单独的 diagnostic head，也没有让 `p_split` 区分 semantic ambiguity 与 geometry anisotropy。
4. 没有辅助监督或 proxy target。主 occupancy loss 能告诉模型“最终占据预测好不好”，但不会自动告诉它“这块是语义歧义、那块是覆盖不足”。没有 proxy label 时，router 可能学习到降低 loss 的捷径，而不是学习到我们想要的概念。
5. clone / split / attenuation 的贡献量存在 shortcut。clone 会输出 parent + child，split 输出两个 child，atten 输出低 opacity 原位 Gaussian。router 可能偏向贡献更多 opacity 的分支来提升 recall，而不是因为真的识别出了 coverage insufficiency。
6. 当前 full soft candidate bank 会把所有 risky operation 都物化进 renderer，训练显存开销非常大。`allocation_ratio=0.3` 时输出 Gaussian 数从 `N` 变成 `2.2N`，而 local aggregation 的峰值显存主要由 Gaussian-tile overlap 数 `R` 决定；低概率 candidate 仍然会参与 tile 展开、排序和 backward context 保存。

因此，V5 目前更准确的定位是：

```text
TopK candidate generator with soft opacity routing
```

而不是：

```text
semantic / geometry / coverage / redundancy aware allocator
```

如果目标是“真的学习到知识”，不能直接把 full 5-candidate soft bank 当成默认训练形态。更合理的路线是把 V5 升级为显式诊断式、显存预算约束的 allocator：先预测四个 diagnostic score，再由这些 score 影响 TopK selector 和 operation gate；训练时只物化少量必要 candidate，并用 proxy target / ranking loss / straight-through routing 维持可学习性。

## 2. 当前 V5 的实际机制

### 2.1 State encoder

V5 的 state 输入来自：

- `instance_feature`
- geometry cue：`scale(3), mean_scale, log_volume, anisotropy_ratio`
- semantic cue：`entropy, top2 margin, top1 confidence`

对应代码：

- `build_geometry_cue()`：`adaptive_allocationv5.py:145-152`
- `build_semantic_cue()`：`adaptive_allocationv5.py:154-161`
- `encode_allocation_state()`：`adaptive_allocationv5.py:163-172`

这说明当前模块对 semantic ambiguity 和 geometry anisotropy 有最基础的单体 cue：

- semantic ambiguity 可以由高 entropy、低 margin、低 confidence 表示；
- geometry anisotropy 可以由 `max(scale) / min(scale)` 表示。

但这里没有显式输入：

- Gaussian center `means`
- opacity
- normalized position
- 到边界/相机可见区的关系
- 邻居数量
- 邻居 overlap
- 局部 Gaussian density
- 当前 Gaussian 对 GT occupied voxel 的 coverage residual

所以 coverage insufficiency 和 spatial redundancy 的信息入口明显不足。

### 2.2 Risky TopK selector

当前 selector 逻辑是：

```python
risky_score = sigmoid(risky_score_head(h))
topk_idx = topk(risky_score, k=K)
```

对应代码：`adaptive_allocationv5.py:174-183`。

问题是 `risky_score` 后续没有进入输出 Gaussian 的任何连续属性，只用于索引选择。`torch.topk` 的索引选择对 score head 不提供有意义的可微学习路径。forward 中也可以看到：

- `risky_score, selected_mask, topk_idx = self.compute_risky_topk(h, N)`：`adaptive_allocationv5.py:502`
- 后面使用的是 `topk_idx` 和 `selected_mask`，`risky_score` 本身不再参与输出。

因此 `risky_score_head` 很可能长期没有梯度，或者没有直接服务于“哪些 Gaussian 风险更高”的学习目标。state encoder 仍可能通过 selected Gaussian 的 branch/gate 得到梯度，但 TopK selection 的排序准则本身不稳定、不可解释。

这是当前 V5 最关键的学习瓶颈。

### 2.3 Risky operation gate

operation gate 输出三类概率：

```python
p_clone, p_split, p_atten = softmax(risky_gate(h) / temperature)
```

对应代码：`adaptive_allocationv5.py:185-187` 和 `adaptive_allocationv5.py:541-544`。

这三类概率会通过 `_with_effective_opacity()` 乘到 candidate opacity 上：

- clone parent opacity 乘 `p_clone`
- clone child opacity 乘 `p_clone`
- split child opacity 乘 `p_split`
- attenuation candidate opacity 乘 `p_atten`

对应代码：`adaptive_allocationv5.py:189-207` 和 `adaptive_allocationv5.py:559-568`。

这部分是 V5 的有效设计：主任务 loss 可以通过 candidate opacity 回传到 risky gate。因此 `p_clone / p_split / p_atten` 有机会学习“哪类候选对最终 occupancy 更有帮助”。

但它们仍然不是四类诊断量：

- `p_split` 可能因为 semantic ambiguity 高而变高；
- `p_split` 也可能因为 geometry anisotropy 高而变高；
- `p_clone` 可能表示 coverage insufficiency；
- `p_clone` 也可能只是因为 clone parent + child 带来更大的 opacity mass；
- `p_atten` 可能表示 spatial redundancy；
- `p_atten` 也可能表示该 Gaussian 的 semantic 错误、位置错误、或者只是被 TopK 选错后的补救。

没有额外约束时，operation probability 只能解释为“这个 candidate bank 的软贡献偏好”，不能解释为“区域知识分类”。

### 2.4 Candidate bank

TopK 内每个 Gaussian 生成 5 个 candidate：

- clone parent
- clone child
- split child 1
- split child 2
- attenuation candidate

非 TopK Gaussian 直接 pass-through。输出数量为：

```text
N - K + 5K = N + 4K
```

对应代码：`adaptive_allocationv5.py:571-627`。

配置里 `allocation_ratio=0.3`，输入 lifter `num_anchor=25600`，所以 densify 后理论数量约为：

```text
25600 * (1 + 4 * 0.3) = 56320
```

对应配置：

- `adaptive_allocationv5_refine.py:98-100,151-159`
- `adaptive_allocationv5_bg_refine.py:98-100,151-159`

这会增加 local aggregation 的 Gaussian 数量和 tile overlap 压力。`localagg_optimize.md` 已经分析过峰值显存主要由 Gaussian-tile overlap 数 `R` 决定，因此 V5 的候选膨胀需要和 renderer 预算联动观察。

### 2.5 Candidate bank 的训练显存风险

当前 V5 的显存风险不是普通的 tensor 拼接，而是 full soft bank 让所有 operation candidate 同时进入 `GaussianHead -> local_aggregate_react`。这会同时放大：

- 参与 render 的 Gaussian 数 `P`
- Gaussian-tile overlap 数 `R`
- autograd 保存到 backward 的 renderer context
- 每个 candidate 的 means / scale / rotation / opacity / semantic 梯度

`localagg_optimize.md` 中已经给出结论：训练峰值显存主导项是 `R=sum_g V_g`，当前 renderer 会把每个 Gaussian 覆盖的 tile 展开成 `R` 条记录并保存排序相关 context 到 backward。因此 candidate bank 的真实代价应按 `R` 估算，而不能只看 Gaussian 数量。

在当前 V5 中，TopK 内每个 Gaussian 无论 router 概率如何，都会物化以下 candidate：

| candidate | 几何 support | 显存含义 |
| --- | --- | --- |
| clone parent | 与 parent 相同 | 重复一次原 support |
| clone child | 新位置，scale 约 `0.7-1.0` parent | 新增一份相近大小 support |
| split child 1 | 新位置，scale 约 `0.45-0.75` parent | 新增较小 support |
| split child 2 | 新位置，scale 约 `0.45-0.75` parent | 新增较小 support |
| attenuation | 与 parent 相同 | 再重复一次原 support |

所以即使 `p_clone` 或 `p_atten` 很低，它们对应的 candidate 仍然会被 renderer 展开。当前 `_with_effective_opacity()` 还有 `opacity_floor`，低概率 candidate 不会真的变成 0 opacity；这对 gate 梯度有帮助，但对显存是最坏情况，因为每个 candidate 都必须参与 forward/backward。

粗略估算：

```text
P_out = N + 4K = N * (1 + 4 * allocation_ratio)
allocation_ratio = 0.3 -> P_out = 2.2N
```

`R` 的放大不是严格 `2.2x`，因为 split child scale 会变小；但 clone parent 和 attenuation 都复制 parent support，clone child 也常常接近 parent 大小。如果被 TopK 选中的又恰好是大尺度或高风险 Gaussian，`R` 放大可能接近甚至超过 `P` 的放大直觉。更关键的是，renderer context 会保留到 backward，所以这部分开销不是 forward 临时峰值，而是训练全程活跃的高水位。

因此，当前 full soft bank 适合作为功能验证版本，不适合作为长期训练默认版本。真正的 V5+ 应先解决：

```text
不要为了让 gate 可微，就把所有 operation 的完整几何 candidate 都送进 renderer。
```

## 3. 四类知识的现状判断

### 3.1 Semantic ambiguity

当前支持程度：中等。

已有 cue：

- semantic entropy
- top-2 class margin
- top-1 confidence

这些 cue 可以表达单个 Gaussian 的分类不确定性。若一个 Gaussian 的 logits 接近多类混合，则 entropy 会高、margin 会低。

缺口：

1. 这只是预测分布的 ambiguity，不一定是真实区域的 semantic ambiguity。模型不准、语义未校准、类别长尾、opacity 太低都会造成高 entropy。
2. 没有用 GT voxel label histogram 监督 Gaussian 的局部语义纯度。
3. split branch 虽然能输出两个 semantic delta，但没有约束两个 child 变得更纯、且彼此对应不同 semantic mode。
4. `p_split` 同时承载 semantic ambiguity 和 geometry anisotropy，解释上会混在一起。

建议目标：

```text
semantic_ambiguity_score 高
  -> Gaussian 支持域内 GT/pred label histogram entropy 高
  -> split probability 上升
  -> split children 的 semantic entropy 下降
  -> split children 之间有合理的 semantic diversity
```

### 3.2 Geometry anisotropy

当前支持程度：中等偏高，但更像规则 cue，不是知识。

已有 cue：

- `anisotropy_ratio = max(scale) / min(scale)`
- split direction 会结合最大 scale 轴：`get_max_scale_axis()`
- split children scale ratio 固定在约 `[0.45, 0.75]`

对应代码：

- anisotropy cue：`adaptive_allocationv5.py:145-152`
- max-scale axis：`adaptive_allocationv5.py:36-60`
- split branch：`adaptive_allocationv5.py:267-306`

优点是 split branch 的几何操作与 anisotropy 有天然匹配关系。

缺口：

1. 高 anisotropy 不一定是坏事。road、wall、fence、sidewalk 等结构天然是长条或薄片。盲目把 anisotropy 当风险会破坏合理的大尺度结构。
2. 只有单体 scale，没有局部 occupied voxel 的几何分布。无法判断 anisotropy 是结构真实形状，还是 Gaussian 过度拉伸。
3. 没有监督 `p_split` 与 bad anisotropy/mismatch 对齐。

建议把目标从简单 `geometry_anisotropy` 改成更有意义的：

```text
geometry_anisotropy_mismatch
```

也就是：Gaussian 自身很各向异性，且它覆盖的真实/预测 occupied support 并不支持这种拉伸，或者该拉伸导致局部 semantic/occupancy error。

### 3.3 Coverage insufficiency

当前支持程度：低。

当前模块没有显式 coverage cue。state encoder 看不到周围是否有未覆盖 occupied voxel，也看不到当前区域的 Gaussian density 是否不足。

主任务 occupancy loss 可以间接鼓励 clone：如果 clone child 增加某些 voxel 的正确类别贡献，`p_clone` 会得到正向梯度。但这条信号是结果级的，不会自动形成“coverage insufficiency score”。

缺口：

1. inference 阶段没有 GT occupied voxel，必须依靠当前特征和 Gaussian 分布预测 coverage demand。
2. 当前输入缺少 local Gaussian coverage proxy，例如 nearby Gaussian count、accumulated opacity、nearest uncovered residual、local density。
3. clone direction 是自由学习方向，只乘 mean scale，未显式朝向 coverage residual。
4. `p_clone` 可能因为 opacity mass 变大而被偏好，而不是真的识别了 coverage insufficiency。

建议目标：

```text
coverage_insufficiency_score 高
  -> 当前 Gaussian 附近存在低 accumulated density / 未覆盖 occupied support
  -> clone probability 上升
  -> clone child 朝 coverage residual barycenter 或低密度方向移动
  -> clone 后局部 recall / coverage gain 上升
```

### 3.4 Spatial redundancy

当前支持程度：低。

atten branch 可以降低 opacity：

- `m = m_min + (1 - m_min) * sigmoid(atten_opacity_head(h))`
- attenuation candidate 再乘 `p_atten`

对应代码：`adaptive_allocationv5.py:343-360`。

但 spatial redundancy 本质上是局部集合属性：需要知道附近是否已经有足够多、语义相近、空间高度重叠的 Gaussian。当前 V5 的 state encoder 没有邻居信息。

缺口：

1. 没有 Gaussian-Gaussian overlap cue。
2. 没有 local density / duplicate score。
3. 没有“移除或降低该 Gaussian 后预测不变”的冗余目标。
4. `p_atten` 可能学成“坏 Gaussian 抑制器”，不一定是 redundancy detector。

建议目标：

```text
spatial_redundancy_score 高
  -> 附近有多个语义相近且 support 重叠的 Gaussian
  -> 当前 Gaussian 的 marginal contribution 低
  -> attenuation probability 上升
  -> 降低 opacity 后 occupancy loss 不上升或更低
```

## 4. 为什么主 loss 不足以自动学到四类知识

当前训练配置主要使用 `OccupancyLoss`：

- CE 权重为 10
- Lovasz 权重为 1
- `use_sem_geo_scal_loss=False`

对应配置：`adaptive_allocationv5_refine.py:33-54` 和 `adaptive_allocationv5_bg_refine.py:33-54`。

`GaussianHead` 会把 Gaussian 渲染/聚合到 voxel prediction，再交给 occupancy loss：

- local aggregator：`gaussian_head.py:160-169`
- output `pred_occ`：`gaussian_head.py:193-204`
- CE / Lovasz：`occupancy_loss.py:113-154`

这条路径优化的是最终 voxel 分类质量，不是 Gaussian 级别的原因归因。对 V5 来说，主 loss 只能回答：

```text
这个输出 Gaussian bank 是否让 voxel prediction 更接近 GT？
```

它不能直接回答：

```text
为什么这个区域错？
是 semantic ambiguity？
是 geometry anisotropy mismatch？
是 coverage insufficiency？
还是 spatial redundancy？
```

没有显式诊断头和 proxy target 时，模型可能学到如下 shortcut：

- 高 `p_clone`：因为 clone 带来更多 opacity mass，而不是因为 coverage 不足；
- 高 `p_split`：因为 split children 的 semantic delta 更容易修正局部类别，而不是因为几何或语义应当拆分；
- 高 `p_atten`：因为该 Gaussian 会伤害 loss，而不是因为它空间冗余；
- TopK selector：由于 score head 没有连续梯度，选择规则可能更多来自随机初始化和 state encoder 的间接漂移。

## 5. 建议的 V5+ 结构

### 5.1 增加显式四诊断头

建议在 `allocation_state_encoder` 后新增：

```text
diagnostic_head(h) -> [
  semantic_ambiguity_score,
  geometry_anisotropy_score,
  coverage_insufficiency_score,
  spatial_redundancy_score,
]
```

这四个 score 应该被缓存用于可视化：

```text
latest_diagnostic_scores
latest_diagnostic_targets
latest_selected_mask
latest_risky_operation_prob
```

这样预测阶段可以直接画出四张 Gaussian-level heatmap，而不是用 `p_clone / p_split / p_atten` 反推含义。

### 5.2 扩展 state 输入

建议 state 输入从：

```text
instance_feature + scale cue + semantic cue
```

扩展为：

```text
instance_feature
+ normalized xyz
+ opacity
+ scale / log_volume / anisotropy_ratio
+ semantic entropy / margin / confidence
+ local neighbor cue
+ local coverage cue
+ local redundancy cue
```

推荐新增的 cue：

| cue | 作用 |
| --- | --- |
| normalized xyz | 让模型知道远近、边界、z 高度等空间先验 |
| opacity | 判断低贡献、高贡献、可衰减程度 |
| neighbor_count | spatial redundancy / sparse coverage 的基础 |
| mean_neighbor_distance | 判断局部稀疏或拥挤 |
| overlap_sum | 判断 Gaussian-Gaussian support 是否重叠 |
| semantic_neighbor_agreement | 判断冗余是否来自相同语义，还是边界混合 |
| local_density_mass | 判断局部 accumulated opacity 是否不足或过量 |
| boundary_distance | 处理 pc_range 边界 clamp 引入的异常 |

coverage 和 redundancy 是局部集合属性，必须引入邻域统计；否则模型只能在单体 Gaussian 上猜。

### 5.3 让 selector 可学习

当前 `risky_score_head` 的最大问题是没有连续梯度。至少需要做其中一种改造：

方案 A：把 selected risky score 乘到 candidate opacity。

```text
selected_risk = risky_score.gather(topk_idx)
candidate_opacity *= selected_risk
```

这样 score head 对 TopK 内样本有梯度。为了不让 selected Gaussian 被完全抹掉，可以设计为：

```text
risk_weight = risk_floor + (1 - risk_floor) * selected_risk
```

方案 B：TopK 仍 hard，但增加风险排序辅助 loss。

训练期构造四类 proxy target 后：

```text
risk_target = max(
  semantic_ambiguity_target,
  geometry_anisotropy_target,
  coverage_insufficiency_target,
  spatial_redundancy_target,
)
```

用 BCE/MSE/ranking loss 监督 `risky_score`。

方案 C：使用 soft top-k / straight-through k-hot selector。

这会更复杂，但能让 selector 在预算约束下获得更平滑的梯度。

短期优先建议：方案 A + 方案 B。实现成本低，行为也容易 debug。

### 5.4 用诊断量约束 operation gate

可以让三类 operation logit 显式依赖四个诊断 score：

```text
clone_logit_prior = + coverage_insufficiency - spatial_redundancy
split_logit_prior = + semantic_ambiguity + geometry_anisotropy
atten_logit_prior = + spatial_redundancy - coverage_insufficiency
```

再叠加一个 learnable residual：

```text
operation_logits = prior_logits + residual_mlp(h)
```

这样既保留可学习性，也给 operation 概率一个可解释的 inductive bias。

注意：semantic ambiguity 和 geometry anisotropy 都可能走 split，但它们应当在 diagnostic score 上分开，而不是都塞进一个 `p_split`。

### 5.5 显存约束下的 candidate materialization

为了控制训练显存，V5+ 不建议继续默认物化完整 5-candidate soft bank。推荐把“诊断学习”和“candidate 渲染”拆开：

```text
diagnostic / router:
  可以输出 soft score，用 auxiliary loss 和 prior KL 学习。

renderer materialization:
  每个 selected Gaussian 只物化 1 个 operation 的必要 candidate，
  或最多物化 Top-M operation candidate。
```

#### 方案 A：Hard / straight-through 单操作物化

对 TopK Gaussian 仍计算 soft operation probability，但 forward 只选择一个 operation 进入 renderer：

```text
op_id = argmax(p_clone, p_split, p_atten)

clone:
  输出 parent + clone child

split:
  输出 split child 1 + split child 2

atten:
  输出 attenuated parent
```

输出数量从当前：

```text
N - K + 5K = N + 4K
```

下降为：

```text
N - K + 2K_clone + 2K_split + K_atten
= N + K_clone + K_split
<= N + K
```

当 `allocation_ratio=0.3` 时，Gaussian 数上界从 `2.2N` 降到 `1.3N`。更重要的是，clone parent / attenuation 这类重复 support 不会同时进入 renderer，`R` 也会大幅下降。

为了保留 gate 可学习性，可以使用：

- straight-through one-hot gate；
- operation prior KL；
- diagnostic target supervision；
- branch outcome loss；
- 温度退火，从 soft 诊断到 hard materialization。

这条路线牺牲了“所有 expert 都通过主 loss 接收梯度”的便利，但显存收益最大，也更接近推理时希望看到的明确操作语义。

#### 方案 B：Top-M operation bank

如果担心 hard op 早期选错，可以只物化概率最高的 Top-M operation：

```text
M=2:
  每个 selected Gaussian 最多物化两类 operation。
```

例如只保留 `p_clone / p_split / p_atten` 最高的两类，第三类不进 renderer。这样训练早期仍保留一定探索，但比 full 5-candidate bank 省很多。

进一步可以按阶段退火：

```text
warmup: M=2
main training: M=1
fine-tune / visualization: optional full bank on few frames
```

#### 方案 C：R-budget-aware operation choice

只限制 TopK 数量不够，因为大尺度 Gaussian 的 tile cost 可能远高于普通 Gaussian。应该给每个 operation 估计 tile cost：

```text
cost_clone = V_parent + V_clone_child
cost_split = V_split_child_1 + V_split_child_2
cost_atten = V_parent
```

然后在 selector / router 中加入预算：

```text
expected_cost =
  p_clone * cost_clone
+ p_split * cost_split
+ p_atten * cost_atten

L_budget = relu(sum_selected expected_cost - R_budget)
```

如果使用 hard materialization，也可以在选择 TopK 时使用 value / cost：

```text
priority = risk_score / sqrt(expected_tile_cost + eps)
```

这样 selector 会倾向选择“收益高、tile 代价可接受”的 Gaussian，而不是把预算花在少数大尺度背景 Gaussian 上。

#### 方案 D：Effective-opacity pruning

在 renderer 前过滤极低有效 opacity candidate：

```text
keep_candidate = effective_opacity > tau
```

但这与当前 `opacity_floor` 有冲突。若要使用 pruning，应把 `opacity_floor` 作为训练早期的可选项，而不是长期强制每个低概率 candidate 参与 renderer。更稳妥的做法是：

```text
diagnostic / router loss 保留 soft probability 梯度；
renderer 只保留 Top-M 或超过 opacity 阈值的 candidate。
```

不建议单独依赖 opacity pruning 解决问题，因为早期概率未校准，阈值可能误删有价值 candidate；它更适合作为 hard/ST 或 Top-M materialization 的补充。

#### 方案 E：把 full bank 降级为 debug 模式

full 5-candidate bank 的价值主要在于验证每个 branch 的可微贡献和可视化 candidate 类型。建议将它定位为：

```text
materialization_mode="full_bank_debug"
```

只用于：

- 小 batch / 少量 frame 可视化；
- branch 梯度 sanity check；
- 比较 operation outcome；
- 离线分析。

正式训练默认应使用：

```text
materialization_mode="hard_st"
```

或：

```text
materialization_mode="topm", topm=1/2
```

## 6. 训练期 proxy target 设计

### 6.1 Semantic ambiguity target

训练期可以用 Gaussian support 内的 GT voxel label histogram 构造：

```text
semantic_ambiguity_target
  = normalized_entropy(label_histogram)
```

或：

```text
1 - max_class_fraction
```

如果某个 Gaussian 覆盖多个语义类别，target 高；如果几乎只覆盖一个类别，target 低。

也可以结合 prediction uncertainty：

```text
target = alpha * GT_hist_entropy + (1 - alpha) * pred_entropy
```

### 6.2 Geometry anisotropy mismatch target

不要简单把 `max(scale) / min(scale)` 当作坏目标。建议 target 结合误差和局部几何：

```text
raw_anisotropy = log(max(scale) / min(scale))
support_error = local CE / misclassified voxel ratio
support_shape_mismatch = mismatch between Gaussian axis and GT occupied covariance

geometry_anisotropy_target = raw_anisotropy * support_error_or_mismatch
```

这样可以避免把 road/wall 这类合理长条结构全部打成高风险。

### 6.3 Coverage insufficiency target

训练期可以先计算每个 occupied voxel 的 current coverage：

```text
coverage(v) = sum_i opacity_i * exp(-0.5 * mahalanobis_i(v))
```

低 coverage 的 occupied voxel 视为 uncovered residual：

```text
uncovered(v) = 1[GT(v) != empty] * 1[coverage(v) < tau]
```

再把 uncovered residual 分配给附近 Gaussian：

```text
coverage_insufficiency_target(i)
  = sum_{v near i} uncovered(v) * affinity(i, v)
```

同时可构造 clone direction target：

```text
direction_target(i)
  = normalize(weighted_barycenter(uncovered voxels near i) - mean_i)
```

这能让 clone child 朝真正缺覆盖的位置移动，而不是任意漂移。

### 6.4 Spatial redundancy target

可以用 Gaussian-Gaussian overlap + semantic similarity 构造：

```text
overlap_ij = exp(-0.5 * mahalanobis_i(mean_j))
semantic_sim_ij = dot(softmax(sem_i), softmax(sem_j))
redundancy_i = sum_j overlap_ij * semantic_sim_ij * opacity_j
```

再结合 marginal contribution：

```text
spatial_redundancy_target(i) 高
  if redundancy_i 高
  and local coverage 已充分
  and local semantic purity 高
```

这样可以区分：

- 真冗余：多个相同语义 Gaussian 重叠覆盖同一区域；
- 语义边界：多个不同语义 Gaussian 接近，但不应简单 atten。

## 7. 辅助 loss 建议

建议引入轻量辅助损失，而不是只依赖 occupancy loss。

### 7.1 Diagnostic supervision

```text
L_diag =
  BCE/MSE(semantic_ambiguity_score, semantic_ambiguity_target)
+ BCE/MSE(geometry_anisotropy_score, geometry_anisotropy_target)
+ BCE/MSE(coverage_insufficiency_score, coverage_insufficiency_target)
+ BCE/MSE(spatial_redundancy_score, spatial_redundancy_target)
```

### 7.2 Risk selector supervision

```text
risk_target = max(four diagnostic targets)
L_risk = ranking_loss(risky_score, risk_target)
```

ranking loss 比普通 BCE 更适合 TopK selector，因为 selector 关心排序而不是绝对概率。

### 7.3 Operation prior KL

构造 soft operation target：

```text
clone_target = coverage_insufficiency_target
split_target = max(semantic_ambiguity_target, geometry_anisotropy_target)
atten_target = spatial_redundancy_target
op_target = normalize([clone_target, split_target, atten_target])
```

然后：

```text
L_op = KL(op_target || p_clone/p_split/p_atten)
```

这个 loss 权重应较小，避免压死主任务自适应。

### 7.4 Operation outcome regularization

可以增加更贴近行为的约束：

- split 后两个 child 的 semantic entropy 应低于 parent；
- split child 间 semantic distribution 不应完全相同；
- split 后 scale / anisotropy 应下降；
- clone child 应提升 uncovered residual coverage；
- redundancy 高的 Gaussian attenuation 后 local prediction 不应明显变差。

这些 outcome loss 不一定第一阶段全做，可以按风险逐步打开。

## 8. 可观测性与实验检查

### 8.1 必做 sanity check

1. 先记录 densify 前后的 `P_out / P_in`、renderer `R`、`R/P`、peak memory。如果 full bank 已经让 `R` 不可控，后续诊断学习实验没有意义。
2. 检查 `risky_score_head.weight.grad` 是否长期为 `None` 或接近 0。若是，说明 selector 当前没有被学到。
3. 记录 `p_clone / p_split / p_atten` 的均值、熵、最大值分布，观察是否早期塌缩。
4. 记录 `opacity_floor` clamp 命中比例。如果大量 probability 被 clamp 到 floor，对 gate 的梯度解释会失真，同时也会强制低概率 candidate 继续消耗 renderer 显存。
5. 打开 `cache_router_stats=True` 做可视化。当前两个 V5 配置都设为 `False`，不利于分析。
6. 记录 TopK 与四类 proxy target 的 enrichment：TopK 内 target 均值应显著高于 non-TopK。

### 8.2 推荐指标

| 指标 | 目的 |
| --- | --- |
| `AUC(score_sem_amb, target_sem_amb)` | 诊断 semantic ambiguity 是否有效 |
| `AUC(score_geo_aniso, target_geo_aniso)` | 诊断 bad anisotropy 是否有效 |
| `AUC(score_cov_insuf, target_cov_insuf)` | 诊断 coverage insufficiency 是否有效 |
| `AUC(score_redund, target_redund)` | 诊断 spatial redundancy 是否有效 |
| `TopK target enrichment` | selector 是否选择真正高风险 Gaussian |
| `corr(p_clone, target_cov_insuf)` | clone 是否对齐 coverage |
| `corr(p_split, max(target_sem, target_geo))` | split 是否对齐语义/几何拆分需求 |
| `corr(p_atten, target_redund)` | attenuation 是否对齐冗余 |
| `coverage gain after clone` | clone 是否真的补覆盖 |
| `purity gain after split` | split 是否真的提升语义纯度 |
| `mIoU / CE / Lovasz` | 主任务收益 |
| `P_out / P_in` | candidate bank 对 Gaussian 数量的放大 |
| `R / R per Gaussian / peak memory` | renderer 成本 |
| `R_out / R_baseline` | densify 是否造成不可接受的 tile overlap 放大 |
| `materialized_ops_per_topk` | 每个 TopK Gaussian 实际送入 renderer 的 operation 数 |
| `dropped_candidate_ratio` | Top-M / pruning 策略丢弃了多少 candidate |

### 8.3 消融路线

建议按下面顺序消融：

1. 当前 V5 baseline。
2. 当前 V5 + `cache_router_stats=True` + renderer `R/P/peak memory` 观测，确认 full bank 的真实代价。
3. V5 + hard/ST 单操作物化，对比 `P_out/R/peak memory/mIoU`。
4. V5 + Top-M operation bank，比较 `M=2` 与 `M=1`。
5. V5 + selected `risky_score` 参与 opacity 或 risk ranking loss，验证 selector 梯度。
6. V5 + 显式 `xyz / opacity / neighbor overlap` cue。
7. V5 + 四诊断 head，仅缓存不加 loss，观察 score 分布。
8. V5 + diagnostic target supervision。
9. V5 + operation prior KL。
10. V5 + operation outcome regularization。

每一步都比较主任务指标、四诊断指标和 renderer 成本，避免只提升解释性但损害 occupancy。

## 9. 优先级建议

### P0：先把 candidate bank 显存账本打出来

这是当前最紧急的问题。先不要急着加更多诊断头或 proxy loss，而是确认：

```text
P_in
P_out
P_out / P_in
R_before_densify
R_after_densify
R_after / R_before
R / P
peak allocated / reserved memory
materialized_ops_per_topk
```

如果 full soft bank 已经让 `R` 或 peak memory 接近不可训练区间，后续结构应该先切到 memory-bounded materialization。

### P1：把 full bank 改成可选 debug 模式

当前 full bank 的优点是梯度路径直观，但训练成本过高。建议新增配置：

```python
materialization_mode = "full_bank"  # debug only
materialization_mode = "hard_st"
materialization_mode = "topm"
topm = 1 or 2
```

正式训练优先使用 `hard_st` 或 `topm=1/2`。full bank 只用于少量 frame 的可视化和梯度 sanity check。

### P2：先确认 selector 是否真的在学

这是最小但最重要的检查。当前代码结构下，`risky_score_head` 很可能没有有效梯度。建议先在一次训练 backward 后打印：

```text
risky_score_head.weight.grad norm
risky_gate.weight.grad norm
clone/split/atten branch grad norm
```

如果 score head grad 为 0，不要继续假设 TopK selector 已经学会了风险排序。

### P3：给 selector 连续梯度

把 selected `risky_score` 作为 risk intensity 乘到 TopK candidate contribution，或加入 risk ranking loss。否则 TopK 只是 hard index 操作。

如果采用 hard/ST materialization，selector 和 operation gate 更需要 auxiliary / ranking loss，否则只有被 hard 选中的分支能从主 loss 得到直接反馈。

### P4：补充局部空间 cue

coverage insufficiency 和 spatial redundancy 都不是单 Gaussian 属性。至少增加：

- normalized xyz
- opacity
- neighbor count
- overlap sum
- semantic neighbor agreement
- local density mass

### P5：增加四诊断 head 和缓存

即使第一阶段不加 loss，也应该让模型输出四个 score，方便可视化和统计。没有显式 score，就无法判断模型是否学到了我们关心的知识。

### P6：加入训练期 proxy target

用 GT voxel 和当前 Gaussian bank 构造软标签，监督四诊断 head，并用小权重约束 operation gate。这样才能把“想要的概念”注入训练，而不是寄希望于主任务 loss 自发分解。

### P7：把 operation 结果也纳入约束

最终应验证：

- split 是否提升 semantic purity / 降低 bad anisotropy；
- clone 是否提升 coverage；
- atten 是否减少 redundancy 且不伤害局部预测。

## 10. 推荐的最终语义映射

建议把 V5+ 的解释口径固定为：

```text
semantic_ambiguity_score:
  这个 Gaussian 支持域是否混有多个语义模式。

geometry_anisotropy_score:
  这个 Gaussian 的几何拉伸是否可能是不合适的过度覆盖，而不只是合法长条结构。

coverage_insufficiency_score:
  当前局部是否缺 Gaussian support，需要 clone / expansion。

spatial_redundancy_score:
  当前局部是否已有相似 Gaussian 充分覆盖，该 Gaussian 的边际贡献低，需要 attenuation。
```

operation 概率解释为：

```text
p_clone:
  对 coverage insufficiency 的响应。

p_split:
  对 semantic ambiguity 或 geometry anisotropy mismatch 的响应。

p_atten:
  对 spatial redundancy 或 harmful contribution 的响应。
```

但注意：operation 概率不是诊断量本身。诊断量应独立输出，operation 概率只是根据诊断量采取的动作。

## 11. 总结

当前 AdaptiveAllocationV5 的 soft candidate bank 是一个合理的第一步：它让 clone / split / attenuation 都能通过 opacity contribution 接收主任务梯度，避免了 V4 hard routing 中 keep 分支吞噬操作概率的问题。

但是，若目标是让模块“真的学到知识”并且能训得动，当前实现还缺四件关键东西：

1. 可学习的 selector：`risky_score` 必须进入连续计算图或接受排序监督。
2. 足够的信息入口：coverage 和 redundancy 必须引入局部空间/邻域 cue。
3. 显式诊断监督：四个概念需要独立 score、proxy target 和可视化指标。
4. 显存预算约束：不能默认把 clone / split / attenuation 的完整 candidate bank 全部物化进 renderer。

推荐将下一版目标定义为：

```text
AdaptiveAllocationV5+
  = differentiable risky selector
  + four diagnostic heads
  + local spatial coverage/redundancy cues
  + diagnosis-conditioned clone/split/atten gate
  + memory-bounded candidate materialization
  + lightweight proxy supervision
```

更具体地说，full 5-candidate bank 应该降级为 debug / visualization 模式；正式训练应优先采用 hard/ST 单操作物化、Top-M operation bank 或 R-budget-aware routing。这样它才有机会在预测阶段稳定地区分 semantic ambiguity、geometry anisotropy、coverage insufficiency 和 spatial redundancy，而不是用一组难以解释、且训练显存昂贵的 candidate opacity 权重硬撑。
