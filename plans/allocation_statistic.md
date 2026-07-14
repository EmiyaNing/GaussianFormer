# AdaptiveAllocationV4 高斯操作统计与操作着色可视化任务流设计

## 目标

统计 `model/encoder/gaussian_encoder/topk_module/adaptive_allocationv4.py` 中 `AdaptiveAllocationV4` 处理的输入高斯，在一次前向推理或验证流程中分别有多少高斯执行了：

| 操作 | 模块内编号 | 含义 |
| --- | --- | --- |
| keep | `0` | 保留原高斯，不生成新高斯，不修改 opacity |
| clone | `1` | 原高斯保留，并额外生成 1 个 clone child |
| split | `2` | 原高斯被 child 1 替换，并额外追加 child 2 |
| opacity attenuation | `3` | 原高斯保留，但 opacity 被衰减 |

统计对象应以“输入到 `AdaptiveAllocationV4.forward` 的原始 N 个高斯”为单位，而不是以最终输出高斯数量为单位。

在统计基础上，还需要将操作类型映射为固定颜色，并在 `visualize.py` 调用高斯可视化时使用该颜色方案：

| 操作 | 颜色 | 可视化含义 |
| --- | --- | --- |
| keep | 灰色 | 该高斯最终保持原状 |
| clone | 蓝色 | 该高斯触发 clone，并产生 1 个额外 child |
| split | 红色 | 该高斯触发 split，原位置被 child 1 替换并追加 child 2 |
| opacity attenuation | 黄色 | 该高斯 opacity 被衰减 |

这个需求包含两个目标：

```text
1. 统计每类操作的数量和比例。
2. 在高斯可视化结果中，用固定颜色标出每个高斯对应的操作类型。
```

## 关键代码路径理解

`AdaptiveAllocationV4.forward` 的核心流程可以概括为：

```text
输入高斯 N
   |
   v
状态编码 h
   |
   v
allocation_score
   |
   v
TopK 选中 selected_mask
   |
   v
operation_head 得到 4 类 op_id
   |
   v
未选中 TopK 的高斯强制设为 keep
   |
   v
按 op_id 生成 keep / clone / split / attenuation 分支结果
```

最适合统计和记录操作标签的位置是在：

```text
op_logits, op_prob, op_onehot, op_id = compute_operation_routing(...)
```

之后、逐 batch 分支循环开始统计 mask 之前。

原因：

- 此时 `op_id` 已经是最终硬路由结果。
- 未进入 TopK 的高斯已经被强制改成 `KEEP`。
- 后续 `keep_mask`、`clone_mask`、`split_mask`、`atten_mask` 都直接由 `op_id` 得到，统计这里和实际执行分支一致。
- 统计对象仍然是一开始的 N 个高斯，尚未受到 clone/split 追加新高斯的影响。
- 可视化所需的 per-gaussian 操作颜色也可以从同一份 `op_id` 派生，避免统计和可视化使用两套逻辑。

## 统计口径

建议同时保留两种口径，避免后续分析时混淆。

### 1. 原始高斯操作口径

统计输入 N 个高斯各自被路由到了哪个操作：

| 指标 | 定义 |
| --- | --- |
| keep_count | `op_id == KEEP` 的数量 |
| clone_count | `op_id == CLONE` 的数量 |
| split_count | `op_id == SPLIT` 的数量 |
| atten_count | `op_id == ATTEN` 的数量 |
| total_input | 输入高斯数量 N |

这个口径回答的问题是：

```text
有多少原始高斯执行了 keep / clone / split / opacity attenuation？
```

用户当前需求主要对应这一口径。

### 2. 输出高斯数量变化口径

`clone` 和 `split` 都会让最终输出数量增加，但语义不同：

| 操作 | 对原始位置的处理 | 追加数量 | 输出数量贡献 |
| --- | --- | --- | --- |
| keep | 原样保留 | 0 | 1 |
| clone | 原样保留 | 1 | 2 |
| split | child 1 替换原始位置 | 1 | 2 |
| attenuation | opacity 衰减后保留 | 0 | 1 |

因此某个 batch 的输出高斯数量为：

```text
M = N + clone_count + split_count
```

这个口径回答的问题是：

```text
AdaptiveAllocationV4 让高斯数量增加了多少？
```

## keep 的定义边界

当前模块中 keep 来源有两类：

| keep 类型 | 来源 | 是否建议单独统计 |
| --- | --- | --- |
| non_topk_keep | 没有进入 TopK，被强制设为 keep | 建议统计 |
| routed_keep | 进入 TopK 后，operation router 仍选择 keep | 建议统计 |

总 keep 数量为：

```text
keep_count = non_topk_keep_count + routed_keep_count
```

建议在统计结果中同时给出三者：

| 字段 | 含义 |
| --- | --- |
| keep_total | 所有最终为 keep 的高斯 |
| keep_from_non_topk | 未被 TopK 选中而 keep 的高斯 |
| keep_from_router | 被 TopK 选中且 router 选择 keep 的高斯 |

这样可以区分：

- keep 是因为预算没有选中；
- 还是模型主动在候选高斯中选择 keep。

## 低侵入 / 零侵入实现模式分析

上一版方案建议在 `AdaptiveAllocationV4.forward` 内部保存 `input_op_id`、`output_op_id` 和统计信息。这个方案逻辑最直接，但确实会触碰模块内部主流程，尤其是需要在输出高斯组装时同步维护 `output_op_id`，对 `AdaptiveAllocationV4` 的侵入偏大。

新的设计应优先满足：

```text
1. 不改变 AdaptiveAllocationV4.forward 的返回值。
2. 尽量不修改 adaptive_allocationv4.py。
3. 统计和可视化逻辑主要放在 visualize.py 或独立辅助模块中。
4. 如果必须修改 AdaptiveAllocationV4，也只增加可选诊断接口，不改核心计算路径。
5. 所有统计 / 着色逻辑默认关闭，只在可视化脚本显式开启时运行。
6. 正常 train.py / eval.py / 非可视化推理路径不注册 hook、不复算 op_id、不保存 per-gaussian 标签。
```

下面给出几种可选模式。

### 模式 A：零修改 V4，使用 forward hook 复算 op_id

这是最推荐的低风险方案。

核心思想：

```text
不修改 AdaptiveAllocationV4 源码。
在 visualize.py 中给 AdaptiveAllocationV4 注册 forward hook。
hook 读取 forward 输入和输出。
在 hook 中调用模块已有 helper 复算 h、selected_mask、op_id。
再根据 op_id 构造统计结果和 output_op_id。
```

数据流：

```text
visualize.py
   |
   |-- 找到模型中的 AdaptiveAllocationV4 实例
   |-- 注册 forward hook
   v
AdaptiveAllocationV4.forward 正常执行
   |
   v
hook 捕获:
   - 输入 instance_feature / anchor / gaussian
   - 输出 result_anchors / result_gaussian / result_features
   |
   v
hook 复算:
   - h
   - allocation_score
   - selected_mask
   - op_id
   |
   v
hook 生成:
   - allocation statistics
   - input_op_id
   - output_op_id
```

为什么可以复算：

| 信息 | 来源 |
| --- | --- |
| `instance_feature` | `AdaptiveAllocationV4.forward` 输入 |
| `gaussian.scales` | `AdaptiveAllocationV4.forward` 输入 |
| `gaussian.semantics` | `AdaptiveAllocationV4.forward` 输入 |
| `encode_allocation_state` | 模块已有方法 |
| `compute_allocation_topk` | 模块已有方法 |
| `compute_operation_routing` | 模块已有方法 |

优点：

- `adaptive_allocationv4.py` 完全不用改。
- 不改变模型结构、checkpoint、forward 返回值。
- 统计逻辑集中在 `visualize.py` 或单独的诊断工具文件里。
- 适合只在可视化 / 验证时启用。

缺点：

- 会额外执行一次 allocation state encoding、TopK 和 routing，带来少量重复计算。
- 如果训练模式下存在随机层，复算可能和原 forward 不完全一致；但当前 `AdaptiveAllocationV4` 中没有 dropout，且可视化阶段通常是 eval 模式，这个风险较低。
- hook 需要小心处理 DDP 包装后的模型模块查找。

对输出着色的处理：

```text
input_op_id: 直接由复算得到的 op_id。
output_op_id:
  - 前 N 个位置使用 input_op_id。
  - clone 追加 child 的位置标记为 clone。
  - split 追加 child 的位置标记为 split。
  - padding 位置标记为 -1。
```

由于 `AdaptiveAllocationV4.forward` 当前组装顺序是：

```text
前 N 个位置: 原始高斯或被 split child 1 / attenuation 替换后的高斯
追加位置: clone child，然后 split child 2
```

hook 可以根据 `clone_indices`、`split_indices` 在外部重建与输出 M 对齐的 `output_op_id`。

推荐使用场景：

| 场景 | 是否推荐 |
| --- | --- |
| 可视化脚本统计 | 强烈推荐 |
| 临时实验分析 | 强烈推荐 |
| 长期训练日志 | 不建议默认启用；如确实需要，应单独加轻量开关 |

### 模式 B：零修改 V4，运行时包装 compute_operation_routing

核心思想：

```text
不改 adaptive_allocationv4.py。
在 visualize.py 中临时包装 AdaptiveAllocationV4 实例的 compute_operation_routing 方法。
原方法正常执行，wrapper 只截获返回的 op_id。
再配合 forward hook 在 forward 结束后生成统计和 output_op_id。
```

数据流：

```text
visualize.py
   |
   |-- 找到 AdaptiveAllocationV4 实例
   |-- 保存原 compute_operation_routing
   |-- 替换为 wrapper
   v
AdaptiveAllocationV4.forward
   |
   |-- 调用 wrapper
   |-- wrapper 内部调用原方法
   |-- wrapper 记录 op_id / selected_mask
   v
forward hook 读取记录并生成统计
```

优点：

- `adaptive_allocationv4.py` 不需要修改。
- 不需要重复计算 `h`、TopK 和 routing，统计结果与原 forward 完全同源。
- 比模式 A 更省计算。

缺点：

- 运行时 monkey patch 方法，可读性和可维护性弱于 forward hook 复算。
- 如果多线程、多 dataloader worker 或多个同类模块同时存在，需要确保记录不会互相覆盖。
- 如果后续 `AdaptiveAllocationV4` 内部方法名改变，wrapper 会失效。

推荐使用场景：

| 场景 | 是否推荐 |
| --- | --- |
| 单卡可视化 | 推荐 |
| 多卡 DDP 可视化 | 可用，但要更谨慎 |
| 长期稳定工具 | 次推荐，hook 复算更清晰 |

### 模式 C：外部诊断 wrapper / subclass，不改原始 V4

核心思想：

```text
新增一个诊断版模块，例如 DiagnosticAdaptiveAllocationV4。
它继承 AdaptiveAllocationV4。
原始 adaptive_allocationv4.py 尽量不动。
通过配置把 type 从 AdaptiveAllocationV4 切到 DiagnosticAdaptiveAllocationV4。
```

可选实现方式：

| 方式 | 说明 |
| --- | --- |
| subclass 覆写 `compute_operation_routing` | 捕获 `op_id`，主 forward 仍用父类 |
| subclass 增加 hook 辅助属性 | 保存最近一次统计结果 |
| subclass 覆写 `forward` | 能完整维护 `output_op_id`，但会复制大量父类逻辑，不推荐 |

优点：

- 原始 V4 文件基本不动，训练模型逻辑保持干净。
- 诊断逻辑独立成模块，后续可以开关式使用。
- 比 monkey patch 更显式，适合长期维护。

缺点：

- 需要改配置或注册新模块。
- 如果只覆写 `compute_operation_routing`，仍然需要配合 hook 或额外逻辑构造 `output_op_id`。
- 如果覆写 `forward`，容易和父类实现漂移，维护成本较高。

推荐使用场景：

| 场景 | 是否推荐 |
| --- | --- |
| 长期保留的分析功能 | 推荐 |
| 不想污染主模块 | 推荐 |
| 只做一次性可视化 | 不如模式 A 简洁 |

### 模式 D：极小侵入 V4，增加可选 observer 回调

核心思想：

```text
只在 AdaptiveAllocationV4 中加入一个可选 observer。
forward 主逻辑不变。
当 observer 存在时，把 op_id、selected_mask、必要尺寸信息传给外部 observer。
统计和文件输出仍在 visualize.py 或 observer 中完成。
```

侵入点控制在：

```text
op_id 得到之后，调用一次 observer。
```

概念数据流：

```text
AdaptiveAllocationV4.forward
   |
   |-- 正常得到 op_id / selected_mask
   |-- 如果 observer 存在，则通知 observer
   v
后续 forward 主逻辑不变
```

优点：

- 不需要复算。
- 不需要 monkey patch。
- 统计数据与真实 forward 完全一致。
- 对主返回值无影响。

缺点：

- 仍然需要修改 `adaptive_allocationv4.py`。
- 如果 observer 设计不克制，后续可能把诊断逻辑逐渐塞回主模块。

推荐约束：

| 约束 | 原因 |
| --- | --- |
| observer 默认为 `None` | 默认训练 / 推理完全不受影响 |
| observer 只接收 tensor，不做文件 IO | 避免 V4 依赖可视化脚本 |
| observer 中 detach 数据 | 避免保留计算图 |
| observer 不改变任何 forward 输出 | 保持模块接口稳定 |

推荐使用场景：

| 场景 | 是否推荐 |
| --- | --- |
| 需要长期稳定且结果必须同源 | 推荐 |
| 严格要求不修改 V4 | 不适用 |

### 模式 E：离线根据输入输出差异推断操作

核心思想：

```text
完全不读取 op_id。
只比较 allocation 前后的高斯数量、位置、opacity 等变化，尝试反推出 keep / clone / split / attenuation。
```

这个方案不推荐作为主方案。

原因：

| 问题 | 说明 |
| --- | --- |
| clone 和 split 都会增加 1 个输出高斯 | 仅靠数量无法区分 |
| split child 1 会替换原位置 | 原始高斯不再存在，匹配不稳定 |
| clone child 与父高斯位置可能接近 | 最近邻匹配容易误判 |
| attenuation 只改 opacity | 可能和其他数值变化混淆 |
| padding 会进一步干扰输出对齐 | 需要额外 mask |

推荐使用场景：

| 场景 | 是否推荐 |
| --- | --- |
| 没法访问模型内部方法 | 勉强可作为粗略分析 |
| 需要准确统计四类操作 | 不推荐 |
| 需要操作着色可视化 | 不推荐 |

### 推荐优先级

综合侵入性、准确性、实现复杂度，推荐顺序如下：

| 优先级 | 模式 | 侵入 AdaptiveAllocationV4 | 准确性 | 实现复杂度 | 推荐结论 |
| ---: | --- | --- | --- | --- | --- |
| 1 | 模式 A：forward hook 复算 | 无 | 高 | 中 | 最推荐，适合当前需求 |
| 2 | 模式 B：运行时包装 routing 方法 | 无 | 最高 | 中 | 可选，省计算但更脆 |
| 3 | 模式 C：外部诊断 subclass | 无或极低 | 高 | 中高 | 适合长期工具化 |
| 4 | 模式 D：可选 observer | 极低 | 最高 | 低中 | 若允许小改 V4，这是最干净的同源方案 |
| 5 | 模式 E：离线差异推断 | 无 | 低 | 高 | 不建议 |

当前需求建议采用：

```text
首选：模式 A
备选：模式 B
如果后续要长期维护：模式 C 或模式 D
```

## 只在可视化时启用的性能隔离设计

为了避免统计和操作着色逻辑影响正常训练、常规 eval 和普通推理，建议把该功能设计成“可视化专用旁路”。

核心原则：

```text
默认状态:
  - 不注册 forward hook
  - 不 monkey patch 模块方法
  - 不复算 op_id
  - 不保存 input_op_id / output_op_id
  - 不改变 draw_gaussian_params

仅当 visualize.py 中显式开启可视化统计参数时:
  - 注册 collector
  - 注册 hook 或 wrapper
  - 执行统计和颜色映射
  - 在可视化结束后移除 hook / 恢复 wrapper
```

### 开关设计

建议在 `visualize.py` 中使用独立参数控制：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--vis-gaussian-allocation-color` | `False` | 按 keep / clone / split / attenuation 给高斯着色 |
| `--allocation-statistic` | `False` | 输出 allocation 操作统计 |

也可以将两者合并成一个开关：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--vis-gaussian-allocation-stat` | `False` | 同时启用统计和操作颜色 |

推荐拆成两个开关，因为有时只需要统计，不一定需要保存高斯图；有时只想看图，也可以顺便输出轻量统计。

### 启用条件

collector 只应在以下条件满足时创建：

```text
args.vis_gaussian_allocation_color == True
或
args.allocation_statistic == True
```

如果两个参数都为 `False`：

```text
collector = None
不遍历模型查找 AdaptiveAllocationV4
不注册 hook
不创建统计缓存
```

这样正常训练、常规 eval、普通可视化都不会承担任何额外计算和显存开销。

### 生命周期管理

hook / wrapper 的生命周期应严格限制在可视化循环内：

```text
构建模型并加载权重
   |
   v
如果启用 allocation statistic / allocation color:
   注册 collector 和 hook
   |
   v
执行 visualize.py 的 validation / visualization loop
   |
   v
循环结束后:
   remove hook
   清空 collector 中的 GPU tensor 引用
```

关键约束：

| 约束 | 目的 |
| --- | --- |
| 保存 hook handle | 便于结束后调用 remove |
| 每次 iter 后只保留 CPU 数值和必要 CPU label | 防止 GPU tensor 被 collector 长期引用 |
| 不在 collector 中保存 `gaussian` / `feature` 完整对象 | 避免显存无法释放 |
| 如果使用 method wrapper，结束后恢复原方法 | 避免后续流程继续被拦截 |

### no_grad 与 detach 策略

可视化脚本本身通常已经在 `torch.no_grad()` 下运行，但 collector 内仍应遵守以下规则：

| 数据 | 推荐处理 |
| --- | --- |
| 统计计数 | 立即转成 Python int / float |
| `input_op_id` | detach 后按需转 CPU |
| `output_op_id` | detach 后按需转 CPU；绘制前再传入绘图函数 |
| 中间 `h` / `allocation_score` / `op_prob` | 用完即丢，不保存 |

collector 中不应保存任何带计算图的 tensor。

### 显存控制

模式 A 会额外复算一次 state encoding 和 routing。为了限制开销：

```text
1. 只在需要 allocation statistic / allocation color 的 visualize.py 中启用。
2. 复算过程放在 no_grad 语义下。
3. 不保存 h、op_prob、allocation_score 等中间大 tensor。
4. 每个 iter 结束后清理 last input / output 引用。
5. 如果只输出全局统计，不保存 per-gaussian op_id 历史。
```

对于 per-iter 可视化，只需要保留当前 iter 的 `output_op_id`，绘图完成后即可释放。

### 对训练和常规 eval 的影响边界

| 运行入口 | 是否启用 allocation collector | 额外计算 | 额外显存 |
| --- | --- | --- | --- |
| 正常训练 | 否 | 无 | 无 |
| 常规 eval | 否 | 无 | 无 |
| `visualize.py` 但未开启 allocation 参数 | 否 | 无 | 无 |
| `visualize.py --allocation-statistic` | 是 | 有，仅可视化阶段 | 很小，需及时释放 |
| `visualize.py --vis-gaussian-allocation-color` | 是 | 有，仅可视化阶段 | 需要当前 iter 的 label |

因此，功能实现不应写入模型默认 forward 路径，也不应在 config 中默认替换为诊断模块。

## 推荐任务流

### Step 1：确定统计触发范围

先明确统计发生在哪个阶段：

| 场景 | 推荐方式 |
| --- | --- |
| 单次可视化 / 推理 | 在 `visualize.py` 中通过 collector 获取统计 |
| 完整验证集统计 | 每个 validation iter 累加统计，epoch 结束后输出汇总 |
| 训练过程监控 | 默认不启用；如确实需要，应设计单独轻量开关 |

建议优先实现“验证 / 可视化阶段统计”，因为不会影响训练梯度，也更符合当前可视化脚本的使用方式。

### Step 2：获取 batch 级操作标签

若采用低侵入方案，统计不一定要写在 `AdaptiveAllocationV4.forward` 内部。

推荐按模式选择操作标签来源：

| 模式 | 操作标签来源 |
| --- | --- |
| 模式 A | `visualize.py` 的 forward hook 中复算 `op_id` |
| 模式 B | wrapper 捕获 `compute_operation_routing` 的返回值 |
| 模式 C | 诊断 subclass 暴露最近一次 `op_id` |
| 模式 D | observer 接收 `op_id` |

无论采用哪种模式，最终都应得到每个 batch item 的 `input_op_id`。

对每个 batch item 分别统计：

| 字段 | 说明 |
| --- | --- |
| batch_index | 当前 batch 内样本编号 |
| input_gaussians | 输入高斯数量 N |
| selected_topk | TopK 选中的数量 K |
| keep_total | 最终 keep 数 |
| keep_from_non_topk | 未选中 TopK 导致的 keep 数 |
| keep_from_router | TopK 内 router 选择 keep 数 |
| clone | clone 数 |
| split | split 数 |
| opacity_attenuation | attenuation 数 |
| output_gaussians_expected | 预期输出高斯数量，等于 `N + clone + split` |

### Step 3：保存最近一次统计

低侵入设计下，统计结果不一定保存在 `AdaptiveAllocationV4` 本体上，也可以由 `visualize.py` 中的 collector 保存。

推荐保存内容：

```text
allocation_collector.last_stats
  - per_batch: 每个 batch item 的统计表
  - summary: 当前 forward 的合计统计
  - ratios: 各操作占输入高斯数量的比例
  - input_op_id: 输入 N 个高斯的操作标签
  - output_op_id: 与输出 M 个高斯对齐的操作标签
```

注意事项：

- 统计数据应从 tensor 转成普通数值，避免保留计算图。
- 不应影响 forward 的返回接口，避免破坏现有 encoder 调用。
- 统计开关建议放在 `visualize.py` 参数中。
- 如果一直保存 per-gaussian 标签，会增加少量内存；可视化阶段通常可以接受。
- collector 不应作为模型属性长期挂载；若临时挂载，也必须在可视化结束后清理。
- 不需要保存历史 `input_op_id` / `output_op_id`，除非用户明确要求导出 per-gaussian 明细。

### Step 4：在外层验证 / 可视化脚本中聚合

外层脚本需要找到模型中的 `AdaptiveAllocationV4` 实例，并在每次 forward 后读取其统计。

推荐聚合维度：

| 聚合维度 | 用途 |
| --- | --- |
| per_iter | 查看每一帧 / 每个样本的操作分布 |
| global_summary | 查看整个可视化列表或验证集的总体分布 |
| ratio_summary | 查看 keep / clone / split / attenuation 占比 |
| optional scene_summary | 流式或按场景验证时，查看每个 scene 的分布 |

全局合计时应累加原始高斯操作数量：

```text
global_keep += keep_total
global_clone += clone
global_split += split
global_atten += opacity_attenuation
global_input += input_gaussians
global_output_expected += output_gaussians_expected
```

### Step 5：输出统计文件

建议输出到当前 `work_dir` 下，例如：

```text
work_dir/
  allocation_stats/
    allocation_stats_summary.json
    allocation_stats_per_iter.csv
    allocation_stats_report.md
```

各文件职责：

| 文件 | 内容 |
| --- | --- |
| summary.json | 机器可读的全局统计 |
| per_iter.csv | 每个 iter / batch item 的行式记录 |
| report.md | 人类可读的汇总表和说明 |

如果只做最小实现，也可以先只输出一个 Markdown 或 JSON 文件。

## 推荐统计表格式

### 全局汇总表

| Metric | Count | Ratio |
| --- | ---: | ---: |
| input gaussians | total_input | 100% |
| keep total | keep_total | keep_total / total_input |
| keep from non-TopK | keep_from_non_topk | keep_from_non_topk / total_input |
| keep from router | keep_from_router | keep_from_router / total_input |
| clone | clone | clone / total_input |
| split | split | split / total_input |
| opacity attenuation | opacity_attenuation | opacity_attenuation / total_input |
| expected output gaussians | output_gaussians_expected | output_gaussians_expected / total_input |

### 单 iter 明细表

| iter | batch | input | topk | keep | non_topk_keep | routed_keep | clone | split | atten | expected_output |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | N | K | ... | ... | ... | ... | ... | ... | ... |

## 验证逻辑

每个 batch item 都应该满足以下一致性检查：

```text
keep_total + clone + split + opacity_attenuation = input_gaussians
```

```text
keep_from_non_topk + keep_from_router = keep_total
```

```text
selected_topk = keep_from_router + clone + split + opacity_attenuation
```

```text
expected_output_gaussians = input_gaussians + clone + split
```

如果 batch 内不同样本 clone/split 数不同，模块末尾会 padding 到同一长度。统计时应使用 `expected_output_gaussians` 作为真实有效输出数量，而不是 padding 后 tensor 的第二维长度。

## 操作着色可视化设计

### 颜色映射

推荐使用固定、不随随机种子变化的颜色：

| op_id | 操作 | RGB 颜色 | 说明 |
| ---: | --- | --- | --- |
| 0 | keep | `(0.55, 0.55, 0.55)` | 中性灰色，表示保留 |
| 1 | clone | `(0.10, 0.35, 1.00)` | 蓝色，表示扩增覆盖 |
| 2 | split | `(1.00, 0.10, 0.10)` | 红色，表示分裂重构 |
| 3 | opacity attenuation | `(1.00, 0.85, 0.05)` | 黄色，表示透明度衰减 |

颜色应作为 operation visualization 的专用模式，不应复用当前基于 opacity 或 semantic 的自适应颜色逻辑。

### 可视化对象对齐原则

需要特别注意 `AdaptiveAllocationV4` 的输出高斯数量可能大于输入高斯数量：

```text
output_gaussians = input_gaussians + clone_count + split_count
```

而 `op_id` 原始形状对应的是输入 N 个高斯：

```text
op_id: input N
result_gaussian: output M
```

因此有两种可视化口径可选。

#### 口径 A：只可视化输入高斯的操作类型

该口径把颜色赋给原始 N 个高斯，最直接回答：

```text
哪些输入高斯执行了 keep / clone / split / attenuation？
```

优点：

- 统计和颜色完全一一对应。
- 不需要处理 clone child / split child 的颜色继承问题。
- 适合作为第一阶段实现。

限制：

- 如果最终传给绘制函数的是 `result_dict['gaussian']`，它可能已经是 allocation 后的 M 个高斯，不能直接用 N 个颜色数组套上去。
- 需要拿到 allocation 前或 allocation 对齐后的高斯对象，或者在模块内额外保存一个与输出 M 对齐的 operation label。

#### 口径 B：可视化输出高斯，并让 child 继承父操作颜色

该口径把颜色扩展到最终输出 M 个高斯：

| 输出高斯来源 | 推荐颜色 |
| --- | --- |
| keep 原始高斯 | 灰色 |
| clone 原始高斯 | 蓝色 |
| clone child | 蓝色 |
| split child 1 | 红色 |
| split child 2 | 红色 |
| attenuation 后原始高斯 | 黄色 |

优点：

- 可以直接作用于最终 `result_gaussian`，更适合当前 `visualize.py` 中的高斯绘制流程。
- 可视化画面中的所有高斯都有颜色，不会因为 M 和 N 不一致出现缺失。

限制：

- 颜色表示的是“该输出高斯来自哪种输入操作”，不是每个输出高斯自身重新执行了一次操作。
- 需要在 `AdaptiveAllocationV4.forward` 组装输出时同步组装 `result_op_id` 或 `result_operation_color_label`。

推荐优先采用口径 B，因为当前需求是在“对 adaptive_allocationv4 的高斯进行可视化的基础上进行统计”，最终可视化通常会使用 allocation 后的高斯结果。

### 输出 label 对齐规则

若采用口径 B，需要得到两份操作标签。低侵入实现中，这两份标签由 `visualize.py` 中的 collector 在外部保存或构造，不要求 `AdaptiveAllocationV4` 内部维护：

| 标签 | 形状语义 | 用途 |
| --- | --- | --- |
| input_op_id | 输入 N 个高斯的最终路由 | 统计原始操作数量 |
| output_op_id | 输出 M 个高斯对应的父操作类型 | 给最终高斯着色 |

`output_op_id` 的构造逻辑应与高斯组装逻辑保持一致：

| 分支 | 原位置 label | 追加高斯 label |
| --- | --- | --- |
| keep | keep | 无 |
| clone | clone | clone |
| split | split | split |
| attenuation | attenuation | 无 |

如果 batch 内 M 不同并发生 padding，`output_op_id` 也应同步 padding。padding label 不参与统计，也不应被绘制。

建议为 padding 使用独立的 invalid label，例如：

| label | 含义 |
| ---: | --- |
| -1 | padding / invalid gaussian |

这样可以避免 padding 被误认为 keep，因为 keep 的合法编号是 `0`。

### visualize.py 中的功能入口

建议在 `visualize.py` 增加一个独立开关：

```text
--vis-gaussian-allocation-color
```

该开关的语义是：

```text
当可视化 gaussian 时，优先使用 collector 得到的 output_op_id 进行固定操作着色。
```

与已有参数的关系：

| 参数 | 作用 | 优先级建议 |
| --- | --- | --- |
| `--vis-gaussian-allocation-color` | 按 keep/clone/split/attenuation 着色 | 最高 |
| `--vis-gaussian-adaptive-color` | 按已有 adaptive color 逻辑着色 | 次级 |
| 默认语义/opacity 颜色 | 原有绘制方式 | 最低 |

这样可以避免两个颜色模式同时生效时互相覆盖。

### 绘制函数参数传递

当前 `visualize.py` 通过 `draw_gaussian_params` 将可视化参数传给 `save_gaussian` / `save_gaussian_point` 等函数。

建议延续这个结构，新增以下概念性参数：

| 参数 | 含义 |
| --- | --- |
| allocation_color | 是否启用操作着色 |
| allocation_op_ids | 与待绘制 gaussian 对齐的操作标签 |
| allocation_color_map | keep/clone/split/attenuation 到 RGB 的映射 |

绘制函数内部如果发现 `allocation_color` 开启，应直接使用 `allocation_op_ids` 映射颜色，而不是再调用当前 adaptive color 工具。

### 统计与可视化的数据流

采用首选的 hook collector 模式时，整体数据流建议如下：

```text
visualize.py
   |
   |-- 注册 AdaptiveAllocationV4 forward hook
   v
AdaptiveAllocationV4.forward 正常执行
   |
   |-- hook 捕获输入和输出
   |-- collector 外部复算 input_op_id
   |-- collector 外部构造 output_op_id
   |-- collector 外部生成 allocation statistics
   v
visualize.py
   |
   |-- 读取统计并累加
   |-- 读取 output_op_id 并放入 draw_gaussian_params
   v
save_gaussian / save_gaussian_point
   |
   v
按操作颜色绘制高斯
```

低侵入模式下，更推荐让 `visualize.py` 的 collector 直接持有最近一次统计结果，而不是要求模型 `result_dict` 或 `AdaptiveAllocationV4` 保存这些字段。

如果后续要长期维护，也可以考虑模式 C 或模式 D，把诊断信息显式化，但不建议在第一版就大改主模块。

## 可选增强项

### 操作分布直方图

可以将四类操作比例绘制成柱状图：

```text
keep | ########################
clone | #######
split | ###
atten | ####
```

### TopK 内部路由分布

为了分析 operation router 的偏好，建议额外给出 TopK 内部比例：

| 操作 | TopK 内数量 | TopK 内比例 |
| --- | ---: | ---: |
| routed keep | keep_from_router | keep_from_router / selected_topk |
| clone | clone | clone / selected_topk |
| split | split | split / selected_topk |
| attenuation | opacity_attenuation | opacity_attenuation / selected_topk |

这个表能排除 non-TopK keep 对 keep 总比例的稀释影响。

### 保留 per-gaussian 操作标签

如果后续需要可视化哪一个高斯执行了哪种操作，可以额外保存每个输入高斯的 `op_id`。

在当前新需求下，per-gaussian 操作标签已经不只是调试信息，而是操作着色可视化的必要输入。

建议至少保存：

| 标签 | 是否必要 | 原因 |
| --- | --- | --- |
| input_op_id | 必要 | 保证统计口径准确 |
| output_op_id | 推荐必要 | 保证最终高斯可视化颜色与 M 个输出高斯对齐 |

如果只保存 `input_op_id`，则需要额外保证绘制对象也是输入 N 个高斯，否则颜色数量会和高斯数量不一致。

### 图例输出

为了让生成的可视化图片可解释，建议同步输出一个轻量图例说明：

| 颜色 | 操作 |
| --- | --- |
| 灰色 | keep |
| 蓝色 | clone |
| 红色 | split |
| 黄色 | opacity attenuation |

图例可以写入 Markdown 报告，也可以在保存目录中输出单独的 legend 文本文件。若绘图工具支持叠加图例，也可以在图片上加入颜色说明。

## 最小可行实现建议

最小任务闭环如下：

```text
1. 不修改 adaptive_allocationv4.py。
2. 在 visualize.py 中增加 allocation operation color / statistic 开关。
3. 当两个开关都关闭时，不创建 collector，不查找 AdaptiveAllocationV4，不注册 hook。
4. 只有开关开启时，才在 visualize.py 中为 AdaptiveAllocationV4 注册 forward hook。
5. hook 捕获 V4 的输入和输出，并在 no_grad / detach 语义下复算 input_op_id。
6. collector 根据 input_op_id 统计 keep / clone / split / attenuation，并立即转成普通数值。
7. collector 按 V4 输出组装规则构造当前 iter 的 output_op_id。
8. visualize.py 将当前 iter 的 output_op_id 传给高斯绘制函数。
9. 绘制函数按 keep=灰色、clone=蓝色、split=红色、attenuation=黄色进行可视化。
10. 当前 iter 绘制结束后，释放不再需要的 GPU tensor 引用。
11. 可视化循环结束后，remove hook，并清空 collector 缓存。
12. 验证统计一致性，并检查 output_op_id 数量与实际绘制高斯数量一致。
```

这个最小方案对 `AdaptiveAllocationV4` 源文件零侵入，也不改变 `AdaptiveAllocationV4.forward` 的返回值。

在该方案下，正常训练、常规 eval、以及未开启 allocation 参数的 `visualize.py` 都不会调用统计逻辑，也不会增加额外显存占用。

如果后续需要长期稳定、减少 hook 复算成本，可以再升级到模式 B、模式 C 或模式 D。
