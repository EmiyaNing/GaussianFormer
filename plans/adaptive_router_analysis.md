# AdaptiveAllocationV4 Router 偏向 keep 的原因分析

## 现象复盘

根据你给出的统计结果：

| 指标 | 数量 | 占输入比例 | 占 TopK 比例 |
| --- | ---: | ---: | ---: |
| input gaussians | 27345 | 100.00% | - |
| keep from non TopK | 5470 | 20.00% | - |
| keep from router | 20548 | 75.14% | 93.93% |
| clone | 844 | 3.09% | 3.86% |
| split | 328 | 1.20% | 1.50% |
| opacity attenuation | 155 | 0.57% | 0.71% |

这里真正异常的不是总 keep 高，而是 **TopK 内部 router 仍然把 93.93% 的候选高斯判成 keep**。因为 non-TopK keep 只占 20.00%，说明这次统计中 TopK 候选约为：

```text
selected_topk = 20548 + 844 + 328 + 155 = 21875
selected_topk / input = 21875 / 27345 ~= 80.00%
```

也就是说，这次运行里大部分高斯已经获得了被修改的资格，但 router 自身几乎没有真正选择 clone/split/attenuation。

## 代码路径对应关系

`AdaptiveAllocationV4` 的流程分成两级：

1. `score_head` 计算 `allocation_score`，再用 `allocation_ratio` 做 TopK。
2. `operation_head` 对 TopK 内高斯做 4 类 argmax：`keep / clone / split / atten`。
3. 未进入 TopK 的高斯被强制设置为 keep。

关键代码位于 `model/encoder/gaussian_encoder/topk_module/adaptive_allocationv4.py`：

```python
allocation_score = torch.sigmoid(self.score_head(h)).squeeze(-1)
K = max(1, int(self.allocation_ratio * N))
_, topk_idx = torch.topk(allocation_score, k=K, dim=1)
```

```python
op_logits = self.operation_head(h)
op_prob = op_logits.softmax(dim=-1)
op_id = op_prob.argmax(dim=-1)
op_id = torch.where(selected_mask, op_id, torch.full_like(op_id, KEEP))
```

后续真正执行分支时，直接由 `op_id` 得到 boolean mask：

```python
keep_mask = (cur_op_id == KEEP)
clone_mask = (cur_op_id == CLONE)
split_mask = (cur_op_id == SPLIT)
atten_mask = (cur_op_id == ATTEN)
```

因此统计中的 `keep_from_router` 基本等价于：进入 TopK 后，`operation_head(h).softmax(...).argmax` 的第 0 类胜出。

## 主要原因

### 1. `operation_head` 缺少显式操作监督或比例约束

当前配置的 loss 只有 occupancy 相关监督，没有看到针对 operation router 的目标项，例如：

- 监督某个高斯应该 clone/split/attenuate 的 pseudo label；
- 约束 clone/split/atten 的最小占比；
- 惩罚 keep 概率过高；
- 鼓励 TopK 内 `p_keep` 与 `allocation_score` 保持一致。

在这种情况下，router 只能通过最终 occupancy loss 的间接收益学习操作选择。但 keep 是最保守、扰动最小的操作，而 clone/split/atten 都会改变输出高斯分布、数量、opacity、feature、semantic。若没有额外正则鼓励探索或约束操作分布，训练很容易收敛到“保持原状最稳定”的策略。

### 2. straight-through onehot 被计算了，但没有参与输出

`compute_operation_routing()` 中确实构造了：

```python
op_onehot = op_onehot_hard.detach() - op_prob.detach() + op_prob
```

但在 `forward()` 后续逻辑里，实际使用的是 `op_id` 派生出来的 `clone_mask / split_mask / atten_mask`，没有用 `op_onehot` 去加权 candidate bank 或参与任何辅助 loss。这样会带来一个严重问题：

- `argmax` 和 boolean mask 本身不可导；
- `op_prob` 没有进入最终输出；
- `op_onehot` 也没有进入最终输出；
- 因此 `operation_head` 很可能拿不到有效梯度。

如果这一判断成立，router 的类别偏好主要来自初始化、checkpoint 中已有参数、输入分布和随机线性层的偶然偏置，而不是由训练稳定学到的策略。若 keep logit 一开始略占优，它会长期占优。

### 3. TopK 选择和操作选择是解耦的，TopK 不等于必须修改

`score_head` 只决定哪些高斯进入候选集合，不决定这些高斯一定要 clone/split/atten。进入 TopK 后仍然可以被 router 判为 keep。

你们这次统计中 TopK 约为 80%，说明 `allocation_ratio` 或实际统计口径使大多数高斯进入候选集合。但 TopK 扩大只会增加 router 可选择的样本数；如果 router 本身偏向 keep，扩大 TopK 反而会让 `keep_from_router` 数量显著上升。

### 4. 配置中没有显式传入 `allocation_ratio`

当前 `config/adaptive_allocation/adaptive_allocationv4.py` 的 `densify_layer` 只传入了 `feat_embed_dim / semantic_dim / pc_range / scale_range / unit_xyz`，没有显式设置 `allocation_ratio`。代码默认值是 0.4。

但你给出的数字反推 TopK 比例约为 0.8。这说明至少需要检查运行配置、继承后的最终 config、checkpoint 或统计口径：

```text
non_topk_keep / input = 5470 / 27345 ~= 20.00%
selected_topk / input ~= 80.00%
```

这个问题不直接导致 TopK 内 keep 过高，但会影响 keep 来源的解释。如果实际 TopK 是 80%，那么“大部分 keep”主要不是预算没选中造成的，而是 router 主动选择 keep 造成的。

### 5. clone/split/atten 的短期风险比 keep 更高

从分支实现看：

- clone 会追加 child，并修改 child 的位置、scale、opacity、semantic、feature；
- split 会用 child 1 替换原始高斯，并追加 child 2；
- atten 会降低 opacity。

这些操作都可能在训练早期破坏已有 Gaussian 表达。尤其当前模型从 `raydn_r50_flash_704_bs2_seq_428q_nui_60e.pth` 加载已有权重时，原始高斯/特征可能已经能提供较稳定的 occupancy 表达。相比之下，keep 基本不引入额外风险，所以在只看最终 occupancy loss 的情况下，keep 更容易成为低风险解。

## 最可能的结论

这组统计更像是 **operation router 没有被有效训练，或者没有被足够强的目标约束去选择非 keep 操作**，而不是 TopK 预算太小。

证据是：

1. non-TopK keep 只有 5470，占 20.00%；
2. TopK 内 routed keep 有 20548，占 TopK 的 93.93%；
3. `operation_head` 输出经过 hard argmax 后直接变成 mask；
4. 代码中计算出的 ST `op_onehot` 没有参与输出或 loss；
5. 当前 loss/config 没有看到 operation-level 监督、分布约束或反 keep 正则。

## 建议验证

建议在同一批数据上额外统计以下信息：

1. `op_logits.mean(dim=(0,1))` 和 `op_prob.mean(dim=(0,1))`，确认 keep logit/prob 是否系统性最高。
2. 只在 TopK 内统计 `op_prob` 均值，而不只看 argmax 后的 `op_id`。
3. 训练时打印 `operation_head.weight.grad.abs().mean()` 和 `score_head.weight.grad.abs().mean()`。如果接近 0 或为 None，就能验证 router/topk head 基本没有有效梯度。
4. 打印最终 merged config 中 `allocation_ratio`，确认为什么统计反推约为 0.8，而当前代码默认是 0.4。
5. 对比 checkpoint 刚加载后和训练若干 iter 后的 operation 分布。如果分布几乎不变，说明 router 基本没有被训练改写。

## 可考虑的改进方向

短期调试可以先加诊断，不必立刻改结构：

- 暴露并记录 TopK 内 `op_logits/op_prob` 的类别均值和 argmax 分布；
- 记录 `operation_head` 与 `score_head` 的梯度；
- 显式在 config 中写出 `allocation_ratio`，避免统计解释混乱。

如果要让 router 真正学习非 keep 策略，可以考虑：

- 增加 operation distribution regularization，例如限制 TopK 内 keep 比例上限或鼓励 clone/split/atten 的最小使用率；
- 给 `operation_head` 增加辅助 loss，例如基于 semantic entropy、scale anisotropy、opacity/confidence 构造 pseudo target；
- 让 `op_onehot` 真正参与 candidate bank 的可导混合，或加入基于 `op_prob` 的软路由训练阶段；
- 增加 `L_cons = |allocation_score - (1 - p_keep)|` 一类约束，使高 allocation score 的高斯更倾向非 keep；
- 对 keep logit 加温度、bias 初始化或训练期退火，避免早期直接塌缩到 keep。

