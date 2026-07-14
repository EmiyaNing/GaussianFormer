# 自适应语义高斯优化模块

### 核心需求
1. 帮我们基于当前代码中model/encoder/gaussian_encoder/topk_module/topk_densify_v2.py中定义的DensifyOnly，完成一个新的adaptive_allocation模块。
2. 新的adaptive_allocation模块的输入输出变量形式和DensifyOnly保持一致。
3. 新的adaptive_allocation模块不同于DensifyOnly，该模块会自适应的从输入的semantic gaussian中学习一个决策因子。
4. 然后学习到的决策因子会根据该决策因子来决定对该semantic gaussian到底执行split,clone还是keep。
5. 给定输入的instance_feature, anchor, 假设instance_feature的shape是B * num_gaussian * embedim，那么学习到的决策因子应该是B * num_gaussian * 3,其最后一个坐标轴熵的第一个元素表示keep，第二个元素表示clone，第三个元素表示split。
6. 对于clone和split请设置单独的两个网络，其中clone部分网络的操作逻辑和当前的densify_only保持一致；其中split部分操作请使用prediction的结果来修改对应的instance_feature, 以及gaussian的几何属性。
7. 其中clone和split操作对gaussian几何属性进行优化的时候参考densify_only，请使用残差优化的模式。
8. 最后生成的模块类名为AdaptiveAllocation,生成的代码放在model/encoder/gaussian_encoder/topk_module/adaptive_allocation.py当中。

### 注意事项
1. 使用最少量的代码来完成我们的需求。
2. 设计新模块时尽量不要修改已有代码，在修改已有代码时务必先和我确认。
3. 编写的所有代码要保证高的可读性，同时有充足的英文注释。
4. 不要编写所谓的一次性代码，确保代码可复用。
5. 在设计模块之前先将整个任务进行拆分，拆分为不可分割的子任务后对每个子任务进行分析和设计任务流