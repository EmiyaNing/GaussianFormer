# AdaptiveAllocationV4 模式 B 可视化统计 Wrapper 设计文档

## 设计目标

在不修改 `AdaptiveAllocationV4` 源文件、不改变模型 forward 返回值、不影响正常训练和常规 eval 的前提下，实现：

| 目标 | 说明 |
| --- | --- |
| 操作统计 | 统计 keep / clone / split / opacity attenuation 的数量和比例 |
| 操作着色 | 在 Gaussian 可视化中用灰 / 蓝 / 红 / 黄显示四类操作 |
| 单帧报告 | 每完成一帧 / 一个 batch item 的可视化，立即输出该帧操作统计 |
| 低侵入 | 不改 `adaptive_allocationv4.py`，只在 `visualize.py` 运行时安装 wrapper |
| 可控开销 | 默认关闭，仅显式开启可视化统计参数时运行 |
| 同源结果 | 直接截获 `compute_operation_routing` 的真实输出，而不是离线猜测 |

## 模式 B 核心思想

模式 B 使用“运行时方法包装 + forward hook”：

```text
不修改 AdaptiveAllocationV4 源码
   |
   v
visualize.py 在运行时找到 AdaptiveAllocationV4 实例
   |
   v
临时包装 compute_operation_routing
   |
   v
原始 routing 方法照常执行
   |
   v
wrapper 截获 selected_mask 和 op_id
   |
   v
forward hook 在模块 forward 结束后读取输出形状
   |
   v
collector 生成统计和 output_op_id
```

该方案的关键点是：

```text
operation routing 仍然由 AdaptiveAllocationV4 原逻辑产生。
wrapper 只观察结果，不改变结果。
```

## 模块层级图

```text
visualize.py
|
+-- AllocationOperationCollector
|   |
|   +-- runtime routing wrapper
|   |   |
|   |   +-- 调用原 compute_operation_routing
|   |   +-- 截获 selected_mask / op_id
|   |
|   +-- forward hook
|   |   |
|   |   +-- 读取 AdaptiveAllocationV4 输出 shape
|   |   +-- 构造统计表
|   |   +-- 构造 output_op_id
|   |
|   +-- statistic writer
|       |
|       +-- per-frame json / markdown
|       +-- summary json
|       +-- per-iter csv
|       +-- markdown report
|
+-- draw_gaussian_params
|   |
|   +-- adaptive color 参数
|   +-- allocation color 参数
|
+-- save_gaussian / save_gaussian_point
    |
    +-- 如果存在 allocation_op_ids，则按操作类型着色
    +-- 否则退回原 adaptive / semantic / opacity 颜色
```

## 运行时数据流

```text
模型 forward 开始
   |
   v
AdaptiveAllocationV4.forward
   |
   v
compute_operation_routing 被调用
   |
   v
runtime wrapper:
   - 调用原 routing 方法
   - 原结果返回给 AdaptiveAllocationV4
   - 旁路记录 selected_mask / op_id
   |
   v
AdaptiveAllocationV4 继续正常生成 Gaussian 输出
   |
   v
forward hook:
   - 读取输出 Gaussian 数量
   - 读取 wrapper 旁路记录
   - 汇总 per-batch operation stats
   - 构造 output_op_id
   |
   v
visualize.py:
   - 读取 collector 当前 output_op_id
   - 注入 Gaussian 绘制参数
   |
   v
save_gaussian / save_gaussian_point:
   - 按 output_op_id 映射固定颜色
   |
   v
collector:
   - 当前帧可视化结束后立即输出 frame report
```

## Wrapper 与原模块的关系

```text
原 AdaptiveAllocationV4:
   输入不变
   输出不变
   参数不变
   checkpoint 不变
   forward 主流程不变

Wrapper:
   只在 visualize.py 中临时存在
   只包装 compute_operation_routing
   只读取 routing 输出
   不修改 op_id
   不修改 selected_mask
   不修改 Gaussian 输出
```

## 启用条件

collector 默认不启用。

```text
默认运行:
   不创建 collector
   不查找 AdaptiveAllocationV4
   不安装 wrapper
   不注册 hook
   不保存统计

显式开启:
   --allocation-statistic
   或
   --vis-gaussian-allocation-color
```

只有满足显式开启条件，并且处于 rank 0 时，才安装模式 B 组件。

## 生命周期

```text
构建模型
   |
   v
加载 checkpoint
   |
   v
判断是否开启 allocation 统计 / 着色
   |
   +-- 否:
   |     不做任何额外动作
   |
   +-- 是:
         创建 AllocationOperationCollector
         包装 compute_operation_routing
         注册 forward hook
         |
         v
         执行可视化循环
         |
         v
         输出统计文件
         |
         v
         移除 hook
         恢复原 routing 方法
         清理缓存
```

## 统计口径

统计对象是输入到 `AdaptiveAllocationV4.forward` 的原始高斯。

| 字段 | 含义 |
| --- | --- |
| input_gaussians | 输入高斯数量 |
| selected_topk | TopK 选中的高斯数量 |
| keep_total | 最终 keep 的输入高斯数量 |
| keep_from_non_topk | 未进入 TopK、被强制 keep 的数量 |
| keep_from_router | 进入 TopK 后 router 选择 keep 的数量 |
| clone | 执行 clone 的输入高斯数量 |
| split | 执行 split 的输入高斯数量 |
| opacity_attenuation | 执行 opacity attenuation 的输入高斯数量 |
| expected_output_gaussians | 预期有效输出高斯数量 |
| padded_output_gaussians | padding 后输出张量长度 |

一致性关系：

```text
keep_total + clone + split + opacity_attenuation = input_gaussians

keep_from_non_topk + keep_from_router = keep_total

selected_topk = keep_from_router + clone + split + opacity_attenuation

expected_output_gaussians = input_gaussians + clone + split
```

## output_op_id 对齐规则

`op_id` 对应输入高斯数量 N。

可视化使用的最终 Gaussian 可能是输出数量 M。

```text
M = N + clone_count + split_count
```

因此 collector 会构造与输出 Gaussian 对齐的 `output_op_id`：

| 输出区域 | label 来源 |
| --- | --- |
| 前 N 个位置 | 输入高斯的原始 op_id |
| clone 追加区域 | clone label |
| split 追加区域 | split label |
| padding 区域 | invalid label |

输出高斯的颜色表示：

```text
该输出高斯来自哪一种输入操作。
```

## 颜色映射

| 操作 | 颜色 |
| --- | --- |
| keep | 灰色 |
| clone | 蓝色 |
| split | 红色 |
| opacity attenuation | 黄色 |
| invalid / padding | 不参与绘制或退回空白 |

颜色优先级：

```text
allocation operation color
   >
adaptive gaussian color
   >
semantic / opacity default color
```

## 绘制模块数据流

```text
collector.current_output_op_id
   |
   v
draw_gaussian_params
   |
   v
save_gaussian / save_gaussian_point
   |
   v
与 Gaussian 数量进行长度检查
   |
   +-- 数量匹配:
   |     根据 op_id 映射颜色
   |
   +-- 数量不匹配:
         打印提示
         关闭 allocation 着色
         回退原有颜色逻辑
```

## 输出文件

普通可视化输出位置：

```text
work_dir / vis_ep{epoch} / allocation_stats
```

流式可视化输出位置：

```text
work_dir / stream_vis_ep{epoch} / allocation_stats
```

文件结构：

```text
allocation_stats
|
+-- per_frame
|   |
|   +-- {frame}_batch{idx}.json
|   +-- {frame}_batch{idx}.md
+-- allocation_stats_summary.json
+-- allocation_stats_per_iter.csv
+-- allocation_stats_report.md
```

| 文件 | 用途 |
| --- | --- |
| per-frame json | 保存单帧 / 单 batch item 的机器可读统计 |
| per-frame markdown | 保存单帧 / 单 batch item 的表格和文本柱状图 |
| summary json | 保存全局统计和比例 |
| per-iter csv | 保存每个 iter / batch 的统计明细 |
| markdown report | 保存人类可读的汇总表和颜色图例 |

## 单帧即时统计可视化

每个可视化样本完成后，collector 会立即输出一轮该帧统计。

普通可视化：

```text
vis_ep{epoch}
|
+-- allocation_stats
    |
    +-- per_frame
        |
        +-- val_{iter}_batch{idx}.json
        +-- val_{iter}_batch{idx}.md
```

流式可视化：

```text
stream_vis_ep{epoch}
|
+-- allocation_stats
    |
    +-- per_frame
        |
        +-- {scene_token}_frame_{frame_index}_batch{idx}.json
        +-- {scene_token}_frame_{frame_index}_batch{idx}.md
```

单帧 Markdown 报告包含：

| 内容 | 说明 |
| --- | --- |
| frame id | 当前帧 / 当前 iter 标识 |
| module name | 触发统计的 AdaptiveAllocationV4 模块 |
| metric table | keep / clone / split / attenuation 计数与比例 |
| operation bars | 四类操作的文本柱状图 |

该输出发生在当前帧可视化之后、最终全局 summary 之前，因此可以边跑可视化边查看每帧分布。

## 性能边界

默认关闭时：

```text
无 wrapper
无 hook
无额外计算
无额外显存占用
无额外统计文件
```

开启后：

```text
只在 visualize.py 中运行
只在 rank 0 安装
只保存 detach 后的 op_id / selected_mask
不保存完整 feature / gaussian 对象
可视化结束后恢复原方法并移除 hook
```

相比 forward hook 复算方案，模式 B 的额外开销更低：

| 项目 | 模式 A hook 复算 | 模式 B wrapper |
| --- | --- | --- |
| 是否重复 state encoder | 是 | 否 |
| 是否重复 topk | 是 | 否 |
| 是否重复 routing | 是 | 否 |
| 是否同源截获真实 op_id | 否，复算得到 | 是 |
| 对 V4 源码侵入 | 无 | 无 |

## 适用范围

适合：

| 场景 | 说明 |
| --- | --- |
| 单次可视化 | 低侵入，结果同源 |
| validation 可视化 | 可以输出全局统计 |
| stream 可视化 | 可以按帧累计统计 |
| 操作颜色诊断 | 颜色和统计来自同一份 op_id |

不建议：

| 场景 | 原因 |
| --- | --- |
| 默认训练路径 | 没必要承担统计逻辑 |
| 默认 eval 路径 | 会增加诊断逻辑复杂度 |
| FIFO 融合后的 Gaussian 着色 | 融合高斯不再与当前 V4 output_op_id 严格对齐 |

## 总结

模式 B 的设计本质是：

```text
让 AdaptiveAllocationV4 照常工作；
在它说出真实 op_id 的一瞬间旁听；
在可视化侧把这份旁听结果变成统计和颜色。
```

它避免了修改 V4 内部主流程，也避免了复算带来的额外计算，是当前需求下准确性、性能和侵入性之间较平衡的方案。
