# 历史帧条件 MixUp Occupancy 数据增强设计

## 1. 结论概览

在当前 GaussianFormer 代码环境中，通过新增一个 dataset pipeline transform 来实现“从历史帧选取环绕视角图像，并将历史帧 occupancy 转到当前坐标系后再混合”的增强是可行的。

但需要区分两种目标：

| 目标 | 仅新增 pipeline 是否足够 | 推荐程度 | 说明 |
| --- | --- | --- | --- |
| 图像线性混合 + occupancy 硬标签空间混合 | 足够 | 推荐首版 | 不改模型和 loss，适合当前训练链路 |
| 图像线性混合 + occupancy soft label 混合 | 不足够 | 不建议首版 | 当前 `OccupancyLoss` 走交叉熵，标签需要离散类别 |
| 历史图像几何严格对齐后再混合 | 不足够或成本较高 | 二期考虑 | 需要图像重投影、深度/可见性或历史视角分支 |
| 追加历史视角并保留历史投影矩阵 | pipeline 可做，但依赖模型配置 | 可作为历史模型版本 | 需要使用已有 `HistoryCrossAttention` 一类能消费多帧图像的配置 |

因此，首版建议把该增强定义为：

> 条件历史帧 MaskMix / Hard-Mix：图像按系数与同一 scene 的历史帧环视图混合；occupancy 先从历史 LiDAR 坐标系变换到当前 LiDAR 坐标系，再按空间 mask 或类别条件覆盖/融合为离散标签。

这更接近 segmentation 中 CutMix/MaskMix 的 3D 版本，而不是严格 soft-label MixUp。这样可以保持现有 `GaussianHead`、`OccupancyLoss` 和 dataloader 返回字段不变。

## 2. 当前代码环境中的可用条件

### 2.1 Pipeline 机制天然支持新增 transform

当前数据集会按配置顺序构建 `OPENOCC_TRANSFORMS` 中的 transform，并在 `NuScenesDataset.__getitem__` 中顺序执行 pipeline。也就是说，只要新增一个注册到 `OPENOCC_TRANSFORMS` 的 transform，就能通过配置插入训练数据流。

现有主流程如下：

| 阶段 | 现有模块 | 关键输出 |
| --- | --- | --- |
| 样本元信息 | `NuScenesDataset.get_data_info` | 当前图像路径、当前 `lidar_pose`、`ego_pose`、`history_context` |
| 当前图像加载 | `LoadMultiViewImageFromFiles` | `img`，当前 6 个 camera 图像列表 |
| occupancy 加载 | `LoadOccupancySurroundOcc` 或 `LoadOccupancyOcc3D` | `occ_label`、`occ_xyz`、`occ_cam_mask`，Occ3D 还可能有 `occ_mask` |
| 图像几何增强 | `ResizeCropFlipImage` | 统一处理所有 `img`，并同步更新投影矩阵 |
| 图像颜色增强 | `PhotoMetricDistortionMultiViewImage` | 光度扰动 |
| 归一化与格式化 | `NormalizeMultiviewImage`、`DefaultFormatBundle` | tensor 化 |
| meta 适配 | `NuScenesAdaptor` | `projection_mat`、`image_wh` |

新增 mixup transform 建议插在：

```text
LoadMultiViewImageFromFiles
        |
LoadOccupancySurroundOcc / LoadOccupancyOcc3D
        |
HistoryOccupancyMixUp      <-- 新增
        |
ResizeCropFlipImage
        |
PhotoMetricDistortionMultiViewImage
        |
NormalizeMultiviewImage
        |
DefaultFormatBundle
        |
NuScenesAdaptor
```

这样做有三个好处：

| 位置优势 | 原因 |
| --- | --- |
| 能同时访问图像和 occupancy | mixup 需要 `img` 与 `occ_label` 都已经存在 |
| 仍在 resize/crop/flip 之前 | 混合后的图像会继续走统一几何增强，减少额外处理 |
| 不影响后续模型输入字段 | 输出仍是 `img`、`occ_label`、`occ_xyz`、mask 等原字段 |

### 2.2 历史帧上下文已经在数据集中提供

`NuScenesDataset.get_data_info` 当前会在 `history_context` 中提供：

| 字段 | 用途 |
| --- | --- |
| `scene_infos` | 访问同一 scene 内其他帧的原始 info |
| `scene_token` | 确保历史帧来自同一 scene |
| `frame_index` | 确定当前帧位置，只从过去帧采样 |
| `data_path` | 拼接历史图像与 LiDAR 路径 |
| `sensor_types` | 保持 camera 顺序和当前帧一致 |

已有 `LoadMultiViewImageHistory` 已经证明：在 transform 内读取历史图像、计算历史 camera 到当前 LiDAR 的投影关系是现有框架支持的。

### 2.3 坐标变换工具已经具备

当前代码中已有 `get_lidar2global` 与 `get_img2global`。历史 occupancy 转到当前坐标系需要的核心变换是：

```text
历史 LiDAR 坐标
      |
      |  hist_lidar2global
      v
全局坐标
      |
      |  inverse(current_lidar2global)
      v
当前 LiDAR 坐标
```

记作：

```text
T_current_from_history = inverse(T_global_from_current_lidar) x T_global_from_history_lidar
```

历史 voxel 中心点经过该变换后，再离散化回当前 occupancy grid。

## 3. 新增 transform 的职责边界

建议新增一个独立 transform，暂命名为 `HistoryOccupancyMixUp`。它不改模型，不改 loss，不改 dataloader，只在 pipeline 中改写训练样本。

### 3.1 输入字段

| 字段 | 是否必须 | 说明 |
| --- | --- | --- |
| `img` | 必须 | 当前帧 6 视角图像 |
| `occ_label` | 必须 | 当前帧 occupancy 语义标签 |
| `occ_xyz` | 必须 | 当前 occupancy 网格中心坐标，通常保持不变 |
| `lidar_pose` | 必须 | 当前 LiDAR 到 global 的变换 |
| `history_context` | 必须 | 用于采样历史帧 |
| `occ_cam_mask` | 建议 | SurroundOcc 主要监督 mask |
| `occ_mask` | 可选 | Occ3D 中常见 |

### 3.2 输出字段

| 字段 | 操作 |
| --- | --- |
| `img` | 当前图像与历史图像混合后的 6 视角图像，或历史模式下追加图像 |
| `occ_label` | 混合后的离散 occupancy 标签 |
| `occ_cam_mask` | 与混合区域同步更新 |
| `occ_mask` | 如果输入中存在，则同步更新 |
| `occ_xyz` | 保持当前坐标系网格，不建议改动 |
| `mixup_meta` | 可选，仅用于 debug，不加入默认 return_keys |

`occ_xyz` 不应被替换为历史坐标。它代表当前训练样本的监督查询点集合，历史 occupancy 应该被映射到这些当前查询点上。

## 4. 数据流设计

整体流程图：

```text
当前样本 t
  |
  |-- 当前 6 视角图像 I_t
  |-- 当前 occupancy Y_t
  |-- 当前位姿 T_global_from_lidar_t
  |-- history_context
  |
  v
按条件采样历史帧 h, h < t, same scene
  |
  |-- 历史 6 视角图像 I_h
  |-- 历史 occupancy Y_h
  |-- 历史位姿 T_global_from_lidar_h
  |
  v
历史 occupancy 坐标变换
  |
  |-- voxel_h -> xyz_h
  |-- xyz_h -> xyz_t
  |-- xyz_t -> voxel_t
  |
  v
构建混合区域 M
  |
  |-- 时间条件
  |-- 空间条件
  |-- 类别条件
  |-- mask 条件
  |
  v
图像混合 + occupancy 混合
  |
  |-- I_mix = image_mix(I_t, I_h)
  |-- Y_mix = occ_mix(Y_t, transform(Y_h), M)
  |
  v
后续原 pipeline
```

## 5. 历史帧采样策略

历史帧不是任意全局样本，而是从当前 scene 的过去帧中选择。

### 5.1 基础约束

| 条件 | 建议 |
| --- | --- |
| scene | 必须与当前帧相同 |
| 时间方向 | 只选历史帧，即 `history_index < current_index` |
| camera 完整性 | 6 个 camera 文件均存在 |
| occupancy 完整性 | 历史 occupancy 文件可读取 |
| 位姿有效性 | 当前和历史 LiDAR pose 均可计算 |
| 最大时间窗 | 建议 1 到 6 帧内起步 |
| 最小历史数量 | 当前为 scene 首帧或无有效历史帧时跳过增强 |

### 5.2 推荐采样分布

| 策略 | 说明 | 推荐用途 |
| --- | --- | --- |
| 均匀采样 | 在最近 K 帧中随机选 1 帧 | 首版最简单 |
| 近帧优先 | 越近的历史帧概率越高 | 降低几何错位 |
| 间隔采样 | 避免只采上一帧，加入一定时间差 | 增强多样性 |
| 熵/覆盖率采样 | 复用已有 entropy history 思路 | 二期增强 |

首版建议使用“近帧优先”。例如大多数样本来自前 1 到 3 帧，少量样本来自前 4 到 6 帧。

### 5.3 触发概率

| 参数 | 建议初值 | 说明 |
| --- | --- | --- |
| `prob` | 0.2 到 0.5 | mixup 不是每个样本都启用 |
| `warmup_epochs` | 0 到 2 | 可选，前期关闭或降低概率 |
| `max_history_gap` | 3 或 6 | 历史窗口过大时图像几何错位更明显 |

## 6. 图像混合设计

### 6.1 基础图像混合

对于当前帧每个 camera，选择同 camera 的历史图像进行混合：

```text
CAM_FRONT 当前图像      + CAM_FRONT 历史图像      -> CAM_FRONT 混合图像
CAM_FRONT_RIGHT 当前图像 + CAM_FRONT_RIGHT 历史图像 -> CAM_FRONT_RIGHT 混合图像
...
CAM_BACK_RIGHT 当前图像  + CAM_BACK_RIGHT 历史图像  -> CAM_BACK_RIGHT 混合图像
```

图像混合系数建议从较保守范围开始：

| 参数 | 建议 |
| --- | --- |
| 历史图像权重 alpha | 0.1 到 0.4 |
| 当前图像权重 | 1 - alpha |
| 每视角 alpha | 首版所有 camera 共用同一个 alpha |
| 图像 dtype | 保持 float32，与当前 loader 一致 |

### 6.2 图像几何一致性的风险

当前 `adaptive_allocationv5.py` 这类普通 6 视角配置只保留当前帧的 6 个投影矩阵。若直接把历史图像线性混进当前图像槽位，模型仍会使用当前投影矩阵采样图像特征。

这会带来一个天然不一致：

```text
历史图像像素来自历史相机位姿
        |
        | 但模型投影使用当前相机位姿
        v
历史图像成分与当前 3D 点投影不严格对齐
```

因此，首版应把图像 mixup 当作外观/上下文扰动，而不是严格几何监督。对应约束是：

| 控制项 | 建议 |
| --- | --- |
| 历史帧间隔 | 不宜过大 |
| alpha | 不宜过大 |
| 同 scene | 必须 |
| 同 camera | 必须 |
| 是否用于验证 | 不用于验证，只用于 train |

### 6.3 更严格的图像模式

如果后续使用 `LoadMultiViewImageHistory` 和 `HistoryCrossAttention` 这类能消费多帧图像的配置，可以采用更严格的模式：

| 模式 | 图像处理 | 投影矩阵 | 模型要求 |
| --- | --- | --- | --- |
| 当前槽位混合 | 6 张当前图像被混入历史图像 | 当前投影矩阵 | 普通 6-view 模型即可 |
| 追加历史视角 | 当前 6 张 + 历史 6 张 | 每帧各自正确投影到当前 LiDAR | 需要历史图像分支 |
| 重投影后混合 | 将历史图像 warp 到当前相机平面后混合 | 当前投影矩阵 | 需要深度/可见性，成本高 |

首版建议使用“当前槽位混合”。如果实验目标更偏几何一致，建议切到“追加历史视角”而不是做像素级 warp。

## 7. Occupancy 混合设计

### 7.1 历史 occupancy 加载

根据数据集类型分别加载：

| 数据集 | 历史标签来源 | 当前代码约定 |
| --- | --- | --- |
| SurroundOcc | `occ_path` 下按 LiDAR 文件名保存的 `.npy` | 稀疏索引转为 200 x 200 x 16 dense label |
| Occ3D | scene / sample token 下的 `labels.npz` | 直接读取 semantics 和 mask |

首版建议优先支持当前配置常用的 SurroundOcc，再扩展 Occ3D。

### 7.2 坐标变换

历史 occupancy 的标签本来定义在历史帧 LiDAR 坐标系下，不能直接和当前 `occ_label` 按数组下标混合。必须先变换到当前 LiDAR 坐标系。

变换流程：

```text
历史 occupancy label grid
  |
  |  取非空或有效 voxel
  v
历史 voxel index
  |
  |  index -> 历史 voxel center xyz_h
  v
历史 LiDAR 坐标点 xyz_h
  |
  |  T_current_from_history
  v
当前 LiDAR 坐标点 xyz_t
  |
  |  当前 pc_range + voxel_size 离散化
  v
当前 voxel index
  |
  |  边界过滤 + 冲突处理
  v
历史 occupancy 在当前坐标系下的 label grid
```

### 7.3 离散化规则

当前常用 grid 为：

| 维度 | 范围 | 分辨率 | 数量 |
| --- | --- | --- | --- |
| x | -50 到 50 | 0.5 | 200 |
| y | -50 到 50 | 0.5 | 200 |
| z | -5 到 3 | 0.5 | 16 |

历史点变换到当前坐标后，只有落在该范围内的 voxel 才参与混合。越界点直接丢弃。

### 7.4 冲突处理

历史 occupancy 重投影到当前 grid 时，多个历史 voxel 可能落入同一个当前 voxel。建议按以下优先级处理：

| 优先级 | 规则 |
| --- | --- |
| 1 | 有效 mask 内的 label 优先于无效 label |
| 2 | 非 empty label 优先于 empty label |
| 3 | 若多个非 empty label 冲突，保留距离当前 voxel center 最近的历史 voxel |
| 4 | 若距离仍相同，保留出现次数更多的类别 |

这样可以减少旋转和平移离散化带来的随机噪声。

## 8. 混合 mask 设计

为了避免整帧硬覆盖导致标签剧烈漂移，建议引入 3D 混合区域 `M`。

### 8.1 推荐 mask 类型

| mask 类型 | 说明 | 推荐阶段 |
| --- | --- | --- |
| 随机 3D cuboid | 在当前 occupancy grid 中随机采样一个或多个长方体区域 | 首版推荐 |
| BEV 区域 mask | 只在 x-y 平面采样区域，覆盖所有 z | 简单稳定 |
| 前景类别 mask | 只混合非 empty 或指定类别 | 可选增强 |
| 可见区域 mask | 只混合 camera/lidar mask 有效区域 | 推荐开启 |
| 时间稳定类别 mask | 优先混合道路、建筑、植被等静态类 | 二期推荐 |

### 8.2 首版建议

首版建议使用：

```text
M = 随机 BEV 区域
    AND 历史 occupancy 变换后仍在当前 grid 内
    AND 历史 label 有效
    AND 当前/历史 mask 有效
```

混合后：

```text
Y_mix = Y_history_in_current, where M is true
Y_mix = Y_current, otherwise
```

这保持 `occ_label` 仍为离散类别，不需要改 loss。

### 8.3 类别条件建议

动态物体在历史帧中被变换到当前坐标后，可能并不符合当前时刻真实位置。是否混入动态类取决于实验目标：

| 策略 | 优点 | 风险 |
| --- | --- | --- |
| 只混静态类 | 几何更稳定 | 增强多样性较弱 |
| 静态类高概率，动态类低概率 | 折中 | 参数更多 |
| 所有类都混 | 增强最强 | 可能引入动态目标伪标签 |

建议首版默认：

```text
静态背景类：允许混合
动态前景类：低概率混合或先关闭
empty 类：不主动覆盖非 empty
```

如果目标是提高动态物体鲁棒性，可以在第二阶段打开动态类混合。

## 9. Mask 与监督字段的更新

### 9.1 SurroundOcc

SurroundOcc 当前主要输出：

| 字段 | 说明 |
| --- | --- |
| `occ_label` | dense semantic label |
| `occ_cam_mask` | 当前实现中由 label 生成的 mask |
| `occ_xyz` | 当前坐标系网格中心 |

混合后需要同步：

| 字段 | 操作 |
| --- | --- |
| `occ_label` | 在 `M` 内替换为历史变换后的 label |
| `occ_cam_mask` | 在 `M` 内合并历史有效 mask |
| `occ_xyz` | 保持当前帧原值 |

### 9.2 Occ3D

Occ3D 当前还会输出：

| 字段 | 说明 |
| --- | --- |
| `occ_mask` | semantics 非 empty 区域 |
| `occ_cam_mask` | camera visible mask |
| `occ_lidar_mask` | lidar visible mask |

混合后应同步更新：

| 字段 | 操作 |
| --- | --- |
| `occ_label` | 在 `M` 内替换 |
| `occ_mask` | 在 `M` 内使用历史变换后的非 empty / 有效 mask |
| `occ_cam_mask` | 在 `M` 内合并历史 camera mask |
| `occ_lidar_mask` | 如后续使用，则同步合并 |

注意：如果新增字段没有出现在 dataset `return_keys` 中，训练端不会拿到。首版应优先复用已有字段，而不是依赖新增字段。

## 10. 与现有 loss 的兼容性

当前 `GaussianHead` 从 `metas` 中读取：

| 字段 | 用途 |
| --- | --- |
| `occ_xyz` | 聚合器采样点 |
| `occ_label` | voxel 监督标签 |
| `occ_cam_mask` | 返回给下游 |
| `occ_mask` | 如果存在，会进入 loss mask |

当前 `OccupancyLoss` 使用交叉熵、Lovasz 等离散标签损失。它期望 `sampled_label` 是类别 id，而不是 one-hot 或 soft label。

因此：

| 方案 | 是否兼容当前 loss | 说明 |
| --- | --- | --- |
| 硬标签 mask 替换 | 兼容 | 推荐 |
| soft label 线性混合 | 不兼容 | 需要改 loss |
| 额外输出 mixup 权重 | 默认无效 | loss 当前不会消费 |

如果未来要做真正 MixUp，可扩展为：

```text
目标分布 = (1 - lambda) x 当前 one-hot + lambda x 历史 one-hot
```

但这要求 `OccupancyLoss` 改为支持 soft target 的 CE/KL，并同步处理 Lovasz 类损失。该路径不属于“只加 pipeline”的范畴。

## 11. 推荐配置形态

### 11.1 当前 6-view 模型首版

适合 `adaptive_allocationv5.py` 这种当前帧 6 图输入配置。

| 配置项 | 建议 |
| --- | --- |
| pipeline 插入位置 | occupancy loader 之后，resize 之前 |
| 图像模式 | 当前 6 图槽位线性混合 |
| occupancy 模式 | 历史 label 转当前坐标后硬标签 mask 替换 |
| alpha | 0.1 到 0.4 |
| 历史窗口 | 最近 1 到 3 帧优先 |
| 混合区域 | 小到中等 BEV 区域 |
| 动态类 | 默认关闭或低概率 |

优点：

```text
不改模型
不改 loss
不改 dataloader
能快速验证增强是否有效
```

风险：

```text
历史图像成分与当前投影矩阵不严格对齐
alpha 过大时可能干扰图像特征学习
动态物体历史标签可能成为噪声
```

### 11.2 历史图像模型版本

适合 `img_voxel_crossframe.py`、`dynamic_fusion.py` 这类已有历史图像注意力设计的配置。

| 配置项 | 建议 |
| --- | --- |
| 图像模式 | 当前图像保留，历史图像作为额外 frame 输入 |
| 投影矩阵 | 使用历史相机到当前 LiDAR 的矩阵 |
| occupancy | 仍然转到当前 LiDAR 坐标系 |
| 模型要求 | encoder 中启用 history attention |

这个版本几何更自洽：

```text
历史图像像素使用历史 camera pose
历史 occupancy label 转到当前 LiDAR pose
模型在跨帧 attention 中学习对齐关系
```

但它不再是单纯“把历史图像混进当前 6 图”的增强，而是多帧训练输入增强。

## 12. 可行性评估

| 维度 | 评估 | 依据 |
| --- | --- | --- |
| pipeline 接入 | 高 | transform registry 和 pipeline 顺序已经存在 |
| 历史帧采样 | 高 | `history_context` 已提供 scene 信息和 frame index |
| 历史图像加载 | 高 | `LoadMultiViewImageHistory` 已有类似逻辑 |
| occupancy 坐标转换 | 高 | 当前已有 LiDAR to global pose 工具 |
| hard-label 混合 | 高 | 与当前 `occ_label` 和 CE loss 兼容 |
| soft-label MixUp | 低 | 当前 loss 不支持 soft target |
| 图像几何严格一致 | 中低 | 直接线性混合历史图像会和当前投影矩阵不完全匹配 |
| Occ3D 兼容 | 中 | 可做，但路径和 mask 规则比 SurroundOcc 更多 |
| 训练稳定性 | 中 | 需要控制 alpha、时间窗、类别和空间 mask |

整体判断：

```text
首版作为 pipeline-only hard-mask mix 可行性高。
若追求严格 soft MixUp 或严格跨时刻图像几何一致，需要改 loss 或模型输入形式。
```

## 13. 建议实现拆分

虽然本文档不写真实代码，但实现上建议按下面的职责拆分，避免 transform 过大。

| 组件 | 职责 |
| --- | --- |
| 历史帧采样器 | 根据 `history_context` 选合法历史帧 |
| 历史图像加载器 | 读取同 camera 顺序的 6 视角图像 |
| 历史 occupancy 加载器 | 根据数据集类型读取历史标签 |
| occupancy 坐标变换器 | 历史 voxel -> 当前 voxel |
| mask 生成器 | 生成 BEV/cuboid/class/valid mask |
| 混合器 | 更新 `img`、`occ_label`、mask 字段 |
| debug 记录 | 可选保存 alpha、历史 gap、混合比例 |

首版最小闭环：

```text
同 scene 最近 K 帧采样
        |
读取历史 6 图
        |
读取历史 SurroundOcc label
        |
历史 label 转当前 grid
        |
随机 BEV mask
        |
图像 alpha blend
        |
occupancy hard replace
```

## 14. 验证计划

### 14.1 单样本可视化检查

| 检查项 | 期望 |
| --- | --- |
| 历史帧选择 | 来自同 scene，且 index 小于当前帧 |
| 图像顺序 | 6 个 camera 顺序与当前帧一致 |
| alpha | 落在设定范围 |
| occupancy 越界率 | 不应异常高 |
| 混合 voxel 比例 | 在合理范围，例如 5% 到 30% |
| mask 更新 | `occ_label` 与 mask 形状一致 |

### 14.2 坐标变换 sanity check

建议选连续两帧做检查：

```text
历史非空 voxel center
        |
变换到当前坐标
        |
投回当前 grid
        |
统计保留比例、越界比例、类别分布
```

合理现象：

| 指标 | 期望 |
| --- | --- |
| 保留比例 | 近帧应较高 |
| 越界比例 | 随时间 gap 增大而上升 |
| 类别分布 | 不应只剩 empty 或单一类别 |
| 静态结构 | 道路、建筑、植被应较稳定 |

### 14.3 训练 A/B

| 实验 | 目的 |
| --- | --- |
| baseline | 当前配置，无 mixup |
| image-only mix | 判断图像扰动影响 |
| occ-only mix | 判断 occupancy 标签增强影响 |
| image + occ mix | 完整方案 |
| static-only mix | 检查动态类噪声 |
| all-class mix | 检查增强上限 |

### 14.4 关键指标

| 指标 | 关注点 |
| --- | --- |
| 训练 loss | 是否出现异常震荡 |
| mIoU | 总体性能 |
| foreground IoU | 是否改善稀有类或前景 |
| empty / non-empty 准确性 | 是否破坏几何占据 |
| 每类 IoU | 动态类是否被噪声伤害 |

## 15. 风险与缓解

| 风险 | 表现 | 缓解 |
| --- | --- | --- |
| 历史图像与当前投影不一致 | 训练不稳定或图像特征变差 | 降低 alpha，缩小 history gap |
| 动态物体标签错位 | car/pedestrian 类 IoU 下降 | 首版只混静态类或降低动态类概率 |
| occupancy 离散化冲突 | 局部 label 噪声 | 使用距离优先或投票规则 |
| mask 过大 | 监督分布突变 | 限制混合比例 |
| empty 覆盖 non-empty | 几何监督变弱 | empty 不主动覆盖非 empty |
| Occ3D/SurroundOcc mask 语义不同 | loss mask 异常 | 分数据集维护 mask 合并规则 |
| return_keys 不含新字段 | debug 信息丢失 | debug 字段仅本地使用，不影响训练 |

## 16. 推荐首版参数

| 参数 | 建议值 |
| --- | --- |
| 启用阶段 | train only |
| 触发概率 | 0.3 |
| 历史窗口 | 最近 3 帧 |
| 历史采样 | 近帧优先 |
| 图像 alpha | 0.1 到 0.3 |
| occupancy mask | 随机 BEV 矩形 |
| 混合体素比例 | 5% 到 20% |
| 类别策略 | 静态类优先，动态类默认关闭 |
| empty 策略 | empty 不覆盖 non-empty |
| soft label | 不启用 |

## 17. 最终建议

建议先实现一个 pipeline-only 的 `HistoryOccupancyMixUp`，只做硬标签空间混合：

```text
图像：当前环视图与同 scene 历史环视图按小 alpha 混合
标签：历史 occupancy 先转到当前 LiDAR 坐标系
监督：在受控 3D mask 内用历史 label 替换当前 label
损失：继续使用当前离散标签 loss
```

这个版本最符合当前代码结构，工程改动小，风险可控，也能直接回答“历史帧图像 + 当前坐标系 occupancy mixup”这一核心想法是否有效。

如果首版实验有效，再推进两个增强方向：

| 方向 | 需要改动 | 价值 |
| --- | --- | --- |
| soft-label occupancy MixUp | 改 `OccupancyLoss` | 更接近原始 MixUp |
| 历史视角几何一致输入 | 使用 history attention 配置 | 减少图像投影错位 |

