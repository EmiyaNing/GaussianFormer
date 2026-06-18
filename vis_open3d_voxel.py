# vis_standalone_3d_open3d_voxel_optimized.py
import os
import gc
import psutil
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from pyquaternion import Quaternion
from mpl_toolkits.axes_grid1 import ImageGrid
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui

from visualize_color_utils import get_adaptive_gaussian_colors

gui.Application.instance.initialize()

# 内存监控装饰器
def memory_monitor(func):
    """监控函数内存使用的装饰器"""
    def wrapper(*args, **kwargs):
        try:
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"[Memory] 函数 {func.__name__} 开始: {start_memory:.2f} MB")
            
            result = func(*args, **kwargs)
            
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            print(f"[Memory] 函数 {func.__name__} 结束: {end_memory:.2f} MB, 使用: {memory_used:.2f} MB")
            
            return result
        except ImportError:
            # 如果psutil不可用，跳过内存监控
            return func(*args, **kwargs)
    return wrapper

# 内存清理函数
def clear_memory():
    """强制清理内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# 修复safe_sigmoid函数
def safe_sigmoid(x):
    """安全的sigmoid函数，支持Tensor和numpy数组"""
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    elif isinstance(x, np.ndarray):
        # 对numpy数组使用数值稳定的sigmoid
        x = np.clip(x, -500, 500)  # 防止数值溢出
        return 1 / (1 + np.exp(-x))
    else:
        # 尝试转换为tensor
        return torch.sigmoid(torch.tensor(x))

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing='ij')
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def get_nuscenes_colormap():
    """NuScenes 颜色映射"""
    colors = np.array(
        [
            [  0,   0,   0, 255],       # others
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            [139, 137, 137, 255],       # other_flat
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
        ]
    ).astype(np.float32) / 255.
    return colors

def get_sphere_template(resolution=4, device='cuda'):
    """获取单位球体的顶点和面模板，缓存在内存中"""
    if not hasattr(get_sphere_template, '_cache'):
        get_sphere_template._cache = {}
    key = (resolution, device)
    if key not in get_sphere_template._cache:
        # 在CPU上创建球体
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
        vertices = np.asarray(sphere.vertices, dtype=np.float32)  # (V, 3)
        triangles = np.asarray(sphere.triangles, dtype=np.float32)  # (F, 3)
        # 转换为PyTorch张量并移到指定设备
        vertices_tensor = torch.from_numpy(vertices).to(device)
        triangles_tensor = torch.from_numpy(triangles).to(device)
        get_sphere_template._cache[key] = (vertices_tensor, triangles_tensor)
    return get_sphere_template._cache[key]

@memory_monitor
def create_voxel_grid_from_occupancy(occ_data, voxel_size, vox_origin, sem=False, dataset='nusc', max_voxels=50000):
    """从占用数据创建Open3D体素网格可视化（使用正方体网格VoxelGrid）"""
    # 确保数据是numpy数组
    if isinstance(occ_data, torch.Tensor):
        voxels = occ_data[0].cpu().to(torch.int).numpy()
    else:
        voxels = occ_data[0].astype(np.int32)

    # 设置边界值用于颜色映射
    voxels[0, 0, 0] = 1
    voxels[-1, -1, -1] = 1

    # 获取颜色映射表
    if sem:
        colormap = get_nuscenes_colormap()

    # 计算体素世界坐标和对应标签值
    grid_coords = get_grid_coords(voxels.shape, voxel_size) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    grid_values = voxels.reshape(-1)

    # 获取FOV内的体素
    if sem:
        if dataset == 'nusc':
            mask = (grid_values >= 0) & (grid_values < 17)
        elif dataset == 'kitti360':
            mask = (grid_values > 0) & (grid_values < 19)
        else:
            mask = (grid_values > 0) & (grid_values < 20)
    else:
        mask = (grid_values > 0) & (grid_values < 100)

    fov_coords = grid_coords[mask]
    fov_values = grid_values[mask]

    print(f"[create_voxel_grid] 有效体素数量: {len(fov_values)}")

    # 如果体素数量过多，进行采样
    if len(fov_values) > max_voxels:
        print(f"⚠ 体素数量过多 ({len(fov_values)})，进行采样到 {max_voxels}")
        indices = np.random.choice(len(fov_values), max_voxels, replace=False)
        fov_coords = fov_coords[indices]
        fov_values = fov_values[indices]

    # 创建 Open3D VoxelGrid（正方体网格）
    vox_origin_arr = np.array(vox_origin, dtype=np.float64)
    voxel_size_f = float(voxel_size[0])  # 各向同性体素大小

    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.origin = vox_origin_arr
    voxel_grid.voxel_size = voxel_size_f

    # 计算每个体素在 VoxelGrid 中的整数网格索引并通过 add_voxel 逐个添加
    half_voxel = np.array(voxel_size, dtype=np.float64) / 2.0
    voxels_added = 0
    for i in range(len(fov_coords)):
        # 从世界坐标反推整数网格索引
        grid_idx_arr = ((fov_coords[i] - vox_origin_arr - half_voxel) / voxel_size_f).astype(np.int32)

        if sem:
            sem_value = int(fov_values[i])
            if 0 <= sem_value < len(colormap):
                color = colormap[sem_value][:3].tolist()
            else:
                color = [0.5, 0.5, 0.5]
        else:
            color = [0.5, 0.5, 0.5]

        voxel = o3d.geometry.Voxel(grid_idx_arr.tolist(), color)
        voxel_grid.add_voxel(voxel)
        voxels_added += 1

    print(f"[create_voxel_grid] VoxelGrid 创建完成，含 {voxels_added} 个正方体体素")

    # 清理临时变量释放内存
    del grid_coords, fov_coords, fov_values
    clear_memory()

    return [voxel_grid]


def save_occ(save_dir, occ_data, name, sem=False, cap=2, dataset='nusc', show_window=True):
    """使用Open3D体素网格的3D占用可视化 - 优化内存版本"""
    print(f"[save_occ] {name}")
    
    try:
        import open3d as o3d
        print("✓ Open3D导入成功")
    except Exception as e:
        print(f"✗ 无法导入Open3D: {e}")
        return

    if dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]

    # 创建体素网格
    voxel_geometries = create_voxel_grid_from_occupancy(occ_data, voxel_size, vox_origin, sem, dataset)
    
    # 创建Open3D可视化
    vis = o3d.visualization.Visualizer()
    
    # 根据参数决定是否显示窗口
    if show_window:
        vis.create_window(window_name=f"3D Occupancy Voxels: {name}", width=1200, height=800)
        print("✓ 创建交互式3D体素窗口")
    else:
        vis.create_window(width=2560, height=1440, visible=False)
        print("✓ 使用离屏渲染")

    # 添加体素几何体到可视化 - 分批处理避免内存峰值
    geometries_added = 0
    batch_size = 1000
    
    for batch_start in range(0, len(voxel_geometries), batch_size):
        batch_end = min(batch_start + batch_size, len(voxel_geometries))
        for geometry in voxel_geometries[batch_start:batch_end]:
            vis.add_geometry(geometry)
            geometries_added += 1
        
        # 清理批次内存
        clear_memory()
        print(f"[save_occ] 已添加 {geometries_added}/{len(voxel_geometries)} 个几何体")
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])  # 白色背景
    render_option.mesh_show_wireframe = False
    render_option.mesh_show_back_face = False
    render_option.show_coordinate_frame = True
    # VoxelGrid 专用选项：显示正方体体素线框（让每个体素立方体边缘更清晰）
    if hasattr(render_option, 'voxel_grid_show_wireframe'):
        render_option.voxel_grid_show_wireframe = True
    
    # 设置相机参数
    ctr = vis.get_view_control()
    
    # 设置合适的相机视角
    ctr.set_front([0, 0, -1])  # 从上方看
    ctr.set_up([0, -1, 0])     # 上方向
    ctr.set_zoom(0.3)
    
    # 更新渲染
    vis.poll_events()
    vis.update_renderer()
    
    # 保存截图
    filepath = os.path.join(save_dir, f'{name}.png')
    vis.capture_screen_image(filepath)
    print(f"✓ 截图保存到: {filepath}")
    
    # 如果显示窗口，则保持打开状态
    if show_window:
        print("🖱️  交互式3D体素窗口已打开，可以:")
        print("   - 鼠标拖拽旋转视角")
        print("   - 滚轮缩放")
        print("   - 按 'Q' 或关闭窗口继续")
        vis.run()  # 这会阻塞直到窗口关闭
    
    vis.destroy_window()
    
    # 清理内存
    del voxel_geometries
    clear_memory()
    
    print(f"[save_occ] 完成 {name}")


def save_occ_error(save_dir, pred_occ, gt_occ, name, dataset='nusc', show_window=True, empty_label=17):
    """预测错误分类可视化。

    对于预测或GT中任一为占用的体素（pred_occ != empty_label OR gt_occ != empty_label），
    按以下规则着色：
      - 🟢 绿色 (0.0, 1.0, 0.0): 位置和语义类别均预测正确
          条件：gt_occ != empty_label AND pred_occ == gt_occ
      - 🔴 深红色 (0.8, 0.0, 0.0): 位置预测正确但语义类别错误
          条件：gt_occ != empty_label AND pred_occ != empty_label AND pred_occ != gt_occ
      - ⚫ 黑色 (0.0, 0.0, 0.0): 假阳性（GT为空但预测为非空）
          条件：gt_occ == empty_label AND pred_occ != empty_label
      - 🔵 蓝色 (0.0, 0.0, 1.0): 假阴性 / 漏检（GT为非空但预测为空）
          条件：gt_occ != empty_label AND pred_occ == empty_label

    Args:
        save_dir:    保存目录
        pred_occ:    (H, W, D) 预测的占用/语义标签（argmax 结果）
        gt_occ:      (H, W, D) 真实的占用/语义标签
        name:        文件名前缀
        dataset:     数据集名称（'nusc' 或 'kitti'）
        show_window: 是否显示交互式窗口
        empty_label: 空体素的标签值（NuScenes 中为 17）
    """
    print(f"[save_occ_error] {name}")
    print(f"  pred_occ shape: {pred_occ.shape}, gt_occ shape: {gt_occ.shape}")

    try:
        import open3d as o3d
    except Exception as e:
        print(f"✗ 无法导入 Open3D: {e}")
        return

    if dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
    else:
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]

    # 确保是 numpy 数组
    if isinstance(pred_occ, torch.Tensor):
        pred_occ_np = pred_occ.cpu().numpy()
    else:
        pred_occ_np = pred_occ
    if isinstance(gt_occ, torch.Tensor):
        gt_occ_np = gt_occ.cpu().numpy()
    else:
        gt_occ_np = gt_occ

    # 确保是 3D (H, W, D)
    if pred_occ_np.ndim == 4:
        pred_occ_np = pred_occ_np[0]
    if gt_occ_np.ndim == 4:
        gt_occ_np = gt_occ_np[0]

    # 获取体素网格坐标
    grid_coords = get_grid_coords(pred_occ_np.shape, voxel_size)
    grid_coords = grid_coords + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    # 筛选出预测或GT中任一为占用的体素（并集掩码，确保假阴性也可见）
    pred_occ_flat = pred_occ_np.reshape(-1)
    gt_occ_flat = gt_occ_np.reshape(-1)
    pred_mask = pred_occ_flat != empty_label
    gt_mask = gt_occ_flat != empty_label
    any_mask = pred_mask | gt_mask

    pred_voxels = pred_occ_flat[any_mask]
    gt_voxels = gt_occ_flat[any_mask]
    coords = grid_coords[any_mask]

    n_pred_occ = pred_mask.sum()
    n_gt_occ = gt_mask.sum()
    print(f"  预测占用的体素数: {n_pred_occ}, GT占用的体素数: {n_gt_occ}")
    print(f"  并集体素数（可视化范围）: {len(pred_voxels)}")

    if len(pred_voxels) == 0:
        print("  ⚠ 预测和GT均为空，跳过可视化")
        return

    # 分类并着色
    voxel_colors = []
    for i in range(len(pred_voxels)):
        if gt_voxels[i] != empty_label:
            # GT 为占用
            if pred_voxels[i] == empty_label:
                # 预测为空 → 假阴性 / 漏检 → 蓝色
                voxel_colors.append([0.0, 0.0, 1.0])
            elif pred_voxels[i] == gt_voxels[i]:
                # 语义类别正确 → 绿色
                voxel_colors.append([0.0, 1.0, 0.0])
            else:
                # 语义类别错误 → 深红色
                voxel_colors.append([0.8, 0.0, 0.0])
        else:
            # GT 为空 → 假阳性 → 黑色
            voxel_colors.append([0.0, 0.0, 0.0])

    voxel_colors = np.array(voxel_colors, dtype=np.float32)

    # 统计各类别数量及百分比
    green_mask  = (voxel_colors[:, 0] == 0.0) & (voxel_colors[:, 1] == 1.0) & (voxel_colors[:, 2] == 0.0)
    red_mask    = (voxel_colors[:, 0] == 0.8) & (voxel_colors[:, 1] == 0.0) & (voxel_colors[:, 2] == 0.0)
    black_mask  = (voxel_colors[:, 0] == 0.0) & (voxel_colors[:, 1] == 0.0) & (voxel_colors[:, 2] == 0.0)
    blue_mask   = (voxel_colors[:, 0] == 0.0) & (voxel_colors[:, 1] == 0.0) & (voxel_colors[:, 2] == 1.0)
    n_green  = green_mask.sum()
    n_red    = red_mask.sum()
    n_black  = black_mask.sum()
    n_blue   = blue_mask.sum()
    n_total = n_green + n_red + n_black + n_blue
    if n_total > 0:
        print(f"    🟢 位置和语义均正确 (绿色):   {n_green} ({n_green/n_total*100:.1f}%)")
        print(f"    🔴 位置正确但语义错误 (深红):  {n_red} ({n_red/n_total*100:.1f}%)")
        print(f"    ⚫ 假阳性 (黑色):              {n_black} ({n_black/n_total*100:.1f}%)")
        print(f"    🔵 假阴性/漏检 (蓝色):         {n_blue} ({n_blue/n_total*100:.1f}%)")
    else:
        print(f"    🟢 位置和语义均正确 (绿色):   {n_green}")
        print(f"    🔴 位置正确但语义错误 (深红):  {n_red}")
        print(f"    ⚫ 假阳性 (黑色):              {n_black}")
        print(f"    🔵 假阴性/漏检 (蓝色):         {n_blue}")

    # 创建 VoxelGrid（正方体网格）
    vox_origin_arr = np.array(vox_origin, dtype=np.float64)
    voxel_size_f = float(voxel_size[0])
    half_voxel = np.array(voxel_size, dtype=np.float64) / 2.0

    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.origin = vox_origin_arr
    voxel_grid.voxel_size = voxel_size_f

    for i in range(len(coords)):
        # 从世界坐标反推整数网格索引
        grid_idx_arr = ((coords[i] - vox_origin_arr - half_voxel) / voxel_size_f).astype(np.int32)
        color = voxel_colors[i].tolist()
        voxel = o3d.geometry.Voxel(grid_idx_arr.tolist(), color)
        voxel_grid.add_voxel(voxel)

    print(f"  VoxelGrid 创建完成，含 {len(coords)} 个正方体体素")

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    if show_window:
        vis.create_window(window_name=f"Prediction Error Map: {name}", width=1200, height=800)
    else:
        vis.create_window(width=2560, height=1440, visible=False)

    vis.add_geometry(voxel_grid)

    # 渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])  # 白色背景
    render_option.show_coordinate_frame = True
    # VoxelGrid 线框显示
    if hasattr(render_option, 'voxel_grid_show_wireframe'):
        render_option.voxel_grid_show_wireframe = True

    # 相机视角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.3)

    vis.poll_events()
    vis.update_renderer()

    # 保存截图
    filepath = os.path.join(save_dir, f'{name}_error_map.png')
    vis.capture_screen_image(filepath)
    print(f"  ✓ 截图保存到: {filepath}")

    if show_window:
        print("  🖱️  交互式窗口已打开，按 'Q' 或关闭窗口继续")
        vis.run()

    vis.destroy_window()
    del voxel_grid, voxel_colors
    clear_memory()
    print(f"[save_occ_error] 完成 {name}")


def save_gaussian_with_gt_occ(save_dir, gaussian_data, gt_occ, name,
                                scalar=1.5, ignore_opa=False, filter_zsize=False,
                                dataset='nusc', show_window=True,
                                max_gaussians=25600, max_voxels=50000,
                                adaptive_color=False, adaptive_color_seed=42):
    """同时可视化 Semantic Gaussian（语义着色）和 GT Occupancy（灰色）。

    Args:
        save_dir:        保存目录
        gaussian_data:   GaussianPrediction 对象
        gt_occ:          (H, W, D) GT 占用/语义标签
        name:            文件名前缀
        scalar:          Gaussian 缩放系数
        ignore_opa:      是否忽略透明度
        filter_zsize:    是否过滤 Z 轴异常值
        dataset:         数据集名称
        show_window:     是否显示交互式窗口
        max_gaussians:   最大高斯点数
        max_voxels:      最大体素数
    """
    print(f"[save_gaussian_with_gt_occ] {name}")

    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    if dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
    else:
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]

    # ========== Part 1: 构建 Gaussian 椭球体网格（与 save_gaussian 一致） ==========
    if len(gaussian_data.means) > 0:
        means = gaussian_data.means[0].detach().cpu().numpy()
        scales = gaussian_data.scales[0].detach().cpu().numpy()
        rotations = gaussian_data.rotations[0].detach().cpu().numpy()
    else:
        print("⚠ 没有高斯数据可处理")
        means = np.array([])

    if len(means) > 0 and gaussian_data.semantics.shape[0]:
        sems = gaussian_data.semantics[0].detach().cpu().numpy()
        pred = np.argmax(sems, axis=-1)
    elif len(means) > 0:
        pred = np.ones(len(means))
    else:
        pred = np.array([])

    if len(means) > 0:
        if ignore_opa:
            opas = np.ones(len(means))
        else:
            if gaussian_data.opacities.shape[0]:
                opas = gaussian_data.opacities[0].squeeze().detach().cpu().numpy()
            else:
                opas = np.ones(len(means))
        mask = (pred != empty_label)
        if filter_zsize:
            zdist, zbins = np.histogram(means[:, 2], bins=min(100, len(means)))
            zidx = np.argsort(zdist)[::-1]
            for idx2 in zidx[:10]:
                binl = zbins[idx2]
                binr = zbins[idx2 + 1]
                zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
                mask = mask & zmsk
            z_small_mask = scales[:, 2] > 0.1
            mask = z_small_mask & mask
        means = means[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opas = opas[mask]
        pred = pred[mask]

    print(f"[save_gaussian_with_gt_occ] 有效高斯点数量: {len(means)}")

    # 构建 Gaussian 椭球体合并网格
    if len(means) > 0:
        resolution = 16
        template_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
        base_vertices = np.asarray(template_sphere.vertices, dtype=np.float32)
        base_triangles = np.asarray(template_sphere.triangles, dtype=np.int32)

        all_vertices = []
        all_triangles = []
        all_colors = []
        vertex_offset = 0

        for idx in range(min(len(means), max_gaussians)):
            center = means[idx]
            radii = scales[idx] * scalar
            rot_matrix = Quaternion(rotations[idx]).rotation_matrix
            base_color = sem_cmap[pred[idx]][:3].copy()
            if np.allclose(base_color, [1.0, 1.0, 1.0], atol=0.1):
                continue
            # 根据不透明度混入白色背景：透明度越低 → 越接近白色
            opa_val = float(opas[idx])
            opa_val = np.clip(opa_val, 0.0, 1.0)
            color = base_color * opa_val + np.array([1.0, 1.0, 1.0]) * (1.0 - opa_val)
            transformed_vertices = base_vertices * radii
            transformed_vertices = np.dot(transformed_vertices, rot_matrix.T)
            transformed_vertices += center
            all_vertices.append(transformed_vertices)
            all_triangles.append(base_triangles + vertex_offset)
            all_colors.append(np.tile(color, (len(base_vertices), 1)))
            vertex_offset += len(base_vertices)

        gaussian_mesh_available = len(all_vertices) > 0
        if gaussian_mesh_available:
            all_vertices = np.vstack(all_vertices)
            all_triangles = np.vstack(all_triangles)
            all_colors = np.vstack(all_colors)
            gaussian_mesh = o3d.geometry.TriangleMesh()
            gaussian_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
            gaussian_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
            gaussian_mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)
            gaussian_mesh.compute_vertex_normals()
            print(f"[save_gaussian_with_gt_occ] Gaussian网格：顶点 {len(all_vertices)}，面片 {len(all_triangles)}")
        else:
            print("⚠ 没有可渲染的高斯球（可能全被过滤为白色）")
    else:
        gaussian_mesh_available = False

    # ========== Part 2: 构建 GT Occupancy 语义着色 VoxelGrid（与 save_occ 一致）==========
    if isinstance(gt_occ, torch.Tensor):
        gt_np = gt_occ.cpu().numpy()
    else:
        gt_np = gt_occ
    if gt_np.ndim == 4:
        gt_np = gt_np[0]

    grid_coords = get_grid_coords(gt_np.shape, voxel_size)
    grid_coords = grid_coords + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    gt_flat = gt_np.reshape(-1)

    if dataset == 'nusc':
        gt_mask = (gt_flat >= 0) & (gt_flat < 17)
    elif dataset == 'kitti360':
        gt_mask = (gt_flat > 0) & (gt_flat < 19)
    else:
        gt_mask = (gt_flat > 0) & (gt_flat < 20)

    gt_coords = grid_coords[gt_mask]
    gt_values = gt_flat[gt_mask]
    print(f"[save_gaussian_with_gt_occ] GT 占用体素数: {len(gt_coords)}")

    if len(gt_coords) > max_voxels:
        print(f"⚠ 体素数量过多 ({len(gt_coords)})，采样到 {max_voxels}")
        indices = np.random.choice(len(gt_coords), max_voxels, replace=False)
        gt_coords = gt_coords[indices]
        gt_values = gt_values[indices]

    # 获取语义颜色映射（与 save_occ / create_voxel_grid_from_occupancy 一致）
    occ_colormap = get_nuscenes_colormap()

    vox_origin_arr = np.array(vox_origin, dtype=np.float64)
    voxel_size_f = float(voxel_size[0])
    half_voxel = np.array(voxel_size, dtype=np.float64) / 2.0

    occ_voxel_grid = o3d.geometry.VoxelGrid()
    occ_voxel_grid.origin = vox_origin_arr
    occ_voxel_grid.voxel_size = voxel_size_f

    occ_voxels_available = len(gt_coords) > 0
    if occ_voxels_available:
        for i in range(len(gt_coords)):
            grid_idx_arr = ((gt_coords[i] - vox_origin_arr - half_voxel) / voxel_size_f).astype(np.int32)
            sem_value = int(gt_values[i])
            if 0 <= sem_value < len(occ_colormap):
                color = occ_colormap[sem_value][:3].tolist()
            else:
                color = [0.5, 0.5, 0.5]
            voxel = o3d.geometry.Voxel(grid_idx_arr.tolist(), color)
            occ_voxel_grid.add_voxel(voxel)
        print(f"[save_gaussian_with_gt_occ] GT VoxelGrid 创建完成，含 {len(gt_coords)} 个语义着色体素")
    else:
        print("⚠ GT 中没有占用体素")

    # ========== Part 3: 合并可视化 ==========
    if not gaussian_mesh_available and not occ_voxels_available:
        print("⚠ 没有可可视化的内容，跳过")
        return

    vis = o3d.visualization.Visualizer()
    if show_window:
        vis.create_window(window_name=f"Gaussian + GT Occupancy: {name}", width=1200, height=800)
    else:
        vis.create_window(width=2560, height=1440, visible=False)

    if gaussian_mesh_available:
        vis.add_geometry(gaussian_mesh)
    if occ_voxels_available:
        vis.add_geometry(occ_voxel_grid)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])
    render_option.show_coordinate_frame = True
    if hasattr(render_option, 'voxel_grid_show_wireframe'):
        render_option.voxel_grid_show_wireframe = True

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.3)

    vis.poll_events()
    vis.update_renderer()

    filepath = os.path.join(save_dir, f'{name}_gaussian_occ.png')
    vis.capture_screen_image(filepath)
    print(f"  ✓ 截图保存到: {filepath}")

    if show_window:
        print("  🖱️  交互式窗口已打开，按 'Q' 或关闭窗口继续")
        vis.run()

    vis.destroy_window()
    del occ_voxel_grid
    if gaussian_mesh_available:
        del gaussian_mesh
    clear_memory()
    print(f"[save_gaussian_with_gt_occ] 完成 {name}")


def vis_gaussian_occ_match(save_dir, gaussian_data, gt_occ, name,
                           scalar=1.5, ignore_opa=False, filter_zsize=False,
                           dataset='nusc', show_window=True,
                           max_gaussians=25600, max_voxels=50000,
                           adaptive_color=False, adaptive_color_seed=42):
    """可视化高斯球与GT Occupancy的几何匹配程度。

    只展示能够几何包裹住某个/某些GT Occupancy网格中心的高斯球。
    对每个保留的高斯球，若其包裹的GT网格多数语义与该高斯球语义一致 → 语义颜色；
    否则 → 深灰色 ([0.3, 0.3, 0.3])。
    同时正常展示GT Occupancy VoxelGrid（语义着色）。

    Args:
        save_dir:        保存目录
        gaussian_data:   GaussianPrediction 对象
        gt_occ:          (H, W, D) GT 占用/语义标签
        name:            文件名前缀
        scalar:          Gaussian 缩放系数（同时影响包裹判定半径与可视化椭球大小）
        ignore_opa:      是否忽略透明度过滤
        filter_zsize:    是否过滤 Z 轴异常高斯点
        dataset:         数据集名称
        show_window:     是否显示交互式窗口
        max_gaussians:   最大高斯点数（超出则随机采样）
        max_voxels:      最大 GT 体素数（超出则随机采样）
    """
    print(f"[vis_gaussian_occ_match] {name}")

    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    if dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
    else:
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]

    # ==================================================================
    # 子任务2 前序: 提取并预处理高斯参数（与 save_gaussian_with_gt_occ 一致）
    # ==================================================================
    if len(gaussian_data.means) > 0:
        means = gaussian_data.means[0].detach().cpu().numpy()
        scales = gaussian_data.scales[0].detach().cpu().numpy()
        rotations = gaussian_data.rotations[0].detach().cpu().numpy()
    else:
        print("⚠ 没有高斯数据可处理")
        means = np.array([])

    if len(means) > 0 and gaussian_data.semantics.shape[0]:
        sems = gaussian_data.semantics[0].detach().cpu().numpy()
        pred = np.argmax(sems, axis=-1)
    elif len(means) > 0:
        pred = np.ones(len(means))
    else:
        pred = np.array([])

    if len(means) > 0:
        if ignore_opa:
            opas = np.ones(len(means))
        else:
            if gaussian_data.opacities.shape[0]:
                opas = gaussian_data.opacities[0].squeeze().detach().cpu().numpy()
            else:
                opas = np.ones(len(means))
        mask = (pred != empty_label)
        if filter_zsize:
            zdist, zbins = np.histogram(means[:, 2], bins=min(100, len(means)))
            zidx = np.argsort(zdist)[::-1]
            for idx2 in zidx[:10]:
                binl = zbins[idx2]
                binr = zbins[idx2 + 1]
                zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
                mask = mask & zmsk
            z_small_mask = scales[:, 2] > 0.1
            mask = z_small_mask & mask
        means = means[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opas = opas[mask]
        pred = pred[mask]

    print(f"[vis_gaussian_occ_match] 有效高斯点数量: {len(means)}")

    if len(means) > max_gaussians:
        print(f"⚠ 高斯点数量过多 ({len(means)})，采样到 {max_gaussians}")
        indices = np.random.choice(len(means), max_gaussians, replace=False)
        means = means[indices]
        scales = scales[indices]
        rotations = rotations[indices]
        opas = opas[indices]
        pred = pred[indices]

    # ==================================================================
    # 子任务1: GT Occupancy 网格中心点提取
    # ==================================================================
    if isinstance(gt_occ, torch.Tensor):
        gt_np = gt_occ.cpu().numpy()
    else:
        gt_np = gt_occ
    if gt_np.ndim == 4:
        gt_np = gt_np[0]

    grid_coords = get_grid_coords(gt_np.shape, voxel_size)
    grid_coords = grid_coords + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    gt_flat = gt_np.reshape(-1)

    # 有效（非空）体素掩码
    if dataset == 'nusc':
        gt_mask = (gt_flat >= 0) & (gt_flat < 17)
    elif dataset == 'kitti360':
        gt_mask = (gt_flat > 0) & (gt_flat < 19)
    else:
        gt_mask = (gt_flat > 0) & (gt_flat < 20)

    gt_centers = grid_coords[gt_mask]      # (N_gt, 3)  世界坐标系
    gt_labels = gt_flat[gt_mask]           # (N_gt,)    语义标签（整数）
    print(f"[vis_gaussian_occ_match] GT 占用体素数: {len(gt_centers)}")

    if len(gt_centers) > max_voxels:
        print(f"⚠ GT体素数量过多 ({len(gt_centers)})，采样到 {max_voxels}")
        indices = np.random.choice(len(gt_centers), max_voxels, replace=False)
        gt_centers = gt_centers[indices]
        gt_labels = gt_labels[indices]

    # ==================================================================
    # 子任务2+3: 几何包含判断（马氏距离） + 高斯球筛选
    #
    # 椭球方程: (p - μ)ᵀ·R·Σ⁻²·Rᵀ·(p - μ) ≤ 1
    # 其中 Σ = diag(scales * scalar)
    #
    # 计算流程（对每个高斯球 g）:
    #   delta = gt_centers - means[g]             ← 世界坐标系下的偏移
    #   delta_local = delta @ R_g                 ← 变换到椭球局部坐标系
    #   maha_sq = Σ_j (delta_local[:,j] / radii[j])²
    #   inside = maha_sq ≤ 1.0
    # ==================================================================
    containment_map = {}  # g_idx → np.ndarray of gt_indices
    n_contained_total = 0

    if len(means) > 0 and len(gt_centers) > 0:
        for g_idx in range(len(means)):
            center = means[g_idx]
            radii = scales[g_idx] * scalar                     # (3,)  缩放后的椭球半径
            rot_matrix = Quaternion(rotations[g_idx]).rotation_matrix  # (3,3) local→world

            # 世界坐标 → 局部坐标: p_local = Rᵀ · (p_world - μ)
            delta = gt_centers - center                        # (N_gt, 3)
            delta_local = delta @ rot_matrix                   # (N_gt, 3)  = Rᵀ @ deltaᵀ 再转置

            # 马氏距离平方（归一化后的欧氏距离平方）
            maha_sq = np.sum((delta_local / radii) ** 2, axis=1)  # (N_gt,)

            inside = maha_sq <= 1.0
            gt_indices = np.where(inside)[0]

            if len(gt_indices) > 0:
                containment_map[g_idx] = gt_indices
                n_contained_total += len(gt_indices)

    filtered_indices = list(containment_map.keys())
    print(f"[vis_gaussian_occ_match] 包裹GT的高斯球: {len(filtered_indices)} / {len(means)}"
          f"  (共包裹 {n_contained_total} 个GT网格中心)")

    # ==================================================================
    # 子任务4: 语义匹配判定与着色
    #
    # 判定规则:
    #   - 取被包裹GT中心的语义标签众数 majority_label
    #   - 若 gaussian_sem == majority_label → 语义颜色
    #   - 否则 → 深灰色 [0.3, 0.3, 0.3]
    # ==================================================================
    filtered_colors = []
    match_count = 0
    matched_gt_set = set()     # 被"语义匹配"的高斯球覆盖的 GT 网格中心索引集合

    if len(filtered_indices) > 0:
        for g_idx in filtered_indices:
            gt_idx_list = containment_map[g_idx]
            wrapped_labels = gt_labels[gt_idx_list]

            # 多数投票：众数语义标签
            unique_labels, counts = np.unique(wrapped_labels, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]

            gaussian_sem = pred[g_idx]

            if gaussian_sem == majority_label:
                # 语义匹配 → 使用对应语义类别的颜色
                color = sem_cmap[gaussian_sem][:3].copy()
                match_count += 1
                # 记录该匹配高斯球所覆盖的全部 GT 网格中心
                matched_gt_set.update(gt_idx_list.tolist())
            else:
                # 语义不匹配 → 深灰色
                color = np.array([0.3, 0.3, 0.3], dtype=np.float32)

            filtered_colors.append(color)

        # 统计信息输出
        n_total_gaussians = len(means)
        n_total_gt = len(gt_centers)
        n_matched_gaussians = match_count
        n_matched_gt = len(matched_gt_set)

        pct_matched_gaussians = (n_matched_gaussians / n_total_gaussians * 100.0) if n_total_gaussians > 0 else 0.0
        pct_matched_gt = (n_matched_gt / n_total_gt * 100.0) if n_total_gt > 0 else 0.0

        print(f"[vis_gaussian_occ_match] 语义匹配统计:")
        print(f"  匹配高斯球: {n_matched_gaussians} / {n_total_gaussians}  ({pct_matched_gaussians:.1f}%)")
        print(f"  被匹配GT网格: {n_matched_gt} / {n_total_gt}  ({pct_matched_gt:.1f}%)")

    # ==================================================================
    # 子任务5 Part A: 构建筛选后的高斯椭球体合并网格
    # ==================================================================
    gaussian_mesh_available = False
    if len(filtered_indices) > 0:
        resolution = 16
        template_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
        base_vertices = np.asarray(template_sphere.vertices, dtype=np.float32)
        base_triangles = np.asarray(template_sphere.triangles, dtype=np.int32)

        all_vertices = []
        all_triangles = []
        all_colors = []
        vertex_offset = 0

        for idx_in_filtered, g_idx in enumerate(filtered_indices):
            center = means[g_idx]
            radii = scales[g_idx] * scalar
            rot_matrix = Quaternion(rotations[g_idx]).rotation_matrix
            base_color = filtered_colors[idx_in_filtered].copy()

            # 根据不透明度混入白色背景：透明度越低 → 越接近白色
            opa_val = float(opas[g_idx])
            opa_val = np.clip(opa_val, 0.0, 1.0)
            color = base_color * opa_val + np.array([1.0, 1.0, 1.0]) * (1.0 - opa_val)

            # 缩放 → 旋转 → 平移
            transformed_vertices = base_vertices * radii
            transformed_vertices = np.dot(transformed_vertices, rot_matrix.T)
            transformed_vertices += center

            all_vertices.append(transformed_vertices)
            all_triangles.append(base_triangles + vertex_offset)
            all_colors.append(np.tile(color, (len(base_vertices), 1)))
            vertex_offset += len(base_vertices)

        if len(all_vertices) > 0:
            all_vertices = np.vstack(all_vertices)
            all_triangles = np.vstack(all_triangles)
            all_colors = np.vstack(all_colors)
            gaussian_mesh = o3d.geometry.TriangleMesh()
            gaussian_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
            gaussian_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
            gaussian_mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)
            gaussian_mesh.compute_vertex_normals()
            gaussian_mesh_available = True
            print(f"[vis_gaussian_occ_match] Gaussian网格：顶点 {len(all_vertices)}，面片 {len(all_triangles)}")

    # ==================================================================
    # 子任务5 Part B: 构建 GT Occupancy VoxelGrid（语义着色，与 save_gaussian_with_gt_occ 一致）
    # ==================================================================
    occ_colormap = get_nuscenes_colormap()
    vox_origin_arr = np.array(vox_origin, dtype=np.float64)
    voxel_size_f = float(voxel_size[0])
    half_voxel = np.array(voxel_size, dtype=np.float64) / 2.0

    occ_voxel_grid = o3d.geometry.VoxelGrid()
    occ_voxel_grid.origin = vox_origin_arr
    occ_voxel_grid.voxel_size = voxel_size_f

    occ_voxels_available = len(gt_centers) > 0
    if occ_voxels_available:
        for i in range(len(gt_centers)):
            grid_idx_arr = ((gt_centers[i] - vox_origin_arr - half_voxel) / voxel_size_f).astype(np.int32)
            sem_value = int(gt_labels[i])
            if 0 <= sem_value < len(occ_colormap):
                color = occ_colormap[sem_value][:3].tolist()
            else:
                color = [0.5, 0.5, 0.5]
            voxel = o3d.geometry.Voxel(grid_idx_arr.tolist(), color)
            occ_voxel_grid.add_voxel(voxel)
        print(f"[vis_gaussian_occ_match] GT VoxelGrid 创建完成，含 {len(gt_centers)} 个语义着色体素")
    else:
        print("⚠ GT 中没有占用体素")

    # ==================================================================
    # 子任务5 Part C: 合并可视化 + Open3D 渲染
    # ==================================================================
    if not gaussian_mesh_available and not occ_voxels_available:
        print("⚠ 没有可可视化的内容，跳过")
        return

    vis = o3d.visualization.Visualizer()
    if show_window:
        vis.create_window(window_name=f"Gaussian-GT Match: {name}", width=1200, height=800)
    else:
        vis.create_window(width=2560, height=1440, visible=False)

    if gaussian_mesh_available:
        vis.add_geometry(gaussian_mesh)
    if occ_voxels_available:
        vis.add_geometry(occ_voxel_grid)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])
    render_option.show_coordinate_frame = True
    if hasattr(render_option, 'voxel_grid_show_wireframe'):
        render_option.voxel_grid_show_wireframe = True

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.3)

    vis.poll_events()
    vis.update_renderer()

    filepath = os.path.join(save_dir, f'{name}_gaussian_match.png')
    vis.capture_screen_image(filepath)
    print(f"  ✓ 截图保存到: {filepath}")

    if show_window:
        print("  🖱️  交互式窗口已打开，按 'Q' 或关闭窗口继续")
        vis.run()

    vis.destroy_window()
    if gaussian_mesh_available:
        del gaussian_mesh
    del occ_voxel_grid
    clear_memory()
    print(f"[vis_gaussian_occ_match] 完成 {name}")


def create_ellipsoid(center, radii, rotation, color, opacity=1.0, resolution=4):
    """创建椭球体网格 - 优化内存版本"""
    # 创建单位球体 - 使用更低的分辨率减少内存使用
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    
    # 缩放为椭球体
    vertices = np.asarray(sphere.vertices)
    vertices = vertices * radii
    sphere.vertices = o3d.utility.Vector3dVector(vertices)
    
    # 旋转
    sphere.rotate(rotation, center=(0, 0, 0))
    
    # 平移
    sphere.translate(center)
    
    # 设置颜色和透明度
    sphere.paint_uniform_color(color[:3])  # 只使用RGB
    
    return sphere


@memory_monitor
def save_gaussian(
        save_dir, gaussian_data, name, scalar=1.5, ignore_opa=False,
        filter_zsize=False, show_window=True, max_gaussians=25600,
        adaptive_color=False, adaptive_color_seed=42,
        allocation_color=False, allocation_op_ids=None,
        allocation_color_map=None):
    print(f"[save_gaussian] 开始处理 {name}")

    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    # ---------- 提取高斯参数 ----------
    if len(gaussian_data.means) > 0:
        means = gaussian_data.means[0].detach().cpu().numpy()
        scales = gaussian_data.scales[0].detach().cpu().numpy()
        rotations = gaussian_data.rotations[0].detach().cpu().numpy()
    else:
        print("⚠ 没有高斯数据可处理")
        return

    if gaussian_data.opacities.shape[0]:
        opas = gaussian_data.opacities[0]
        if opas.numel() == 0:
            opas = torch.ones_like(gaussian_data.means[0][..., :1]) if gaussian_data.means else torch.ones(1, 1)
        opas = opas.squeeze().detach().cpu().numpy()
    else:
        opas = np.array([1.0])

    if gaussian_data.semantics.shape[0]:
        sems = gaussian_data.semantics[0].detach().cpu().numpy()
        pred = np.argmax(sems, axis=-1)
    else:
        pred = np.ones(len(means)) if len(means) > 0 else np.array([])

    allocation_ids = None
    if allocation_color and allocation_op_ids is not None:
        allocation_ids = np.asarray(allocation_op_ids).reshape(-1)
        if len(allocation_ids) != len(means):
            print(
                f"⚠ allocation_op_ids 数量({len(allocation_ids)})与高斯数量({len(means)})不一致，关闭操作着色")
            allocation_ids = None

    # 过滤条件
    if ignore_opa and not adaptive_color and allocation_ids is None:
        opas[:] = 1.
    mask = (pred != empty_label)

    if filter_zsize:
        if len(means) > 0:
            zdist, zbins = np.histogram(means[:, 2], bins=min(100, len(means)))
            zidx = np.argsort(zdist)[::-1]
            for idx in zidx[:10]:
                binl = zbins[idx]
                binr = zbins[idx + 1]
                zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
                mask = mask & zmsk
            z_small_mask = scales[:, 2] > 0.1
            mask = z_small_mask & mask

    if len(means) > 0:
        means = means[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opas = opas[mask]
        pred = pred[mask]
        if allocation_ids is not None:
            allocation_ids = allocation_ids[mask]

    print(f"[save_gaussian] 有效高斯点数量: {len(means)}")
    if len(means) == 0:
        print("⚠ 没有有效的高斯点可可视化")
        return

    if len(means) > max_gaussians:
        print(f"⚠ 高斯点数量过多 ({len(means)})，进行采样到 {max_gaussians}")
        indices = np.random.choice(len(means), max_gaussians, replace=False)
        means = means[indices]
        scales = scales[indices]
        rotations = rotations[indices]
        opas = opas[indices]
        pred = pred[indices]
        if allocation_ids is not None:
            allocation_ids = allocation_ids[indices]

    adaptive_colors = None
    allocation_colors = None
    if allocation_ids is not None:
        cmap = allocation_color_map or {
            0: [0.55, 0.55, 0.55],
            1: [0.05, 0.20, 0.95],
            2: [0.15, 0.65, 1.00],
            3: [1.00, 0.10, 0.10],
            4: [1.00, 0.85, 0.05],
        }
        allocation_colors = np.array(
            [cmap.get(int(op_id), [1.0, 1.0, 1.0]) for op_id in allocation_ids],
            dtype=np.float32)
    elif adaptive_color:
        adaptive_colors = get_adaptive_gaussian_colors(
            pred, scales, opas, seed=adaptive_color_seed)

    # ---------- 合并所有椭球体为单个网格 ----------
    resolution = 16
    template_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    base_vertices = np.asarray(template_sphere.vertices, dtype=np.float32)   # (V, 3)
    base_triangles = np.asarray(template_sphere.triangles, dtype=np.int32)   # (F, 3)

    all_vertices = []
    all_triangles = []
    all_colors = []

    vertex_offset = 0

    for idx in range(len(means)):
        center = means[idx]
        radii = scales[idx] * scalar
        rot_matrix = Quaternion(rotations[idx]).rotation_matrix

        if allocation_colors is not None:
            base_color = allocation_colors[idx].copy()
        elif adaptive_colors is not None:
            base_color = adaptive_colors[idx].copy()
        else:
            base_color = sem_cmap[pred[idx]][:3].copy()
        if np.allclose(base_color, [1.0, 1.0, 1.0], atol=0.1):
            continue

        if allocation_colors is not None:
            color = base_color
        elif adaptive_colors is not None:
            color = base_color
        else:
            # 根据不透明度混入白色背景：透明度越低 → 越接近白色
            opa_val = float(opas[idx])
            opa_val = np.clip(opa_val, 0.0, 1.0)
            color = base_color * opa_val + np.array([1.0, 1.0, 1.0]) * (1.0 - opa_val)

        # 缩放 -> 旋转 -> 平移
        transformed_vertices = base_vertices * radii
        transformed_vertices = np.dot(transformed_vertices, rot_matrix.T)
        transformed_vertices += center

        all_vertices.append(transformed_vertices)
        all_triangles.append(base_triangles + vertex_offset)
        all_colors.append(np.tile(color, (len(base_vertices), 1)))

        vertex_offset += len(base_vertices)

    if len(all_vertices) == 0:
        print("⚠ 没有可渲染的高斯球（可能全被过滤为白色）")
        return

    all_vertices = np.vstack(all_vertices)
    all_triangles = np.vstack(all_triangles)
    all_colors = np.vstack(all_colors)

    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    combined_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    combined_mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)
    combined_mesh.compute_vertex_normals()

    print(f"[save_gaussian] 合并网格：顶点数 {len(all_vertices)}，面片数 {len(all_triangles)}")

    # ---------- 可视化（确保 GUI 已初始化）----------
    vis = o3d.visualization.Visualizer()
    
    # 根据参数决定是否显示窗口
    if show_window:
        vis.create_window(window_name=f"3D Gaussian Points: {name}", width=1200, height=800)
        print("✓ 创建交互式3D窗口")
    else:
        vis.create_window(width=2560, height=1440, visible=False)
        print("✓ 使用离屏渲染")
    
    # 添加点云到可视化
    vis.add_geometry(combined_mesh)
    
    # 设置渲染选项 - 调整点的大小以获得更好的可视化效果
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])  # 白色背景
    render_option.point_size = 3.0  # 设置点的大小
    render_option.show_coordinate_frame = True
    
    # 设置相机
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)
    
    # 更新渲染
    vis.poll_events()
    vis.update_renderer()
    
    # 保存截图
    filepath = os.path.join(save_dir, f'{name}_point.png')
    vis.capture_screen_image(filepath)
    print(f"✓ 截图保存到: {filepath}")
    
    # 如果显示窗口，则保持打开状态
    if show_window:
        print("🖱️  交互式3D窗口已打开，可以:")
        print("   - 鼠标拖拽旋转视角")
        print("   - 滚轮缩放")
        print("   - 按 'Q' 或关闭窗口继续")
        vis.run()  # 这会阻塞直到窗口关闭
    
    vis.destroy_window()
    
    # 清理内存
    #del pcd, means, opas, pred, point_colors
    #clear_memory()
    
    print(f"[save_gaussian] 完成 {name}")

def generate_sphere_points(resolution=3):
    """生成单位球面上的27个均匀分布的点"""
    # 创建一个3x3x3的网格点，然后归一化到单位球面
    points = []
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                # 将坐标从[0, resolution-1]映射到[-1, 1]
                x = (i - (resolution-1)/2) * 2 / (resolution-1) if resolution > 1 else 0
                y = (j - (resolution-1)/2) * 2 / (resolution-1) if resolution > 1 else 0
                z = (k - (resolution-1)/2) * 2 / (resolution-1) if resolution > 1 else 0
                
                # 归一化到单位球面
                length = np.sqrt(x*x + y*y + z*z)
                if length > 0:
                    x /= length
                    y /= length
                    z /= length
                points.append([x, y, z])
    
    return np.array(points)


def save_gaussian_point(
        save_dir, gaussian_data, name, scalar=1.5, ignore_opa=False,
        filter_zsize=False, show_window=True, max_gaussians=25600,
        adaptive_color=False, adaptive_color_seed=42,
        allocation_color=False, allocation_op_ids=None,
        allocation_color_map=None):
    """使用Open3D点云的高斯分布3D可视化 - 优化内存版本（每个高斯球用27个点表示形状）"""
    print(f"[save_gaussian_point] 开始处理 {name}")
    
    try:
        import open3d as o3d
    except Exception as e:
        print(f"✗ 无法导入Open3D: {e}")
        return

    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    # 保存高斯属性（可选）- 只在需要时保存
    try:
        if len(gaussian_data.means) > 0 and len(gaussian_data.means[0]) > 0:
            torch.save(gaussian_data, os.path.join(save_dir, f'{name}_attr.pth'))
    except:
        print("⚠ 无法保存高斯属性文件")

    # 提取高斯参数 - 优化内存使用
    if len(gaussian_data.means) > 0:
        means = gaussian_data.means[0].detach().cpu().numpy()
        scales = gaussian_data.scales[0].detach().cpu().numpy()
        rotations = gaussian_data.rotations[0].detach().cpu().numpy()
    else:
        print("⚠ 没有高斯数据可处理")
        return
    
    if gaussian_data.opacities.shape[0]:
        opas = gaussian_data.opacities[0]
        if opas.numel() == 0:
            opas = torch.ones_like(gaussian_data.means[0][..., :1]) if gaussian_data.means else torch.ones(1, 1)
        opas = opas.squeeze().detach().cpu().numpy()
    else:
        opas = np.array([1.0])
    
    if gaussian_data.semantics.shape[0]:
        sems = gaussian_data.semantics[0].detach().cpu().numpy()
        pred = np.argmax(sems, axis=-1)
    else:
        pred = np.ones(len(means)) if len(means) > 0 else np.array([])

    allocation_ids = None
    if allocation_color and allocation_op_ids is not None:
        allocation_ids = np.asarray(allocation_op_ids).reshape(-1)
        if len(allocation_ids) != len(means):
            print(
                f"⚠ allocation_op_ids 数量({len(allocation_ids)})与高斯数量({len(means)})不一致，关闭操作着色")
            allocation_ids = None

    # 过滤条件
    if ignore_opa and not adaptive_color and allocation_ids is None:
        opas[:] = 1.
        mask = (pred != empty_label)
    elif adaptive_color or allocation_ids is not None:
        mask = (pred != empty_label)
    else:
        mask = (pred != empty_label) & (opas > 0.1)

    if filter_zsize:
        if len(means) > 0:
            zdist, zbins = np.histogram(means[:, 2], bins=min(100, len(means)))
            zidx = np.argsort(zdist)[::-1]
            for idx in zidx[:10]:
                binl = zbins[idx]
                binr = zbins[idx + 1]
                zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
                mask = mask & zmsk
            
            z_small_mask = scales[:, 2] > 0.1
            mask = z_small_mask & mask

    if len(means) > 0:
        means = means[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opas = opas[mask]
        pred = pred[mask]
        if allocation_ids is not None:
            allocation_ids = allocation_ids[mask]

    print(f"[save_gaussian_point] 有效高斯点数量: {len(means)}")

    if len(means) == 0:
        print("⚠ 没有有效的高斯点可可视化")
        return

    # 如果高斯点数量过多，进行采样
    if len(means) > max_gaussians:
        print(f"⚠ 高斯点数量过多 ({len(means)})，进行采样到 {max_gaussians}")
        indices = np.random.choice(len(means), max_gaussians, replace=False)
        means = means[indices]
        scales = scales[indices]
        rotations = rotations[indices]
        opas = opas[indices]
        pred = pred[indices]
        if allocation_ids is not None:
            allocation_ids = allocation_ids[indices]

    adaptive_colors = None
    allocation_colors = None
    if allocation_ids is not None:
        cmap = allocation_color_map or {
            0: [0.55, 0.55, 0.55],
            1: [0.05, 0.20, 0.95],
            2: [0.15, 0.65, 1.00],
            3: [1.00, 0.10, 0.10],
            4: [1.00, 0.85, 0.05],
        }
        allocation_colors = np.array(
            [cmap.get(int(op_id), [1.0, 1.0, 1.0]) for op_id in allocation_ids],
            dtype=np.float32)
    elif adaptive_color:
        adaptive_colors = get_adaptive_gaussian_colors(
            pred, scales, opas, seed=adaptive_color_seed)

    # 生成单位球面上的27个点
    sphere_points = generate_sphere_points(resolution=3)
    print(f"[save_gaussian_point] 生成了 {len(sphere_points)} 个球面点")

    # 为每个高斯球创建27个形状点
    all_points = []
    all_colors = []
    
    batch_size = 1000  # 分批处理避免内存峰值
    total_gaussians = len(means)
    
    for batch_start in range(0, total_gaussians, batch_size):
        batch_end = min(batch_start + batch_size, total_gaussians)
        batch_points = []
        batch_colors = []
        
        for idx in range(batch_start, batch_end):
            center = means[idx]
            scale = scales[idx]
            
            # 将四元数转换为旋转矩阵
            rot_matrix = Quaternion(rotations[idx]).rotation_matrix
            
            # 获取基础颜色
            if allocation_colors is not None:
                base_color = allocation_colors[idx].copy()
            elif adaptive_colors is not None:
                base_color = adaptive_colors[idx].copy()
            elif len(pred) > idx:
                base_color = sem_cmap[pred[idx]][:3].copy()  # 只取RGB
            else:
                base_color = sem_cmap[0][:3].copy()  # 默认颜色
            
            # 跳过白色高斯（可选）
            if np.allclose(base_color, [1.0, 1.0, 1.0], atol=0.1):
                continue
            
            if allocation_colors is not None:
                color = base_color
            elif adaptive_colors is not None:
                color = base_color
            else:
                # 根据不透明度混入白色背景：透明度越低 → 越接近白色
                opa_val = float(opas[idx])
                opa_val = np.clip(opa_val, 0.0, 1.0)
                color = base_color * opa_val + np.array([1.0, 1.0, 1.0]) * (1.0 - opa_val)
            
            # 为当前高斯球生成27个形状点
            for sphere_point in sphere_points:
                # 应用缩放
                scaled_point = sphere_point * scale
                # 应用旋转
                rotated_point = np.dot(rot_matrix, scaled_point)
                # 应用平移
                final_point = rotated_point + center
                
                batch_points.append(final_point)
                batch_colors.append(color)
        
        # 添加到总列表
        all_points.extend(batch_points)
        all_colors.extend(batch_colors)
        
        # 清理批次内存
        del batch_points, batch_colors
        clear_memory()
        
        print(f"[save_gaussian_point] 已处理 {min(batch_end, total_gaussians)}/{total_gaussians} 个高斯球")
    
    # 转换为numpy数组
    all_points = np.array(all_points)
    all_colors = np.array(all_colors)
    
    print(f"[save_gaussian_point] 总共生成了 {len(all_points)} 个形状点")

    # 如果点数量过多，进行采样
    if len(all_points) > max_gaussians * 27:
        max_total_points = max_gaussians * 27
        print(f"⚠ 形状点数量过多 ({len(all_points)})，进行采样到 {max_total_points}")
        indices = np.random.choice(len(all_points), max_total_points, replace=False)
        all_points = all_points[indices]
        all_colors = all_colors[indices]

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # 设置点的大小（通过设置渲染选项）
    print(f"[save_gaussian_point] 创建了点云，包含 {len(all_points)} 个形状点")

    # 创建Open3D可视化
    vis = o3d.visualization.Visualizer()
    
    # 根据参数决定是否显示窗口
    if show_window:
        vis.create_window(window_name=f"3D Gaussian Points: {name}", width=1200, height=800)
        print("✓ 创建交互式3D窗口")
    else:
        vis.create_window(width=2560, height=1440, visible=False)
        print("✓ 使用离屏渲染")
    
    # 添加点云到可视化
    vis.add_geometry(pcd)
    
    # 设置渲染选项 - 调整点的大小以获得更好的可视化效果
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])  # 白色背景
    render_option.point_size = 3.0  # 设置点的大小
    render_option.show_coordinate_frame = True
    
    # 设置相机
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)
    
    # 更新渲染
    vis.poll_events()
    vis.update_renderer()
    
    # 保存截图
    filepath = os.path.join(save_dir, f'{name}_point.png')
    vis.capture_screen_image(filepath)
    print(f"✓ 截图保存到: {filepath}")
    
    # 如果显示窗口，则保持打开状态
    if show_window:
        print("🖱️  交互式3D窗口已打开，可以:")
        print("   - 鼠标拖拽旋转视角")
        print("   - 滚轮缩放")
        print("   - 按 'Q' 或关闭窗口继续")
        vis.run()  # 这会阻塞直到窗口关闭
    
    vis.destroy_window()
    
    # 清理内存
    #del pcd, means, opas, pred, point_colors
    #clear_memory()
    
    print(f"[save_gaussian_point] 完成 {name}")

def save_gaussian_topdown(save_dir, anchor_init, gaussian, name):
    """高斯俯视图可视化"""
    print(f"[save_gaussian_topdown] 开始处理 {name}")
    
    # 修复数据类型问题
    if isinstance(anchor_init, np.ndarray):
        anchor_init_tensor = torch.from_numpy(anchor_init)
    else:
        anchor_init_tensor = anchor_init
    
    # 确保使用修复后的safe_sigmoid
    init_means = safe_sigmoid(anchor_init_tensor[:, :2]) * 100 - 50
    
    # 处理means列表
    means = [init_means]
    if gaussian is not None:
        for g in gaussian:
            if hasattr(g, 'means') and g.means:
                g_means = g.means[0]
                if isinstance(g_means, torch.Tensor):
                    g_means = g_means.detach().cpu().numpy()
                # 确保有足够的数据点
                if g_means.shape[0] > 0 and g_means.shape[1] >= 2:
                    means.append(g_means[:, :2])  # 取前两列

    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=(24., 16.))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, 5),
                    axes_pad=0.,
                    share_all=True
                    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    
    for ax, im in zip(grid, means):
        if isinstance(im, torch.Tensor):
            im = im.cpu().numpy()
        ax.scatter(im[:, 0], im[:, 1], s=0.1, marker='o')
    
    filepath = os.path.join(save_dir, f"{name}.jpg")
    plt.savefig(filepath)
    plt.clf()
    plt.cla()
    
    print(f"[save_gaussian_topdown] 完成 {name}")
