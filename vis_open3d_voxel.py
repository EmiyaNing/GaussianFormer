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

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
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
    """从占用数据创建Open3D点云可视化 - 优化内存版本（使用点云代替网格）"""
    # 确保数据是numpy数组
    if isinstance(occ_data, torch.Tensor):
        voxels = occ_data[0].cpu().to(torch.int).numpy()
    else:
        voxels = occ_data[0].astype(np.int32)

    # 设置边界值用于颜色映射
    voxels[0, 0, 0] = 1
    voxels[-1, -1, -1] = 1

    # 计算体素坐标 - 优化内存使用
    grid_coords = get_grid_coords(voxels.shape, voxel_size) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # 获取FOV内的体素
    if sem:
        if dataset == 'nusc':
            fov_voxels = grid_coords[
                (grid_coords[:, 3] >= 0) & (grid_coords[:, 3] < 17)
            ]
        elif dataset == 'kitti360':
            fov_voxels = grid_coords[
                (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 19)
            ]
        else:
            fov_voxels = grid_coords[
                (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 20)
            ]
    else:
        fov_voxels = grid_coords[
            (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 100)
        ]
    
    print(f"[create_voxel_grid] 有效体素数量: {len(fov_voxels)}")
    
    # 如果体素数量过多，进行采样
    if len(fov_voxels) > max_voxels:
        print(f"⚠ 体素数量过多 ({len(fov_voxels)})，进行采样到 {max_voxels}")
        indices = np.random.choice(len(fov_voxels), max_voxels, replace=False)
        fov_voxels = fov_voxels[indices]
    
    # 直接创建点云进行可视化，不再创建网格
    points = fov_voxels[:, :3]
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 为点云着色
    if sem:
        colors = get_nuscenes_colormap()
        point_colors = []
        
        for i in range(len(fov_voxels)):
            sem_value = int(fov_voxels[i, 3])
            if 0 <= sem_value < len(colors):
                color = colors[sem_value][:3]  # 只取RGB
            else:
                color = [0.5, 0.5, 0.5]  # 默认灰色
            point_colors.append(color)
        
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
    else:
        # 非语义模式使用统一颜色
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
    
    # 清理临时变量释放内存
    del grid_coords, fov_voxels
    clear_memory()
    
    return [pcd]


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
def save_gaussian(save_dir, gaussian_data, name, scalar=1.5, ignore_opa=False, filter_zsize=False, show_window=True, max_gaussians=25600):
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

    # 过滤条件
    if ignore_opa:
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

    print(f"[save_gaussian] 有效高斯点数量: {len(means)}")
    if len(means) == 0:
        print("⚠ 没有有效的高斯点可可视化")
        return


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

        color = sem_cmap[pred[idx]][:3]
        if np.allclose(color, [1.0, 1.0, 1.0], atol=0.1):
            continue

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


def save_gaussian_point(save_dir, gaussian_data, name, scalar=1.5, ignore_opa=False, filter_zsize=False, show_window=True, max_gaussians=25600):
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

    # 过滤条件
    if ignore_opa:
        opas[:] = 1.
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
            
            # 获取颜色
            if len(pred) > idx:
                color = sem_cmap[pred[idx]][:3]  # 只取RGB
            else:
                color = sem_cmap[0][:3]  # 默认颜色
            
            # 跳过白色高斯（可选）
            if np.allclose(color, [1.0, 1.0, 1.0], atol=0.1):
                continue
            
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