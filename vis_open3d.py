# 使用 open3d 替代 mayavi 的可视化工具
import os
import numpy as np
from PIL import Image

def _to_numpy(x):
    try:
        import torch
        has_torch = True
    except Exception:
        has_torch = False

    # Torch tensor -> numpy on CPU
    if has_torch and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    # dict: recursively convert values
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}

    # list/tuple: recursively convert elements, then try to make ndarray if possible
    if isinstance(x, (list, tuple)):
        converted = [_to_numpy(v) for v in x]
        try:
            arr = np.asarray(converted)
            # 当转换后是 object dtype（不能形成规则 ndarray）时，返回与输入类型一致的容器
            if arr.dtype == object:
                return converted if isinstance(x, list) else tuple(converted)
            return arr
        except Exception:
            return converted if isinstance(x, list) else tuple(converted)

    # objects exposing .numpy()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        try:
            return _to_numpy(x.numpy())
        except Exception:
            pass

    # already numpy array
    if isinstance(x, np.ndarray):
        return x

    # scalar or other convertible objects
    try:
        return np.asarray(x)
    except Exception:
        return x
    

def _ensure_xyz(arr):
    """
    将 arr 转为 np.float32 ndarray，shape 为 (N,3) 或 (1,3)。
    返回 None 表示无法转换/无效数据。
    """
    try:
        a = np.asarray(arr)
    except Exception:
        return None
    if a.dtype == object:
        # 尝试将元素逐一转为浮点数三元组
        try:
            lst = []
            for el in arr:
                e = np.asarray(el, dtype=np.float32)
                if e.size == 3:
                    lst.append(e.reshape(3,))
                elif e.ndim == 1 and e.shape[0] >= 3:
                    lst.append(e[:3].astype(np.float32))
                else:
                    return None
            if len(lst) == 0:
                return None
            return np.vstack(lst).astype(np.float32)
        except Exception:
            return None
    # 普通数值数组
    try:
        a = a.astype(np.float32)
    except Exception:
        return None
    if a.ndim == 1:
        if a.size == 3:
            return a.reshape(1, 3)
        else:
            return None
    if a.ndim == 2:
        if a.shape[1] >= 3:
            return a[:, :3].astype(np.float32)
        else:
            return None
    # 其他维度，尝试压扁最后三维
    if a.ndim > 2:
        try:
            flat = a.reshape(-1, a.shape[-1])
            if flat.shape[1] >= 3:
                return flat[:, :3].astype(np.float32)
        except Exception:
            return None
    return None

def _render_and_save(pcd, out_png, width=800, height=600, background=(1.0,1.0,1.0,1.0)):
    try:
        import open3d as o3d
        # 尝试离屏渲染 (new Open3D)
        try:
            from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
            r = OffscreenRenderer(width, height)
            mat = MaterialRecord()
            mat.shader = "defaultUnlit"
            r.scene.add_geometry("pcd", pcd, mat)
            bbox = pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = max(bbox.get_extent()) or 1.0
            r.setup_camera(60.0, center, center + np.array([0, 0, extent*2]), np.array([0,1,0]))
            img = r.render_to_image()
            # img is o3d.geometry.Image
            o3d.io.write_image(out_png, img)
            r.release()
            return True
        except Exception:
            # 回退到 GUI 截图（若有 X server）
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=width, height=height, visible=False)
            vis.add_geometry(pcd)
            ctr = vis.get_view_control()
            bbox = pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = max(bbox.get_extent()) or 1.0
            ctr.set_lookat(center)
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            ctr.set_zoom(0.6)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(out_png, do_render=True)
            vis.destroy_window()
            return True
    except Exception as e:
        # 无法渲染时忽略
        return False

def save_occ(save_dir, occ, name, save_png=True, z_slice=0, dataset='nusc'):
    """
    occ: numpy/torch array with shape (N, X, Y, Z) or (X,Y,Z)
    将非零体素导出为点云并保存 ply（和 png 截图若支持）
    """
    import open3d as o3d
    occ = _to_numpy(occ)
    if occ.ndim == 4:
        occ = occ[0]
    # threshold >0
    pts = np.argwhere(occ > 0.5)
    if pts.size == 0:
        return
    # 将索引转为中心坐标（可以根据实际体素大小更改 scale/offset）
    # 假设坐标系： (x: width, y: height, z: depth)
    coords = pts.astype(np.float32)
    # swap axes to x,y,z order if needed: currently (X,Y,Z) -> (x,y,z)
    # 确保 coords 为 (N,3) float32
    coords = _ensure_xyz(coords)
    if coords is None:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    colors = np.ones_like(coords) * 0.5
    colors = colors / colors.max()
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    os.makedirs(save_dir, exist_ok=True)
    ply_path = os.path.join(save_dir, f'{name}.ply')
    o3d.io.write_point_cloud(ply_path, pcd)
    if save_png:
        png_path = os.path.join(save_dir, f'{name}.png')
        _render_and_save(pcd, png_path)

def save_gaussian(save_dir, gaussian, name, scalar=1.5, ignore_opa=False, filter_zsize=False):
    """
    简化实现：将高斯的中心（如果有）作为点云保存
    支持格式：
      - numpy array or torch tensor with shape (N, >=3) -> use [:, :3] as centers
      - dict with keys 'means' / 'centers'
    """
    import open3d as o3d
    ga = gaussian
    import pdb
    pdb.set_trace()
    ga = _to_numpy(ga)
    if isinstance(ga, (list, tuple)):
        if len(ga) == 0:
            return
        ga = ga[0]

    if isinstance(ga, dict):
        if 'means' in ga:
            centers = _to_numpy(ga['means'])
        elif 'centers' in ga:
            centers = _to_numpy(ga['centers'])
        else:
            return
    else:
        if not hasattr(ga, 'ndim'):
            # 保护性检查，尝试转为 ndarray
            ga = _to_numpy(ga)
        if hasattr(ga, 'ndim') and ga.ndim == 1:
            # 单个高斯
            if ga.size >= 3:
                centers = ga[:3].reshape(1,3)
            else:
                return
        elif hasattr(ga, 'ndim') and ga.ndim >= 2:
            centers = ga[:, :3]
        else:
            return
    # 确保 centers 为 (N,3) float32 ndarray
    centers = _ensure_xyz(centers)
    if centers is None:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    # 如果有权重，使用权重映射颜色
    if isinstance(ga, np.ndarray) and ga.ndim >= 2 and ga.shape[1] >= 4:
        w = ga[:, 3]
        w = (w - w.min()) / (w.ptp() + 1e-9)
        colors = np.stack([1-w, w, 0.2*np.ones_like(w)], axis=1)
    else:
        colors = np.repeat(np.array([[1.0, 0.2, 0.2]]), centers.shape[0], axis=0)
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    os.makedirs(save_dir, exist_ok=True)
    ply_path = os.path.join(save_dir, f'{name}.ply')
    o3d.io.write_point_cloud(ply_path, pcd)
    png_path = os.path.join(save_dir, f'{name}.png')
    _render_and_save(pcd, png_path)

def save_gaussian_topdown(save_dir, anchor_init, gaussians, name):
    """
    简化顶视图：将高斯中心投影到 XY 平面并保存为简单图像（PNG）和点云
    """
    import open3d as o3d
    ga = _to_numpy(gaussians)
    if isinstance(ga, (list, tuple)):
        if len(ga) == 0:
            return
        ga = ga[0]

    if isinstance(ga, dict) and 'means' in ga:
        centers = _to_numpy(ga['means'])
    elif hasattr(ga, 'ndim') and ga.ndim >= 2:
        centers = ga[:, :3]
    else:
        return
    # 确保 centers 为 (N,3) float32 ndarray
    centers = _ensure_xyz(centers)
    if centers is None:
        return
    # topdown projection: use x,y
    pts2d = centers[:, :2]
    # normalize to image coords
    minv = pts2d.min(axis=0)
    maxv = pts2d.max(axis=0)
    span = (maxv - minv) + 1e-6
    img_size = (512, 512)
    grid = np.zeros(img_size + (3,), dtype=np.uint8) + 255
    uv = ((pts2d - minv) / span * (np.array(img_size) - 1)).astype(int)
    for u,v in uv:
        if 0 <= v < img_size[0] and 0 <= u < img_size[1]:
            grid[img_size[0]-1 - v, u] = (255, 50, 50)
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f'{name}.png')
    Image.fromarray(grid).save(img_path)
    # also save point cloud (use x,y,z)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1,0.2,0.2]], centers.shape[0], axis=0))
    ply_path = os.path.join(save_dir, f'{name}.ply')
    o3d.io.write_point_cloud(ply_path, pcd)