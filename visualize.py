r"""Visualize OPUS occupancy and semantic Gaussian ellipsoids on Occ3D nuScenes.

Examples (one GPU; explicit --vis-occ/--vis-gaussian also open interactive windows)::

    python3.10 visualize.py --py-config config/mini_occ3d_4096_1024/opusv1.py \
        --checkpoint work_dirs/mini_occ3d_4096_1024/opusv1/epoch_5.pth --vis-occ

    python3.10 visualize.py --py-config config/gf2_probability_decomposition/probability.py \
        --checkpoint work_dirs/gf2_probability_decomposition/train/probability/epoch_5.pth

Without visualization flags, export occupancy and, when available, Gaussians.
Use --no-show for batch PNG export, or --data-only to export NPZ without Open3D.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


# Occ3D labels 0..16 are foreground, including class 0 (others); 17 is empty.
CLASS_NAMES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
)
# Occ3D / nuScenes semantic colors, indexed by CLASS_NAMES above.
# Source: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
COLORS = np.array([
    [0, 0, 0],        # others
    [112, 128, 144],  # barrier
    [220, 20, 60],    # bicycle
    [255, 127, 80],   # bus
    [255, 158, 0],    # car
    [233, 150, 70],   # construction_vehicle
    [255, 61, 99],    # motorcycle
    [0, 0, 230],     # pedestrian
    [47, 79, 79],    # traffic_cone
    [255, 140, 0],   # trailer
    [255, 99, 71],   # truck
    [0, 207, 191],   # driveable_surface
    [175, 0, 75],    # other_flat
    [75, 0, 75],     # sidewalk
    [112, 180, 60],  # terrain
    [222, 184, 135], # manmade
    [0, 175, 0],     # vegetation
], dtype=np.float64) / 255.0


def as_numpy(value):
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def occupancy_data(output, metas, mask='none'):
    """Read final predictions and actual model-frame coordinates without edits."""
    xyz = as_numpy(output.get('sampled_xyz', metas['occ_xyz']))[0].reshape(-1, 3)
    pred = as_numpy(output['final_occ'])[0].reshape(-1).astype(np.int64)
    gt = as_numpy(output.get('sampled_label', metas['occ_label']))[0].reshape(-1).astype(np.int64)
    if not (len(xyz) == len(pred) == len(gt)):
        raise ValueError('final_occ, sampled_xyz and sampled_label must describe the same voxels')
    selected = np.isfinite(xyz).all(-1) & (gt >= 0) & (gt <= 17)
    if mask != 'none':
        key = {'camera': 'occ_cam_mask', 'lidar': 'occ_lidar_mask'}[mask]
        if key not in metas:
            raise ValueError(f'{mask} mask requested, but this dataset does not return {key}')
        selected &= as_numpy(metas[key])[0].reshape(-1).astype(bool)
    return dict(xyz=xyz, prediction=pred, ground_truth=gt, mask=selected)


def gaussian_data(gaussian, valid=None, min_opacity=0.0):
    """Flatten grouped/flat Gaussians; keep semantic colors independent of opacity."""
    arrays = {key: as_numpy(getattr(gaussian, key))[0].reshape(-1, width)
              for key, width in [('means', 3), ('scales', 3), ('rotations', 4)]}
    count = len(arrays['means'])
    sem = as_numpy(gaussian.semantics)
    if sem.shape[-1] not in (17, 18):
        raise ValueError('Expected 17 foreground classes, optionally followed by empty')
    arrays['semantics'] = sem[0].reshape(-1, sem.shape[-1])
    opacity = as_numpy(gaussian.opacities)
    arrays['opacities'] = opacity[0].reshape(-1) if opacity.size else np.ones(count)
    if any(len(array) != count for array in arrays.values()):
        raise ValueError('Gaussian attributes must have the same primitive count')
    arrays['labels'] = arrays['semantics'].argmax(-1)
    selected = np.ones(count, dtype=bool)
    for array in arrays.values():
        if count:
            selected &= np.isfinite(array).reshape(count, -1).all(-1)
    selected &= (arrays['scales'] > 0).all(-1)
    selected &= np.linalg.norm(arrays['rotations'], axis=-1) > 1e-8
    selected &= (arrays['labels'] < 17) & (arrays['opacities'] >= min_opacity)
    if valid is not None:
        selected &= as_numpy(valid)[0].reshape(-1).astype(bool)
    return dict(**{key: value[selected] for key, value in arrays.items()},
                source_indices=np.flatnonzero(selected))


def gaussian_transforms(scales, quaternions):
    """Row-vector transform T with covariance T.T @ T, matching the CUDA head.

    The head uses T = diag(scale) @ R, not R @ diag(scale). Consequently a
    unit sphere vertex u is mapped to u @ T; using R.T here mirrors rotations.
    Quaternions use the model's (w, x, y, z) order.
    """
    q = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    w, x, y, z = q.T
    rotation = np.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ], axis=-1).reshape(-1, 3, 3)
    return scales[:, :, None] * rotation


def voxel_size_from_config(cfg):
    head = cfg.model.head
    size = cfg.get('grid_size', head.get('grid_size', head.get('cuda_kwargs', {}).get('grid_size')))
    if size is None:
        raise ValueError('Set grid_size in the config; voxel size is never inferred from a dataset name')
    size = np.broadcast_to(np.asarray(size, dtype=float), (3,)).copy()
    if not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError('grid_size must contain positive finite voxel dimensions')
    return size


def configure_display():
    """Select X11/XWayland before Open3D initializes GLFW on Linux."""
    if sys.platform.startswith('linux') and os.environ.get('DISPLAY'):
        os.environ.pop('WAYLAND_DISPLAY', None)
        os.environ['XDG_SESSION_TYPE'] = 'x11'


class SceneRenderer:
    """Open3D windows/screenshots with shared framing and semantic colors."""

    def __init__(self, bounds, view='top', show=False):
        configure_display()
        if show and sys.platform.startswith('linux') and not (
                os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('Interactive Open3D requires a desktop display. '
                               'Use --no-show or --data-only for headless export.')
        import open3d as o3d
        self.o3d, self.show = o3d, show
        lower, upper = np.asarray(bounds).reshape(2, 3)
        self.center = (lower + upper) / 2
        padding = max(upper-lower) * .05
        self.bounds = np.stack([lower-padding, upper+padding])
        self.front = [0., 0., 1.] if view == 'top' else [1., -1.4, 1.3]
        self.up = [0, 1, 0] if view == 'top' else [0, 0, 1]

    def mesh(self, template, centers, transforms, labels):
        vertices = np.asarray(template.vertices)
        triangles = np.asarray(template.triangles)
        # Keep per-draw buffers bounded. Every primitive
        # is retained; only the GPU submission is split into smaller meshes.
        chunk = max(1, 200000 // len(vertices))
        if len(centers) > chunk:
            return [self.mesh(template, centers[i:i+chunk], transforms[i:i+chunk], labels[i:i+chunk])
                    for i in range(0, len(centers), chunk)]
        world = np.einsum('vj,njk->nvk', vertices, transforms) + centers[:, None, :]
        faces = triangles[None] + (np.arange(len(centers)) * len(vertices))[:, None, None]
        mesh = self.o3d.geometry.TriangleMesh()
        mesh.vertices = self.o3d.utility.Vector3dVector(world.reshape(-1, 3))
        mesh.triangles = self.o3d.utility.Vector3iVector(faces.reshape(-1, 3))
        mesh.vertex_colors = self.o3d.utility.Vector3dVector(np.repeat(COLORS[labels], len(vertices), axis=0))
        mesh.compute_vertex_normals()
        return mesh

    def occupancy(self, data, key, voxel_size):
        labels = data[key]
        keep = data['mask'] & (labels >= 0) & (labels < 17)
        cube = self.o3d.geometry.TriangleMesh.create_box()
        cube.translate([-.5, -.5, -.5])
        transforms = np.broadcast_to(np.diag(voxel_size), (keep.sum(), 3, 3))
        return self.mesh(cube, data['xyz'][keep], transforms, labels[keep])

    def gaussians(self, data, sigma, points=False):
        if points:
            cloud = self.o3d.geometry.PointCloud()
            cloud.points = self.o3d.utility.Vector3dVector(data['means'])
            cloud.colors = self.o3d.utility.Vector3dVector(COLORS[data['labels']])
            return cloud
        sphere = self.o3d.geometry.TriangleMesh.create_sphere(resolution=4)
        transform = gaussian_transforms(data['scales'] * sigma, data['rotations'])
        return self.mesh(sphere, data['means'], transform, data['labels'])

    def open_window(self, parts, title, visible):
        window = self.o3d.visualization.Visualizer()
        if not window.create_window(window_name=title, width=1280, height=960, visible=visible):
            raise RuntimeError('Open3D could not create a window. Set DISPLAY for interaction; '
                               'for headless export use xvfb-run -a ... --no-show, or --data-only.')
        for part in parts:
            window.add_geometry(part)
        # White bounds points only stabilize framing; they are not predictions.
        # This also gives an empty scene the same camera as a nonempty scene.
        frame = self.o3d.geometry.PointCloud()
        frame.points = self.o3d.utility.Vector3dVector(self.bounds)
        frame.colors = self.o3d.utility.Vector3dVector(np.ones((2, 3)))
        window.add_geometry(frame)
        options = window.get_render_option()
        options.background_color = np.ones(3)
        options.light_on = False
        options.point_size = 3.
        window.poll_events()
        view = window.get_view_control()
        view.set_lookat(self.center)
        view.set_front(self.front)
        view.set_up(self.up)
        view.set_zoom(.7)
        return window

    def save(self, geometry, path):
        parts = geometry if isinstance(geometry, list) else [geometry]
        parts = [part for part in parts if not part.is_empty()]
        window = self.open_window(parts, path.stem, visible=self.show)
        try:
            window.poll_events()
            window.update_renderer()
            window.capture_screen_image(str(path), do_render=True)
            if self.show:
                print(f'Open3D: {path.stem}; drag to rotate, scroll to zoom, Q to continue.', flush=True)
                window.run()
        finally:
            window.destroy_window()
        if not path.is_file():
            raise RuntimeError(f'Could not write {path}')


def render_file(folder, name, bounds, view, show, sigma, points):
    """Fresh graphics process, isolated from the model's CUDA context."""
    folder = Path(folder)
    source = 'gaussians' if name == 'gaussians' else 'occupancy'
    with np.load(folder / f'{source}.npz') as archive:
        data = dict(archive)
    renderer = SceneRenderer(bounds, view, show)
    if source == 'gaussians':
        geometry = renderer.gaussians(data, sigma, points)
    else:
        key = 'prediction' if name == 'occupancy' else 'ground_truth'
        geometry = renderer.occupancy(data, key, data['voxel_size'])
    renderer.save(geometry, folder / f'{name}.png')


def render_image(folder, name, bounds, args):
    params = dict(folder=str(folder.resolve()), name=name, bounds=np.asarray(bounds).tolist(),
                  view=args.view, show=args.show, sigma=args.gaussian_sigma,
                  points=args.vis_gaussian_point)
    subprocess.run([sys.executable, '-c',
                    'import json, sys; from visualize import render_file; render_file(**json.loads(sys.argv[1]))',
                    json.dumps(params)], cwd=Path(__file__).resolve().parent, check=True)


def load_model(cfg, checkpoint, device):
    import model  # noqa: F401 -- register the project models
    from mmseg.models import build_segmentor
    network = build_segmentor(cfg.model)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = state.get('state_dict', state)
    if state and all(key.startswith('module.') for key in state):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
    # A mismatched/backbone-only checkpoint must not silently visualize random heads.
    network.load_state_dict(state, strict=True)
    return network.to(device).eval().requires_grad_(False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--checkpoint', '--resume-from', dest='checkpoint',
                        help='Trained checkpoint; defaults to WORK_DIR/latest.pth')
    parser.add_argument('--work-dir', default='out/visualize')
    parser.add_argument('--vis-index', nargs='+', type=int,
                        help='Validation dataset indices; otherwise take the first NUM_SAMPLES')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--vis-occ', action='store_true', help='Export predicted and GT occupancy')
    parser.add_argument('--vis-gaussian', action='store_true', help='Export semantic Gaussian ellipsoids')
    parser.add_argument('--vis-gaussian-point', action='store_true', help='Display Gaussian centers instead of ellipsoids')
    parser.add_argument('--vis-opus-points', action='store_true', help='Also export OPUS sparse points to NPZ')
    parser.add_argument('--mask', choices=['auto', 'camera', 'lidar', 'none'], default='auto',
                        help='Occupancy only; auto follows occ3d_eval_mask in the config')
    parser.add_argument('--gaussian-sigma', type=float, default=2., help='Ellipsoid radius in standard deviations')
    parser.add_argument('--min-opacity', type=float, default=0., help='Optional Gaussian filter; default keeps all valid primitives')
    parser.add_argument('--view', choices=['top', 'perspective'], default='top')
    display = parser.add_mutually_exclusive_group()
    display.add_argument('--show', dest='show', action='store_true',
                         help='Open interactive windows; default for explicit visualization flags')
    display.add_argument('--no-show', dest='show', action='store_false', help='Save PNGs without interactive windows')
    parser.set_defaults(show=None)
    parser.add_argument('--data-only', action='store_true', help='Export NPZ without loading Open3D')
    parser.add_argument('--device', default='cuda:0', help='Single inference device; no automatic DDP')
    parser.add_argument('--seed', type=int, default=928)
    args = parser.parse_args(argv)
    if args.num_samples < 1 or not np.isfinite(args.gaussian_sigma) or args.gaussian_sigma <= 0:
        parser.error('num-samples and gaussian-sigma must be positive')
    if not 0 <= args.min_opacity <= 1:
        parser.error('min-opacity must be in [0, 1]')
    if args.show is None:
        args.show = not args.data_only and (args.vis_occ or args.vis_gaussian or args.vis_gaussian_point)
    elif args.show and args.data_only:
        parser.error('--show requires image rendering; remove --data-only')
    return args


def main(args):
    if not args.data_only:
        configure_display()
    from mmengine import Config
    from mmengine.runner import set_random_seed
    from torch.utils.data import DataLoader, Subset
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal

    set_random_seed(args.seed)
    cfg = Config.fromfile(args.py_config)
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', False)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.work_dir) / 'latest.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Trained checkpoint not found: {checkpoint}. Supply --checkpoint.')
    mask = cfg.get('occ3d_eval_mask', 'none') if args.mask == 'auto' else args.mask
    if mask not in ('none', 'camera', 'lidar'):
        raise ValueError(f'Unsupported occupancy mask: {mask}')
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    indices = args.vis_index if args.vis_index is not None else list(range(args.num_samples))
    if not indices or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError(f'vis-index must be in [0, {len(dataset)-1}]')
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=custom_collate_fn_temporal)
    network = load_model(cfg, checkpoint, device)
    auto = not (args.vis_occ or args.vis_gaussian or args.vis_gaussian_point or args.vis_opus_points)
    voxel_size = voxel_size_from_config(cfg)
    destination = Path(args.work_dir) / 'visualization'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for index, batch in zip(indices, loader):
            data = {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()}
            images = data.pop('img')
            with torch.autocast(device_type=device.type, enabled=cfg.get('amp', False)):
                output = network(imgs=images, metas=data)
            draw_occ = auto or args.vis_occ
            draw_gaussian = args.vis_gaussian or args.vis_gaussian_point or (auto and 'gaussian' in output)
            if draw_gaussian and 'gaussian' not in output:
                raise ValueError('This point-only model has no Gaussian output; use --vis-occ or --vis-opus-points')
            occ = occupancy_data(output, data, mask)
            folder = destination / f'frame_{index:06d}'
            folder.mkdir(parents=True, exist_ok=True)
            bounds = cfg.get('pc_range')
            if bounds is None:
                bounds = np.concatenate([occ['xyz'].min(0)-voxel_size/2,
                                         occ['xyz'].max(0)+voxel_size/2])
            record = dict(index=index, mask=mask, voxel_size=voxel_size.tolist(),
                          gaussian_sigma=args.gaussian_sigma, min_opacity=args.min_opacity,
                          files=[])
            if draw_occ:
                np.savez_compressed(folder / 'occupancy.npz', **occ, voxel_size=voxel_size)
                record['files'].append('occupancy.npz')
                record['occupied_voxels'] = int((occ['mask'] & (occ['prediction'] >= 0) & (occ['prediction'] < 17)).sum())
                if not args.data_only:
                    for name in ['occupancy', 'ground_truth']:
                        render_image(folder, name, bounds, args)
                        record['files'].append(f'{name}.png')
            if draw_gaussian:
                valid = output.get('opus_point_valid_masks', [None])[-1]
                gaussian = gaussian_data(output['gaussian'], valid, args.min_opacity)
                np.savez_compressed(folder / 'gaussians.npz', **gaussian)
                record['files'].append('gaussians.npz')
                record['gaussian_count'] = len(gaussian['means'])
                if not args.data_only:
                    render_image(folder, 'gaussians', bounds, args)
                    record['files'].append('gaussians.png')
            if args.vis_opus_points:
                if 'opus_points' not in output:
                    raise ValueError('This model does not return opus_points')
                valid = output.get('opus_point_valid_mask')
                if valid is None:
                    valid = output.get('opus_point_valid_masks', [None])[-1]
                keep = as_numpy(valid)[0].reshape(-1).astype(bool) if valid is not None else slice(None)
                np.savez_compressed(folder / 'opus_points.npz', **{
                    key: as_numpy(output[source])[0][keep] for key, source in
                    [('points', 'opus_points'), ('labels', 'opus_labels'), ('scores', 'opus_scores')]})
                record['files'].append('opus_points.npz')
            (folder / 'metadata.json').write_text(json.dumps(record, indent=2) + '\n')
            records.append(record)
            print(f'Frame {index}: {folder} ({", ".join(record["files"])})', flush=True)
            del output, data, images
    manifest = dict(config=str(Path(args.py_config).resolve()), checkpoint=str(checkpoint.resolve()),
                    device=str(device), view=args.view, show=args.show, data_only=args.data_only,
                    class_names=CLASS_NAMES,
                    colors_rgb=np.rint(COLORS * 255).astype(int).tolist(), frames=records)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main(parse_args())
