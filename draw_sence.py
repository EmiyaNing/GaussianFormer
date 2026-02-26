import pdb
import numpy as np
from open3d_vis_utils import draw_scenes

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

pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]

front = np.load("densify_front.npy")[0]
front[:, 0] = front[:, 0] * 100 - 50
front[:, 1] = front[:, 1] * 100 - 50
front[:, 2] = front[:, 2] * 8 - 5


after = np.load("densify_after.npy")[0]
after[:, 0] = after[:, 0] * 100 - 50
after[:, 1] = after[:, 1] * 100 - 50
after[:, 2] = after[:, 2] * 8 - 5

print("current display the densify front")
draw_scenes(front)

print("current display the densify after")
draw_scenes(after)

residual_count = after.shape[0] - front.shape[0]
residual_pts = after[-residual_count:]
print("added points during the densify")
print("current the added point's x_range:", residual_pts[:, 0].min(), residual_pts[:, 0].max())
print("current the added point's y_range:", residual_pts[:, 1].min(), residual_pts[:, 1].max())
print("current the added point's z_range:", residual_pts[:, 2].min(), residual_pts[:, 2].max())
draw_scenes(residual_pts)

front_means = np.load("densify_front_means.npy")[0]
front_sems  = np.load("densify_front_semantics.npy")[0]
front_class = front_sems.argmax(-1)
after_means = np.load("densify_after_means.npy")[0]
after_sems  = np.load("densify_after_semantics.npy")[0]
after_class = after_sems.argmax(-1)
color_maps  = get_nuscenes_colormap()

front_colors = color_maps[front_class][:, :3]
after_colors = color_maps[after_class][:, :3]


print("current display the densify front gaussian's center")
draw_scenes(front_means, point_colors=front_colors)

print("current display the densify after gaussian's center")
draw_scenes(after_means, point_colors=after_colors)


residual_means = after_means[-residual_count:]
residual_colors= after_colors[-residual_count:]
print("current the added point's x_range:", residual_means[:, 0].min(), residual_means[:, 0].max())
print("current the added point's y_range:", residual_means[:, 1].min(), residual_means[:, 1].max())
print("current the added point's z_range:", residual_means[:, 2].min(), residual_means[:, 2].max())
draw_scenes(residual_means, point_colors=residual_colors)
