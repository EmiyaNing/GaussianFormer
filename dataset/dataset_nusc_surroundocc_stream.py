import os
import numpy as np
from torch.utils import data
import pickle
from mmcv.image.io import imread
from pyquaternion import Quaternion
from . import OPENOCC_DATASET


@OPENOCC_DATASET.register_module()
class NuScenes_Scene_SurroundOcc_Dataset_Stream(data.Dataset):
    def __init__(
        self,
        data_path,
        num_frames=None,
        grid_size_occ=[200, 200, 16],
        empty_idx=17,
        imageset=None,
        phase='train',
        scene_name=None,
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.num_frames = num_frames
        self.grid_size_occ = np.array(grid_size_occ).astype(np.uint32)
        self.empty_idx = empty_idx
        self.phase = phase
        if scene_name is None:
            self.scene_names = list(self.nusc_infos.keys())
        else:
            self.scene_names = [scene_name]
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

    def __len__(self):
        'Denotes the total number of scenes'
        return len(self.scene_names)

    def __getitem__(self, index):
        if self.num_frames is None:
            start_idx, end_idx = 0, self.scene_lens[index]
        else:
            while self.scene_lens[index] - self.num_frames  <= 0:
                index = np.random.randint(0, len(self.scene_lens))
            start_idx = np.random.randint(0, self.scene_lens[index] - self.num_frames)
            end_idx = start_idx + self.num_frames
        scene_name = self.scene_names[index]

        imgs_seq, metas_seq, occ_seq = [], [], []
        prev_lidar2global = None
        for i in range(start_idx, end_idx):
            info = self.nusc_infos[scene_name][i]
            data_info = self.get_data_info(info)
            # load image
            imgs = []
            for filename in data_info['img_filename']:
                imgs.append(imread(filename, 'unchanged').astype(np.float32))
            imgs_seq.append(np.stack(imgs, 0))
            # load metas
            metas = {'scene_name': scene_name}
            metas['lidar2img'] = [data_info['lidar2img']]
            if prev_lidar2global is not None:
                metas['lidar2global'] = [prev_lidar2global, data_info['lidar2global']]
            prev_lidar2global = data_info['lidar2global']
            metas_seq.append([metas])
            # load surroundocc label
            label_file = os.path.join('data/surroundocc/samples', data_info['pts_filename'].split('/')[-1]+'.npy')
            label_idx = np.load(label_file)
            occ_label = np.ones(self.grid_size_occ, dtype=np.int64) * self.empty_idx
            occ_label[label_idx[:, 0], label_idx[:, 1], label_idx[:, 2]] = label_idx[:, 3]
            occ_seq.append(occ_label)

        imgs = np.stack(imgs_seq, 0)    # F, N, H, W, C
        occ = np.stack(occ_seq, 0)      # F, H, W, D
        data_tuple = (imgs, metas_seq, occ)
        return data_tuple
    
    def get_data_info(self, info):
        # standard protocal modified from SECOND.Pytorch
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = info['lidar2ego_translation']
        ego2lidar = np.linalg.inv(lidar2ego)
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']
        lidar2global = np.dot(ego2global, lidar2ego)

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            ego2lidar=ego2lidar,
            lidar2global=lidar2global,
        )

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        return input_dict