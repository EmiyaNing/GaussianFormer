import pdb
import pickle


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['infos']


#daocc_info_path = './data/nuscenes_infos_train_w_3occ.pkl'
#info_path = './data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl'
info_path = './data/nuscenes_temporal_infos_train.pkl'
infos = load_pkl(info_path)
pdb.set_trace()