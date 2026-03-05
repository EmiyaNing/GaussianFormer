import os
import cv2
import numpy as np

from tqdm import tqdm

def get_file_list(dir_path):
    file_names = os.listdir(dir_path)
    file_names = sorted(file_names)
    return file_names


def main():
    file_root_path = './samples/mask/'
    cam_back_dir   = os.path.join(file_root_path, 'CAM_BACK')
    cam_backl_dir  = os.path.join(file_root_path, 'CAM_BACK_LEFT')
    cam_backr_dir  = os.path.join(file_root_path, 'CAM_BACK_RIGHT')
    cam_front_dir  = os.path.join(file_root_path, 'CAM_FRONT')
    cam_frontl_dir = os.path.join(file_root_path, 'CAM_FRONT_LEFT')
    cam_frontr_dir = os.path.join(file_root_path, 'CAM_FRONT_RIGHT')
    cam_back_list, cam_backl_list, cam_backr_list, cam_front_list, cam_frontl_list, cam_frontr_list = \
    get_file_list(cam_back_dir), get_file_list(cam_backl_dir), get_file_list(cam_backr_dir), get_file_list(cam_front_dir), get_file_list(cam_frontl_dir), get_file_list(cam_frontr_dir)

    for back, backr, backl, front, frontr, frontl in tqdm(zip(cam_back_list, cam_backr_list, \
                                                         cam_backl_list, cam_front_list, cam_frontr_list, cam_frontl_list)):
        back_img_mask   = ((cv2.imread(os.path.join(cam_back_dir, back)).sum(-1) > 0) * 255).astype(np.uint8) 
        backr_img_mask  = ((cv2.imread(os.path.join(cam_backr_dir, backr)).sum(-1) > 0) * 255).astype(np.uint8) 
        backl_img_mask  = ((cv2.imread(os.path.join(cam_backl_dir, backl)).sum(-1) > 0) * 255).astype(np.uint8) 
        front_img_mask  = ((cv2.imread(os.path.join(cam_front_dir, front)).sum(-1) > 0) * 255).astype(np.uint8) 
        frontr_img_mask = ((cv2.imread(os.path.join(cam_frontr_dir, frontr)).sum(-1) > 0) * 255).astype(np.uint8) 
        frontl_img_mask = ((cv2.imread(os.path.join(cam_frontl_dir, frontl)).sum(-1) > 0) * 255).astype(np.uint8) 
        cv2.imwrite(os.path.join(cam_back_dir, back), back_img_mask)
        cv2.imwrite(os.path.join(cam_backr_dir, backr), backr_img_mask)
        cv2.imwrite(os.path.join(cam_backl_dir, backl), backl_img_mask)
        cv2.imwrite(os.path.join(cam_front_dir, front), front_img_mask)
        cv2.imwrite(os.path.join(cam_frontr_dir, frontr), frontr_img_mask)
        cv2.imwrite(os.path.join(cam_frontl_dir, frontl), frontl_img_mask)
 


if __name__ == '__main__':
    main()