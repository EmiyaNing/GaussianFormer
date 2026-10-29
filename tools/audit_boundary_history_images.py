"""Check all 48 normalized camera images, independently of GPU predictions."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mmengine import Config
from dataset import OPENOCC_DATASET
from tools.boundary_gaussian_stats.data import EvaluationFrames


def main():
    torch.set_num_threads(2)
    cfg=Config.fromfile(ROOT/'config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py')
    cases=[]
    for split in ('train','val'):
        options=deepcopy(cfg.val_dataset_config)
        options.imageset=cfg[split+'_dataset_config'].imageset
        native=OPENOCC_DATASET.build(options)
        fast=EvaluationFrames(cfg,split);images={}
        for index in (0,7):
            original=native[index];cached=fast[index]
            keys=cached['feature_frame_keys']
            for key in keys:
                if key not in images:images[key]=fast.base[fast.lookup[key]]['img']
            reassembled=torch.cat([images[key] for key in keys])
            torch.testing.assert_close(original['img'],reassembled,rtol=0,atol=0)
            torch.testing.assert_close(torch.from_numpy(original['projection_mat']),
                                       torch.from_numpy(cached['projection_mat']),rtol=0,atol=0)
            cases.append(dict(split=split,index=index,images=48,
                history=cached['history_frame_indices'].tolist(),
                all_images_bitwise_equal=True,projections_bitwise_equal=True))
        del native,fast,images
    path=ROOT/'work_dirs/boundary_gaussian_full/history_image_parity.json'
    path.write_text(json.dumps(dict(status='passed',cases=cases),indent=2))
    print(path)


if __name__=='__main__':main()
