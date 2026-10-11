#!/usr/bin/env python3
"""An identity-selected intervention example, not selected for effect strength."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmengine import Config
from torch.utils.data import DataLoader,Subset
from dataset.utils import custom_collate_fn_temporal
from tools.diagnose_teacher_alignment import loader_for
from tools.sparseworld_region_utils import ROOT
from tools.sparseworld_region_ablation_common import OUT,REPORT,CENTERS,records,masks,window,windows,patch_mask,shifted_mask
from tools.run_sparseworld_region_ablations import roi_mask

def main():
    candidates=[]
    for r in records():
        p=OUT/'val'/(r['token']+'.json')
        if not p.exists():continue
        d=json.loads(p.read_text())['interventions']['future_only_static']
        if d['available'] and d['F_mask_pixels']>0 and d['P_mask_pixels']>0:candidates.append(r)
    r=candidates[0]
    cfg=Config.fromfile('config/opus/future2current_feature_identity.py')
    loader,_=loader_for(cfg,'val',512,0,cfg.identity_sample_seed)
    batch=next(iter(DataLoader(Subset(loader.dataset.dataset,[r['index']]),batch_size=1,num_workers=0,collate_fn=custom_collate_fn_temporal)))
    imgs=batch.pop('img');pair=dict(np.load(ROOT/'pairs'/(r['token']+'.npz')));attrs=dict(np.load(ROOT/'attributes'/(r['token']+'.npz')))
    groups=masks(r,2,pair,attrs);roi,xy=roi_mask(groups[2]);context=(np.linalg.norm(CENTERS[:,:2]-xy,axis=1)<=8)&groups[1]
    fig,axes=plt.subplots(2,3,figsize=(12,4.5),layout='constrained');chosen={}
    for row,a in enumerate(('F','P')):
        images,_=window(batch,imgs,windows(6)['H' if a=='F' else 'P'],0 if a=='F' else 6)
        ma=patch_mask(CENTERS[context if a=='F' else roi],pair[a+'_projection'][2],pair[a+'_wh'][2],images.shape[-2],images.shape[-1])
        rand=shifted_mask(ma,20261008);view=int(ma.sum((1,2)).argmax());chosen[a]=view
        rgb=images[0,view].permute(1,2,0).numpy()*np.array([58.395,57.12,57.375])+np.array([123.675,116.28,103.53])
        rgb=np.clip(rgb/255,0,1)
        for col,mask in enumerate([None,ma[view],rand[view]]):
            shown=rgb.copy()
            if mask is not None:shown[mask]=np.array([123.675,116.28,103.53])/255
            axes[row,col].imshow(shown);axes[row,col].axis('off');axes[row,col].set_title(a+' / '+['original','targeted mask','same-area random'][col])
    fig.suptitle('3 s static future-window-only ROI | '+r['token']+'\nIdentity-selected example; view with largest mask area')
    dest=REPORT/'figures';dest.mkdir(exist_ok=True,parents=True);fig.savefig(dest/'a4_mask_example.png',dpi=160);plt.close(fig)
    (OUT/'mask_example.json').write_text(json.dumps(dict(token=r['token'],scene=r['scene'],roi_center=xy.tolist(),views=chosen),indent=2))
if __name__=='__main__':main()
