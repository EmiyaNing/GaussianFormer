"""Full 2^4 Tiny attribute diagnostic, fitted to each evaluation frame's GT.

All 15 nonempty subsets of C(center), S(scale), R(rotation), O(opacity).
This is a label-assisted output-fitting experiment, NOT normal evaluation.
"""
from pathlib import Path
from itertools import combinations
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader, Subset
from tools.stat_boundary_gaussians import read, write, sha, forward, to_cuda
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.eval_gaussian_sizes import paths, build, CLASS_NAMES, CLASS_COLUMNS
from tools.eval_gaussian_center_oracle import objective, loss_terms, TRANSITIONS
from tools.semantic_field_support import extension

RUN = ROOT / 'work_dirs/gaussian_attribute_oracle_full_val_v1'
HISTORY = ROOT / 'work_dirs/gaussian_sizes_full_val'
CENTER_HISTORY = ROOT / 'work_dirs/gaussian_center_oracle_full_val_v1/tiny'
ATTRIBUTES = 'CSRO'
ARMS = tuple(''.join(x) for n in range(1, 5) for x in combinations(ATTRIBUTES, n))
STEPS = (0, 5, 10, 20)
ROTATION_LIMIT = math.pi / 4
LR = {'C': .02, 'S': .02, 'R': .05, 'O': .05}
ATTR_STATS = (
    'GS', 'center_sum_m', 'center_max_m', 'center_moving', 'center_at_limit',
    'scale_abs_change_sum_m', 'scale_abs_change_max_m', 'scale_moving_GS',
    'scale_at_min_axes', 'scale_at_max_axes', 'rotation_sum_rad', 'rotation_max_rad',
    'rotation_moving', 'rotation_at_limit', 'opacity_abs_change_sum',
    'opacity_abs_change_max', 'opacity_moving', 'opacity_at_zero', 'opacity_at_one',
    'covariance_relative_change_sum', 'initial_near_spherical',
    'near_spherical_rotation_sum_rad', 'initial_mean_aspect_ratio_sum')
OVERLAP_NAMES = ('rescued_binary_FN', 'removed_binary_FP',
                 'corrected_foreground_class', 'all_corrected_predictions')


def initialize():
    config, checkpoint = paths('tiny')
    cfg = Config.fromfile(config)
    old = read(HISTORY/'tiny/protocol.json')
    assert sha(checkpoint) == old['checkpoint_sha256']
    for name, digest in old['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    branch = cfg.model.encoder
    assert branch.get('scale_min', .08) == .08 and branch.get('scale_max', .4) == .4
    files = list(dict.fromkeys([Path(__file__), ROOT/'tools/eval_gaussian_center_oracle.py',
        ROOT/'loss/occupancy_loss.py', ROOT/'loss/utils/lovasz_softmax.py',
        ROOT/'model/head/localagg/local_aggregate/__init__.py',
        ROOT/'model/head/localagg/src/backward.cu', ROOT/'model/utils/utils.py',
        *[ROOT/name for name in old['source_sha256']]]))
    files = [p for p in files if p.is_file()]
    protocol = dict(version=1, model='tiny', frames=6019, scenes=150, GS=76800,
        queries=600, children=128, arms=list(ARMS), steps=list(STEPS), seed=928,
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
        config=str(config.relative_to(ROOT)),
        resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        annotation_sha256=sha(cfg.val_dataset_config.imageset),
        domain='All camera-valid GT labels 0..17; CE includes empty; Lovasz ignores 17',
        loss='Unmodified configured final-stage Occupancy CE+Lovasz, no Set/coverage objectives',
        optimizer=dict(type='Adam', betas=[.9,.999], eps=1e-8, weight_decay=0,
                       learning_rates=LR, independent_zero_state_per_frame_per_arm=True),
        constraints=dict(C='Euclidean displacement <=0.4m; native ROI',
            S='Direct output scales, each axis in [0.08,0.4] m',
            R='3D local rotation vector, norm <=pi/4 rad; compose with initial quaternion',
            O='Direct output opacity in [0,1]; originally zero-opacity GS remain zero'),
        fixed='Network, all semantic values, empty Gaussian/scalar/bias, GS count and query parent; inactive attributes exactly fixed',
        interpretation='Per-frame validation-GT-assisted fitting, not deployable accuracy, not a global optimum or a causal allocation of training errors',
        comparison='Budget-specific 20-update corrective gains; groups have different physical constraints and degrees of freedom',
        overlap=list(OVERLAP_NAMES), attribute_statistics=list(ATTR_STATS),
        transitions=list(TRANSITIONS), class_columns=CLASS_COLUMNS,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    if (RUN/'protocol.json').exists():
        assert read(RUN/'protocol.json') == protocol, 'Inputs changed; use a new run directory'
    else:
        write(RUN/'protocol.json', protocol)
        cfg.dump(RUN/'resolved_config.py')
        for p in files:
            dest = RUN/'sources'/p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    return cfg, protocol


def quaternion_product(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), -1)


def rotate(initial, vector):
    angle = vector.norm(dim=-1, keepdim=True)
    delta = torch.cat(((angle/2).cos(), .5*torch.sinc(angle/(2*math.pi))*vector), -1)
    return quaternion_product(delta, initial)


class Frame:
    def __init__(self, net, out, batch, protocol):
        self.net, self.head, self.protocol = net, net.head, protocol
        self.labels = batch['occ_label'].flatten().int().contiguous()
        self.valid = batch['occ_cam_mask'].flatten().bool() & (self.labels >= 0) & (self.labels < 18)
        self.target = self.labels[self.valid].long()
        self.xyz_full = out['sampled_xyz'][0].detach().contiguous()
        self.xyz = self.xyz_full[self.valid].contiguous()
        self.base_logits = out['pred_occ'][-1][0, :, self.valid].detach().clone()
        self.base_pred = self.base_logits.argmax(0)
        g = out['gaussian']
        self.base = type(g)(*(v.detach().clone() if torch.is_tensor(v) else v for v in g))
        self.args = tuple(v.detach() for v in self.head.prepare_gaussian_args(self.base))
        self.empty_logits = self.base_logits[17].clone()
        self.agg = self.head.aggregator
        self.lower = self.agg.pc_min[:, None, :] + 1e-4
        self.upper = self.agg.pc_min[:, None, :] + self.base.means.new_tensor(
            [self.agg.H,self.agg.W,self.agg.D])*self.agg.grid_size-1e-4
        self.zero_opacity = self.base.opacities == 0
        self.foreground_ids = torch.where(self.valid & (self.labels < 17))[0]
        self.lookup = torch.full_like(self.labels, -1)
        self.lookup[self.foreground_ids] = torch.arange(len(self.foreground_ids),
            device=self.labels.device, dtype=torch.int32)
        self.classes = self.base.semantics[0].argmax(-1).int().contiguous()
        self.base_cov = torch.linalg.inv(self.args[-1][:, :-1]).detach()

    def parameters(self, arm):
        params = {}
        if 'C' in arm: params['C'] = torch.nn.Parameter(torch.zeros_like(self.base.means))
        if 'S' in arm: params['S'] = torch.nn.Parameter(self.base.scales.clone())
        if 'R' in arm: params['R'] = torch.nn.Parameter(torch.zeros_like(self.base.means))
        if 'O' in arm: params['O'] = torch.nn.Parameter(self.base.opacities.clone())
        return params

    def gaussian(self, params):
        values = {}
        if 'C' in params: values['means'] = self.base.means+params['C']
        if 'S' in params: values['scales'] = params['S']
        if 'R' in params: values['rotations'] = rotate(self.base.rotations, params['R'])
        if 'O' in params: values['opacities'] = params['O']
        return self.base._replace(**values)

    def render(self, params):
        g = self.gaussian(params)
        if 'S' in params or 'R' in params:
            args = self.head.prepare_gaussian_args(g)
        else:
            args = self.args
        inverse = args[-1][:, :-1]
        z = self.agg(self.xyz[None], g.means, g.opacities.reshape(1,-1),
            self.args[2][:,:-1], g.scales, inverse).T
        return torch.cat((z[:17],self.empty_logits[None]),0), g, inverse

    @torch.no_grad()
    def project(self, params):
        if 'C' in params:
            x = params['C']
            x.mul_((.4/x.norm(dim=-1,keepdim=True).clamp_min(.4)).clamp(max=1))
            x.copy_((self.base.means+x).maximum(self.lower).minimum(self.upper)-self.base.means)
        if 'S' in params: params['S'].clamp_(.08,.4)
        if 'R' in params:
            x = params['R']
            x.mul_((ROTATION_LIMIT/x.norm(dim=-1,keepdim=True).clamp_min(ROTATION_LIMIT)).clamp(max=1))
        if 'O' in params:
            params['O'].clamp_(0,1)
            params['O'][self.zero_opacity] = 0

    @torch.no_grad()
    def coverage(self, g, inverse):
        means = g.means[0].contiguous()
        alpha = g.opacities.flatten().contiguous()
        inv = inverse[0].flatten(1)[:,[0,4,8,1,5,2]].contiguous()
        centers = ((means-self.agg.pc_min)/self.agg.grid_size).int().contiguous()
        radii = torch.ceil(g.scales[0].max(-1).values*self.agg.scale_multiplier/self.agg.grid_size).int().contiguous()
        _,counts,_ = extension().support(means,inv,alpha,centers,radii,self.xyz_full,
            self.lookup,self.labels,self.classes,self.classes,len(self.foreground_ids),
            self.protocol['queries'],self.protocol['children'],self.agg.H,self.agg.W,self.agg.D)
        hit = counts[:,[0,1,2,3,4,5,7]] > 0
        cc = torch.zeros((17,8),device=means.device,dtype=torch.int64)
        y = self.labels[self.foreground_ids].long()
        cc[:,0] = torch.bincount(y,minlength=17)
        for j in range(7): cc[:,j+1] = torch.bincount(y[hit[:,j]],minlength=17)
        return cc

    @torch.no_grad()
    def statistics(self, params, g, inverse):
        n = g.means.shape[1]
        d = (g.means-self.base.means).norm(dim=-1).double()
        scale = (g.scales-self.base.scales).abs().double()
        angle = params['R'].norm(dim=-1).double() if 'R' in params else d.new_zeros(d.shape)
        opacity = (g.opacities-self.base.opacities).abs().double().flatten()
        cov = torch.linalg.inv(inverse.detach())
        relative = (cov-self.base_cov).flatten(-2).norm(dim=-1)/self.base_cov.flatten(-2).norm(dim=-1)
        aspect = (self.base.scales.max(-1).values/self.base.scales.min(-1).values).double()
        spherical = aspect <= 1.05
        vals = [d.new_tensor(n),d.sum(),d.max(),(d>1e-5).sum(),(d>=.4-1e-5).sum(),
            scale.sum(),scale.max(),(scale.max(-1).values>1e-5).sum(),
            (g.scales<=.08+1e-5).sum(),(g.scales>=.4-1e-5).sum(),
            angle.sum(),angle.max(),(angle>1e-5).sum(),(angle>=ROTATION_LIMIT-1e-5).sum(),
            opacity.sum(),opacity.max(),(opacity>1e-5).sum(),
            (g.opacities<=1e-5).sum(),(g.opacities>=1-1e-5).sum(),
            relative.double().sum(),spherical.sum(),angle[spherical].sum(),aspect.sum()]
        return torch.stack(vals)

    @torch.no_grad()
    def observe(self, params, z, g, inverse):
        pred = z.argmax(0)
        cm = torch.bincount(self.target*18+pred,minlength=324).reshape(18,18)
        fg,oldfg,newfg = self.target<17,self.base_pred<17,pred<17
        oldok,newok = self.base_pred==self.target,pred==self.target
        masks = [fg&~oldfg&newfg,fg&oldfg&~newfg,
            ~fg&oldfg&~newfg,~fg&~oldfg&newfg,~oldok&newok,oldok&~newok,
            fg&oldfg&newfg&~oldok&newok,fg&oldfg&newfg&oldok&~newok]
        transitions = torch.stack([x.sum() for x in masks])
        corrections = torch.stack((masks[0],masks[2],masks[6],masks[4]))
        return dict(confusion=cm,class_coverage=self.coverage(g,inverse),
            transitions=transitions,attribute_stats=self.statistics(params,g,inverse)), corrections


def refine_arm(frame, arm, occ, baseline, audit=False):
    params = frame.parameters(arm)
    optimizer = torch.optim.Adam([dict(params=[v],lr=LR[k]) for k,v in params.items()],
                                 betas=(.9,.999),eps=1e-8,weight_decay=0)
    trajectory,rows,gradients = [],{},[]
    for step in range(21):
        optimizer.zero_grad(set_to_none=True)
        z,g,inverse = frame.render(params)
        if step == 0:
            torch.testing.assert_close(z,frame.base_logits,rtol=0,atol=0)
        values = loss_terms(z,frame.target,occ)
        if not bool(torch.isfinite(values).all()): raise FloatingPointError((arm,step,'loss'))
        trajectory.append(values.detach())
        if step in STEPS:
            if step == 0:
                row = baseline
            else:
                row,correct = frame.observe(params,z.detach(),g,inverse)
            for name,value in row.items(): rows.setdefault(name,[]).append(value.detach())
        if step == 20: break
        values[0].backward()
        grad = torch.stack([v.grad.abs().max() if v.grad is not None else values.new_tensor(float('nan')) for v in params.values()])
        if not bool(torch.isfinite(grad).all()): raise FloatingPointError((arm,step,'gradient'))
        gradients.append(grad.max().detach())
        optimizer.step()
        frame.project(params)
    allowed = {'C':'means','S':'scales','R':'rotations','O':'opacities'}
    active = {allowed[k] for k in arm}
    for name in ('means','scales','rotations','opacities','semantics'):
        if name not in active:
            assert torch.equal(getattr(g,name),getattr(frame.base,name)),(arm,name)
    assert all(p.grad is None for p in frame.net.parameters())
    if 'C' in params: assert float(params['C'].norm(dim=-1).max()) <= .40001
    if 'S' in params: assert float(g.scales.min()) >= .08-1e-6 and float(g.scales.max()) <= .4+1e-6
    if 'R' in params: assert float(params['R'].norm(dim=-1).max()) <= ROTATION_LIMIT+1e-5
    if 'O' in params: assert float(g.opacities.min()) >= 0 and float(g.opacities.max()) <= 1
    result = {name:torch.stack(values).cpu().numpy() for name,values in rows.items()}
    result['loss_trajectory'] = torch.stack(trajectory).cpu().numpy()
    result['max_gradient'] = torch.stack(gradients).max().cpu().numpy()
    if audit:
        from tools.eval_gaussian_center_oracle import FrozenRenderer
        zero = torch.zeros_like(frame.base.means,requires_grad=True)
        native = FrozenRenderer(frame.head,frame.base,frame.xyz,frame.base_logits[17])
        if arm == 'C':
            torch.testing.assert_close(native.render(zero),frame.base_logits,rtol=0,atol=0)
    return result,correct


def numerical_audit(net, protocol):
    """Check all four derivative paths against real-renderer finite differences."""
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
    device = next(net.parameters()).device
    means = torch.tensor([[-.13,-.17,.31],[.43,.29,.71],[-.53,.41,.51]],device=device)[None]
    scales = means.new_tensor([[.23,.29,.31],[.28,.25,.33],[.27,.32,.24]])[None]
    rotation = means.new_tensor([[1.,0,0,0]]).repeat(1,3,1)
    alpha = means.new_tensor([.8,.9,.7]).reshape(1,3,1)
    semantic = means.new_full((1,3,17),.1)
    semantic[0,0,4]=3.;semantic[0,1,11]=3.;semantic[0,2,16]=3.
    g = GaussianPrediction(means,scales,rotation,alpha,semantic)
    x = torch.arange(-.6,.61,.4,device=device)
    z = torch.arange(-.4,1.21,.4,device=device)
    xyz = torch.stack(torch.meshgrid(x,x,z,indexing='ij'),-1).reshape(-1,3)
    with torch.no_grad():
        args = net.head.prepare_gaussian_args(g)
        m,a,s,scale,inv = args
        logits = net.head.aggregator(xyz[None],m,a.reshape(1,-1),s,scale,inv).T
        logits = torch.cat((logits[:17],logits[17:]-net.head.empty_bias),0)
    # The derivative check does not call grid-indexed coverage.
    fake = dict(occ_label=torch.full((len(xyz),),17,device=device),
                occ_cam_mask=torch.ones(len(xyz),device=device,dtype=torch.bool))
    fake['occ_label'][:len(xyz)//2]=4
    out=dict(sampled_xyz=xyz[None],pred_occ=[logits[None]],gaussian=g)
    frame=Frame(net,out,fake,protocol)
    answer={}
    for arm in ATTRIBUTES:
        params=frame.parameters(arm)
        z,_,_=frame.render(params)
        torch.testing.assert_close(z,logits,rtol=0,atol=0)
        val=torch.nn.functional.cross_entropy(z[None].double(),frame.target[None])
        grad,=torch.autograd.grad(val,params[arm])
        assert torch.isfinite(grad).all()
        errors=[]
        for idx in grad.abs().flatten().argsort(descending=True)[:3].tolist():
            p=frame.parameters(arm);q=frame.parameters(arm)
            with torch.no_grad():
                p[arm].flatten()[idx]+=.002;q[arm].flatten()[idx]-=.002
                zp=frame.render(p)[0];zq=frame.render(q)[0]
                fd=(torch.nn.functional.cross_entropy(zp[None].double(),frame.target[None])-
                    torch.nn.functional.cross_entropy(zq[None].double(),frame.target[None]))/.004
            exact=grad.flatten()[idx]
            err=float((fd-exact).abs()/exact.abs().clamp_min(1e-3))
            assert err<.04,(arm,idx,float(fd),float(exact),err)
            errors.append(err)
        answer[arm]=dict(coordinates=3,max_relative_error=max(errors))
    v=means.new_tensor([[[.2,-.1,.15]]]).repeat(1,3,1)
    from model.utils.utils import get_rotation_matrix
    q=rotate(rotation,v)
    rr=get_rotation_matrix(q.double())
    sphere=torch.eye(3,device=device,dtype=torch.float64)*.09
    rotated=rr.transpose(-1,-2)@sphere@rr
    assert float((rotated-sphere).abs().max())<1e-12
    return dict(status='passed',finite_differences=answer,rotation_at_zero_finite=True,
                isotropic_rotation_covariance_invariant=True)


def process_frame(net,out,batch,occ,protocol,audit=False):
    frame=Frame(net,out,batch,protocol)
    with torch.no_grad():
        baseline,_=frame.observe({},frame.base_logits,frame.base,frame.args[-1][:,:-1])
    rows,corrections={},[]
    for arm in ARMS:
        result,correct=refine_arm(frame,arm,occ,baseline,audit)
        for name,value in result.items(): rows.setdefault(name,[]).append(value)
        corrections.append(correct)
    masks=torch.stack(corrections) # [arms,categories,valid voxels]
    overlap=[]
    for c in range(4):
        mat=masks[:,c].float()
        overlap.append((mat@mat.T).round().long())
    overlaps=torch.stack(overlap)
    if audit:
        raw=masks.cpu().numpy()
        for c in range(4):
            expected=np.array([[(raw[i,c]&raw[j,c]).sum() for j in range(15)] for i in range(15)])
            np.testing.assert_array_equal(overlaps[c].cpu().numpy(),expected)
    result={name:np.stack(values) for name,values in rows.items()}
    result['correction_overlap']=overlaps.cpu().numpy()
    return result


def check_history(result, original, centers, frame_index):
    for j in range(15):
        np.testing.assert_array_equal(result['confusion'][j,0],original['confusion'][frame_index])
        np.testing.assert_array_equal(result['class_coverage'][j,0],original['class_coverage'][frame_index])
    ci=ARMS.index('C')
    for name in ('confusion','class_coverage','transitions','loss_trajectory'):
        np.testing.assert_array_equal(result[name][ci],centers[name][frame_index],err_msg=name)


def pilot():
    from dataset.utils import custom_collate_fn_temporal
    cfg,protocol=initialize()
    net=build(cfg,'tiny');net.requires_grad_(False)
    occ=objective(cfg)
    numeric=numerical_audit(net,protocol)
    dataset=EvaluationFrames(cfg,'val');cache=FeatureCache(net,dataset)
    try:
        batch=to_cuda(custom_collate_fn_temporal([dataset[0]]))
        with torch.no_grad(): out=forward(net,batch)
        start=time.monotonic()
        result=process_frame(net,out,batch,occ,protocol,True)
        scene=dataset.keyframes[0][0]
        with np.load(HISTORY/'tiny/scenes'/f'{scene}.npz') as original, np.load(CENTER_HISTORY/'scenes'/f'{scene}.npz') as centers:
            check_history(result,original,centers,0)
        np.savez_compressed(RUN/'pilot_arrays.npz',**result)
        info=dict(status='passed',numeric=numeric,all_15_zero_update_logits_exact=True,
            historical_confusion_and_coverage_exact=True,C_entire_trajectory_bitwise_exact=True,
            inactive_attributes_fixed=True,no_network_gradients=True,
            overlap_counts_independently_verified=True,seconds_all_15_arms=time.monotonic()-start)
        write(RUN/'pilot.json',info)
        print(json.dumps(info),flush=True)
    finally:
        cache.close();net.flush_memory_phase()
        del net,cache,dataset
        gc.collect();torch.cuda.empty_cache()


def evaluate(workers):
    from dataset.utils import custom_collate_fn_temporal
    cfg,protocol=initialize()
    assert read(RUN/'pilot.json')['status']=='passed'
    if (RUN/'complete.json').exists(): return
    net=build(cfg,'tiny');net.requires_grad_(False)
    occ=objective(cfg)
    dataset=EvaluationFrames(cfg,'val')
    assert len(dataset)==6019
    scenes={}
    for i,(scene,_) in enumerate(dataset.keyframes): scenes.setdefault(scene,[]).append(i)
    assert len(scenes)==150
    pending=[]
    for scene,ids in scenes.items():
        path=RUN/'scenes'/f'{scene}.json'
        if path.exists():
            m=read(path);assert m['indices']==ids and m['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert path.with_suffix('.npz').exists()
        else: pending+=ids
    loader=DataLoader(Subset(dataset,pending),batch_size=1,num_workers=workers,shuffle=False,
        collate_fn=custom_collate_fn_temporal,pin_memory=True,persistent_workers=workers>0,
        **(dict(prefetch_factor=2) if workers else {}))
    cache=FeatureCache(net,dataset);started=time.monotonic();done=0;active=None
    try:
        for index,batch in zip(pending,loader):
            scene,f=dataset.keyframes[index]
            if active is None:
                active=scene;arrays={};indices=[];tokens=[]
                original=np.load(HISTORY/'tiny/scenes'/f'{scene}.npz')
                centers=np.load(CENTER_HISTORY/'scenes'/f'{scene}.npz')
            assert active==scene
            batch=to_cuda(batch)
            with torch.no_grad(): out=forward(net,batch)
            result=process_frame(net,out,batch,occ,protocol)
            check_history(result,original,centers,len(indices))
            for name,value in result.items(): arrays.setdefault(name,[]).append(value)
            indices.append(index);tokens.append(dataset.base.scene_infos[scene][f]['token']);done+=1
            if index==scenes[scene][-1]:
                original.close();centers.close()
                path=RUN/'scenes'/f'{scene}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,**{name:np.stack(values) for name,values in arrays.items()})
                path.with_suffix('.tmp').replace(path)
                write(path.with_suffix('.json'),dict(status='complete',scene=scene,indices=indices,
                    tokens=tokens,frames=len(indices),checkpoint_sha256=protocol['checkpoint_sha256']))
                active=None
            if done%5==0 or active is None:
                status=dict(status='running',frames=6019-len(pending)+done,expected_frames=6019,
                    scenes=len(list((RUN/'scenes').glob('*.json'))),expected_scenes=150,
                    arms=15,steps=20,elapsed_seconds=time.monotonic()-started,
                    diagnostic_frames_per_second=done/(time.monotonic()-started))
                write(RUN/'progress.json',status);print(json.dumps(status),flush=True)
            del out,batch,result
        all_paths=list((RUN/'scenes').glob('*.json'))
        ids=[i for p in all_paths for i in read(p)['indices']]
        assert len(all_paths)==150 and sorted(ids)==list(range(6019))
        for name,digest in protocol['source_sha256'].items(): assert sha(ROOT/name)==digest,name
        assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
        status=dict(status='complete',frames=6019,scenes=150,arms=15,steps=20,
            seconds=time.monotonic()-started,checkpoint_sha256=protocol['checkpoint_sha256'])
        write(RUN/'complete.json',status);write(RUN/'progress.json',status)
    finally:
        cache.close();net.flush_memory_phase()
        del net,cache,dataset,loader
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('pilot','run'),default='run')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(RUN/'status.json',dict(status='running',action=args.action,pid=os.getpid()))
        try:
            if args.action=='pilot': pilot()
            else: evaluate(args.workers)
            write(RUN/'status.json',dict(status=args.action+'_complete'))
        except BaseException as exc:
            write(RUN/'status.json',dict(status='failed',error=repr(exc),pid=os.getpid()));raise


if __name__=='__main__': main()
