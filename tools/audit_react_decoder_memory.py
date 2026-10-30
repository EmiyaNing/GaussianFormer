"""Audit fixed real DAF inputs, isolating decoder changes from sparse repeats."""

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from mmengine import Config
from mmseg.models import build_segmentor

import model  # noqa: F401
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal, move_batch_to_device
from model.encoder.gaussian_encoder.deformable_module import DeformableFeatureAggregation as DFA
from model.encoder.gaussian_encoder.ops import DeformableAggregationFunction as DAF


class Captured(Exception):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',type=int,default=128)
    parser.add_argument('--history',type=int,default=9)
    parser.add_argument('--isolate',action='store_true')
    parser.add_argument('--normalized-sampling',action='store_true',
                        help='Compare v4 normalization+DAF against the probability-free fused operator')
    parser.add_argument('--out',type=Path,default=Path('work_dirs/react_execution_optimization_v4/real_daf_audit.json'))
    args=parser.parse_args()
    cfg=Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast_v3.py')
    cfg.val_dataset_config.num_lidar_history=args.history
    torch.manual_seed(928)
    dataset=OPENOCC_DATASET.build(cfg.val_dataset_config)
    batch=move_batch_to_device(custom_collate_fn_temporal([dataset[args.sample]]),'cuda')
    imgs=batch.pop('img')
    network=build_segmentor(cfg.model).cuda().train()
    layer=next(layer for layer in network.encoder.layers if isinstance(layer,DFA))
    captured={}
    def save(module,inputs):
        captured['features'],captured['anchor'],captured['embed']=[v.detach().cpu() for v in inputs[:3]]
        captured['maps']=[v.detach().cpu() for v in inputs[3]]
        captured['metas']={key:inputs[4][key].detach().cpu() for key in ('projection_mat','image_wh')}
        raise Captured()
    hook=layer.register_forward_pre_hook(save)
    try:
        with torch.no_grad():
            network(imgs=imgs,metas=batch)
    except Captured:
        pass
    finally:
        hook.remove()
    del network,dataset,batch,imgs
    gc.collect()
    torch.cuda.empty_cache()
    features,anchor,embed=[captured[key].cuda().requires_grad_() for key in ('features','anchor','embed')]
    maps=[value.cuda().requires_grad_() for value in captured['maps']]
    metas={key:value.cuda() for key,value in captured['metas'].items()}
    prepared=DAF.feature_maps_format(maps)
    prepared=(prepared[0].contiguous().float(),prepared[1].int(),prepared[2].int())
    variables=(features,anchor,embed)+tuple(maps)+tuple(layer.parameters())
    names=['features','anchor','anchor_embed']+[f'image_level_{i}' for i in range(len(maps))]
    names += [f'parameter/{name}' for name,_ in layer.named_parameters()]
    torch.manual_seed(173)
    channels=layer.embed_dims*(2 if layer.residual_mode=='cat' else 1)
    probe=torch.randn(*features.shape[:2],channels,device=features.device)
    baseline_output=baseline_gradients=None
    report=dict(sample=args.sample,history=args.history,gaussians=features.shape[1],
                gradient_order=names,variants={})
    variants=(['baseline','projection_only','factor_only','mask_only','compact','fused']
              if args.isolate else ['baseline','compact','fused'])
    if args.normalized_sampling:
        variants=['baseline','normalized_sampling']
    for variant in variants:
        layer.batched_projection=args.normalized_sampling or variant in ('projection_only','compact','fused')
        layer.factor_camera_weights=args.normalized_sampling or variant in ('factor_only','compact','fused')
        layer.compact_weight_mask=args.normalized_sampling or variant in ('mask_only','compact','fused')
        layer.fused_weight_softmax=args.normalized_sampling or variant=='fused'
        layer.fuse_normalized_sampling=variant=='normalized_sampling'
        torch.manual_seed(938)
        output=layer(features,anchor,embed,maps,metas,formatted_feature_maps=prepared)
        gradients=torch.autograd.grad(output,variables,probe,retain_graph=True)
        assert all(torch.isfinite(value).all() for value in gradients)
        if baseline_output is None:
            baseline_output=output.detach().cpu()
            baseline_gradients=[value.detach().cpu() for value in gradients]
        else:
            actual=output.detach().cpu()
            actual_gradients=[value.detach().cpu() for value in gradients]
            relative=[float((a-b).norm()/b.norm().clamp_min(1e-12))
                      for a,b in zip(actual_gradients,baseline_gradients)]
            entry=dict(output_max_abs=float((actual-baseline_output).abs().max()),
                       output_mean_abs=float((actual-baseline_output).abs().mean()),
                       all_input_and_parameter_gradient_max_relative_l2=max(relative),
                       parameter_gradient_max_relative_l2=max(relative[3+len(maps):]),
                       largest_gradient_difference=names[relative.index(max(relative))],
                       gradient_relative_l2=relative)
            torch.testing.assert_close(actual,baseline_output,atol=2e-4,rtol=2e-4)
            assert max(relative)<.002,entry
            report['variants'][variant]=entry
        del output,gradients
    with torch.no_grad():
        points=layer.kps_generator(anchor,features)
        expected,mask=layer.project_points(points,metas['projection_mat'],metas['image_wh'])
        actual,actual_mask=layer.project_points_batched(points,metas['projection_mat'],metas['image_wh'])
        report['projection_mask_mismatches']=int((mask!=actual_mask).sum())
        report['valid_projection_max_abs']=float((expected[mask]-actual[mask]).abs().max())
        report['feature_cell_floor_changes']={}
        for height,width in prepared[1].tolist():
            scale=expected.new_tensor([width,height])
            before=(expected*scale-.5).floor()
            after=(actual*scale-.5).floor()
            report['feature_cell_floor_changes'][f'{height}x{width}']=int(
                ((before!=after).any(-1)&mask).sum())
        assert report['projection_mask_mismatches']==0
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
