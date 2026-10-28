"""Meaningful interface checks without touching the repository's test suite."""
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config

from .encoder import SparseWorldBiasGaussianEncoder, TimestampSelfAttention
from .metrics import HorizonMetrics
from .model import SparseWorldBiasRenderer
from .train import set_lr


def test_timestamp_mask_blocks_later_queries():
    attention=TimestampSelfAttention(embed_dims=8,num_heads=2,dropout=0.,pc_range=(0,0,0,2,2,2))
    attention.eval()
    attention.ind_mask=torch.tensor([[0.,-1e5],[0.,0.]])
    points=torch.full((1,2,1,3),.5)
    features=torch.randn(1,2,8)
    other=features.clone()
    other[:,1]+=torch.arange(8)*100.
    with torch.no_grad():
        first=attention(points,features)
        second=attention(points,other)
    torch.testing.assert_close(first[:,0],second[:,0])
    assert not torch.allclose(first[:,1],second[:,1])


def test_raw_points_preserved_and_outside_opacity_zero():
    encoder=SparseWorldBiasGaussianEncoder(embed_dims=8,num_decoder=1,
        num_frames=1,num_views=1,num_groups=1,num_heads=2,num_levels=1,
        num_refines=(2,),num_classes=3,feedforward_channels=16,
        pc_range=(0,0,0,2,2,2),attribute_stages=(0,),decoder_checkpoint=False)
    points=torch.tensor([[[[.5,.5,.5],[1.2,.5,.5]]]],requires_grad=True)
    logits=torch.tensor([[[[-2.,0.,2.],[-2.,0.,2.]]]],requires_grad=True)
    valid=torch.ones(1,1,dtype=torch.bool)
    gaussian=encoder._make_gaussian(torch.zeros(1,1,8),points,logits,valid,0)
    assert points[0,0,1,0].item()>1
    assert gaussian.opacities[0,0,0].item()>0
    assert gaussian.opacities[0,0,1].item()==0
    torch.testing.assert_close(gaussian.semantics,torch.nn.functional.softplus(logits))
    assert logits[0,0,0,0].item()==-2
    (gaussian.semantics.sum()+gaussian.scales.sum()).backward()
    assert logits.grad is not None
    assert encoder.attribute_branches['0'].net[-1].weight.grad is not None


def test_bias_is_shared_and_receives_gradient():
    head=SparseWorldBiasRenderer(apply_loss_type='fixed_5',max_render_scale=.4,
        initial_empty_bias=1.,num_classes=18,empty_label=17,with_empty=True,
        empty_args=dict(mean=[0,0,0],scale=[100,100,8]),
        cuda_kwargs=dict(scale_multiplier=3,H=2,W=2,D=2,pc_min=[0,0,0],grid_size=.4))
    head._render_batched=lambda xyz,*args:torch.zeros(1,18,8)
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
    gaussian=GaussianPrediction(torch.full((1,1,1,3),.2),torch.full((1,1,1,3),.2),
        torch.tensor([[[[1.,0.,0.,0.]]]]),torch.ones(1,1,1,1),torch.ones(1,1,1,17))
    head.train()
    train=head.render_state(gaussian,torch.zeros(1,8,3))
    head.eval()
    eval_output=head.render_state(gaussian,torch.zeros(1,8,3))
    torch.testing.assert_close(train,eval_output)
    assert train[0,17,0].item()==-1
    train.sum().backward()
    assert head.empty_bias.grad.item()==-8


def test_metrics_use_all_seven_horizons_and_fixed_class_divisor():
    metric=HorizonMetrics()
    grid=torch.tensor([[[[0,17]]]])
    result=dict(final_occ_grid=grid,future_occ_grids=[grid.clone() for _ in range(6)],
                pred_traj=torch.zeros(1,6,2))
    metas=dict(occ_label=grid,future_occ_labels=torch.stack([grid]*6,dim=1),
               temporal_trajs=torch.zeros(1,6,2))
    metric.add(result,metas)
    result=metric.results()
    assert len(result['horizons'])==7
    assert result['selection_mean_miou']==pytest.approx(100/17)
    assert result['horizons'][6]['seconds']==3
    assert result['horizons'][0]['class_iou'][17]==100
    assert result['horizons'][0]['occupied_iou']==100


def test_optimizer_schedule_preserves_parameter_group_ratios():
    p=torch.nn.Parameter(torch.ones(1))
    q=torch.nn.Parameter(torch.ones(1))
    optimizer=torch.optim.AdamW([dict(params=[p],lr=.1,base_lr=.1),dict(params=[q],lr=2.5,base_lr=2.5)])
    for step in (0,499,500,999):
        set_lr(optimizer,step,1000,500,.001)
        assert optimizer.param_groups[1]['lr']/optimizer.param_groups[0]['lr']==pytest.approx(25)


def test_configs_are_standalone_and_share_experiment_protocol():
    paths=[Path(f'config/sparseworld_opus_gaussian_bias/{v}_4096_1024_15e.py')
           for v in ('sparseworld_strict','sparseworld_opus_gaussian_bias')]
    configs=[Config.fromfile(str(p)) for p in paths]
    assert all('_base_' not in p.read_text() for p in paths)
    for key in ('seed','max_epochs','train_dataset_config','val_dataset_config','gradient_accumulation'):
        assert configs[0][key]==configs[1][key]
    assert configs[1].model.pts_bbox_head.transformer.encoder.num_refines[-1]==128


def test_final_report_rejects_incomplete_training(tmp_path):
    from .report import gather
    import json
    (tmp_path/'completed.json').write_text(json.dumps(dict(epochs=14)))
    with pytest.raises(ValueError,match='all 15 epochs'):
        gather(tmp_path)


def test_report_produces_seven_horizon_class_comparison_from_common_best(tmp_path):
    import csv
    import json
    from .report import create_report, VARIANTS
    root=tmp_path/'runs'
    for variant_index,variant in enumerate(VARIANTS):
        directory=root/variant
        directory.mkdir(parents=True)
        (directory/'best_recheck').mkdir()
        for epoch in range(1,16):
            metric=HorizonMetrics()
            metric.samples=1024
            metric.trajectory_count[:]=1024
            for h in range(7):
                correct=epoch+variant_index
                metric.confusion[h,0,0]=correct
                metric.confusion[h,0,17]=1024-correct
                metric.confusion[h,17,17]=1024
            value=dict(metric.results(),epoch=epoch)
            (directory/f'epoch_{epoch:02d}_metrics.json').write_text(json.dumps(value))
            np.savez_compressed(directory/f'epoch_{epoch:02d}_confusion.npz',confusion=metric.confusion)
        (directory/'best.json').write_text(json.dumps(value))
        (directory/'best_mIoU.pth').write_bytes(b'synthetic test fixture only')
        (directory/'best_recheck'/'epoch_15_metrics.json').write_text(json.dumps(value))
        np.savez_compressed(directory/'best_recheck'/'epoch_15_confusion.npz',confusion=metric.confusion)
        (directory/'initialization.json').write_text(json.dumps(dict(sha256='same_image_initialization')))
        (directory/'completed.json').write_text(json.dumps(dict(epochs=15,manifest_sha256={'train':'same','val':'same'})))
    output=tmp_path/'report'
    create_report(root,output)
    rows=list(csv.DictReader((output/'per_horizon_per_class_iou.csv').open()))
    assert len(rows)==7*18
    assert all(row['strict_epoch']=='15' and row['gaussian_epoch']=='15' for row in rows)
    assert {float(row['seconds']) for row in rows}=={h*.5 for h in range(7)}
    assert (output/'report.md').exists()
