from pathlib import Path
import os

import numpy as np
import pytest
import torch

from experiments.cleanup_out_20261015 import apply_plan, make_plan, sha


def checkpoint(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':{'layer.weight':torch.full((2,),float(value))}},path)


def test_cleanup_preserves_logs_caches_and_latest_and_removes_only_requested_files(tmp_path):
    root,audit=tmp_path/'out',tmp_path/'audit'
    root.mkdir();audit.mkdir()
    run=root/'model';checkpoint(run/'epoch_1.pth',1);checkpoint(run/'epoch_2.pth',2)
    (run/'latest.pth').symlink_to(run/'epoch_2.pth')
    (run/'train.log').write_bytes(b'keep exact logs\n')
    cache=root/'features';cache.mkdir();torch.save({'features':torch.arange(8)},cache/'0001.pth')
    unlabelled=root/'unlabelled';unlabelled.mkdir();torch.save({'points':torch.zeros(2,3)},unlabelled/'predictions.pth')
    np.savez(root/'keep.npz',x=np.ones(3))
    distill=root/'sparseworld_fast_motion_distillation';distill.mkdir();np.savez(distill/'remove.npz',x=np.ones(3))
    (root/'alias').symlink_to(run,target_is_directory=True)
    plan=make_plan(root)
    assert plan['model_checkpoint_files_to_delete']==2 and plan['distillation_npz_to_delete']==1
    result=apply_plan(root,audit,plan)
    assert result['status']=='completed'
    assert (run/'latest.pth').is_file() and not (run/'latest.pth').is_symlink()
    assert not (run/'epoch_2.pth').exists()
    assert (run/'train.log').read_bytes()==b'keep exact logs\n'
    assert (cache/'0001.pth').exists() and (unlabelled/'predictions.pth').exists()
    assert (root/'keep.npz').exists() and not (distill/'remove.npz').exists()
    assert torch.load(run/'latest.pth',weights_only=False)['state_dict']['layer.weight'][0]==2


def test_broken_latest_repaired_to_original_basename_before_deleting_old_checkpoints(tmp_path):
    root,audit=tmp_path/'out',tmp_path/'audit';root.mkdir();audit.mkdir()
    checkpoint(root/'epoch_1.pth',1);checkpoint(root/'epoch_2.pth',2)
    (root/'latest.pth').symlink_to('/old/unavailable/location/epoch_1.pth')
    plan=make_plan(root)
    apply_plan(root,audit,plan)
    assert torch.load(root/'latest.pth',weights_only=False)['state_dict']['layer.weight'][0]==1


def test_missing_latest_uses_newest_model_without_treating_features_as_model(tmp_path):
    root,audit=tmp_path/'out',tmp_path/'audit';root.mkdir();audit.mkdir()
    checkpoint(root/'old.pth',1);checkpoint(root/'new.pth',2)
    os.utime(root/'old.pth',(1000,1000));os.utime(root/'new.pth',(2000,2000))
    torch.save({'features':torch.ones(2,3)},root/'features.pth')
    plan=make_plan(root)
    apply_plan(root,audit,plan)
    assert (root/'features.pth').exists()
    assert torch.load(root/'latest.pth',weights_only=False)['state_dict']['layer.weight'][0]==2


def test_external_directory_alias_and_external_latest_target_never_modified(tmp_path):
    root,audit=tmp_path/'out',tmp_path/'audit';root.mkdir();audit.mkdir()
    external=tmp_path/'external';checkpoint(external/'epoch_1.pth',1)
    before=sha(external/'epoch_1.pth')
    (root/'alias').symlink_to(external,target_is_directory=True)
    run=root/'run';run.mkdir();(run/'latest.pth').symlink_to(external/'epoch_1.pth')
    plan=make_plan(root);apply_plan(root,audit,plan)
    assert sha(external/'epoch_1.pth')==before and (root/'alias').is_symlink()
    assert (run/'latest.pth').is_file() and not (run/'latest.pth').is_symlink()


def test_changed_deletion_target_stops_before_cleanup(tmp_path):
    root,audit=tmp_path/'out',tmp_path/'audit';root.mkdir();audit.mkdir()
    checkpoint(root/'epoch_1.pth',1)
    plan=make_plan(root)
    checkpoint(root/'epoch_1.pth',2)
    with pytest.raises(ValueError,match='changed'):
        apply_plan(root,audit,plan)
    assert not (root/'latest.pth').exists() and (root/'epoch_1.pth').exists()
