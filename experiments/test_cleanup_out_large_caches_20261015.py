import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.cleanup_out_large_caches_20261015 import TARGETS,apply_plan,make_plan


def fixture_tree(tmp_path):
    out,audit=tmp_path/'out',tmp_path/'audit'
    out.mkdir();audit.mkdir()
    for name in TARGETS:
        path=out/name;path.mkdir(parents=True)
        torch.save({'features':torch.ones(2,3)},path/'cache.pth')
        np.savez(path/'cache.npz',x=np.ones(3))
        (path/'train.log').write_text('preserve logs\n')
        (path/'metrics.json').write_text('{"score": 1}\n')
    torch.save({'state_dict':{'linear.weight':torch.ones(2,3)}},out/TARGETS[-1]/'latest.pth')
    unrelated=out/'unrelated';unrelated.mkdir();np.savez(unrelated/'keep.npz',x=np.ones(3))
    return out,audit


def test_exact_three_roots_cache_only_with_latest_and_logs_preserved(tmp_path):
    out,audit=fixture_tree(tmp_path)
    latest=out/TARGETS[-1]/'latest.pth';before=latest.read_bytes()
    plan=make_plan(out)
    assert plan['counts']=={'.pth':3,'.npz':3}
    result=apply_plan(out,audit,plan)
    assert result['deleted_files']==6 and result['remaining_requested_caches']==0
    assert latest.read_bytes()==before
    assert (out/'unrelated/keep.npz').exists()
    for target in TARGETS:
        assert (out/target/'train.log').read_text()=='preserve logs\n'
        assert (out/target/'metrics.json').exists()


def test_target_change_rejected_before_any_file_removed(tmp_path):
    out,audit=fixture_tree(tmp_path)
    plan=make_plan(out)
    path=out/TARGETS[0]/'cache.npz';path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        apply_plan(out,audit,plan)
    assert all((out/target/'cache.pth').exists() for target in TARGETS)
