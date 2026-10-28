import ast
from pathlib import Path

import pytest
import torch
from mmengine.config import Config

from .prepare import equal_tree, migrate
from .protocol import ANALYSIS, CONFIG, SOURCE_CONFIG
from .train import should_restore_query_statistics


def test_output_only_standalone_config_keeps_training_and_fifteen_epoch_budget():
    old, new = Config.fromfile(str(SOURCE_CONFIG)).to_dict(), Config.fromfile(str(CONFIG)).to_dict()
    old.pop('report_directory')
    assert new.pop('report_directory') == str(ANALYSIS)
    assert old == new
    assert new['max_epochs']==15 and new['gradient_accumulation']==2
    assert new['train_loader']['batch_size']==2 and 4096//2//2*15==15360
    assert not any(text in CONFIG.read_text() for text in ('_base_', 'Config.fromfile', 'runpy'))


@pytest.mark.parametrize('rng,index,phase,expected',[(None,0,'train',False),({},0,'train',False),({},50,'train',True),({},4096,'evaluate',True)])
def test_epoch_boundary_resets_query_statistics_but_interruption_restores(rng,index,phase,expected):
    assert should_restore_query_statistics(rng,index,phase) is expected


def test_checkpoint_output_hash_migration_preserves_all_other_values(tmp_path):
    source,target = tmp_path/'source.pth',tmp_path/'target.pth'
    saved = dict(config_sha256='old',state_dict={'weight':torch.arange(12).reshape(3,4)},
                 optimizer={'state':{0:{'step':torch.tensor(1024.),'exp_avg':torch.arange(4.)}}},
                 rng={'torch':torch.get_rng_state()},query_stamps=torch.arange(7),epoch=1,updates=1024)
    torch.save(saved,source)
    before=source.read_bytes()
    migrated=migrate(source,target,'old','new')
    assert source.read_bytes()==before and migrated.pop('config_sha256')=='new'
    saved.pop('config_sha256')
    assert equal_tree(saved,migrated)


def test_every_epoch_evaluation_body_matches_verified_first_epoch_body():
    original=Path('experiments/sparseworld_opus_gaussian_bias/semantic_occupancy_first_epoch/diagnostics.py')
    current=Path(__file__).with_name('diagnostics.py')
    def function(path):
        return next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='evaluate')
    old,new=function(original),function(current)
    assert isinstance(old.body[0],ast.If)
    old.body.pop(0)
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
