"""Token selection after full SparseWorld eligibility/history construction."""
import json
import numpy as np

from dataset import OPENOCC_DATASET
from dataset.sparseworld_strict import NuScenesSparseWorldStrictDataset


@OPENOCC_DATASET.register_module()
class SparseWorldBiasSubsetDataset(NuScenesSparseWorldStrictDataset):
    def __init__(self, selection_manifest, **kwargs):
        super().__init__(**kwargs)
        with open(selection_manifest, encoding='utf-8') as handle:
            manifest = json.load(handle)
        eligible = {self.scene_infos[s][i]['token']: (s, i)
                    for s, i in self.keyframes}
        tokens = [sample['token'] for sample in manifest['samples']]
        if len(set(tokens)) != len(tokens):
            raise ValueError('subset manifest contains duplicate tokens')
        missing = sorted(set(tokens) - set(eligible))
        if missing:
            raise ValueError(f'subset tokens no longer eligible: {missing[:5]}')
        self.keyframes = [eligible[token] for token in tokens]
        if len(self.keyframes) != manifest['num_samples']:
            raise ValueError('subset size differs from manifest')
        self.selection_manifest = selection_manifest
        self.epoch = 0
        self.augmentation_seed = 17
        # Parent history indexing still uses the original full metadata.

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        state = np.random.get_state()
        np.random.seed(self.augmentation_seed + self.epoch * len(self) + int(index))
        try:
            return super().__getitem__(index)
        finally:
            np.random.set_state(state)

