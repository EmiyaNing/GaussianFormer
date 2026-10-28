"""Generate literal, independent C/D/E snapshots and an explicit change audit."""
import math

from mmengine.config import Config

from ..ab_first_epoch.report import read
from ..train import atomic_json, digest
from .protocol import ANALYSIS, ARMS, C_ANALYSIS, C_CONFIG, D, E


def main():
    base = Config.fromfile(str(C_CONFIG))
    if digest(C_CONFIG) != read(C_ANALYSIS / 'configuration_audit.json')['config_sha256']:
        raise ValueError('preserved C configuration changed')
    frequency_path = ANALYSIS / 'training_class_frequencies_mean_balance.json'
    frequency = read(frequency_path)
    if digest(base.train_dataset_config.selection_manifest) != frequency['train_manifest_sha256']:
        raise ValueError('frequency evidence used another training subset')
    audits = {}
    paths = dict(ARMS, C_reference_eval=ARMS[D].with_name('sparseworld_opus_gaussian_bias_C_reference_eval_DE_4096_1024_15e.py'))
    for name, path in paths.items():
        if path.exists():
            raise ValueError(f'refusing to replace a standalone configuration: {path}')
        cfg = Config(base.to_dict())
        cfg.experiment_name = f'sparseworld_opus_gaussian_bias_{name}'
        cfg.report_directory = str(ANALYSIS / name)
        cfg.route_arm = name
        cfg.custom_imports.imports.append('experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch')
        if name == D:
            cfg.loss.occupancy_loss.type = 'BalancedConditionalSemanticOccupancyLoss'
            cfg.loss.occupancy_loss.conditional_class_multipliers = frequency['auxiliary_multipliers']
            change = 'Only rebalance the 17-class conditional auxiliary using clipped sqrt inverse training frequency.'
        elif name == E:
            cfg.model.type = 'SparseWorldFactorizedOccupancy'
            cfg.model.pts_bbox_head.transformer.encoder.type = 'FactorizedOccupancyGaussianEncoder'
            cfg.model.pts_bbox_head.transformer.encoder.initial_occupancy_logit = -4.59511985013459
            cfg.model.gaussian_head.type = 'FactorizedOccupancyGaussianRenderer'
            cfg.model.gaussian_head.occupancy_threshold = .5
            cfg.model.gaussian_head.occupancy_prior_offset = math.log(17.)
            change = 'Independent query-to-child occupancy evidence; factorized joint probabilities and fixed binary gate=0.5.'
        else:
            change = 'Read-only re-evaluation of preserved C checkpoint to collect paired scene counts.'
        expected = cfg.to_dict()
        for key in ('experiment_name', 'report_directory', 'custom_imports'):
            expected[key] = base.to_dict()[key]
        expected.pop('route_arm')
        if name == D:
            expected['loss']['occupancy_loss']['type'] = base.loss.occupancy_loss.type
            expected['loss']['occupancy_loss'].pop('conditional_class_multipliers')
        elif name == E:
            expected['model']['type'] = base.model.type
            expected['model']['pts_bbox_head']['transformer']['encoder']['type'] = base.model.pts_bbox_head.transformer.encoder.type
            expected['model']['pts_bbox_head']['transformer']['encoder'].pop('initial_occupancy_logit')
            expected['model']['gaussian_head']['type'] = base.model.gaussian_head.type
            expected['model']['gaussian_head'].pop('occupancy_threshold')
            expected['model']['gaussian_head'].pop('occupancy_prior_offset')
        if expected != base.to_dict():
            raise ValueError(f'{name} changes an undeclared setting')
        cfg.dump(str(path))
        if '_base_' in path.read_text():
            raise ValueError('configuration inheritance is forbidden')
        audit = dict(config=str(path), config_sha256=digest(path), source_config=str(C_CONFIG), source_sha256=digest(C_CONFIG),
                     training_change=change, max_epochs=15, total_optimizer_updates=15360, first_epoch_updates=1024,
                     runtime_stop_after_epoch=1, diagnostic_epoch=1, supervision_mask='none', seed=17,
                     train_samples=4096, validation_samples=1024, config_inheritance=False,
                     free_relative_weight=.1, conditional_semantic_weight=1., CE_coefficient=10., Lovasz_coefficient=1.,
                     geometry_change=False, temporal_structure_change=False,
                     frequency_evidence=str(frequency_path) if name == D else None,
                     frequency_evidence_sha256=digest(frequency_path) if name == D else None,
                     occupancy_threshold=.5 if name == E else None,
                     model_initialization='same unchanged C tensors; E adds a zero-weight, 1%-prior occupancy branch')
        target = ANALYSIS / name
        target.mkdir(parents=True, exist_ok=True)
        atomic_json(target / 'configuration_audit.json', audit)
        audits[name] = audit
    atomic_json(ANALYSIS / 'configuration_audit.json', dict(baseline='C_conditional_semantic_ce', arms=audits,
        geometry_research_deferred=True, independent_routes=True, combined_DE_not_run=True,
        scope='First-epoch screening under the original 15-epoch schedule; no claim of a completed 15-epoch optimum.'))
    print('Standalone D/E and read-only C evaluation configurations prepared.', flush=True)


if __name__ == '__main__':
    main()

