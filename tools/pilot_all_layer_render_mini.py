"""Real-frame checks of initialization, loss strength and six GS-head gradients."""
from pathlib import Path
import gc
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.all_layer_render_experiment import *


def pilot():
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    prepare_annotation()
    cfg = config('baseline')
    dataset = OPENOCC_DATASET.build(cfg.train_dataset_config)
    order = np.random.default_rng(cfg.seed).permutation(len(dataset))
    assert len(dataset) == 4090 and len({s for s, _ in dataset.keyframes}) == 698
    batch = custom_collate_fn_temporal([TrainingFrames(dataset, cfg.seed)[int(order[0])]])
    digest = fingerprint(batch)
    batch = to_cuda(batch)
    references = {}
    stage_checks = []
    for arm in ARMS:
        cfg, net, opt, objective, state = build(arm, training=True)
        del state, opt
        seed_all(cfg.seed * 1000000 + 200000)
        torch.cuda.reset_peak_memory_stats()
        out = forward(net, batch)
        assert out['supervised_stage_indices'] == ([5] if arm == 'baseline' else list(range(6)))
        inputs = loss_input(cfg, out, batch)
        total, details = objective(inputs)
        assert torch.isfinite(total)
        if arm == 'baseline':
            references = dict(set_loss=details['OPUSSetLoss'],
                occupancy=details['OccupancyLoss'], logits=out['pred_occ'][-1].detach().cpu(),
                gaussians=[{key: getattr(g, key).detach().cpu().clone()
                            for key in ('means', 'scales', 'rotations', 'opacities', 'semantics')}
                           for g in out['gaussians']])
            total.backward()
        else:
            torch.testing.assert_close(out['pred_occ'][-1].detach().cpu(), references['logits'], rtol=0, atol=0)
            assert details['OPUSSetLoss'] == references['set_loss']
            assert details['AllLayerOccupancyLoss/layer_6'] == references['occupancy']
            terms = [details[f'AllLayerOccupancyLoss/layer_{i+1}'] for i in range(6)]
            np.testing.assert_allclose(details['AllLayerOccupancyLoss'], sum(terms), rtol=1e-7)
            for i, g in enumerate(out['gaussians']):
                for key in ('means', 'semantics'):
                    torch.testing.assert_close(getattr(g, key).detach().cpu(), references['gaussians'][i][key], rtol=0, atol=0)
            for key in ('means', 'scales', 'rotations', 'opacities', 'semantics'):
                torch.testing.assert_close(getattr(out['gaussians'][-1], key).detach().cpu(), references['gaussians'][-1][key], rtol=0, atol=0)
            occupancy, _ = objective.losses[1](inputs)
            parameters = [net.encoder.attribute_branches[str(i)].net[-1].bias for i in range(6)]
            gradients = torch.autograd.grad(occupancy, parameters, retain_graph=True)
            for i, (gradient, k) in enumerate(zip(gradients, cfg.model.encoder.num_refines)):
                value = gradient.reshape(k, 8)
                assert torch.isfinite(value).all()
                norms = dict(scale=float(value[:, :3].norm()), rotation=float(value[:, 3:7].norm()), opacity=float(value[:, 7].norm()))
                assert all(v > 0 for v in norms.values()), (i, norms)
                stage_checks.append(dict(layer=i+1, **norms, occupancy_loss=terms[i]))
            total.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
        print(arm, 'loss', float(total.detach()), 'grad_norm', float(norm),
              'peak_GB', torch.cuda.max_memory_allocated()/1e9, flush=True)
        if arm == 'all_layers':
            peak = torch.cuda.max_memory_allocated()/1e9
        release(net)
        del net, out, inputs, objective, total
        gc.collect()
        torch.cuda.empty_cache()
    write(RUN / 'pilot.json', dict(status='passed', input_fingerprint=digest, train_index=int(order[0]),
        original_final_logits_and_GS_exact=True, all_six_Set_losses_unchanged=True,
        layer6_occupancy_unchanged=True, occupancy_sum_full_weight=True,
        all_six_attribute_heads_have_scale_rotation_opacity_gradients=stage_checks,
        all_layers_peak_cuda_GB=peak))


if __name__ == '__main__':
    pilot()
