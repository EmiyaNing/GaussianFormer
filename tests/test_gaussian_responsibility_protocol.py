"""Check that the intervention preserves earlier stages and geometric gradients."""
import torch

from loss.center_support_diagnostic_loss import CenterSupportDiagnosticSetLoss
from loss.gaussian_responsibility_loss import GaussianResponsibilitySetLoss
from model.encoder.gaussian_encoder.utils import GaussianPrediction


def test_final_stage_only_and_identical_geometry_gradients():
    torch.manual_seed(928)
    gt = torch.tensor([[[-.2, -.2, .2], [1.8, 1.8, .2]]])
    points = [(gt + .01).clone().requires_grad_(True) for _ in range(2)]
    logits = [torch.randn(1, 2, 17, requires_grad=True) for _ in range(2)]
    gaussian = GaussianPrediction(points[-1], torch.ones(1, 2, 3) * .08,
        torch.tensor([[[1., 0, 0, 0], [1., 0, 0, 0]]]), torch.ones(1, 2, 1) * .1, logits[-1])
    inputs = dict(opus_pred_points=points, opus_pred_logits=logits,
        opus_point_valid_masks=[torch.ones(1, 2, dtype=torch.bool)] * 2,
        gaussian=gaussian, metas=dict(occ_xyz=gt,
            occ_label=torch.tensor([[3, 4]]), occ_cam_mask=torch.ones(1, 2, dtype=torch.bool)))
    settings = dict(stage_weights=(1., 1.), class_weights=[1.] * 17, final_gt_domain='all')
    original = CenterSupportDiagnosticSetLoss(**settings)
    original_value, old = original(inputs)
    baseline_geometry = torch.autograd.grad(original_value, points, retain_graph=True)
    baseline_early_logits = torch.autograd.grad(original_value, logits[0], retain_graph=True)[0]
    baseline_last_norm = torch.autograd.grad(original_value, logits[-1], retain_graph=True)[0].norm()
    ce_values, ce_gradients = [], []
    for assignment in ('nearest', 'render', 'transport'):
        updated = GaussianResponsibilitySetLoss(assignment=assignment, **settings)
        value, detail = updated(inputs)
        for name, old_value in old.items():
            if name == 'stage_1/classification':
                continue
            torch.testing.assert_close(torch.as_tensor(detail[name]), torch.as_tensor(old_value))
        new_geometry = torch.autograd.grad(value, points, retain_graph=True)
        for a, b in zip(baseline_geometry, new_geometry):
            torch.testing.assert_close(a, b)
        early = torch.autograd.grad(value, logits[0], retain_graph=True)[0]
        torch.testing.assert_close(early, baseline_early_logits)
        last = torch.autograd.grad(value, logits[-1], retain_graph=True)[0]
        torch.testing.assert_close(last.norm(), baseline_last_norm)
        ce_values.append(value); ce_gradients.append(last)
    # Disjoint single-class support must reduce to nearest-label CE.
    for value, gradient in zip(ce_values[1:], ce_gradients[1:]):
        torch.testing.assert_close(value, ce_values[0])
        torch.testing.assert_close(gradient, ce_gradients[0])
