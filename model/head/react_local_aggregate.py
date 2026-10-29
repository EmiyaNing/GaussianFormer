"""Project-owned React wrapper; reuse the installed CUDA/autograd extension.

Keeping Python execution changes here avoids requiring a pip reinstall or a
CUDA rebuild when local_aggregate_react was installed as a copied package.
"""

import torch
from local_aggregate_react import LocalAggregator, _LocalAggregate


class PreparedReactAggregator(LocalAggregator):
    def __init__(self, *args, validate_inputs=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate_inputs = validate_inputs

    def forward(self, pts, means3D, opacities, semantics, scales, cov3D,
                points_int=None):
        if pts.shape[0] != 1 or pts.requires_grad:
            raise ValueError('React rendering requires one batch of fixed query points')
        pts = pts.squeeze(0)
        means3D = means3D.squeeze(0)
        opacities = opacities.squeeze(0)
        semantics = semantics.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)
        if points_int is None:
            points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int32)
        means_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int32)
        radii = torch.ceil(scales * self.scale_multiplier / self.grid_size).to(torch.int32)
        if self.validate_inputs:
            assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
            assert means_int.min() >= 0 and means_int[:, 0].max() < self.H and means_int[:, 1].max() < self.W and means_int[:, 2].max() < self.D
            assert radii.min() >= 1
        inverse = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]
        if self.inv_softmax:
            raise NotImplementedError('inverse softmax is not supported by React')
        return _LocalAggregate.apply(pts, points_int, means3D, means_int, opacities,
                                     semantics, radii, inverse, self.H, self.W, self.D)
