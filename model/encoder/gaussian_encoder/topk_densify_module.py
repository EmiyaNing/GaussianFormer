from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction
from ...utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class TopkDensifyModule(BaseModule):
    '''
    Current version densify module only densify the most valueable topk gaussian ball.
        The topk count can be seted mannual.
        The shift of gaussian ball's mean, scale, rot, semantic, opacity all learned from instance features.
    '''
    def __init__(
        self,
        feat_embed_dim = 128,
        semantic_dim = 17,
        topk_count = 2560,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        **kwargs,
    ):
        super(TopkDensifyModule, self).__init__()
        self.topk_count = topk_count
        self.pc_range   = pc_range
        self.scale_range= scale_range
        unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
        unit_prob = [4 * unit_prob[i] for i in range(3)]
        self.unit_sigmoid = unit_prob

        self.feature_proj = nn.Sequential(
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU()
        )

        self.location_shift = nn.Linear(feat_embed_dim, 3)
        self.rotation_shift = nn.Linear(feat_embed_dim, 4)
        self.scale_shift    = nn.Linear(feat_embed_dim, 3)
        self.semantic_shift = nn.Linear(feat_embed_dim, semantic_dim)
        self.opacities_shift= nn.Linear(feat_embed_dim, 1)

        


    def forward(
            self,
            instance_feature: torch.Tensor,
            anchor: torch.Tensor,
            gaussian: GaussianPrediction
        ):
        '''
            This module reference the gaussian refine module, 
             generate some new gaussian ball from predicted high confidence gaussian ball.
        '''

        opacities = gaussian.opacities # B, N, 1
        means     = gaussian.means
        scales    = gaussian.scales
        rotations = gaussian.rotations
        semantics = gaussian.semantics

        batch_size= opacities.shape[0]
        new_features = []
        new_outputs  = []
        new_opacitices = []
        new_semantics  = []
        new_rotations  = []
        new_means      = []
        new_scales     = []


        for b in range(batch_size):
            cur_feats = instance_feature[b]
            cur_opa   = opacities[b] # N, 1
            cur_means = means[b]     # N, 3
            cur_scales= scales[b]    # N, 3
            cur_rots  = rotations[b] # N, 4 
            cur_sems  = semantics[b] # N, 17

            _, indices = torch.topk(cur_opa[:, 0], self.topk_count, dim=-1)
            filter_feats  = cur_feats[indices]
            filter_opa    = cur_opa[indices]
            filter_scales = cur_scales[indices]
            filter_rots   = cur_rots[indices]
            filter_sems   = cur_sems[indices]
            filter_means  = cur_means[indices]


            densified_feats = self.feature_proj(filter_feats)
            new_features.append(densified_feats)


            # refine module use the restrict xyz
            delta_xyz_sigmoid = self.location_shift(densified_feats)
            delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1
            mean_shift_out = torch.stack([
                delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
                delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
                delta_xyz_prob[..., 2] * self.unit_sigmoid[2]
            ], dim=-1)





            scale_shift_out = self.scale_shift(densified_feats)
            rots_shift_out  = F.normalize(self.rotation_shift(densified_feats))
            sem_shift_out   = self.semantic_shift(densified_feats)
            opa_shift_out   = self.opacities_shift(densified_feats)
            new_outputs.append(torch.cat([mean_shift_out, scale_shift_out, rots_shift_out, opa_shift_out, sem_shift_out], dim=-1))



            means_shift = (safe_sigmoid(mean_shift_out) - 0.5) * 2 # shift the center of gaussian ball
            scale_shift = (safe_sigmoid(scale_shift_out) - 0.5) * 2    # shift the scale of each gaussian ball
            rots_shift  = rots_shift_out
            sem_shift   = safe_sigmoid(sem_shift_out)
            opa_shift   = safe_sigmoid(opa_shift_out)

            cur_new_means = filter_means + means_shift * filter_scales
            cur_new_means_x = torch.clamp(cur_new_means[:, 0], self.pc_range[0], self.pc_range[3])
            cur_new_means_y = torch.clamp(cur_new_means[:, 1], self.pc_range[1], self.pc_range[4])
            cur_new_means_z = torch.clamp(cur_new_means[:, 2], self.pc_range[2], self.pc_range[5])
            cur_new_means = torch.stack([cur_new_means_x, cur_new_means_y, cur_new_means_z], dim=-1)

            cur_new_scales = filter_scales + filter_scales * scale_shift
            cur_new_scales = torch.clamp(cur_new_scales, self.scale_range[0], self.scale_range[1])

            new_means.append(cur_new_means)
            new_scales.append(cur_new_scales)
            new_rotations.append(filter_rots / 2 + rots_shift / 2)
            new_semantics.append(filter_sems * sem_shift)
            new_opacitices.append(filter_opa * opa_shift)

        new_features = torch.stack(new_features)
        new_outputs  = torch.stack(new_outputs)

        new_opacitices = torch.stack(new_opacitices)
        new_semantics  = torch.stack(new_semantics)
        new_means      = torch.stack(new_means)
        new_scales     = torch.stack(new_scales)
        new_rotations  = torch.stack(new_rotations)    

        result_features = torch.cat([instance_feature, new_features], dim=1)
        result_outputs  = torch.cat([anchor, new_outputs], dim=1)
        result_opacitices = torch.cat([opacities, new_opacitices], dim=1)
        result_semantics  = torch.cat([semantics, new_semantics], dim=1)
        result_scales     = torch.cat([scales, new_scales], dim=1)
        result_means      = torch.cat([means, new_means], dim=1)
        result_rotations  = torch.cat([rotations, new_rotations], dim=1)

        reuslt_gaussian = GaussianPrediction(
            means=result_means,
            scales=result_scales,
            rotations=result_rotations,
            opacities=result_opacitices,
            semantics=result_semantics
        )
        return result_outputs, reuslt_gaussian, result_features