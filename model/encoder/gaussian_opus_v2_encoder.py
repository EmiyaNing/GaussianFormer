"""Gaussian-OPUS V2 encoder with Gaussian-template 4D image sampling.

Tensor-symbol convention used throughout this file:

``B`` batch size, ``Q`` template/query count, ``R`` number of child Gaussians
predicted by one stage, ``T`` temporal frame count, ``N`` camera count, ``G``
feature group count, ``K`` keypoint count, ``L`` FPN level count, and ``D``
token/channel dimension.  Gaussian centres and scales are always expressed in
metres in the ego/world coordinate system unless a variable explicitly says
``normalized`` or ``prob``.
"""
import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from model.ops.opus_msmv_sampling import msmv_sampling
from model.utils.safe_ops import safe_inverse_sigmoid, safe_sigmoid
from model.utils.utils import get_rotation_matrix
from .gaussian_encoder.utils import GaussianPrediction
from .opus_encoder import _StrictAdaptiveMixing, _StrictOPUSSelfAttention, _opus_encode_points


def _scene_prob(points, lower, upper, eps=1e-4):
    """Convert world-coordinate points to bounded normalized scene coordinates.

    Args:
        points (Tensor): World-coordinate positions, normally ``[B,Q,3]`` or
            ``[B,Q,R,3]`` in metres.
        lower (Tensor): Scene lower bound, broadcastable to ``points`` and
            normally ``[3]`` in metres.
        upper (Tensor): Scene upper bound, broadcastable to ``points`` and
            normally ``[3]`` in metres.
        eps (float): Keeps probabilities away from exactly 0/1 so that a later
            ``safe_inverse_sigmoid`` remains finite.

    Returns:
        Tensor: Same shape and dtype as ``points``; normalized coordinates in
        the closed interval ``[eps, 1-eps]``.
    """
    # Clamping is deliberately done after coordinate normalization: bounds are
    # per axis and not a radial clamp in world space.
    return ((points - lower) / (upper - lower)).clamp(eps, 1.0 - eps)


def _next_parent(gaussian, keep_geometry_grad):
    """Build the parent Gaussian consumed by the next decoder stage.

    Args:
        gaussian (GaussianPrediction): Current-stage grouped prediction.
            ``means/scales`` have shape ``[B,Q,R,3]``; ``rotations`` is
            ``[B,Q,R,4]``; ``opacities`` is ``[B,Q,R,1]``; and ``semantics`` is
            ``[B,Q,R,C]``.
        keep_geometry_grad (bool): When True, later-stage loss can propagate
            into earlier means/scales. When False, all geometry is detached.

    Returns:
        GaussianPrediction: Same shapes as ``gaussian``. Only means/scales may
        retain an autograd connection; attributes not read by V2 sampling are
        always detached to avoid retaining an unused graph.
    """
    geometry = (lambda x: x) if keep_geometry_grad else (lambda x: x.detach())
    return GaussianPrediction(
        geometry(gaussian.means), geometry(gaussian.scales),
        gaussian.rotations.detach(), gaussian.opacities.detach(), gaussian.semantics.detach())


def world_to_sampling_gaussian(parent):
    """Reduce a grouped parent prediction to one Gaussian used for sampling.

    Args:
        parent (GaussianPrediction): Grouped stage output. Required fields are
            ``means [B,Q,R,3]``, ``scales [B,Q,R,3]`` and
            ``rotations [B,Q,R,4]``.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - center: ``[B,Q,3]`` mean child centre in metres.
            - scale: ``[B,Q,3]`` mean child scale in metres, lower-bounded by
              ``1e-4`` to keep keypoint generation well defined.
            - rotation: ``[B,Q,4]`` unit quaternion obtained by averaging
              child quaternions. A near-zero average uses identity rotation.
    """
    center = parent.means.mean(dim=2)
    scale = parent.scales.mean(dim=2).clamp_min(1e-4)
    rotation = parent.rotations.mean(dim=2)
    norm = rotation.norm(dim=-1, keepdim=True)
    # Opposite quaternions can cancel during averaging. Use identity rather
    # than normalizing a near-zero vector, which would produce unstable axes.
    identity = torch.zeros_like(rotation)
    identity[..., 0] = 1.0
    rotation = F.normalize(torch.where(norm > 1e-6, rotation, identity), dim=-1, eps=1e-6)
    return center, scale, rotation


class GaussianTemplateKeypointGenerator(BaseModule):
    """Generate rotated, scale-aware world keypoints around each Gaussian.

    This is the V2 world-coordinate counterpart of
    ``SparseGaussian3DKeyPointsGenerator``. It does not require an intermediate
    sigmoid-logit anchor: a fixed local template is multiplied by the current
    sampling Gaussian scale, rotated by its quaternion, and translated to its
    centre.

    Owned attributes:
        template (Tensor buffer): ``[K_fixed,3]`` local offsets after
            multiplication by this stage's ``stage_step``. It is moved with the
            module but is not trainable.
        num_learnable_pts (int): Number of token-conditioned offsets appended
            to ``template``.
        learnable_fixed_scale (float): Maximum magnitude multiplier applied to
            learned offsets after they are mapped to ``[-0.5, 0.5]``.
        offset_fc (Linear | None): Maps token ``[B,Q,D]`` to
            ``[B,Q,K_learnable,3]`` offsets; absent when no learned points are
            requested.

    Methods:
        ``__init__`` validates and stores the stage-scaled template.
        ``forward`` creates world-coordinate keypoints for every query.
    """
    def __init__(self, embed_dims, template, stage_step, num_learnable_pts=0,
                 learnable_fixed_scale=1.0):
        """Initialize the fixed and optional learned keypoint template.

        Args:
            embed_dims (int): Token dimension ``D``; input width of
                ``offset_fc`` when learned points are enabled.
            template (Sequence[Sequence[float]] | Tensor): ``[K_fixed,3]``
                dimensionless local offsets. It must include ``[0,0,0]``.
            stage_step (float): Positive stage-specific multiplier for fixed
                offsets. Larger values sample a larger Gaussian neighbourhood.
            num_learnable_pts (int): Number of additional learned keypoints.
            learnable_fixed_scale (float): Offset range multiplier for learned
                points before multiplication by Gaussian scale.
        """
        super().__init__()
        template = torch.as_tensor(template, dtype=torch.float32)
        if template.ndim != 2 or template.shape[-1] != 3:
            raise ValueError('sampling_template must have shape [K, 3]')
        if not torch.any(template.abs().sum(dim=-1) == 0):
            raise ValueError('sampling_template must include the center [0, 0, 0]')
        if stage_step <= 0:
            raise ValueError('stage_step must be positive')
        self.register_buffer('template', template * float(stage_step))
        self.num_learnable_pts = num_learnable_pts
        self.learnable_fixed_scale = learnable_fixed_scale
        self.offset_fc = (nn.Linear(embed_dims, num_learnable_pts * 3)
                          if num_learnable_pts else None)

    def forward(self, center, scale, rotation, token):
        """Generate world-coordinate sampling keypoints.

        Args:
            center (Tensor): ``[B,Q,3]`` Gaussian centres in metres.
            scale (Tensor): ``[B,Q,3]`` per-axis Gaussian scales in metres.
            rotation (Tensor): ``[B,Q,4]`` quaternion rotations; it is
                normalized defensively before conversion to a rotation matrix.
            token (Tensor): ``[B,Q,D]`` query token. Used only when
                ``num_learnable_pts > 0``.

        Returns:
            Tensor: ``[B,Q,K_fixed+K_learnable,3]`` keypoints in metres.
        """
        batch, queries = center.shape[:2]
        offsets = self.template.to(center).view(1, 1, -1, 3).expand(batch, queries, -1, -1)
        if self.offset_fc is not None:
            learned = safe_sigmoid(self.offset_fc(token).view(
                batch, queries, self.num_learnable_pts, 3)) - .5
            offsets = torch.cat([offsets, learned * self.learnable_fixed_scale], dim=2)
        # First scale local template axes, then rotate, then translate. This
        # order makes an anisotropic Gaussian's template follow its orientation.
        local = offsets * scale.unsqueeze(2)
        rotation_matrix = get_rotation_matrix(F.normalize(rotation, dim=-1, eps=1e-6))
        return center.unsqueeze(2) + torch.matmul(rotation_matrix.unsqueeze(2), local.unsqueeze(-1)).squeeze(-1)


def project_keypoints_4d(keypoints, metas, num_frames, num_views, num_groups, eps=1e-5):
    """Project world keypoints into OPUS/MSMV 4D sampling locations.

    A single first-valid camera is selected independently for every
    ``(B,T,Q,G,K)`` entry. The final third MSMV coordinate stores that selected
    camera index normalized to ``[0,1]``.

    Args:
        keypoints (Tensor): ``[B,Q,K,3]`` world-coordinate positions in metres.
        metas (dict): Must contain ``projection_mat [B,T*N,4,4]`` (ego/world
            homogeneous point to image projection) and ``image_wh [B,T*N,2]``
            pixel ``(width,height)`` pairs.
        num_frames (int): Temporal frame count ``T``.
        num_views (int): Number of cameras per frame ``N``.
        num_groups (int): Feature group count ``G``. Geometry is duplicated for
            groups because each group has a separate feature-channel partition.
        eps (float): Positive depth threshold and denominator lower bound.

    Returns:
        tuple[Tensor, Tensor]:
            - locations: ``[B*T*G,Q,K,3]`` normalized MSMV locations
              ``(u,v,view_index/(N-1))``.
            - selected_valid: bool ``[B,T,Q,G,K]``; False means no camera
              observes the keypoint and its sampled feature must be masked.
    """
    batch, queries, keypoint_count, _ = keypoints.shape
    projection = metas['projection_mat'].to(keypoints)
    image_wh = metas['image_wh'].to(keypoints)
    if projection.shape[1] != num_frames * num_views:
        raise ValueError('projection_mat camera dimension must equal T * N')
    projection = projection.view(batch, num_frames, num_views, 4, 4)
    image_wh = image_wh.view(batch, num_frames, num_views, 2)
    # Copy keypoints across frames/groups before projection; no temporal motion
    # transform is applied here because metas already supplies per-frame camera
    # projection matrices in the chosen ego/world convention.
    homogeneous = torch.cat([keypoints, torch.ones_like(keypoints[..., :1])], dim=-1)
    homogeneous = homogeneous[:, None, :, None].expand(
        batch, num_frames, queries, num_groups, keypoint_count, 4)
    projected = torch.einsum('btnij,btqgkj->btnqgki', projection, homogeneous)
    depth = projected[..., 2:3]
    uv = projected[..., :2] / depth.clamp_min(eps)
    uv = uv / image_wh[:, :, :, None, None, None, :]
    valid = ((depth[..., 0] > eps) & (uv[..., 0] > 0.) & (uv[..., 0] < 1.) &
             (uv[..., 1] > 0.) & (uv[..., 1] < 1.))
    # argmax returns camera 0 for all-miss entries; selected_valid records that
    # case separately so downstream aggregation can gate its sampled feature.
    selected_view = valid.float().argmax(dim=2)  # [B,T,Q,G,K]
    gather_index = selected_view[:, :, None, :, :, :, None].expand(-1, -1, 1, -1, -1, -1, 2)
    selected_uv = torch.gather(uv, 2, gather_index).squeeze(2)
    selected_valid = torch.gather(valid, 2, selected_view[:, :, None]).squeeze(2)
    view_coordinate = selected_view.to(selected_uv.dtype) / max(num_views - 1, 1)
    locations = torch.cat([selected_uv, view_coordinate.unsqueeze(-1)], dim=-1)
    locations = locations.permute(0, 1, 3, 2, 4, 5).reshape(
        batch * num_frames * num_groups, queries, keypoint_count, 3).contiguous()
    return locations, selected_valid  # selected_valid: [B,T,Q,G,K]


class GaussianOPUS4DFeatureAggregation(BaseModule):
    """Template keypoint based 4D aggregation for OPUS adaptive mixing.

    The returned feature keeps its [group, time * keypoint, channel] axes so
    ``_StrictAdaptiveMixing`` can perform the original query-conditioned point
    and channel mixing instead of receiving a prematurely summed feature.
    
    Owned attributes:
        num_frames, num_views, num_groups, num_levels (int): ``T/N/G/L``
            sampling dimensions validated by the caller's image feature layout.
        keypoint_generator (GaussianTemplateKeypointGenerator): Generates the
            ``K`` world points sampled for each query.
        attn_drop (float): Training-only Bernoulli drop probability for FPN
            fusion weights.
        group_dims (int): Per-group channel dimension ``D/G``.
        num_keypoints (int): Total fixed plus learned keypoint count ``K``.
        weight_fc (Linear): Produces per-query FPN-level logits with output
            width ``G*K*L``.

    Methods:
        ``__init__`` records sampling dimensions and builds ``weight_fc``.
        ``forward`` generates keypoints, projects them, fuses FPN levels, and
        returns the strict OPUS adaptive-mixing tensor.
    """
    def __init__(self, embed_dims, num_frames, num_views, num_groups, num_levels,
                 keypoint_generator, attn_drop=0.):
        """Initialize 4D image-feature aggregation.

        Args:
            embed_dims (int): Full token/channel dimension ``D``.
            num_frames (int): Number of temporal frames ``T``.
            num_views (int): Cameras per frame ``N``.
            num_groups (int): Channel groups ``G``; must divide ``embed_dims``.
            num_levels (int): FPN levels ``L`` consumed by MSMV sampling.
            keypoint_generator (GaussianTemplateKeypointGenerator): Per-stage
                keypoint generator that determines ``K``.
            attn_drop (float): Optional dropout on already-normalized FPN
                weights during training.
        """
        super().__init__()
        if embed_dims % num_groups:
            raise ValueError('embed_dims must be divisible by num_groups')
        self.num_frames, self.num_views = num_frames, num_views
        self.num_groups, self.num_levels = num_groups, num_levels
        self.keypoint_generator = keypoint_generator
        self.attn_drop = attn_drop
        self.group_dims = embed_dims // num_groups
        # Learnable points change K at construction time, so include them explicitly.
        self.num_keypoints = keypoint_generator.template.shape[0] + keypoint_generator.num_learnable_pts
        self.weight_fc = nn.Linear(embed_dims, num_groups * self.num_keypoints * num_levels)

    def forward(self, token, center, scale, rotation, grouped_features, metas):
        """Sample and fuse multi-level image features without collapsing points.

        Args:
            token (Tensor): ``[B,Q,D]`` query features that produce FPN-level
                weights and optional learned keypoint offsets.
            center (Tensor): ``[B,Q,3]`` sampling Gaussian centres in metres.
            scale (Tensor): ``[B,Q,3]`` sampling Gaussian scales in metres.
            rotation (Tensor): ``[B,Q,4]`` sampling Gaussian quaternions.
            grouped_features (list[Tensor]): Length ``L``. Each entry has
                ``[B*T*G,N,H_l,W_l,D/G]`` and is channel-last for MSMV.
            metas (dict): Projection/image-size metadata required by
                ``project_keypoints_4d``.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - sampled: ``[B,Q,G,T*K,D/G]`` FPN-fused features, preserved
                  for query-conditioned OPUS adaptive mixing.
                - visible: bool ``[B,Q]``; True when any keypoint is visible in
                  any frame/group.
                - keypoints: ``[B,Q,K,3]`` world-coordinate points in metres;
                  returned for diagnostics.
        """
        batch, queries, _ = token.shape
        keypoints = self.keypoint_generator(center, scale, rotation, token)
        locations, selected_valid = project_keypoints_4d(
            keypoints, metas, self.num_frames, self.num_views, self.num_groups)
        weights = self.weight_fc(token).view(
            batch, queries, self.num_groups, self.num_keypoints, self.num_levels)
        weights = weights[:, :, None].expand(-1, -1, self.num_frames, -1, -1, -1)
        # [B,Q,T,G,K,L], with an independently normalized FPN distribution.
        valid = selected_valid.permute(0, 2, 1, 3, 4).unsqueeze(-1)
        # An all-miss keypoint is sampled outside the image and therefore has
        # zero feature value.  Feed zero logits to softmax to avoid NaNs; the
        # projection mask, rather than a fragile -inf path, enforces zero data.
        weights = torch.where(valid, weights, torch.zeros_like(weights)).softmax(dim=-1)
        if self.training and self.attn_drop:
            keep = torch.rand_like(weights) >= self.attn_drop
            weights = weights * keep.to(weights.dtype) / (1. - self.attn_drop)
        # MSMV expects its leading batch axis to be B*T*G, exactly matching
        # grouped_features and locations produced above.
        msmv_weights = weights.permute(0, 2, 3, 1, 4, 5).reshape(
            batch * self.num_frames * self.num_groups, queries, self.num_keypoints, self.num_levels).contiguous()
        sampled = msmv_sampling(grouped_features, locations, msmv_weights)
        sampled = sampled.view(batch, self.num_frames, self.num_groups, queries,
                               self.num_keypoints, self.group_dims)
        # A behind-camera point can have an in-image UV.  Gate sampled values
        # explicitly so an all-miss keypoint cannot leak image features.
        sampled = sampled * selected_valid.permute(0, 1, 3, 2, 4).unsqueeze(-1).to(sampled.dtype)
        # [B,T,G,Q,K,D/G] -> [B,Q,G,T*K,D/G], the strict OPUS contract.
        sampled = sampled.permute(0, 3, 2, 1, 4, 5).reshape(
            batch, queries, self.num_groups,
            self.num_frames * self.num_keypoints, self.group_dims)
        visible = selected_valid.any(dim=(1, 3, 4))
        return sampled, visible, keypoints


class GaussianOPUSV2RefinementHead(BaseModule):
    """Decode every query token into raw outputs and bounded child Gaussians.

    The head intentionally keeps one semantic branch in *every* V2 stage, so
    ``apply_loss_type='all'`` can directly supervise every stage. Geometry uses
    11 values per child: center residual logits (3), scale logits (3), raw
    quaternion (4), and opacity logit (1).

    Owned attributes:
        num_refine (int): Child Gaussian count ``R`` emitted by this stage.
        semantic_dim (int): Non-empty semantic score channels ``C``.
        stage_step (float): Multiplies the OPUS-style bounded center offset.
        center_step_multiplier (float): Additional user-controlled center range
            multiplier; ``1`` exactly uses ``stage_step * parent_scale``.
        pc_lower, pc_upper (Tensor buffers): ``[3]`` world scene bounds.
        scale_min, scale_max (Tensor buffers): ``[3]`` allowed decoded scale
            bounds in metres.
        trunk (Sequential): Shared token MLP producing ``[B,Q,D]`` features.
        geometry (Linear): Maps trunk features to ``R*11`` raw geometry values.
        semantic (Linear): Maps trunk features to ``R*C`` raw semantic values.

    Methods:
        ``__init__`` creates branches and initializes a stable identity/
        low-opacity Gaussian prior.
        ``forward`` emits raw tensors plus decoded ``GaussianPrediction``.
    """
    def __init__(self, embed_dims, num_refine, semantic_dim, pc_range, scale_range,
                 stage_step, center_step_multiplier=1.):
        """Initialize the per-stage Gaussian refinement head.

        Args:
            embed_dims (int): Token dimension ``D``.
            num_refine (int): Number of child Gaussians ``R`` per query.
            semantic_dim (int): Non-empty semantic channels ``C`` per child.
            pc_range (Sequence[float]): ``(x_min,y_min,z_min,x_max,y_max,z_max)``
                world bounds in metres.
            scale_range (Sequence[Sequence[float]]): Min/max per-axis decoded
                Gaussian scale, each ``[3]`` in metres.
            stage_step (float): Local center residual range multiplier.
            center_step_multiplier (float): Additional global multiplier for
                center refinement; defaults to one.
        """
        super().__init__()
        self.num_refine, self.semantic_dim = num_refine, semantic_dim
        self.stage_step = stage_step
        self.center_step_multiplier = center_step_multiplier
        self.register_buffer('pc_lower', torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer('pc_upper', torch.tensor(pc_range[3:], dtype=torch.float32))
        self.register_buffer('scale_min', torch.tensor(scale_range[0], dtype=torch.float32))
        self.register_buffer('scale_max', torch.tensor(scale_range[1], dtype=torch.float32))
        self.trunk = nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
                                   nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True))
        self.geometry = nn.Linear(embed_dims, num_refine * 11)
        self.semantic = nn.Linear(embed_dims, num_refine * semantic_dim)
        # Zero output weights make the initial prediction deterministic: parent
        # center, midpoint scale, identity quaternion, and 0.1 opacity.
        nn.init.zeros_(self.geometry.weight); nn.init.zeros_(self.geometry.bias)
        nn.init.zeros_(self.semantic.weight); nn.init.constant_(self.semantic.bias, -4.59511985)
        with torch.no_grad():
            self.geometry.bias.view(num_refine, 11)[:, 6] = 1.
            self.geometry.bias.view(num_refine, 11)[:, 10] = safe_inverse_sigmoid(torch.tensor(.1))

    def forward(self, token, parent, valid_mask):
        """Decode child Gaussian parameters from a stage token.

        Args:
            token (Tensor): Transformer output ``[B,Q,D]``.
            parent (GaussianPrediction): Previous grouped Gaussian with
                ``means/scales [B,Q,R_parent,3]``. Only these two fields are
                used for the OPUS-style local center proposal.
            valid_mask (Tensor): bool ``[B,Q]`` query validity mask. Invalid
                queries retain legal geometry but receive zero opacity.

        Returns:
            tuple[Tensor, Tensor, GaussianPrediction]:
                - geometry_logits: ``[B,Q,R,11]`` raw network output.
                - semantic_logits: ``[B,Q,R,C]`` raw network output before
                  ``softplus``.
                - gaussian: Decoded grouped prediction with means/scales
                  ``[B,Q,R,3]``, rotations ``[B,Q,R,4]``, opacities
                  ``[B,Q,R,1]``, and non-negative semantics ``[B,Q,R,C]``.
        """
        batch, queries, _ = token.shape
        hidden = self.trunk(token)
        geometry_logits = self.geometry(hidden).view(batch, queries, self.num_refine, 11)
        semantic_logits = self.semantic(hidden).view(batch, queries, self.num_refine, self.semantic_dim)
        parent_center = parent.means.mean(dim=2, keepdim=True)
        parent_scale = parent.scales.mean(dim=2, keepdim=True).clamp_min(1e-4)
        extent = self.pc_upper.to(token) - self.pc_lower.to(token)
        # Convert parent centre to logit space, add a bounded metric-scaled
        # residual, then map it back. This keeps all means inside pc_range.
        center_logit = safe_inverse_sigmoid(_scene_prob(parent_center, self.pc_lower.to(token), self.pc_upper.to(token)))
        # Match OPUS's bounded local proposal range: stage_step × parent scale.
        center_logit = center_logit + (self.center_step_multiplier * self.stage_step *
                                       torch.tanh(geometry_logits[..., :3]) * (parent_scale / extent))
        means = self.pc_lower.to(token) + extent * safe_sigmoid(center_logit)
        # Unlike V1's exp/log-scale residual, V2 treats scale logits as direct
        # raw output and maps them safely into the configured physical range.
        scales = self.scale_min.to(token) + (self.scale_max.to(token) - self.scale_min.to(token)) * safe_sigmoid(geometry_logits[..., 3:6])
        rotations = F.normalize(geometry_logits[..., 6:10], dim=-1, eps=1e-6)
        opacity = safe_sigmoid(geometry_logits[..., 10:11])
        semantics = F.softplus(semantic_logits)
        opacity = opacity * valid_mask[:, :, None, None].to(opacity.dtype)
        return geometry_logits, semantic_logits, GaussianPrediction(means, scales, rotations, opacity, semantics)


class GaussianOPUSV2Layer(BaseModule):
    """One V2 decoder stage: Gaussian sampling, OPUS mixing/attention, refine.

    Owned attributes:
        pc_range (tuple[float]): Six world-coordinate scene bounds.
        image_aggregation (GaussianOPUS4DFeatureAggregation): Produces
            ``[B,Q,G,T*K,D/G]`` image features from Gaussian keypoints.
        position_encoder (Sequential): Encodes normalized center and relative
            scale ``[B,Q,6]`` into a token positional feature ``[B,Q,D]``.
        mixing (_StrictAdaptiveMixing): Original OPUS query-conditioned point
            and channel mixing over the sampled ``T*K`` positions.
        self_attn (_StrictOPUSSelfAttention): Original geometry-aware attention
            using pairwise query-center distances.
        ffn (Sequential), norm1/norm2/norm3 (LayerNorm): Standard decoder token
            update blocks.
        refine (GaussianOPUSV2RefinementHead): Emits this stage's ``R`` child
            Gaussians and semantic scores.

    Methods:
        ``__init__`` creates all stage-specific modules.
        ``forward`` runs one parent-to-child Gaussian refinement stage.
    """
    def __init__(self, embed_dims, num_frames, num_views, num_groups, num_levels,
                 num_heads, feedforward_channels, dropout, num_refine, semantic_dim,
                 stage_step, pc_range, scale_range, sampling_template,
                 num_learnable_pts, learnable_fixed_scale, center_step_multiplier,
                 attn_drop):
        """Initialize one decoder stage.

        Args:
            embed_dims, num_heads, feedforward_channels, dropout: Standard
                transformer dimensions/hyperparameters (``D`` and head count).
            num_frames, num_views, num_groups, num_levels: Image sampling
                dimensions ``T/N/G/L``.
            num_refine (int): Child Gaussians ``R`` emitted by this stage.
            semantic_dim (int): Per-child semantic channels ``C``.
            stage_step (float): Fixed-template scale and center residual range.
            pc_range, scale_range: World scene bounds and decoded scale bounds.
            sampling_template: ``[K_fixed,3]`` local template offsets.
            num_learnable_pts, learnable_fixed_scale: Optional learned-keypoint
                settings passed to ``GaussianTemplateKeypointGenerator``.
            center_step_multiplier, attn_drop: Refinement range and FPN-weight
                regularization controls.
        """
        super().__init__()
        self.pc_range = tuple(pc_range)
        generator = GaussianTemplateKeypointGenerator(embed_dims, sampling_template, stage_step,
                                                       num_learnable_pts, learnable_fixed_scale)
        self.image_aggregation = GaussianOPUS4DFeatureAggregation(
            embed_dims, num_frames, num_views, num_groups, num_levels, generator,
            attn_drop)
        self.position_encoder = nn.Sequential(nn.Linear(6, embed_dims), nn.LayerNorm(embed_dims),
                                              nn.ReLU(inplace=True), nn.Linear(embed_dims, embed_dims))
        self.mixing = _StrictAdaptiveMixing(
            embed_dims, num_frames * self.image_aggregation.num_keypoints, num_groups, 32)
        self.self_attn = _StrictOPUSSelfAttention(embed_dims, num_heads, dropout, pc_range)
        self.ffn = nn.Sequential(nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(feedforward_channels, embed_dims), nn.Dropout(dropout))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(embed_dims), nn.LayerNorm(embed_dims), nn.LayerNorm(embed_dims)
        self.refine = GaussianOPUSV2RefinementHead(embed_dims, num_refine, semantic_dim,
                                                    pc_range, scale_range, stage_step,
                                                    center_step_multiplier)

    def forward(self, parent, token, grouped_features, metas, valid_mask):
        """Run sampling, transformer update, and Gaussian refinement for one stage.

        Args:
            parent (GaussianPrediction): Grouped parent Gaussian, normally
                ``means/scales [B,Q,R_parent,3]``.
            token (Tensor): Incoming query features ``[B,Q,D]``.
            grouped_features (list[Tensor]): Length ``L`` image feature maps,
                each ``[B*T*G,N,H_l,W_l,D/G]``.
            metas (dict): Batched projection and image-size metadata.
            valid_mask (Tensor): bool ``[B,Q]`` mask for active queries.

        Returns:
            tuple[Tensor, Tensor, Tensor, GaussianPrediction, Tensor]:
                - token: Updated and validity-masked token ``[B,Q,D]``.
                - geometry: Raw geometry logits ``[B,Q,R,11]``.
                - semantics: Raw semantic logits ``[B,Q,R,C]``.
                - gaussian: Decoded grouped child Gaussian.
                - visible: bool ``[B,Q]`` image visibility summary.
        """
        center, scale, rotation = world_to_sampling_gaussian(parent)
        lower = center.new_tensor(self.pc_range[:3]); upper = center.new_tensor(self.pc_range[3:])
        position = torch.cat([_scene_prob(center, lower, upper), scale / (upper - lower)], dim=-1)
        token = token + self.position_encoder(position)
        sampled, visible, _ = self.image_aggregation(
            token, center, scale, rotation, grouped_features, metas)
        token = self.norm1(self.mixing(sampled, token))
        # Strict OPUS attention expects a point-set axis. V2 uses one sampling
        # center per query, hence the singleton ``R=1`` axis.
        normalized_center = _opus_encode_points(center, self.pc_range).unsqueeze(2)
        token = self.norm2(self.self_attn(normalized_center, token))
        token = self.norm3(token + self.ffn(token))
        geometry, semantics, gaussian = self.refine(token, parent, valid_mask)
        return token * valid_mask.unsqueeze(-1).to(token.dtype), geometry, semantics, gaussian, visible


@MODELS.register_module()
class GaussianOPUSV2Encoder(BaseModule):
    """Top-level V2 encoder that iteratively refines grouped Gaussians.

    The encoder starts from one Gaussian template per query, runs ``num_decoder``
    stages, and stores every stage output for renderer deep supervision.

    Owned attributes:
        num_frames, num_views, num_groups (int): ``T/N/G`` image layout used
            by ``_group_features`` and every V2 stage.
        cross_stage_geometry_grad (bool): Whether means/scales remain connected
            across stages for end-to-end geometry refinement.
        layers (ModuleList[GaussianOPUSV2Layer]): Ordered stage modules; stage
            ``s`` emits ``num_refines[s]`` child Gaussians per query.

    Methods:
        ``__init__`` validates global configuration and constructs stages.
        ``_group_features`` converts backbone/FPN maps to the MSMV layout.
        ``forward`` runs all stages and returns their representations.
    """
    def __init__(self, embed_dims=256, num_decoder=5, num_frames=1, num_views=6,
                 num_levels=4, num_groups=4, num_heads=8, feedforward_channels=512,
                 dropout=.1, semantic_dim=17, num_refines=(1, 4, 8, 16, 32),
                 stage_steps=(4., 3.6, 3.2, 2.8, 2.4),
                 sampling_template=((0, 0, 0), (.45, 0, 0), (-.45, 0, 0),
                                    (0, .45, 0), (0, -.45, 0), (0, 0, .45), (0, 0, -.45)),
                 num_learnable_pts=0, learnable_fixed_scale=1.,
                 scale_range=((.08, .08, .08), (.8, .8, .8)), center_step_multiplier=1.,
                 attn_drop=.0, cross_stage_geometry_grad=True,
                 pc_range=(-40., -40., -1., 40., 40., 5.4), init_cfg=None):
        """Initialize a multi-stage Gaussian-OPUS V2 decoder.

        Args:
            embed_dims (int): Query token dimension ``D``.
            num_decoder (int): Number of decoder/refinement stages.
            num_frames, num_views, num_levels, num_groups: Image sampling
                dimensions ``T/N/L/G``.
            num_heads, feedforward_channels, dropout: Transformer parameters.
            semantic_dim (int): Non-empty semantic channels ``C``.
            num_refines (Sequence[int]): Child count ``R_s`` for each stage.
            stage_steps (Sequence[float]): Per-stage template and center-offset
                ranges; must have ``num_decoder`` positive entries.
            sampling_template: Fixed local offsets ``[K_fixed,3]``.
            num_learnable_pts, learnable_fixed_scale: Optional learned keypoint
                configuration shared by all stages.
            scale_range: ``([3] min, [3] max)`` decoded scale range in metres.
            center_step_multiplier, attn_drop: Refinement and sampling controls.
            cross_stage_geometry_grad (bool): Preserve means/scales gradients
                across stages when True.
            pc_range (Sequence[float]): Six world-coordinate scene bounds.
            init_cfg (dict | None): MMEngine initialization configuration.
        """
        super().__init__(init_cfg)
        if len(num_refines) != num_decoder or len(stage_steps) != num_decoder:
            raise ValueError('num_refines and stage_steps must provide one entry per decoder layer')
        if max(scale_range[1]) >= 1.0:
            raise ValueError('decoded Gaussian scales must remain below 1m for local_aggregate')
        self.num_frames, self.num_views, self.num_groups = num_frames, num_views, num_groups
        self.cross_stage_geometry_grad = cross_stage_geometry_grad
        self.layers = nn.ModuleList([
            GaussianOPUSV2Layer(embed_dims, num_frames, num_views, num_groups, num_levels, num_heads,
                                 feedforward_channels, dropout, refine_count, semantic_dim, step,
                                 pc_range, scale_range, sampling_template, num_learnable_pts,
                                 learnable_fixed_scale, center_step_multiplier, attn_drop)
            for refine_count, step in zip(num_refines, stage_steps)
        ])

    def _group_features(self, features, batch):
        """Convert FPN maps from segmentor layout to OPUS/MSMV grouped layout.

        Args:
            features (list[Tensor]): Length ``L`` feature maps, each
                ``[B,T*N,D,H_l,W_l]`` in channel-first backbone layout.
            batch (int): Expected batch size ``B``; checked implicitly by
                reshape and supplied by ``forward``.

        Returns:
            list[Tensor]: Length ``L``; each map is
            ``[B*T*G,N,H_l,W_l,D/G]`` and contiguous/channel-last, the exact
            ``msmv_sampling`` feature contract.
        """
        grouped = []
        for feature in features:
            _, cameras, channels, height, width = feature.shape
            if cameras != self.num_frames * self.num_views or channels % self.num_groups:
                raise ValueError('V2 image feature shape must be [B,T*N,D,H,W] with D divisible by groups')
            # Split channels into groups before moving camera/spatial axes to
            # MSMV's channel-last representation.
            grouped.append(feature.reshape(batch, self.num_frames, self.num_views, self.num_groups,
                                           channels // self.num_groups, height, width).permute(
                0, 1, 3, 2, 5, 6, 4).reshape(batch * self.num_frames * self.num_groups,
                                               self.num_views, height, width, channels // self.num_groups).contiguous())
        return grouped

    def forward(self, query_features, query_templates, ms_img_feats, metas, query_valid_mask=None, **kwargs):
        """Run all V2 stages from semantic Gaussian templates to child Gaussians.

        Args:
            query_features (Tensor): Initial learned query tokens ``[B,Q,D]``.
            query_templates (GaussianPrediction): Template attributes with
                means/scales ``[B,Q,3]``, rotations ``[B,Q,4]``, opacities
                ``[B,Q,1]`` and semantics ``[B,Q,C]``.
            ms_img_feats (list[Tensor]): FPN maps, each ``[B,T*N,D,H_l,W_l]``.
            metas (dict): Must expose camera ``projection_mat`` and ``image_wh``
                tensors described in ``project_keypoints_4d``.
            query_valid_mask (Tensor | None): Optional bool ``[B,Q]``. None
                means every query is active.
            **kwargs: Accepted for generic segmentor compatibility; unused.

        Returns:
            dict: ``{'representation': states}``, where ``states`` has one
            dictionary per stage. Each dictionary contains query features
            ``[B,Q,D]``, grouped decoded Gaussian tensors, raw geometry/semantic
            logits, query mask, and visibility ``[B,Q]``.
        """
        batch, queries = query_features.shape[:2]
        if query_valid_mask is None:
            query_valid_mask = torch.ones(batch, queries, device=query_features.device, dtype=torch.bool)
        grouped_features = self._group_features(ms_img_feats, batch)
        # Decoder stages always consume grouped Gaussians. The lifter template
        # therefore receives a singleton child axis ``R_parent=1``.
        parent = GaussianPrediction(query_templates.means.unsqueeze(2), query_templates.scales.unsqueeze(2),
                                    query_templates.rotations.unsqueeze(2), query_templates.opacities.unsqueeze(2),
                                    query_templates.semantics.unsqueeze(2))
        representation = []
        token = query_features
        for layer in self.layers:
            token, raw_geometry, raw_semantics, gaussian, visible = layer(
                parent, token, grouped_features, metas, query_valid_mask)
            representation.append({'query_features': token, 'gaussian': gaussian,
                                   'raw_geometry': raw_geometry, 'raw_semantics': raw_semantics,
                                   'query_valid_mask': query_valid_mask, 'visible': visible})
            # Only means/scales affect V2's next-stage sampling geometry.
            parent = _next_parent(gaussian, self.cross_stage_geometry_grad)
        return {'representation': representation}
