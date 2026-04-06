# Adapted from https://github.com/niki-amini-naieni/CountGD
"""
Deformable Attention Modules.
Refactored implementation of multi-scale attention mechanisms.
"""
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


def _check_power_of_2(val):
    """Verifies if the integer is a power of two."""
    if not isinstance(val, int) or val < 0:
        raise ValueError(f"Input must be a positive int, got {val} ({type(val)})")
    return (val != 0) and (val & (val - 1) == 0)


def ms_deform_attn_core_pytorch(
    feat_value: torch.Tensor,
    shapes: torch.Tensor,
    sampling_coords: torch.Tensor,
    attn_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of Multi-Scale Deformable Attention.
    """
    # 
    
    batch_sz, _, n_heads, d_embed = feat_value.shape
    _, n_queries, n_heads, n_levels, n_points, _ = sampling_coords.shape
    
    # Split features by level based on H*W
    split_lens = [h * w for h, w in shapes]
    feat_splits = feat_value.split(split_lens, dim=1)
    
    # Convert [0, 1] coords to [-1, 1] for grid_sample
    grid_coords = 2 * sampling_coords - 1
    sampled_list = []
    
    for lvl, (h, w) in enumerate(shapes):
        # Flatten and reshape features: (BS, Len, Heads, Dim) -> (BS*Heads, Dim, H, W)
        lvl_feat = (
            feat_splits[lvl]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_sz * n_heads, d_embed, h, w)
        )
        
        # Prepare grid: (BS*Heads, Queries, Points, 2)
        lvl_grid = grid_coords[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        
        # Bilinear sampling
        sampled = F.grid_sample(
            lvl_feat, lvl_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampled_list.append(sampled)

    # Reshape weights: (BS, Heads, 1, Queries, Levels*Points)
    attn_weights = attn_weights.transpose(1, 2).reshape(
        batch_sz * n_heads, 1, n_queries, n_levels * n_points
    )
    
    # Weighted aggregation
    stacked = torch.stack(sampled_list, dim=-2).flatten(-2)
    output = (stacked * attn_weights).sum(-1).view(batch_sz, n_heads * d_embed, n_queries)
    
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module.
    
    

    Adapts standard attention to attend to sparse sampling points across
    multiple feature resolutions.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        batch_first: bool = False,
    ):
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by heads ({num_heads})")
        
        d_head = embed_dim // num_heads
        self.batch_first = batch_first

        if not _check_power_of_2(d_head):
            warnings.warn("Efficiency Warning: Head dimension is not a power of 2.")

        self.im2col_step = img2col_step
        self.d_model = embed_dim
        self.n_heads = num_heads
        self.n_levels = num_levels
        self.n_points = num_points

        # Offset and Weight Projections
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_layer_weights()

    def _reset_parameters(self):
        return self._init_layer_weights()

    def _init_layer_weights(self):
        """Initializes weights with grid priors for offsets."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        
        # Create initial reference grid
        angles = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        base_grid = torch.stack([angles.cos(), angles.sin()], -1)
        
        # Normalize and reshape
        base_grid = (
            (base_grid / base_grid.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        
        # Scale by point index
        for i in range(self.n_points):
            base_grid[:, :, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(base_grid.view(-1))
            
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def freeze_sampling_offsets(self):
        print("Locking sampling offset gradients.")
        self.sampling_offsets.weight.requires_grad = False
        self.sampling_offsets.bias.requires_grad = False

    def freeze_attention_weights(self):
        print("Locking attention weight gradients.")
        self.attention_weights.weight.requires_grad = False
        self.attention_weights.bias.requires_grad = False

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Scale Deformable Attention.
        """
        if value is None:
            value = query

        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        batch_sz, n_queries, _ = query.shape
        _, n_values, _ = value.shape

        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != n_values:
            raise ValueError("Spatial shapes do not match total value length.")

        # Project values
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        
        value = value.view(batch_sz, n_values, self.n_heads, -1)
        
        # Predict offsets and weights
        sampling_offsets = self.sampling_offsets(query).view(
            batch_sz, n_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attn_weights = self.attention_weights(query).view(
            batch_sz, n_queries, self.n_heads, self.n_levels * self.n_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1).view(
            batch_sz, n_queries, self.n_heads, self.n_levels, self.n_points
        )

        # Apply offsets to reference points
        if reference_points.shape[-1] == 2:
            offset_scale = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sample_locs = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_scale[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sample_locs = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(f"Reference points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        # Compute attention
        output = ms_deform_attn_core_pytorch(
            value, spatial_shapes, sample_locs, attn_weights
        )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


def generate_missing_class_stub(cls_name, dep_name, extra_msg=""):
    """Generates a dummy class that raises ImportError."""
    err_msg = f"Missing dependency '{dep_name}': Class '{cls_name}' unavailable. {extra_msg}"

    class _StubMeta(type):
        def __getattr__(cls, _):
            raise ImportError(err_msg)

    class _Stub(object, metaclass=_StubMeta):
        def __init__(self, *args, **kwargs):
            raise ImportError(err_msg)

    return _Stub


def generate_missing_func_stub(func_name, dep_name, extra_msg=""):
    """Generates a dummy function that raises ImportError."""
    err_msg = f"Missing dependency '{dep_name}': Function '{func_name}' unavailable. {extra_msg}"

    if isinstance(dep_name, (list, tuple)):
        dep_name = ",".join(dep_name)

    def _stub(*args, **kwargs):
        raise ImportError(err_msg)

    return _stub
