# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange

XFORMERS_AVAILABLE = False


class OutAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_bias: bool = True,
        attn_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
        weight_norm: str = "softmax",  # support "softmax", "sigmoid", "none", "sparsemax", "entmax", "topk_softmax"
        # parameters for spatial smoothness
        spatial_smooth: bool = True,  # whether to enable spatial smoothness
        smooth_method: str = "avgpool",  # smoothness method: "gaussian", "avgpool", "maxpool"
        kernel_size: int = 5,  # kernel size
        gaussian_sigma: float = 1.0,  # gaussian standard deviation
        use_mask: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.weight_norm = weight_norm
        self.use_mask = use_mask
        
        # parameters for spatial smoothness
        self.spatial_smooth = spatial_smooth
        self.smooth_method = smooth_method
        self.kernel_size = kernel_size
        self.gaussian_sigma = gaussian_sigma
        
        # Multi-head projections
        self.q_proj = nn.Linear(dim, dim, bias=qk_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qk_bias)
        
        # Normalization layers
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.rope = rope
        
        # precompute gaussian kernel (only used when gaussian smoothness is enabled)
        if self.spatial_smooth and self.smooth_method == "gaussian":
            self._gaussian_kernel = self._create_gaussian_kernel(
                self.kernel_size, self.gaussian_sigma
            )
        else:
            self._gaussian_kernel = None

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """create gaussian kernel"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # create 2D kernel
        kernel = g[:, None] * g[None, :]
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

    def _apply_gaussian_smooth(self, weight_map, num_refs):
        """apply gaussian smoothness to enhance spatial continuity"""
        # calculate spatial dimensions
        B, Nr, HW = weight_map.shape
        H = W = int(HW ** 0.5)
        
        if H * W != HW:
            # if not a perfect square, skip spatial smoothness
            return weight_map
        
        # Reshape to spatial dimensions
        weight_spatial = weight_map.view(B, Nr, H, W)  # [B, Nr, H, W]
        
        # Apply Gaussian filtering to each reference separately
        smoothed_weights = []
        kernel = self._gaussian_kernel.to(weight_map.device)
        padding = self.kernel_size // 2
        
        for nr in range(Nr):
            # Extract single reference weights [B, 1, H, W]
            single_ref = weight_spatial[:, nr:nr+1, :, :]
            
            # Apply Gaussian convolution with padding
            smoothed = F.conv2d(single_ref, kernel, padding=padding)
            smoothed_weights.append(smoothed)
        
        # Concatenate back [B, Nr, H, W]
        smoothed_spatial = torch.cat(smoothed_weights, dim=1)
        
        # Reshape back to flat format
        return smoothed_spatial.view(B, Nr, HW)

    def _apply_avgpool_smooth(self, weight_map, num_refs):
        """apply average pooling smoothness to enhance spatial continuity"""
        # calculate spatial dimensions
        B, Nr, HW = weight_map.shape
        H = W = int(HW ** 0.5)
        
        if H * W != HW:
            # if not a perfect square, skip spatial smoothness
            return weight_map
        
        # Reshape to spatial dimensions
        weight_spatial = weight_map.view(B, Nr, H, W)  # [B, Nr, H, W]
        
        # Apply average pooling with same output size
        kernel_size = self.kernel_size
        padding = kernel_size // 2
        
        # use average pooling for smoothness, keep original size
        smoothed_spatial = F.avg_pool2d(
            weight_spatial, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        
        # Reshape back to flat format
        return smoothed_spatial.view(B, Nr, HW)
    
    def _apply_maxpool_smoothing(self, weight_map, num_refs):
        """apply max pooling smoothness to enhance spatial continuity"""
        # calculate spatial dimensions
        B, Nr, HW = weight_map.shape
        H = W = int(HW ** 0.5)
        
        if H * W != HW:
            # if not a perfect square, skip spatial smoothness
            return weight_map
        
        # Reshape to spatial dimensions
        weight_spatial = weight_map.view(B, Nr, H, W)  # [B, Nr, H, W]
        
        # Apply max pooling with same output size
        kernel_size = self.kernel_size
        padding = kernel_size // 2
        
        # use max pooling for smoothness, keep original size
        smoothed_spatial = F.max_pool2d(
            weight_spatial, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        
        # Reshape back to flat format
        return smoothed_spatial.view(B, Nr, HW)

    def _apply_spatial_smoothing(self, weight_map, num_refs):
        """apply spatial smoothness"""
        if not self.spatial_smooth:
            return weight_map
        
        if self.smooth_method == "gaussian":
            return self._apply_gaussian_smooth(weight_map, num_refs)
        elif self.smooth_method == "avgpool":
            return self._apply_avgpool_smooth(weight_map, num_refs)
        else:
            return weight_map

    def forward(self, x: Tensor, context: Tensor, num_refs: int, q_pos=None, k_pos=None, ref_alphas=None) -> Tensor:
        """
        Args:
            x: target pose features [B, N_q, C] - query from target pose
            context: ref image features [B, N_k, C] - key from n ref images, N_k = Nr * Hr * Wr
            num_refs: number of ref images, Nr
            q_pos: query position encoding
            k_pos: key position encoding
            ref_alphas: ref image alphas [B, Nr, H, W]
        Returns:
            weight_map: [B, Nr*Hr*Wr] - weight map for each ref image pixel
        """
        B, N_q, C = x.shape
        B_ctx, N_k, C_ctx = context.shape
        assert B == B_ctx, "x and context must have the same batch size"
        assert C == C_ctx, "x and context must have the same feature dimension"
        assert N_k % num_refs == 0, "context length must be divisible by num_refs"
        
        # project to multi-head space
        q = self.q_proj(x).reshape(B, N_q, self.num_heads, self.head_dim)  # [B, N_q, H, D]
        k = self.k_proj(context).reshape(B, N_k, self.num_heads, self.head_dim)  # [B, N_k, H, D]
        
        # reorder head dimension
        q = q.transpose(1, 2)  # [B, H, N_q, D]
        k = k.transpose(1, 2)  # [B, H, N_k, D]
        
        # apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # apply rotary position encoding (if provided)
        if self.rope is not None:
            if q_pos is not None:
                q = self.rope(q, q_pos)
            if k_pos is not None:
                k = self.rope(k, k_pos)
        
        # compute attention scores (keep original scores, no softmax)
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, H, N_q, N_k]
        # no softmax here, keep original attention scores for weight map calculation
        attn = self.attn_drop(attn)
        
        # average aggregation across heads
        attn = attn.mean(dim=1)  # [B, N_q, N_k]
        
        # compute average importance for each ref image pixel
        weight_map = attn.mean(dim=1)  # [B, N_k]
        
        # reshape to ref dimension for normalization
        weight_map = rearrange(weight_map, 'B (Nr HW) -> B Nr HW', Nr=num_refs)
        
        # apply spatial smoothness to enhance continuity (before normalization)
        weight_map = self._apply_spatial_smoothing(weight_map, num_refs)
        
        # apply weight normalization strategy
        if self.weight_norm == "softmax":
            # softmax on ref dimension, force competitive selection
            weight_map = F.softmax(weight_map, dim=1)
        elif self.weight_norm == "sigmoid":
            # independent activation, each ref can be independently important
            # use temperature parameter to make sigmoid steeper, temperature smaller -> steeper
            weight_map = torch.sigmoid(weight_map)
        elif self.weight_norm == "sparsemax":
            # sparsemax: sparse probability distribution, allows exact zeros
            weight_map = self._sparsemax(weight_map, dim=1)
        elif self.weight_norm == "entmax":
            # entmax: interpolation between softmax and sparsemax
            weight_map = self._entmax(weight_map, dim=1, alpha=1.5)
        elif self.weight_norm == "topk_softmax":
            # top-k softmax: only top-k refs get non-zero weights
            k = min(3, num_refs)  # select top-3 or all if less than 3
            weight_map = self._topk_softmax(weight_map, dim=1, k=k)
        elif self.weight_norm == "adaptive_softmax":
            # adaptive softmax: temperature based on variance
            weight_map = self._adaptive_softmax(weight_map, dim=1)
        elif self.weight_norm == "none":
            # no normalization, keep original attention weights
            weight_map = F.relu(weight_map)  # only ensure non-negative
        else:
            raise ValueError(f"Unknown weight_norm: {self.weight_norm}")
        
        if ref_alphas is not None and self.use_mask:
            ref_alphas = ref_alphas.unsqueeze(-1)
            ref_alphas = rearrange(ref_alphas, "B Nr H W 1 -> (B Nr) 1 H W")
            HW = weight_map.shape[2]
            H_weight = W_weight = int(HW ** 0.5)
            ref_alphas = F.interpolate(ref_alphas, size=(H_weight, W_weight), mode="nearest")
            ref_alphas = rearrange(ref_alphas, "(B Nr) 1 H_weight W_weight -> B Nr (H_weight W_weight)", Nr=num_refs)
            weight_map = weight_map * ref_alphas
        
        # reshape back to original format
        weight_map = rearrange(weight_map, 'B Nr HW -> B (Nr HW)')
        
        return weight_map
    
    def _sparsemax(self, logits, dim=-1):
        """Sparsemax activation: sparse probability distribution"""
        # Sort logits in descending order
        sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
        
        # Compute cumulative sum
        cumsum = torch.cumsum(sorted_logits, dim=dim)
        
        # Find the threshold
        k = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype)
        if dim != -1:
            k = k.view([1] * dim + [-1] + [1] * (logits.dim() - dim - 1))
        
        threshold = (cumsum - 1) / k
        mask = sorted_logits > threshold
        
        # Find the last True position for each sample
        k_star = mask.sum(dim=dim, keepdim=True)
        threshold = threshold.gather(dim, k_star - 1)
        
        # Apply sparsemax
        return F.relu(logits - threshold)
    
    def _entmax(self, logits, dim=-1, alpha=1.5):
        """Entmax activation: interpolation between softmax (alpha=1) and sparsemax (alpha=2)"""
        if alpha == 1.0:
            return F.softmax(logits, dim=dim)
        elif alpha == 2.0:
            return self._sparsemax(logits, dim=dim)
        else:
            # Simplified entmax implementation
            # For alpha=1.5, this gives a good balance
            scaled_logits = logits * (2 - alpha)
            return self._sparsemax(scaled_logits, dim=dim)
    
    def _topk_softmax(self, logits, dim=-1, k=3):
        """Top-k softmax: only top-k elements get non-zero probabilities"""
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(logits, k=k, dim=dim)
        
        # Create output tensor filled with zeros
        output = torch.zeros_like(logits)
        
        # Apply softmax only to top-k values
        topk_probs = F.softmax(topk_values, dim=dim)
        
        # Scatter the probabilities back
        output.scatter_(dim, topk_indices, topk_probs)
        
        return output
    
    def _adaptive_softmax(self, logits, dim=-1):
        """Adaptive softmax: temperature based on variance"""
        # Compute variance along the specified dimension
        variance = torch.var(logits, dim=dim, keepdim=True)
        
        # Adaptive temperature: higher variance -> lower temperature (sharper)
        # lower variance -> higher temperature (smoother)
        temperature = torch.clamp(1.0 / (variance + 1e-6), min=0.5, max=2.0)
        
        # Apply adaptive softmax
        return F.softmax(logits / temperature, dim=dim)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class MemEffSelfAttention(SelfAttention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    Cross Attention layer, query from x, key and value from context
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, context: Tensor, q_pos=None, k_pos=None) -> Tensor:
        """
        Args:
            x: query [B, N_q, C]
            context: key and value [B, N_k, C]
            q_pos: query position encoding
            k_pos: key position encoding
        Returns:
            output [B, N_q, C]
        """
        B, N_q, C = x.shape
        B_ctx, N_k, C_ctx = context.shape
        
        assert B == B_ctx, "x and context must have the same batch size"
        assert C == C_ctx, "x and context must have the same feature dimension"
        
        # compute query
        q = self.q_proj(x).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # compute key and value
        kv = self.kv_proj(context).reshape(B, N_k, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        q, k = self.q_norm(q), self.k_norm(k)

        # apply rotary position encoding
        if self.rope is not None:
            if q_pos is not None:
                q = self.rope(q, q_pos)
            if k_pos is not None:
                k = self.rope(k, k_pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, num_heads, N_q, N_k]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, num_heads, N_q, head_dim]

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

