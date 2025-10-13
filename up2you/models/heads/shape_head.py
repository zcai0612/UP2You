import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from up2you.models.layers import Mlp
from up2you.models.layers.block import SelfAttnBlock
from up2you.models.heads.head_act import base_shape_act


class ShapeHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 1536,
        trunk_depth: int = 2,
        target_dim: int = 10, # Number of SMPLX Shape Parameters
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        act_type: str = "linear",
    ):
        super().__init__()

        self.target_dim = target_dim
        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                SelfAttnBlock(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )
        self.act_type = act_type

        # Normalizations for shape token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        self.empty_shape_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_shape = nn.Linear(self.target_dim, dim_in)

        # Modulation for shape token.
        self.shapeLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.shape_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)
        
    
    def forward(self, shape_tokens: torch.Tensor, num_iterations: int = 4) -> list:
        """
        Forward pass to predict smplx shape.

        Args:
            shape_tokens (torch.Tensor): shape tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: A list of predicted shape encodings (post-activation) from each iteration.
        """
        shape_tokens = self.token_norm(shape_tokens)
        shape_token_enc_list = self.trunk_fn(shape_tokens, num_iterations)

        return shape_token_enc_list

    def trunk_fn(self, shape_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine smplx shape predictions.

        Args:
            shape_tokens (torch.Tensor): Normalized shape tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated shape encodings from each iteration.
        """
        B, S, C = shape_tokens.shape  # S is expected to be 1.
        pred_shape_enc = None
        pred_shape_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty shape for the first iteration.
            if pred_shape_enc is None:
                module_input = self.embed_shape(self.empty_shape_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_shape_enc = pred_shape_enc.detach()
                module_input = self.embed_shape(pred_shape_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.shapeLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            shape_tokens_modulated = gate_msa * modulate(self.adaln_norm(shape_tokens), shift_msa, scale_msa)
            shape_tokens_modulated = shape_tokens_modulated + shape_tokens

            shape_tokens_modulated = self.trunk(shape_tokens_modulated)
            # Compute the delta update for the shape encoding.
            pred_shape_enc_delta = self.shape_branch(self.trunk_norm(shape_tokens_modulated))

            if pred_shape_enc is None:
                pred_shape_enc = pred_shape_enc_delta
            else:
                pred_shape_enc = pred_shape_enc + pred_shape_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_shape = base_shape_act(
                pred_shape_enc, act_type=self.act_type
            )
            pred_shape_enc_list.append(activated_shape)

        pred_shape_enc = pred_shape_enc_list[-1]

        return pred_shape_enc
    

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift