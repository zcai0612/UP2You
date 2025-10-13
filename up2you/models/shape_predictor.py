import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any
from einops import rearrange
from up2you.models.layers.block import SelfAttnBlock, CrossAttnBlock
from up2you.models.heads.shape_head import ShapeHead

class ShapePredictor(nn.Module):
    def __init__(
        self,
        num_queries=1,
        embed_dim=1024,
        depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=['cross', 'cross', 'cross'],
        qk_norm=True,
        init_values=0.01,
        # shape head
        trunk_depth=2,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            self.get_attn_block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=None,
                aa_type=aa_type,
            )
            for _ in range(depth) for aa_type in aa_order
            
        ])

        self.depth = depth
        self.aa_order = aa_order

        self.shape_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        nn.init.normal_(self.shape_tokens, std=1e-6)

        self.shape_head = ShapeHead(
            dim_in=embed_dim,
            trunk_depth=trunk_depth,
            target_dim=10,
        )

        self.use_reentrant = False

    def _process_attention(self, x, context, q_pos, k_pos):
        tokens = x
        for block in self.blocks:
            if isinstance(block, SelfAttnBlock):
                if self.training:
                    tokens = checkpoint(block, tokens, q_pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = block(tokens, q_pos)
            elif isinstance(block, CrossAttnBlock):
                if self.training:
                    tokens = checkpoint(block, tokens, context, q_pos, k_pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = block(tokens, context, q_pos, k_pos)
        return tokens

    def get_attn_block(
        self, 
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias,
        proj_bias,
        ffn_bias,
        init_values,
        qk_norm,
        rope,
        aa_type,
    ):
        if aa_type == 'self':
            attn_block = SelfAttnBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                norm_layer=nn.LayerNorm,
                qk_norm=qk_norm,
                rope=rope,
            )
        elif aa_type == 'cross':
            attn_block = CrossAttnBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                norm_layer=nn.LayerNorm,
                qk_norm=qk_norm,
                rope=rope,
            )
        return attn_block

    def forward(
        self, ref_img_feats, # [B, Nr, H, W, C]
    ):
        B, Nr, H, W, C = ref_img_feats.shape

        ref_img_feats = rearrange(ref_img_feats, "B Nr H W C -> B (Nr H W) C")
        shape_tokens = self.shape_tokens.repeat(B, 1, 1)

        shape_tokens = self._process_attention(shape_tokens, ref_img_feats, None, None)
        shape_params = self.shape_head(shape_tokens)

        return shape_params