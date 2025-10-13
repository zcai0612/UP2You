# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from up2you.models.layers.block import SelfAttnBlock, CrossAttnBlock
from up2you.models.layers.attention import OutAttention
from up2you.models.layers.patch_embed import PatchEmbed
from up2you.models.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from up2you.models.pose_encoder import PoseEncoder
from einops import rearrange

class FeatureAggregator(nn.Module):
    def __init__(
        self,
        pose_img_size=518, # follow DINOv2 Large
        pose_img_in_chans=3,
        pose_patch_embed_type="patch_embed", # "patch_embed" or "pose_encoder"
        embed_dim=1024, # follow DINOv2 Large
        patch_size=14, # follow DINOv2 Large
        depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=['self', 'self', 'self', 'cross'],
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        weight_norm="none",
        use_mask=False,
        spatial_smooth=False,
        smooth_method="avgpool",
        kernel_size=5,
    ):
        super().__init__()

        self.resize = Resize((pose_img_size, pose_img_size))

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        if pose_patch_embed_type == "patch_embed":
            self.pose_patch_embed = PatchEmbed(
                img_size=pose_img_size,
                patch_size=patch_size,
                in_chans=pose_img_in_chans,
                embed_dim=embed_dim,
            )
        elif pose_patch_embed_type == "pose_encoder":
            self.pose_patch_embed = PoseEncoder(
                in_channels=pose_img_in_chans,
                downsample_ratio=patch_size,
                out_channels=embed_dim,
                image_size=pose_img_size,
            )
        else:
            raise ValueError(f"Invalid pose_patch_embed_type: {pose_patch_embed_type}")
        
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

        self.out_attn = OutAttention(
            dim=embed_dim, 
            num_heads=16,
            qk_bias=qkv_bias, 
            attn_drop=0.0, 
            qk_norm=False, 
            rope=self.rope,
            weight_norm=weight_norm,
            use_mask=use_mask,
            spatial_smooth=spatial_smooth,
            smooth_method=smooth_method,
            kernel_size=kernel_size,
        )

        self.pose_img_size = pose_img_size
        self.pose_patch_size = patch_size
        self.embed_dim = embed_dim

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

    '''
    Input:
        target_pose_imgs
        ref_img_feats: [B, Nr, H, W, C] from DINOv2
    '''

    def forward(
        self, 
        target_pose_imgs, 
        ref_img_feats, 
        ref_alphas=None,
    ) -> List[torch.Tensor]:

        B, Nv, H_v, W_v, C_v = target_pose_imgs.shape
        B, Nr, H_r, W_r, C_r = ref_img_feats.shape

        assert C_r == self.embed_dim, "ref_img_feats.shape[-1] should be equal to embed_dim"

        target_pose_imgs = rearrange(target_pose_imgs, "B Nv H_v W_v C -> (B Nv) C H_v W_v")
        if not H_v == W_v == self.pose_img_size:
            target_pose_imgs = self.resize(target_pose_imgs)
        target_pose_tokens = self.pose_patch_embed(target_pose_imgs)

        ref_img_feats = rearrange(ref_img_feats, "B Nr H_r W_r C -> B (Nr H_r W_r) C")
        target_pose_tokens = rearrange(target_pose_tokens, "(B Nv) S C -> B Nv S C", B=B, Nv=Nv)
        
        q_pos = None
        k_pos = None

        if self.rope is not None:
            q_pos = self.position_getter(B*Nv, H_v//self.pose_patch_size, W_v//self.pose_patch_size, device=target_pose_imgs.device)
            k_pos = self.position_getter(B*Nr, H_r, W_r, device=ref_img_feats.device)

        q_pos = rearrange(q_pos, "(B Nv) S C -> B Nv S C", B=B, Nv=Nv)
        k_pos = rearrange(k_pos, "(B Nr) (H W) C -> B (Nr H W) C", B=B, Nr=Nr, H=H_r, W=W_r)

        weight_maps = []
        for view_idx in range(Nv):
            view_target_pose_tokens = target_pose_tokens[:, view_idx, :, :]
            q_pos_view = q_pos[:, view_idx, :, :]
            view_target_pose_tokens = self._process_attention(view_target_pose_tokens, ref_img_feats, q_pos_view, k_pos)
            
            view_weight_map = self.out_attn(view_target_pose_tokens, ref_img_feats, num_refs=Nr, ref_alphas=ref_alphas)
            view_weight_map = rearrange(view_weight_map, "B (Nr H_r W_r) -> B Nr H_r W_r", B=B, Nr=Nr, H_r=H_r, W_r=W_r)
            weight_maps.append(view_weight_map.unsqueeze(-1))

        return weight_maps

if __name__ == "__main__":
    feature_aggregator = FeatureAggregator()
    feature_aggregator.eval()
    target_pose_imgs = torch.randn(1, 6, 518, 518, 3)
    ref_img_feats = torch.randn(1, 7, 37, 37, 1024)
    weight_maps = feature_aggregator(target_pose_imgs, ref_img_feats)
    for i in range(len(weight_maps)):
        print(weight_maps[i].shape)