import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Optional, Tuple, Union
from einops import rearrange

from up2you.models.layers.block import CrossAttnBlock
from up2you.models.layers.patch_embed import PatchEmbed
from up2you.models.layers.rope import RotaryPositionEmbedding2D, PositionGetter


class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        patch_size=14,
        depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=['self', 'self', 'cross', 'cross', 'cross'],
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        mask_ratio=0.95,
    ):
        super().__init__()
        
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_ratio = mask_ratio
        
        # 计算patch网格大小
        self.patch_grid_size = (self.img_size[0] // self.patch_size[0], 
                               self.img_size[1] // self.patch_size[1])
        self.num_patches = self.patch_grid_size[0] * self.patch_grid_size[1]
        
        # Patch embedding层
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        
        # Mask token - 用于被mask的patch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # 编码器到解码器的投影
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # 特征投影层 - 将输入feats投影到decoder维度
        self.feat_proj = nn.Linear(embed_dim, decoder_embed_dim, bias=True)  # 假设feats与embed_dim相同
        
        # 位置编码
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False
        )
        
        # RoPE位置编码
        self.rope = RoPE(
            freq=rope_freq,
            dim=decoder_embed_dim // num_heads,
        )
        
        # Cross attention blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            CrossAttnBlock(
                dim=decoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                init_values=init_values,
                qk_norm=qk_norm,
                rope=self.rope,
            )
            for i in range(depth)
        ])
        
        # 最终的norm层
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # 预测头 - 将decoder输出投影回像素空间
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, 
            patch_size[0] * patch_size[1] * in_chans if isinstance(patch_size, tuple) 
            else patch_size * patch_size * in_chans, 
            bias=True
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化权重"""
        # 初始化位置编码
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            self.patch_grid_size,
            n_cls_token=0
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # 初始化mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # 初始化线性层
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        """
        将图像转换为patches
        imgs: [B, C, H, W]
        return: [B, N, patch_size**2 * C]
        """
        p = self.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], imgs.shape[1], h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * imgs.shape[1])
        return x
    
    def unpatchify(self, x):
        """
        将patches转换回图像
        x: [B, N, patch_size**2 * C]
        return: [B, C, H, W]
        """
        p = self.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, -1)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], -1, h * p, h * p)
        return imgs
    
    def create_foreground_mask(self, img_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在前景区域内创建随机mask
        
        Args:
            img_mask: [B, 1, H, W] 前景mask
            
        Returns:
            patch_mask: [B, N] patch级别的mask，1表示保留，0表示mask
            foreground_patches: [B, N] 前景patch的标识
        """
        B, _, H, W = img_mask.shape
        
        # 将img_mask转换为patch级别
        # 首先resize到patch grid大小
        patch_mask = F.interpolate(
            img_mask.float(), 
            size=self.patch_grid_size, 
            mode='bilinear', 
            align_corners=False
        )  # [B, 1, H_patch, W_patch]
        
        patch_mask = patch_mask.squeeze(1).flatten(1)  # [B, N]
        
        # 确定前景patches（mask值 > 0.5的patch被认为是前景）
        foreground_patches = (patch_mask > 0.5).float()  # [B, N]
        
        # 在前景区域内进行随机mask
        final_mask = torch.ones_like(foreground_patches)  # [B, N]
        
        for b in range(B):
            fg_indices = torch.where(foreground_patches[b] > 0.5)[0]
            if len(fg_indices) > 0:
                # 在前景patches中随机选择需要mask的patches
                num_mask = int(len(fg_indices) * self.mask_ratio)
                mask_indices = torch.randperm(len(fg_indices))[:num_mask]
                selected_indices = fg_indices[mask_indices]
                final_mask[b, selected_indices] = 0
        
        return final_mask, foreground_patches
    
    def forward(
        self, 
        img: torch.Tensor, 
        img_mask: torch.Tensor, 
        feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            img: [B, C, H, W] 输入图像
            img_mask: [B, 1, H, W] 前景mask
            feats: [B, N_feat, C_feat] 用于重建的特征
            
        Returns:
            pred: [B, C, H, W] 重建的图像
            patch_mask: [B, N] patch级别的mask
            loss_mask: [B, N] 用于计算loss的mask（只在被mask的前景区域计算loss）
        """
        B, C, H, W = img.shape
        
        # 1. Patch embedding
        x = self.patch_embed(img)  # [B, N, embed_dim]
        
        # 2. 创建前景mask
        patch_mask, foreground_patches = self.create_foreground_mask(img_mask)  # [B, N]
        
        # 3. 投影到decoder维度
        x = self.decoder_embed(x)  # [B, N, decoder_embed_dim]
        
        # 4. 应用mask - 将被mask的patch替换为mask token
        mask_tokens = self.mask_token.repeat(B, self.num_patches, 1)  # [B, N, decoder_embed_dim]
        w = patch_mask.unsqueeze(-1).type_as(x)  # [B, N, 1]
        x = x * w + mask_tokens * (1 - w)  # [B, N, decoder_embed_dim]
        
        # 5. 添加位置编码
        x = x + self.decoder_pos_embed  # [B, N, decoder_embed_dim]
        
        # 6. 处理context特征
        context = self.feat_proj(feats)  # [B, N_feat, decoder_embed_dim]
        
        # 7. 通过cross attention blocks
        # 生成位置编码
        q_pos = self.get_position_encoding(B, self.patch_grid_size, x.device)  # [B, N, pos_dim]
        k_pos = None  # context没有特定的2D位置编码
        
        for blk in self.blocks:
            x = blk(x, context, q_pos=q_pos, k_pos=k_pos)  # [B, N, decoder_embed_dim]
        
        # 8. 最终normalization
        x = self.decoder_norm(x)  # [B, N, decoder_embed_dim]
        
        # 9. 预测像素值
        x = self.decoder_pred(x)  # [B, N, patch_size**2 * C]
        
        # 10. 转换回图像格式
        pred = self.unpatchify(x)  # [B, C, H, W]
        
        # 11. 计算loss mask - 只在被mask的前景区域计算loss
        loss_mask = foreground_patches * (1 - patch_mask)  # [B, N]
        
        return pred, patch_mask, loss_mask
    
    def get_position_encoding(self, batch_size: int, grid_size: Tuple[int, int], device: torch.device):
        """生成2D位置编码用于RoPE"""
        H, W = grid_size
        
        # 创建2D网格
        y_pos = torch.arange(H, dtype=torch.float32, device=device)
        x_pos = torch.arange(W, dtype=torch.float32, device=device)
        
        y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing='ij')
        
        # 展平并组合
        pos = torch.stack([y_pos.flatten(), x_pos.flatten()], dim=-1)  # [H*W, 2]
        pos = pos.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, H*W, 2]
        
        return pos