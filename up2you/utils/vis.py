import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Optional, Union
from einops import rearrange


def weight_map_to_heatmap(
    weight_maps: List[torch.Tensor],
    ref_img_masks: torch.Tensor=None,
    colormap: str = "jet",
    normalize: bool = True,
    temperature: float = 1.0,
    res: int = 768, 
    return_tensor: bool = True,
    eps: float = 1e-8
) -> List[torch.Tensor]:
    """
    将weight map列表转换为热力图可视化。
    
    Args:
        weight_maps: List of weight maps, 每个元素形状为 [B, Nr, H, W, 1]
        ref_img_masks: 元素形状为 [B, Nr, H, W]
        colormap: matplotlib colormap名称，如 "jet", "viridis", "hot"等
        normalize: 是否进行归一化，对所有pose的Nr个weight map同时归一化
        temperature: 温度参数，用于调节热力图的对比度
        return_tensor: 是否返回tensor格式，否则返回numpy格式
        eps: 小的数值，避免除零错误
        
    Returns:
        List of heatmaps, 每个元素形状为 [B, Nr, H, W, 3] (RGB热力图)
    """
    # 初始化结果列表
    heatmaps = []
    
    # 获取目标尺寸
    if ref_img_masks is not None:
        target_h, target_w = ref_img_masks.shape[-2:]
    else:
        target_h, target_w = res, res
    
    # 遍历每个pose的weight map
    for pose_idx, weight_map in enumerate(weight_maps):
        B, Nr, H, W, _ = weight_map.shape
        
        if ref_img_masks is not None:
            weight_map = F.interpolate(weight_map.squeeze(-1), size=(target_h, target_w), mode="bilinear", align_corners=False)
        else:
            weight_map = F.interpolate(weight_map.squeeze(-1), size=(res, res), mode="bilinear", align_corners=False)

        # 将weight map展平为 [B, Nr*H*W]
        weight_flat = rearrange(weight_map, 'b nr h w -> b (nr h w)')
        
        if normalize and ref_img_masks is not None:
            # 只对前景部分进行归一化
            mask_flat = rearrange(ref_img_masks, 'b nr h w -> b (nr h w)')  # [B, Nr*H*W]
            # 只对前景部分进行Softmax归一化
            weight_flat = torch.where(mask_flat > 0, 
                                    F.softmax(weight_flat, dim=-1),
                                    weight_flat)
            # 使用mask获取前景部分的weight
            masked_weight = weight_flat * mask_flat
            
            # 只在mask为1的位置计算min和max
            weight_min = torch.where(mask_flat > 0, masked_weight, torch.tensor(float('inf')).to(weight_flat.device)).min(dim=-1, keepdim=True)[0]
            weight_max = torch.where(mask_flat > 0, masked_weight, torch.tensor(float('-inf')).to(weight_flat.device)).max(dim=-1, keepdim=True)[0]

            weight_flat = torch.where(mask_flat > 0, 
                                    (weight_flat - weight_min) / (weight_max - weight_min + eps),
                                    weight_flat)
        elif normalize:
            weight_flat = F.softmax(weight_flat, dim=-1)
            weight_min = weight_flat.min(dim=-1, keepdim=True)[0]
            weight_max = weight_flat.max(dim=-1, keepdim=True)[0]
            weight_flat = (weight_flat - weight_min) / (weight_max - weight_min + eps)

        # 应用temperature
        weight_flat = weight_flat ** (1.0 / temperature)
        
        # 重新整形为 [B, Nr, H, W]
        weight_norm = rearrange(weight_flat, 'b (nr h w) -> b nr h w', nr=Nr, h=target_h, w=target_w)
        
        # # 如果需要插值到目标尺寸
        # if ref_img_masks is not None:
        #     weight_norm = F.interpolate(
        #         weight_norm,
        #         size=(target_h, target_w),
        #         mode='bilinear',
        #         align_corners=False
        #     )
        
        # 创建热力图
        cmap = plt.get_cmap(colormap)
        weight_np = weight_norm.detach().cpu().numpy()
        heatmap = torch.from_numpy(cmap(weight_np)[..., :3]).to(weight_map.device)
        
        # 如果有mask，只保留前景部分
        if ref_img_masks is not None:
            mask = ref_img_masks.unsqueeze(-1)  # [B, Nr, H, W, 1]
            heatmap = heatmap * mask
        
        if not return_tensor:
            heatmap = heatmap.detach().cpu().numpy()
        
        heatmaps.append(heatmap)
    
    return heatmaps
    