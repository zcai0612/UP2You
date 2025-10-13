import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Optional, Union
from einops import rearrange
import matplotlib.pyplot as plt

def weight_map_to_heatmap(
    weight_maps: List[torch.Tensor],
    ref_img_masks: torch.Tensor=None,
    colormap: str = "jet",
    normalize: bool = True,
    temperature: float = 1.0,
    return_tensor: bool = True,
    eps: float = 1e-8
) -> List[torch.Tensor]:
    """
    Convert weight map list to heatmap visualization.
    
    Args:
        weight_maps: List of weight maps, each element shape is [B, Nr, H, W, 1]
        ref_img_masks: each element shape is [B, Nr, H, W]
        colormap: matplotlib colormap name, such as "jet", "viridis", "hot"
        normalize: whether to normalize, normalize all Nr weight maps of all poses simultaneously
        temperature: temperature parameter, used to adjust the contrast of the heatmap
        return_tensor: whether to return tensor format, otherwise return numpy format
        eps: small number to avoid division by zero
        
    Returns:
        List of heatmaps, each element shape is [B, Nr, H, W, 3] (RGB heatmap)
    """
    # initialize result list
    heatmaps = []
    
    # get target size
    if ref_img_masks is not None:
        target_h, target_w = ref_img_masks.shape[-2:]
    
    # iterate over each pose's weight map
    for pose_idx, weight_map in enumerate(weight_maps):
        B, Nr, H, W, _ = weight_map.shape
        
        if ref_img_masks is not None:
            weight_map = F.interpolate(weight_map.squeeze(-1), size=(target_h, target_w), mode="bilinear", align_corners=False)

        # flatten weight map to [B, Nr*H*W]
        weight_flat = rearrange(weight_map, 'b nr h w -> b (nr h w)')
        
        if normalize and ref_img_masks is not None:
            # normalize only the foreground part
            mask_flat = rearrange(ref_img_masks, 'b nr h w -> b (nr h w)')  # [B, Nr*H*W]
            # normalize only the foreground part with Softmax
            weight_flat = torch.where(mask_flat > 0, 
                                    F.softmax(weight_flat, dim=-1),
                                    weight_flat)
            # use mask to get the weight of the foreground part
            masked_weight = weight_flat * mask_flat
            
            # calculate min and max only at the positions where mask is 1
            weight_min = torch.where(mask_flat > 0, masked_weight, torch.tensor(float('inf')).to(weight_flat.device)).min(dim=-1, keepdim=True)[0]
            weight_max = torch.where(mask_flat > 0, masked_weight, torch.tensor(float('-inf')).to(weight_flat.device)).max(dim=-1, keepdim=True)[0]

            weight_flat = torch.where(mask_flat > 0, 
                                    (weight_flat - weight_min) / (weight_max - weight_min + eps),
                                    weight_flat)
        # apply temperature
        weight_flat = weight_flat ** (1.0 / temperature)
        
        # reshape to [B, Nr, H, W]
        weight_norm = rearrange(weight_flat, 'b (nr h w) -> b nr h w', nr=Nr, h=target_h, w=target_w)
        
        
        # create heatmap
        cmap = plt.get_cmap(colormap)
        weight_np = weight_norm.detach().cpu().numpy()
        heatmap = torch.from_numpy(cmap(weight_np)[..., :3]).to(weight_map.device)
        
        # if there is mask, only keep the foreground part
        if ref_img_masks is not None:
            mask = ref_img_masks.unsqueeze(-1)  # [B, Nr, H, W, 1]
            heatmap = heatmap * mask
        
        if not return_tensor:
            heatmap = heatmap.detach().cpu().numpy()
        
        heatmaps.append(heatmap)
    
    return heatmaps
    


def visualize_weight_map_grid(
    weight_maps: Union[torch.Tensor, List[torch.Tensor]],
    ref_images: Optional[torch.Tensor] = None,
    alpha_blend: float = 0.6,
    colormap: str = "jet",
    return_combined: bool = True
) -> torch.Tensor:
    """
    Create grid visualization of weight map
    
    Args:
        weight_maps: weight map data
        ref_images: reference images [B, Nr, H, W, 3], used for alpha blending
        alpha_blend: alpha blending, 0 means only show the original image, 1 means only show the heatmap
        colormap: heatmap color map
        return_combined: whether to return the combined grid image
        
    Returns:
        Visualization result tensor [B, H_grid, W_grid, 3]
    """
    
    # convert to heatmap
    heatmaps = weight_map_to_heatmap(
        weight_maps, 
        colormap=colormap, 
        normalize=True,
        return_tensor=True
    )
    
    if isinstance(heatmaps, list):
        # if there are multiple views, only take the first view for visualization
        heatmaps = heatmaps[0]
    
    B, Nr, H, W, C = heatmaps.shape
    
    if ref_images is not None:
        # alpha blend heatmap and original image
        if ref_images.shape[-1] != 3:
            ref_images = ref_images.permute(0, 1, 3, 4, 2)  # [B, Nr, H, W, 3]
        
        # ensure the size matches
        if ref_images.shape[2:4] != (H, W):
            ref_images = F.interpolate(
                ref_images.view(B*Nr, H, W, 3).permute(0, 3, 1, 2),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1).view(B, Nr, H, W, 3)
        
        # alpha blend
        combined = alpha_blend * heatmaps + (1 - alpha_blend) * ref_images
    else:
        combined = heatmaps
    
    if return_combined:
        # create grid layout [B, H*1, W*Nr, 3]
        grid_images = rearrange(combined, "B Nr H W C -> B H (Nr W) C")
        return grid_images
    else:
        return combined


def weight_map_statistics(weight_maps: Union[torch.Tensor, List[torch.Tensor]]) -> dict:
    """
    Calculate the statistics of weight map
    
    Args:
        weight_maps: weight map data
        
    Returns:
        Statistics dictionary
    """
    stats = {}
    
    if isinstance(weight_maps, list):
        stats['num_views'] = len(weight_maps)
        all_stats = []
        
        for i, weight_map in enumerate(weight_maps):
            if weight_map.dim() == 5:
                weight_map = weight_map.squeeze(-1)
            
            view_stats = {
                'shape': list(weight_map.shape),
                'min': weight_map.min().item(),
                'max': weight_map.max().item(),
                'mean': weight_map.mean().item(),
                'std': weight_map.std().item(),
            }
            
            # statistics for each reference
            B, Nr, H, W = weight_map.shape
            ref_stats = []
            for nr in range(Nr):
                ref_weight = weight_map[:, nr]
                ref_stat = {
                    'min': ref_weight.min().item(),
                    'max': ref_weight.max().item(), 
                    'mean': ref_weight.mean().item(),
                    'std': ref_weight.std().item(),
                }
                ref_stats.append(ref_stat)
            view_stats['per_ref'] = ref_stats
            all_stats.append(view_stats)
            
        stats['per_view'] = all_stats
    else:
        if weight_maps.dim() == 5:
            weight_maps = weight_maps.squeeze(-1)
            
        stats['shape'] = list(weight_maps.shape)
        stats['min'] = weight_maps.min().item()
        stats['max'] = weight_maps.max().item()
        stats['mean'] = weight_maps.mean().item()
        stats['std'] = weight_maps.std().item()
        
        # statistics for each reference
        B, Nr, H, W = weight_maps.shape
        ref_stats = []
        for nr in range(Nr):
            ref_weight = weight_maps[:, nr]
            ref_stat = {
                'min': ref_weight.min().item(),
                'max': ref_weight.max().item(),
                'mean': ref_weight.mean().item(), 
                'std': ref_weight.std().item(),
            }
            ref_stats.append(ref_stat)
        stats['per_ref'] = ref_stats
    
    return stats 