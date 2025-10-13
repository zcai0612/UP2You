import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange

class FeatureSelector:
    def __init__(
        self,
        mode="topk", # topk, sum
    ):
        self.mode = mode

    def _select_sum(
        self,
        ref_img_feats: torch.Tensor,
        view_weight_maps: torch.Tensor, # B Nr H W C
        ref_img_masks: torch.Tensor = None,
    ):
        B, Nr, H, W, C = ref_img_feats.shape
        device = ref_img_feats.device
        dtype = ref_img_feats.dtype

        
        view_weight_maps = view_weight_maps.to(device=device, dtype=dtype)
        view_weight_maps = rearrange(view_weight_maps, "B Nr H W 1 -> (B Nr) 1 H W")

        if ref_img_masks is not None:
            H_r, W_r = ref_img_masks.shape[-2:]
            ref_img_masks = rearrange(ref_img_masks, "B Nr H W -> (B Nr) H W").unsqueeze(1)
            view_weight_maps = F.interpolate(view_weight_maps, size=(H_r, W_r), mode="bilinear", align_corners=True)
            view_weight_maps = view_weight_maps * ref_img_masks

        view_weight_maps = F.interpolate(view_weight_maps, size=(H, W), mode="bilinear", align_corners=True)
        view_weight_maps = rearrange(view_weight_maps, "(B Nr) 1 H W -> B Nr H W 1", B=B, Nr=Nr)

        weighted_ref_features = ref_img_feats * view_weight_maps
        weighted_ref_features = weighted_ref_features.sum(dim=1)

        weighted_ref_features = rearrange(weighted_ref_features, "B H W C -> B (H W) C")
        return weighted_ref_features

    def _select_topk(
        self,
        ref_img_feats: torch.Tensor,
        view_weight_maps: torch.Tensor,
        ref_img_masks: torch.Tensor = None,
        max_select_ratio: float = 1.0,
        random_select_ratio: float = 0.0,
    ):
        B, Nr, H, W, C = ref_img_feats.shape
        device = ref_img_feats.device
        dtype = ref_img_feats.dtype
        
        view_weight_maps = view_weight_maps.to(device=device, dtype=dtype)
        view_weight_maps = rearrange(view_weight_maps, "B Nr H W 1 -> (B Nr) 1 H W")

        orig_view_weight_maps = view_weight_maps.clone()

        if ref_img_masks is not None:
            H_r, W_r = ref_img_masks.shape[-2:]
            ref_img_masks = rearrange(ref_img_masks, "B Nr H W -> (B Nr) H W").unsqueeze(1)
            view_weight_maps = F.interpolate(view_weight_maps, size=(H_r, W_r), mode="bilinear", align_corners=True)
            view_weight_maps = view_weight_maps * ref_img_masks

        view_weight_maps = F.interpolate(view_weight_maps, size=(H, W), mode="bilinear", align_corners=True)
        orig_view_weight_maps = F.interpolate(orig_view_weight_maps, size=(H, W), mode="bilinear", align_corners=True)
        view_weight_maps = view_weight_maps.squeeze(1)
        orig_view_weight_maps = orig_view_weight_maps.squeeze(1)

        
        if max_select_ratio > Nr:
            max_selected_ratio = Nr
        else:
            max_selected_ratio = max_select_ratio
        num_all = int(max_selected_ratio * H * W)
        
        num_random = int(num_all * random_select_ratio)
        num_topk = int(num_all - num_random)

        ref_features_flatten = rearrange(ref_img_feats, "B Nr H W C -> B (Nr H W) C")
        weights = rearrange(view_weight_maps, "(B Nr) H W -> B (Nr H W)", Nr=Nr)
        orig_weights = rearrange(orig_view_weight_maps, "(B Nr) H W -> B (Nr H W)", Nr=Nr)

        # print("*"*100)
        # print(B, Nr, H, W, C)
        # print(f"weights: {weights.shape}")
        # print(f"orig_weights: {orig_weights.shape}")
        # print(f"num_topk: {num_topk}")
        # print(f"num_random: {num_random}")
        
        # 选择topk特征 (topk操作不可微，但这是预期的)
        _, topk_indices = torch.topk(weights, k=num_topk, dim=1)  # B, num_topk
        
        # 按原始索引顺序排序topk_indices以保持特征的原有排列顺序
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=1)
        
        topk_features = torch.gather(ref_features_flatten, 1, topk_indices_sorted.unsqueeze(-1).expand(-1, -1, C))  # B, num_topk, C
        topk_weights = torch.gather(weights, 1, topk_indices_sorted)  # B, num_topk
        
        # 随机选择剩余特征
        if num_random > 0:
            # 创建掩码排除已选择的topk索引（保持可微性）
            mask = torch.ones(B, Nr * H * W, dtype=torch.bool, device=device)
            mask.scatter_(1, topk_indices.detach(), False)  # detach避免影响梯度流
            
            # 获取剩余索引的数量
            num_remaining = mask.sum(dim=1).min().item()  # 取最小值确保所有batch都有足够的索引
            
            if num_remaining >= num_random:
                random_weights_for_sampling = torch.rand(B, Nr * H * W, device=device, dtype=dtype)
                random_weights_for_sampling = random_weights_for_sampling.masked_fill(~mask, -float('inf'))
                
                # 选择随机索引 (topk操作不可微，但用于索引选择是合理的)
                _, random_indices = torch.topk(random_weights_for_sampling, k=num_random, dim=1)
                
                # 按原始索引顺序排序random_indices以保持特征的原有排列顺序
                random_indices_sorted, _ = torch.sort(random_indices, dim=1)
                
                # 获取对应的特征和权重
                random_features = torch.gather(ref_features_flatten, 1, random_indices_sorted.unsqueeze(-1).expand(-1, -1, C))
                random_weights = torch.gather(orig_weights, 1, random_indices_sorted)
                
                # 合并topk和随机选择的特征，需要再次排序以保持整体顺序
                all_selected_indices = torch.cat([topk_indices_sorted, random_indices_sorted], dim=1)
                all_selected_indices_sorted, sort_order = torch.sort(all_selected_indices, dim=1)
                
                all_selected_features = torch.cat([topk_features, random_features], dim=1)
                all_selected_weights = torch.cat([topk_weights, random_weights], dim=1)
                
                # 根据排序顺序重新排列特征和权重
                selected_features = torch.gather(all_selected_features, 1, sort_order.unsqueeze(-1).expand(-1, -1, C))
                selected_weights = torch.gather(all_selected_weights, 1, sort_order)
            else:
                # 如果剩余特征不足，只使用topk特征
                selected_features = topk_features
                selected_weights = topk_weights
        else:
            selected_features = topk_features
            selected_weights = topk_weights
        
        # 特征与权重相乘
        selected_features = selected_features * selected_weights.unsqueeze(-1)  # B, selected_num, C
        
        return selected_features

    def forward(
        self,
        ref_img_feats: torch.Tensor,
        weight_maps: List[torch.Tensor],
        ref_img_masks: torch.Tensor = None,
        max_select_ratio: float = 2.0,
        random_select_ratio: float = 0.0,
    ) -> torch.Tensor:
        selected_ref_img_feats = []

        if self.mode == "topk":
            for view_weight_maps in weight_maps:
                selected_ref_img_feats.append(self._select_topk(ref_img_feats, view_weight_maps, ref_img_masks, max_select_ratio, random_select_ratio))
        elif self.mode == "sum":
            for view_weight_maps in weight_maps:
                selected_ref_img_feats.append(self._select_sum(ref_img_feats, view_weight_maps, ref_img_masks))
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        selected_ref_img_feats = torch.cat(selected_ref_img_feats, dim=0)

        return selected_ref_img_feats