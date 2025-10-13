import os
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import math 
import random

from smplx.lbs import batch_rodrigues


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


class Camera:
    def __init__(self, device='cuda'):
        self.device = device

    def get_orthogonal_camera(self, views, radius=1.0):
        """Initialize orthogonal cameras for rendering.
        
        Args:
            views: List of viewing angles
            radius: Viewing range radius, defines viewing volume as [-radius, radius] (default: 1.0)
        """
        rot = batch_rodrigues(
            torch.tensor([
                [0, np.radians(angle), 0] for angle in views 
            ])).float().to(self.device) 
        extrinsic = torch.eye(4)[None].expand(len(views), -1, -1).clone() .to(self.device)
        extrinsic[:, :3, :3] = rot

        extrinsic[:, 3, 3] = radius
        
        intrinsic = torch.eye(4)[None].expand(len(views), -1, -1).clone() .to(self.device)
        R = batch_rodrigues(torch.tensor([np.pi, 0, 0]).reshape(1, 3)).reshape(3, 3).float()   
        intrinsic[:, :3, :3] = R
        
        # Scale the projection based on radius
        # For orthogonal projection, we need to scale by 1/radius to map [-radius, radius] to [-1, 1]
        # scale_factor = 1.0 / radius
        # intrinsic[:, 0, 0] *= scale_factor  # Scale X
        # intrinsic[:, 1, 1] *= scale_factor  # Scale Y
        
        mvps = torch.bmm(intrinsic, extrinsic) 
        extrinsic = extrinsic

        return mvps, rot, extrinsic, intrinsic
    
    
    def get_predefined_camera(
        self, 
        shifts, # list
        radius, # list
        elevation, # list
        azimuth, # list
        fov, # list
        shifts_jitter_range=[[-0.1,-0.1,-0.1], [0.1,0.1,0.1]],
        radius_jitter_range=[-0.05, 0.05],
        elevation_jitter_range=[-10, 10],
        azimuth_jitter_range=[-10, 10],
        fov_jitter_range=[10, -10],
    ):
        num_views = len(shifts)
        if not (len(shifts) == len(radius) == len(elevation) == len(azimuth) == len(fov)):
            raise ValueError("shifts, radius, elevation, azimuth and fov must have the same length")
        
        mvps = []
        rots = []
        extrinsics = [] 
        intrinsics = []

        for i in range(num_views):
            shift = torch.tensor(shifts[i]).float().to(self.device)
            curr_radius = radius[i]
            curr_elevation = elevation[i]
            curr_azimuth = azimuth[i]
            curr_fov = fov[i]

            shift_jitter = [random.uniform(shifts_jitter_range[0][j], shifts_jitter_range[1][j]) for j in range(3)]
            radius_jitter = random.uniform(radius_jitter_range[0], radius_jitter_range[1])
            elevation_jitter = random.uniform(elevation_jitter_range[0], elevation_jitter_range[1])
            azimuth_jitter = random.uniform(azimuth_jitter_range[0], azimuth_jitter_range[1])
            fov_jitter = random.uniform(fov_jitter_range[0], fov_jitter_range[1])
            
            final_shift = (shift + torch.tensor(shift_jitter).float().to(self.device)).view(1, 3)
            final_radius = curr_radius + radius_jitter
            final_elevation = curr_elevation + elevation_jitter
            final_azimuth = curr_azimuth + azimuth_jitter
            final_fov = curr_fov + fov_jitter
            
            elev_rad = np.radians(final_elevation)
            azim_rad = np.radians(final_azimuth)
            
            centers = torch.tensor([
                final_radius * np.sin(elev_rad) * np.sin(azim_rad),
                final_radius * np.cos(elev_rad),
                final_radius * np.sin(elev_rad) * np.cos(azim_rad)
            ]).float().to(self.device) + final_shift
            
            targets = final_shift + torch.zeros_like(centers)
            forward_vector = safe_normalize(centers - targets)
            up_vector = torch.FloatTensor([0, 1, 0]).to(self.device).view(1, 3)
            right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
            up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

            poses = torch.eye(4, dtype=torch.float, device=self.device)
            poses[:3, :3] = torch.stack([right_vector, up_vector, forward_vector], dim=-1).squeeze(0)
            poses[:3, 3] = centers.squeeze(0)
            
            extrinsic = torch.inverse(poses)
            rotation_matrix = extrinsic[:3, :3]
            R_x = batch_rodrigues(torch.tensor([[np.pi, 0, 0]])).float().to(self.device)
            R_y = batch_rodrigues(torch.tensor([[0, np.pi, 0]])).float().to(self.device)
            rotation_matrix = (rotation_matrix @ R_x @ R_y).to(self.device)
            extrinsic[:3, :3] = rotation_matrix
            
            fov_rad = np.radians(final_fov)
            focal_length = 1.0 / np.tan(fov_rad / 2.0)
            
            intrinsic = torch.eye(4).float().to(self.device)
            intrinsic[0, 0] = focal_length  # fx
            intrinsic[1, 1] = focal_length  # fy
            intrinsic[2, 2] = -1.0  # invert Z axis for right-handed coordinate system
            intrinsic[3, 2] = -1.0  # perspective division
            
            mvp = intrinsic @ extrinsic
            
            mvps.append(mvp.unsqueeze(0))
            rots.append(rotation_matrix)
            extrinsics.append(extrinsic.unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0))
        
        mvps = torch.cat(mvps, dim=0)
        rots = torch.cat(rots, dim=0)
        extrinsics = torch.cat(extrinsics, dim=0)
        intrinsics = torch.cat(intrinsics, dim=0)
        
        return mvps, rots, extrinsics, intrinsics

    def get_random_camera(
        self, 
        elevation_range, # list, different center has different elevation range
        azimuth_range, # list, different center has different azimuth range
        fov_range, 
        radius_ranges, # list, different center has different radius
        num_views, 
        shifts # list, different center
    ):
        if not (len(shifts) == len(radius_ranges) == len(elevation_range) == len(azimuth_range)):
            raise ValueError("shifts, radius_ranges, elevation_range and azimuth_range must have the same length")
        
        mvps = []
        rots = []
        extrinsics = []
        intrinsics = []

        num_centers = len(shifts)
        if num_views < num_centers:
            raise ValueError(f"num_views ({num_views}) 必须大于等于中心点数量 ({num_centers})")
        
        indices = list(range(num_centers))  # ensure each index is selected at least once
        remaining_views = num_views - num_centers
        if remaining_views > 0:
            indices.extend([random.randint(0, num_centers - 1) for _ in range(remaining_views)])
        
        random.shuffle(indices)
        
        for view_idx in range(num_views):
            shift_idx = indices[view_idx]
            shift = torch.tensor(shifts[shift_idx]).float().to(self.device)
            radius_range = radius_ranges[shift_idx]
            elev_range = elevation_range[shift_idx]
            azim_range = azimuth_range[shift_idx]
            
            elev = random.randint(elev_range[0], elev_range[1])
            azim = random.randint(azim_range[0], azim_range[1])
            fov = random.randint(fov_range[0], fov_range[1])
            radius = random.uniform(radius_range[0], radius_range[1])
            
            elev_rad = np.radians(elev)
            azim_rad = np.radians(azim)
            
            shift = torch.as_tensor(shift, device=self.device).view(1, 3)
            centers = torch.tensor([
                radius * np.sin(elev_rad) * np.sin(azim_rad),
                radius * np.cos(elev_rad),
                radius * np.sin(elev_rad) * np.cos(azim_rad)
            ]).float().to(self.device) + shift
            
            targets = shift + torch.zeros_like(centers)
            forward_vector = safe_normalize(centers - targets)
            up_vector = torch.FloatTensor([0, 1, 0]).to(self.device).view(1, 3)
            right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
            up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

            poses = torch.eye(4, dtype=torch.float, device=self.device)
            poses[:3, :3] = torch.stack([right_vector, up_vector, forward_vector], dim=-1).squeeze(0)
            poses[:3, 3] = centers.squeeze(0)
            
            extrinsic = torch.inverse(poses)
            rotation_matrix = extrinsic[:3, :3]
            R_x = batch_rodrigues(torch.tensor([[np.pi, 0, 0]])).float().to(self.device)
            R_y = batch_rodrigues(torch.tensor([[0, np.pi, 0]])).float().to(self.device)
            rotation_matrix = (rotation_matrix @ R_x @ R_y).to(self.device)
            extrinsic[:3, :3] = rotation_matrix
            # extrinsic = torch.eye(4).float().to(self.device)
            # rot = batch_rodrigues(torch.tensor([[0, np.radians(azim), 0]])
            # ) @ batch_rodrigues(torch.tensor([[np.radians(elev), 0, 0]]))  
            # R = batch_rodrigues(torch.tensor([[np.pi, 0, 0]]))
            # rotation_matrix = (rot.float() @ R.float()).to(self.device)
            # extrinsic[:3, :3] = rotation_matrix
            # extrinsic[:3, 3] = camera_position
            
            fov_rad = np.radians(fov)
            focal_length = 1.0 / np.tan(fov_rad / 2.0)
            
            intrinsic = torch.eye(4).float().to(self.device)
            intrinsic[0, 0] = focal_length  # fx
            intrinsic[1, 1] = focal_length  # fy
            intrinsic[2, 2] = -1.0  # invert Z axis for right-handed coordinate system
            intrinsic[3, 2] = -1.0  # perspective division
            
            mvp = intrinsic @ extrinsic
            
            mvps.append(mvp.unsqueeze(0))
            rots.append(rotation_matrix)
            extrinsics.append(extrinsic.unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0))
        
        mvps = torch.cat(mvps, dim=0)
        rots = torch.cat(rots, dim=0) 
        extrinsics = torch.cat(extrinsics, dim=0)
        intrinsics = torch.cat(intrinsics, dim=0)
        
        return mvps, rots, extrinsics, intrinsics