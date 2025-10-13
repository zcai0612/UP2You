import os
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import sys
from kiui.mesh import Mesh
from .render import Renderer
from .camera import Camera
from .mesh import normalize_vertices

class CommonRenderer(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        ortho_views: list[int] = [0, 45, 90, 180, 270, 315],
        background_color: str = "gray",
        normal_type: str = "world",
        return_rgba: bool = False,
        resolution: int = 768,
    ):
        super().__init__()

        self.camera = Camera(device=device)
        self.device = device

        self.renderer = Renderer(device=device)

        self.ortho_views = ortho_views
        self.mvps, self.rots, _, _ = self.camera.get_orthogonal_camera([-view % 360 for view in self.ortho_views])
        self.bg_color = self.renderer.get_bg_color(background_color).to(self.device)
        self.resolution = resolution
        self.return_rgba = return_rgba
        self.normal_type = normal_type

    def camera_to_world_normal(self, normals_camera, masks, rot, bg_color):
        normals_world = normals_camera * masks * 2 - 1
        normals_world = F.normalize(normals_world, dim=-1)
        normals_world = normals_world * masks - (1 - masks)
        
        rot_transpose = rot.transpose(1, 2)
        normals_world = torch.bmm(normals_world.reshape(len(normals_camera), -1, 3), rot_transpose).reshape(*normals_camera.shape)
        
        normals_world = (normals_world + 1) / 2
        normals_world = normals_world * masks + (1 - masks) * bg_color
        
        return normals_world
    
    def render_ortho_views(
        self, mesh
    ):
        smpl_pkg = self.renderer(
            mesh,
            mvp=self.mvps,
            h=self.resolution, w=self.resolution, shading_mode='albedo',
            bg_color=self.bg_color,
        )
        if self.normal_type == "world":
            smpl_normal = self.camera_to_world_normal(smpl_pkg['normal'], smpl_pkg['alpha'], self.rots, self.bg_color)
        elif self.normal_type == "camera":
            smpl_normal = smpl_pkg['normal']
        else:
            raise ValueError(f"Invalid normal type: {self.normal_type}")
        
        if self.return_rgba:
            smpl_normal_rgba = torch.cat([smpl_normal, smpl_pkg['alpha']], dim=-1)
            return smpl_normal_rgba.permute(0, 3, 1, 2)
        else:
            return smpl_normal.permute(0, 3, 1, 2)

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ):
        vertices = vertices.detach().to(self.device)
        vertices = normalize_vertices(vertices, bound=0.9)

        faces = faces.detach().to(self.device)

        mesh = Mesh(v=vertices, f=faces, device=self.device)
        mesh.auto_normal()

        return self.render_ortho_views(mesh)