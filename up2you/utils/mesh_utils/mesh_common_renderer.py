import os
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import sys
from kiui.mesh import Mesh
from ..smpl_utils.render import Renderer
from ..smpl_utils.camera import Camera
from ..smpl_utils.mesh import normalize_vertices

class CommonRenderer(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        ortho_views: list[int] = [0, 45, 90, 180, 270, 315],
        return_rgba: bool = False,
        resolution: int = 768,
    ):
        super().__init__()

        self.camera = Camera(device=device)
        self.device = device

        self.renderer = Renderer(device=device)

        self.ortho_views = ortho_views
        self.mvps, self.rots, _, _ = self.camera.get_orthogonal_camera([-view % 360 for view in self.ortho_views])
        self.resolution = resolution
        self.return_rgba = return_rgba

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
        self, mesh, mvps=None, rots=None, normal_type="world", shading_mode="albedo", background_color: str = "gray",
    ):
        if mvps is None:
            mvps = self.mvps
        if rots is None:
            rots = self.rots
        bg_color = self.renderer.get_bg_color(background_color).to(self.device)
        mesh_pkg = self.renderer(
            mesh,
            mvp=mvps,
            h=self.resolution, w=self.resolution, shading_mode=shading_mode,
            bg_color=bg_color,
        )
        mesh_image = mesh_pkg['image']
        if normal_type == "world":
            mesh_normal = self.camera_to_world_normal(mesh_pkg['normal'], mesh_pkg['alpha'], rots, bg_color)
        else:
            mesh_normal = mesh_pkg['normal']
        if mesh_image is None:
                mesh_image = torch.zeros_like(mesh_normal)
        if self.return_rgba:
            mesh_image_rgba = torch.cat([mesh_image, mesh_pkg['alpha']], dim=-1)
            mesh_normal_rgba = torch.cat([mesh_normal, mesh_pkg['alpha']], dim=-1)
            return mesh_image_rgba.permute(0, 3, 1, 2), mesh_normal_rgba.permute(0, 3, 1, 2)
        else:
            return mesh_image.permute(0, 3, 1, 2), mesh_normal.permute(0, 3, 1, 2)

    def forward(
        self,
        obj_path,
        albedo_path,
        mvps=None,
        rots=None,
        normal_type="world",
        shading_mode="albedo",
        background_color: str = "gray",
    ):
        if obj_path.endswith(".obj"):
            mesh = Mesh.load_obj(obj_path, albedo_path=albedo_path, device=self.device)
        else:
            mesh = Mesh.load(obj_path, device=self.device)
        mesh.v = normalize_vertices(mesh.v, bound=1.85/2)
        mesh.auto_normal()

        return self.render_ortho_views(mesh, mvps, rots, normal_type, shading_mode, background_color)

    def render_video(
        self,
        obj_path,
        albedo_path,    
        num_frames=64,
        normal_type="camera",
        shading_mode="albedo",
        background_color: str = "gray",
    ):
        render_views = [int(360 // num_frames * i) for i in range(num_frames)]
        mvps, rots, _, _ = self.camera.get_orthogonal_camera([-view % 360 for view in render_views])
        if obj_path.endswith(".obj"):
            mesh = Mesh.load_obj(obj_path, albedo_path=albedo_path, device=self.device)
        else:
            mesh = Mesh.load(obj_path, device=self.device)
        mesh.v = normalize_vertices(mesh.v, bound=1.85/2)
        mesh.auto_normal()
        return self.render_ortho_views(mesh, mvps, rots, normal_type, shading_mode, background_color)