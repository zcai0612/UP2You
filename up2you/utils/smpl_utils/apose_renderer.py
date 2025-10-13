import os
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import sys
import smplx
import json

from PIL import Image
from tqdm import tqdm
from kiui.mesh import Mesh 

from .render import Renderer
from .camera import Camera
from .mesh import normalize_vertices

from matplotlib import cm as mpl_cm, colors as mpl_colors

def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)
    for part_idx, (k, v) in enumerate(part_segm.items()):
        vertex_labels[v]= part_idx
    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()
    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3]= alpha
    vertex_colors[:,:3]= cm(norm_gt(vertex_labels))[:, :3]
    return vertex_colors

class AposeRenderer(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        smplx_model_path: str = "human_models/models/smplx/SMPLX_NEUTRAL.npz",
        part_segmentation_path: str = "human_models/smplx_vert_segmentation.json",
        ortho_views: list[int] = [0, 45, 90, 180, 270, 315],
        background_color: str = "gray",
    ):
        super().__init__()
        self.device = device
        self.body_model = smplx.SMPLX(
            smplx_model_path, use_pce=False, flat_hand_mean=True
        ).to(self.device).eval()

        with open(part_segmentation_path, "r") as f:
            self.part_segmentation = json.load(f)
        
        # Load APose
        body_pose = torch.zeros((1, self.body_model.NUM_BODY_JOINTS, 3))
        body_pose[:, 15, :] = torch.tensor([0.0, 0.0, -np.pi/4])
        body_pose[:, 16, :] = torch.tensor([0.0, 0.0, +np.pi/4])
        self.body_pose = body_pose.float().to(self.device).unsqueeze(0)

        self.camera = Camera(device=device)

        self.renderer = Renderer(device=device)

        self.ortho_views_6 = [0, 45, 90, 180, 270, 315]
        self.ortho_views_4 = [0, 90, 180, 270]
        self.mvps_6, self.rots_6, _, _ = self.camera.get_orthogonal_camera([-view % 360 for view in self.ortho_views_6])
        self.mvps_4, self.rots_4, _, _ = self.camera.get_orthogonal_camera([-view % 360 for view in self.ortho_views_4])
        self.bg_color = self.renderer.get_bg_color(background_color).to(self.device)
        # self.normal_type = normal_type
        # self.rotation_matrix = np.array([[1, 0, 0],
        #                     [0, -1, 0],
        #                     [0, 0, -1]], dtype=np.float32)
        
        

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
        self, mesh, height, width, normal_type, return_rgba, num_views,
    ):
        if num_views == 6:
            mvps = self.mvps_6
            rots = self.rots_6
        elif num_views == 4:
            mvps = self.mvps_4
            rots = self.rots_4
        else:
            raise ValueError(f"Invalid number of views: {num_views}")
        smpl_pkg = self.renderer(
            mesh,
            mvp=mvps,
            h=height, w=width, shading_mode='albedo',
            bg_color=self.bg_color,
        )
        if normal_type == "world":
            smpl_normal = self.camera_to_world_normal(smpl_pkg['normal'], smpl_pkg['alpha'], rots, self.bg_color)
        elif normal_type == "camera":
            smpl_normal = smpl_pkg['normal']
        else:
            raise ValueError(f"Invalid normal type: {normal_type}")

        smpl_semantic = smpl_pkg['image']
        
        if return_rgba:
            smpl_normal_rgba = torch.cat([smpl_normal, smpl_pkg['alpha']], dim=-1)
            smpl_semantic_rgba = torch.cat([smpl_semantic, smpl_pkg['alpha']], dim=-1)
            return smpl_normal_rgba.permute(0, 3, 1, 2), smpl_semantic_rgba.permute(0, 3, 1, 2)
        else:
            return smpl_normal.permute(0, 3, 1, 2), smpl_semantic.permute(0, 3, 1, 2)

    def get_tpose_mesh(self, betas: torch.Tensor):
        betas = torch.tensor(betas).float().to(self.device)
        res = self.body_model(
            betas=betas,
        )
        vertices = res.vertices.squeeze(0).detach()
        faces = torch.from_numpy(self.body_model.faces.astype(np.int32)).to(self.device)
        mesh = Mesh(v=vertices, f=faces, device=self.device)
        mesh.auto_normal()
        return mesh

    def forward(
        self,
        betas: torch.Tensor,
        height: int = 1024,
        width: int = 1024,
        normal_type: str = "camera",
        return_rgba: bool = False,
        num_views: int = 6,
        return_mesh: bool = False,
    ):
        res = self.body_model(
            body_pose=self.body_pose,
            betas=betas,
        )
        vertices = res.vertices.squeeze(0).detach().cpu().numpy()
        vertices = normalize_vertices(vertices, bound=0.9)
        faces = self.body_model.faces
        
        vertices_copy = vertices.copy()
        faces_copy = faces.copy()
        
        vertex_colors = part_segm_to_vertex_colors(self.part_segmentation, vertices.shape[0])
        vertex_colors = torch.from_numpy(vertex_colors).float().to(self.device)
        # vertex_colors = vertex_colors.unsqueeze(0)

        # Apply Rotation
        # vertices_copy = vertices_copy @ self.rotation_matrix.T
        
        vertices_tensor = torch.from_numpy(vertices_copy).float().to(self.device)
        faces_tensor = torch.from_numpy(faces_copy.astype(np.int32)).to(self.device)
        mesh = Mesh(v=vertices_tensor, f=faces_tensor, vc=vertex_colors, device=self.device)
        mesh.auto_normal()

        smpl_normals, smpl_semantics = self.render_ortho_views(mesh, height, width, normal_type, return_rgba=return_rgba, num_views=num_views)

        if return_mesh:
            return smpl_normals, smpl_semantics, vertices_tensor, faces_tensor
        else:
            return smpl_normals, smpl_semantics
    