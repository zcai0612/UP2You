import os
import kornia
import numpy as np
import torch
import torch.optim as optim
import trimesh

from core.opt import MeshOptimizer

from tqdm import tqdm
from PIL import Image
from kiui.mesh import Mesh 

from up2you.utils.smpl_utils.render import Renderer
from up2you.utils.smpl_utils.camera import Camera
from up2you.utils.smpl_utils.mesh import normalize_vertices
from .mesh_util import poisson
from .project_mesh import multiview_color_projection, get_cameras_list

from up2you.utils.smpl_utils.smpl_util import SMPLX, part_removal, apply_vertex_mask

bg_color = np.array([1,1,1])

def to_py3d_mesh(vertices, faces, normals=None):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.textures import TexturesVertex
    mesh = Meshes(verts=[vertices], faces=[faces], textures=None)
    if normals is None:
        normals = mesh.verts_normals_packed()
    # set normals as vertext colors
    mesh.textures = TexturesVertex(verts_features=[normals / 2 + 0.5])
    return mesh

def save_mesh(save_name, vertices, faces,  color=None):
    trimesh.Trimesh(
        vertices.detach().cpu().numpy(), 
        faces.detach().cpu().numpy(), 
        vertex_colors=(color.detach().cpu().numpy() * 255).astype(np.uint8) if color is not None else None) \
    .export(save_name)


class Reconstructor:
    def __init__(
        self,
        device,
        normal_views=[0, 45, 90, 180, 270, 315],
        color_views=[0, 45, 90, 180, 270, 315],
        scale=2.0,
        resolution=1024,
        iters=700,
    ) -> None:
        self.device = device
        self.renderer = Renderer(device=device)

        self.camera = Camera(device=device)
        self.normal_views = normal_views
        self.color_views = color_views

        self.mvps, self.rots, _, _ = self.camera.get_orthogonal_camera([-view % 360 for view in self.normal_views], radius=scale)
        self.bg_color = torch.from_numpy(bg_color).to(self.device)

        self.resolution = resolution
        self.scale = scale
        self.iters = iters

        # for color projection
        self.weights = torch.Tensor([1., 0.4, 0.8, 1.0, 0.8, 0.4]).view(6,1,1,1).to(self.device)

        # for rescale
        self.mesh_center = None
        self.mesh_scale = None
        

    def preprocess(self, normal_pils):
        ###------------------ load target images -------------------------------
        # color_pils: 6views
        # normal_pils: 4views
        kernal = torch.ones(3, 3)
        erode_iters = 2
        normals = []
        masks = []

        idx_normal = 0
        for normal in normal_pils:
            normal = normal.resize((self.resolution, self.resolution), Image.BILINEAR)
            normal = np.array(normal).astype(np.float32) / 255.
            mask = normal[..., 3:]
            mask_torch = torch.from_numpy(mask).unsqueeze(0)
            for _ in range(erode_iters):
                mask_torch = kornia.morphology.erosion(mask_torch, kernal)
            mask_erode = mask_torch.squeeze(0).numpy()
            masks.append(mask_erode)
            normal = normal[..., :3] * mask_erode 
            normals.append(normal)

        masks = np.stack(masks, 0)
        masks = torch.from_numpy(masks).to(self.device)
        normals = np.stack(normals, 0) 
        target_normals = torch.from_numpy(normals).to(self.device)

        return masks, target_normals

    def proj_texture(self, fused_images, vertices, faces):
        mesh = to_py3d_mesh(vertices, faces)
        mesh = mesh.to(self.device)
        camera_focal =  1/2
        cameras_list = get_cameras_list(self.color_views, device=self.device, focal=camera_focal)
        mesh = multiview_color_projection(mesh, fused_images, camera_focal=camera_focal, resolution=self.resolution, weights=self.weights.squeeze().cpu().numpy(),
                                        device=self.device, complete_unseen=True, confidence_threshold=0.2, cameras_list=cameras_list)
        return mesh


    def render(self, mesh: Mesh):
        mesh.f = mesh.f.to(torch.int32)
        mesh.auto_normal()
        render_pkg = self.renderer(
            mesh, mvp=self.mvps, h=self.resolution, w=self.resolution, shading_mode='albedo',
            bg_color=self.bg_color,
        )
        normals = render_pkg['normal']
        alphas = render_pkg['alpha']
        normals = normals * alphas
        return normals, alphas


    def geometry_optimization(
        self, normal_masks, target_normals, smplx_v, smplx_f, output_dir, replace_hand=False
    ):
        mesh_smpl = trimesh.Trimesh(vertices=smplx_v.detach().cpu().numpy(), faces=smplx_f.detach().cpu().numpy())
        nrm_opt = MeshOptimizer(smplx_v.detach(), smplx_f.detach().to(torch.int64), edge_len_lims=[0.01, 0.1])
        vertices, faces = nrm_opt.vertices, nrm_opt.faces
        optim_mesh = Mesh(v=vertices, f=faces, device=self.device)

        for i in tqdm(range(self.iters)):
            nrm_opt.zero_grad()

            normals, alphas= self.render(
                optim_mesh
            )

            loss_normal = (normals - target_normals).abs().mean()
            loss_alpha = (alphas - normal_masks).abs().mean()

            loss = loss_normal + loss_alpha
            loss.backward()
            nrm_opt.step()

            vertices,faces = nrm_opt.remesh()

            optim_mesh.v = vertices
            optim_mesh.f = faces
            torch.cuda.empty_cache() 

        mesh_remeshed = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())

        verts_rescale = (vertices / self.mesh_scale) + self.mesh_center
        mesh_remeshed_save = trimesh.Trimesh(vertices=verts_rescale.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
        mesh_remeshed_save.export(os.path.join(output_dir, "mesh_remeshed.obj"))
        smpl_data = SMPLX()
        if replace_hand:
            hand_mask = torch.zeros(smpl_data.smplx_verts.shape[0], )
            hand_mask.index_fill_(
                0, torch.tensor(smpl_data.smplx_mano_vid_dict["left_hand"]), 1.0
            )
            hand_mask.index_fill_(
                0, torch.tensor(smpl_data.smplx_mano_vid_dict["right_hand"]), 1.0
            )
            hand_mesh = apply_vertex_mask(mesh_smpl.copy(), hand_mask)
            body_mesh = part_removal(
                mesh_remeshed.copy(),
                hand_mesh,
                0.08,
                self.device,
                mesh_smpl.copy(),
                region="hand"
            )
            final = poisson(sum([hand_mesh, body_mesh]), f'{output_dir}/mesh_final.obj', 10, False)
        else:
            final = poisson(mesh_remeshed, f'{output_dir}/mesh_final.obj', 10, False)


        return final

    def color_projection(self, final_mesh, color_pils, output_dir):

        vertices = torch.from_numpy(final_mesh.vertices).float().to(self.device)
        faces = torch.from_numpy(final_mesh.faces).long().to(self.device)

        masked_color = []
        for tmp in color_pils:
            # tmp = Image.open(f'{self.opt.mv_path}/{case}/color_{view}_masked.png')
            tmp = tmp.resize((self.resolution, self.resolution), Image.BILINEAR)
            tmp = np.array(tmp).astype(np.float32) / 255.
            masked_color.append(torch.from_numpy(tmp).permute(2, 0, 1).to(self.device))

        meshes = self.proj_texture(masked_color, vertices, faces)
        vertices = meshes.verts_packed().float()
        faces = meshes.faces_packed().long()
        colors = meshes.textures.verts_features_packed().float()
        verts_rescale = (vertices / self.mesh_scale) + self.mesh_center

        final_mesh_path = os.path.join(output_dir, "result_clr.obj")
        save_mesh(final_mesh_path, verts_rescale, faces, colors)
        return final_mesh_path

    def run(
        self,
        color_pils,
        normal_pils,
        smplx_obj_path,
        output_dir,
        replace_hand=False
    ):
        smplx_mesh = Mesh.load_obj(smplx_obj_path)
        os.makedirs(output_dir, exist_ok=True)
        smplx_v, smplx_f = smplx_mesh.v, smplx_mesh.f
        smplx_v, center, scale = normalize_vertices(smplx_v, bound=self.scale*0.9, return_params=True)
        self.mesh_center = center
        self.mesh_scale = scale
        normal_masks, target_normals = self.preprocess(normal_pils)
        final_mesh = self.geometry_optimization(normal_masks, target_normals, smplx_v, smplx_f, output_dir, replace_hand)
        final_mesh_path = self.color_projection(final_mesh, color_pils, output_dir)
        return final_mesh_path
