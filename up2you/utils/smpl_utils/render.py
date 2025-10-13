
import os 
import random 
import torch
import torch.nn.functional as F
import numpy as np
from time import time 
import nvdiffrast.torch as dr
from rich.progress import track
from kiui.op import scale_img_nhwc, safe_normalize
from typing import Tuple, List, NewType, Union
Tensor = NewType('Tensor', torch.Tensor)


class Renderer(torch.nn.Module):
    def __init__(self, gui=False, shading=False, hdr_path=None, device="cuda"):
        super().__init__()    
        if not gui or os.name == 'nt': 
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()

        self.light_dir = np.array([0, 0])
        self.ambient_ratio = 0.5
        self.device = device
        
        if shading and hdr_path is not None:
            import envlight
            if hdr_path is None:
                hdr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/lights/mud_road_puresky_1k.hdr') 
            self.light = envlight.EnvLight(hdr_path, scale=2, device=self.device)
            self.FG_LUT = torch.from_numpy(np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/lights/bsdf_256_256.bin"), dtype=np.float32).reshape(1, 256, 256, 2)).to(self.device)
            self.metallic_factor = 1
            self.roughness_factor = 1

    def render_bg(self, envmap_path):
        '''render with shading'''
        from PIL import Image 
        envmap = Image.open(envmap_path)
        h,w = self.res
        pos_int = torch.arange(w*h, dtype = torch.int32, device=self.device)
        pos = 0.5 - torch.stack((pos_int % w, pos_int // w), dim=1) / torch.tensor((w,h), device=self.device)
        a = np.deg2rad(self.fov_x)/2
        r = w/h
        f = torch.tensor((2*np.tan(a),  2*np.tan(a)/r), device=self.device, dtype=torch.float32)
        rays = torch.cat((pos*f, torch.ones((w*h,1), device=self.device), torch.zeros((w*h,1), device=self.device)), dim=1)
        rays_norm = (rays.transpose(0,1) / torch.norm(rays, dim=1)).transpose(0,1)
        rays_view = torch.matmul(rays_norm, self.view_mats.inverse().transpose(1,2)).reshape((self.view_mats.shape[0],h,w,-1))
        theta = torch.acos(rays_view[..., 1])
        phi = torch.atan2(rays_view[..., 0], rays_view[..., 2])
        envmap_uvs = torch.stack([0.75-phi/(2*np.pi), theta / np.pi], dim=-1)
        self.bgs = dr.texture(envmap[None, ...], envmap_uvs, filter_mode='linear').flip(1)
        self.bgs[..., -1] = 0 # Set alpha to 0

    def shading(self, albedo, normal=None, mode='albedo'):  
        if mode == "albedo":
            return albedo 
        elif mode == "lambertian":
            assert normal is not None, "normal and light direction should be provided" 
            light_d = np.deg2rad(self.light_dir)
            light_d = np.array([
                np.cos(light_d[0]) * np.sin(light_d[1]),
                -np.sin(light_d[0]),
                np.cos(light_d[0]) * np.cos(light_d[1]),
            ], dtype=np.float32)
            light_d = torch.from_numpy(light_d).to(albedo.device)
            lambertian = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ light_d).float().clamp(min=0)
            albedo = albedo * lambertian.unsqueeze(-1) 
            return albedo

        elif mode == "pbr": 
            # xyzs, _ = dr.interpolate(self.mesh.v.unsqueeze(0), rast, self.mesh.f) # [1, H, W, 3]
            # viewdir = safe_normalize(xyzs - pose[:3, 3])

            # n_dot_v = (normal * viewdir).sum(-1, keepdim=True) # [1, H, W, 1]
            # reflective = n_dot_v * normal * 2 - viewdir

            # diffuse_albedo = (1 - metallic) * albedo

            # fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1) # [H, W, 2]
            # fg = dr.texture(
            #     self.FG_LUT,
            #     fg_uv.reshape(1, -1, 1, 2).contiguous(),
            #     filter_mode="linear",
            #     boundary_mode="clamp",
            # ).reshape(1, H, W, 2)
            # F0 = (1 - metallic) * 0.04 + metallic * albedo
            # specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]

            # diffuse_light = self.light(normal)
            # specular_light = self.light(reflective, roughness)

            # color = diffuse_albedo * diffuse_light + specular_albedo * specular_light # [H, W, 3]
            # color = color * alpha + self.bg_color * (1 - alpha)

            # buffer = color[0].detach().cpu().numpy()
            raise NotImplementedError("PBR shading is not implemented")
        
        return albedo 

    def get_orthogonal_cameras(self, n=4, yaw_range=(0, 360), endpoint=False, return_all=False):
        """
        generate orthogonal cameras mvp matrix along yaw axis
        Args:
            n: int number of cameras
            yaw_range: tuple (min_yaw, max_yaw) in degrees 
        Returns:
            mvp: torch.Tensor [n, 4, 4], batch of orthogonal cameras 
        """
        # extrinsic: rotate object along yaw axis
        extrinsic = torch.eye(4)[None].expand(n, -1, -1).clone()   
        min_yaw, max_yaw = np.radians(yaw_range)
        angles = np.linspace(min_yaw, max_yaw, n, endpoint=endpoint)
        R = batch_rodrigues(torch.tensor([[0, angle, 0] for angle in angles]).reshape(-1, 3)).float() 
        extrinsic[:, :3, :3] = R 

        # intrinsic: orthogonal projection
        intrinsic = torch.eye(4)[None].expand(n, -1, -1).clone()   
        R = batch_rodrigues(torch.tensor([np.pi, 0, 0]).reshape(1, 3)).reshape(3, 3).float()   
        intrinsic[:, :3, :3] = R

        mvp = torch.bmm(intrinsic, extrinsic) 
        
        if return_all:
            return mvp, extrinsic, intrinsic
        
        return mvp
    

    def get_bg_color(self, phrase):
        if phrase is None:
            return torch.ones(3)
        if isinstance(phrase, str):
            if phrase == 'white':
                return torch.ones(3)
            elif phrase == 'black':
                return torch.zeros(3)
            elif phrase == 'gray':
                return torch.ones(3) * 0.5 
            else:
                raise ValueError('')
        elif isinstance(phrase, List):
            return torch.tensor(phrase)
        else:
            raise ValueError('')

    def render_360views(
        self, input_mesh, num_views, res=512, bg='white',  
        loop=1, shading_mode='albedo', 
        yaw_range=(0, 360), 
        resize=True, size=1.0, spp=1, 
        **kwargs
        ):
        mesh = input_mesh.clone()
        device = mesh.v.device  
        
        # bg_color = torch.ones(3).to(device) if bg == 'white' else torch.zeros(3).to(device) 
        bg_color = self.get_bg_color(bg).to(device)

        if mesh.vn is None:
            mesh.auto_normal() 
        
        if resize:
            mesh.auto_size(size) 

        # camera 
        yaw_range = (yaw_range[0], yaw_range[1]*loop)
        mvps = self.get_orthogonal_cameras(num_views, yaw_range).to(device)
        # mvps[:, 2, 3] += 0.1  # move camera back a bit
        # print('debuging camera dist ', mvps[:, 2, 3])
        pkg = self.forward(mesh, mvps, spp=spp, bg_color=bg_color, h=res, w=res, shading_mode=shading_mode, **kwargs)
        return pkg
    

    def forward(self, mesh, mvp, h=512, w=512,
                light_d=None,
                ambient_ratio=1.,
                shading_mode='albedo',
                spp=1,
                show_wire=False, 
                wire_width=0.05,
                wire_color=[0.5, 0.5, 0.5], 
                bg_color=None):
        """
        Args: 
            mesh: Mesh object
            mvp: [batch, 4, 4]
            h: int
            w: int
            light_d:
            spp: int
            ambient_ratio: float
            shading_mode: str rendering type albedo, lambertian or pbr
        Returns:
            color: [batch, h, w, 3]
            alpha: [batch, h, w, 1] 
        """
        assert shading_mode in ['albedo', 'lambertian', 'pbr'], "shading_mode should be albedo, lambertian or pbr"
        
        if not h % 8 == 0 and w % 8 == 0:
            raise ValueError("h and w should be multiples of 8")
        
        B = mvp.shape[0] 
        v_homo = F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1)
        v_clip = torch.bmm(v_homo, torch.transpose(mvp, 1, 2)).float()  # [B, N, 4]

        
        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)
        alpha = (rast[..., 3:] > 0).float() # [B, H, W, 1] 
        
        # alpha, _ = dr.interpolate(torch.ones_like(v_clip[..., :1]), rast, mesh.f)  # [B, H, W, 1] 

        ### normal  
        if mesh.vn is None:
            mesh.auto_normal()
            
        normal, _ = dr.interpolate(mesh.vn[None, ...].float(), rast, mesh.f)
        normal = (normal + 1) / 2.
        
        if mesh.vt is not None and mesh.ft is not None:
            texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs='all') 
        else:
            texc = None 
        
        ### albedo   
        if mesh.albedo is not None:
            color = dr.texture(mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear')  # [B, H, W, 3] 
        elif mesh.vc is not None:
            color, _ = dr.interpolate(mesh.vc[None, ..., :3].contiguous().float(), rast, mesh.f)
        else:
            color = None

        ### shading  
        if color is not None:
            color = torch.where(rast[..., 3:] > 0, color, torch.tensor(0).to(color.device))  # remove background
            color = self.shading(color, normal, mode=shading_mode)
            
        ### antialias
        normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        if color is not None:
            color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        
        if show_wire:
            u = rast[..., 0] # [1, h, w]
            v = rast[..., 1] # [1, h, w]
            _w = 1 - u - v
            # mask = rast[..., 2]
            near_edge = (((_w < wire_width) | (u < wire_width) | (v < wire_width)) & (alpha[..., 0] > 0)) # [B, h, w]
            color[near_edge] = torch.tensor(wire_color).float().to(u.device) 
            
        ### inverse super-sampling
        if spp > 1:
            if color is not None:
                color = scale_img_nhwc(color, (h, w))
            alpha = scale_img_nhwc(alpha, (h, w))
            normal = scale_img_nhwc(normal, (h, w)) 

        ### background
        if bg_color is not None:
            if color is not None:
                color = color * alpha + bg_color * (1 - alpha)
            normal = normal * alpha + bg_color * (1 - alpha)
        
        return {
            'image': color,
            'normal': normal,
            'alpha': alpha, 
            'uvs': texc, 
            'pix_to_face': rast[..., 3].long()
        }



def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat