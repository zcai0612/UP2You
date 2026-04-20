import os
import random
import smplx
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.utils import save_image
from ..utils.core import find
from ..utils.typing import *

from .base import BaseSystem

from ..models.encoder.dinov2_wrapper import Dinov2Wrapper
from ..models.shape_predictor import ShapePredictor
from ..utils.smpl_utils.smplx_common_renderer import CommonRenderer

class MVPuzzleMV2ShapeSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        # Model
        pretrained_shape_predictor_name_or_path: Optional[str] = None
        human_model_path: str = "human_models/SMPLX_NEUTRAL.npz"
        init_shape_predictor_kwargs: Dict[str, Any] = field(default_factory=dict)
        init_image_encoder_kwargs: Dict[str, Any] = field(default_factory=dict)
        
        # Training
        train_shape_predictor: bool = True
        num_reference_image_range: List[int] = field(default_factory=lambda: [3, 7])
        front_reference_prob: float = 0.2

        batch_size: int = 1

    cfg: Config

    def configure(self):
        super().configure()
        init_shape_predictor_kwargs = OmegaConf.to_container(self.cfg.init_shape_predictor_kwargs)
        init_image_encoder_kwargs = OmegaConf.to_container(self.cfg.init_image_encoder_kwargs)
        
        self.image_encoder = Dinov2Wrapper(
            **init_image_encoder_kwargs
        )
        self.shape_predictor = ShapePredictor(
            **init_shape_predictor_kwargs
        )
        self.renderer = CommonRenderer(
            normal_type="camera",
            return_rgba=True,
        )
        self.body_model_train = smplx.SMPLX(
            self.cfg.human_model_path,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=self.cfg.batch_size
        )
        self.body_model_val = smplx.SMPLX(
            self.cfg.human_model_path,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1
        )
        if self.cfg.pretrained_shape_predictor_name_or_path:
            self.shape_predictor.load_state_dict(
                torch.load(self.cfg.pretrained_shape_predictor_name_or_path)
            )
        
        self.shape_predictor.requires_grad_(self.cfg.train_shape_predictor)

    def preprocess_reference_images(
        self, 
        ref_rgbs: Tensor, 
        ref_alphas: Tensor,
        num_refs: int,
    ):
        num_refs_range = self.cfg.num_reference_image_range
        num_refs_range = (num_refs_range[0], num_refs_range[1])
        num_refs_ = random.randint(num_refs_range[0], num_refs_range[1])

        ref_rgbs_ = rearrange(ref_rgbs, "(B Nr) C H W -> B Nr C H W", Nr=num_refs)
        ref_rgbs_ = ref_rgbs_[:, :num_refs_, :, :, :]
        ref_rgbs_ = rearrange(ref_rgbs_, "B Nr C H W -> (B Nr) C H W")

        ref_alphas_ = rearrange(ref_alphas, "(B Nr) H W -> B Nr H W", Nr=num_refs)
        ref_alphas_ = ref_alphas_[:, :num_refs_, :, :]
        ref_alphas_ = rearrange(ref_alphas_, "B Nr H W -> (B Nr) H W")
        return ref_rgbs_, ref_alphas_, num_refs_
        

    def forward(
        self,
        ref_rgbs: Tensor,
        num_refs: int,
    ):
        # Process reference latents to obtain reference features
        with torch.no_grad():
            ref_img_feats = self.image_encoder(ref_rgbs)
            ref_img_feats = rearrange(
                ref_img_feats,
                "(B Nr) H W C -> B Nr H W C",
                Nr=num_refs
            )

        pred_shape = self.shape_predictor(ref_img_feats)
        return pred_shape


    def training_step(self, batch, batch_idx):
        num_refs = batch["num_refs"]
        ref_rgbs = batch["ref_rgbs"]
        ref_alphas = batch["ref_alphas"]
        target_shapes = batch["target_shapes"].view(-1, 10)
        bs = target_shapes.shape[0]
    

        ref_rgbs, ref_alphas, num_refs = self.preprocess_reference_images(
            ref_rgbs, ref_alphas, num_refs
        )

        pred_shapes = self(
            ref_rgbs,
            num_refs
        )
        # print(f"pred_shapes: {pred_shapes.shape}")
        # print(f"target_shapes: {target_shapes.shape}")
        # print(f"ref_rgbs: {ref_rgbs.shape}")
        # print(f"num_refs: {num_refs}")

        res_pred = self.body_model_train(
            betas=pred_shapes.view(bs, 10),
        )
        res_gt = self.body_model_train(
            betas=target_shapes.view(bs, 10),
        )
        vertices_pred = res_pred.vertices
        vertices_gt = res_gt.vertices
        
        loss = 0.0

        loss_params = F.mse_loss(pred_shapes.contiguous(), target_shapes.detach().contiguous(), reduction="mean")
        loss_vertices = F.mse_loss(vertices_pred.contiguous(), vertices_gt.detach().contiguous(), reduction="sum")

        loss = loss_params + loss_vertices

        self.log("loss_params", loss_params, prog_bar=True, logger=True)
        self.log("loss_vertices", loss_vertices, prog_bar=True, logger=True)

        # self.log("loss", loss, prog_bar=True, logger=True)
        self.check_train(batch)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass


    def generate_images(self, batch, **kwargs):
        num_refs = batch["num_refs"]

        with torch.no_grad():
            ref_img_feats = self.image_encoder(batch["ref_rgbs"])
            ref_img_feats = rearrange(
                ref_img_feats,
                "(B Nr) H W C -> B Nr H W C",
                Nr=num_refs
            )

        pred_shapes = self.shape_predictor(ref_img_feats)
        gt_shapes = batch["target_shapes"]
        res_pred = self.body_model_val(
            betas=pred_shapes.squeeze(1),
        )
        res_gt = self.body_model_val(
            betas=gt_shapes.squeeze(1),
        )
        vertices_pred = res_pred.vertices.float().squeeze(0)
        vertices_gt = res_gt.vertices.float().squeeze(0)
        faces = torch.from_numpy(self.body_model_val.faces.astype(np.int32)).to(vertices_pred.device)
        pred_smpl_normals = self.renderer(
            vertices_pred,
            faces,
        )
        gt_smpl_normals = self.renderer(
            vertices_gt,
            faces,
        )
        return pred_smpl_normals, gt_smpl_normals

    def on_save_checkpoint(self, checkpoint):
        if self.global_rank == 0:
            save_dir = os.path.join(os.path.dirname(self.get_save_dir()), "weights")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                self.shape_predictor.state_dict(),
                os.path.join(save_dir, "shape_predictor.pt")
            )

    def on_check_train(self, batch):
        pass
    
    def validation_step(self, batch, batch_idx):
        pred_smpl_normals, gt_smpl_normals = self.generate_images(batch)

        if (
            self.cfg.check_val_limit_rank > 0
            and self.global_rank < self.cfg.check_val_limit_rank
        ):
            val_img_save_step_dir = os.path.join(self.get_save_dir(), "val", f"step_{self.true_global_step}", f"{self.global_rank}_{batch_idx}")
            os.makedirs(val_img_save_step_dir, exist_ok=True)

            ref_rgbs = batch["ref_rgbs"]
            save_image(ref_rgbs, os.path.join(val_img_save_step_dir, "ref_rgbs.png"))
            save_image(pred_smpl_normals, os.path.join(val_img_save_step_dir, "pred_smplx_normals.png"))
            save_image(gt_smpl_normals, os.path.join(val_img_save_step_dir, "gt_smplx_normals.png"))
        
    
    def on_validation_epoch_end(self):
        pass
    
    def test_step(self, batch, batch_idx):
        pred_smpl_normals, gt_smpl_normals = self.generate_images(batch)

        test_img_save_dir = os.path.join(self.get_save_dir(), "test", f"{self.global_rank}_{batch_idx}")
        os.makedirs(test_img_save_dir, exist_ok=True)

        ref_rgbs = batch["ref_rgbs"]
        save_image(ref_rgbs, os.path.join(test_img_save_dir, "ref_rgbs.png"))
        save_image(pred_smpl_normals, os.path.join(test_img_save_dir, "pred_smplx_normals.png"))
        save_image(gt_smpl_normals, os.path.join(test_img_save_dir, "gt_smplx_normals.png"))
        
    
    def on_test_end(self):
        pass