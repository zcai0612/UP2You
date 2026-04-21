import os
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models import AutoencoderKL
from diffusers.training_utils import compute_snr
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from ..pipelines.pipeline_mvpuzzle_i2mv_sd21 import UP2YouI2MVSDPipeline
from ..schedulers.scheduling_shift_snr import ShiftSNRScheduler
from ..utils.core import find
from ..utils.typing import *
from .base import BaseSystem
from .utils import encode_prompt, vae_encode

from ..models.encoder.dinov2_wrapper import Dinov2Wrapper
from ..models.feature_aggregator import FeatureAggregator
from ..utils.vis import weight_map_to_heatmap

class MVPuzzleI2MVSDSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        # Model / Adapter
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        pretrained_vae_name_or_path: Optional[str] = None
        pretrained_adapter_name_or_path: Optional[str] = None
        pretrained_unet_name_or_path: Optional[str] = None
        pretrained_feature_aggregator_name_or_path: Optional[str] = None

        use_fp16_vae: bool = True
        use_fp16_clip: bool = True

        init_adapter_kwargs: Dict[str, Any] = field(default_factory=dict)
        init_image_encoder_kwargs: Dict[str, Any] = field(default_factory=dict)
        init_feature_aggregator_kwargs: Dict[str, Any] = field(default_factory=dict)

        max_select_ratio: float = 2.0
        random_select_ratio: float = 0.2
        val_max_select_ratio: float = 1.0
        val_random_select_ratio: float = 0.0

        # Training
        trainable_modules: List[str] = field(default_factory=list)
        train_cond_encoder: bool = True
        train_feature_aggregator: bool = True

        prompt_drop_prob: float = 0.0
        image_drop_prob: float = 0.0
        cond_drop_prob: float = 0.0

        num_reference_image_range: List[int] = field(default_factory=lambda: [3, 7]) # number of reference image range
        front_reference_prob: float = 0.2 # probability of using front view image as reference image

        gradient_checkpointing: bool = False

        # Noise sampler
        noise_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
        noise_offset: float = 0.0
        input_perturbation: float = 0.0
        snr_gamma: Optional[float] = 5.0
        prediction_type: Optional[str] = None
        shift_noise: bool = False
        shift_noise_mode: str = "interpolated"
        shift_noise_scale: float = 1.0

        use_weight_map_loss: bool = False

        # Evaluation
        eval_seed: int = 0
        eval_num_inference_steps: int = 30
        eval_guidance_scale: float = 1.0
        eval_height: int = 1024
        eval_width: int = 1024
    
    cfg: Config

    def configure(self):
        super().configure()

        # Prepare pipeline
        pipeline_kwargs = {}
        if self.cfg.pretrained_vae_name_or_path is not None:
            pipeline_kwargs["vae"] = AutoencoderKL.from_pretrained(
                self.cfg.pretrained_vae_name_or_path
            )
        if self.cfg.pretrained_unet_name_or_path is not None:
            pipeline_kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                self.cfg.pretrained_unet_name_or_path
            )

        pipeline: UP2YouI2MVSDPipeline
        unet = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="unet"
        )

        # Load all required components from the base stable diffusion model
        pipeline = UP2YouI2MVSDPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            unet=unet,
            **pipeline_kwargs,
        )
        pipeline.enable_vae_slicing()
        
        init_adapter_kwargs = OmegaConf.to_container(self.cfg.init_adapter_kwargs)
        
        if "self_attn_processor" in init_adapter_kwargs:
            self_attn_processor = init_adapter_kwargs["self_attn_processor"]
            if self_attn_processor is not None and isinstance(self_attn_processor, str):
                self_attn_processor = find(self_attn_processor)
                init_adapter_kwargs["self_attn_processor"] = self_attn_processor
        pipeline.init_custom_adapter(**init_adapter_kwargs)

        if self.cfg.pretrained_adapter_name_or_path:
            pretrained_path = os.path.dirname(self.cfg.pretrained_adapter_name_or_path)
            adapter_name = os.path.basename(self.cfg.pretrained_adapter_name_or_path)
            pipeline.load_custom_adapter(pretrained_path, weight_name=adapter_name)

        noise_scheduler = DDPMScheduler.from_config(
            pipeline.scheduler.config, **self.cfg.noise_scheduler_kwargs
        )
        if self.cfg.shift_noise:
            noise_scheduler = ShiftSNRScheduler.from_scheduler(
                noise_scheduler,
                shift_mode=self.cfg.shift_noise_mode,
                shift_scale=self.cfg.shift_noise_scale,
                scheduler_class=DDPMScheduler,
            )
        pipeline.scheduler = noise_scheduler

        # Prepare models
        self.pipeline: UP2YouI2MVSDPipeline = pipeline
        self.vae = self.pipeline.vae.to(
            dtype=torch.float16 if self.cfg.use_fp16_vae else torch.float32
        )
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder.to(
            dtype=torch.float16 if self.cfg.use_fp16_clip else torch.float32
        )
        self.cond_encoder = self.pipeline.cond_encoder

        self.unet = self.pipeline.unet
        self.noise_scheduler = self.pipeline.scheduler
        self.feature_selector = self.pipeline.feature_selector
        self.inference_scheduler = DDPMScheduler.from_config(
            self.noise_scheduler.config
        )
        self.pipeline.scheduler = self.inference_scheduler
        if self.cfg.prediction_type is not None:
            self.noise_scheduler.register_to_config(
                prediction_type=self.cfg.prediction_type
            )


        init_image_encoder_kwargs = OmegaConf.to_container(self.cfg.init_image_encoder_kwargs)
        init_feature_aggregator_kwargs = OmegaConf.to_container(self.cfg.init_feature_aggregator_kwargs)

        self.image_encoder = Dinov2Wrapper(
            **init_image_encoder_kwargs
        )
        self.feature_aggregator = FeatureAggregator(
            **init_feature_aggregator_kwargs
        )

        if self.cfg.pretrained_feature_aggregator_name_or_path:
            self.feature_aggregator.load_state_dict(
                torch.load(self.cfg.pretrained_feature_aggregator_name_or_path)
            )
        
        # Prepare trainable / non-trainable modules
        trainable_modules = self.cfg.trainable_modules
        if trainable_modules and len(trainable_modules) > 0:
            self.unet.requires_grad_(False)
            for name, module in self.unet.named_modules():
                for trainable_module in trainable_modules:
                    if trainable_module in name:
                        module.requires_grad_(True)
        else:
            self.unet.requires_grad_(True)
        self.cond_encoder.requires_grad_(self.cfg.train_cond_encoder)
        self.feature_aggregator.requires_grad_(self.cfg.train_feature_aggregator)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Others
        # Prepare gradient checkpointing
        if self.cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def preprocess_reference_images(
        self, 
        ref_rgbs: Tensor, 
        ref_alphas: Tensor,
        target_rgbs: Tensor,    
        target_alphas: Tensor,
        num_refs: int,
        num_views: int,
    ):
        front_ref_prob = self.cfg.front_reference_prob
        if random.random() < front_ref_prob:
            H_ref, W_ref = ref_rgbs.shape[-2:]
            target_rgbs_ = rearrange(target_rgbs, "(B Nv) C H W -> B Nv C H W", Nv=num_views)
            target_alphas_ = rearrange(target_alphas, "(B Nv) H W -> B Nv H W", Nv=num_views)
            ref_rgbs_ = target_rgbs_[:, 0:1, :, :, :]
            ref_alphas_ = target_alphas_[:, 0:1, :, :]
            ref_rgbs_ = rearrange(ref_rgbs_, "B 1 C H W -> B C H W")
            ref_rgbs_ = F.interpolate(ref_rgbs_, size=(H_ref, W_ref), mode="bilinear")
            ref_alphas_ = rearrange(ref_alphas_, "B 1 H W -> B H W")
            ref_alphas_ = F.interpolate(ref_alphas_.unsqueeze(1), size=(H_ref, W_ref), mode="nearest").squeeze(1)
            return ref_rgbs_, ref_alphas_, 1
        else:
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
        noisy_latents: Tensor,
        target_pose_camera: Tensor, # target pose
        timesteps: Tensor,
        ref_latents: Tensor,
        ref_rgbs: Tensor,
        ref_alphas: Tensor,
        prompts: List[str],
        num_views: int,
        num_refs: int,
    ):
        bsz = noisy_latents.shape[0]
        b_samples = bsz // num_views
        bsz_ref = ref_latents.shape[0]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            input_ids = self.tokenizer(
                prompts,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].to(noisy_latents.device)
            encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[
                0
            ].float()

        prompt_drop_mask = (
            torch.rand(b_samples, device=noisy_latents.device)
            < self.cfg.prompt_drop_prob
        )
        image_drop_mask = (
            torch.rand(b_samples, device=noisy_latents.device)
            < self.cfg.image_drop_prob
        )
        cond_drop_mask = (
            torch.rand(b_samples, device=noisy_latents.device) < self.cfg.cond_drop_prob
        )

        prompt_drop_mask = prompt_drop_mask | cond_drop_mask
        image_drop_mask = image_drop_mask | cond_drop_mask

        image_drop_mask_ref = image_drop_mask.repeat_interleave(num_views, dim=0)

        encoder_hidden_states[prompt_drop_mask] = 0.0

        # Process reference latents to obtain reference features
        with torch.no_grad():
            ref_img_feats = self.image_encoder(ref_rgbs)
            ref_img_feats = rearrange(
                ref_img_feats,
                "(B Nr) H W C -> B Nr H W C",
                B=b_samples,
                Nr=num_refs
            )
        target_pose_imgs = rearrange(target_pose_camera, "(B Nv) C H W -> B Nv H W C", B=b_samples, Nv=num_views)
        ref_alphas = rearrange(ref_alphas, "(B Nr) H W -> B Nr H W", B=b_samples, Nr=num_refs)
        weight_maps = self.feature_aggregator(
            target_pose_imgs=target_pose_imgs,
            ref_img_feats=ref_img_feats,
            ref_alphas=ref_alphas,
        )

        # print(weight_maps[0].mean(), weight_maps[0].max(), weight_maps[0].min(), weight_maps[0].requires_grad)

        with torch.no_grad():
            ref_timesteps = torch.zeros_like(timesteps[:1]).repeat_interleave(bsz_ref, dim=0)
            ref_hidden_states = {}
            self.unet(
                ref_latents,
                ref_timesteps,
                encoder_hidden_states=encoder_hidden_states.repeat_interleave(num_refs, dim=0),
                cross_attention_kwargs={
                    "cache_hidden_states": ref_hidden_states,
                    "use_mv": False,
                    "use_ref": False,
                },
                return_dict=False,
            )
        for k, v in ref_hidden_states.items():
            _, N, _ = v.shape
            Hr = Wr = int(N ** 0.5)
            v_ = rearrange(v, "(B Nr) (Hr Wr) C -> B Nr Hr Wr C", B=b_samples, Nr=num_refs, Hr=Hr, Wr=Wr)
            v_ = self.feature_selector.forward(
                ref_img_feats=v_,
                weight_maps=weight_maps,
                # ref_img_masks=ref_alphas,
                max_select_ratio=self.cfg.max_select_ratio,
                random_select_ratio=self.cfg.random_select_ratio,
            )
            v_[image_drop_mask_ref] = 0.0
            ref_hidden_states[k] = v_

        conditioning_features = self.cond_encoder(target_pose_camera)

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states.repeat_interleave(num_views, dim=0),
            down_intrablock_additional_residuals=conditioning_features,
            cross_attention_kwargs={
                "ref_hidden_states": ref_hidden_states,
                "num_views": num_views,
            },
        ).sample

        return {
            "noise_pred": noise_pred,
            "weight_maps": weight_maps, # [B, Nr, H, W, 1] * Nv
        }
    
    def training_step(self, batch, batch_idx):
        num_views = batch["num_views"]
        num_refs = batch["num_refs"]

        ref_rgbs = batch["ref_rgbs"]
        ref_alphas = batch["ref_alphas"]
        target_rgbs = batch["target_rgbs"]
        target_alphas = batch["target_alphas"]



        ref_rgbs, ref_alphas, num_refs = self.preprocess_reference_images(
            ref_rgbs, ref_alphas, target_rgbs, target_alphas, num_refs, num_views
        )

        H_r, W_r = ref_rgbs.shape[-2:]

        vae_max_slice = 8
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            latents = []
            for i in range(0, target_rgbs.shape[0], vae_max_slice):
                latents.append(
                    vae_encode(
                        self.vae,
                        target_rgbs[i : i + vae_max_slice].to(self.vae.dtype) * 2 - 1,
                        sample=True,
                        apply_scale=True,
                    ).float()
                )
            latents = torch.cat(latents, dim=0)

        ref_vae_max_slice = 8
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            ref_latents = []
            for i in range(0, ref_rgbs.shape[0], ref_vae_max_slice):
                ref_latents.append(
                    vae_encode(
                        self.vae,
                        ref_rgbs[i : i + ref_vae_max_slice].to(self.vae.dtype) * 2 - 1,
                        sample=True,
                        apply_scale=True,
                    ).float()
                )
            ref_latents = torch.cat(ref_latents, dim=0)


        bsz = latents.shape[0]
        b_samples = bsz // num_views

        noise = torch.randn_like(latents)
        if self.cfg.noise_offset is not None:
            # # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        noise_mask = (
            batch["noise_mask"]
            if "noise_mask" in batch
            else torch.ones((bsz,), dtype=torch.bool, device=latents.device)
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (b_samples,),
            device=latents.device,
            dtype=torch.long,
        )
        timesteps = timesteps.repeat_interleave(num_views)
        timesteps[~noise_mask] = 0

        if self.cfg.input_perturbation is not None:
            new_noise = noise + self.cfg.input_perturbation * torch.randn_like(noise)
            noisy_latents = self.noise_scheduler.add_noise(
                latents, new_noise, timesteps
            )
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noisy_latents[~noise_mask] = latents[~noise_mask]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unsupported prediction type {self.noise_scheduler.config.prediction_type}"
            )


        model_pred = self(
            noisy_latents, 
            batch["target_poses_camera"], 
            timesteps, 
            ref_latents, 
            ref_rgbs, 
            ref_alphas,
            batch["prompts"], 
            num_views, 
            num_refs  
        )

        noise_pred = model_pred["noise_pred"]
        weight_maps = model_pred["weight_maps"] # [B, Nr, H, W, 1] * Nv
        
        noise_pred = noise_pred[noise_mask]
        target = target[noise_mask]

        loss = 0.0

        if self.cfg.snr_gamma is None:
            loss_sd = F.mse_loss(noise_pred, target, reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )

            loss_sd = F.mse_loss(noise_pred, target, reduction="none")
            loss_sd = loss_sd.mean(dim=list(range(1, len(loss_sd.shape)))) * mse_loss_weights
            loss_sd = loss_sd.mean()
        
        if self.cfg.use_weight_map_loss:
            weight_maps = torch.cat(weight_maps, dim=0).squeeze(-1)
            weight_maps = F.interpolate(weight_maps, size=(H_r, W_r), mode="bilinear", align_corners=True) # B*Nv Nr H W
            ref_alphas = rearrange(ref_alphas, "(B Nr) H W -> B Nr H W", Nr=num_refs)
            ref_alphas_expand = ref_alphas.repeat_interleave(num_views, dim=0) # B*Nv Nr H W

            bkg_weight_maps = weight_maps * (1.0 - ref_alphas_expand)
            
            loss_weight_map = bkg_weight_maps.sum(dim=1).mean()
        else:
            loss_weight_map = 0.0

        loss += loss_sd
        loss += loss_weight_map

        self.log("loss", loss_sd, prog_bar=True, logger=True)
        # self.log("loss_weight_map", loss_weight_map, prog_bar=True, logger=True)
        # will execute self.on_check_train every self.cfg.check_train_every_n_steps steps
        self.check_train(batch)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    def generate_images(self, batch, **kwargs):    
        num_views = batch["num_views"]
        num_refs = batch["num_refs"]
        ref_alphas = batch["ref_alphas"]

        with torch.no_grad():
            ref_img_feats = self.image_encoder(batch["ref_rgbs"])
            ref_img_feats = rearrange(
                ref_img_feats,
                "(B Nr) H W C -> B Nr H W C",
                Nr=num_refs
            )
        target_pose_imgs = rearrange(batch["target_poses_camera"], "(B Nv) C H W -> B Nv H W C", Nv=num_views)
        ref_alphas = rearrange(ref_alphas, "(B Nr) H W -> B Nr H W", Nr=num_refs)
        weight_maps = self.feature_aggregator(
            target_pose_imgs=target_pose_imgs,
            ref_img_feats=ref_img_feats,
            ref_alphas=ref_alphas,
        )

        return self.pipeline(
            prompt=batch["prompts"],
            control_image=batch["target_poses_camera"],
            num_images_per_prompt=batch["num_views"],
            generator=torch.Generator(device=self.device).manual_seed(
                self.cfg.eval_seed
            ),
            num_inference_steps=self.cfg.eval_num_inference_steps,
            guidance_scale=self.cfg.eval_guidance_scale,
            height=self.cfg.eval_height,
            width=self.cfg.eval_width,
            reference_rgbs=batch["ref_rgbs"],
            weight_maps=weight_maps,
            # ref_img_masks=ref_alphas,
            max_select_ratio=self.cfg.val_max_select_ratio,
            random_select_ratio=self.cfg.val_random_select_ratio,
            output_type="pt",
        ).images, weight_maps

    def on_save_checkpoint(self, checkpoint):
        if self.global_rank == 0:
            save_dir = os.path.join(os.path.dirname(self.get_save_dir()), "weights")
            os.makedirs(save_dir, exist_ok=True)
            self.pipeline.save_custom_adapter(
                save_dir,
                "custom_adapter.safetensors",
                safe_serialization=True,
                include_keys=self.cfg.trainable_modules,
            )
            torch.save(
                self.feature_aggregator.state_dict(),
                os.path.join(save_dir, "feature_aggregator.pt")
            )

    def on_check_train(self, batch):
        train_img_save_step_dir = os.path.join(self.get_save_dir(), "train", f"step_{self.true_global_step}")
        os.makedirs(train_img_save_step_dir, exist_ok=True)

        target_poses_camera = batch["target_poses_camera"]
        ref_rgbs = batch["ref_rgbs"]
        target_rgbs = batch["target_rgbs"]

        save_image(target_poses_camera, os.path.join(train_img_save_step_dir, "target_poses_camera.png"))
        save_image(ref_rgbs, os.path.join(train_img_save_step_dir, "ref_rgbs.png"))
        save_image(target_rgbs, os.path.join(train_img_save_step_dir, "target_rgbs.png"))



    def validation_step(self, batch, batch_idx):
        images, weight_maps = self.generate_images(batch)
        ref_img_masks = batch["ref_alphas"]
        ref_img_masks = rearrange(ref_img_masks, "(B Nr) H W -> B Nr H W", Nr=batch["num_refs"])
        # 使用新的热力图转换功能
        weight_map_heatmaps = weight_map_to_heatmap(
            weight_maps, 
            ref_img_masks=ref_img_masks,
            colormap="jet", 
            normalize=True,
            temperature=1.0,
            return_tensor=True
        )
        
        if (
            self.cfg.check_val_limit_rank > 0
            and self.global_rank < self.cfg.check_val_limit_rank
        ):
            val_img_save_step_dir = os.path.join(self.get_save_dir(), "val", f"step_{self.true_global_step}", f"{self.global_rank}_{batch_idx}")
            os.makedirs(val_img_save_step_dir, exist_ok=True)

            target_poses_camera = batch["target_poses_camera"]
            ref_rgbs = batch["ref_rgbs"]

            save_image(target_poses_camera, os.path.join(val_img_save_step_dir, "target_poses_camera.png"))
            save_image(ref_rgbs, os.path.join(val_img_save_step_dir, "ref_rgbs.png"))
            save_image(images, os.path.join(val_img_save_step_dir, "gen_images.png"))
            for pose_idx, weight_map_heatmap in enumerate(weight_map_heatmaps):
                weight_map_heatmap = rearrange(weight_map_heatmap, "B Nr H W C -> (B Nr) C H W")
                save_image(weight_map_heatmap, os.path.join(val_img_save_step_dir, f"weight_map_heatmap_{pose_idx}.png"))

    def on_validation_epoch_end(self):
        pass
    
    def test_step(self, batch, batch_idx):
        images, weight_maps = self.generate_images(batch)
        ref_img_masks = batch["ref_alphas"]
        ref_img_masks = rearrange(ref_img_masks, "(B Nr) H W -> B Nr H W", Nr=batch["num_refs"])
        
        # 使用新的热力图转换功能
        weight_map_heatmaps = weight_map_to_heatmap(
            weight_maps, 
            ref_img_masks=ref_img_masks,
            colormap="jet", 
            normalize=True,
            temperature=1.0,
            return_tensor=True
        )

        test_img_save_dir = os.path.join(self.get_save_dir(), "test", f"{self.global_rank}_{batch_idx}")
        os.makedirs(test_img_save_dir, exist_ok=True)

        target_poses_camera = batch["target_poses_camera"]
        ref_rgbs = batch["ref_rgbs"]

        save_image(target_poses_camera, os.path.join(test_img_save_dir, "target_poses_camera.png"))
        save_image(ref_rgbs, os.path.join(test_img_save_dir, "ref_rgbs.png"))
        save_image(images, os.path.join(test_img_save_dir, "gen_images.png"))
        
        for pose_idx, weight_map_heatmap in enumerate(weight_map_heatmaps):
            weight_map_heatmap = rearrange(weight_map_heatmap, "B Nr C H W -> (B Nr) C H W")
            save_image(weight_map_heatmap, os.path.join(test_img_save_dir, f"weight_map_heatmap_{pose_idx}.png"))


    def on_test_end(self):
        pass
