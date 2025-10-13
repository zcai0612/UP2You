import os
from typing import List

import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange
import torch.nn as nn

class CLIPWrapper(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        model_name: str = "/mnt/home/public/public_nas/weights/clip/image_encoder",
        freeze: bool = True,
    ):
        super().__init__()
        self.device = device

        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        if freeze:
            self._freeze()

    def _freeze(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        pil_images = []
        for i in range(image.shape[0]):
            pil_images.append(Image.fromarray((image[i] * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)))
        clip_image = self.clip_image_processor(images=pil_images, return_tensors="pt").pixel_values
        features = self.model(clip_image.to(self.device, dtype=torch.float16), output_hidden_states=True).hidden_states[-2]
        features = features[:, 1:, :]

        B, N, C = features.shape
        H = W = int(N ** 0.5)
        features = rearrange(features, "B (H W) C -> B H W C", H=H, W=W)
        features = features.float()
        return features
