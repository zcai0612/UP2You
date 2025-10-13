import torch
import torch.nn as nn
from torchvision.transforms import Resize
from einops import rearrange

model_dims = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class Dinov2Wrapper(nn.Module):
    def __init__(
        self, 
        device='cuda', 
        dtype=torch.float16, 
        model_name='dinov2_vitb14', 
        freeze=True,
        image_size=None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = self._build_dinov2(model_name).to(device, dtype=dtype) # better use float16 to enable xformers
        if freeze:
            self._freeze()

        # DINOv2 preprocessing (ImageNet normalization)
        self.vit_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype)[:, None, None]
        self.vit_std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype)[:, None, None]
        self.feat_dim = model_dims.get(model_name, 768)

        self.image_size = image_size
        if self.image_size is not None:
            self.resize_transform = Resize((self.image_size, self.image_size))

    def _freeze(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def _build_dinov2(self, model_name: str):
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        return model

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        image_input = image.to(self.device, dtype=self.dtype)
        if self.image_size is not None:
            image_input = self.resize_transform(image_input)
        image_input = (image_input - self.vit_mean) / self.vit_std

        out = self.model.forward_features(image_input)
        # DINOv2 returns dict with 'x_norm_patchtokens' (patch tokens) and other features
        if isinstance(out, dict):
            if 'x_norm_patchtokens' in out:
                features = out['x_norm_patchtokens']
            else:
                features = out.get('last_hidden_state', next(iter(out.values())))
        else:
            features = out

        B, N, C = features.shape
        H = W = int(N ** 0.5)
        features = rearrange(features, "B (H W) C -> B H W C", H=H, W=W)
        features = features.float()
        return features

    @torch.no_grad()
    def get_cls_embeds(self, image: torch.Tensor):
        image_input = image.to(self.device, dtype=self.dtype)
        if self.image_size is not None:
            image_input = self.resize_transform(image_input)
        image_input = (image_input - self.vit_mean) / self.vit_std

        out = self.model.forward_features(image_input)
    
        features = out.get('last_hidden_state', next(iter(out.values())))
        return features