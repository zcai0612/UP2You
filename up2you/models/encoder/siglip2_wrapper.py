import torch
import torch.nn as nn
from transformers import AutoModel

# channels: 1152
class Siglip2Wrapper(nn.Module):
    def __init__(
        self, 
        device: str = "cuda", 
        dtype: torch.dtype = torch.float16, 
        model_name: str = "facebook/siglip-2-base",
        freeze: bool = True,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(model_name).to(device, dtype=dtype)
        if freeze:
            self._freeze()
    
        self.vit_mean = torch.tensor([0.5, 0.5, 0.5], device=self.device, dtype=self.dtype)[:, None, None]
        self.vit_std = torch.tensor([0.5, 0.5, 0.5], device=self.device, dtype=self.dtype)[:, None, None]

    def _freeze(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        image_input = image.to(self.device, dtype=self.dtype)
        image = (image_input - self.vit_mean) / self.vit_std
        return self.model.get_image_features(image).float()