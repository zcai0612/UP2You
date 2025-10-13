import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from einops import rearrange

class DepthwiseSeparableConv(nn.Module):
    """depthwise separable convolution, used to reduce the number of parameters and computational complexity"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LightweightResBlock(nn.Module):
    """lightweight residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, stride=1)
        
        # 1x1 convolution for residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return F.relu(out, inplace=True)

class PoseEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        downsample_ratio=16,
        out_channels=768,
        image_size=None,
    ):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        
        # initial convolution layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # lightweight ResNet layer
        self.layer1 = self._make_layer(32, 64, 2, stride=1)    # 1/2
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 1/4
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 1/8
        
        # decide whether to use more layers based on downsample_ratio
        if downsample_ratio == 16:
            self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 1/16
            final_channels = 512
        else:
            self.layer4 = None
            final_channels = 256
        
        # global average pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final_proj = nn.Sequential(
            nn.Conv2d(final_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        
        if image_size is not None:
            self.resize = Resize((image_size, image_size))
        else:
            self.resize = None

        # weight initialization
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(LightweightResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(LightweightResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, pose_img):
        if self.resize is not None:
            pose_img = self.resize(pose_img)

        # pose_img: [B, C, H, W]
        x = self.stem(pose_img)      # [B, 32, H/2, W/2]
        x = self.layer1(x)           # [B, 64, H/2, W/2]
        x = self.layer2(x)           # [B, 128, H/4, W/4]
        x = self.layer3(x)           # [B, 256, H/8, W/8]
        
        if self.layer4 is not None:
            x = self.layer4(x)       # [B, 512, H/16, W/16]
        
        # keep spatial dimensions for feature extraction
        features = self.final_proj(x)  # [B, out_channels, H/ratio, W/ratio]
        features = rearrange(features, "b c h w -> b (h w) c")
        
        return features
    