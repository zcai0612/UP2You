# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroConv2d(nn.Module):
    """
    Zero Convolution Module
    
    This is a special convolution layer that outputs zero values at the initial training stage.
    By initializing weights and biases to zero, the module does not affect the input at the 
    beginning of training, and gradually learns useful features as training progresses.
    
    Commonly used in:
    - Conditional injection in Diffusion models
    - ControlNet architectures
    - Networks requiring progressive training
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size, default 1x1 convolution
            stride: Stride
            padding: Padding
            dilation: Dilation rate
            groups: Number of groups for grouped convolution
            bias: Whether to use bias
            padding_mode: Padding mode
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        
        # Zero initialize weights and biases
        self._zero_init()
    
    def _zero_init(self):
        """Initialize weights and biases to zero"""
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, out_channels, H', W')
        """
        return self.conv(x)


class ZeroConv1d(nn.Module):
    """
    1D Zero Convolution Module
    
    Zero convolution for sequence data, commonly used in time series or text processing
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        
        self._zero_init()
    
    def _zero_init(self):
        """Initialize weights and biases to zero"""
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, L)
            
        Returns:
            Output tensor of shape (B, out_channels, L')
        """
        return self.conv(x)


class ZeroLinear(nn.Module):
    """
    Zero Linear Layer
    
    Zero-initialized version of linear layer, used for fully connected layers requiring zero initialization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        
        # self._zero_init()
    
    def _zero_init(self):
        """Initialize weights and biases to zero"""
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        return self.linear(x)


class AdaptiveZeroConv2d(nn.Module):
    """
    Adaptive Zero Convolution Module
    
    Zero convolution with learnable scaling factor, allowing more flexible training control
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        init_scale: float = 0.0
    ):
        """
        Args:
            init_scale: Initial value of scaling factor, default 0.0 (completely zero output)
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        
        # Learnable scaling factor
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        
        # Zero initialize weights and biases
        self._zero_init()
    
    def _zero_init(self):
        """Initialize weights and biases to zero"""
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, out_channels, H', W')
        """
        return self.conv(x) * self.scale


class ConditionZeroConv2d(nn.Module):
    """
    Conditional Zero Convolution Module
    
    Used for architectures requiring conditional input such as ControlNet
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_channels: Optional[int] = None,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        """
        Args:
            condition_channels: Number of channels for conditional input, if None then no conditional input is used
        """
        super().__init__()
        
        # Main convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        
        # Conditional convolution layer (if needed)
        self.condition_conv = None
        if condition_channels is not None:
            self.condition_conv = nn.Conv2d(
                in_channels=condition_channels,
                out_channels=out_channels,
                kernel_size=1,  # 1x1 convolution for conditional injection
                bias=False
            )
            nn.init.zeros_(self.condition_conv.weight)
        
        # Zero initialize main convolution layer
        self._zero_init()
    
    def _zero_init(self):
        """Initialize weights and biases to zero"""
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Main input tensor of shape (B, C, H, W)
            condition: Conditional input tensor of shape (B, condition_channels, H, W)
            
        Returns:
            Output tensor of shape (B, out_channels, H', W')
        """
        output = self.conv(x)
        
        if condition is not None and self.condition_conv is not None:
            # Pass conditional input through convolution layer and add to main output
            condition_output = self.condition_conv(condition)
            # Interpolate if spatial dimensions don't match
            if condition_output.shape[-2:] != output.shape[-2:]:
                condition_output = F.interpolate(
                    condition_output, 
                    size=output.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            output = output + condition_output
        
        return output


# Convenience functions
def zero_conv2d(in_channels: int, out_channels: int, **kwargs) -> ZeroConv2d:
    """Convenience function to create 2D zero convolution layer"""
    return ZeroConv2d(in_channels, out_channels, **kwargs)


def zero_conv1d(in_channels: int, out_channels: int, **kwargs) -> ZeroConv1d:
    """Convenience function to create 1D zero convolution layer"""
    return ZeroConv1d(in_channels, out_channels, **kwargs)


def zero_linear(in_features: int, out_features: int, **kwargs) -> ZeroLinear:
    """Convenience function to create zero linear layer"""
    return ZeroLinear(in_features, out_features, **kwargs)
