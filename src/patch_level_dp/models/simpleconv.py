"""Simple ConvNet model architecture for classification."""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelArchitecture


class SimpleConvNet(nn.Module):
    """A simple convolutional neural network for classification.
    
    Architecture:
        - 3 convolutional blocks (conv -> batchnorm -> relu -> maxpool)
        - 2 fully connected layers
        - Dropout for regularization
    
    Designed to work with various input sizes (e.g., 28x28 MNIST, 224x224 DTD).
    """
    
    def __init__(self, num_classes: int, in_channels: int = 3, 
                 base_channels: int = 32, input_size: int = 28):
        """Initialize SimpleConvNet.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            base_channels: Base number of channels (doubled after each block)
            input_size: Expected input image size (height/width, assumes square)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.input_size = input_size
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate flattened size after conv layers
        # Each maxpool reduces size by 2, so after 3 pools: input_size / 8
        final_size = input_size // 8
        if final_size < 1:
            final_size = 1
        
        flat_features = base_channels * 4 * final_size * final_size
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self._flat_features = flat_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class SimpleConvNetArchitecture(ModelArchitecture):
    """SimpleConvNet architecture configuration for classification."""
    
    def __init__(self, input_size: int = 28, base_channels: int = 32):
        """Initialize SimpleConvNet architecture.
        
        Args:
            input_size: Expected input image size (default: 28 for MNIST)
            base_channels: Base number of channels (default: 32)
        """
        self.input_size = input_size
        self.base_channels = base_channels
    
    @property
    def name(self) -> str:
        return "simpleconv"
    
    def create_model(self, num_classes: int, **kwargs) -> nn.Module:
        """Create SimpleConvNet model instance.
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional parameters:
                - in_channels (int): Number of input channels (default: 3)
                - input_size (int): Input image size (overrides init value)
                - base_channels (int): Base channels (overrides init value)
            
        Returns:
            SimpleConvNet model instance
        """
        in_channels = kwargs.get("in_channels", 3)
        input_size = kwargs.get("input_size", self.input_size)
        base_channels = kwargs.get("base_channels", self.base_channels)
        
        model = SimpleConvNet(
            num_classes=num_classes,
            in_channels=in_channels,
            base_channels=base_channels,
            input_size=input_size
        )
        
        print(f"Successfully initialized SimpleConvNet with {num_classes} output classes, "
              f"input_size={input_size}, base_channels={base_channels}")
        
        return model
    
    def get_model_specific_params(self) -> Dict[str, Any]:
        """Get SimpleConvNet specific parameters."""
        return {
            "input_size": self.input_size,
            "base_channels": self.base_channels,
            "task": "classification",
        }
    
    def get_loss_function_name(self) -> str:
        return "cross_entropy"
    
    def requires_pretrained_backbone(self) -> bool:
        # SimpleConvNet has no pretrained weights
        return False
    
    def get_output_stride(self) -> int:
        # Not applicable for classification
        return 1
    
    def requires_interpolation(self) -> bool:
        # Classification models don't need interpolation
        return False




