"""ResNet model architecture wrapper for classification."""

from typing import Dict, Any

import torch.nn as nn
from torchvision import models

from .base import ModelArchitecture


class ResNetArchitecture(ModelArchitecture):
    """ResNet architecture configuration for classification.
    
    Supports ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152.
    """
    
    def __init__(self, variant: str = "resnet18"):
        """Initialize ResNet architecture.
        
        Args:
            variant: ResNet variant ("resnet18", "resnet34", "resnet50", 
                     "resnet101", "resnet152")
        """
        self.variant = variant
        self._model_map = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        
        if variant not in self._model_map:
            raise ValueError(f"Unknown ResNet variant: {variant}. "
                           f"Available: {list(self._model_map.keys())}")
    
    @property
    def name(self) -> str:
        return self.variant
    
    def create_model(self, num_classes: int, **kwargs) -> nn.Module:
        """Create ResNet model instance.
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional parameters:
                - pretrained (bool): Use pretrained ImageNet weights (default: False)
                - weights (str): Specific weights to load (e.g., "IMAGENET1K_V1")
            
        Returns:
            ResNet model instance with modified final layer
        """
        pretrained = kwargs.get("pretrained", False)  # Train from scratch by default
        weights = kwargs.get("weights", None)
        
        # Get the model constructor
        model_fn = self._model_map[self.variant]
        
        # Handle weights parameter
        if weights is not None:
            model = model_fn(weights=weights)
        elif pretrained:
            # Use default pretrained weights
            model = model_fn(weights="IMAGENET1K_V1")
        else:
            model = model_fn(weights=None)
        
        # Replace final fully connected layer for target num_classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        pretrained_str = "pretrained" if pretrained else "from scratch"
        print(f"Successfully initialized {self.variant} ({pretrained_str}) with {num_classes} output classes.")
        
        return model
    
    def get_model_specific_params(self) -> Dict[str, Any]:
        """Get ResNet specific parameters."""
        return {
            "variant": self.variant,
            "pretrained": False,  # Train from scratch by default
            "task": "classification",
        }
    
    def get_loss_function_name(self) -> str:
        return "cross_entropy"
    
    def requires_pretrained_backbone(self) -> bool:
        return False  # Train from scratch by default
    
    def get_output_stride(self) -> int:
        # Not applicable for classification
        return 1
    
    def requires_interpolation(self) -> bool:
        # Classification models don't need interpolation
        return False

