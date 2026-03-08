"""VGG model architecture wrapper for classification."""

from typing import Dict, Any

import torch
import torch.nn as nn
from torchvision import models

from .base import ModelArchitecture


class VGGArchitecture(ModelArchitecture):
    """VGG architecture configuration for classification.
    
    Supports VGG-11, VGG-13, VGG-16, and VGG-19 (with and without batch normalization).
    """
    
    def __init__(self, variant: str = "vgg11"):
        """Initialize VGG architecture.
        
        Args:
            variant: VGG variant ("vgg11", "vgg13", "vgg16", "vgg19",
                     "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn")
        """
        self.variant = variant
        self._model_map = {
            "vgg11": models.vgg11,
            "vgg13": models.vgg13,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
            "vgg11_bn": models.vgg11_bn,
            "vgg13_bn": models.vgg13_bn,
            "vgg16_bn": models.vgg16_bn,
            "vgg19_bn": models.vgg19_bn,
        }
        
        if variant not in self._model_map:
            raise ValueError(f"Unknown VGG variant: {variant}. "
                           f"Available: {list(self._model_map.keys())}")
    
    @property
    def name(self) -> str:
        return self.variant
    
    def create_model(self, num_classes: int, **kwargs) -> nn.Module:
        """Create VGG model instance.
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional parameters:
                - pretrained (bool): Use pretrained ImageNet weights (default: False)
                - weights (str): Specific weights to load (e.g., "IMAGENET1K_V1")
                - dropout_rate (float): Dropout rate in classifier (default: 0.5, VGG default)
                - small_input (bool): Adapt for small images like MNIST (default: False)
            
        Returns:
            VGG model instance with modified classifier
        """
        pretrained = kwargs.get("pretrained", False)
        weights = kwargs.get("weights", None)
        small_input = kwargs.get("small_input", False)
        freeze_mode = kwargs.get("freeze_mode", "none")
        
        # Get the model constructor
        model_fn = self._model_map[self.variant]
        
        # Handle weights parameter
        if weights is not None:
            model = model_fn(weights=weights)
        elif pretrained:
            model = model_fn(weights="IMAGENET1K_V1")
        else:
            model = model_fn(weights=None)
        
        # Adapt for small input images (e.g., MNIST 28x28)
        if small_input:
            features = list(model.features.children())
            pooling_indices = [i for i, layer in enumerate(features) if isinstance(layer, nn.MaxPool2d)]
            if len(pooling_indices) >= 2:
                for idx in sorted(pooling_indices[-2:], reverse=True):
                    features.pop(idx)
                model.features = nn.Sequential(*features)
                print(f"Adapted VGG for small input: removed last 2 max pooling layers ({len(pooling_indices)} -> {len(pooling_indices)-2})")
            
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 28, 28)
                dummy_features = model.features(dummy_input)
                feature_h, feature_w = dummy_features.shape[2], dummy_features.shape[3]
            
            model.avgpool = nn.AdaptiveAvgPool2d((feature_h, feature_w))
            print(f"Adjusted avgpool to output size: ({feature_h}, {feature_w})")
            
            # Calculate new input features for classifier
            new_in_features = 512 * feature_h * feature_w
            
            # Rebuild classifier with correct input size
            model.classifier = nn.Sequential(
                nn.Linear(new_in_features, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes),
            )
            print(f"Adjusted classifier input: {new_in_features} features (512 x {feature_h} x {feature_w})")
        else:
            # Replace final classifier layer for target num_classes
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)
        
        pretrained_str = "pretrained" if pretrained else "from scratch"
        print(f"Successfully initialized {self.variant} ({pretrained_str}) with {num_classes} output classes.")
        
        if freeze_mode != "none":
            print(f"Warning: freeze_mode '{freeze_mode}' not implemented for VGG, training all layers")
        
        return model
    
    def get_model_specific_params(self) -> Dict[str, Any]:
        """Get VGG specific parameters."""
        return {
            "variant": self.variant,
            "pretrained": False,
            "task": "classification",
        }
    
    def get_loss_function_name(self) -> str:
        return "cross_entropy"
    
    def requires_pretrained_backbone(self) -> bool:
        return False
    
    def get_output_stride(self) -> int:
        return 1
    
    def requires_interpolation(self) -> bool:
        return False
