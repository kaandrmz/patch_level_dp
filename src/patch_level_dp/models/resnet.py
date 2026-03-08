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
                - small_input (bool): Use 3x3 conv instead of 7x7 in first layer (default: False)
                - freeze_mode (str): Freezing strategy - "none", "backbone", or "last_k" (default: "none")
                - freeze_k (int): Number of last layers to train when freeze_mode="last_k" (default: 1)
            
        Returns:
            ResNet model instance with modified final layer
        """
        pretrained = kwargs.get("pretrained", False)
        weights = kwargs.get("weights", None)
        small_input = kwargs.get("small_input", False)
        freeze_mode = kwargs.get("freeze_mode", "none")
        freeze_k = kwargs.get("freeze_k", 1)
        
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
        
        # Replace 7x7 conv with 3x3 conv
        if small_input:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
            print("Using small input mode: 3x3 conv (stride 2, keeping maxpool)")
        
        # Replace final fully connected layer for target num_classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        pretrained_str = "pretrained" if pretrained else "from scratch"
        print(f"Successfully initialized {self.variant} ({pretrained_str}) with {num_classes} output classes.")
        
        # Apply freezing strategy
        if freeze_mode == "backbone":
            self._freeze_backbone(model)
        elif freeze_mode == "last_k":
            self._freeze_except_last_k_layers(model, k=freeze_k)
        elif freeze_mode != "none":
            print(f"Warning: Unknown freeze_mode '{freeze_mode}', training all layers")
        
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
    
    def _freeze_backbone(self, model: nn.Module) -> nn.Module:
        """Freeze all layers except the final classification head.
        
        Args:
            model: ResNet model instance
            
        Returns:
            Model with frozen backbone
        """
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final classification head (fc layer)
        for param in model.fc.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"Frozen all layers except final FC layer")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        return model
    
    def _freeze_except_last_k_layers(self, model: nn.Module, k: int = 1) -> nn.Module:
        """Freeze all layers except the last k residual blocks and FC layer.
        
        Args:
            model: ResNet model instance
            k: Number of last residual blocks to train (1-4)
                k=1: train layer4 + fc
                k=2: train layer3 + layer4 + fc
                k=3: train layer2 + layer3 + layer4 + fc
                k=4: train layer1 + layer2 + layer3 + layer4 + fc
        
        Returns:
            Model with partially frozen backbone
        """
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Define layers to unfreeze based on k
        layers_to_unfreeze = []
        if k >= 4:
            layers_to_unfreeze.append('layer1')
        if k >= 3:
            layers_to_unfreeze.append('layer2')
        if k >= 2:
            layers_to_unfreeze.append('layer3')
        if k >= 1:
            layers_to_unfreeze.append('layer4')
        
        # Unfreeze selected layers
        for layer_name in layers_to_unfreeze:
            layer = getattr(model, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
        
        # Always unfreeze the final FC layer
        for param in model.fc.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        trainable_layers = ', '.join(layers_to_unfreeze + ['fc'])
        print(f"Frozen backbone except: {trainable_layers}")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        return model
