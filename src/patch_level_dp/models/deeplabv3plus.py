"""DeepLabV3+ model architecture wrapper."""

from typing import Dict, Any

import torch.nn as nn

from patch_level_dp.models.architectures.deeplabv3plus.network import modeling

from .base import ModelArchitecture


class DeepLabV3PlusArchitecture(ModelArchitecture):
    """DeepLabV3+ architecture configuration."""
    
    @property
    def name(self) -> str:
        return "deeplabv3plus"
    
    def create_model(self, num_classes: int, **kwargs) -> nn.Module:
        """Create DeepLabV3+ model instance.
        
        Args:
            num_classes: Number of segmentation classes
            **kwargs: Additional parameters (output_stride, pretrained_backbone)
            
        Returns:
            DeepLabV3+ model instance
        """
        output_stride = kwargs.get("output_stride", self.get_output_stride())
        pretrained_backbone = kwargs.get("pretrained_backbone", False)
        model = modeling.deeplabv3plus_resnet101(
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone
        )
        return model
    
    def get_model_specific_params(self) -> Dict[str, Any]:
        """Get DeepLabV3+ specific parameters."""
        return {
            "output_stride": 8,
            "pretrained_backbone": False,
            "backbone": "resnet101",
            "decoder_channels": 256,
            "atrous_rates": [12, 24, 36],
        }
    
    def get_loss_function_name(self) -> str:
        return "cross_entropy"
    
    def requires_pretrained_backbone(self) -> bool:
        return True
    
    def get_output_stride(self) -> int:
        return 8
    
    def requires_interpolation(self) -> bool:
        return False