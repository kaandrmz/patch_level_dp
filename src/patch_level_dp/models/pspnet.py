"""PSPNet model architecture wrapper."""

from typing import Dict, Any

import torch.nn as nn

from patch_level_dp.models.architectures.pspnet.pspnet import PSPNet

from .base import ModelArchitecture


class PSPNetArchitecture(ModelArchitecture):
    """PSPNet architecture configuration."""
    
    @property
    def name(self) -> str:
        return "pspnet"
    
    def create_model(self, num_classes: int, **kwargs) -> nn.Module:
        """Create PSPNet model instance.
        
        Args:
            num_classes: Number of segmentation classes
            **kwargs: Additional parameters (sizes, psp_size, backend, pretrained)
            
        Returns:
            PSPNet model instance
        """
        sizes = kwargs.get("sizes", (1, 2, 3, 6))
        psp_size = kwargs.get("psp_size", 2048)
        backend = kwargs.get("backend", "resnet101")
        pretrained = kwargs.get("pretrained", True)
        
        model = PSPNet(
            n_classes=num_classes, 
            sizes=sizes,
            psp_size=psp_size,
            backend=backend, 
            pretrained=pretrained
        )
        
        print(f"Successfully initialized PSPNet with {backend} backbone.")
        
        return model
    
    def get_model_specific_params(self) -> Dict[str, Any]:
        """Get PSPNet specific parameters."""
        return {
            "sizes": (1, 2, 3, 6),
            "psp_size": 2048,
            "backend": "resnet101",
            "pretrained": True,
        }
    
    def get_loss_function_name(self) -> str:
        return "nll_loss"
    
    def requires_pretrained_backbone(self) -> bool:
        return True
    
    def get_output_stride(self) -> int:
        return 8
    
    def requires_interpolation(self) -> bool:
        return True