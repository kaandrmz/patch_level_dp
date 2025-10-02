"""Base classes for model architectures."""
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn


class ModelArchitecture(ABC):
    """Abstract base class for model architectures."""
    
    @property
    @abstractmethod  
    def name(self) -> str:
        """Model architecture name identifier."""
        pass
    
    @abstractmethod
    def create_model(self, num_classes: int, **kwargs) -> nn.Module:
        """Create model instance.
        
        Args:
            num_classes: Number of output classes
            **kwargs: Additional model-specific parameters
            
        Returns:
            PyTorch model instance
        """
        pass
    
    @abstractmethod
    def get_model_specific_params(self) -> Dict[str, Any]:
        """Get model-specific parameters and configurations.
        
        Returns:
            Dictionary with model-specific settings
        """
        pass
    
    @abstractmethod
    def get_loss_function_name(self) -> str:
        """Get the appropriate loss function name for this model.
        
        Returns:
            Loss function name (e.g., "cross_entropy", "nll_loss")
        """
        pass
    
    def requires_pretrained_backbone(self) -> bool:
        """Whether this model requires pretrained backbone weights.
        
        Returns:
            True if pretrained backbone is recommended
        """
        return False
    
    def get_output_stride(self) -> int:
        """Get the output stride for segmentation models.
        
        Returns:
            Output stride value
        """
        return 8
    
    def requires_interpolation(self) -> bool:
        """Whether model output needs interpolation to match target size.
        
        Returns:
            True if interpolation is needed
        """
        return True
