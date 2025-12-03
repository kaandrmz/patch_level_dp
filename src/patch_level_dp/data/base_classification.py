"""Base classes for classification dataset configurations."""

from abc import ABC, abstractmethod
from typing import Tuple, Type, Any, Optional

from torch.utils.data import Dataset


class ClassificationDatasetConfig(ABC):
    """Abstract base class for classification dataset configurations.
    
    Unlike segmentation datasets, classification datasets return (image, label)
    where label is a scalar class index, not an image mask.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        pass
    
    @property  
    @abstractmethod
    def image_size(self) -> Tuple[int, int]:
        """Image size as (height, width)."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classification classes."""
        pass
    
    @property
    @abstractmethod
    def epoch_size(self) -> int:
        """Number of training images in the dataset."""
        pass
    
    @property
    def num_channels(self) -> int:
        """Number of input channels (default: 3 for RGB)."""
        return 3
    
    @abstractmethod
    def get_transforms(self, mode: str = "train") -> Any:
        """Get transforms for the specified mode.
        
        Args:
            mode: One of "train", "val", or "test"
            
        Returns:
            Transform composition for classification (applied to image only)
        """
        pass
    
    @abstractmethod
    def create_dataset(self, root: str, split: str, transform: Any = None, **kwargs) -> Dataset:
        """Create dataset instance.
        
        Args:
            root: Root directory for the dataset
            split: Dataset split ("train", "val", or "test")
            transform: Optional transform to apply
            **kwargs: Additional dataset-specific arguments
            
        Returns:
            PyTorch Dataset instance
        """
        pass

