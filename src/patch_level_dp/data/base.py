"""Base classes for dataset configurations."""

from abc import ABC, abstractmethod
from typing import Tuple, Type, Any, Optional

from torch.utils.data import Dataset


class DatasetConfig(ABC):
    """Abstract base class for dataset configurations."""
    
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
    def crop_size(self) -> int:
        """Crop size for training."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of segmentation classes."""
        pass
    
    @property
    @abstractmethod
    def epoch_size(self) -> int:
        """Number of training images in the dataset."""
        pass
    
    @abstractmethod
    def get_dataset_class(self) -> Type[Dataset]:
        """Get the dataset class."""
        pass
    
    @abstractmethod
    def get_transforms(self, mode: str = "train", crop_size: Optional[int] = None) -> Any:
        """Get transforms for the specified mode."""
        pass
    
    @abstractmethod
    def create_dataset(self, root: str, split: str, transform: Any = None, **kwargs) -> Dataset:
        """Create dataset instance."""
        pass
