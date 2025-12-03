"""MNIST dataset configuration for classification."""

from typing import Tuple, Type, Any, Optional

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from .base_classification import ClassificationDatasetConfig


class MNISTConfig(ClassificationDatasetConfig):
    """Configuration for MNIST dataset."""
    
    @property
    def name(self) -> str:
        return "mnist"
    
    @property  
    def image_size(self) -> Tuple[int, int]:
        return (28, 28)
    
    @property
    def crop_size(self) -> int:
        return 24  # Smaller than image_size (28) for random crop augmentation
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def epoch_size(self) -> int:
        return 60000  # MNIST training set size
    
    @property
    def num_channels(self) -> int:
        return 1  # MNIST is grayscale
    
    def get_transforms(self, mode: str = "train", crop_size: Optional[int] = None) -> Any:
        """Get transforms for MNIST.
        
        Converts grayscale to 3-channel for compatibility with pretrained models,
        and applies standard ImageNet normalization.
        """
        if crop_size is None:
            crop_size = self.crop_size
            
        if mode == "train":
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert to RGB
                transforms.RandomRotation(degrees=10),
                transforms.RandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
                transforms.ToTensor(),
            ])
        elif mode in ("val", "test"):
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert to RGB
                transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Unknown transform mode: {mode}")
    
    def create_dataset(self, root: str, split: str, transform: Any = None, **kwargs) -> Dataset:
        """Create MNIST dataset instance.
        
        Args:
            root: Root directory for the dataset
            split: Dataset split ("train", "val", or "test")
            transform: Optional transform to apply
            **kwargs: Additional arguments (download=True by default)
            
        Returns:
            MNIST Dataset instance
        """
        # MNIST only has train/test splits, map val to test
        train = (split == "train")
        
        if transform is None:
            transform = self.get_transforms(mode=split)
        
        download = kwargs.pop("download", True)
        
        return MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
            **kwargs
        )

