"""MNIST dataset configuration for classification."""

from typing import Tuple, Type, Any, Optional

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
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
        return 24
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def epoch_size(self) -> int:
        return 60000  # MNIST training set size
    
    @property
    def num_channels(self) -> int:
        return 1  # MNIST is grayscale
    
    def get_transforms(self, mode: str = "train", crop_size: Optional[int] = None, padding: int = 0,
                        gaussian_augmentation: bool = False, gaussian_noise_std: float = None) -> Any:
        """Get transforms for MNIST.

        Converts grayscale to 3-channel for compatibility with pretrained models.
        When gaussian_augmentation=True, uses minimal transforms (crop + noise only).

        Args:
            mode: "train", "val", or "test"
            crop_size: Size for random crop (only used in train mode)
            padding: Padding to add around images (e.g., 2 to go from 28x28 to 32x32)
            gaussian_augmentation: If True, uses Gaussian data augmentation baseline
            gaussian_noise_std: Noise std in pixel space [0, 255] (required if gaussian_augmentation=True)
        """
        if crop_size is None:
            crop_size = self.crop_size

        transform_list = [transforms.Grayscale(num_output_channels=3)]  # Convert to RGB

        if padding > 0:
            transform_list.append(transforms.Pad(padding))

        if mode == "train":
            if gaussian_augmentation:
                if gaussian_noise_std is None:
                    raise ValueError("gaussian_noise_std must be provided when gaussian_augmentation=True")
                noise_std_normalized = gaussian_noise_std / 255.0
                transform_list.append(transforms.RandomCrop(size=(crop_size, crop_size), pad_if_needed=True))
                transform_list.append(transforms.ToTensor())
                transform_list.append(transforms_v2.GaussianNoise(mean=0.0, sigma=noise_std_normalized, clip=True))
                return transforms.Compose(transform_list)
            transform_list.append(transforms.RandomCrop(size=(crop_size, crop_size), pad_if_needed=True))

        transform_list.append(transforms.ToTensor())

        return transforms.Compose(transform_list)

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
