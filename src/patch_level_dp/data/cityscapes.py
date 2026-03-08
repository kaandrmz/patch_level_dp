"""Cityscapes dataset configuration."""

from typing import Tuple, Type, Any, Optional

from torch.utils.data import Dataset

from patch_level_dp.models.architectures.deeplabv3plus.datasets.cityscapes import Cityscapes
import patch_level_dp.models.architectures.deeplabv3plus.utils.ext_transforms as et

from .base import DatasetConfig


class CityscapesConfig(DatasetConfig):
    """Configuration for Cityscapes dataset."""
    
    @property
    def name(self) -> str:
        return "cityscapes"
    
    @property  
    def image_size(self) -> Tuple[int, int]:
        return (1024, 2048)
    
    @property
    def crop_size(self) -> int:
        return 505
    
    @property
    def num_classes(self) -> int:
        return 19
    
    @property
    def epoch_size(self) -> int:
        return 2975
    
    def get_dataset_class(self) -> Type[Dataset]:
        return Cityscapes
    
    def get_transforms(self, mode: str = "train", crop_size: Optional[int] = None, padding: int = 0, 
                       gaussian_augmentation: bool = False, gaussian_noise_std: float = None) -> Any:
        """Get transforms for Cityscapes.
        
        Args:
            mode: "train", "val", or "test"
            crop_size: Size for random cropping
            padding: Padding for images
            gaussian_augmentation: If True, uses Gaussian data augmentation baseline
            gaussian_noise_std: Standard deviation for Gaussian noise (in [0, 255] scale)
        """
        if crop_size is None:
            crop_size = self.crop_size
            
        if mode == "train":
            if gaussian_augmentation:
                if gaussian_noise_std is None:
                    raise ValueError("gaussian_noise_std must be provided when gaussian_augmentation=True")
                
                # Convert noise std from [0, 255] scale to [0, 1] scale (after ToTensor)
                noise_std_normalized = gaussian_noise_std / 255.0
                
                # Check if we need cropping (crop_size < image dimensions)
                image_h, image_w = self.image_size
                needs_crop = (crop_size < image_h or crop_size < image_w)
                
                if needs_crop:
                    # Crop first, then add noise to the cropped image
                    return et.ExtCompose([
                        et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
                        et.ExtToTensor(),
                        et.ExtGaussianNoise(std=noise_std_normalized),
                        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                else:
                    # Use full images (no cropping)
                    return et.ExtCompose([
                        et.ExtToTensor(),
                        et.ExtGaussianNoise(std=noise_std_normalized),
                        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            else:
                # Standard training with cropping
                return et.ExtCompose([
                    et.ExtRandomHorizontalFlip(),
                    et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
                    et.ExtToTensor(),
                    et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        elif mode in ("val", "test"):
            return et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError(f"Unknown transform mode: {mode}")
    
    def create_dataset(self, root: str, split: str, transform: Any = None, **kwargs) -> Dataset:
        """Create Cityscapes dataset instance."""
        if transform is None:
            transform = self.get_transforms(mode=split)
        
        return Cityscapes(
            root=root,
            split=split,
            transform=transform,
            **kwargs
        )