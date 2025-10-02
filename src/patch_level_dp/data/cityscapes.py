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
    
    def get_transforms(self, mode: str = "train", crop_size: Optional[int] = None) -> Any:
        """Get transforms for Cityscapes."""
        if crop_size is None:
            crop_size = self.crop_size
            
        if mode == "train":
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