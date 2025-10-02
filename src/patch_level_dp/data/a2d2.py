"""A2D2 dataset configuration."""

from typing import Tuple, Type, Any, Optional

from torch.utils.data import Dataset

from patch_level_dp.models.architectures.deeplabv3plus.datasets.a2d2 import A2D2Dataset
import patch_level_dp.models.architectures.deeplabv3plus.utils.ext_transforms as et

from .base import DatasetConfig


class A2D2Config(DatasetConfig):
    """Configuration for A2D2 dataset."""
    
    @property
    def name(self) -> str:
        return "a2d2"
    
    @property  
    def image_size(self) -> Tuple[int, int]:
        return (1208, 1920)
    
    @property
    def crop_size(self) -> int:
        return 505
    
    @property
    def num_classes(self) -> int:
        return 18
    
    @property
    def epoch_size(self) -> int:
        return 18557
    
    def get_dataset_class(self) -> Type[Dataset]:
        return A2D2Dataset
    
    def get_transforms(self, mode: str = "train", crop_size: Optional[int] = None) -> Any:
        """Get transforms for A2D2."""
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
    
    def map_split(self, split: str) -> str:
        """Map common split names to A2D2-specific names."""
        split_mapping = {
            'train': 'training',
            'val': 'validation',
            'test': 'test'
        }
        return split_mapping.get(split, split)
    
    def create_dataset(self, root: str, split: str, transform: Any = None, **kwargs) -> Dataset:
        """Create A2D2 dataset instance."""
        mapped_split = self.map_split(split)
        
        if transform is None:
            transform = self.get_transforms(mode=split)
        
        return A2D2Dataset(
            root=root,
            split=mapped_split,
            transform=transform,
            **kwargs
        )