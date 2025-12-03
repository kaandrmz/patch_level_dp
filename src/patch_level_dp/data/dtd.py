"""DTD (Describable Textures Dataset) configuration for classification."""

from typing import Tuple, Type, Any, Optional, List

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import DTD
from PIL import Image

from .base_classification import ClassificationDatasetConfig


class FilteredDTD(Dataset):
    """DTD dataset wrapper that filters out images smaller than min_size.
    
    This ensures all images can be center-cropped to min_size without padding.
    """
    
    def __init__(self, dtd_dataset: DTD, min_size: int = 300):
        """Initialize filtered DTD dataset.
        
        Args:
            dtd_dataset: Original DTD dataset instance
            min_size: Minimum dimension required (images smaller than this are dropped)
        """
        self.dtd = dtd_dataset
        self.min_size = min_size
        self.valid_indices: List[int] = []
        
        # Pre-filter indices based on image size
        for idx in range(len(self.dtd)):
            img_path = self.dtd._image_files[idx]
            with Image.open(img_path) as img:
                w, h = img.size
                if min(w, h) >= min_size:
                    self.valid_indices.append(idx)
        
        print(f"FilteredDTD: Kept {len(self.valid_indices)}/{len(self.dtd)} images "
              f"(dropped {len(self.dtd) - len(self.valid_indices)} with min dim < {min_size})")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        original_idx = self.valid_indices[idx]
        return self.dtd[original_idx]


class DTDConfig(ClassificationDatasetConfig):
    """Configuration for DTD (Describable Textures Dataset).
    
    DTD contains 5640 images, organized according to 47 texture categories.
    Images are variable-sized (300-640), so we center crop to 300x300 and
    drop the 3 images with min dimension < 300.
    """
    
    @property
    def name(self) -> str:
        return "dtd"
    
    @property  
    def image_size(self) -> Tuple[int, int]:
        return (300, 300)  # All images center-cropped to this size
    
    @property
    def crop_size(self) -> int:
        return 200  # Smaller than image_size (300) for random crop augmentation
    
    @property
    def num_classes(self) -> int:
        return 47  # 47 texture categories
    
    @property
    def epoch_size(self) -> int:
        # Original: 1880 per split. After filtering (min dim < 300):
        # train: 1879 (1 dropped), val: 1878 (2 dropped), test: 1880 (0 dropped)
        return 1879
    
    def get_transforms(self, mode: str = "train", crop_size: Optional[int] = None) -> Any:
        """Get transforms for DTD.
        
        First center crops to image_size (300x300), then applies augmentation.
        """
        if crop_size is None:
            crop_size = self.crop_size
        
        image_size = self.image_size[0]  # 300
            
        if mode == "train":
            return transforms.Compose([
                # transforms.CenterCrop(image_size),  # Crop to 300x300 first
                transforms.Resize((300, 300)),  # Resize to fixed size for batching
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif mode in ("val", "test"):
            return transforms.Compose([
                # transforms.CenterCrop(image_size),  # Crop to 300x300
                transforms.Resize((300, 300)),  # Resize to fixed size for batching
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError(f"Unknown transform mode: {mode}")
    
    def create_dataset(self, root: str, split: str, transform: Any = None, **kwargs) -> Dataset:
        """Create filtered DTD dataset instance.
        
        Args:
            root: Root directory for the dataset
            split: Dataset split ("train", "val", or "test")
            transform: Optional transform to apply
            **kwargs: Additional arguments (partition=1 by default, download=True)
            
        Returns:
            FilteredDTD Dataset instance (drops images with min dim < 300)
        """
        if transform is None:
            transform = self.get_transforms(mode=split)
        
        partition = kwargs.pop("partition", 1)
        download = kwargs.pop("download", True)
        
        # Create base DTD dataset
        base_dtd = DTD(
            root=root,
            split=split,
            partition=partition,
            transform=transform,
            download=download,
            **kwargs
        )
        
        return FilteredDTD(base_dtd, min_size=self.image_size[0])

