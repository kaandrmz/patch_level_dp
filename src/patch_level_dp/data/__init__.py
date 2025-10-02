from typing import Dict, Type, List

from .base import DatasetConfig
from .cityscapes import CityscapesConfig
from .a2d2 import A2D2Config

DATASET_CONFIGS: Dict[str, Type[DatasetConfig]] = {
    "cityscapes": CityscapesConfig,
    "a2d2": A2D2Config,
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Factory function to get dataset configuration by name.
    
    Args:
        name: Dataset name (e.g., "cityscapes", "a2d2")
        
    Returns:
        DatasetConfig instance for the specified dataset
        
    Raises:
        ValueError: If dataset name is not registered
    """
    if name not in DATASET_CONFIGS:
        available = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
    
    config_class = DATASET_CONFIGS[name]
    return config_class()


def list_available_datasets() -> List[str]:
    """Get list of available dataset names.
    
    Returns:
        List of available dataset names
    """
    return list(DATASET_CONFIGS.keys())


def register_dataset_config(name: str, config_class: Type[DatasetConfig]) -> None:
    """Register a new dataset configuration.
    
    Args:
        name: Dataset name identifier
        config_class: DatasetConfig subclass
        
    Raises:
        ValueError: If name is already registered
    """
    if name in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{name}' is already registered")
    
    if not issubclass(config_class, DatasetConfig):
        raise ValueError(f"config_class must be a subclass of DatasetConfig")
    
    DATASET_CONFIGS[name] = config_class


__all__ = [
    "DatasetConfig",
    "CityscapesConfig",
    "A2D2Config",
    "DATASET_CONFIGS",
    "get_dataset_config",
    "list_available_datasets",
    "register_dataset_config",
]