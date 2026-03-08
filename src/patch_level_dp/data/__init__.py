from typing import Dict, Type, Union

from .base import DatasetConfig
from .base_classification import ClassificationDatasetConfig
from .cityscapes import CityscapesConfig
from .a2d2 import A2D2Config
from .mnist import MNISTConfig
from .dtd import DTDConfig

DATASET_CONFIGS: Dict[str, Type[DatasetConfig]] = {
    "cityscapes": CityscapesConfig,
    "a2d2": A2D2Config,
}

CLASSIFICATION_DATASET_CONFIGS: Dict[str, Type[ClassificationDatasetConfig]] = {
    "mnist": MNISTConfig,
    "dtd": DTDConfig,
}

ALL_DATASET_CONFIGS: Dict[str, Type[Union[DatasetConfig, ClassificationDatasetConfig]]] = {
    **DATASET_CONFIGS,
    **CLASSIFICATION_DATASET_CONFIGS,
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


def get_classification_dataset_config(name: str) -> ClassificationDatasetConfig:
    """Factory function to get classification dataset configuration by name.
    
    Args:
        name: Dataset name (e.g., "mnist", "dtd")
        
    Returns:
        ClassificationDatasetConfig instance for the specified dataset
        
    Raises:
        ValueError: If dataset name is not registered
    """
    if name not in CLASSIFICATION_DATASET_CONFIGS:
        available = list(CLASSIFICATION_DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown classification dataset '{name}'. Available: {available}")
    
    config_class = CLASSIFICATION_DATASET_CONFIGS[name]
    return config_class()


__all__ = [
    "DatasetConfig",
    "ClassificationDatasetConfig",
    "CityscapesConfig",
    "A2D2Config",
    "MNISTConfig",
    "DTDConfig",
    "DATASET_CONFIGS",
    "CLASSIFICATION_DATASET_CONFIGS",
    "ALL_DATASET_CONFIGS",
    "get_dataset_config",
    "get_classification_dataset_config",
]
