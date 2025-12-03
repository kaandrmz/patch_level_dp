from typing import Dict, Type, List, Union

from .base import DatasetConfig
from .base_classification import ClassificationDatasetConfig
from .cityscapes import CityscapesConfig
from .a2d2 import A2D2Config
from .mnist import MNISTConfig
from .dtd import DTDConfig

# Segmentation datasets (DatasetConfig subclasses)
DATASET_CONFIGS: Dict[str, Type[DatasetConfig]] = {
    "cityscapes": CityscapesConfig,
    "a2d2": A2D2Config,
}

# Classification datasets (ClassificationDatasetConfig subclasses)
CLASSIFICATION_DATASET_CONFIGS: Dict[str, Type[ClassificationDatasetConfig]] = {
    "mnist": MNISTConfig,
    "dtd": DTDConfig,
}

# Combined registry for convenience (both types)
ALL_DATASET_CONFIGS: Dict[str, Type[Union[DatasetConfig, ClassificationDatasetConfig]]] = {
    **DATASET_CONFIGS,
    **CLASSIFICATION_DATASET_CONFIGS,
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Factory function to get dataset configuration by name.
    
    Args:
        name: Dataset name (e.g., "cityscapes", "a2d2", "mnist", "dtd")
        
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


def get_any_dataset_config(name: str) -> Union[DatasetConfig, ClassificationDatasetConfig]:
    """Factory function to get any dataset configuration by name.
    
    Searches both segmentation and classification datasets.
    
    Args:
        name: Dataset name (e.g., "cityscapes", "mnist")
        
    Returns:
        Dataset config instance for the specified dataset
        
    Raises:
        ValueError: If dataset name is not registered
    """
    if name in DATASET_CONFIGS:
        return DATASET_CONFIGS[name]()
    elif name in CLASSIFICATION_DATASET_CONFIGS:
        return CLASSIFICATION_DATASET_CONFIGS[name]()
    else:
        available = list(ALL_DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")


def list_available_datasets() -> List[str]:
    """Get list of available segmentation dataset names.
    
    Returns:
        List of available dataset names
    """
    return list(DATASET_CONFIGS.keys())


def list_classification_datasets() -> List[str]:
    """Get list of available classification dataset names.
    
    Returns:
        List of available classification dataset names
    """
    return list(CLASSIFICATION_DATASET_CONFIGS.keys())


def list_all_datasets() -> List[str]:
    """Get list of all available dataset names.
    
    Returns:
        List of all available dataset names
    """
    return list(ALL_DATASET_CONFIGS.keys())


def register_dataset_config(name: str, config_class: Type[DatasetConfig]) -> None:
    """Register a new segmentation dataset configuration.
    
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
    ALL_DATASET_CONFIGS[name] = config_class


def register_classification_dataset_config(
    name: str, config_class: Type[ClassificationDatasetConfig]
) -> None:
    """Register a new classification dataset configuration.
    
    Args:
        name: Dataset name identifier
        config_class: ClassificationDatasetConfig subclass
        
    Raises:
        ValueError: If name is already registered
    """
    if name in CLASSIFICATION_DATASET_CONFIGS:
        raise ValueError(f"Classification dataset '{name}' is already registered")
    
    if not issubclass(config_class, ClassificationDatasetConfig):
        raise ValueError(f"config_class must be a subclass of ClassificationDatasetConfig")
    
    CLASSIFICATION_DATASET_CONFIGS[name] = config_class
    ALL_DATASET_CONFIGS[name] = config_class


__all__ = [
    # Base classes
    "DatasetConfig",
    "ClassificationDatasetConfig",
    # Segmentation datasets
    "CityscapesConfig",
    "A2D2Config",
    # Classification datasets
    "MNISTConfig",
    "DTDConfig",
    # Registries
    "DATASET_CONFIGS",
    "CLASSIFICATION_DATASET_CONFIGS",
    "ALL_DATASET_CONFIGS",
    # Factory functions
    "get_dataset_config",
    "get_classification_dataset_config",
    "get_any_dataset_config",
    # List functions
    "list_available_datasets",
    "list_classification_datasets",
    "list_all_datasets",
    # Registration functions
    "register_dataset_config",
    "register_classification_dataset_config",
]