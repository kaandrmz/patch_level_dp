from typing import Dict, Type, List

from .base import ModelArchitecture
from .deeplabv3plus import DeepLabV3PlusArchitecture
from .pspnet import PSPNetArchitecture
from .resnet import ResNetArchitecture
from .simpleconv import SimpleConvNetArchitecture, SimpleConvNet

# Segmentation models
SEGMENTATION_ARCHITECTURES: Dict[str, Type[ModelArchitecture]] = {
    "deeplabv3plus": DeepLabV3PlusArchitecture,
    "pspnet": PSPNetArchitecture,
}

# Classification models
CLASSIFICATION_ARCHITECTURES: Dict[str, Type[ModelArchitecture]] = {
    "resnet18": lambda: ResNetArchitecture("resnet18"),
    "resnet34": lambda: ResNetArchitecture("resnet34"),
    "resnet50": lambda: ResNetArchitecture("resnet50"),
    "resnet101": lambda: ResNetArchitecture("resnet101"),
    "resnet152": lambda: ResNetArchitecture("resnet152"),
    "simpleconv": SimpleConvNetArchitecture,
}

# Combined registry (backward compatible)
MODEL_ARCHITECTURES: Dict[str, Type[ModelArchitecture]] = {
    "deeplabv3plus": DeepLabV3PlusArchitecture,
    "pspnet": PSPNetArchitecture,
    "resnet18": lambda: ResNetArchitecture("resnet18"),
    "resnet34": lambda: ResNetArchitecture("resnet34"),
    "resnet50": lambda: ResNetArchitecture("resnet50"),
    "resnet101": lambda: ResNetArchitecture("resnet101"),
    "resnet152": lambda: ResNetArchitecture("resnet152"),
    "simpleconv": SimpleConvNetArchitecture,
}


def get_model_architecture(name: str) -> ModelArchitecture:
    """Factory function to get model architecture by name.
    
    Args:
        name: Model architecture name (e.g., "deeplabv3plus", "pspnet", "resnet18")
        
    Returns:
        ModelArchitecture instance for the specified model
        
    Raises:
        ValueError: If model name is not registered
    """
    if name not in MODEL_ARCHITECTURES:
        available = list(MODEL_ARCHITECTURES.keys())
        raise ValueError(f"Unknown model architecture '{name}'. Available models: {available}")
    
    architecture_class_or_factory = MODEL_ARCHITECTURES[name]
    # Handle both class types and lambda factories
    return architecture_class_or_factory()


def get_classification_model(name: str) -> ModelArchitecture:
    """Factory function to get classification model architecture by name.
    
    Args:
        name: Model name (e.g., "resnet18", "simpleconv")
        
    Returns:
        ModelArchitecture instance for the specified model
        
    Raises:
        ValueError: If model name is not registered
    """
    if name not in CLASSIFICATION_ARCHITECTURES:
        available = list(CLASSIFICATION_ARCHITECTURES.keys())
        raise ValueError(f"Unknown classification model '{name}'. Available: {available}")
    
    architecture_class_or_factory = CLASSIFICATION_ARCHITECTURES[name]
    return architecture_class_or_factory()


def get_segmentation_model(name: str) -> ModelArchitecture:
    """Factory function to get segmentation model architecture by name.
    
    Args:
        name: Model name (e.g., "deeplabv3plus", "pspnet")
        
    Returns:
        ModelArchitecture instance for the specified model
        
    Raises:
        ValueError: If model name is not registered
    """
    if name not in SEGMENTATION_ARCHITECTURES:
        available = list(SEGMENTATION_ARCHITECTURES.keys())
        raise ValueError(f"Unknown segmentation model '{name}'. Available: {available}")
    
    architecture_class = SEGMENTATION_ARCHITECTURES[name]
    return architecture_class()


def list_available_models() -> List[str]:
    """Get list of all available model architecture names.
    
    Returns:
        List of available model architecture names
    """
    return list(MODEL_ARCHITECTURES.keys())


def list_segmentation_models() -> List[str]:
    """Get list of available segmentation model names.
    
    Returns:
        List of segmentation model names
    """
    return list(SEGMENTATION_ARCHITECTURES.keys())


def list_classification_models() -> List[str]:
    """Get list of available classification model names.
    
    Returns:
        List of classification model names
    """
    return list(CLASSIFICATION_ARCHITECTURES.keys())


def register_model_architecture(name: str, architecture_class: Type[ModelArchitecture]) -> None:
    """Register a new model architecture.
    
    Args:
        name: Model architecture name identifier
        architecture_class: ModelArchitecture subclass
        
    Raises:
        ValueError: If name is already registered or invalid class
    """
    if name in MODEL_ARCHITECTURES:
        raise ValueError(f"Model architecture '{name}' is already registered")
    
    if not issubclass(architecture_class, ModelArchitecture):
        raise ValueError(f"architecture_class must be a subclass of ModelArchitecture")
    
    MODEL_ARCHITECTURES[name] = architecture_class


__all__ = [
    # Base class
    "ModelArchitecture",
    # Segmentation models
    "DeepLabV3PlusArchitecture",
    "PSPNetArchitecture",
    # Classification models
    "ResNetArchitecture",
    "SimpleConvNetArchitecture",
    "SimpleConvNet",
    # Registries
    "MODEL_ARCHITECTURES",
    "SEGMENTATION_ARCHITECTURES",
    "CLASSIFICATION_ARCHITECTURES",
    # Factory functions
    "get_model_architecture",
    "get_segmentation_model",
    "get_classification_model",
    # List functions
    "list_available_models",
    "list_segmentation_models",
    "list_classification_models",
    # Registration
    "register_model_architecture",
]