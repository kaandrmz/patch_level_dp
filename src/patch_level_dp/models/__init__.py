from typing import Dict, Type, List

from .base import ModelArchitecture
from .deeplabv3plus import DeepLabV3PlusArchitecture
from .pspnet import PSPNetArchitecture

MODEL_ARCHITECTURES: Dict[str, Type[ModelArchitecture]] = {
    "deeplabv3plus": DeepLabV3PlusArchitecture,
    "pspnet": PSPNetArchitecture,
}


def get_model_architecture(name: str) -> ModelArchitecture:
    """Factory function to get model architecture by name.
    
    Args:
        name: Model architecture name (e.g., "deeplabv3plus", "pspnet")
        
    Returns:
        ModelArchitecture instance for the specified model
        
    Raises:
        ValueError: If model name is not registered
    """
    if name not in MODEL_ARCHITECTURES:
        available = list(MODEL_ARCHITECTURES.keys())
        raise ValueError(f"Unknown model architecture '{name}'. Available models: {available}")
    
    architecture_class = MODEL_ARCHITECTURES[name]
    return architecture_class()


def list_available_models() -> List[str]:
    """Get list of available model architecture names.
    
    Returns:
        List of available model architecture names
    """
    return list(MODEL_ARCHITECTURES.keys())


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
    "ModelArchitecture",
    "DeepLabV3PlusArchitecture",
    "PSPNetArchitecture",
    "MODEL_ARCHITECTURES",
    "get_model_architecture",
    "list_available_models",
    "register_model_architecture",
]