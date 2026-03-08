from typing import Dict, Type

from .base import ModelArchitecture
from .deeplabv3plus import DeepLabV3PlusArchitecture
from .pspnet import PSPNetArchitecture
from .resnet import ResNetArchitecture
from .vgg import VGGArchitecture

MODEL_ARCHITECTURES: Dict[str, Type[ModelArchitecture]] = {
    "deeplabv3plus": DeepLabV3PlusArchitecture,
    "pspnet": PSPNetArchitecture,
    "resnet18": lambda: ResNetArchitecture("resnet18"),
    "resnet34": lambda: ResNetArchitecture("resnet34"),
    "resnet50": lambda: ResNetArchitecture("resnet50"),
    "resnet101": lambda: ResNetArchitecture("resnet101"),
    "resnet152": lambda: ResNetArchitecture("resnet152"),
    "vgg11": lambda: VGGArchitecture("vgg11"),
    "vgg13": lambda: VGGArchitecture("vgg13"),
    "vgg16": lambda: VGGArchitecture("vgg16"),
    "vgg19": lambda: VGGArchitecture("vgg19"),
    "vgg11_bn": lambda: VGGArchitecture("vgg11_bn"),
    "vgg13_bn": lambda: VGGArchitecture("vgg13_bn"),
    "vgg16_bn": lambda: VGGArchitecture("vgg16_bn"),
    "vgg19_bn": lambda: VGGArchitecture("vgg19_bn"),
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
    return architecture_class_or_factory()


__all__ = [
    "ModelArchitecture",
    "DeepLabV3PlusArchitecture",
    "PSPNetArchitecture",
    "ResNetArchitecture",
    "VGGArchitecture",
    "MODEL_ARCHITECTURES",
    "get_model_architecture",
]
