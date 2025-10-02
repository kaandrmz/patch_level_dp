from .data import get_dataset_config, list_available_datasets
from .models import get_model_architecture, list_available_models


def get_train_dp_model():
    """Lazy import of train_dp_model to avoid heavy dependencies on import."""
    from .experiments import train_dp_model
    return train_dp_model


def get_dp_segmentation_model():
    """Lazy import of DPSegmentationModel to avoid heavy dependencies on import."""
    from .experiments import DPSegmentationModel
    return DPSegmentationModel


__all__ = [
    "get_train_dp_model",
    "get_dp_segmentation_model",
    "get_dataset_config",
    "get_model_architecture", 
    "list_available_datasets",
    "list_available_models",
]