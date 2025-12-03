def get_dp_segmentation_model():
    """Lazy import of DPSegmentationModel."""
    from .dp_model import DPSegmentationModel
    return DPSegmentationModel


def get_dp_classification_model():
    """Lazy import of DPClassificationModel."""
    from .dp_classification_model import DPClassificationModel
    return DPClassificationModel


def get_train_functions():
    """Lazy import of training functions.""" 
    from .train_dp_model import train_dp_model, test_dp_model
    return train_dp_model, test_dp_model


# Segmentation imports
try:
    from .train_dp_model import train_dp_model, test_dp_model
except ImportError:
    train_dp_model, test_dp_model = None, None

# Classification imports
try:
    from .train_dp_classification_model import train_dp_classification_model, test_dp_classification_model
except ImportError:
    train_dp_classification_model, test_dp_classification_model = None, None

from .utils import (
    seed_everything,
    setup_memory_optimization,
    format_metrics,
    log_model_info,
    save_experiment_config,
    load_experiment_config,
    MetricsLogger,
    get_checkpoint_info,
    compute_privacy_metrics,
)


__all__ = [
    # Segmentation
    "get_dp_segmentation_model",
    "get_train_functions",
    "train_dp_model", 
    "test_dp_model",
    # Classification
    "get_dp_classification_model",
    "train_dp_classification_model",
    "test_dp_classification_model",
    # Utils
    "seed_everything",
    "setup_memory_optimization",
    "format_metrics",
    "log_model_info",
    "save_experiment_config",
    "load_experiment_config",
    "MetricsLogger",
    "get_checkpoint_info",
    "compute_privacy_metrics",
]
