def get_dp_segmentation_model():
    """Lazy import of DPSegmentationModel."""
    from .dp_model import DPSegmentationModel
    return DPSegmentationModel


def get_train_functions():
    """Lazy import of training functions.""" 
    from .train_dp_model import train_dp_model, test_dp_model
    return train_dp_model, test_dp_model


try:
    from .train_dp_model import train_dp_model, test_dp_model
except ImportError:
    train_dp_model, test_dp_model = None, None

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
    "get_dp_segmentation_model",
    "get_train_functions",
    "train_dp_model", 
    "test_dp_model",
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