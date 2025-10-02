"""Experiment utilities for DP segmentation training.

This module contains shared utilities used across experiments,
including seeding, metrics reporting, and checkpoint management.
"""
import os
import random
from typing import Dict, Any, Optional
import numpy as np
import torch

import lightning as L
from lightning.pytorch.callbacks import Callback


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    L.seed_everything(seed)


def setup_memory_optimization() -> None:
    """Setup environment variables for memory optimization."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dictionary for pretty printing.
    
    Args:
        metrics: Dictionary of metric names to values
        prefix: Optional prefix for metric names
        
    Returns:
        Formatted string representation of metrics
    """
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{prefix}{key}: {value:.4f}")
        else:
            formatted.append(f"{prefix}{key}: {value}")
    return ", ".join(formatted)


def log_model_info(model: torch.nn.Module) -> None:
    """Log information about the model.
    
    Args:
        model: PyTorch model to analyze
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")


def save_experiment_config(config: Dict[str, Any], save_path: str) -> None:
    """Save experiment configuration to file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path to save configuration file
    """
    import json
    
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)  # Test if serializable
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Experiment configuration saved to: {save_path}")


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


class MetricsLogger(Callback):
    """Callback to log detailed metrics during training."""
    
    def __init__(self, log_every_n_epochs: int = 1):
        """Initialize metrics logger.
        
        Args:
            log_every_n_epochs: How often to log detailed metrics
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log training metrics at epoch end."""
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            if hasattr(pl_module, 'train_evaluator'):
                score = pl_module.train_evaluator.get_results()
                metrics_str = format_metrics(score, prefix="train_")
                print(f"Epoch {trainer.current_epoch:3d} | {metrics_str}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at epoch end."""
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            if hasattr(pl_module, 'val_evaluator'):
                score = pl_module.val_evaluator.get_results()
                metrics_str = format_metrics(score, prefix="val_")
                print(f"Epoch {trainer.current_epoch:3d} | {metrics_str}")


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Get information about a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        return {"exists": False}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        info = {
            "exists": True,
            "epoch": checkpoint.get("epoch", None),
            "global_step": checkpoint.get("global_step", None),
            "state_dict_keys": list(checkpoint.get("state_dict", {}).keys())[:10],  # First 10 keys
        }
        
        if "callbacks" in checkpoint:
            for callback_data in checkpoint["callbacks"].values():
                if "best_model_score" in callback_data:
                    info["best_score"] = callback_data["best_model_score"]
                    break
        
        return info
        
    except Exception as e:
        return {"exists": True, "error": str(e)}


def compute_privacy_metrics(
    epsilon: float, 
    delta: float, 
    num_queries: int,
    sensitivity: float
) -> Dict[str, float]:
    """Compute and format privacy-related metrics.
    
    Args:
        epsilon: Privacy budget epsilon
        delta: Privacy budget delta
        num_queries: Number of privacy queries
        sensitivity: Sensitivity parameter
        
    Returns:
        Dictionary with privacy metrics
    """
    return {
        "epsilon": epsilon,
        "delta": delta,
        "num_queries": num_queries,
        "sensitivity": sensitivity,
        "privacy_loss_per_query": epsilon / max(num_queries, 1),
    }
