"""Training function for DP classification models."""

import os
import math
from typing import Dict, Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from ..data import ALL_DATASET_CONFIGS
from ..privacy import DPDataLoader, UniformWithoutReplacementSampler
from .dp_classification_model import DPClassificationModel


def train_dp_classification_model(
    model_name: str,
    dataset_name: str,
    dataset_root: str,
    epsilon: float,
    sensitivity: float,
    clip_norm: float,
    num_epochs: int,
    batch_size: int,
    max_physical_batch_size: int,
    crop_size: int,
    delta: Optional[float] = None,
    batch_sampling_prob: Optional[float] = None,
    lr: float = 0.01,
    privacy_patch_size: Tuple[int, int] = (10, 10),
    padding: int = 0,
    standard_deviation: Optional[float] = None,
    baseline_privacy: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    seed_value: int = 516,
    checkpoint_dir: str = "/nfs/students/duk/checkpoints",
    check_val_every_n_epoch: int = 2,
    num_sanity_val_steps: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """Train a DP classification model.
    
    Args:
        model_name: Model architecture name (e.g., "resnet18", "simpleconv")
        dataset_name: Dataset name (e.g., "mnist", "dtd")
        dataset_root: Root directory for dataset
        epsilon: Privacy budget epsilon
        delta: Privacy budget delta (if None, calculated as 1.0 / epoch_size)
        sensitivity: Sensitivity parameter for DP
        batch_sampling_prob: Batch sampling probability (if None, calculated as batch_size / epoch_size)
        clip_norm: Gradient clipping norm
        num_epochs: Number of training epochs
        batch_size: Logical batch size
        max_physical_batch_size: Maximum physical batch size for memory management
        lr: Learning rate
        crop_size: Crop size for training
        privacy_patch_size: Size of privacy-sensitive patches
        padding: Padding for images
        standard_deviation: Fixed noise level (if None, calculated from epsilon)
        resume_from_checkpoint: Path to checkpoint to resume from
        seed_value: Random seed
        checkpoint_dir: Directory to save checkpoints
        check_val_every_n_epoch: How often to run validation
        num_sanity_val_steps: Number of sanity validation steps
        **kwargs: Additional model-specific arguments (e.g., baseline_privacy)
        
    Returns:
        Dictionary with training results including model, epsilon, and metrics
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    L.seed_everything(seed_value)
    
    # Get dataset config
    if dataset_name not in ALL_DATASET_CONFIGS:
        available = list(ALL_DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    dataset_config = ALL_DATASET_CONFIGS[dataset_name]()
    
    train_transform = dataset_config.get_transforms(mode="train", crop_size=crop_size)
    train_dataset = dataset_config.create_dataset(
        root=dataset_root,
        split="train", 
        transform=train_transform
    )
    
    val_transform = dataset_config.get_transforms(mode="val")
    val_dataset = dataset_config.create_dataset(
        root=dataset_root,
        split="val",
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    train_loader = DPDataLoader.from_data_loader(train_loader, batch_size=batch_size, replacement=False)
    
    assert isinstance(train_loader.batch_sampler, UniformWithoutReplacementSampler), \
        f"Expected UniformWithoutReplacementSampler but got {type(train_loader.batch_sampler).__name__}"
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    epoch_size = dataset_config.epoch_size
    
    calculated_delta = 1.0 / (epoch_size ** 1.0)
    print(f"Using epoch_size={epoch_size}, calculated_delta={calculated_delta}")
    
    if delta is None:
        delta = calculated_delta
    
    calculated_batch_sampling_prob = batch_size / epoch_size
    print(f"Using batch_sampling_prob={calculated_batch_sampling_prob}")
    
    if batch_sampling_prob is None:
        batch_sampling_prob = calculated_batch_sampling_prob
    
    num_queries = math.ceil(epoch_size / batch_size) * num_epochs
    print(f"Privacy parameters: epoch_size={epoch_size}, delta={delta}, batch_sampling_prob={batch_sampling_prob}, num_queries={num_queries}")
    
    model = DPClassificationModel(
        model_name=model_name,
        dataset_name=dataset_name,
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
        batch_sampling_prob=batch_sampling_prob,
        max_physical_batch_size=max_physical_batch_size,
        batch_size=batch_size,
        clip_norm=clip_norm,
        num_queries=num_queries,
        train_loader=train_loader,
        crop_size=crop_size,
        privacy_patch_size=privacy_patch_size,
        padding=padding,
        lr=lr,
        standard_deviation=standard_deviation,
        baseline_privacy=baseline_privacy,
        **kwargs
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'best-{model_name}-{dataset_name}' + '-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        save_top_k=1,
        mode='max',
        save_weights_only=False
    )
    
    # Create logger with meaningful experiment name
    experiment_name = f"{model_name}_{dataset_name}_clip{clip_norm}_lr{lr}_eps{epsilon}"
    if baseline_privacy:
        experiment_name += "_baseline"
    
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=experiment_name,
        default_hp_metric=False,
    )
    print(f"Logging to: lightning_logs/{experiment_name}")
    
    trainer = L.Trainer(
        accelerator="auto", 
        devices="auto", 
        strategy="auto", 
        max_epochs=num_epochs, 
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        logger=logger,
        num_sanity_val_steps=num_sanity_val_steps
    )
    
    trainer.logger._log_graph = True
    trainer.fit(model, val_dataloaders=val_loader, ckpt_path=resume_from_checkpoint)
    
    final_epsilon = model.composed_pld.get_epsilon_for_delta(delta)
    
    results = {
        "model": model,
        "epsilon": final_epsilon,
        "delta": delta,
        "best_model_path": checkpoint_callback.best_model_path,
        "best_val_acc": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
    }
    
    if checkpoint_callback.best_model_path:
        print(f"Loading best model from {checkpoint_callback.best_model_path} "
              f"with val_acc: {checkpoint_callback.best_model_score:.4f}")
        
        best_model = DPClassificationModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            train_loader=train_loader
        )
        
        best_model.composed_pld = model.composed_pld
        
        results["model"] = best_model
        results["best_model_loaded"] = True
    else:
        print("No checkpoint found, returning the last model")
        results["best_model_loaded"] = False
    
    return results


def test_dp_classification_model(
    model: DPClassificationModel,
    dataset_root: str,
    batch_size: int = 8,
    **kwargs
) -> Dict[str, Any]:
    """Test a trained DP classification model.
    
    Args:
        model: Trained DP classification model
        dataset_root: Root directory for dataset  
        batch_size: Batch size for testing
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with test results and metrics
    """
    dataset_config = model.dataset_config
    
    test_transform = dataset_config.get_transforms(mode="test")
    test_dataset = dataset_config.create_dataset(
        root=dataset_root,
        split="test",
        transform=test_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto", 
        strategy="auto"
    )
    
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)
    
    return {
        "test_results": test_results,
        "test_acc": test_results[0].get("test_acc", None),
        "test_loss": test_results[0].get("test_loss", None),
    }
