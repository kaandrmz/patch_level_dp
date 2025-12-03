"""SEML script for training DP classification models."""

import math
from typing import Dict, Any

from seml import Experiment
from patch_level_dp.experiments import train_dp_classification_model, test_dp_classification_model


ex = Experiment()


def init_privacy_kwargs(
    model_name: str,
    dataset_name: str,
    dataset_root: str,
    epsilon: float,
    batch_size: int,
    num_epochs: int,
    clip_norm: float,
    max_physical_batch_size: int,
    privacy_patch_size: list,
    **kwargs
) -> Dict[str, Any]:
    """Initialize privacy and training parameters.
    
    Args:
        model_name: Model architecture name
        dataset_name: Dataset name
        dataset_root: Dataset root directory
        epsilon: Privacy budget epsilon
        batch_size: Batch size
        num_epochs: Number of epochs
        clip_norm: Gradient clipping norm
        max_physical_batch_size: Max physical batch size
        privacy_patch_size: Privacy patch size
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with initialized parameters
    """
    epoch_sizes = {
        "mnist": 60000,
        "dtd": 1879,
    }
    
    epoch_size = epoch_sizes.get(dataset_name, 60000)
    if dataset_name not in epoch_sizes:
        print(f"Warning: Unknown dataset {dataset_name}, using default epoch_size={epoch_size}")
    
    delta = 1.0 / (epoch_size ** 1.0)
    batch_sampling_prob = batch_size / epoch_size
    sensitivity = clip_norm * 2
    num_queries = math.ceil(epoch_size / batch_size) * num_epochs

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_root": dataset_root,
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": sensitivity,
        "batch_sampling_prob": batch_sampling_prob,
        "clip_norm": clip_norm,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "max_physical_batch_size": max_physical_batch_size,
        "privacy_patch_size": tuple(privacy_patch_size),
        **kwargs
    }


@ex.automain
def run(
    model_name: str,
    dataset_name: str,
    dataset_root: str,
    epsilon: float,
    batch_size: int,
    num_epochs: int,
    lr: float,
    crop_size: int,
    padding: int,
    clip_norm: float,
    max_physical_batch_size: int,
    privacy_patch_size: list,
    sensitivity: float = None,
    batch_sampling_prob: float = None,
    delta: float = None,
    baseline_privacy: bool = False,
    pretrained: bool = False,
    seed_value: int = 516,
    checkpoint_dir: str = "/nfs/students/duk/checkpoints",
    check_val_every_n_epoch: int = 2,
    num_sanity_val_steps: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """Main SEML experiment function for training DP classification models."""
    
    print(f"Training {model_name} on {dataset_name} with epsilon={epsilon}")
    print(f"Dataset root: {dataset_root}")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {lr}")
    print(f"Crop size: {crop_size}, Padding: {padding}")
    print(f"Clip norm: {clip_norm}, Max physical batch size: {max_physical_batch_size}")
    print(f"Privacy patch size: {privacy_patch_size}")
    print(f"Baseline privacy: {baseline_privacy}")
    print(f"Pretrained: {pretrained}")
    print(f"Seed: {seed_value}")
    
    train_params = init_privacy_kwargs(
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        epsilon=epsilon,
        batch_size=batch_size,
        num_epochs=num_epochs,
        clip_norm=clip_norm,
        max_physical_batch_size=max_physical_batch_size,
        privacy_patch_size=privacy_patch_size,
        lr=lr,
        crop_size=crop_size,
        padding=padding,
        baseline_privacy=baseline_privacy,
        pretrained=pretrained,
        seed_value=seed_value,
        checkpoint_dir=checkpoint_dir,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=num_sanity_val_steps,
        **kwargs
    )
    
    print("\nFinal training parameters:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")
    
    results = train_dp_classification_model(**train_params)
    
    print(f"\nTraining completed!")
    print(f"Final epsilon: {results['epsilon']}")
    print(f"Best validation accuracy: {results.get('best_val_acc', 'N/A')}")
    
    if results.get('best_model_loaded', False):
        print("\nTesting best model...")
        test_results = test_dp_classification_model(
            model=results['model'],
            dataset_root=dataset_root,
            batch_size=batch_size
        )
        print(f"Test results: {test_results}")
        results.update(test_results)
    
    return {
        "final_epsilon": results['epsilon'],
        "final_delta": results['delta'],
        "best_val_acc": results.get('best_val_acc'),
        "test_acc": results.get('test_acc'),
        "test_loss": results.get('test_loss'),
        "best_model_loaded": results.get('best_model_loaded', False),
        "checkpoint_path": results.get('best_model_path'),
    }
