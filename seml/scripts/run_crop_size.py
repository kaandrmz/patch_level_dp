"""SEML script for running crop size parameter dependency experiments."""

from typing import Dict, Any

from seml import Experiment
from utils import calculate_common_parameters, add_common_metadata
from patch_level_dp.privacy.calculations import calc_intersection_prob, create_pld

ex = Experiment()


def calculate_epsilon_for_crop_size(
    noise_level: float,
    crop_size: int,
    image_size: tuple,
    padding: int,
    fixed_delta: float,
    sensitivity: float,
    batch_sampling_prob: float,
    steps: int,
    fixed_private_patch_size: tuple
) -> float:
    """Calculate epsilon for a given crop size."""
    crop_patch_size = (crop_size, crop_size)
    
    intersection_prob = calc_intersection_prob(
        image_size=image_size,
        crop_patch_size=crop_patch_size,
        private_patch_size=fixed_private_patch_size,
        padding=padding
    )
    
    if intersection_prob == 0:
        return float('inf')
    
    sampling_prob = batch_sampling_prob * intersection_prob
    if sampling_prob == 0:
        return float('inf')
    
    single_step_pld = create_pld(noise_level, sensitivity, sampling_prob)
    total_pld = single_step_pld.self_compose(steps)
    epsilon = total_pld.get_epsilon_for_delta(fixed_delta)
    
    return epsilon


@ex.automain
def run(
    image_size: list,
    padding: int,
    clip_norm: float,
    batch_size: int,
    epoch_size: int,
    num_epochs: int,
    fixed_delta: float,
    noise_level: float,
    fixed_private_patch_size: list,
    crop_size: int,
    **kwargs
) -> Dict[str, Any]:
    """Calculate epsilon for crop size parameter dependency experiment."""
    
    # Calculate common parameters
    common_params = calculate_common_parameters(image_size, clip_norm, batch_size, epoch_size, num_epochs)
    
    # Calculate epsilon
    epsilon = calculate_epsilon_for_crop_size(
        noise_level=noise_level,
        crop_size=int(crop_size),
        image_size=common_params['image_size_tuple'],
        padding=padding,
        fixed_delta=fixed_delta,
        sensitivity=common_params['sensitivity'],
        batch_sampling_prob=common_params['batch_sampling_prob'],
        steps=common_params['steps'],
        fixed_private_patch_size=tuple(fixed_private_patch_size)
    )
    
    results = {
        "experiment_type": "crop_size",
        "noise_level": noise_level,
        "crop_size": int(crop_size),
        "epsilon": epsilon,
        "fixed_private_patch_size": fixed_private_patch_size,
    }
    
    # Add common metadata
    results = add_common_metadata(
        results, image_size, padding, fixed_delta, 
        common_params['sensitivity'], common_params['batch_sampling_prob'], 
        common_params['steps'], clip_norm, batch_size, epoch_size, num_epochs
    )
    
    return results
