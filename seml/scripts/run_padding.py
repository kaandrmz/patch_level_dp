"""SEML script for running padding parameter dependency experiments."""

from typing import Dict, Any, Optional

from seml import Experiment
from utils import calculate_common_parameters, add_common_metadata
from patch_level_dp.privacy.calculations import calc_intersection_prob, create_pld

ex = Experiment()


def calculate_epsilon_for_padding(
    noise_level: float,
    padding_value: int,
    image_size: tuple,
    crop_size: int,
    private_patch_size: int,
    fixed_delta: float,
    sensitivity: float,
    batch_sampling_prob: float,
    steps: int,
    target_delta: Optional[float] = None,
    use_intersection: bool = True
) -> float:
    """Calculate epsilon for a given padding value."""
    if use_intersection:
        intersection_prob = calc_intersection_prob(
            image_size=image_size,
            crop_patch_size=(crop_size, crop_size),
            private_patch_size=(private_patch_size, private_patch_size),
            padding=padding_value
        )
        if intersection_prob == 0:
            return float('inf')
        sampling_prob = batch_sampling_prob * intersection_prob
    else:
        sampling_prob = batch_sampling_prob
    
    if sampling_prob == 0:
        return float('inf')
    
    single_step_pld = create_pld(
        standard_deviation=noise_level,
        sensitivity=sensitivity, 
        sampling_prob=sampling_prob
    )
    total_pld = single_step_pld.self_compose(steps)
    
    delta = target_delta if target_delta is not None else fixed_delta
    epsilon = total_pld.get_epsilon_for_delta(delta)
    
    return epsilon


@ex.automain
def run(
    image_size: list,
    padding_value: int,
    clip_norm: float,
    batch_size: int,
    epoch_size: int,
    num_epochs: int,
    fixed_delta: float,
    experiment_set: str,
    crop_size: int,
    private_patch_size: int,
    noise_levels: Optional[list] = None,
    noise_labels: Optional[list] = None,
    noise_set: Optional[str] = None,
    target_delta: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """Calculate epsilon for padding parameter dependency experiment."""
    
    # Calculate common parameters
    common_params = calculate_common_parameters(image_size, clip_norm, batch_size, epoch_size, num_epochs)
    
    # Handle noise level sets
    if noise_set:
        noise_sets = {
            "high": ([4.0, 4.5, 5.0], ["4", "4.5", "5"]),
            "medium": ([2.5, 3.0, 3.5], ["2.5", "3", "3.5"]), 
            "low": ([1.0, 1.5, 2.0], ["1", "1.5", "2"])
        }
        current_noise_levels, current_noise_labels = noise_sets[noise_set]
    else:
        current_noise_levels = noise_levels
        current_noise_labels = noise_labels
    
    # Calculate epsilon for both intersection-based and baseline methods
    padding_results = {}
    
    for use_intersection in [True, False]:
        method_key = "intersection" if use_intersection else "baseline"
        method_results = {}
        
        for noise_level in current_noise_levels:
            epsilon = calculate_epsilon_for_padding(
                noise_level=noise_level,
                padding_value=padding_value,
                image_size=common_params['image_size_tuple'],
                crop_size=crop_size,
                private_patch_size=private_patch_size,
                fixed_delta=fixed_delta,
                sensitivity=common_params['sensitivity'],
                batch_sampling_prob=common_params['batch_sampling_prob'],
                steps=common_params['steps'],
                target_delta=target_delta,
                use_intersection=use_intersection
            )
            method_results[noise_level] = epsilon
        
        padding_results[method_key] = method_results
    
    results = {
        "experiment_type": "padding",
        "experiment_set": experiment_set,
        "crop_size": crop_size,
        "private_patch_size": private_patch_size,
        "padding_value": padding_value,
        "noise_levels": current_noise_levels,
        "noise_labels": current_noise_labels,
        "target_delta": target_delta,
        "padding_results": padding_results
    }
    
    # Add common metadata
    results = add_common_metadata(
        results, image_size, padding_value, fixed_delta, 
        common_params['sensitivity'], common_params['batch_sampling_prob'], 
        common_params['steps'], clip_norm, batch_size, epoch_size, num_epochs
    )
    
    return results
