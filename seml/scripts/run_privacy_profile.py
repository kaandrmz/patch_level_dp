"""SEML script for running privacy profile experiments."""

import math
import numpy as np
from typing import Dict, Any

from seml import Experiment
from patch_level_dp.privacy.calculations import (
    calc_intersection_prob, 
    calculate_delta_for_epsilon_multi, 
    SAMPLING_METHODS
)

ex = Experiment()


@ex.automain
def run(
    experiment_set: str,
    image_size: list,
    padding: int,
    clip_norm: float,
    batch_size: int,
    epoch_size: int,
    num_epochs: int,
    epsilon_min: float,
    epsilon_max: float,
    epsilon_num_points: int,
    crop_size: int,
    private_patch_size: int,
    noise_level: float,
    sigma_data: float,
    **kwargs
) -> Dict[str, Any]:
    """Calculate privacy profile (epsilon-delta curves) for given configuration."""
    
    # Calculate derived parameters
    image_size_tuple = tuple(image_size)
    sensitivity = clip_norm * 2
    batch_sampling_prob = batch_size / epoch_size
    steps = math.ceil(epoch_size / batch_size) * num_epochs
    
    # Generate epsilon values (log scale)
    epsilon_values = np.logspace(np.log10(epsilon_min), np.log10(epsilon_max), epsilon_num_points)
    
    # Calculate delta values for all sampling methods and epsilon values
    results = {}
    for sampling_method in SAMPLING_METHODS:
        method_results = {}
        for epsilon in epsilon_values:
            args = (
                epsilon, image_size_tuple, padding, crop_size, private_patch_size,
                batch_sampling_prob, noise_level, sensitivity, steps, 
                sampling_method, sigma_data
            )
            delta = calculate_delta_for_epsilon_multi(args)
            method_results[float(epsilon)] = delta
        
        results[sampling_method] = method_results
    
    # Flatten results for easier database storage
    flattened_results = {
        f"{epsilon},{sampling_method}": delta
        for sampling_method in SAMPLING_METHODS
        for epsilon, delta in results[sampling_method].items()
    }
    
    # Calculate intersection probability for reference
    intersection_prob = calc_intersection_prob(
        image_size=image_size_tuple,
        crop_patch_size=(crop_size, crop_size),
        private_patch_size=(private_patch_size, private_patch_size),
        padding=padding
    )
    
    return {
        "experiment_set": experiment_set,
        "image_size": image_size,
        "crop_size": crop_size,
        "private_patch_size": private_patch_size,
        "noise_level": noise_level,
        "sigma_data": sigma_data,
        "padding": padding,
        "clip_norm": clip_norm,
        "sensitivity": sensitivity,
        "batch_sampling_prob": batch_sampling_prob,
        "steps": steps,
        "epsilon_min": epsilon_min,
        "epsilon_max": epsilon_max,
        "epsilon_num_points": epsilon_num_points,
        "intersection_prob": intersection_prob,
        "privacy_profile_results": flattened_results,
        "num_epsilon_points": len(epsilon_values),
        "num_methods": len(SAMPLING_METHODS)
    }
