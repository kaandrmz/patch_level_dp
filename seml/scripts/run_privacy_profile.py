"""SEML script for running privacy profile experiments."""

import math
import numpy as np
from typing import Dict, Any
from multiprocessing import Pool, cpu_count

from seml import Experiment
from patch_level_dp.privacy.calculations import (
    calc_intersection_prob, 
    calculate_delta_for_epsilon_multi, 
    get_patch_pixels_and_bbox,
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
    private_patch_size,  # int or list [width, height]
    noise_level: float,
    sigma_data: float,
    priv_patch_shape: str = "square",
    **kwargs
) -> Dict[str, Any]:
    """Calculate privacy profile (epsilon-delta curves) for given configuration.
    
    Args:
        priv_patch_shape: Shape of the private patch. Defaults to "square" for backward compatibility.
        private_patch_size: Size of the private patch. Can be:
            - int: Single size value (rectangles use 2:1 ratio, squares use size x size)
            - [width, height]: Custom dimensions (from YAML list)
    """
    
    # Calculate derived parameters
    image_size_tuple = tuple(image_size)
    sensitivity = clip_norm * 2
    batch_sampling_prob = batch_size / epoch_size
    steps = math.ceil(epoch_size / batch_size) * num_epochs
    
    # Convert list to tuple if needed (YAML lists become Python lists)
    if isinstance(private_patch_size, list):
        private_patch_size = tuple(private_patch_size)
    
    # Generate pixel representation for the private patch shape
    private_patch_pixels, private_patch_bbox = get_patch_pixels_and_bbox(
        priv_patch_shape, private_patch_size
    )
    
    # Generate epsilon values (log scale)
    epsilon_values = np.logspace(np.log10(epsilon_min), np.log10(epsilon_max), epsilon_num_points)
    
    # PRE-CALCULATE intersection probability once (not 50 times!)
    intersection_prob = calc_intersection_prob(
        image_size=image_size_tuple,
        crop_patch_size=(crop_size, crop_size),
        private_patch_pixels=private_patch_pixels,
        private_patch_bbox=private_patch_bbox,
        padding=padding
    )
    
    # Calculate delta values for all sampling methods and epsilon values
    # Parallelize across epsilon values for speedup
    num_workers = min(cpu_count(), 16)  # Use up to 16 CPUs for better parallelization
    
    results = {}
    for sampling_method in SAMPLING_METHODS:
        # Prepare arguments for all epsilon values
        args_list = [
            (
                epsilon, image_size_tuple, padding, crop_size, private_patch_size,
                batch_sampling_prob, noise_level, sensitivity, steps, 
                sampling_method, sigma_data, priv_patch_shape, intersection_prob
            )
            for epsilon in epsilon_values
        ]
        
        # Parallel computation
        with Pool(num_workers) as pool:
            deltas = pool.map(calculate_delta_for_epsilon_multi, args_list)
        
        # Store results
        method_results = {float(epsilon): delta for epsilon, delta in zip(epsilon_values, deltas)}
        results[sampling_method] = method_results
    
    # Flatten results for easier database storage
    flattened_results = {
        f"{epsilon},{sampling_method}": delta
        for sampling_method in SAMPLING_METHODS
        for epsilon, delta in results[sampling_method].items()
    }
    
    return {
        "experiment_set": experiment_set,
        "image_size": image_size,
        "crop_size": crop_size,
        "private_patch_size": private_patch_size,
        "priv_patch_shape": priv_patch_shape,
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
