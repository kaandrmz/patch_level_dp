"""Common utilities for parameter dependency experiments."""

import math
# from patch_level_dp.privacy.calculations import calc_intersection_prob, create_pld


def calculate_common_parameters(image_size, clip_norm, batch_size, epoch_size, num_epochs):
    """Calculate common parameters used across all experiment types."""
    image_size_tuple = tuple(image_size)
    sensitivity = clip_norm * 2
    batch_sampling_prob = batch_size / epoch_size
    num_queries = math.ceil(epoch_size / batch_size) * num_epochs
    steps = num_queries
    
    return {
        'image_size_tuple': image_size_tuple,
        'sensitivity': sensitivity,
        'batch_sampling_prob': batch_sampling_prob,
        'steps': steps
    }


def add_common_metadata(results, image_size, padding, fixed_delta, sensitivity, 
                       batch_sampling_prob, steps, clip_norm, batch_size, 
                       epoch_size, num_epochs):
    """Add common metadata to results dictionary."""
    results.update({
        "image_size": image_size,
        "padding": padding,
        "fixed_delta": fixed_delta,
        "sensitivity": sensitivity,
        "batch_sampling_prob": batch_sampling_prob,
        "steps": steps,
        "clip_norm": clip_norm,
        "batch_size": batch_size,
        "epoch_size": epoch_size,
        "num_epochs": num_epochs
    })
    return results
