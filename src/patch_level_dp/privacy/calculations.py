"""Privacy calculations: epsilon, noise, and intersection probability calculations."""

import math
from typing import Tuple

from dp_accounting.pld import common
from dp_accounting.pld.accountant import (
    get_smallest_gaussian_noise,
    get_smallest_parameter,
)
from dp_accounting.pld.privacy_loss_distribution import (
    PrivacyLossDistribution,
    _create_pld_pmf_from_additive_noise,
)
from dp_accounting.pld.privacy_loss_mechanism import (
    AdjacencyType,
    GaussianPrivacyLoss,
)

from .pld import PldPmf, SwitchingPrivacyLoss
import dp_accounting.pld.pld_pmf

dp_accounting.pld.pld_pmf.SparsePLDPmf = PldPmf


def calc_intersection_prob(
    image_size: Tuple[int, int],
    crop_patch_size: Tuple[int, int],
    private_patch_size: Tuple[int, int],
    padding: int = 0,
) -> float:
    """Calculate the worst case probability of intersection between the crop patch and the private patch."""
    
    def _intersection_prob_fix_location(
        image_size, crop_patch_size, private_patch_size, padding, private_patch_loc
    ):
        image_h, image_w = image_size
        image_h, image_w = image_h + 2 * padding, image_w + 2 * padding
        priv_h, priv_w = private_patch_size
        crop_h, crop_w = crop_patch_size
        Sx = image_w - crop_w
        Sy = image_h - crop_h
        total_positions = (Sx + 1) * (Sy + 1)
        if total_positions == 0:
            return 1.0

        priv_x, priv_y = private_patch_loc
        priv_x += padding
        priv_y += padding

        min_intersect_x = max(0, priv_x - crop_w + 1)
        max_intersect_x = min(Sx, priv_x + priv_w - 1)
        min_intersect_y = max(0, priv_y - crop_h + 1)
        max_intersect_y = min(Sy, priv_y + priv_h - 1)

        if max_intersect_x < min_intersect_x or max_intersect_y < min_intersect_y:
            return 0.0

        intersect_positions = (max_intersect_x - min_intersect_x + 1) * (
            max_intersect_y - min_intersect_y + 1
        )
        return intersect_positions / total_positions

    max_intersection_prob = 0
    for x in range(padding, image_size[1] - private_patch_size[1] + padding + 1):
        for y in range(padding, image_size[0] - private_patch_size[0] + padding + 1):
            curr_intersection_prob = _intersection_prob_fix_location(
                image_size,
                crop_patch_size,
                private_patch_size,
                padding,
                private_patch_loc=(x, y),
            )
            max_intersection_prob = max(max_intersection_prob, curr_intersection_prob)
    return max_intersection_prob


def create_pld(
    standard_deviation: float,
    sensitivity: float,
    sampling_prob: float
) -> PrivacyLossDistribution:
    """Create a privacy loss distribution for the switching mechanism."""
    switching_loss = SwitchingPrivacyLoss(
        epsilon_threshold=0.0,
        below_threshold_pl=GaussianPrivacyLoss(
            standard_deviation=standard_deviation,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            adjacency_type=AdjacencyType.ADD,
        ),
        above_threshold_pl=GaussianPrivacyLoss(
            standard_deviation=standard_deviation,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            adjacency_type=AdjacencyType.REMOVE,
        ),
    )
    pld_pmf = _create_pld_pmf_from_additive_noise(switching_loss, use_connect_dots=True)
    return PrivacyLossDistribution(pld_pmf)


def get_smallest_noise_for_switching(
    privacy_parameters: common.DifferentialPrivacyParameters,
    sensitivity: int,
    sampling_prob: float,
    num_queries: int = 1,
) -> float:
    """Get the smallest noise for the switching mechanism."""
    upper_bound = get_smallest_gaussian_noise(
        privacy_parameters, num_queries, sensitivity
    )
    search_parameters = common.BinarySearchParameters(0, upper_bound)

    def create_pld_wrapper(standard_deviation):
        return create_pld(standard_deviation, sensitivity, sampling_prob)

    parameter = get_smallest_parameter(
        privacy_parameters, num_queries, create_pld_wrapper, search_parameters
    )
    if parameter is not None:
        return parameter
    else:
        print("Warning: No noise found for switching pl, returning upper bound!")
        return upper_bound


def calc_noise(
    epsilon: float,
    delta: float,
    sensitivity: float,
    batch_sampling_prob: float,
    num_queries: int,
    image_size: Tuple[int, int],
    crop_size: int,
    private_patch_size: Tuple[int, int],
    padding: int,
    baseline_privacy: bool = False,
) -> Tuple[float, float]:
    """Calculates the noise (standard deviation) for the switching mechanism."""
    sampling_prob = _calc_sampling_prob(
        image_size=image_size,
        crop_size=crop_size,
        private_patch_size=private_patch_size,
        padding=padding,
        batch_sampling_prob=batch_sampling_prob,
        baseline_privacy=baseline_privacy,
    )

    noise = get_smallest_noise_for_switching(
        privacy_parameters=common.DifferentialPrivacyParameters(epsilon, delta),
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        num_queries=num_queries,
    )
    return noise, sampling_prob


def calc_epsilon(
    standard_deviation: float,
    delta: float,
    sensitivity: float,
    batch_sampling_prob: float,
    num_queries: int,
    image_size: Tuple[int, int],
    crop_size: int,
    private_patch_size: Tuple[int, int],
    padding: int,
    baseline_privacy: bool = False,
) -> float:
    """Calculates the epsilon for a given noise (standard deviation)."""
    sampling_prob = _calc_sampling_prob(
        image_size=image_size,
        crop_size=crop_size,
        private_patch_size=private_patch_size,
        padding=padding,
        batch_sampling_prob=batch_sampling_prob,
        baseline_privacy=baseline_privacy,
    )

    pld = create_pld(
        standard_deviation=standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
    )

    composed_pld = pld.self_compose(num_queries)

    return composed_pld.get_epsilon_for_delta(delta)


def _calc_sampling_prob(
    image_size: Tuple[int, int],
    crop_size: int,
    private_patch_size: Tuple[int, int],
    padding: int,
    batch_sampling_prob: float,
    baseline_privacy: bool = False,
) -> float:
    """Helper to calculate the effective sampling probability."""
    if baseline_privacy: # CLASSIC ANALYSIS
        sampling_prob = batch_sampling_prob
        print(f"Baseline privacy: using minibatch sampling instead of patch-level sampling")
        return sampling_prob
    
    intersection_prob = calc_intersection_prob(
        image_size=image_size,
        crop_patch_size=(crop_size, crop_size),
        private_patch_size=private_patch_size,
        padding=padding,
    )
    print(f"Intersection Probability: {intersection_prob}")

    sampling_prob = intersection_prob * batch_sampling_prob # OUR ANALYSIS
    print(f"Effective Sampling Probability: {sampling_prob}")
    return sampling_prob


def calc_sampling_prob(
    image_size: Tuple[int, int],
    crop_size: int,
    private_patch_size: Tuple[int, int],
    padding: int,
    batch_sampling_prob: float,
    baseline_privacy: bool = False,
) -> float:
    """Calculates the effective sampling probability."""
    return _calc_sampling_prob(
        image_size=image_size,
        crop_size=crop_size,
        private_patch_size=private_patch_size,
        padding=padding,
        batch_sampling_prob=batch_sampling_prob,
        baseline_privacy=baseline_privacy,
    )


# Define sampling methods for privacy profile experiments
SAMPLING_METHODS = ['intersection', 'batch_only', 'full_sampling']


def calculate_delta_for_epsilon_multi(args):
    """Calculate delta for a single epsilon value with specified sampling method and noise level."""
    epsilon, image_size, padding, crop_size, private_patch_size, batch_sampling_prob, standard_deviation, sensitivity, steps, sampling_method, full_sampling_std_dev = args
    
    current_sensitivity = sensitivity
    current_steps = steps
    current_std_dev = standard_deviation

    if sampling_method == 'intersection':
        # Calculate intersection probability and multiply with batch probability (our method)
        intersection_prob = calc_intersection_prob(
            image_size=image_size, 
            crop_patch_size=(crop_size, crop_size), 
            private_patch_size=(private_patch_size, private_patch_size), 
            padding=padding
        )
        sampling_prob = batch_sampling_prob * intersection_prob
    elif sampling_method == 'batch_only':
        # Use only batch probability (baseline method)
        sampling_prob = batch_sampling_prob
    elif sampling_method == 'full_sampling':
        # Use sampling probability of 1.0, don't compose, and use a different sensitivity
        sampling_prob = 1.0
        current_sensitivity = math.sqrt(private_patch_size * private_patch_size * 3) * 255
        current_steps = 1
        # The provided standard deviation (noise) is too small for this high sensitivity, causing memory errors.
        # We override it here with a fixed value to make the calculation feasible.
        current_std_dev = full_sampling_std_dev

    else:
        raise ValueError(f"Unknown sampling_method: {sampling_method}")
    
    # Create PLD and calculate delta
    single_step_pld = create_pld(
        standard_deviation=current_std_dev,
        sensitivity=current_sensitivity,
        sampling_prob=sampling_prob,
    )
    
    if current_steps > 1:
        total_pld = single_step_pld.self_compose(current_steps)
    else:
        total_pld = single_step_pld
        
    delta = total_pld.get_delta_for_epsilon(epsilon)
        
    return delta