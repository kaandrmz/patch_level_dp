"""Privacy calculations: epsilon, noise, and intersection probability calculations."""

import math
from typing import Tuple, List, Union

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


def calc_gaussian_tvd(standard_deviation: float, sensitivity: float) -> float:
    """Calculate Total Variation Distance (TVD) between N(0, sigma) and N(r, sigma).
    
    This is used for Gaussian data augmentation amplification by subsampling.
    Based on Schuchardt et al. (2025) approach.
    
    Args:
        standard_deviation: Noise standard deviation (sigma)
        sensitivity: Maximum L2 change in data (r)
        
    Returns:
        Total variation distance, a value in [0, 1]
    """
    privacy_loss = GaussianPrivacyLoss(standard_deviation, sensitivity)
    tvd = float(privacy_loss.get_delta_for_epsilon(0))
    assert 0 <= tvd <= 1, f"TVD must be in [0, 1], got {tvd}"
    return tvd


def generate_rectangle_pixels(width: int, height: int) -> List[Tuple[int, int]]:
    """Generate all pixel coordinates for a rectangle."""
    return [(x, y) for x in range(width) for y in range(height)]


def generate_circle_pixels(radius: int) -> List[Tuple[int, int]]:
    """Generate pixel coordinates for a filled circle.
    
    Args:
        radius: Radius of the circle
        
    Returns:
        List of (x, y) pixel coordinates within the circle
    """
    pixels = []
    # The circle is centered at (radius, radius) in a (2*radius+1) x (2*radius+1) box
    for x in range(2 * radius + 1):
        for y in range(2 * radius + 1):
            # Check if pixel is within circle: (x - radius)^2 + (y - radius)^2 <= radius^2
            if (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2:
                pixels.append((x, y))
    return pixels


def generate_crescent_pixels(size: int) -> List[Tuple[int, int]]:
    """Generate pixel coordinates for a crescent/blob shape.
    
    Creates a crescent by taking a circle and subtracting a smaller offset circle.
    The result has approximately size^2 pixels.
    
    Args:
        size: Size parameter that determines the scale of the crescent
        
    Returns:
        List of (x, y) pixel coordinates forming the crescent
    """
    # Create a crescent with approximately size^2 pixels
    # Use a larger circle with a cut-out offset circle
    # For size=10, we want ~100 pixels
    
    # Main circle radius to get approximately size^2 pixels in the crescent
    # A full circle of radius r has π*r^2 pixels, crescent has roughly half
    # So we want 2*size^2/π ≈ 0.64*size^2, meaning r ≈ size*sqrt(2/π) ≈ 0.8*size
    main_radius = int(size * 0.9)
    
    # Offset and radius for the cut-out circle
    cutout_radius = int(main_radius * 0.6)
    offset_x = int(main_radius * 0.4)  # Shift cutout circle to create crescent
    offset_y = 0
    
    pixels = []
    box_size = 2 * main_radius + 1
    center = main_radius
    
    for x in range(box_size):
        for y in range(box_size):
            # Check if pixel is in main circle
            dist_main = (x - center) ** 2 + (y - center) ** 2
            in_main_circle = dist_main <= main_radius ** 2
            
            # Check if pixel is in cutout circle (offset from center)
            dist_cutout = (x - center - offset_x) ** 2 + (y - center - offset_y) ** 2
            in_cutout_circle = dist_cutout <= cutout_radius ** 2
            
            # Pixel is in crescent if it's in main circle but not in cutout
            if in_main_circle and not in_cutout_circle:
                pixels.append((x, y))
    
    return pixels


def generate_dual_circle_pixels(size: int) -> List[Tuple[int, int]]:
    """Generate pixel coordinates for two circles side by side (distributed cluster).
    
    Creates two circles with approximately size^2/2 pixels each, for a total of ~size^2 pixels.
    The circles are separated by a small gap.
    
    Args:
        size: Size parameter that determines the scale (total area ≈ size^2)
        
    Returns:
        List of (x, y) pixel coordinates forming the dual circle cluster
    """
    # For size=10, we want ~100 pixels total, so ~50 pixels per circle
    # π*r^2 = size^2/2, so r = size/sqrt(2π) ≈ size * 0.4
    import math
    radius = int(size * 0.4)
    
    # Spacing between circle centers
    gap = radius // 2  # Small gap between circles
    center_offset = radius + gap // 2
    
    pixels = []
    # Box needs to fit both circles side by side
    box_width = 2 * center_offset + 2 * radius + 1
    box_height = 2 * radius + 1
    
    # Left circle center
    left_center_x = radius
    center_y = radius
    
    # Right circle center
    right_center_x = radius + 2 * center_offset
    
    for x in range(box_width):
        for y in range(box_height):
            # Check if pixel is in left circle
            dist_left = (x - left_center_x) ** 2 + (y - center_y) ** 2
            in_left_circle = dist_left <= radius ** 2
            
            # Check if pixel is in right circle
            dist_right = (x - right_center_x) ** 2 + (y - center_y) ** 2
            in_right_circle = dist_right <= radius ** 2
            
            # Add pixel if it's in either circle
            if in_left_circle or in_right_circle:
                pixels.append((x, y))
    
    return pixels


def get_patch_pixels_and_bbox(shape: str, size: Union[int, Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """Generate pixel coordinates and bounding box for a privacy patch shape.
    
    Args:
        shape: Shape name - "square", "rectangle", "circle", "blob", or "cluster"
        size: Size parameter that determines the scale of the shape
              - For rectangles: int uses hardcoded 2:1 ratio, tuple (width, height) for custom dimensions
              - For other shapes: int for size, tuple (width, height) to override default size calculation
        
    Returns:
        Tuple of (pixel_list, bounding_box) where:
        - pixel_list: List of (x, y) coordinates of pixels in the shape
        - bounding_box: (width, height) tuple of the bounding box
    """
    if shape == "square":
        # Square with equal width and height (size x size)
        if isinstance(size, tuple):
            width, height = size
        else:
            width = size
            height = size
        pixels = generate_rectangle_pixels(width, height)
        return pixels, (width, height)
    
    elif shape == "rectangle":
        # Rectangle: if tuple provided, use exact dimensions; otherwise use 2:1 aspect ratio
        if isinstance(size, tuple):
            width, height = size
        else:
            # Default 2:1 aspect ratio (2*size wide, size tall)
            width = 2 * size
            height = size
        pixels = generate_rectangle_pixels(width, height)
        return pixels, (width, height)
    
    elif shape == "circle":
        # Circle with radius = size // 2
        if isinstance(size, tuple):
            # If tuple provided, use first value as radius
            radius = size[0] // 2
        else:
            radius = size // 2
        pixels = generate_circle_pixels(radius)
        bbox_size = 2 * radius + 1
        return pixels, (bbox_size, bbox_size)
    
    elif shape == "blob":
        # Crescent/blob with approximately size^2 pixels
        if isinstance(size, tuple):
            # If tuple provided, use first value for blob generation
            blob_size = size[0]
        else:
            blob_size = size
        pixels = generate_crescent_pixels(blob_size)
        # Calculate actual bounding box from pixels
        if not pixels:
            return pixels, (0, 0)
        max_x = max(x for x, y in pixels)
        max_y = max(y for x, y in pixels)
        return pixels, (max_x + 1, max_y + 1)
    
    elif shape == "cluster":
        # Dual circle cluster with approximately size^2 pixels total
        if isinstance(size, tuple):
            # If tuple provided, use first value for cluster generation
            cluster_size = size[0]
        else:
            cluster_size = size
        pixels = generate_dual_circle_pixels(cluster_size)
        # Calculate actual bounding box from pixels
        if not pixels:
            return pixels, (0, 0)
        max_x = max(x for x, y in pixels)
        max_y = max(y for x, y in pixels)
        return pixels, (max_x + 1, max_y + 1)
    
    else:
        raise ValueError(f"Unknown shape: {shape}. Must be 'square', 'rectangle', 'circle', 'blob', or 'cluster'")


def calc_intersection_prob(
    image_size: Tuple[int, int],
    crop_patch_size: Tuple[int, int],
    private_patch_pixels: List[Tuple[int, int]],
    private_patch_bbox: Tuple[int, int],
    padding: int = 0,
) -> float:
    """Calculate the worst case probability of intersection between the crop patch and the private patch.
    
    Args:
        image_size: (height, width) of the image
        crop_patch_size: (height, width) of the crop patch
        private_patch_pixels: List of (x, y) pixel coordinates in the private patch (relative to patch anchor)
        private_patch_bbox: (width, height) bounding box of the private patch
        padding: Padding around the image
        
    Returns:
        Maximum intersection probability across all possible private patch positions
    """
    
    def _intersection_prob_fix_location(
        image_size, crop_patch_size, private_patch_pixels, private_patch_bbox, padding, private_patch_loc
    ):
        """Calculate intersection probability for a fixed private patch location.
        
        Uses fast geometric calculation for rectangles, optimized set-based for other shapes.
        """
        image_h, image_w = image_size
        image_h, image_w = image_h + 2 * padding, image_w + 2 * padding
        crop_h, crop_w = crop_patch_size
        
        # Calculate valid crop position ranges
        Sx = image_w - crop_w
        Sy = image_h - crop_h
        total_positions = (Sx + 1) * (Sy + 1)
        if total_positions == 0:
            return 1.0

        priv_x, priv_y = private_patch_loc
        priv_x += padding
        priv_y += padding

        bbox_w, bbox_h = private_patch_bbox
        
        # Check if this is a rectangle (all pixels fill the bounding box)
        is_rectangle = len(private_patch_pixels) == bbox_w * bbox_h
        
        if is_rectangle:
            # FAST PATH: Use geometric calculation for rectangles
            priv_w, priv_h = bbox_w, bbox_h
            
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
        
        else:
            # OPTIMIZED PATH: Set-based approach for non-rectangular shapes (circle, blob)
            # Instead of iterating all crop positions, for each pixel we directly calculate
            # which crop positions would include it, then use a set to avoid double-counting.
            # Complexity: O(num_pixels × crop_size²) vs old O(crop_positions × num_pixels)
            # For 1000×1000 image with 100×100 crop: ~160x faster!
            intersecting_crops = set()
            
            for dx, dy in private_patch_pixels:
                pixel_abs_x = priv_x + dx
                pixel_abs_y = priv_y + dy
                
                # For this pixel, calculate which crop positions would include it
                # Crop at (cx, cy) covers [cx, cx+crop_w) × [cy, cy+crop_h)
                # So pixel at (pixel_abs_x, pixel_abs_y) is included if:
                #   cx <= pixel_abs_x < cx+crop_w  =>  pixel_abs_x-crop_w+1 <= cx <= pixel_abs_x
                #   cy <= pixel_abs_y < cy+crop_h  =>  pixel_abs_y-crop_h+1 <= cy <= pixel_abs_y
                
                cx_min = max(0, pixel_abs_x - crop_w + 1)
                cx_max = min(Sx, pixel_abs_x)
                cy_min = max(0, pixel_abs_y - crop_h + 1)
                cy_max = min(Sy, pixel_abs_y)
                
                for cx in range(cx_min, cx_max + 1):
                    for cy in range(cy_min, cy_max + 1):
                        intersecting_crops.add((cx, cy))
            
            return len(intersecting_crops) / total_positions

    # Iterate over all possible private patch positions
    bbox_w, bbox_h = private_patch_bbox
    max_intersection_prob = 0
    
    # Calculate total iterations for progress tracking
    total_priv_positions = (image_size[1] - bbox_w + padding + 1) * (image_size[0] - bbox_h + padding + 1)
    is_rectangle = len(private_patch_pixels) == bbox_w * bbox_h
    
    # Log calculation start
    import sys
    print(f"[calc_intersection_prob] Starting calculation...", flush=True)
    print(f"  Image: {image_size}, Crop: {crop_patch_size}, Patch pixels: {len(private_patch_pixels)}, BBox: {private_patch_bbox}", flush=True)
    print(f"  Total private patch positions to check: {total_priv_positions:,}", flush=True)
    print(f"  Using {'FAST geometric' if is_rectangle else 'OPTIMIZED set-based'} algorithm", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Private patch anchor can be placed anywhere such that the bounding box fits in the image
    position_count = 0
    report_interval = max(1, total_priv_positions // 20)  # Report every 5%
    
    for x in range(padding, image_size[1] - bbox_w + padding + 1):
        for y in range(padding, image_size[0] - bbox_h + padding + 1):
            curr_intersection_prob = _intersection_prob_fix_location(
                image_size,
                crop_patch_size,
                private_patch_pixels,
                private_patch_bbox,
                padding,
                private_patch_loc=(x, y),
            )
            max_intersection_prob = max(max_intersection_prob, curr_intersection_prob)
            
            position_count += 1
            if position_count % report_interval == 0:
                progress_pct = 100.0 * position_count / total_priv_positions
                import sys
                print(f"  Progress: {position_count:,}/{total_priv_positions:,} ({progress_pct:.1f}%) - Max prob so far: {max_intersection_prob:.6f}", flush=True)
                sys.stdout.flush()
    
    import sys
    print(f"[calc_intersection_prob] COMPLETED! Final max intersection probability: {max_intersection_prob:.6f}", flush=True)
    sys.stdout.flush()
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
    private_patch_size: Union[int, Tuple[int, int]],
    padding: int,
    baseline_privacy: bool = False,
    shape: str = "square",
    gaussian_augmentation: bool = False,
    gaussian_augmentation_sigma: float = None,
) -> Tuple[float, float]:
    """Calculates the noise (standard deviation) for the switching mechanism.
    
    For gaussian_augmentation mode, this function iteratively searches for the noise level
    that satisfies the privacy budget using binary search.
    """
    if gaussian_augmentation:
        def get_epsilon_for_noise(noise_level):
            """Helper to calculate epsilon for a given noise level."""
            sampling_prob = _calc_sampling_prob(
                image_size=image_size,
                crop_size=crop_size,
                private_patch_size=private_patch_size,
                padding=padding,
                batch_sampling_prob=batch_sampling_prob,
                baseline_privacy=baseline_privacy,
                shape=shape,
                gaussian_augmentation=gaussian_augmentation,
                gaussian_augmentation_sigma=noise_level,
            )
            
            pld = create_pld(
                standard_deviation=noise_level,
                sensitivity=sensitivity,
                sampling_prob=sampling_prob,
            )
            composed_pld = pld.self_compose(num_queries)
            return composed_pld.get_epsilon_for_delta(delta), sampling_prob
        
        # Binary search for the noise level
        lower_noise = 0.001
        upper_noise = 100000.0  # Very large noise
        tolerance = 0.01
        
        print(f"Searching for noise level to achieve epsilon={epsilon} with delta={delta}...")
        
        while upper_noise - lower_noise > tolerance:
            mid_noise = (lower_noise + upper_noise) / 2
            mid_epsilon, mid_sampling_prob = get_epsilon_for_noise(mid_noise)
            
            if mid_epsilon > epsilon:
                lower_noise = mid_noise
            else:
                upper_noise = mid_noise
        
        final_noise = (lower_noise + upper_noise) / 2
        final_epsilon, final_sampling_prob = get_epsilon_for_noise(final_noise)
        print(f"Found noise level: {final_noise:.4f} (achieved epsilon: {final_epsilon:.4f})")
        
        return final_noise, final_sampling_prob
    
    sampling_prob = _calc_sampling_prob(
        image_size=image_size,
        crop_size=crop_size,
        private_patch_size=private_patch_size,
        padding=padding,
        batch_sampling_prob=batch_sampling_prob,
        baseline_privacy=baseline_privacy,
        shape=shape,
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
    private_patch_size: Union[int, Tuple[int, int]],
    padding: int,
    baseline_privacy: bool = False,
    shape: str = "square",
    gaussian_augmentation: bool = False,
) -> float:
    """Calculates the epsilon for a given noise (standard deviation)."""
    sampling_prob = _calc_sampling_prob(
        image_size=image_size,
        crop_size=crop_size,
        private_patch_size=private_patch_size,
        padding=padding,
        batch_sampling_prob=batch_sampling_prob,
        baseline_privacy=baseline_privacy,
        shape=shape,
        gaussian_augmentation=gaussian_augmentation,
        gaussian_augmentation_sigma=standard_deviation,
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
    private_patch_size: Union[int, Tuple[int, int]],
    padding: int,
    batch_sampling_prob: float,
    baseline_privacy: bool = False,
    shape: str = "square",
    gaussian_augmentation: bool = False,
    gaussian_augmentation_sigma: float = None,
) -> float:
    """Helper to calculate the effective sampling probability.
    
    Three mutually exclusive modes:
    1. baseline_privacy=True: Standard DP-SGD (batch subsampling only)
    2. gaussian_augmentation=True: Gaussian data augmentation (batch x TVD)
    3. Both False (default): Patch-level DP (batch x intersection probability)
    
    Args:
        shape: Shape of the private patch - "square", "rectangle", "circle", or "blob". 
               Defaults to "square" for backward compatibility.
        gaussian_augmentation: If True, uses Gaussian data augmentation baseline (Schuchardt et al. 2025).
        gaussian_augmentation_sigma: Noise standard deviation for Gaussian augmentation.
    """
    # Validate mutually exclusive modes
    if baseline_privacy and gaussian_augmentation:
        raise ValueError(
            "baseline_privacy and gaussian_augmentation cannot both be True. "
            "These are mutually exclusive privacy modes:\n"
            "  - baseline_privacy=True: Standard DP-SGD\n"
            "  - gaussian_augmentation=True: Gaussian data augmentation baseline\n"
            "  - Both False: Patch-level DP (your method)"
        )
    
    if baseline_privacy: # MODE 1: STANDARD DP-SGD
        sampling_prob = batch_sampling_prob
        print(f"Baseline privacy: using minibatch sampling instead of patch-level sampling")
        return sampling_prob
    
    if gaussian_augmentation: # MODE 2: GAUSSIAN DATA AUGMENTATION (Schuchardt et al. 2025)
        # Use private_patch_size for sensitivity (for fair comparison with patch-level DP)
        # This represents the L2 norm of changes within the private patch
        if isinstance(private_patch_size, tuple):
            patch_h, patch_w = private_patch_size
        else:
            patch_h = private_patch_size
            patch_w = private_patch_size
        
        # Max L2 change in [0, 255] scale
        # For RGB images: sqrt(height * width * 3 channels) * 255 (max pixel value)
        max_l2_change = math.sqrt(patch_h * patch_w * 3) * 255
        size_description = f"private patch {patch_h}×{patch_w}"
        
        if gaussian_augmentation_sigma is None:
            raise ValueError("gaussian_augmentation_sigma must be provided when gaussian_augmentation=True")
        
        tvd = calc_gaussian_tvd(gaussian_augmentation_sigma, max_l2_change)
        
        sampling_prob = batch_sampling_prob * tvd
        
        print(f"Gaussian data augmentation baseline (Schuchardt et al. 2025)")
        print(f"Using {size_description}, Max L2 change: {max_l2_change:.2f}")
        print(f"Noise sigma: {gaussian_augmentation_sigma:.4f}, TVD: {tvd:.6f}")
        print(f"Effective Sampling Probability: {sampling_prob:.6f}")
        return sampling_prob
    
    # MODE 3: PATCH-LEVEL DP (our method)
    # Generate pixel representation based on shape
    # private_patch_size can be int or tuple - get_patch_pixels_and_bbox handles both
    private_patch_pixels, private_patch_bbox = get_patch_pixels_and_bbox(shape, private_patch_size)
    
    intersection_prob = calc_intersection_prob(
        image_size=image_size,
        crop_patch_size=(crop_size, crop_size),
        private_patch_pixels=private_patch_pixels,
        private_patch_bbox=private_patch_bbox,
        padding=padding,
    )
    print(f"Intersection Probability: {intersection_prob}")

    sampling_prob = intersection_prob * batch_sampling_prob # OUR ANALYSIS
    print(f"Effective Sampling Probability: {sampling_prob}")
    return sampling_prob


def calc_sampling_prob(
    image_size: Tuple[int, int],
    crop_size: int,
    private_patch_size: Union[int, Tuple[int, int]],
    padding: int,
    batch_sampling_prob: float,
    baseline_privacy: bool = False,
    shape: str = "square",
    gaussian_augmentation: bool = False,
    gaussian_augmentation_sigma: float = None,
) -> float:
    """Calculates the effective sampling probability."""
    return _calc_sampling_prob(
        image_size=image_size,
        crop_size=crop_size,
        private_patch_size=private_patch_size,
        padding=padding,
        batch_sampling_prob=batch_sampling_prob,
        baseline_privacy=baseline_privacy,
        shape=shape,
        gaussian_augmentation=gaussian_augmentation,
        gaussian_augmentation_sigma=gaussian_augmentation_sigma,
    )


# Define sampling methods for privacy profile experiments
SAMPLING_METHODS = ['intersection', 'batch_only', 'full_sampling']


def calculate_delta_for_epsilon_multi(args):
    """Calculate delta for a single epsilon value with specified sampling method and noise level."""
    epsilon, image_size, padding, crop_size, private_patch_size, batch_sampling_prob, standard_deviation, sensitivity, steps, sampling_method, full_sampling_std_dev, shape, intersection_prob = args
    
    current_sensitivity = sensitivity
    current_steps = steps
    current_std_dev = standard_deviation

    if sampling_method == 'intersection':
        # Use pre-calculated intersection probability and multiply with batch probability (our method)
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