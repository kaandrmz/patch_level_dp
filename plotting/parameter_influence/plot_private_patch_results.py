"""Plotting script for private patch parameter dependency experiments using SEML results."""

import math
import os
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

from seml.database import get_collection


def get_seml_results(collection_name: str) -> List[Dict]:
    """Retrieve experiment results from SEML database."""
    collection = get_collection(collection_name)

    results = list(collection.find({"status": "COMPLETED"}))
    print(f"Found {len(results)} completed experiments in collection '{collection_name}'")
    return results


def format_noise_label(noise_level: float) -> str:
    """Format noise level for display (remove .0 for whole numbers)."""
    if noise_level == int(noise_level):
        return str(int(noise_level))
    else:
        return str(noise_level)


def process_private_patch_results(results: List[Dict], filter_image_size: Optional[Tuple[int, int]] = None) -> Tuple[Dict, List[float], List[Tuple[int, int]]]:
    """Process private patch experiment results.
    
    Args:
        results: List of experiment results from database
        filter_image_size: If provided, only process results matching this image size
        
    Returns:
        Tuple of (results_by_noise, noise_levels, image_sizes_found)
    """
    results_by_noise = {}
    noise_levels = set()
    image_sizes_found = set()

    for result in results:
        if result.get('result', {}).get('experiment_type') != 'private_patch':
            continue

        image_size_raw = result['result'].get('image_size', [1000, 1000])
        
        if isinstance(image_size_raw, dict):
            if 'py/object' in image_size_raw and 'py/seq' in image_size_raw:
                image_size = tuple(image_size_raw['py/seq'])
            else:
                print(f"Warning: Unexpected image_size format (dict): {image_size_raw}")
                continue
        elif isinstance(image_size_raw, list):
            image_size = tuple(image_size_raw)
        elif isinstance(image_size_raw, tuple):
            image_size = image_size_raw
        else:
            try:
                image_size = tuple(image_size_raw)
            except (TypeError, ValueError):
                print(f"Warning: Could not convert image_size to tuple: {image_size_raw}")
                continue
        
        image_sizes_found.add(image_size)
        
        if filter_image_size and image_size != filter_image_size:
            continue

        noise_level = result['result']['noise_level']
        private_patch_size = result['result']['private_patch_size']
        epsilon = result['result']['epsilon']

        if noise_level not in results_by_noise:
            results_by_noise[noise_level] = {}

        results_by_noise[noise_level][private_patch_size] = epsilon
        noise_levels.add(noise_level)

    noise_levels = sorted(list(noise_levels))
    image_sizes_found = sorted(list(image_sizes_found))
    return results_by_noise, noise_levels, image_sizes_found


def calculate_baseline_epsilon(noise_level: float, common_params: Dict) -> float:
    """Calculate baseline epsilon (intersection_prob=1) for a given noise level."""
    from patch_level_dp.privacy.calculations import create_pld

    sampling_prob = common_params['batch_sampling_prob']
    sensitivity = common_params['sensitivity']
    steps = common_params['steps']
    fixed_delta = common_params['fixed_delta']

    single_step_pld = create_pld(noise_level, sensitivity, sampling_prob)
    total_pld = single_step_pld.self_compose(steps)
    epsilon = total_pld.get_epsilon_for_delta(fixed_delta)

    return epsilon


def plot_private_patch_results(
    results_by_noise: Dict,
    noise_levels: List[float],
    common_params: Dict,
    image_width: int,
    output_dir: str = "."
):
    """Create plot for private patch size vs epsilon results.

    Args:
        results_by_noise: Dictionary mapping noise levels to {private_patch_size: epsilon}
        noise_levels: List of noise levels (should be 3 for proper coloring)
        common_params: Common experiment parameters
        image_width: Image width for filename
        output_dir: Directory to save plots
    """
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
        'font.size': 30,
        'axes.labelsize': 30,
        'legend.fontsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'lines.linewidth': 2.5,
        'font.family': 'serif'
    })

    # Colors matching experiment.py
    plot_colors = ['#377eb8', '#ff7f0e', '#4daf4a']  # Blue, Orange, Green
    baseline_colors = ['#a6cee3', '#fdbf6f', '#b2df8a']  # Lighter shades

    noise_labels = [format_noise_label(nl) for nl in noise_levels]

    fig, ax = plt.subplots(figsize=(12, 10))

    for noise_idx, noise_level in enumerate(noise_levels):
        if noise_level not in results_by_noise:
            print(f"Warning: No results found for noise level {noise_level}")
            continue

        patch_sizes = sorted(results_by_noise[noise_level].keys())
        epsilons = [results_by_noise[noise_level][ps] for ps in patch_sizes]

        # Filter out infinite values
        valid_points = [(ps, eps) for ps, eps in zip(patch_sizes, epsilons) if eps != float('inf')]
        if valid_points:
            valid_patch_sizes, valid_epsilons = zip(*valid_points)
            ax.plot(valid_patch_sizes, valid_epsilons, linestyle='-',
                    color=plot_colors[noise_idx], zorder=2)
        else:
            print(f"Warning: No valid points for noise level {noise_level}")

        # Plot baseline (intersection_prob=1)
        baseline_eps = calculate_baseline_epsilon(noise_level, common_params)
        if baseline_eps != float('inf'):
            ax.axhline(y=baseline_eps, color=baseline_colors[noise_idx],
                       linestyle='--', zorder=1)

    ax.set_xlabel('Private patch side length [pixels]', labelpad=15)
    ax.set_ylabel(r'$\varepsilon$', labelpad=15)
    ax.grid(True, which="both", ls="-", alpha=0.3)

    # Create custom legends (matching experiment.py style)
    # Top-left legend for line styles
    legend1_lines = [Line2D([0], [0], color='black', lw=2, linestyle='-'),
                     Line2D([0], [0], color='black', lw=2, linestyle='--')]
    legend1_labels = ['Patch-Level Subsampling', 'Minibatch Subsampling']
    ax.legend(legend1_lines, legend1_labels, loc='upper left')

    # Bottom-right legend for noise levels
    legend2_lines = [Line2D([0], [0], color=plot_colors[i], lw=2) for i in range(len(noise_levels))]
    legend2_labels = [f'{noise_labels[i]}' for i in range(len(noise_levels))]
    leg2 = Legend(ax, legend2_lines, legend2_labels, loc='lower right', title=r'$\sigma$')
    ax.add_artist(leg2)
    leg2.get_frame().set_edgecolor('black')
    leg2.get_frame().set_alpha(0.63)
    ax.get_legend().get_frame().set_edgecolor('black')
    ax.get_legend().get_frame().set_alpha(0.63)

    plt.tight_layout()

    # Filename format matching experiment.py
    filename = f'experiment_eps_vs_priv_patch_size_delta{common_params["fixed_delta"]:.0e}_noise{noise_levels[0]}_imgw{image_width}_two.pdf'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Private patch plot saved as '{filepath}'")
    plt.close(fig)


def plot_collection(
    collection_name: str,
    output_dir: str = ".",
    noise_groups: Optional[List[List[float]]] = None
):
    """Plot results from a specific SEML collection.

    Args:
        collection_name: Name of the SEML collection
        output_dir: Directory to save plots
        noise_groups: List of noise level groups to plot separately (e.g., [[1,1.5,2], [2.5,3,3.5]])
                     If None, will plot all noise levels together (up to 3)
    """
    print(f"\n=== Plotting {collection_name} ===")
    try:
        patch_results = get_seml_results(collection_name)
        if not patch_results:
            print(f"No results found in collection '{collection_name}'")
            print(f"Make sure experiments have been run and completed:")
            print(f"  seml {collection_name} status")
            return

        _, _, image_sizes_found = process_private_patch_results(patch_results)
        
        if not image_sizes_found:
            print("No valid private patch results found!")
            return
            
        print(f"Found results for image sizes: {image_sizes_found}")

        for image_size in image_sizes_found:
            print(f"\n--- Processing image size {image_size} ---")
            
            results_by_noise, noise_levels, _ = process_private_patch_results(
                patch_results, 
                filter_image_size=image_size
            )
            
            if not results_by_noise:
                print(f"No results for image size {image_size}")
                continue

            first_result = None
            for result in patch_results:
                res_img_size_raw = result['result'].get('image_size', [1000, 1000])
                
                if isinstance(res_img_size_raw, dict) and 'py/seq' in res_img_size_raw:
                    res_img_size = tuple(res_img_size_raw['py/seq'])
                elif isinstance(res_img_size_raw, list):
                    res_img_size = tuple(res_img_size_raw)
                elif isinstance(res_img_size_raw, tuple):
                    res_img_size = res_img_size_raw
                else:
                    continue
                    
                if res_img_size == image_size:
                    first_result = result['result']
                    break
            
            if not first_result:
                print(f"Could not find result for image size {image_size}")
                continue
                
            common_params = {
                'fixed_delta': first_result['fixed_delta'],
                'sensitivity': first_result['sensitivity'],
                'batch_sampling_prob': first_result['batch_sampling_prob'],
                'steps': first_result['steps']
            }

            image_width = image_size[1] if isinstance(image_size, tuple) else image_size

            if noise_groups:
                for group in noise_groups:
                    group_results = {nl: results_by_noise[nl] for nl in group if nl in results_by_noise}
                    if group_results:
                        plot_private_patch_results(
                            group_results,
                            group,
                            common_params,
                            image_width,
                            output_dir
                        )
            else:
                if len(noise_levels) > 3:
                    print(f"Warning: Found {len(noise_levels)} noise levels, but plots are designed for 3.")
                    print("Consider using noise_groups parameter to split into groups of 3.")
                plot_private_patch_results(
                    results_by_noise,
                    noise_levels[:3] if len(noise_levels) > 3 else noise_levels,
                    common_params,
                    image_width,
                    output_dir
                )

        print(f"Plotting completed for {collection_name}!")

    except Exception as e:
        print(f"Error plotting {collection_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main plotting function for private patch experiments."""
    print("=== Plotting Private Patch Results ===")

    # Define noise level groups (as used in experiment.py)
    noise_groups = [
        [1.0, 1.5, 2.0],
        [2.5, 3.0, 3.5],
        [4.0, 4.5, 5.0]
    ]

    # Output directory for plots
    output_dir = "output/plots/parameter_influence/private_patch"
    os.makedirs(output_dir, exist_ok=True)

    # Plot collections - can use either consolidated or separate collections
    collections_to_plot = [
        "parameter_influence_private_patch_all_experiments",  # Consolidated (preferred)
        # Or separate collections:
        # "parameter_influence_private_patch_imgw1000_experiments",
        # "parameter_influence_private_patch_imgw2000_experiments"
    ]

    for collection_name in collections_to_plot:
        plot_collection(collection_name, output_dir, noise_groups)


if __name__ == "__main__":
    main()