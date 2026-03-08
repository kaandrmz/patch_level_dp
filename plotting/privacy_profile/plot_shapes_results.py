"""Plotting script for privacy profile shapes experiments using SEML results."""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from seml.database import get_collection
from patch_level_dp.privacy.calculations import SAMPLING_METHODS

METHOD_LABELS = ['Patch-Level Subsampling', 'Minibatch Subsampling', 'Data-Level Noise']

# Shape labels for nice titles
SHAPE_LABELS = {
    'square': 'Square',
    'rectangle': 'Rectangle',
    'circle': 'Circle',
    'blob': 'Blob',
    'cluster': 'Cluster'
}


def get_seml_results(collection_name: str) -> List[Dict]:
    """Retrieve experiment results from SEML database."""
    collection = get_collection(collection_name)

    results = list(collection.find({"status": "COMPLETED"}))
    print(f"Found {len(results)} completed experiments in collection '{collection_name}'")
    return results


def process_shapes_results(results: List[Dict]) -> Dict:
    """Process privacy profile shapes experiment results.
    
    Returns a dictionary keyed by shape, containing the privacy profile data.
    Filters for 30 epsilon points and removes duplicates.
    """
    processed_results = {}

    for result in results:
        if 'result' not in result:
            continue

        result_data = result['result']
        
        # Only process experiments with 30 epsilon points
        if result_data.get('num_epsilon_points', 0) != 30:
            print(f"Skipping experiment with {result_data.get('num_epsilon_points')} epsilon points")
            continue

        shape = result_data.get('priv_patch_shape', 'unknown')
        
        # Skip if we already have this shape (removes duplicates)
        if shape in processed_results:
            print(f"Duplicate found for shape '{shape}', keeping first one")
            continue

        # Extract privacy profile results
        privacy_results = {}
        for key, delta_value in result_data['privacy_profile_results'].items():
            epsilon_str, method = key.split(',')
            epsilon = float(epsilon_str)

            if method not in privacy_results:
                privacy_results[method] = {}

            # Handle different value formats
            if isinstance(delta_value, dict) and 'value' in delta_value:
                delta_value = delta_value['value']

            privacy_results[method][epsilon] = delta_value

        processed_results[shape] = {
            'privacy_profiles': privacy_results,
            'metadata': {
                'experiment_set': result_data.get('experiment_set', 'main'),
                'intersection_prob': result_data.get('intersection_prob', 0),
                'epsilon_min': result_data.get('epsilon_min', 0.1),
                'epsilon_max': result_data.get('epsilon_max', 1000.0),
                'private_patch_size': result_data.get('private_patch_size', 10),
                'crop_size': result_data.get('crop_size', 100),
                'noise_level': result_data.get('noise_level', 1.0),
            }
        }

    return processed_results


def plot_shape_privacy_profile(shape: str, data: Dict, output_dir: str = "."):
    """Create privacy profile plot for a single shape showing all three methods."""
    
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

    plt.figure(figsize=(12, 8))

    colors = ['#377eb8', '#ff7f0e', '#4daf4a']

    privacy_results = data['privacy_profiles']

    # Plot each method
    for method_idx, method in enumerate(SAMPLING_METHODS):
        if method not in privacy_results:
            print(f"Warning: Method '{method}' not found in results for {shape}")
            continue

        method_data = privacy_results[method]
        epsilons = sorted(method_data.keys())
        deltas = [method_data[eps] for eps in epsilons]

        if epsilons:
            plt.semilogx(epsilons, deltas,
                         color=colors[method_idx],
                         linewidth=2.5,
                         label=METHOD_LABELS[method_idx],
                         alpha=0.8)

    plt.xlim(1e-1, 1e3)
    plt.xlabel(r'$\varepsilon$', labelpad=15)
    plt.ylabel(r'$\delta(\varepsilon)$', labelpad=15)

    plt.grid(True, alpha=0.3)
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_alpha(0.63)

    plt.tight_layout()

    # Create filename based on shape
    metadata = data['metadata']
    shape_label = SHAPE_LABELS.get(shape, shape)
    filename = f'privacy_profile_shape_{shape}.pdf'

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Privacy profile plot for {shape_label} saved as '{filepath}'")
    print(f"  Intersection probability: {metadata['intersection_prob']:.6f}")
    print(f"  Private patch size: {metadata['private_patch_size']}")
    print(f"  Crop size: {metadata['crop_size']}")
    
    return filename


def main():
    """Main plotting function."""
    output_dir = "output/plots/privacy_profile_shapes"

    print("=== Processing Privacy Profile Shapes Experiments ===")
    shapes_results = get_seml_results("rebut_priv_prof_shapes")
    
    if not shapes_results:
        print("No results found in rebut_priv_prof_shapes collection")
        return

    processed_shapes = process_shapes_results(shapes_results)
    print(f"\nProcessing {len(processed_shapes)} unique shape configurations")
    print(f"Shapes found: {list(processed_shapes.keys())}")
    
    # Expected shapes
    expected_shapes = ['square', 'rectangle', 'circle', 'blob', 'cluster']
    
    # Plot each shape
    for shape in expected_shapes:
        if shape in processed_shapes:
            print(f"\n--- Plotting {SHAPE_LABELS.get(shape, shape)} ---")
            plot_shape_privacy_profile(shape, processed_shapes[shape], output_dir)
        else:
            print(f"\n--- Shape '{shape}' not found (may still be running) ---")

    print("\n" + "=" * 80)
    print("All privacy profile shape plots completed!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

