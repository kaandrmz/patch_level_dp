"""Plotting script for privacy profile experiments using SEML results."""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from seml.database import get_collection
from patch_level_dp.privacy.calculations import SAMPLING_METHODS

METHOD_LABELS = ['Patch-Level Subsampling', 'Minibatch Subsampling', 'Data-Level Noise']


def get_seml_results(collection_name: str) -> List[Dict]:
    """Retrieve experiment results from SEML database."""
    collection = get_collection(collection_name)

    results = list(collection.find({"status": "COMPLETED"}))
    print(f"Found {len(results)} completed experiments in collection '{collection_name}'")
    return results


def process_privacy_profile_results(results: List[Dict]) -> Dict:
    """Process privacy profile experiment results."""
    processed_results = {}

    for result in results:
        if 'result' not in result:
            continue

        result_data = result['result']

        image_size_raw = result_data['image_size']
        if isinstance(image_size_raw, dict) and 'py/seq' in image_size_raw:
            image_size = tuple(image_size_raw['py/seq'])
        elif isinstance(image_size_raw, list):
            image_size = tuple(image_size_raw)
        elif isinstance(image_size_raw, tuple):
            image_size = image_size_raw
        else:
            print(f"Warning: Unexpected image_size format: {image_size_raw}")
            continue

        config_key = (
            result_data['crop_size'],
            result_data['private_patch_size'],
            result_data['noise_level'],
            result_data['sigma_data'],
            image_size
        )

        privacy_results = {}
        for key, delta_value in result_data['privacy_profile_results'].items():
            epsilon_str, method = key.split(',')
            epsilon = float(epsilon_str)

            if method not in privacy_results:
                privacy_results[method] = {}

            if isinstance(delta_value, dict) and 'value' in delta_value:
                delta_value = delta_value['value']

            privacy_results[method][epsilon] = delta_value

        processed_results[config_key] = {
            'privacy_profiles': privacy_results,
            'metadata': {
                'experiment_set': result_data['experiment_set'],
                'intersection_prob': result_data['intersection_prob'],
                'epsilon_min': result_data['epsilon_min'],
                'epsilon_max': result_data['epsilon_max']
            }
        }

    return processed_results


def plot_privacy_profile(config_key: Tuple, data: Dict, output_dir: str = "."):
    """Create privacy profile plot for a single configuration."""
    crop_size, private_patch_size, noise_level, sigma_data, image_size = config_key

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

    for method_idx, method in enumerate(SAMPLING_METHODS):
        if method not in privacy_results:
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

    if image_size == (1000, 1000):
        filename = f'privacy_profile_crop{crop_size}_priv{private_patch_size}_sdata{int(sigma_data)}_noise{noise_level}.pdf'
    else:
        filename = f'privacy_profile_imgw{image_size[1]}_crop{crop_size}_priv{private_patch_size}_sdata{int(sigma_data)}_noise{noise_level}.pdf'

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Privacy profile plot saved as '{filepath}'")
    return filename


def main():
    """Main plotting function."""
    output_dir = "output/plots/privacy_profile"

    print("=== Processing Main Privacy Profile Experiments ===")
    main_results = get_seml_results("privacy_profile_experiments")
    if main_results:
        processed_main = process_privacy_profile_results(main_results)
        print(f"Processing {len(processed_main)} main experiment configurations")
        for config_key, data in processed_main.items():
            plot_privacy_profile(config_key, data, output_dir)
    else:
        print("No main privacy profile results found")

    print("\n=== Processing Sigma Data Variation Experiments ===")
    sigma_data_results = get_seml_results("privacy_profile_sigma_data_experiments")
    if sigma_data_results:
        processed_sigma_data = process_privacy_profile_results(sigma_data_results)
        print(f"Processing {len(processed_sigma_data)} sigma data variation configurations")
        for config_key, data in processed_sigma_data.items():
            plot_privacy_profile(config_key, data, output_dir)
    else:
        print("No sigma data variation privacy profile results found")

    print("\n=== Processing Large Image Experiments ===")
    large_results = get_seml_results("privacy_profile_large_image_experiments")
    if large_results:
        processed_large = process_privacy_profile_results(large_results)
        print(f"Processing {len(processed_large)} large image configurations")
        for config_key, data in processed_large.items():
            plot_privacy_profile(config_key, data, output_dir)
    else:
        print("No large image privacy profile results found")

    print("\nAll privacy profile plots completed!")


if __name__ == "__main__":
    main()