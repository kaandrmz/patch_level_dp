"""Plotting script for padding parameter dependency experiments using SEML results."""

import math
import os
from typing import Dict, List, Tuple

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


def process_padding_results(results: List[Dict]) -> Dict:
    """Process padding experiment results."""
    processed_results = {}

    for result in results:
        if 'result' not in result:
            continue

        result_data = result['result']

        if result_data.get('experiment_type') != 'padding':
            continue

        noise_levels_raw = result_data['noise_levels']
        if isinstance(noise_levels_raw, dict) and 'py/seq' in noise_levels_raw:
            noise_levels = tuple(noise_levels_raw['py/seq'])
        elif isinstance(noise_levels_raw, list):
            noise_levels = tuple(noise_levels_raw)
        elif isinstance(noise_levels_raw, tuple):
            noise_levels = noise_levels_raw
        else:
            print(f"Warning: Unexpected noise_levels format: {noise_levels_raw}")
            continue

        noise_labels_raw = result_data['noise_labels']
        if isinstance(noise_labels_raw, dict) and 'py/seq' in noise_labels_raw:
            noise_labels = [str(x) for x in noise_labels_raw['py/seq']]
        elif isinstance(noise_labels_raw, list):
            noise_labels = [str(x) for x in noise_labels_raw]
        else:
            print(f"Warning: Unexpected noise_labels format: {noise_labels_raw}")
            noise_labels = [str(nl) for nl in noise_levels]

        config_key = (
            result_data['experiment_set'],
            result_data['crop_size'],
            result_data['private_patch_size'],
            noise_levels
        )

        if config_key not in processed_results:
            processed_results[config_key] = {
                'padding_data': {},
                'metadata': {
                    'noise_labels': noise_labels,
                    'target_delta': result_data['target_delta']
                }
            }

        padding_value = result_data['padding_value']
        padding_results = result_data['padding_results']

        processed_results[config_key]['padding_data'][padding_value] = padding_results

    return processed_results


def calculate_baseline_epsilon(noise_level: float, common_params: Dict) -> float:
    """Calculate baseline epsilon (intersection_prob=1) for a given noise level."""
    from patch_level_dp.privacy.calculations import create_pld

    sampling_prob = common_params['batch_sampling_prob']
    sensitivity = common_params['sensitivity']
    steps = common_params['steps']
    target_delta = common_params['target_delta']

    single_step_pld = create_pld(noise_level, sensitivity, sampling_prob)
    total_pld = single_step_pld.self_compose(steps)
    epsilon = total_pld.get_epsilon_for_delta(target_delta)

    return epsilon


def plot_padding_results(config_key: Tuple, data: Dict, output_dir: str = "."):
    """Create plot for padding vs epsilon results."""
    experiment_set, crop_size, private_patch_size, noise_levels = config_key

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

    plt.figure(figsize=(12, 10))

    plot_colors = ['#377eb8', '#ff7f0e', '#4daf4a']
    baseline_colors = ['#a6cee3', '#fdbf6f', '#b2df8a']
    noise_labels = data['metadata']['noise_labels']

    padding_data = data['padding_data']
    sorted_padding_values = sorted(padding_data.keys())

    for noise_idx, noise_level in enumerate(noise_levels):
        padding_values_for_line = []
        epsilons_for_line = []

        for padding_val in sorted_padding_values:
            if padding_val in padding_data:
                intersection_results = padding_data[padding_val].get('intersection', {})
                json_key = f'json://{noise_level}'
                epsilon_val = None
                if json_key in intersection_results:
                    epsilon_val = intersection_results[json_key]
                elif noise_level in intersection_results:
                    epsilon_val = intersection_results[noise_level]
                elif str(noise_level) in intersection_results:
                    epsilon_val = intersection_results[str(noise_level)]
                
                if epsilon_val is not None and epsilon_val != float('inf') and not math.isnan(epsilon_val):
                    padding_values_for_line.append(padding_val)
                    epsilons_for_line.append(epsilon_val)

        if padding_values_for_line:
            plt.plot(padding_values_for_line, epsilons_for_line,
                     color=plot_colors[noise_idx % len(plot_colors)],
                     linestyle='-',
                     linewidth=2.5,
                     zorder=2)

        baseline_results = None
        for padding_val in sorted_padding_values:
            if padding_val in padding_data:
                baseline_results = padding_data[padding_val].get('baseline', {})
                break

        if baseline_results:
            baseline_eps = None
            json_key = f'json://{noise_level}'
            if json_key in baseline_results:
                baseline_eps = baseline_results[json_key]
            elif noise_level in baseline_results:
                baseline_eps = baseline_results[noise_level]
            elif str(noise_level) in baseline_results:
                baseline_eps = baseline_results[str(noise_level)]
            
            if baseline_eps is not None and baseline_eps != float('inf') and not math.isnan(baseline_eps):
                plt.axhline(y=baseline_eps, color=baseline_colors[noise_idx],
                            linestyle='--', zorder=1)

    plt.xlabel('Padding Amount (pixels)', labelpad=15)
    plt.ylabel(r'Epsilon ($\varepsilon$)', labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.3)

    legend1_lines = [Line2D([0], [0], color='black', lw=2, linestyle='-'),
                     Line2D([0], [0], color='black', lw=2, linestyle='--')]
    legend1_labels = ['Patch-Level Subsampling', 'Minibatch Subsampling']
    plt.legend(legend1_lines, legend1_labels, loc='upper left')

    legend2_lines = [Line2D([0], [0], color=plot_colors[i], lw=2) for i in range(len(noise_levels))]
    legend2_labels = noise_labels
    leg2 = Legend(plt.gca(), legend2_lines, legend2_labels, loc='lower right', title=r'$\sigma$')
    plt.gca().add_artist(leg2)
    leg2.get_frame().set_edgecolor('black')
    leg2.get_frame().set_alpha(0.63)
    plt.gca().get_legend().get_frame().set_edgecolor('black')
    plt.gca().get_legend().get_frame().set_alpha(0.63)

    plt.xlim(0, 50)
    plt.tight_layout()

    first_noise = noise_levels[0]
    noise_str = f"{first_noise:.1f}" if isinstance(first_noise, float) else str(first_noise)
    filename = f'seml_epsilon_vs_padding_crop{crop_size}_priv{private_patch_size}_noise{noise_str}.pdf'

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Padding plot saved as '{filepath}'")
    return filename


def main():
    """Main plotting function."""
    output_dir = "output/plots/parameter_influence/padding"

    print("=== Processing Padding Experiments ===")
    padding_results = get_seml_results("parameter_influence_padding_all_experiments")
    if padding_results:
        processed_padding = process_padding_results(padding_results)
        print(f"Processing {len(processed_padding)} padding configurations")

        for config_key, data in processed_padding.items():
            plot_padding_results(config_key, data, output_dir)
    else:
        print("No padding results found")

    print("\nAll padding plots completed!")


if __name__ == "__main__":
    main()