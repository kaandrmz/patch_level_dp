"""Plotting script for 3-line variance plots including Gaussian Noise Augmentation.

Produces one PDF per model/dataset combination, each showing:
  - Patch-Level Subsampling (baseline_privacy=False, gaussian_augmentation=False)
  - Minibatch Subsampling (baseline_privacy=True, gaussian_augmentation=False)
  - Gaussian Noise Augmentation (gaussian_augmentation=True)
"""

import matplotlib.pyplot as plt
import pandas as pd
from seml.evaluation import get_results


EXPERIMENT_CONFIGS = [
    {
        'collection': 'deeplabv3plus_cityscapes_experiments',
        'title': 'DeepLabV3+ on Cityscapes',
        'filename': 'deeplabv3plus_cityscapes_performance_variance_gaussian.pdf',
    },
    {
        'collection': 'deeplabv3plus_a2d2_experiments',
        'title': 'DeepLabV3+ on A2D2',
        'filename': 'deeplabv3plus_a2d2_performance_variance_gaussian.pdf',
    },
    {
        'collection': 'pspnet_cityscapes_experiments',
        'title': 'PSPNet on Cityscapes',
        'filename': 'pspnet_cityscapes_performance_variance_gaussian.pdf',
    },
    {
        'collection': 'pspnet_a2d2_experiments',
        'title': 'PSPNet on A2D2',
        'filename': 'pspnet_a2d2_performance_variance_gaussian.pdf',
    },
]


def get_results_for_collection(collection_name):
    """Fetch experiment results from a unified SEML collection."""
    try:
        df = get_results(
            collection_name,
            fields=[
                'result.epsilon',
                'result.test_miou',
                'config.baseline_privacy',
                'config.gaussian_augmentation',
                'config.seed',
            ],
            states=['COMPLETED'],
            to_data_frame=True,
        )

        df.dropna(subset=['result.epsilon', 'result.test_miou'], inplace=True)

        print(f"Found {len(df)} completed experiments in '{collection_name}'")
        return df

    except Exception as e:
        print(f"Could not fetch results for '{collection_name}': {e}")
        return pd.DataFrame()


def process_three_methods(df):
    """Split results into patch-level, minibatch, and Gaussian augmentation."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_patch = df[
        (df['config.baseline_privacy'] == False)
        & (df['config.gaussian_augmentation'] == False)
    ].copy()

    df_minibatch = df[
        (df['config.baseline_privacy'] == True)
        & (df['config.gaussian_augmentation'] == False)
    ].copy()

    df_gaussian = df[
        df['config.gaussian_augmentation'] == True
    ].copy()

    print(f"  Patch-Level Subsampling: {len(df_patch)}")
    print(f"  Minibatch Subsampling:   {len(df_minibatch)}")
    print(f"  Gaussian Noise Aug.:     {len(df_gaussian)}")

    return df_patch, df_minibatch, df_gaussian


def calculate_stats(df, method_name):
    """Calculate mean and standard deviation for each epsilon group."""
    if df.empty:
        return pd.DataFrame()

    df['epsilon_group'] = df['result.epsilon'].round(0)
    stats = df.groupby('epsilon_group').agg(
        **{
            'result.epsilon': ('result.epsilon', 'mean'),
            'mean': ('result.test_miou', 'mean'),
            'std': ('result.test_miou', 'std'),
            'count': ('result.test_miou', 'count'),
        }
    ).reset_index()

    stats = stats.sort_values(by='result.epsilon')

    print(f"\n  {method_name} Statistics:")
    for _, row in stats.iterrows():
        print(
            f"    \u03b5\u2248{row['epsilon_group']:.0f}: "
            f"\u03bc={row['mean']:.4f}, \u03c3={row['std']:.4f}, n={row['count']}"
        )

    return stats


def create_three_line_variance_plot(patch_stats, minibatch_stats, gaussian_stats,
                                    title, filename):
    """Create a variance plot with error bars for all three methods."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
        'font.size': 30,
        'axes.labelsize': 30,
        'legend.fontsize': 27,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'lines.linewidth': 2.5,
        'font.family': 'serif',
    })

    fig, ax = plt.subplots(figsize=(12, 10))

    if not patch_stats.empty:
        _, _, bars = ax.errorbar(
            patch_stats['result.epsilon'],
            patch_stats['mean'],
            yerr=patch_stats['std'],
            marker='o',
            linestyle='-',
            color='#377eb8',
            label='Patch-Level Subsampling',
            capsize=5,
            ecolor='#a6cee3',
            capthick=2,
            zorder=2,
        )
        if bars:
            bars[0].set_linestyle('--')

    if not minibatch_stats.empty:
        _, _, bars = ax.errorbar(
            minibatch_stats['result.epsilon'],
            minibatch_stats['mean'],
            yerr=minibatch_stats['std'],
            marker='o',
            linestyle='-',
            color='#ff7f0e',
            label='Minibatch Subsampling',
            capsize=5,
            ecolor='#fdbf6f',
            capthick=2,
            zorder=2,
        )
        if bars:
            bars[0].set_linestyle('--')

    if not gaussian_stats.empty:
        _, _, bars = ax.errorbar(
            gaussian_stats['result.epsilon'],
            gaussian_stats['mean'],
            yerr=gaussian_stats['std'],
            marker='o',
            linestyle='-',
            color='#4daf4a',
            label='Gaussian Noise Augmentation',
            capsize=5,
            ecolor='#b2df8a',
            capthick=2,
            zorder=2,
        )
        if bars:
            bars[0].set_linestyle('--')

    ax.set_xlabel(r'$\varepsilon$', labelpad=15)
    ax.set_ylabel('mIoU', labelpad=15)
    ax.set_title(title, pad=20)
    ax.grid(True, alpha=0.3)

    legend = ax.legend(loc='lower right')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_alpha(0.63)

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n  Plot saved as '{filename}'")
    plt.close(fig)

    return filename


def run_experiment(config):
    """Run the full pipeline for a single experiment configuration."""
    print(f"\n{'=' * 70}")
    print(f"  {config['title']}")
    print(f"{'=' * 70}")

    df = get_results_for_collection(config['collection'])

    if df.empty:
        print(f"  No results found. Check: seml {config['collection']} status")
        return

    df_patch, df_minibatch, df_gaussian = process_three_methods(df)

    patch_stats = calculate_stats(df_patch, "Patch-Level Subsampling")
    minibatch_stats = calculate_stats(df_minibatch, "Minibatch Subsampling")
    gaussian_stats = calculate_stats(df_gaussian, "Gaussian Noise Augmentation")

    if patch_stats.empty and minibatch_stats.empty and gaussian_stats.empty:
        print("  No data available for plotting")
        return

    create_three_line_variance_plot(
        patch_stats, minibatch_stats, gaussian_stats,
        config['title'], config['filename'],
    )


def main():
    """Generate 3-line variance plots for all model/dataset combinations."""
    print("=== Gaussian Noise Augmentation Performance Analysis ===")

    for config in EXPERIMENT_CONFIGS:
        run_experiment(config)

    print(f"\n{'=' * 70}")
    print("All plots generated.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
