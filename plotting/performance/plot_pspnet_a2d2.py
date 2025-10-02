"""Plotting script for PSPNet on A2D2 performance experiments using SEML results."""

import matplotlib.pyplot as plt
import pandas as pd
from seml.evaluation import get_results


def get_pspnet_a2d2_results():
    """Retrieve PSPNet A2D2 experiment results from SEML database."""
    try:
        df = get_results(
            'pspnet_a2d2_experiments',
            fields=[
                'result.epsilon',
                'result.test_miou',
                'config.baseline_privacy',
                'config.seed_value'
            ],
            states=['COMPLETED'],
            to_data_frame=True
        )

        df.dropna(subset=['result.epsilon', 'result.test_miou'], inplace=True)

        print(f"Found {len(df)} completed PSPNet A2D2 experiments")
        return df

    except Exception as e:
        print(f"Could not fetch results for 'pspnet_a2d2_experiments' collection: {e}")
        return pd.DataFrame()


def process_results(df):
    """Process results and separate into patch-level and minibatch subsampling."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_patch_level = df[df['config.baseline_privacy'] == False].copy()
    df_minibatch = df[df['config.baseline_privacy'] == True].copy()

    print(f"Patch-Level Subsampling experiments: {len(df_patch_level)}")
    print(f"Minibatch Subsampling experiments: {len(df_minibatch)}")

    return df_patch_level, df_minibatch


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
            'count': ('result.test_miou', 'count')
        }
    ).reset_index()

    stats = stats.sort_values(by='result.epsilon')

    print(f"\n{method_name} Statistics:")
    for _, row in stats.iterrows():
        print(f"  ε≈{row['epsilon_group']:.0f}: μ={row['mean']:.4f}, σ={row['std']:.4f}, n={row['count']}")

    return stats


def create_variance_plot(patch_stats, minibatch_stats):
    """Create the variance plot with error bars."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
        'font.size': 27,
        'axes.labelsize': 30,
        'legend.fontsize': 27,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'lines.linewidth': 2.5,
        'font.family': 'serif'
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
            zorder=2
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
            zorder=2
        )
        if bars:
            bars[0].set_linestyle('--')

    ax.set_xlabel(r'$\varepsilon$', labelpad=15)
    ax.set_ylabel('mIoU', labelpad=15)
    ax.set_title('PSPNet on A2D2', pad=20)
    ax.grid(True, alpha=0.3)

    legend = ax.legend(loc='lower right')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_alpha(0.63)

    plt.tight_layout()

    filename = 'pspnet_a2d2_performance_variance.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nVariance plot saved as '{filename}'")
    plt.close(fig)

    return filename


def calculate_improvements(patch_stats, minibatch_stats):
    """Calculate performance improvements between methods."""
    if patch_stats.empty or minibatch_stats.empty:
        print("Cannot calculate improvements - missing data for one or both methods")
        return

    merged_stats = pd.merge(
        patch_stats,
        minibatch_stats,
        on='epsilon_group',
        suffixes=('_patch', '_minibatch')
    )

    if merged_stats.empty:
        print("No overlapping epsilon values found between methods")
        return

    merged_stats['improvement'] = (
        (merged_stats['mean_patch'] - merged_stats['mean_minibatch']) / merged_stats['mean_minibatch']
    ) * 100

    print("\nPerformance Comparison at each Epsilon Level:")
    print("=" * 80)
    for _, row in merged_stats.iterrows():
        print(
            f"ε≈{row['epsilon_group']:.0f}: "
            f"Patch-Level = {row['mean_patch']:.4f} ± {row['std_patch']:.4f}, "
            f"Minibatch = {row['mean_minibatch']:.4f} ± {row['std_minibatch']:.4f}, "
            f"Improvement = {row['improvement']:.2f}%"
        )

    avg_improvement = merged_stats['improvement'].mean()
    print(f"\nAverage Performance Improvement: {avg_improvement:.2f}%")

    return merged_stats


def main():
    """Main plotting function for PSPNet A2D2 performance experiments."""
    print("=== PSPNet A2D2 Performance Analysis ===")

    df = get_pspnet_a2d2_results()

    if df.empty:
        print("No results found. Make sure PSPNet A2D2 experiments have been run:")
        print("  seml pspnet_a2d2_experiments status")
        return

    df_patch, df_minibatch = process_results(df)

    patch_stats = calculate_stats(df_patch, "Patch-Level Subsampling")
    minibatch_stats = calculate_stats(df_minibatch, "Minibatch Subsampling")

    if not patch_stats.empty or not minibatch_stats.empty:
        plot_file = create_variance_plot(patch_stats, minibatch_stats)

        calculate_improvements(patch_stats, minibatch_stats)

        print(f"\nAnalysis complete! Plot saved as '{plot_file}'")
    else:
        print("No data available for plotting")


if __name__ == "__main__":
    main()
