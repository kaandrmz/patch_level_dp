"""Plotting script for PSPNet performance experiments using SEML results."""

import matplotlib.pyplot as plt
import pandas as pd
from seml.evaluation import get_results


def get_vgg_dtd_results():
    """Retrieve VGG-DTD experiment results from SEML database."""
    try:
        df = get_results(
            'vgg_dtd_main',
            fields=[
                'result.final_epsilon',
                'result.test_acc',
                'config.baseline_privacy',
                'config.seed',
                'config.gaussian_augmentation'
            ],
            states=['COMPLETED'],
            to_data_frame=True
        )

        df.dropna(subset=['result.final_epsilon', 'result.test_acc'], inplace=True)

        if 'config.gaussian_augmentation' not in df.columns:
            df['config.gaussian_augmentation'] = False
        df['config.gaussian_augmentation'] = df['config.gaussian_augmentation'].fillna(False)

        print(f"Found {len(df)} completed VGG-DTD experiments")
        return df

    except Exception as e:
        print(f"Could not fetch results for 'vgg_dtd_main' collection: {e}")
        return pd.DataFrame()


def process_results(df):
    """Process results and separate into patch-level, minibatch subsampling, and gaussian augmentation."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_gaussian = df[df['config.gaussian_augmentation'] == True].copy()
    df_non_gaussian = df[df['config.gaussian_augmentation'] != True].copy()

    df_patch_level = df_non_gaussian[df_non_gaussian['config.baseline_privacy'] == False].copy()
    df_minibatch = df_non_gaussian[df_non_gaussian['config.baseline_privacy'] == True].copy()

    print(f"Patch-Level Subsampling experiments: {len(df_patch_level)}")
    print(f"Minibatch Subsampling experiments: {len(df_minibatch)}")
    print(f"Gaussian Noise Augmentation experiments: {len(df_gaussian)}")

    return df_patch_level, df_minibatch, df_gaussian


def calculate_stats(df, method_name):
    """Calculate mean and standard deviation for each epsilon group."""
    if df.empty:
        return pd.DataFrame()

    df['epsilon_group'] = df['result.final_epsilon'].round(0)
    stats = df.groupby('epsilon_group').agg(
        **{
            'result.final_epsilon': ('result.final_epsilon', 'mean'),
            'mean': ('result.test_acc', 'mean'),
            'std': ('result.test_acc', 'std'),
            'count': ('result.test_acc', 'count')
        }
    ).reset_index()

    stats = stats.sort_values(by='result.final_epsilon')

    print(f"\n{method_name} Statistics:")
    for _, row in stats.iterrows():
        print(f"  ε≈{row['epsilon_group']:.0f}: μ={row['mean']:.4f}, σ={row['std']:.4f}, n={row['count']}")

    return stats


def print_per_seed_stats(df, method_name):
    """Print per-seed statistics for each epsilon group."""
    if df.empty:
        return
    
    seeds = [10, 20, 30, 516]
    df['epsilon_group'] = df['result.final_epsilon'].round(0)
    
    print(f"\n{method_name} - Per-Seed Statistics:")
    print("=" * 80)
    
    epsilon_groups = sorted(df['epsilon_group'].unique())
    
    for eps in epsilon_groups:
        df_eps = df[df['epsilon_group'] == eps]
        print(f"\nε≈{eps:.0f}:")
        for seed in seeds:
            df_seed = df_eps[df_eps['config.seed'] == seed]
            if not df_seed.empty:
                acc_values = df_seed['result.test_acc'].values
                print(f"  Seed {seed:3d}: {acc_values[0]:.4f}")
            else:
                print(f"  Seed {seed:3d}: N/A")


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
            patch_stats['result.final_epsilon'],
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
            minibatch_stats['result.final_epsilon'],
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
    ax.set_ylabel('Accuracy', labelpad=15)
    ax.set_title('VGG-11 on DTD', pad=20)
    ax.grid(True, alpha=0.3)

    legend = ax.legend(loc='lower right')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_alpha(0.63)

    plt.tight_layout()

    filename = 'vgg_dtd_performance_variance.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nVariance plot saved as '{filename}'")
    plt.close(fig)

    return filename


def create_per_seed_plots(df_patch, df_minibatch):
    """Create separate plots for each seed showing patch-level vs minibatch comparison."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
        'font.size': 30,
        'axes.labelsize': 30,
        'legend.fontsize': 27,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'lines.linewidth': 2.5,
        'font.family': 'serif'
    })

    seeds = [10, 20, 30, 516]
    filenames = []
    
    for seed in seeds:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot Patch-Level Subsampling for this seed
        if not df_patch.empty:
            df_seed_patch = df_patch[df_patch['config.seed'] == seed].copy()
            if not df_seed_patch.empty:
                df_seed_patch = df_seed_patch.sort_values(by='result.final_epsilon')
                ax.plot(
                    df_seed_patch['result.final_epsilon'],
                    df_seed_patch['result.test_acc'],
                    marker='o',
                    linestyle='-',
                    color='#377eb8',
                    label='Patch-Level Subsampling',
                    zorder=2
                )
        
        # Plot Minibatch Subsampling for this seed
        if not df_minibatch.empty:
            df_seed_minibatch = df_minibatch[df_minibatch['config.seed'] == seed].copy()
            if not df_seed_minibatch.empty:
                df_seed_minibatch = df_seed_minibatch.sort_values(by='result.final_epsilon')
                ax.plot(
                    df_seed_minibatch['result.final_epsilon'],
                    df_seed_minibatch['result.test_acc'],
                    marker='o',
                    linestyle='-',
                    color='#ff7f0e',
                    label='Minibatch Subsampling',
                    zorder=2
                )
        
        ax.set_xlabel(r'$\varepsilon$', labelpad=15)
        ax.set_ylabel('Accuracy', labelpad=15)
        ax.grid(True, alpha=0.3)
        
        legend = ax.legend(loc='lower right')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(0.63)
        
        plt.tight_layout()
        
        filename = f'vgg_dtd_performance_seed_{seed}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Per-seed plot saved as '{filename}'")
        plt.close(fig)
        
        filenames.append(filename)
    
    return filenames


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


def create_variance_plot_3lines(patch_stats, minibatch_stats, gaussian_stats):
    """Create the variance plot with error bars (3-line version with Gaussian Noise Augmentation)."""
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
            patch_stats['result.final_epsilon'],
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
            minibatch_stats['result.final_epsilon'],
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

    if not gaussian_stats.empty:
        _, _, bars = ax.errorbar(
            gaussian_stats['result.final_epsilon'],
            gaussian_stats['mean'],
            yerr=gaussian_stats['std'],
            marker='o',
            linestyle='-',
            color='#4daf4a',
            label='Gaussian Noise Augmentation',
            capsize=5,
            ecolor='#b2df8a',
            capthick=2,
            zorder=2
        )
        if bars:
            bars[0].set_linestyle('--')

    ax.set_xlabel(r'$\varepsilon$', labelpad=15)
    ax.set_ylabel('Accuracy', labelpad=15)
    ax.set_title('VGG-11 on DTD', pad=20)
    ax.grid(True, alpha=0.3)

    legend = ax.legend(loc='lower right')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_alpha(0.63)

    plt.tight_layout()

    filename = 'vgg_dtd_performance_variance_3lines.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n3-line variance plot saved as '{filename}'")
    plt.close(fig)

    return filename


def create_per_seed_plots_3lines(df_patch, df_minibatch, df_gaussian):
    """Create separate plots for each seed showing patch-level vs minibatch vs gaussian comparison."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
        'font.size': 30,
        'axes.labelsize': 30,
        'legend.fontsize': 27,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'lines.linewidth': 2.5,
        'font.family': 'serif'
    })

    seeds = [10, 20, 30, 516]
    filenames = []

    for seed in seeds:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot Patch-Level Subsampling for this seed
        if not df_patch.empty:
            df_seed_patch = df_patch[df_patch['config.seed'] == seed].copy()
            if not df_seed_patch.empty:
                df_seed_patch = df_seed_patch.sort_values(by='result.final_epsilon')
                ax.plot(
                    df_seed_patch['result.final_epsilon'],
                    df_seed_patch['result.test_acc'],
                    marker='o',
                    linestyle='-',
                    color='#377eb8',
                    label='Patch-Level Subsampling',
                    zorder=2
                )

        # Plot Minibatch Subsampling for this seed
        if not df_minibatch.empty:
            df_seed_minibatch = df_minibatch[df_minibatch['config.seed'] == seed].copy()
            if not df_seed_minibatch.empty:
                df_seed_minibatch = df_seed_minibatch.sort_values(by='result.final_epsilon')
                ax.plot(
                    df_seed_minibatch['result.final_epsilon'],
                    df_seed_minibatch['result.test_acc'],
                    marker='o',
                    linestyle='-',
                    color='#ff7f0e',
                    label='Minibatch Subsampling',
                    zorder=2
                )

        # Plot Gaussian Noise Augmentation for this seed
        if not df_gaussian.empty:
            df_seed_gaussian = df_gaussian[df_gaussian['config.seed'] == seed].copy()
            if not df_seed_gaussian.empty:
                df_seed_gaussian = df_seed_gaussian.sort_values(by='result.final_epsilon')
                ax.plot(
                    df_seed_gaussian['result.final_epsilon'],
                    df_seed_gaussian['result.test_acc'],
                    marker='o',
                    linestyle='-',
                    color='#4daf4a',
                    label='Gaussian Noise Augmentation',
                    zorder=2
                )

        ax.set_xlabel(r'$\varepsilon$', labelpad=15)
        ax.set_ylabel('Accuracy', labelpad=15)
        ax.grid(True, alpha=0.3)

        legend = ax.legend(loc='lower right')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(0.63)

        plt.tight_layout()

        filename = f'vgg_dtd_performance_seed_{seed}_3lines.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"3-line per-seed plot saved as '{filename}'")
        plt.close(fig)

        filenames.append(filename)

    return filenames


def main():
    """Main plotting function for VGG performance experiments."""
    print("=== VGG-DTD Performance Analysis ===")

    df = get_vgg_dtd_results()

    if df.empty:
        print("No results found. Make sure VGG-DTD experiments have been run:")
        print("  seml vgg_dtd_main status")
        return

    df_patch, df_minibatch, df_gaussian = process_results(df)

    patch_stats = calculate_stats(df_patch, "Patch-Level Subsampling")
    minibatch_stats = calculate_stats(df_minibatch, "Minibatch Subsampling")
    gaussian_stats = calculate_stats(df_gaussian, "Gaussian Noise Augmentation")

    # Print per-seed statistics
    print_per_seed_stats(df_patch, "Patch-Level Subsampling")
    print_per_seed_stats(df_minibatch, "Minibatch Subsampling")
    print_per_seed_stats(df_gaussian, "Gaussian Noise Augmentation")

    if not patch_stats.empty or not minibatch_stats.empty:
        # Create aggregate plot with variance
        plot_file = create_variance_plot(patch_stats, minibatch_stats)

        # Create per-seed plots (one per seed)
        print("\nGenerating per-seed plots...")
        per_seed_plots = create_per_seed_plots(df_patch, df_minibatch)

        calculate_improvements(patch_stats, minibatch_stats)

        print(f"\nAnalysis complete!")
        print(f"  - Aggregate plot: '{plot_file}'")
        print(f"  - Per-seed plots: {len(per_seed_plots)} files created")
    else:
        print("No data available for plotting")

    # 3-line plots with Gaussian Noise Augmentation
    print("\n" + "=" * 80)
    print("Creating 3-line plots with Gaussian Noise Augmentation")
    print("=" * 80)

    if not patch_stats.empty or not minibatch_stats.empty or not gaussian_stats.empty:
        plot_file_3lines = create_variance_plot_3lines(patch_stats, minibatch_stats, gaussian_stats)

        print("\nGenerating 3-line per-seed plots...")
        per_seed_plots_3lines = create_per_seed_plots_3lines(df_patch, df_minibatch, df_gaussian)

        print(f"\n3-line analysis complete!")
        print(f"  - Aggregate plot: '{plot_file_3lines}'")
        print(f"  - Per-seed plots: {len(per_seed_plots_3lines)} files created")
    else:
        print("No data available for 3-line plotting")

    print("\n" + "=" * 80)
    print("All plots created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
