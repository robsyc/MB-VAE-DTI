"""
Visualization functionality for MB-VAE-DTI datasets.

This module provides functions for visualizing dataset distributions,
comparing datasets, and generating plots for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
from pathlib import Path
from upsetplot import UpSet, from_indicators
from mb_vae_dti.loading.datasets import compute_heavy_atom_counts

# Define paths
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def set_plotting_style():
    """Set the default plotting style for consistent visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_interaction_distribution(
    dfs: List[pd.DataFrame],
    names: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> str:
    """
    Plot the distribution of interaction scores for multiple datasets.
    
    Args:
        dfs: List of dataframes containing interaction data
        names: List of dataset names corresponding to the dataframes
        save_path: Path to save the plot (if None, a default path is used)
        show: Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    set_plotting_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define thresholds and y-limits for each type
    thresholds = {"Y_pKd": 7.0, "Y_pKi": 7.6, "Y_KIBA": 12.1}
    
    # Plot distributions for each interaction type
    for ax, score_type in zip([ax1, ax2, ax3], ["Y_pKd", "Y_pKi", "Y_KIBA"]):
        has_data = False
        for df, name in zip(dfs, names):
            if score_type in df.columns:
                has_data = True
                sns.histplot(
                    data=df,
                    x=score_type,
                    bins=25,
                    label=name,
                    ax=ax,
                    alpha=0.5
                )
                ax.set_xlabel(score_type.replace("Y_", ""))
        
        # Add threshold line and set limits if we have data for this type
        if has_data:
            ax.axvline(x=thresholds[score_type], color='red', linestyle='--', 
                      label=f'Threshold ({thresholds[score_type]})')
            
        ax.set_title(f'Distribution of {score_type.replace("Y_", "")} Values')
        if has_data:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)

    # Create a stacked bar chart with percentages
    bound_percentages = []
    unbound_percentages = []
    total_counts = []
    
    for df in dfs:
        if 'Y' in df.columns:
            total = len(df)
            bound = df['Y'].sum()
            unbound = total - bound
            bound_percentages.append(bound / total * 100)
            unbound_percentages.append(unbound / total * 100)
            total_counts.append(total)
        else:
            bound_percentages.append(0)
            unbound_percentages.append(0)
            total_counts.append(0)
    
    # Create a DataFrame for easier plotting
    plot_df = pd.DataFrame({
        'Dataset': names,
        'Bound (%)': bound_percentages,
        'Unbound (%)': unbound_percentages,
        'Total': total_counts
    })
    
    # Sort by bound percentage
    plot_df = plot_df.sort_values('Bound (%)', ascending=False)
    
    # Create the stacked bar chart
    colors = ['#2ecc71', '#e74c3c']  # Green for bound, red for unbound
    ax4.bar(plot_df['Dataset'], plot_df['Bound (%)'], color=colors[0], label='Bound')
    ax4.bar(plot_df['Dataset'], plot_df['Unbound (%)'], bottom=plot_df['Bound (%)'], 
            color=colors[1], label='Unbound')
    
    # Add percentage labels
    for i, (bound_pct, dataset) in enumerate(zip(plot_df['Bound (%)'], plot_df['Dataset'])):
        unbound_pct = 100 - bound_pct
        # Add bound percentage
        ax4.text(i, bound_pct/2, f"{bound_pct:.1f}%", 
                ha='center', va='center', color='white', fontweight='bold')
        # Add unbound percentage
        ax4.text(i, bound_pct + unbound_pct/2, f"{unbound_pct:.1f}%", 
                ha='center', va='center', color='white', fontweight='bold')
    
    # Add total counts as text above the bars
    for i, total in enumerate(plot_df['Total']):
        ax4.text(i, 103, f"n={total:,}", ha='center', va='bottom', rotation=0,
                fontsize=10, fontweight='bold')
    
    ax4.set_ylim(0, 110)  # Leave room for the count labels
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Bound vs. Unbound Interactions')
    
    plt.tight_layout()
    
    # Save the plot if requested
    if save_path is None:
        dataset_names = "_".join(names)
        save_path = IMAGES_DIR / f"interaction_distribution_{dataset_names}.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(save_path)


def plot_dataset_statistics(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> str:
    """
    Plot heavy atom count and protein sequence length distributions.
    
    Args:
        df: DataFrame to analyze
        title: Title for the plot
        save_path: Path to save the plot (if None, a default path is used)
        show: Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    set_plotting_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Plot heavy atom count distribution instead of SMILES length
    unique_smiles = df['Drug_SMILES'].unique()
    heavy_atoms_dict = compute_heavy_atom_counts(unique_smiles)
    
    # Create a Series of heavy atom counts
    heavy_atom_counts = pd.Series([heavy_atoms_dict[smiles] for smiles in df['Drug_SMILES']])
    
    sns.histplot(heavy_atom_counts, bins=30, ax=ax1, color='blue', alpha=0.7)
    ax1.set_title('Drug Heavy Atom Count Distribution')
    ax1.set_xlabel('Number of Heavy Atoms')
    ax1.set_ylabel('Count')
    
    # 2. Plot protein sequence length distribution
    sns.histplot(df['Target_AA'].str.len(), bins=30, ax=ax2, color='green', alpha=0.7)
    ax2.set_title('Protein Sequence Length Distribution')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    
    # Save the plot if requested
    if save_path is None:
        # Create a filename based on the title
        filename = "dataset_statistics"
        save_path = IMAGES_DIR / f"{filename}.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(save_path)


def plot_dataset_overlap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> str:
    """
    Plot the overlap between different datasets using an UpSet plot.
    
    Args:
        df: DataFrame with dataset indicators (columns starting with 'in_')
        save_path: Path to save the plot (if None, a default path is used)
        show: Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    set_plotting_style()
    
    # Get dataset indicator columns
    dataset_indicators = [col for col in df.columns if col.startswith('in_')]
    if not dataset_indicators:
        raise ValueError("DataFrame does not contain dataset indicators (columns starting with 'in_')")
    
    # Rename columns to remove 'in_' prefix for display
    renamed_df = df[dataset_indicators].copy()
    renamed_df.columns = [col.replace('in_', '') for col in dataset_indicators]
    
    # Convert boolean membership data into a format for UpSetPlot
    upset_data = from_indicators(renamed_df.columns, renamed_df)
    
    # Custom formatter for counts
    def format_counts(count):
        if count >= 1000:
            return f"{count // 1000}k"
        return str(count)
    
    # Create the upset plot
    fig = plt.figure(figsize=(12, 8))
    upset = UpSet(upset_data, subset_size='count', show_counts=True, sort_by='cardinality')
    
    # Plot first so that count texts are added
    upset.plot(fig=fig)
    
    # Now iterate over all axes to update the count texts
    for ax in fig.axes:
        for text in ax.texts:
            try:
                count = int(text.get_text())
                text.set_text(format_counts(count))
            except (ValueError, TypeError):
                # Skip texts that aren't counts
                pass
        
        # Apply k-formatting to y-axis labels for the intersection size plot
        if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'Intersection size':
            yticks = ax.get_yticks()
            yticklabels = [format_counts(int(y)) for y in yticks if y >= 0]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
    
    # Add title
    plt.suptitle('Dataset Overlap', fontsize=16, y=0.98)
    
    # Save the plot if requested
    if save_path is None:
        save_path = IMAGES_DIR / "dataset_overlap.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(save_path)


def plot_promiscuity_analysis(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> str:
    """
    Analyze and visualize drug and target promiscuity in the dataset.
    
    This function creates a 2x2 plot with:
    - Top row: Boxplots showing the distribution of interaction counts for drugs and targets
    - Bottom row: Lorenz curves showing the cumulative distribution of interactions
      (with Gini coefficients) for drugs and targets
    
    Args:
        df: DataFrame containing drug-target interaction data
        save_path: Path to save the plot (if None, a default path is used)
        show: Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    set_plotting_style()
    
    # Count interactions per drug and per target
    drug_counts = df.groupby('Drug_SMILES').size().reset_index(name='interaction_count')
    target_counts = df.groupby('Target_AA').size().reset_index(name='interaction_count')
    
    # Calculate total number of possible interactions
    total_drugs = df['Drug_SMILES'].nunique()
    total_targets = df['Target_AA'].nunique()
    total_possible = total_drugs * total_targets
    total_actual = len(df)
    matrix_coverage = (total_actual / total_possible) * 100
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2)
    
    # ---- Top row: Boxplots of interaction counts ----
    
    # Drug interaction counts (left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Boxplot on primary axis
    sns.boxplot(x=drug_counts['interaction_count'], ax=ax1, color='blue')
    ax1.set_xscale('log')
    ax1.set_title(f'Drug Interaction Counts (n={total_drugs:,})', fontsize=14)
    ax1.set_xlabel('Number of Interactions (log scale)')
    ax1.set_ylabel('Boxplot')
    
    # Calculate and print statistics
    drug_zero_interactions = total_drugs - len(drug_counts)
    drug_one_interaction = len(drug_counts[drug_counts['interaction_count'] == 1])
    drug_max_interactions = drug_counts['interaction_count'].max()
    drug_median_interactions = drug_counts['interaction_count'].median()
    drug_mean_interactions = drug_counts['interaction_count'].mean()
    
    # Calculate percentiles
    drug_percentiles = np.percentile(drug_counts['interaction_count'], [25, 50, 75, 90, 95, 99])
    
    # Print drug statistics to stdout instead of adding to plot
    print(f"Drug Interaction Statistics:")
    print(f"Drugs with 0 interactions: {drug_zero_interactions:,} ({drug_zero_interactions/total_drugs:.1%})")
    print(f"Drugs with 1 interaction: {drug_one_interaction:,} ({drug_one_interaction/total_drugs:.1%})")
    print(f"Mean interactions per drug: {drug_mean_interactions:.1f}")
    print(f"Median interactions per drug: {drug_median_interactions:.1f}")
    print(f"Max interactions per drug: {drug_max_interactions:,}")
    print(f"Percentiles:")
    print(f"25th: {drug_percentiles[0]:.0f}, 50th: {drug_percentiles[1]:.0f}, 75th: {drug_percentiles[2]:.0f}")
    print(f"90th: {drug_percentiles[3]:.0f}, 95th: {drug_percentiles[4]:.0f}, 99th: {drug_percentiles[5]:.0f}")
    print()
    
    # Target interaction counts (right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Boxplot on primary axis
    sns.boxplot(x=target_counts['interaction_count'], ax=ax2, color='green')
    ax2.set_xscale('log')
    ax2.set_title(f'Target Interaction Counts (n={total_targets:,})', fontsize=14)
    ax2.set_xlabel('Number of Interactions (log scale)')
    ax2.set_ylabel('Boxplot')
        
    # Calculate and print statistics
    target_zero_interactions = total_targets - len(target_counts)
    target_one_interaction = len(target_counts[target_counts['interaction_count'] == 1])
    target_max_interactions = target_counts['interaction_count'].max()
    target_median_interactions = target_counts['interaction_count'].median()
    target_mean_interactions = target_counts['interaction_count'].mean()
    
    # Calculate percentiles
    target_percentiles = np.percentile(target_counts['interaction_count'], [25, 50, 75, 90, 95, 99])
    
    # Print target statistics to stdout instead of adding to plot
    print(f"Target Interaction Statistics:")
    print(f"Targets with 0 interactions: {target_zero_interactions:,} ({target_zero_interactions/total_targets:.1%})")
    print(f"Targets with 1 interaction: {target_one_interaction:,} ({target_one_interaction/total_targets:.1%})")
    print(f"Mean interactions per target: {target_mean_interactions:.1f}")
    print(f"Median interactions per target: {target_median_interactions:.1f}")
    print(f"Max interactions per target: {target_max_interactions:,}")
    print(f"Percentiles:")
    print(f"25th: {target_percentiles[0]:.0f}, 50th: {target_percentiles[1]:.0f}, 75th: {target_percentiles[2]:.0f}")
    print(f"90th: {target_percentiles[3]:.0f}, 95th: {target_percentiles[4]:.0f}, 99th: {target_percentiles[5]:.0f}")
    print()


    # ---- Bottom row: Lorenz curves ----
    
    # Function to calculate Lorenz curve and Gini coefficient
    def lorenz_curve(counts):
        # Sort counts in ascending order
        sorted_counts = np.sort(counts)
        # Calculate cumulative sum
        cum_counts = np.cumsum(sorted_counts)
        # Normalize to get cumulative percentage
        cum_pct = cum_counts / cum_counts[-1]
        # X-axis: cumulative percentage of entities (drugs/targets)
        x = np.linspace(0, 1, len(cum_pct))
        # Calculate Gini coefficient
        gini = 1 - 2 * np.trapz(cum_pct, x)
        return x, cum_pct, gini
    
    # Drug Lorenz curve
    ax3 = fig.add_subplot(gs[1, 0])
    x_drug, y_drug, gini_drug = lorenz_curve(drug_counts['interaction_count'].values)
    ax3.plot(x_drug, y_drug, 'b-', linewidth=2, label=f'Drugs (Gini={gini_drug:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect equality')
    
    # Calculate percentage of drugs accounting for different percentages of interactions
    drug_cum_pct = np.cumsum(np.sort(drug_counts['interaction_count'].values)) / np.sum(drug_counts['interaction_count'])
    
    # Find the percentage of drugs accounting for X% of interactions
    def find_entity_pct_for_interaction_pct(cum_pct, target_pct):
        # Find the index where cumulative percentage exceeds target
        idx = np.searchsorted(cum_pct, target_pct)
        if idx >= len(cum_pct):
            return 1.0
        # Return the percentage of entities
        return idx / len(cum_pct)
    
    pct_drugs_50 = find_entity_pct_for_interaction_pct(drug_cum_pct, 0.5)
    pct_drugs_80 = find_entity_pct_for_interaction_pct(drug_cum_pct, 0.8)
    
    # Add markers for these points
    ax3.plot([pct_drugs_50, pct_drugs_50], [0, 0.5], 'r--', alpha=0.7)
    ax3.plot([0, pct_drugs_50], [0.5, 0.5], 'r--', alpha=0.7)
    ax3.scatter([pct_drugs_50], [0.5], color='red', s=50, zorder=5)
    
    ax3.plot([pct_drugs_80, pct_drugs_80], [0, 0.8], 'r--', alpha=0.7)
    ax3.plot([0, pct_drugs_80], [0.8, 0.8], 'r--', alpha=0.7)
    ax3.scatter([pct_drugs_80], [0.8], color='red', s=50, zorder=5)
    
    ax3.set_title('Drug Promiscuity Lorenz Curve', fontsize=14)
    ax3.set_xlabel('Cumulative % of Drugs')
    ax3.set_ylabel('Cumulative % of Interactions')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # Add text with statistics
    stats_text = (
        f"Gini coefficient: {gini_drug:.3f}\n"
        f"{(1-pct_drugs_50):.1%} of drugs account for 50% of interactions\n"
        f"{(1-pct_drugs_80):.1%} of drugs account for 20% of interactions"
    )
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Target Lorenz curve
    ax4 = fig.add_subplot(gs[1, 1])
    x_target, y_target, gini_target = lorenz_curve(target_counts['interaction_count'].values)
    ax4.plot(x_target, y_target, 'g-', linewidth=2, label=f'Targets (Gini={gini_target:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', label='Perfect equality')
    
    # Calculate percentage of targets accounting for different percentages of interactions
    target_cum_pct = np.cumsum(np.sort(target_counts['interaction_count'].values)) / np.sum(target_counts['interaction_count'])
    
    pct_targets_50 = find_entity_pct_for_interaction_pct(target_cum_pct, 0.5)
    pct_targets_80 = find_entity_pct_for_interaction_pct(target_cum_pct, 0.8)
    
    # Add markers for these points
    ax4.plot([pct_targets_50, pct_targets_50], [0, 0.5], 'r--', alpha=0.7)
    ax4.plot([0, pct_targets_50], [0.5, 0.5], 'r--', alpha=0.7)
    ax4.scatter([pct_targets_50], [0.5], color='red', s=50, zorder=5)
    
    ax4.plot([pct_targets_80, pct_targets_80], [0, 0.8], 'r--', alpha=0.7)
    ax4.plot([0, pct_targets_80], [0.8, 0.8], 'r--', alpha=0.7)
    ax4.scatter([pct_targets_80], [0.8], color='red', s=50, zorder=5)
    
    ax4.set_title('Target Promiscuity Lorenz Curve', fontsize=14)
    ax4.set_xlabel('Cumulative % of Targets')
    ax4.set_ylabel('Cumulative % of Interactions')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    # Add text with statistics
    stats_text = (
        f"Gini coefficient: {gini_target:.3f}\n"
        f"{(1-pct_targets_50):.1%} of targets account for 50% of interactions\n"
        f"{(1-pct_targets_80):.1%} of targets account for 20% of interactions"
    )
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add overall title with matrix coverage information
    plt.suptitle(
        f'Drug-Target Interaction Promiscuity Analysis\n'
        f'Matrix Coverage: {matrix_coverage:.2f}% ({total_actual:,} of {total_possible:,} possible interactions)',
        fontsize=16, y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    
    # Save the plot if requested
    if save_path is None:
        save_path = IMAGES_DIR / "promiscuity_analysis.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(save_path)