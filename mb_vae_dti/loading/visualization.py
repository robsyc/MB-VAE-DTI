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


def plot_lorenz_curves(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> str:
    """
    Visualize drug and target promiscuity in the dataset using Lorenz curves.
    
    This function creates a plot with Lorenz curves showing the cumulative distribution 
    of observations and positive interactions for drugs and targets.
    
    Args:
        df: DataFrame containing drug-target interaction data
        save_path: Path to save the plot (if None, a default path is used)
        show: Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    set_plotting_style()
    
    # Count observations per instance
    drug_observations = df.groupby('Drug_SMILES').size()
    target_observations = df.groupby('Target_AA').size()

    # For each instance, count the number of positive interactions
    drug_interactions = df.groupby('Drug_SMILES')['Y'].sum()
    target_interactions = df.groupby('Target_AA')['Y'].sum()
    
    # Calculate total number of possible interactions
    total_drugs = df['Drug_SMILES'].nunique()
    total_targets = df['Target_AA'].nunique()
    total_possible = total_drugs * total_targets
    total_actual = len(df)
    matrix_coverage = (total_actual / total_possible) * 100
    
    # Create figure with 1x2 subplots
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2)
    
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
    
    # Drug Lorenz curves
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot observations curve
    x_drug_obs, y_drug_obs, gini_drug_obs = lorenz_curve(drug_observations.values)
    ax1.plot(x_drug_obs, y_drug_obs, 'b-', linewidth=2, 
             label=f'Observations (Gini={gini_drug_obs:.3f})')
    
    # Plot interactions curve
    x_drug_int, y_drug_int, gini_drug_int = lorenz_curve(drug_interactions.values)
    ax1.plot(x_drug_int, y_drug_int, 'r-', linewidth=2, 
             label=f'Positive Interactions (Gini={gini_drug_int:.3f})')
    
    # Add reference line for perfect equality
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect equality')
    
    ax1.set_title('Drug Distribution Lorenz Curves', fontsize=14)
    ax1.set_xlabel('Cumulative % of Drugs')
    ax1.set_ylabel('Cumulative % of Observations/Interactions')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Target Lorenz curves
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot observations curve
    x_target_obs, y_target_obs, gini_target_obs = lorenz_curve(target_observations.values)
    ax2.plot(x_target_obs, y_target_obs, 'g-', linewidth=2, 
             label=f'Observations (Gini={gini_target_obs:.3f})')
    
    # Plot interactions curve
    x_target_int, y_target_int, gini_target_int = lorenz_curve(target_interactions.values)
    ax2.plot(x_target_int, y_target_int, 'r-', linewidth=2, 
             label=f'Positive Interactions (Gini={gini_target_int:.3f})')
    
    # Add reference line for perfect equality
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect equality')
    
    ax2.set_title('Target Distribution Lorenz Curves', fontsize=14)
    ax2.set_xlabel('Cumulative % of Targets')
    ax2.set_ylabel('Cumulative % of Observations/Interactions')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add overall title with matrix coverage information
    plt.suptitle(
        f'Drug-Target Distribution Lorenz Curves\n'
        f'Matrix Coverage: {matrix_coverage:.2f}% ({total_actual:,} of {total_possible:,} possible interactions)',
        fontsize=16, y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    
    # Save the plot if requested
    if save_path is None:
        save_path = IMAGES_DIR / "lorenz_curves.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(save_path)


