"""
Loading module for MB-VAE-DTI.

This module contains functionality for loading and preprocessing DTI datasets,
visualizing dataset metrics, and handling FASTA files and identifiers.
"""

# from mb_vae_dti.loading import annotation

# Export key functions for easier access
from mb_vae_dti.loading.datasets import (
    load_dataset,
    merge_datasets,
    filter_dataset,
    load_or_create_merged_dataset,
    get_dataset_stats
)

from mb_vae_dti.loading.visualization import (
    plot_interaction_distribution,
    plot_dataset_statistics,
    plot_dataset_overlap,
    plot_promiscuity_analysis
)