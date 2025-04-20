"""
Loading module for MB-VAE-DTI.

This module contains functionality for loading and preprocessing DTI datasets,
visualizing dataset metrics, and handling FASTA files and identifiers.
"""

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
    plot_lorenz_curves,
    plot_interaction_stats
)

from mb_vae_dti.loading.drug_annotation import annotate_drug
from mb_vae_dti.loading.target_annotation import annotate_target

from mb_vae_dti.loading.annotation import (
    generate_unique_ids,
    add_potential_ids,
    annotate_drugs,
    annotate_targets,
    annotate_dti
)

from mb_vae_dti.loading.load_pretrain_drugs import (
    load_drug_generation_datasets,
    save_filtered_dataset
)