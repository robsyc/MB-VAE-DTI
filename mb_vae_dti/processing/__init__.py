"""
Processing module for MB-VAE-DTI.

This module contains functionality for splitting the data,
creating the h5torch file, and generating drug/target embeddings.
It makes use of external repositories like bmfm_sm and ESPF.
"""

from mb_vae_dti.processing.split import (
    add_split_cols,
    add_split_cols_drug_generation
)
from mb_vae_dti.processing.h5factory import (
    create_h5torch,
    load_h5torch_DTI,
    create_h5torch_smiles,
    SMILESDataset
)