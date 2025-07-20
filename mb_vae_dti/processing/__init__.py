"""
Processing module for MB-VAE-DTI.

This module contains functionality for 
- generating drug/target features
- creating the h5torch files
- splitting the data
- h5torch dataloaders

It makes use of external repositories like bmfm_sm and ESPF
and scripts: `embeddings.sh` and `h5torch_creation.py`.
"""

from mb_vae_dti.processing.embedding import (
    run_embedding_script,
    save_dti_to_h5,
    save_pretrain_to_h5
)

from mb_vae_dti.processing.split import (
    add_random_split,
    add_cold_drug_split,
    add_splits
)

from mb_vae_dti.processing.h5factory import (
    create_pretrain_h5torch,
    create_dti_h5torch,
    inspect_h5torch_file
)

from mb_vae_dti.processing.h5datasets import (
    DTIDataset,
    PretrainDataset
)