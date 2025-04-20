# Multi-branch Variational Encoders for Drug-target Interaction Prediction (MB-VAE-DTI)

A machine learning framework for predicting drug-target interactions using multi-branch variational autoencoders leveraging pre-computed embeddings and target-conditioned discrete-diffusion drug-decoding.

## Project Overview

This project implements a novel approach to the dyadic Drug-Target Interaction (DTI) prediction problem using multi-branch variational autoencoders and a diffusion-based drug decoder based on [DiGress](https://github.com/cvignac/DiGress) and [DiffMS](https://github.com/coleygroup/DiffMS). The framework supports multiple embedding strategies for both drugs (molecules) and targets (proteins), allowing for flexible and powerful representation learning.

Key features:
- Multiple drug representations: Morgan fingerprints, graph, image and text (smiles) from [biomed-multi-view](https://github.com/BiomedSciAI/biomed-multi-view)
- Multiple protein representations: ESPF fingerprints, amino acid ([ESM](https://github.com/facebookresearch/esm)) and DNA sequences ([nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer))
- Variational (with discrete-diffusion decoding) and non-variational encoder architectures
- Inspection of latent spaces & exploration of generative capabilities
- Comprehensive evaluation on standard DTI datasets through [tdc](https://tdcommons.ai/) (DAVIS, KIBA, BindingDB and Metz) and a new aggregated dataset of ±400k interactions & pre-computed embeddings, as well as a pre-training strategy for the full drug branch.

## Repository Structure (simplified)

```
MB-VAE-DTI/
├── mb_vae_dti/                 # Main Python package
│   ├── loading/                # Loading and preprocessing datasets
│   │   ├── datasets.py         # Loading, merging, filtering and saving initial DTI datasets
│   │   ├── annotation.py       # Addition of DNA sequences, InChI keys, etc.
│   │   └── visualization.py    # Plotting metrics for loaded data
|   | 
│   ├── processing/             # Embedding & h5torch file creation
│   │   ├── embedding.py        # Embedding generation
│   │   ├── h5factory.py        # h5torch file creation
│   │   └── split.py            # Dataset splitting utilities
|   |
│   ├── training/               # Model training (in progress)
│   │   ├── models.py           # Model architecture definitions
│   │   ├── components.py       # Reusable model components
│   │   └── trainer.py          # Training loop implementation
|   |
│   └── validating/             # Validation and analysis (in progress)
│       ├── metrics.py          # Accuracy metrics computation
│       └── visualization.py    # Result plotting and visualization
|
├── external/                   # External dependencies
│   ├── rdMorganFP/             # Drug fingerprinting utilities
│   ├── biomed-multi-view/      # Multiple drug representation models (graph image, text)
│   ├── ESPF/                   # Protein fingerprinting utilities
│   ├── ESM/                    # Protein language model (ESM-C 6B)
│   ├── nucleotide-transformer/ # DNA sequence transformer (nucleotide-transformer)
│   └── run_embeddings.sh       # Script to run embedding generation
|
├── data/                       # Data directory (gitignored, download from [here](https://test.com))
│   ├── source/                 # Original datasets, populated by loading notebook & tdc
│   ├── processed/              # Processed datasets & embeddings
│   ├── input/                  # Input datasets (h5torch)
│   ├── images/                 # Generated plots and visualizations
│   ├── checkpoints/            # Model checkpoints
│   └── results/                # Model outputs and analysis
|
├── notebooks/                  # Jupyter notebooks for reproducing experiments
│   ├── loading.ipynb           # Data loading, pre-processing and exploration
│   ├── processing.ipynb        # Embedding generation and h5torch file creation
│   ├── training.ipynb          # Model inspection and training processes (in progress)
│   └── validating.ipynb        # Result analysis and validation (in progress)
|
├── scripts/                    # Scripts for running experiments on HPC
│   ├── configs/                # Configuration files
│   │   ├── pretrain.json
│   │   ├── train.json
│   │   └── valid.json
│   ├── embedding.sh            # Shell script for embedding generation
│   ├── pretrain.sh             # Shell script for pretraining
│   ├── train.sh                # Shell script for training
│   ├── validate.sh             # Shell script for validation
│   └── hpc.pbs                 # PBS script with batch indexing
|
├── setup.py                    # Package installation script
├── environment.yml             # Conda environment specification
└── README.md                   # Project documentation
```

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended)
- Conda package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/robsyc/MB-VAE-DTI.git
   cd MB-VAE-DTI
   ```

2. **Run the setup script**
   ```bash
   ./setup_env.sh
   ```
   This script will:
   - Create a conda environment with the necessary CUDA dependencies
   - Install all required Python packages
   - Set up the package in development mode

3. **Activate the environment**
   ```bash
   conda activate mb-vae-dti
   ```

## Quick Start

**Download the complete data folder** including all datasets and pre-trained models:
   - Link: TBA (coming soon)
   - This provides everything needed to immediately start experimenting with the models.

**Note:** This project is currently in development, with the processing section implemented and training/validation components in progress.

## Current Progress

- ✅ Data loading and preprocessing
- 🔄 Embedding generation and processing (in progress)
- 🔄 h5torch file creation and dataset splitting (in progress)
- 🔄 Model training (later)
- 🔄 Model validation and analysis (later)

The training section has not yet been refactored to the new codebase. We are actively working on implementing and testing the quickstart. There currently are still some `archive` folders spread throughout the repository, which contain old code to be refactored.

## Scripts

These scripts are designed to run on an HPC cluster with `Python 3.9`, `PyTorch 2.1.2` and `CUDA 12.1.1`.
They require seperate package-management.

```bash
# Generate embeddings (to be implemented)
bash scripts/embedding.sh

# Pre-training (to be implemented)
bash scripts/pretrain.sh

# Training (to be implemented)
bash scripts/train.sh

# Validation (to be implemented)
bash scripts/validate.sh
```

### Notebooks

...

## Results

The model has been evaluated on standard DTI benchmark datasets:
...

## Citation

If you use this code in your research, please cite:

```
@article{mbvae_dti,
  title={Multi-branch Variational Autoencoders for Drug-target Interaction Prediction and Molecular Generation},
  author={Claeys, Robbe},
  year={2025}
}
```

## Acknowledgments

This work builds upon several open-source projects, including [DiGress](https://github.com/cvignac/DiGress), [DiffMS](https://github.com/coleygroup/DiffMS), and the various embedding models mentioned above.