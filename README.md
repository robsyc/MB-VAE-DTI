# Multi-branch Variational Encoders for Drug-target Interaction Prediction (MB-VAE-DTI)

A machine learning framework for the dyadic drug-target interaction (DTI) prediction problem that combines multi-branch variational autoencoders with discrete diffusion processes. Our architecture uniquely integrates multi-modal drug & target representations - graph structures, fingerprints, images, and SMILES for drug molecules and DNA, amino acid sequences, and functional fingerprints for proteins - for large-scale pre-training as target-conditioned discrete generation of drug candidates.

**Key features of this project:**
- Multiple drug representations: Morgan fingerprints, graph, image and text (smiles)
- Multiple protein representations: ESPF fingerprints, amino acid and DNA sequences
- Variational encoder and discrete-diffusion decoder
- Inspection of latent spaces & exploration of generative capabilities
- Comprehensive evaluation on standard DTI datasets through [tdc](https://tdcommons.ai/) (DAVIS, KIBA, BindingDB and Metz) and a new aggregated dataset of ±400k interactions & pre-computed embeddings.

**Prior work which this project builds upon:**
- [`DiGress`](https://github.com/cvignac/DiGress) and [`DiffMS`](https://github.com/coleygroup/DiffMS) which inspired the diffusion-based drug-decoding.
- [`RDKit`](https://www.rdkit.org/) with Morgan fingerprints for generating drug fingerprints.
- [`MMELON`](https://github.com/BiomedSciAI/biomed-multi-view) model by [Suryanarayanan et al.](https://arxiv.org/abs/2410.19704) for generating drug representation embeddings (graph, image and text).
- [`ESPF`](https://github.com/kexinhuang12345/ESPF) by [Huang et al.](https://static1.squarespace.com/static/58f7aae1e6f2e1a0f9a56616/t/5e370e2d12092f15876d5753/1580666413389/paper.pdf) for generating protein fingerprints.
- [`ESM-C 600M`](https://github.com/evolutionaryscale/esm) by the [ESM Team](https://evolutionaryscale.ai/blog/esm-cambrian) for generating protein language model embeddings.
- [`nucleotide-transformer 500M_multi_species_v2`](https://github.com/instadeepai/nucleotide-transformer) by [Dalla-Torre et al.](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2) for the generation of DNA sequence embeddings.

## Repository Structure

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
│   │   ├── ...
│   │   └── ...
|   |
│   └── validating/             # Validation and analysis (in progress)
│       ├── ...
│       └── ...
|
├── external/                   # External dependencies (each folder has it's own venv and script.py)
│   ├── rdMorganFP/             # Drug fingerprinting utilities
│   ├── biomed-multi-view/      # Biomed-multi-view embedding models
│   ├── ESPF/                   # Protein fingerprinting utilities
│   ├── ESM/                    # Protein language model (ESM-C 6B)
│   ├── nucleotide-transformer/ # DNA sequence transformer (500M_multi_species_v2)
│   └── run_embeddings.sh       # Shell script to run embedding generation
|
├── data/                       # Data directory (gitignored)
│   ├── source/                 # Original datasets, populated by loading notebook & tdc
│   ├── processed/              # Processed datasets & embeddings
│   ├── input/                  # Input datasets (h5torch)
│   ├── images/                 # Plots and visualizations
│   ├── checkpoints/            # Model checkpoints
│   └── results/                # Model outputs and analyses
|
├── notebooks/                  # Jupyter notebooks for reproducing experiments
│   ├── loading.ipynb           # Data loading, pre-processing and exploration
│   ├── processing.ipynb        # Embedding generation and h5torch file creation
│   ├── training.ipynb          # Model inspection and training processes (in progress)
│   └── validating.ipynb        # Result analysis and validation (in progress)
|
├── scripts/                    # Scripts for running experiments on HPC (in progress)
│   ├── configs/                # Configuration files
│   │   ├── pretrain.json
│   │   ├── train.json
│   │   └── valid.json
│   ├── embedding.sh            # Shell script for embedding generation (assumes data & external folders are correctly set up)
│   ├── pretrain.sh             # Shell script for pretraining (in progress)
│   ├── train.sh                # Shell script for training (in progress)
│   ├── validate.sh             # Shell script for validation (in progress)
│   └── hpc.pbs                 # PBS script with batch indexing (in progress)
|
├── setup.py                    # Package installation script (may be in conflict with some external repo dependencies)
├── environment.yml             # Conda environment specification
└── README.md                   # This file :)
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
   - This provides everything needed to immediately start experimenting with the models
   - Alternatively, you can consult the notebooks to generate the data & run the experiments yourself

**Note:** This project is currently in development, with the processing section implemented and training/validation components in progress.

## Current Progress

- ✅ Data loading and preprocessing
- ✅ Embedding generation and processing
- 🔄 h5torch file creation and dataset splitting (in progress)
- 🔄 Model training (later)
- 🔄 Model validation and analysis (later)

The training section has not yet been refactored to the new codebase. We are actively working on implementing and testing the quickstart. There currently are still some `archive` folders spread throughout the repository, which contain old code to be refactored.

## Citation

If you use this code in your research, please cite:

```
@article{mbvae_dti,
   title={Multi-branch VAE for Drug-target Interaction Prediction and Target-conditioned de novo Drug Design},
   author={Claeys, Robbe},
   year={2025},
   url={https://...}
}
```

## Acknowledgments

This work is part of a master thesis project at [UGent](https://www.ugent.be/en) under the supervision of [Prof. Willem Waegeman](https://www.ugent.be/dass/en/research/waegeman), [Gaetan De Waele](https://github.com/gdewael) and [Natan Tourné](https://willemwaegeman.github.io/bioml/members/natan-tourne.html) at the [BioML lab](https://willemwaegeman.github.io/bioml/).