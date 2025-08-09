# Multi-branch Neural Networks for Drug-target Interaction Prediction and Target-conditioned de novo Drug Design (MB-VAE-DTI)

A machine-learning framework for the dyadic drug-target interaction (DTI) prediction problem that combines multi-branch variational autoencoders with discrete diffusion processes. Our architecture uniquely integrates multiple drug/target representations and modalities, pre-trained foundation models, and target-conditioned discrete diffusion-based molecule generation in a unified framework.

**Key features of this project:**
- Multiple drug representations: Morgan fingerprints, graph, image and text (SMILES)
- Multiple protein representations: ESPF fingerprints, amino acid and DNA sequences
- Variational encoders and discrete-diffusion decoder for de novo drug design
- Inspection of latent spaces & exploration of generative capabilities
- Comprehensive evaluation on standard DTI datasets through [tdc](https://tdcommons.ai/) (DAVIS, KIBA, BindingDB and Metz) and a new aggregated dataset of ±300k interactions & pre-computed embeddings.

**Prior work which this project builds upon:**
- [`DiGress`](https://github.com/cvignac/DiGress) by [Vignac et al.](https://arxiv.org/abs/2209.14734) and [`DiffMS`](https://github.com/coleygroup/DiffMS) by [Coley et al.](https://arxiv.org/abs/2409.10000) which inspired the diffusion-based drug-decoding.
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
│   │   └── split.py            # DTI dataset splitting
|   |
│   ├── training/               # Model training (in progress)
│   │   ├── configs/            # Configuration files (incl. gridsearch & ensemble)
│   │   ├── datasets/           # PyTorch Lightning DataModules for DTI & pretraining
│   │   ├── models/             # Model architectures & components
│   │   ├── metrics/            # Pyl metrics for DTI accuracy, and molecular reconstruction
|   |   ├── modules/            # Pyl modules for DTI models
|   |   ├── diffusion/          # Diffusion utilities (in progress)
│   │   ├── utils/              # Training & testing utilities
│   │   └── run.py & test.py    # Main training & testing scripts
|   |
│   └── validating/             # Validation and analysis (in progress)
│       ├── ...
│       └── ...
|
├── external/                   # External dependencies (each has it's own requirements.txt and script.py)
│   ├── rdMorganFP/             # Drug fingerprinting utilities
│   ├── biomed-multi-view/      # Biomed-multi-view embedding models
│   ├── ESPF/                   # Protein fingerprinting utilities
│   ├── ESM/                    # Protein language model (ESM-C 6B)
│   ├── nucleotide-transformer/ # DNA sequence transformer (500M_multi_species_v2)
|   ├── temp/                   # Directory for storing HDF5 files
│   └── run_embeddings.sh       # Shell script to run embedding generation
|
├── data/                       # Data directory (gitignored)
│   ├── source/                 # Original datasets, populated by loading notebook & tdc
│   ├── processed/              # Processed datasets & embeddings
│   ├── input/                  # Input datasets (h5torch)
│   ├── images/                 # Plots and visualizations
│   └── results/                # Model outputs and checkpoints
|
├── notebooks/                  # Jupyter notebooks for reproducing experiments
│   ├── loading.ipynb           # Data loading, pre-processing and exploration
│   ├── processing.ipynb        # Embedding generation and h5torch file creation
│   ├── training.ipynb          # Gridsearch analysis & metrics (in progress)
│   └── validating.ipynb        # Inspection of generative quirks (in progress)
|
├── scripts/                    # Shell scripts for running jobs
│   ├── embedding/
│   ├── training/
│   └── molecular_statistics.py # Molecular properties, marginal distributions, ...
|
├── environment.yml             # Minimal conda setup (Python + RDKit + PyTorch CUDA)
├── pyproject.toml              # All other dependencies (DiffMS + MB-VAE-DTI)
└── README.md                   # This file :)
```

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) package manager
- CUDA-compatible GPU (recommended)

### Setup

```bash
git clone https://github.com/robsyc/MB-VAE-DTI.git
cd MB-VAE-DTI
conda env create -f environment.yml
conda activate mb-vae-dti
```

## Quick Start

**Download the complete data folder** including all datasets and pre-trained models:
   - Link: TBA (coming soon)
   - This provides everything needed to immediately start experimenting with the models
   - Alternatively, you can consult the notebooks to generate the data & run all experiments from scratch
  
  ```bash
  # simplest baseline on random split of DAVIS
  python mb_vae_dti/training/run.py --model baseline --phase finetune --dataset DAVIS --split rand

  # full model on cold split of KIBA
  python mb_vae_dti/training/run.py --model full --phase finetune --dataset KIBA --split cold
  ```

  See `training/run.py` and `training/configs/` for more examples and details.

**Note:** This project is currently in development (training/validation components in progress). There currently are still some `archive` folders spread throughout the repository, which contain old code to be refactored.

## Current Progress

- ✅ Data loading and preprocessing
- ✅ Embedding generation and processing
- ✅ h5torch file creation and dataset splitting
- ✅ Setting up model architectures
- 🔄 Model pre-training w/ contrastive, complexity, (reconstruction) loss
  - Contrastive loss: SimCLR with InfoNCE (1 positive pair and many negatives weighted w/ Tanimoto similarity)
  - Complexity loss: KL divergence between the encoder's output and a standard normal distribution
  - Reconstruction loss: MSE between the diffusion decoder's output and the input (only for the drug branch)
- 🔄 Training baseline model (MLP on FPs & dot-product) and full model (incl. DTI accuracy loss)
- 🔄 Model validation and analysis

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