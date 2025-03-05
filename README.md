# Multi-branch Variational Encoders for Drug-target Interaction Prediction (MB-VAE-DTI)

A machine learning framework for predicting drug-target interactions using multi-branch variational autoencoders with multiple embedding strategies.

## Project Overview

This project implements a novel approach to the dyadic Drug-Target Interaction (DTI) prediction problem using multi-branch variational autoencoders. The framework supports multiple embedding strategies for both drugs (molecules) and protein targets, allowing for flexible and powerful representation learning.

Key features:
- Multiple embedding strategies for drugs (fingerprints, graph embeddings, images, etc.)
- Multiple embedding strategies for proteins (fingerprints, amino acid and DNA sequence embeddings, etc.)
- Variational and non-variational encoder architectures
- Inspection of latent spaces & exploration of generative capabilities
- Comprehensive evaluation on standard DTI datasets (DAVIS, KIBA, BindingDB) and a new aggregated dataset of >400k interactions & pre-computed embeddings

## Repository Structure

```
MB-VAE-DTI/
├── mb_vae_dti/              # Main Python package
│   ├── loading/
│   │   ├── __init__.py
│   │   ├── datasets.py        # Loading initial DTI datasets
│   │   ├── annotation.py      # FASTA files, CID, etc.
│   │   └── visualization.py   # Plotting metrics for loaded data
│   ├── processing/          # Data processing functionality
│   │   ├── h5torch_creation.py # Creating h5torch files
│   │   ├── drug_embedding.py   # Drug embedding generation
│   │   └── prot_embedding.py   # Protein embedding generation
│   ├── training/            # Model training functionality
│   │   ├── models.py        # Model architecture definitions
│   │   ├── components.py    # Reusable model components
│   │   └── trainer.py       # Training loop implementation
│   └── validating/          # Validation and analysis functionality
│       ├── metrics.py       # Accuracy metrics computation
│       └── visualization.py # Result plotting and visualization
├── external/                # External dependencies (gitignored)
│   ├── bmfm_sm/             # Biomedical foundation models
│   └── ESPF/                # Protein encoding utilities
├── data/                    # Data directory
│   ├── source/              # Original datasets
│   ├── processed/           # Processed h5torch files
│   ├── images/              # Generated plots and visualizations
│   ├── checkpoints/         # Model checkpoints
│   └── results/             # Model outputs and analysis
├── notebooks/               # Jupyter notebooks for experiments
│   ├── loading.ipynb        # Data loading and exploration
│   ├── processing.ipynb     # Data processing and embedding generation
│   ├── training.ipynb       # Model building and training
│   └── validating.ipynb     # Result analysis and validation
├── scripts/                 # Scripts for running experiments
│   ├── configs/             # Configuration files
│   │   ├── embedding_config.json # Embedding generation config
│   │   └── model_config.json     # Model training config
│   ├── embedding.sh         # Shell script for embedding generation
│   ├── model.sh             # Shell script for model training
│   └── hpc.pbs              # PBS script with batch indexing
├── setup.py                 # Package installation script
├── environment.yml          # Conda environment specification
└── README.md                # Project documentation
```

## Installation

### Prerequisites

- Python 3.11
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/robsyc/MB-VAE-DTI.git
cd MB-VAE-DTI

# Create and activate a conda environment
conda env create -f environment.yml
conda activate mbvae_env

# Install the package in development mode
pip install -e .

# Create necessary directories
mkdir -p data/source data/processed data/images data/checkpoints data/results
mkdir -p external
```

### External Dependencies

The project relies on two external repositories that need to be set up separately:

```bash
# Create external directory for dependencies
mkdir -p external
cd external

# Clone the biomedical foundation models repository
git clone https://github.com/path/to/bmfm_sm.git
cd bmfm_sm
# Follow installation instructions in the repository's README
cd ..

# Clone the protein encoding utilities repository
git clone https://github.com/path/to/ESPF.git
cd ESPF
# Follow installation instructions in the repository's README
cd ..

# Return to project root
cd ..
```

### Pre-trained Models

Some embedding methods require pre-trained models:

```bash
# Create model directories
mkdir -p models/bmfm_model_dir models/ProstT5_model_dir

# Download models (manual step)
# 1. Download biomed-smmv-base.pth from:
#    https://ad-prod-biomed.s3.us-east-1.amazonaws.com/biomed.multi-view/data_root_os_v1.tar.gz
#    and place in models/bmfm_model_dir/

# 2. Download ProstT5 files from:
#    https://huggingface.co/Rostlab/ProstT5/tree/main
#    and place in models/ProstT5_model_dir/
```

## Usage

### Data Preparation

The data preparation workflow consists of two main steps:

1. **Loading and preprocessing datasets**:
```bash
# Run the loading notebook
jupyter notebook notebooks/loading.ipynb
```

This will:
- Download DTI datasets (DAVIS, KIBA, BindingDB)
- Apply filters and transformations
- Generate intermediate files for embedding generation

2. **Generating embeddings and creating h5torch files**:
```bash
# Run the processing notebook
jupyter notebook notebooks/processing.ipynb

# Or use the shell script for batch processing
bash scripts/embedding.sh
```

This will:
- Generate embeddings for drugs and proteins
- Create h5torch files for efficient data loading
- Save processed data in the data/processed directory

### Model Training

For interactive experimentation, use the training notebook:

```bash
jupyter notebook notebooks/training.ipynb
```

For large-scale experiments on HPC:

```bash
# Submit job to HPC cluster
qsub scripts/hpc.pbs

# Or run locally with specific batch index
bash scripts/model.sh --batch_index 0 --total_batches 12
```

### Analysis and Validation

Analyze results using the validation notebook:

```bash
jupyter notebook notebooks/validating.ipynb
```

## Model Architecture

The core model architecture is a multi-branch variational autoencoder that can process different types of drug and target representations:

1. **Input Branches**:
   - Drug branch: Processes molecular fingerprints or graph embeddings
   - Target branch: Processes protein fingerprints or sequence embeddings

2. **Encoder**:
   - Each branch has its own encoder network
   - Encoders map inputs to a shared latent space
   - Variational encoders produce mean and variance for sampling

3. **Decoder**:
   - Joint decoder processes latent representations
   - Predicts interaction values (binding affinity)

## Results

The model has been evaluated on standard DTI benchmark datasets:

- **DAVIS**: 72 kinase targets and 442 drugs with Kd values
- **KIBA**: 229 targets and 2,111 drugs
- **BindingDB**: Larger dataset with Ki and Kd values

Performance metrics include MSE, RMSE, Pearson correlation, and Spearman correlation.

## Citation

If you use this code in your research, please cite:

```
@article{mbvae_dti,
  title={Multi-branch Variational Autoencoders for Drug-target Interaction Prediction and Molecular Generation},
  author={Claeys, Robbe},
  year={2025}
}
```