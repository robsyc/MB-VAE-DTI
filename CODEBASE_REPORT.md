# MB-VAE-DTI Codebase Report

## Overview

This report provides a comprehensive analysis of the Multi-branch Variational Encoders for Drug-Target Interaction (MB-VAE-DTI) codebase. The project implements a machine learning framework for predicting drug-target interactions using multi-branch variational autoencoders with multiple embedding strategies.

## Current Structure

The codebase is organized into the following main components:

### 1. Forked Repositories

- **bmfm_sm**: A forked repository containing biomedical foundation models for small molecules. This provides pre-trained models for molecular representation.
- **ESPF**: A forked repository containing protein encoding utilities, specifically subword tokenization for proteins.

### 2. Data Management

- **data/**: Contains source datasets (DAVIS, KIBA, BindingDB) in various formats (.tab, .csv)
  - **dataset/**: Processed datasets in h5torch format for efficient loading
  - **logs/**: Training logs and visualization plots
  - **model_saves/**: Saved model checkpoints

### 3. Experiment Workflow

- **notebooks/**: Jupyter notebooks for interactive experimentation
  - **data.ipynb**: Data loading, cleaning, and preprocessing of DTI datasets
  - **models.ipynb**: Model building, training, and evaluation
  - **analysis.ipynb**: Analysis of model results and performance
  - **data_classification.ipynb**: Classification-specific data processing

- **scripts/**: Python and bash scripts for batch processing
  - **run_model.py**: Main script for running model experiments with hyperparameter grid search
  - **get_data.py**: Script for downloading and preparing data
  - **hpc.pbs**: PBS script for running on HPC clusters

### 4. Core Functionality

- **utils/**: Python modules containing the core functionality
  - **dataLoading.py**: Functions for loading data from various sources
  - **dataProcessing.py**: Data preprocessing and transformation functions
  - **preEmbedding.py**: Functions for generating embeddings for drugs and targets
  - **modelBuilding.py**: Model architecture definitions, including the DrugTargetTree model
  - **modelTraining.py**: Training and evaluation functions
  - **model.py** and **data.py**: Older versions of the code (possibly deprecated)
  - **dataProcessingOld.py** and **embeddingOld.py**: Older versions of processing code

## Functional Analysis

### 1. Data Processing Pipeline

The data processing pipeline consists of several steps:

1. **Data Loading**: 
   - Uses the TDC (Therapeutics Data Commons) package to load standard DTI datasets
   - Supports DAVIS, KIBA, and BindingDB datasets
   - Handles different data formats and structures

2. **Data Filtering and Transformation**:
   - Applies filters to remove non-drug-like molecules (based on molecular weight, number of heavy atoms)
   - Filters out protein targets that are too large (>1200 amino acids)
   - Transforms interaction values (Kd, Ki) to logarithmic scale for better modeling

3. **Feature Generation**:
   - For drugs:
     - Molecular fingerprints (ECFP, MACCS)
     - Graph-based embeddings using GNNs
     - Pre-trained embeddings from biomedical foundation models
   - For targets:
     - Protein fingerprints
     - Sequence embeddings using ProstT5
     - Subword tokenization

4. **Data Storage**:
   - Uses h5torch format for efficient storage and loading
   - Organizes data into train/validation/test splits
   - Supports both random and cold-split scenarios

### 2. Model Architecture

The core model architecture is the DrugTargetTree, which has the following components:

1. **Input Branches**:
   - Flexible input handling for different types of drug and target representations
   - Support for multiple input modalities per branch

2. **Encoder Networks**:
   - ResidualMLP blocks for feature extraction
   - Attention mechanisms for feature weighting
   - Support for both standard and variational encoders

3. **Latent Space**:
   - Shared latent space for drug and target representations
   - Variational sampling for regularization
   - KL divergence loss for variational models

4. **Decoder**:
   - Joint processing of latent representations
   - Prediction of interaction values

### 3. Training and Evaluation

The training and evaluation process includes:

1. **Training Loop**:
   - Batch processing with DataLoader
   - Support for various loss functions (MSE, KL divergence)
   - Learning rate scheduling
   - Early stopping

2. **Hyperparameter Optimization**:
   - Grid search over hyperparameters
   - Parallel execution on HPC
   - Result logging and tracking

3. **Evaluation Metrics**:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Pearson correlation coefficient
   - Spearman correlation coefficient

4. **Visualization**:
   - Learning curves
   - Prediction vs. ground truth plots
   - Attention visualization

## Identified Issues and Improvement Opportunities

### 1. Code Organization

**Issues**:
- Multiple versions of similar files (e.g., dataProcessingOld.py and embeddingOld.py)
- Overlapping functionality between utils and notebooks
- Lack of clear module structure
- No proper Python package structure
- External repositories not properly integrated

**Recommendations**:
- Refactor the codebase into a proper Python package with a clear module structure
- Remove deprecated files and consolidate functionality
- Separate core functionality from experiment-specific code
- Implement proper import structure
- Properly manage external dependencies

### 2. Documentation

**Issues**:
- Limited documentation in the README
- No clear explanation of the project's purpose and architecture
- Missing usage examples and results
- Lack of docstrings in many functions

**Recommendations**:
- Create comprehensive documentation with clear explanations of the project's purpose and architecture
- Add detailed usage examples for different scenarios
- Include results and performance metrics
- Add proper docstrings to all functions and classes

### 3. Dependency Management

**Issues**:
- Manual installation steps in README
- No environment.yml for conda environments
- No version pinning for some dependencies
- External repositories not properly managed

**Recommendations**:
- Create a proper environment.yml file for conda environments
- Pin all dependency versions for reproducibility
- Provide a setup.py for package installation
- Consider using Docker for containerization
- Properly manage external repositories

### 4. Code Quality and Testing

**Issues**:
- Lack of unit tests
- Inconsistent code style
- Some hardcoded paths and parameters
- Limited error handling

**Recommendations**:
- Implement unit tests for core functionality
- Apply consistent code style (using tools like black, flake8)
- Use configuration files for parameters
- Improve error handling and logging

### 5. Workflow Improvements

**Issues**:
- Manual steps in the experimental workflow
- Limited automation for large-scale experiments
- No clear pipeline for reproducing results
- Scripts not properly organized for HPC execution

**Recommendations**:
- Create automated pipelines for the entire experimental workflow
- Implement proper logging and result tracking
- Use tools like MLflow or Weights & Biases for experiment tracking
- Create scripts for reproducing key results
- Reorganize scripts for better HPC execution

## Proposed Refactoring Plan

### Phase 1: Code Organization

1. **Create a proper Python package structure**:
   ```
   mb_vae_dti/
   ├── __init__.py
   ├── loading/
   │   ├── __init__.py
   │   ├── datasets.py        # Loading initial DTI datasets
   │   ├── annotation.py      # FASTA files, CID, etc.
   │   └── visualization.py   # Plotting metrics for loaded data
   ├── processing/
   │   ├── __init__.py
   │   ├── h5torch_creation.py # Creating h5torch files
   │   ├── drug_embedding.py   # Drug embedding generation
   │   └── prot_embedding.py   # Protein embedding generation
   ├── training/
   │   ├── __init__.py
   │   ├── models.py          # Model architecture definitions
   │   ├── components.py      # Reusable model components
   │   └── trainer.py         # Training loop implementation
   └── validating/
       ├── __init__.py
       ├── metrics.py         # Accuracy metrics computation
       └── visualization.py   # Result plotting and visualization
   ```

2. **Reorganize external dependencies**:
   ```
   external/                  # Added to .gitignore
   ├── bmfm_sm/               # Biomedical foundation models
   └── ESPF/                  # Protein encoding utilities
   ```

3. **Restructure data directory**:
   ```
   data/
   ├── source/                # Original datasets
   │   ├── davis.tab
   │   ├── kiba.tab
   │   ├── bindingdb_ki.csv
   │   ├── bindingdb_kd.csv
   │   └── Metz.csv
   ├── processed/             # Processed h5torch files
   │   └── merged_dataset.h5t
   ├── images/                # Generated plots and visualizations
   ├── checkpoints/           # Model checkpoints
   └── results/               # Model outputs and analysis
   ```

4. **Reorganize scripts**:
   ```
   scripts/
   ├── configs/               # Configuration files
   │   ├── embedding_config.json
   │   └── model_config.json
   ├── embedding.sh           # Shell script for embedding generation
   ├── model.sh               # Shell script for model training
   └── hpc.pbs                # PBS script with batch indexing
   ```

5. Remove deprecated files and consolidate functionality
6. Implement proper import structure
7. Separate core functionality from experiment-specific code

### Phase 2: Documentation and Testing

1. Add comprehensive docstrings to all functions and classes
2. Create unit tests for core functionality
3. Update README to reflect new structure and workflow

### Phase 3: Dependency Management and Workflow

1. Create environment.yml for conda environments
2. Implement setup.py for package installation
3. Create automated pipelines for experiments
4. Implement proper logging and result tracking
5. Develop configuration-based experiment execution

## Conclusion

The MB-VAE-DTI codebase implements a promising approach to drug-target interaction prediction using multi-branch variational autoencoders. While the core functionality is solid, there are several opportunities for improvement in terms of code organization, documentation, and workflow automation.

By implementing the proposed refactoring plan, the codebase can be transformed into a more maintainable, reproducible, and user-friendly package that can be more easily used by other researchers in the field. The new structure aligns with the experimental workflow, making it easier to understand and extend.

The current implementation already demonstrates the potential of the approach, with promising results on standard DTI benchmark datasets. With the suggested improvements, the project can have a greater impact on the field of drug discovery and computational biology. 