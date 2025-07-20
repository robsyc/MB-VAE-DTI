# Multi-branch Drug-target Interaction model

We are building a model for the dyadic prediction of drug-target interactions.

## Overview of the DTITree Model

The model is a two-branch model, namely, a DTITree model with a drug Branch & a target Branch.

The drug branch is composed of several feature vector encoders, an aggregation module and a discrete denoising diffusion model for the generation of drug molecules. The target branch lacks a decoder module, it simply encodes the target features into an embedding.

In the DTITree model, the drug and target branches' output embeddings are passed on to an aggregation module and a final prediction head.

### Model variants

Several aspects of the model can be varied:
- type of encoding module in the individual branches (MLP or transformer)
- intra-branch-level aggregation module (concat-mlp, attentive-mlp or cross-attention)
- inter-branch-level aggregation module (same)
- size & depth of the encoding & aggregation modules
- variational or non-variational encoding
- inclusion or exclusion of the diffusion decoder module in the drug branch

We want our implementation to be flexible enough to allow for easy experimentation with these variations through config settings.

## Training procedures

One or more of the following training procedures can be run. We want to be able to specify which steps to run and which datasets to use for each step, making sure that weights are transferred correctly between steps in case multiple training procedures are run.

### 1. Pre-training on drugs or targets

We can pre-train the drug or target branch on a large dataset of drug molecules or protein targets respectively. The goal is to learn a good high-level representation of the drug or target molecules. The objectives are: complexity (VAE-style regularization), and/or contrastive learning and/or reconstruction.

### 2. Training on combined DTI dataset -> multi-score prediction

We can train the DTITree model on a combined dataset of drug-target pairs. Multiple DTI scores can be used as targets, and the DTITree model will learn to predict the interaction scores for each of them. The objectives are: accuracy in terms of BCE on the interaction class (0 or 1) and MSE on the interaction scores.

### 3. Fine-tuning on benchmark DTI datasets -> single-score prediction

We can fine-tune the DTITree model on small benchmark DTI datasets (Davis, KIBA, etc.) which are subsets of the prior combined dataset used in training prod 2. The goal is to evaluate the performance of the DTITree model on these benchmark datasets by predicting a single real-values DTI score (pKd, pKi, and KIBA-score).

## Losses

There are several losses used in the training process:
- **Complexity (KL) loss** in case of variational encoding (using $\beta$-approach)
  - drug Branch: KL(q(z|d) || p(z))
  - target Branch: KL(q(z|t) || p(z))
  - DTITree: KL(q(z|d,t) || p(z))
- **Contrastive loss** during Branch pre-training
  - drug Branch: contrastive loss based on Tanimoto similarity on Morgan fingerprints
  - target Branch: contrastive loss based on Tanimoto similarity on ESP fingerprints
- **Reconstruction loss** in the drug Branch (when using the diffusion decoder)
  - cross-entropy over node predictions (X)
  - cross-entropy over edge predictions (E) (weighted 5 times higher)
- **Accuracy loss** in the DTITree model
  - BCE loss over binary interaction class (0 or 1)
  - MSE loss over real-valued interaction scores (one or more of the following)
    - pKd
    - pKi
    - KIBA-score

Our implementation should be flexible enough to allow for different combinations of (and weights for) these losses.

## Datasets

We use the `h5torch` package for storing & loading datasets; it is a wrapper around `h5py` & `torch`.

### Pre-training datasets

At initialisation, the `PretrainDataset` class loads the PyTorch Dataset from the `h5torch` file. The train/validation split is already defined in the file itself, we simply need to specify which subset to load. Sampling for this datasets returns a dictionary with all representations & features. The features are pre-computed foundation model embeddings which we use as inputs for the individual branches.

```python
from pathlib import Path
from mb_vae_dti.processing import PretrainDataset # /home/robsyc/Desktop/thesis/MB-VAE-DTI/mb_vae_dti/processing

output_dir = Path("/home/robsyc/Desktop/thesis/MB-VAE-DTI/data/input")

target_output_file = output_dir / "targets.h5torch"
drug_output_file = output_dir / "drugs.h5torch"

targets_pretrain_training = PretrainDataset(
    h5_path=target_output_file,
    subset_filters={'split_col': 'is_train', 'split_value': True},
    load_in_memory=False
)
sample = targets_pretrain_training[42]
# 2025-05-30 18:15:27,808 - INFO - Subset mask for targets.h5torch: kept 171765 / 190851 items
# 2025-05-30 18:15:27,810 - INFO - Initialized PretrainDataset from targets.h5torch. Size: 171765 items.
# 2025-05-30 18:15:27,811 - INFO -   Features (Axis 0): ['EMB-ESM', 'EMB-NT', 'FP-ESP']
# 2025-05-30 18:15:27,811 - INFO -   Representations (Axis 0): ['aa', 'dna']
# Example sample:
# {
#   'id': 49,
#   'representations': {
#     'aa': 'MAAAMTFCRLLNRCGEAARSLPLGARC...', 
#     'dna': 'ATGGCGGCGGCGATGACCTTCTGCCG...'},
#   'features': { # the inputs are fixed-length pre-computed embeddings
#     'EMB-ESM': array([-0.01264881,  0.00669643, -0.00759549, ..., ], dtype=float32), # 1152 dims
#     'EMB-NT': array([ 0.3568277 ,  0.11620766, -0.11930461, ..., ], dtype=float32), # 1024 dims
#     'FP-ESP': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)} # 4170 dims - this is used for contrastive loss!
# }

drugs_pretrain_validation = PretrainDataset(
    h5_path=drug_output_file,
    subset_filters={'split_col': 'is_train', 'split_value': False}
)
sample = drugs_pretrain_validation[42]
# 2025-05-30 18:21:07,335 - INFO - Subset mask for drugs.h5torch: kept 200000 / 2000000 items
# 2025-05-30 18:21:07,342 - INFO - Initialized PretrainDataset from drugs.h5torch. Size: 200000 items.
# 2025-05-30 18:21:07,342 - INFO -   Features (Axis 0): ['EMB-BiomedGraph', 'EMB-BiomedImg', 'EMB-BiomedText', 'FP-Morgan']
# 2025-05-30 18:21:07,343 - INFO -   Representations (Axis 0): ['smiles']
# Example sample:
# {
#   'id': 313, 
#   'representations': { 'smiles': 'CCOc1cc2c(c(O)c1OCC)C(=O)NC1C2CC(O)C(O)C1O' },
#   'features': {
#     'EMB-BiomedGraph': array([ 3.28338034e-02,  5.93008325e-02, -6.75319880e-02, ...], dtype=float32), # 512 dims
#     'EMB-BiomedImg': array([0.8514107 , 0.41632158, 0.8484036 , ...], dtype=float32), # 512 dims
#     'EMB-BiomedText': array([ 1.031939  ,  0.00464498,  0.59976, ...], dtype=float32), # 768 dims
#     'FP-Morgan': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)} # 2048 dims
# }
```

Notes:
- The pre-train datasets only contain a training & validation split, no test split.
- The discrete diffusion model requires conversion of the 'smiles' representation to a molecular graph. This currencly is not implemented in the `PretrainDataset` class, but we can use the `Chem.MolFromSmiles` function from the `rdkit` package to do this.
- For computing the contrastive loss, we need to compute the Tanimoto similarity between the fingerprint features, this also is not implemented in the `PretrainDataset` class.
- The DiGress/DiffMS diffusion decoder requires dataset statistics (e.g. max_n_heavy_atoms, max_mol_weight, node/edge_type_marginals, etc.) to be initialized. Particulary, the instantiation of the forward diffusion transition models require this information for cleverly adding noise to molecular graphs. We will need to implement functions to compute these statistics from the dataset (and save them somewhere to avoid recomputing them every time).

### Training datasets

Likewise, the DTI training datasets are stored in an `h5torch` file, however, instead of a simple ND-array of samples, we have a COO-matrix of drug-target pairs along with their interaction scores. There are three additional properties we can specify:
- which source datasets to use thanks to provenance tracking
- whether to use the training, validation or test split
- whether to use the random split or the cold-drug split
  - `split_random`: drug-target combinations are randomly assigned to training, validation or test
  - `split_cold`: unique drugs are randomly assigned to training, validation or test

```python
from mb_vae_dti.processing import DTIDataset

dti_output_file = output_dir / "dti.h5torch"

dti_dataset = DTIDataset(
    h5_path=dti_output_file,
    subset_filters={
        'split_col': 'split_cold',
        'split_value': 'train',
        'provenance_cols': ['in_DAVIS', 'in_KIBA']
        }
)
sample = dti_dataset[42]

# 2025-06-27 15:49:58,816 - DEBUG - Calculating subset mask for /home/robsyc/Desktop/thesis/MB-VAE-DTI/data/input/dti.h5torch with filters: {'split_col': 'split_cold', 'split_value': 'train', 'provenance_cols': ['in_DAVIS', 'in_KIBA']}
# 2025-06-27 15:49:58,817 - DEBUG - COO mask size based on central/indices: 339197
# 2025-06-27 15:49:58,818 - DEBUG - Applying split filter: 'split_cold' == 'train'
# 2025-06-27 15:49:58,889 - INFO - Subset mask for dti.h5torch: kept 77656 / 339197 items
# 2025-06-27 15:49:58,895 - INFO - Pre-loaded unstructured Y data for 3 columns: ['Y_KIBA', 'Y_pKd', 'Y_pKi']
# 2025-06-27 15:49:58,895 - INFO - Initialized DTIDataset from dti.h5torch. Size: 77656 interactions.
# 2025-06-27 15:49:58,895 - INFO -   Drug paths (Axis 0): ['Drug_ID', 'Drug_InChIKey', 'EMB-BiomedGraph', 'EMB-BiomedImg', 'EMB-BiomedText', 'FP-Morgan', 'SMILES']
# 2025-06-27 15:49:58,896 - INFO -   Target paths (Axis 1): ['AA', 'DNA', 'EMB-ESM', 'EMB-NT', 'FP-ESP', 'Target_Gene_name', 'Target_ID', 'Target_RefSeq_ID', 'Target_UniProt_ID']
# 2025-06-27 15:49:58,897 - INFO -   Pre-loaded unstructured Y data: ['Y_KIBA', 'Y_pKd', 'Y_pKi']
# Example sample:
# {
#   'id': 42, 
#   'y': {
#       'Y': 0.0, 
#       'Y_KIBA': 11.2, 
#       'Y_pKd': None, 
#       'Y_pKi': None}, 
#   'drug': {
#       'id': {
#           'Drug_ID': 'D000465', 
#           'Drug_InChIKey': 'WOTLXQZLXOXMFD-UHFFFAOYSA-N'}, 
#       'representations': {
#           'SMILES': 'C#Cc1cc2c(cc1OC)-c1[nH]nc(-c3ccc(C#N)nc3)c1C2'}, 
#       'features': {
#           'EMB-BiomedGraph': array([ 3.00971679e-02,  5.47647662e-02, -6.93628863e-02, ...], dtype=float32),
#           'EMB-BiomedImg': array([0.82971275, 0.4069119 , 0.8433203 , ...], dtype=float32), 
#           'EMB-BiomedText': array([ 7.38279283e-01, -4.48157609e-01,  5.84223747e-01, ...], dtype=float32), 
#           'FP-Morgan': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)}
#   }, 
#   'target': {
#       'id': {
#           'Target_Gene_name': 'PLK1', 
#           'Target_ID': 'T000212', 
#           'Target_RefSeq_ID': 'NM_005030', 
#           'Target_UniProt_ID': 'P53350'}, 
#       'representations': {
#           'AA': 'MSAAVTAGKLARAPADPGKAGVPGV...', 
#           'DNA': 'ATGAGTGCTGCAGTGACTGCAGGG...'}, 
#       'features': {
#           'EMB-ESM': array([-0.0009814 ,  0.00653409, -0.00374483, ...], dtype=float32), 
#           'EMB-NT': array([ 0.21353887,  0.08736535, -0.31667355, ...], dtype=float32), 
#           'FP-ESP': array([0., 1., 0., ..., 0., 0., 0.], dtype=float32)}
#   }
# }
```

Notes:
- The `y` key contains the interaction scores for the drug-target pair. One or more of the real-valued scores will be available, from this the binary `Y` class was inferred (0 or 1). Our DTITree accuracy loss should be able to handle this potential lack of some of the real-valued scores; we only want to backpropagate the loss for the available scores in a sample-wise manner.
- We want to evaluate our model(s) in both the random & cold-drug split settings, the latter is more realistic for actual use-cases where drugs are not known in advance.
- Which real-valued scores are available depends on the source dataset(s) used. Note that overlap between the datasets is possible, such that some samples (drug-target pairs) will have multiple scores available.
  - Davis dataset only contains pKd scores
  - KIBA dataset only contains KIBA scores
  - Metz dataset only contains pKi scores
  - BindingDB_Kd & BindingDB_Ki datasets only contain pKd & pKi scores respectively

## DTITree Model configurations

### Baseline uni-modal & single-score model

This is the simplest configuration, no fancy training procedures and only simple inputs & outputs.

Core aspects:
- No pre-training (1.) & no general DTI training (2.), only fine-tuning on benchmark datasets (3.)
- No diffusion decoder so no reconstruction loss
- Only a single drug/target feature (e.g. Morgan & ESP fingerprints) so no intra-branch-level aggregation
- Only single DTI score (e.g. pKd) is predicted so no inter-branch-level aggregation is needed; we can use a simple dot-product prediction head & single MSE loss

Variations to tune:
- Core hyperparameters: learning rate, batch size
- Size, depth and type of the encoding modules in both branches
- Which input features to use (e.g. Morgan & ESP fingerprints, or BioMedGraph & ESM embeddings, etc.)
- Which DTI benchmark dataset to use (& which score to predict (e.g. pKd, pKi, KIBA-score, etc.))

### Multi-modal & single-score model

This slightly more complex configuration adds the idea of using multiple input modalities (fingerprints & embeddings) for the drug & target branches. It is used to assess the benefits of using multiple drug/target representations.

Core aspects:
- No pre-training (1.) & no general DTI training (2.), only fine-tuning on benchmark datasets (3.)
- No diffusion decoder so no reconstruction loss and only a single DTI score is predicted so no inter-branch-level aggregation module is needed; we can again use a simple dot-product prediction head & single MSE loss
- Multiple input modalities so we need to use an intra-branch-level aggregation module in both branches
- Only single DTI score (e.g. pKd) is predicted so no inter-branch-level aggregation is needed

Variations to tune:
- Core hyperparameters: learning rate, batch size
- Size, depth and type of the encoding modules
- Size, depth and type of the intra-branch-level aggregation module
- Which combination of input modalities to use (e.g. embeddings only, fingerprints and embeddings, etc.)
- Which DTI benchmark dataset to use (& which score to predict (e.g. pKd, pKi, KIBA-score, etc.))

### Uni-modal & multi-score model

This slightly more complex configuration adds the general DTI training step (2.) by predicting multiple DTI scores. It is used to evaluate the performance gains of training on the larger combined (but heterogeneous) dataset.

Core aspects:
- No pre-training (1.) but general DTI training (2.) & fine-tuning on benchmark datasets (3.) so we'll want to utilize transfer learning
- Still no diffusion decoder & only a single drug/target feature so no intra-branch-level aggregation
- Multiple DTI scores are predicted so we need an inter-branch-level aggregation module & more intricate prediction head/losses (BCE & multiple MSEs)

Variations to tune:
- Core hyperparameters: learning rate, batch size, **loss weights** for BCE & MSEs
- Size, depth and type of the encoding modules
- Size, depth and type of the inter-branch-level aggregation module
- Again, which input features to use & which DTI benchmark dataset to use in fine-tuning phase (3.)

### Hybdrid model: multi-modal and multi-score

This is a hybrid of the multi-modal & multi-score model. It will be compared to the full model to evaluate the benefits / drawbacks of having a reconstruction objective in the drug branch.

Core aspects:
- Pre-training of both branch using only the contrastive loss (1.)
- General DTI training (2.) & fine-tuning on benchmark datasets (3.)
- No diffusion decoder so no reconstruction loss
- Multiple input modalities so we need an intra-branch-level aggregation module
- Multiple DTI scores are predicted so we need an inter-branch-level aggregation module & intricate prediction head/losses (BCE & multiple MSEs)

Variations to tune:
- Core hyperparameters: learning rate, batch size, **loss weights** for BCE & MSEs
- Size, depth and type of the encoding modules
- Size, depth and type of the inter-branch-level aggregation module
- Which input features to use & which DTI benchmark dataset to use in fine-tuning phase (3.)

### Full model: multi-modal, multi-score, pre-training, variational & diffusion drug-decoder

This is the most complex configuration, and utilizes all relevant components of the DTITree model.

Core aspects:
- Pre-training (1.), general DTI training (2.) & fine-tuning on benchmark datasets (3.), so we'll need to allow for intricate transfer learning
- Diffusion decoder in the drug branch so reconstruction loss is used
- Multiple input modalities per branch so we need intra-branch-level aggregation modules
- Multiple DTI scores are predicted so we need an inter-branch-level aggregation module & intricate prediction head/losses (BCE & multiple MSEs)
- Pre-training with contrastive loss in both branches
- Variational encoding in the drug branch so complexity loss is used

Variations to tune:
- Core hyperparameters: learning rate, batch size, **loss weights** for all components (complexity, contrastive, reconstruction, accuracy) and diffusion-decoder specific hyperparameters (diffusion steps, etc.)
- Size, depth and type of the encoding modules in both branches
- Size, depth and type of the intra-branch-level aggregation modules
- Size, depth and type of the inter-branch-level aggregation module
- Which combination of input modalities to use (e.g. embeddings only, fingerprints and embeddings, etc.)
- Which DTI benchmark dataset to use (& which score to predict (e.g. pKd, pKi, KIBA-score, etc.)) in the fine-tuning phase (3.)

---

> NOTE
> Our implementation should lend itself nicely to tuning of hyperparameters. Our training will be done on an HPC cluster to which job scripts will be submitted. We want to spread experiments (different learning rates, batch sizes, model configurations, etc.) randomly across batches such that we can submit multiple jobs in parallel. We also want to log our results to wandb in a structured way.

> An example job script:
```bash
#!/bin/bash

# Basic parameters
#PBS -N exp2_batch2             ## Job name
#PBS -l nodes=1:ppn=8:gpus=1    ## nodes, processors per node (ppn=all to get a full node), GPUs (H100 with 32gb)
#PBS -l walltime=24:00:00       ## Max time your job will run (no more than 72:00:00)
#PBS -l mem=16gb                ## If not used, memory will be available proportional to the max amount
#PBS -m abe                     ## Email notifications (abe=aborted, begin and end)

# Load the necessary modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1  # Load the PyTorch module
# pip install h5torch

# Change working directory to the location where the job was submmitted
cd /data/gent/454/vsc45450/thesis/MB-VAE-DTI

# Run the script from 0 to 11
python3 ./scripts/run_model.py --batch_index 0 --total_batches 12
```

> NOTE
> The DiGress/DiffMS codebases implement additional validation metrics for tracking e.g. the Tanimoto similarity between the predicted and true molecular graphs, the validity of the generated molecules, etc. We will likely need to implement these as well & incorporate them into the denoising decoder.

---

Testing

1. Baseline uni-modal & single-score model
  CUDA_VISIBLE_DEVICES="" python scripts/training/run.py --model baseline --phase finetune --dataset DAVIS --split cold --override training.max_epochs=1 data.batch_size=16 hardware.gpus=0 hardware.deterministic=false data.pin_memory=false data.num_workers=0