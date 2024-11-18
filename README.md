# Multi-branch Variational Auto-encoder for Drug-target Interaction Prediction

## Quick Start

```bash
# Clone the repository
git clone git@github.com:robsyc/MB-VAE-DTI.git
cd MB-VAE-DTI

# Create and activate a new conda environment
conda create --name mbvae_env python==3.11
conda activate mbvae_env

# Choose the appropriate torch version for your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install the requirements
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Create data directory and fill with data using the data.ipynb notebook
mkdir data

# Create the model directories and download the models
mkdir models/bmfm_model_dir # add biomed-smmv-base.pth from https://ad-prod-biomed.s3.us-east-1.amazonaws.com/biomed.multi-view/data_root_os_v1.tar.gz
mkdir models/ProstT5_model_dir # add files from https://huggingface.co/Rostlab/ProstT5/tree/main
```