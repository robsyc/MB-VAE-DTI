# Multi-branch Variational Auto-encoder for Drug-target Interaction Prediction

## Quick Start

```bash
# Clone the repository
git clone git@github.com:robsyc/MB-VAE-DTI.git
cd MB-VAE-DTI

# Create a new conda environment and install the dependencies (torch-scatter is a little tricky)
conda create --name mbvae_env python==3.11
conda activate mbvae_env
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Add necessary extras
mkdir data # fill with data using the data.ipynb notebook
mkdir models/bmfm_model_dir # add biomed-smmv-base.pth from https://ad-prod-biomed.s3.us-east-1.amazonaws.com/biomed.multi-view/data_root_os_v1.tar.gz
mkdir models/ProstT5_model_dir # add files from https://huggingface.co/Rostlab/ProstT5/tree/main
```