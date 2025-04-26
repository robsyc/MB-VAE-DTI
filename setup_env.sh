#!/bin/bash
set -e

echo "Setting up MB-VAE-DTI environment..."

# Check if we're on HPC with modules
if command -v module &> /dev/null; then
    echo "HPC environment detected. Loading PyTorch module..."
    module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
    # Create a virtual environment if needed
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python -m venv venv
    fi
    source venv/bin/activate
    pip install -e .
else
    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        echo "Conda not found. Please install Conda first."
        exit 1
    fi

    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q mb-vae-dti; then
        echo "Creating conda environment with PyTorch and CUDA support..."
        conda env create -f environment.yml
    else
        echo "Updating conda environment..."
        conda env update -f environment.yml
    fi

    # Activate the environment and install Python packages
    echo "Installing Python packages..."
    eval "$(conda shell.bash hook)"
    conda activate mb-vae-dti
fi

echo """
--------------------------------
 ✅ Environment setup complete
--------------------------------
"""
if command -v module &> /dev/null; then
    echo "🚀 To activate: source venv/bin/activate"
else
    echo "🚀 To activate: conda activate mb-vae-dti"
fi
echo "📝 Verify CUDA: python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'" 