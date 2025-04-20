from setuptools import setup, find_packages

setup(
    name="mb_vae_dti",
    version="0.1.0",
    description="Multi-branch Variational Encoders for Drug-target Interaction Prediction",
    author="Robbe Claeys",
    packages=find_packages(),
    install_requires=[
        "torch==2.1.2",  # Match PyTorch version with HPC
        "wheel>=0.45.1",
        "rdkit>=2023.9.6",
        "PyTDC>=1.0.0",
        "pandas>=2.2.3",
        "numpy>=1.26.4",
        "h5torch>=0.2.14",
        "biopython>=1.85",
        "ipykernel>=6.29.5",
        "UpSetPlot>=0.9.0",
        "tabulate>=0.9.0",
    ],
    python_requires=">=3.9",
) 