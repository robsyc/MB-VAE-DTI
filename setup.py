from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mb_vae_dti",
    version="0.1.0",
    author="Robbe Claeys",
    author_email="robbe.claeys@ugent.be",
    description="Multi-branch Variational Autoencoders for Drug-target Interaction Prediction and Molecular Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robsyc/MB-VAE-DTI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.11",
    install_requires=[
        "wheel",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "ipykernel",
        "notebook",
        "rdkit",
        "PyTDC",
        "biopython",
        "subword-nmt",
        "torch_geometric==2.3.1",
        "h5torch",
        "transformers==4.43.4",
        "sentencepiece==0.2.0",
        "fuse-med-ml==0.3.0",
        "pytorch-fast-transformers==0.4.0",
        "fair-esm",
        "upsetplot",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "isort",
        ],
    },
) 