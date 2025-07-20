import pandas as pd

from mb_vae_dti.processing.h5factory import create_pretrain_h5torch, create_dti_h5torch
from mb_vae_dti.processing.cap_drugs_h5torch import cap_drugs_h5torch
from mb_vae_dti.processing.split import add_splits

import logging
from pathlib import Path

# set logging to debug
logging.basicConfig(level=logging.DEBUG)

temp_dir = Path("/home/robsyc/Desktop/thesis/MB-VAE-DTI/external/temp")
output_dir = Path("/home/robsyc/Desktop/thesis/MB-VAE-DTI/data/input")

# PRE-TRAIN DATASETS
# DRUGS
drug_input_file = [temp_dir / "pretrain_smiles.hdf5"]
drug_output_file = output_dir / "drugs.h5torch"

if drug_output_file.exists():
    print(f"Skipping drug file creation: {drug_output_file} already exists.")
else:
    print(f"Creating drug file: {drug_output_file}")
    if drug_input_file[0].exists():
        print("\n--- Creating H5torch File for Drugs ---")
        create_pretrain_h5torch(
            input_h5_paths=drug_input_file,
            output_h5_path=drug_output_file
        )
        print(f"Limiting drug file to 2 million entities...")
        cap_drugs_h5torch(
            input_h5_path=drug_output_file,
            output_h5_path=drug_output_file,
            target_size=2_000_000
        )
    else:
        print(f"Skipping drug file creation: {drug_input_file[0]} not found.")

# TARGETS
target_input_files = [temp_dir / "pretrain_aa.hdf5", temp_dir / "pretrain_dna.hdf5"]
target_output_file = output_dir / "targets.h5torch"

if target_output_file.exists():
    print(f"Skipping target file creation: {target_output_file} already exists.")
else:
    print(f"Creating target file: {target_output_file}")
    if target_input_files[0].exists():
        print("\n--- Creating H5torch File for Targets ---")
        create_pretrain_h5torch(
            input_h5_paths=target_input_files,
            output_h5_path=target_output_file
        )
    else:
        print(f"Skipping target file creation: {target_input_files[0]} not found.")

# DTI DATASET
drug_input_file = [temp_dir / "dti_smiles.hdf5"]
target_input_files = [temp_dir / "dti_aa.hdf5", temp_dir / "dti_dna.hdf5"]
dti_output_file = output_dir / "dti.h5torch"

if dti_output_file.exists():
    print(f"Skipping DTI file creation: {dti_output_file} already exists.")
else:
    print(f"Creating DTI file: {dti_output_file}")
    if drug_input_file[0].exists() and target_input_files[0].exists():
        print("\n--- Creating H5torch File for DTI ---")
        df = pd.read_csv("data/processed/dti.csv")
        df = add_splits(df, split_fractions=(0.8, 0.1, 0.1), stratify=True, random_state=42)
        create_dti_h5torch(
            drug_input_h5_paths=drug_input_file,
            target_input_h5_paths=target_input_files,
            output_h5_path=dti_output_file,
            df=df,
        )
    else:
        print(f"Skipping DTI file creation: {drug_input_file[0]} or {target_input_files[0]} not found.")