from mb_vae_dti.processing.h5factory import create_pretrain_h5torch
from mb_vae_dti.processing.cap_drugs_h5torch import cap_drugs_h5torch
import logging
from pathlib import Path

# set logging to debug
logging.basicConfig(level=logging.DEBUG)

temp_dir = Path("/home/robsyc/Desktop/thesis/MB-VAE-DTI/external/temp")
output_dir = Path("/home/robsyc/Desktop/thesis/MB-VAE-DTI/data/input")

# PRE-TRAIN DATASETS
# DRUGS
drug_input_file = [temp_dir / "pretrain_smiles.hdf5"]
drug_output_file = output_dir / "drugs_pretrain.h5torch"

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