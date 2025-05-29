#!/usr/bin/env python3
"""
Test script to verify the h5torch fixes for pretrain dataset creation and loading.
"""

import logging
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / "MB-VAE-DTI"))

from mb_vae_dti.processing.h5factory import create_pretrain_h5torch, inspect_h5torch_file
from mb_vae_dti.processing.h5datasets import PretrainDataset

# Set logging to debug
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    temp_dir = Path("/home/robsyc/Desktop/thesis/MB-VAE-DTI/external/temp")
    output_dir = Path("/home/robsyc/Desktop/thesis/MB-VAE-DTI/data/input")
    
    target_input_files = [temp_dir / "pretrain_aa.hdf5", temp_dir / "pretrain_dna.hdf5"]
    target_output_file = output_dir / "targets_pretrain_fixed.h5torch"
    
    # Remove existing file if it exists
    if target_output_file.exists():
        target_output_file.unlink()
        print(f"Removed existing file: {target_output_file}")
    
    if not all(f.exists() for f in target_input_files):
        print(f"Input files not found: {[str(f) for f in target_input_files if not f.exists()]}")
        return
    
    print("\n--- Creating H5torch File with Fixes ---")
    try:
        create_pretrain_h5torch(
            input_h5_paths=target_input_files,
            output_h5_path=target_output_file,
            split_name='is_train'  # Use boolean split column
        )
        print(f"✓ Successfully created: {target_output_file}")
    except Exception as e:
        print(f"✗ Failed to create h5torch file: {e}")
        return
    
    print("\n--- Inspecting H5torch File ---")
    inspect_h5torch_file(target_output_file)
    
    print("\n--- Testing Dataset Loading ---")
    try:
        # Test train dataset
        pretrain_targets_train = PretrainDataset(
            target_output_file,
            subset_filters={'split_col': 'is_train', 'split_value': True}
        )
        print(f"✓ Train dataset size: {len(pretrain_targets_train)}")
        
        # Test validation dataset
        pretrain_targets_val = PretrainDataset(
            target_output_file,
            subset_filters={'split_col': 'is_train', 'split_value': False}
        )
        print(f"✓ Validation dataset size: {len(pretrain_targets_val)}")
        
        # Test sampling
        print("\n--- Testing Data Sampling ---")
        sample = pretrain_targets_train[0]
        print(f"✓ Sample keys: {list(sample.keys())}")
        print(f"✓ Sample ID: {sample['id']}")
        print(f"✓ Representations: {list(sample['representations'].keys())}")
        print(f"✓ Features: {list(sample['features'].keys())}")
        
        # Show sample data
        for repr_name, repr_value in sample['representations'].items():
            print(f"  - {repr_name}: {repr_value[:50]}..." if len(str(repr_value)) > 50 else f"  - {repr_name}: {repr_value}")
        
        for feat_name, feat_value in sample['features'].items():
            print(f"  - {feat_name}: shape={feat_value.shape}, dtype={feat_value.dtype}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 