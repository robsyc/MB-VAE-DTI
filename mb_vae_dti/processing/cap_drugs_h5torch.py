#!/usr/bin/env python3
"""
Script to create a capped version of drugs.h5torch with only 2 million entries.

This script:
- Randomly selects 2 million entries from the original 3.4 million
- Maintains proper alignment across all datasets
- Re-creates the central index and is_train split accordingly
- Preserves the same structure as the original file
"""

import h5py
import h5torch
import numpy as np
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import math

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent / "MB-VAE-DTI"))

from mb_vae_dti.processing.h5factory import inspect_h5torch_file

# Set logging to debug
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1024

def cap_drugs_h5torch(
    input_h5_path: Path,
    output_h5_path: Path,
    target_size: int = 2_000_000,
    random_seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """
    Creates a capped version of the drugs h5torch file with randomly selected entries.
    
    Args:
        input_h5_path: Path to the existing drugs.h5torch file
        output_h5_path: Path where the capped file will be saved
        target_size: Number of entries to keep (default: 2 million)
        random_seed: Random seed for reproducible selection
        batch_size: Batch size for processing large datasets
    """
    logger.info(f"Capping {input_h5_path} to {target_size} entries -> {output_h5_path}")
    
    if not input_h5_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_h5_path}")
    
    # Remove output file if it exists
    if output_h5_path.exists():
        output_h5_path.unlink()
        logger.info(f"Removed existing output file: {output_h5_path}")
    
    input_file = None
    try:
        # Open input file for reading
        input_file = h5py.File(input_h5_path, 'r')
        
        # Get basic info
        original_size = input_file.attrs.get('n_items', len(input_file['central']))
        entity_type = input_file.attrs.get('entity_type', 'drug')
        
        logger.info(f"Original size: {original_size}, Target size: {target_size}")
        
        if target_size >= original_size:
            logger.warning(f"Target size ({target_size}) >= original size ({original_size}). No capping needed.")
            target_size = original_size
        
        # Generate random indices for selection
        np.random.seed(random_seed)
        selected_indices = np.sort(np.random.choice(original_size, size=target_size, replace=False))
        logger.info(f"Selected {len(selected_indices)} random indices (seed={random_seed})")
        
        # Create output file
        with h5torch.File(str(output_h5_path), 'w') as output_file:
            # Set root attributes
            output_file.attrs['entity_type'] = entity_type
            output_file.attrs['n_items'] = target_size
            
            # 1. Create new central dataset (sequential index)
            logger.info("Creating new central dataset...")
            central_data = np.arange(target_size, dtype=np.uint32)
            output_file.register(
                data=central_data,
                axis='central',
                name='index',
                mode='N-D',
                length=target_size
            )
            logger.info(f"Registered central 'index' (Shape: {central_data.shape})")
            
            # 2. Copy features from axis 0 with selected indices
            logger.info("Copying features from axis 0...")
            if '0' in input_file:
                for feat_name in input_file['0'].keys():
                    if feat_name == 'smiles':  # Handle SMILES separately (string data)
                        continue
                    
                    logger.info(f"Processing feature: {feat_name}")
                    feat_dataset = input_file[f'0/{feat_name}']
                    
                    # Determine if this is a fingerprint (needs uint8 save, float32 load)
                    is_fingerprint = feat_name.startswith("FP-")
                    
                    if is_fingerprint:
                        dtype_save = 'uint8'
                        dtype_load = 'float32'
                    else:
                        dtype_save = None  # Use original
                        dtype_load = 'float32'
                    
                    # Get feature dimensions
                    feat_shape = feat_dataset.shape
                    feat_dims = feat_shape[1:] if len(feat_shape) > 1 else ()
                    
                    # Process in batches
                    first_batch_size = min(batch_size, target_size)
                    first_batch_indices = selected_indices[:first_batch_size]
                    first_batch = feat_dataset[first_batch_indices]
                    
                    if is_fingerprint:
                        # Convert to binary for fingerprints (if not already)
                        if first_batch.dtype != np.uint8:
                            first_batch = (first_batch > 0).astype(np.uint8)
                    
                    output_file.register(
                        first_batch,
                        axis=0,
                        name=feat_name,
                        mode='N-D',
                        dtype_save=dtype_save,
                        dtype_load=dtype_load,
                        length=target_size
                    )
                    
                    # Append remaining batches
                    num_batches = math.ceil(target_size / batch_size)
                    if num_batches > 1:
                        logger.info(f"Appending remaining {target_size - first_batch_size} items for {feat_name}...")
                        for i in tqdm(range(1, num_batches), desc=f"Appending {feat_name}", unit="batch"):
                            start_idx = i * batch_size
                            end_idx = min(start_idx + batch_size, target_size)
                            if start_idx >= end_idx:
                                continue
                            
                            batch_indices = selected_indices[start_idx:end_idx]
                            batch_data = feat_dataset[batch_indices]
                            
                            if is_fingerprint and batch_data.dtype != np.uint8:
                                batch_data = (batch_data > 0).astype(np.uint8)
                            
                            output_file.append(batch_data, f"0/{feat_name}")
                    
                    logger.info(f"Registered aligned[0] '{feat_name}' (Shape: {(target_size,) + feat_dims})")
            
            # 3. Handle SMILES (string data) separately
            logger.info("Processing SMILES data...")
            if '0/smiles' in input_file:
                smiles_dataset = input_file['0/smiles']
                logger.info(f"Reading {target_size} selected SMILES strings...")
                
                # Read selected SMILES in chunks to avoid memory issues
                chunk_size = min(100000, target_size)  # Process in 100k chunks for better efficiency
                smiles_list = []
                
                for chunk_start in tqdm(range(0, target_size, chunk_size), desc="Processing SMILES chunks"):
                    chunk_end = min(chunk_start + chunk_size, target_size)
                    chunk_indices = selected_indices[chunk_start:chunk_end]
                    
                    # Read chunk of SMILES
                    chunk_smiles = smiles_dataset[chunk_indices]
                    
                    # Fast conversion using list comprehension
                    if len(chunk_smiles) > 0 and isinstance(chunk_smiles[0], bytes):
                        # Use list comprehension for bytes - much faster than vectorize
                        chunk_converted = [s.decode('utf-8', errors='replace') for s in chunk_smiles]
                    else:
                        # Use list comprehension for non-bytes
                        chunk_converted = [str(s) for s in chunk_smiles]
                    
                    smiles_list.extend(chunk_converted)
                
                # Convert to numpy array at the end
                smiles_array = np.array(smiles_list, dtype=object)
                
                output_file.register(
                    smiles_array,
                    axis=0,
                    name='smiles',
                    mode='N-D',
                    dtype_save='bytes',
                    dtype_load='str'
                )
                logger.info(f"Registered aligned[0] 'SMILES' (Length: {len(smiles_array)})")
            else:
                logger.warning("No SMILES data found in input file")
            
            # 4. Handle split information - regenerate 90/10 split
            logger.info("Generating new train/validation split...")
            # Generate fresh 90/10 split for the capped dataset
            np.random.seed(random_seed + 1)  # Use different seed to avoid correlation with selection
            train_indices = np.random.choice(target_size, size=int(target_size * 0.9), replace=False)

            # Create boolean array: True for train, False for validation
            is_train = np.zeros(target_size, dtype=bool)
            is_train[train_indices] = True

            output_file.register(
                is_train,
                axis='unstructured',
                name='is_train',
                mode='N-D',
                dtype_save='bool',
                dtype_load='bool',
                length=target_size
            )

            num_train = np.sum(is_train)
            num_val = target_size - num_train
            logger.info(f"Registered unstructured 'is_train' (Train: {num_train}, Val: {num_val})")
            
            logger.info(f"Successfully created capped drugs h5torch file: {output_h5_path}")
            logger.info(f"Reduced from {original_size} to {target_size} entries ({target_size/original_size*100:.1f}%)")
    
    except Exception as e:
        logger.error(f"Failed to cap drugs h5torch file: {e}")
        logger.exception("Traceback:")
        raise
    
    finally:
        if input_file:
            try:
                input_file.close()
            except Exception as e:
                logger.error(f"Error closing input file: {e}")

def main():
    output_dir = Path("/home/robsyc/Desktop/thesis/MB-VAE-DTI/data/input")
    
    input_file = output_dir / "drugs.h5torch"
    output_file = output_dir / "drugs_capped.h5torch"
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    
    print("\n--- Inspecting Original File ---")
    inspect_h5torch_file(input_file)
    
    print("\n--- Creating Capped File ---")
    try:
        cap_drugs_h5torch(
            input_file, 
            output_file, 
            target_size=2_000_000,
            random_seed=42
        )
        print(f"✓ Successfully created capped file: {output_file}")
    except Exception as e:
        print(f"✗ Failed to create capped file: {e}")
        return
    
    print("\n--- Inspecting Capped File ---")
    inspect_h5torch_file(output_file)
    
    print("\n--- Testing Dataset Loading ---")
    try:
        # Import here to avoid circular imports
        from mb_vae_dti.processing.h5datasets import PretrainDataset
        
        # Test train dataset
        pretrain_drugs_train = PretrainDataset(
            output_file,
            subset_filters={'split_col': 'is_train', 'split_value': True}
        )
        print(f"✓ Train dataset size: {len(pretrain_drugs_train)}")
        
        # Test validation dataset
        pretrain_drugs_val = PretrainDataset(
            output_file,
            subset_filters={'split_col': 'is_train', 'split_value': False}
        )
        print(f"✓ Validation dataset size: {len(pretrain_drugs_val)}")
        
        # Test sampling
        print("\n--- Testing Data Sampling ---")
        sample = pretrain_drugs_train[0]
        print(f"✓ Sample keys: {list(sample.keys())}")
        print(f"✓ Sample ID: {sample['id']}")
        print(f"✓ Representations: {list(sample['representations'].keys())}")
        print(f"✓ Features: {list(sample['features'].keys())}")
        
        # Show sample data
        for repr_name, repr_value in sample['representations'].items():
            print(f"  - {repr_name}: {repr_value[:50]}..." if len(str(repr_value)) > 50 else f"  - {repr_name}: {repr_value}")
        
        for feat_name, feat_value in sample['features'].items():
            print(f"  - {feat_name}: shape={feat_value.shape}, dtype={feat_value.dtype}")
        
        # Test that indices are properly aligned
        print("\n--- Testing Index Alignment ---")
        sample_5 = pretrain_drugs_train[5]
        sample_10 = pretrain_drugs_train[10]
        print(f"✓ Sample 5 ID: {sample_5['id']}")
        print(f"✓ Sample 10 ID: {sample_10['id']}")
        print(f"✓ IDs are sequential: {sample_5['id'] < sample_10['id']}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 