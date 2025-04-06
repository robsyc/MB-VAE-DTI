"""
This module provides functionality for adding embeddings to existing h5torch files.
It allows for batch processing of features to avoid memory issues.
"""

from typing import Callable, List, Optional, Union, Literal, Dict, Any, Tuple
from pathlib import Path
import h5torch
import h5py
import numpy as np
import pandas as pd
import tqdm
import logging

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

def add_processed_feature(
    file_path: str,
    entity_path: str,
    process_func: Callable[[Union[str, List[str]]], Union[np.ndarray, List[np.ndarray]]],
    feature_name: str,
    batch_size: int = 1,
    dtype_save: str = "float32",
    dtype_load: str = "float32",
    overwrite: bool = False
):
    """
    Generic function to add processed features to h5torch file with batch processing support.
    
    Args:
        file_path (str): Path to the h5torch file
        entity_path (str): Path to entity in h5torch file (e.g. "1/Target_AA")
        process_func (Callable): Function to process entities, can handle batches if batch_size > 1
        feature_name (str): Name for the new feature
        batch_size (int): Number of entities to process at once (1 for no batching)
        dtype_save (str): Data type to save as
        dtype_load (str): Data type to load as
        overwrite (bool): Whether to overwrite existing feature if it exists
    """
    # Parse entity path to get axis
    parts = entity_path.split("/")
    axis = parts[0]
    
    # Open the file in append mode
    f = h5torch.File(file_path, "a")
    
    # Get the axis number if it's a digit
    axis_num = None
    if axis.isdigit():
        axis_num = int(axis)
    
    # Check if feature exists and should be overwritten
    feature_path = f"{axis}/{feature_name}"
    if feature_path in f:
        if overwrite:
            print(f"Found existing {feature_name}, deleting it")
            del f[feature_path]
        else:
            print(f"Feature {feature_name} already exists and overwrite=False. Skipping.")
            f.close()
            return
    
    # Get all entity values
    entities = f[entity_path][:]
    
    # Determine if entities need decoding (from bytes to str)
    needs_decoding = isinstance(entities[0], bytes)
    
    # Process a sample to determine output shape
    sample_entity = entities[0].decode('utf-8') if needs_decoding else entities[0]
    sample_result = process_func(sample_entity)
    result_shape = sample_result.shape if hasattr(sample_result, 'shape') else (1,)
    
    # Create array to store processed results
    n_entities = len(entities)
    output_array = np.zeros((n_entities, *result_shape), dtype=np.float32)
    
    # Adjust batch size if needed
    effective_batch_size = min(batch_size, n_entities)
    
    print(f"Processing {n_entities} entities with batch size {effective_batch_size}, output dimension {result_shape}")
    
    # Process entities in batches
    num_batches = (n_entities + effective_batch_size - 1) // effective_batch_size
    
    for batch_idx in tqdm.tqdm(range(num_batches), desc=f"Processing {feature_name} batches"):
        # Get batch indices
        start_idx = batch_idx * effective_batch_size
        end_idx = min(start_idx + effective_batch_size, n_entities)
        
        # Prepare batch of entities
        batch_entities = entities[start_idx:end_idx]
        
        # Decode if needed
        if needs_decoding:
            batch_entities = [e.decode('utf-8') for e in batch_entities]
        
        # Process batch
        if effective_batch_size == 1:
            # Single item processing
            batch_results = [process_func(batch_entities[0])]
        else:
            # True batch processing
            batch_results = process_func(batch_entities)
            if not isinstance(batch_results, list):
                # If process_func returns a single array for the whole batch
                # Split it into individual results
                batch_results = [batch_results[i] for i in range(len(batch_entities))]
        
        # Store results
        for i, result in enumerate(batch_results):
            idx = start_idx + i
            if idx >= n_entities:
                break
                
            if result_shape == (1,):  # Handle scalar results
                output_array[idx, 0] = result
            else:
                output_array[idx] = result
    
    # Register the processed features
    f.register(
        output_array, 
        mode="N-D", 
        axis=axis_num if axis_num is not None else axis, 
        name=feature_name, 
        dtype_save=dtype_save, 
        dtype_load=dtype_load
    )
    
    # Close the file
    f.close()
    print(f"Successfully added {feature_name} to {file_path}")