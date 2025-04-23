"""
Utility functions for generating embeddings using external repositories.
"""

import argparse
import h5py
import numpy as np
from typing import Callable
from tqdm import tqdm

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate embeddings for a given HDF5 file")
    parser.add_argument("--input", required=True, help="HDF5 file path containing entities to embed")
    return parser.parse_args()

def process_embeddings_in_batches(
    h5_file_path: str,
    embedding_function: Callable[[str], np.ndarray],
    embedding_name: str,
    batch_size: int = 128
) -> None:
    """
    Process and save embeddings in batches to avoid memory issues.
    
    Args:
        h5_file_path: Path to the HDF5 file
        embedding_function: Function that takes a string representation and returns an embedding
        embedding_name: Name to use for the embedding in the HDF5 file
        batch_size: Number of embeddings to process at once
    """
    with h5py.File(h5_file_path, 'r+') as f:
        # Access the data and ensure it exists
        if 'data' not in f:
            raise KeyError("HDF5 file must contain a 'data' dataset with string representations")
        
        data = f['data']
        num_items = len(data)
        
        # Process the first batch to determine embedding shape and dtype
        first_batch_size = min(batch_size, num_items)
        first_batch_data = [data[j].decode('utf-8') for j in range(first_batch_size)]
        first_batch_embeddings = [embedding_function(item) for item in first_batch_data]
        
        # Use the first embedding to determine shape and dtype
        embedding_shape = first_batch_embeddings[0].shape
        embedding_dtype = first_batch_embeddings[0].dtype
        
        # Create the embeddings dataset if it doesn't exist
        embeddings_group = f.require_group('embeddings')
        if embedding_name in embeddings_group:
            del embeddings_group[embedding_name]
        
        embeddings_dataset = embeddings_group.create_dataset(
            embedding_name, 
            shape=(num_items, *embedding_shape),
            dtype=embedding_dtype
        )
        
        # Save the first batch
        for j, embedding in enumerate(first_batch_embeddings):
            embeddings_dataset[j] = embedding
        
        # Calculate number of batches for progress bar
        num_batches = (num_items + batch_size - 1) // batch_size
        
        print(f"Processing {num_items} items in {num_batches} batches...")
        
        # Use tqdm for the overall progress
        with tqdm(total=num_items, desc=f"Generating {embedding_name} embeddings") as pbar:
            # Update progress bar for the first batch we already processed
            pbar.update(first_batch_size)
            
            # Process remaining batches
            for batch_start in range(first_batch_size, num_items, batch_size):
                # Calculate end of current batch
                batch_end = min(batch_start + batch_size, num_items)
                batch_size_actual = batch_end - batch_start
                
                # Get batch data
                batch_data = [data[j].decode('utf-8') for j in range(batch_start, batch_end)]
                
                # Process batch
                batch_embeddings = [embedding_function(item) for item in batch_data]
                
                # Save batch
                for idx, embedding in enumerate(batch_embeddings):
                    j = batch_start + idx
                    embeddings_dataset[j] = embedding
                
                # Update progress bar
                pbar.update(batch_size_actual)
            
        # Add metadata about the embedding
        embeddings_dataset.attrs['name'] = embedding_name
        embeddings_dataset.attrs['shape'] = embedding_shape
        embeddings_dataset.attrs['dtype'] = str(embedding_dtype)
        
        print(f"Successfully added {embedding_name} embeddings with shape {embedding_shape} to {h5_file_path}")
