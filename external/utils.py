"""
Utility functions for generating embeddings using external repositories.
"""

import argparse
import h5py
import numpy as np
from typing import Callable, List, Dict, Any
from tqdm import tqdm
import math

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate embeddings for a given HDF5 file")
    parser.add_argument("--input", required=True, help="HDF5 file path containing entities to embed")
    return parser.parse_args()

def add_embeddings_to_hdf5(
    h5_file_path: str,
    embedding_name: str,
    batch_processing_function: Callable[[List[str]], List[np.ndarray]],
    batch_size: int,
    model_metadata: Dict[str, Any] = None # Optional metadata about the model
) -> None:
    """
    Reads data from an HDF5 file, processes it in batches using a provided function,
    and saves the resulting embeddings back into the same HDF5 file.

    Assumes the HDF5 file has a dataset named 'data' containing the strings to process.
    Embeddings are saved under the 'embeddings' group with the specified embedding_name.

    Args:
        h5_file_path: Path to the HDF5 file.
        embedding_name: Name for the new dataset within the 'embeddings' group.
        batch_processing_function: A function that takes a list of strings (batch)
                                   and returns a list of numpy arrays (embeddings).
        batch_size: The number of items to process in each batch.
        model_metadata: Optional dictionary containing metadata about the embedding model
                        (e.g., model name, layer) to save as attributes.
    """
    print(f"--- Starting embedding generation for '{embedding_name}' ---")
    print(f"Processing HDF5 file: {h5_file_path}")
    print(f"Using batch size: {batch_size}")

    try:
        with h5py.File(h5_file_path, 'r+') as f:
            if 'data' not in f:
                raise KeyError("HDF5 file must contain a 'data' dataset with input strings.")
            
            data = f['data']
            num_items = len(data)
            
            if num_items == 0:
                print("Warning: No items found in the 'data' dataset. Skipping embedding generation.")
                return

            # Decode first batch to determine embedding shape/dtype
            first_batch_size = min(batch_size, num_items)
            try:
                # Decode strings assuming UTF-8
                first_batch_data = [data[j].decode('utf-8') for j in range(first_batch_size)]
            except Exception as e:
                print(f"Error decoding first batch data from HDF5: {e}")
                print("Ensure data in HDF5 is stored correctly (e.g., as UTF-8 strings).")
                raise
            
            print("Processing first batch to determine embedding dimensions...")
            try:
                first_batch_embeddings = batch_processing_function(first_batch_data)
            except Exception as e:
                print(f"Error executing batch_processing_function on the first batch: {e}")
                raise

            if not first_batch_embeddings or not isinstance(first_batch_embeddings, list) or len(first_batch_embeddings) != first_batch_size:
                raise ValueError(
                    f"batch_processing_function did not return a valid list of embeddings "
                    f"for the first batch. Expected list of size {first_batch_size}, got: {first_batch_embeddings}"
                )

            embedding_shape = first_batch_embeddings[0].shape
            embedding_dtype = first_batch_embeddings[0].dtype
            print(f"Determined embedding shape: {embedding_shape}, dtype: {embedding_dtype}")
            
            # Create/Recreate the embeddings dataset
            embeddings_group = f.require_group('embeddings')
            if embedding_name in embeddings_group:
                print(f"Warning: Dataset 'embeddings/{embedding_name}' already exists. Deleting and recreating.")
                del embeddings_group[embedding_name]
            
            embeddings_dataset = embeddings_group.create_dataset(
                embedding_name, 
                shape=(num_items, *embedding_shape),
                dtype=embedding_dtype,
                chunks=(min(batch_size, num_items), *embedding_shape) # Set chunking for potential performance benefits
            )
            print(f"Created HDF5 dataset 'embeddings/{embedding_name}' with shape {(num_items, *embedding_shape)}")
            
            # Save the first batch
            for j, embedding in enumerate(first_batch_embeddings):
                embeddings_dataset[j] = embedding
            
            # Calculate number of remaining batches for progress bar
            num_remaining_items = num_items - first_batch_size
            num_batches = math.ceil(num_remaining_items / batch_size)
            
            print(f"Processing remaining {num_remaining_items} items in {num_batches} batches...")
            
            # Use tqdm for progress, starting after the first batch
            with tqdm(total=num_remaining_items, desc=f"Generating {embedding_name} embeddings") as pbar:
                # Process remaining batches
                for i in range(num_batches):
                    batch_start = first_batch_size + i * batch_size
                    batch_end = min(batch_start + batch_size, num_items)
                    actual_batch_size = batch_end - batch_start

                    if actual_batch_size <= 0: # Should not happen with ceil logic, but good practice
                        continue

                    try:
                        batch_data_decoded = [data[j].decode('utf-8') for j in range(batch_start, batch_end)]
                    except Exception as e:
                        print(f"Error decoding batch data (indices {batch_start}-{batch_end}) from HDF5: {e}")
                        # Decide how to handle: skip batch, raise error? Let's raise for now.
                        raise

                    # Process batch
                    try:
                        batch_embeddings = batch_processing_function(batch_data_decoded)
                    except Exception as e:
                        print(f"Error executing batch_processing_function on batch {i+1} (indices {batch_start}-{batch_end}): {e}")
                        # Decide how to handle: skip batch, raise error? Let's raise for now.
                        raise
                    
                    if not batch_embeddings or not isinstance(batch_embeddings, list) or len(batch_embeddings) != actual_batch_size:
                         raise ValueError(
                            f"batch_processing_function did not return a valid list of embeddings "
                            f"for batch {i+1}. Expected list of size {actual_batch_size}, got: {batch_embeddings}"
                        )

                    # Save batch
                    embeddings_dataset[batch_start:batch_end] = np.array(batch_embeddings)
                    
                    # Update progress bar
                    pbar.update(actual_batch_size)
                
            # Add metadata to the dataset
            embeddings_dataset.attrs['name'] = embedding_name
            embeddings_dataset.attrs['shape'] = embedding_shape
            embeddings_dataset.attrs['dtype'] = str(embedding_dtype)
            if model_metadata:
                for key, value in model_metadata.items():
                    # Ensure metadata is serializable by h5py
                    if isinstance(value, (str, int, float, np.number, np.bool_)): 
                        embeddings_dataset.attrs[key] = value
                    else:
                         print(f"Warning: Metadata '{key}' with value '{value}' (type: {type(value)}) is not directly serializable. Skipping.")
            
            print(f"Successfully added '{embedding_name}' embeddings to {h5_file_path}")

    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {h5_file_path}")
        raise
    except KeyError as e:
        print(f"Error: Missing expected key in HDF5 file: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during embedding generation: {e}")
        raise
    finally:
        print(f"--- Finished embedding generation for '{embedding_name}' ---")
