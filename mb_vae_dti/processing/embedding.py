"""
This module provides functionality for adding embeddings to existing h5torch files.
"""

from typing import Callable, List, Optional, Union, Literal, Dict, Any, Tuple
from pathlib import Path
import h5torch
import numpy as np
import pandas as pd
import tqdm
import subprocess
import tempfile
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embedding')

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

EXTERNAL_DIR = Path("external")
TEMP_DIR = EXTERNAL_DIR / "temp"

# Ensure directories exists
TEMP_DIR.mkdir(exist_ok=True, parents=True)
assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist"
assert PROCESSED_DIR.exists(), f"Processed directory {PROCESSED_DIR} does not exist"

def add_feature_vector_to_h5torch(
    h5torch_file_name: str,
    entity_axis: Literal[0, 1],
    feature_name: str,
    feature_vector: np.ndarray,
    dtype_save: str = "float32",
    dtype_load: str = "float32",
    overwrite: bool = False
):
    """
    Generic function to add feature vectors to DTI h5torch file.
    
    Args:
        h5torch_file_name (str): Path to the h5torch file (e.g. `data.h5torch`)
        entity_axis (Literal[0, 1]): Axis to add the feature to
        feature_name (str): Name for the new feature
        feature_vector (np.ndarray): Vector to add to the file
        dtype_save (str): Data type to save as
        dtype_load (str): Data type to load as
        overwrite (bool): Whether to overwrite existing feature if it exists
    """
    # Open the file in append mode
    f = h5torch.File(PROCESSED_DIR / h5torch_file_name, "a")
    
    # Check if feature exists and should be overwritten
    feature_path = f"{entity_axis}/{feature_name}"
    if feature_path in f:
        if overwrite:
            logger.info(f"Found existing {feature_name}, deleting it")
            del f[feature_path]
        else:
            logger.info(f"Feature {feature_name} already exists and overwrite=False. Skipping.")
            f.close()
            return
    
    # Determine output shape num_entities x num_features
    logger.info(f"Feature {feature_name} shape: {feature_vector.shape}")

    # Register the processed features
    f.register(
        feature_vector, 
        mode="N-D", 
        axis=entity_axis, 
        name=feature_name, 
        dtype_save=dtype_save, 
        dtype_load=dtype_load
    )
    
    # Close the file
    f.close()
    logger.info(f"Successfully added {feature_name} to {h5torch_file_name}")

def generate_embeddings(
    repo_name: str,
    sequences: List[str],
    use_batch: bool = False,  # Default to False for safety
    temp_prefix: Optional[str] = None,
    debug: bool = False
) -> np.ndarray:
    """
    Generate embeddings for a list of sequences using the specified external repository.
    
    Args:
        repo_name (str): Name of the repository in the external directory to use
        sequences (List[str]): List of sequences (SMILES, AA, DNA) to embed
        use_batch (bool): Whether to use batch processing mode
        temp_prefix (Optional[str]): Prefix for temporary files (defaults to repo_name)
        debug (bool): Enable debug mode to keep temporary files
    
    Returns:
        np.ndarray: Embeddings array with shape (len(sequences), embedding_dim)
    """
    if temp_prefix is None:
        temp_prefix = repo_name.lower()
    
    # Create timestamp for unique filenames
    timestamp = int(time.time())
    
    # Create temporary input file with sequences
    input_file = TEMP_DIR / f"{temp_prefix}_input_{timestamp}.txt"
    output_file = TEMP_DIR / f"{temp_prefix}_output_{timestamp}.npy"
    
    # Write sequences to input file
    with open(input_file, 'w') as f:
        for seq in sequences:
            f.write(f"{seq}\n")
    
    logger.info(f"Wrote {len(sequences)} sequences to {input_file}")
    
    # Get absolute paths for command
    abs_input_file = input_file.resolve()
    abs_output_file = output_file.resolve()
    abs_script_path = (EXTERNAL_DIR / "run_embeddings.sh").resolve()
    
    # Ensure run_embeddings.sh is executable
    if not os.access(abs_script_path, os.X_OK):
        logger.warning(f"Making script executable: {abs_script_path}")
        os.chmod(abs_script_path, os.stat(abs_script_path).st_mode | 0o100)
    
    # Build command
    cmd = [
        str(abs_script_path),
        repo_name,
        str(abs_input_file),
        str(abs_output_file)
    ]
    
    if use_batch:
        cmd.append("--batch")
        logger.info(f"Using batch mode for {repo_name}")
    
    logger.info(f"Running embedding command: {' '.join(cmd)}")
    try:
        # Execute the command
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(result.stdout)
        
        # Load embeddings
        if os.path.exists(output_file):
            embeddings = np.load(output_file)
            logger.info(f"Loaded embeddings with shape {embeddings.shape}")
            return embeddings
        else:
            raise FileNotFoundError(f"Output file {output_file} not found after running embedding script")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running embedding script: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise
    finally:
        # Clean up temporary files
        if not debug:
            if input_file.exists():
                input_file.unlink()
            if output_file.exists():
                output_file.unlink()
        else:
            logger.info(f"Debug mode: Keeping temporary files at {input_file} and {output_file}")

def add_embeddings_to_h5torch(
    h5torch_file_name: str,
    repo_name: str,
    entity_axis: Literal[0, 1],
    entity_representation: str,
    feature_name: str,
    use_batch: bool = False, # Default to False for safety
    dtype_save: str = "float32",
    dtype_load: str = "float32",
    overwrite: bool = False,
    debug: bool = False
):
    """
    Generate embeddings for a list of sequences and add them to an h5torch file.
    
    Args:
        h5torch_file_name (str): Path to the h5torch file
        repo_name (str): Name of the repository to use (e.g., 'ESM', 'ESPF')
        entity_axis (Literal[0, 1]): DTI axis of interest (0 for drugs, 1 for targets)
        entity_representation (str): Name of the input representation (e.g., 'Drug_SMILES', 'Target_AA', 'Target_DNA')
        feature_name (str): Name for the new feature
        use_batch (bool): Whether to use batch mode for embedding
        dtype_save (str): Data type to save embeddings as
        dtype_load (str): Data type to load embeddings as
        overwrite (bool): Whether to overwrite existing feature
        debug (bool): Enable debug mode to keep temporary files
    """
    # Open the file to read the sequences
    file_path = PROCESSED_DIR / h5torch_file_name
    f = h5torch.File(file_path, "a")
    
    # Read the sequences from the specified representation dataset
    raw_sequences = f[f"{entity_axis}/{entity_representation}"][:]
    sequences = [s.decode("utf-8") for s in raw_sequences]
    
    logger.info(f"Retrieved {len(sequences)} sequences from {entity_representation}")
    logger.info(f"Generating embeddings using {repo_name}...")
    
    # Generate embeddings
    embeddings = generate_embeddings(
        repo_name=repo_name,
        sequences=sequences,
        use_batch=use_batch,
        debug=debug
    )
    
    # Add embeddings to h5torch file
    add_feature_vector_to_h5torch(
        h5torch_file_name=h5torch_file_name,
        entity_axis=entity_axis,
        feature_name=feature_name,
        feature_vector=embeddings,
        dtype_save=dtype_save,
        dtype_load=dtype_load,
        overwrite=overwrite
    )
    
    # Close the file
    f.close()
    
    return embeddings

