import pandas as pd
from pathlib import Path
import subprocess
from typing import Literal
import logging
import h5py
import numpy as np

# Configure logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embedding')

# Define paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

EXTERNAL_DIR = PROJECT_ROOT / "external"
TEMP_DIR = EXTERNAL_DIR / "temp"

# Ensure directories exists

TEMP_DIR.mkdir(exist_ok=True, parents=True)
assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist"
assert PROCESSED_DIR.exists(), f"Processed directory {PROCESSED_DIR} does not exist"
assert EXTERNAL_DIR.exists(), f"External directory {EXTERNAL_DIR} does not exist"

# Helper functions

def save_representations_to_h5(
    df: pd.DataFrame,
    representation_column: Literal["Drug_SMILES", "Target_AA", "Target_DNA"],
    output_file_name: str
) -> Path:
    """
    Save drug or target string representations (to be used for embedding) to an HDF5 file.
    
    Args:
        df: DataFrame containing the data
        representation_column: Column name for the string representation
        output_file_name: Name for the output HDF5 file
    
    Returns:
        Path to the created HDF5 file
    """
    if not output_file_name.endswith(".hdf5"):
        if "." in output_file_name:
            output_file_name = output_file_name.split(".")[0]
        output_file_name = output_file_name + ".hdf5"

    entity_id = "Drug_ID" if representation_column == "Drug_SMILES" else "Target_ID"
    df = df.sort_values(by=entity_id)
    
    # Get unique entity IDs and representations
    unique_df = df[[entity_id, representation_column]].drop_duplicates()
    unique_ids = unique_df[entity_id].values
    unique_representations = unique_df[representation_column].values

    assert len(unique_ids) == len(unique_representations), "Number of unique IDs and representations must match"
    
    # Create output path
    output_path = TEMP_DIR / output_file_name
    
    # Save to HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Store IDs as string dataset (h5py requires bytes for variable length strings)
        id_dataset = f.create_dataset('ids', (len(unique_ids),), dtype=h5py.special_dtype(vlen=str))
        id_dataset[:] = unique_ids
        
        # Store representations as string dataset
        repr_dataset = f.create_dataset('data', (len(unique_representations),), dtype=h5py.special_dtype(vlen=str))
        repr_dataset[:] = unique_representations
        
        # Create empty group for embeddings that will be populated later
        f.create_group('embeddings')
        
        # Add metadata
        f.attrs['representation_type'] = representation_column.split('_')[-1]
        f.attrs['entity_type'] = 'drug' if representation_column == "Drug_SMILES" else 'target'
    
    logger.info(f"Created H5 file with {len(unique_ids)} unique {entity_id}s at {output_path}")
    return output_path

def run_embedding_script(
    hdf5_file_name: str,
    external_repo_name: Literal[
        "MorganFP", "biomed-multi-view",      # drugs
        "ESPF", "ESM", "nucleotide-transformer" # targets
    ]
) -> None:
    """
    Run the embedding script to add embeddings to an existing HDF5 file.
    
    Args:
        hdf5_file_name: Name of the HDF5 file containing entities to embed
        external_repo_name: Name of the external repository for the embedding method
    """
    hdf5_file_path = (TEMP_DIR / hdf5_file_name).resolve()
    script_path = (EXTERNAL_DIR / "run_embeddings.sh").resolve()
    
    cmd = [
        str(script_path),
        external_repo_name,
        str(hdf5_file_path)
    ]
    
    logger.info(f"Running embedding script with command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running embedding script: {e}")
        raise e
