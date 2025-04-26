"""
Script to generate Morgan fingerprints for drugs using RDKit.
See: https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os
import sys
from pathlib import Path
from typing import List

# Add the parent directory to the Python path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Import necessary functions from utils
from utils import parse_args, add_embeddings_to_hdf5

# --- Constants ---
FP_RADIUS = 2
FP_SIZE = 2048
EMBEDDING_NAME = f"FP-Morgan"
BATCH_SIZE = 512 # RDKit is generally fast

# --- Initialize Fingerprint Generator ---
# Create the generator once globally for efficiency
print(f"Initializing Morgan fingerprint generator (Radius={FP_RADIUS}, Size={FP_SIZE})...")
MFP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
print("Generator initialized.")

# --- Fingerprint Function ---
def get_drug_fingerprint(smiles: str) -> np.ndarray:
    """
    Get the Morgan fingerprint of a drug from its SMILES string.
    Handles invalid SMILES by returning a zero vector.
    
    Args:
        smiles (str): SMILES string of the drug
        
    Returns:
        np.ndarray: Fingerprint of the drug (Morgan fingerprint as float32)
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        # Handle invalid SMILES
        print(f"Warning: Could not parse SMILES: '{smiles}'. Returning zero vector.")
        # Use float32 for consistency
        return np.zeros(FP_SIZE, dtype=np.float32) 
        
    # Use the globally defined generator
    fp = MFP_GENERATOR.GetFingerprint(molecule)
    # Convert RDKit ExplicitBitVect to numpy array
    # Use float32 for consistency
    arr = np.zeros(FP_SIZE, dtype=np.float32)
    # Set on bits to 1.0
    arr[list(fp.GetOnBits())] = 1.0 
    return arr

# --- Batch Processing Wrapper ---
def batch_get_drug_fingerprint(smiles_list: List[str]) -> List[np.ndarray]:
    """
    Wrapper to apply get_drug_fingerprint to a batch of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings.

    Returns:
        List of numpy arrays (fingerprints).
    """
    return [get_drug_fingerprint(s) for s in smiles_list]

# --- Main Execution ---
def main():
    """Main function to parse args and call the HDF5 processing utility."""
    args = parse_args() # Gets only --input
    
    add_embeddings_to_hdf5(
        h5_file_path=args.input,
        embedding_name=EMBEDDING_NAME,
        batch_processing_function=batch_get_drug_fingerprint,
        batch_size=BATCH_SIZE
        # No specific model metadata for Morgan FP
    )

if __name__ == "__main__":
    main()