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

# Add the parent directory to the Python path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import parse_args, process_embeddings_in_batches

# Drug Morgan fingerprint -> size 2048
def get_drug_fingerprint(s: str) -> np.ndarray:
    """
    Get the fingerprint of a drug from its SMILES string.
    
    Args:
        s (str): SMILES string of the drug
        
    Returns:
        np.ndarray: Fingerprint of the drug (Morgan fingerprint with radius 2 and size 2048)
    """
    molecule = Chem.MolFromSmiles(s)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array(mfpgen.GetFingerprint(molecule))

def main():
    """Main function to handle file I/O and call the fingerprinting functions"""
    args = parse_args()
    
    # Process the HDF5 file in batches
    process_embeddings_in_batches(
        h5_file_path=args.input,
        embedding_function=get_drug_fingerprint,
        embedding_name="FP",
        batch_size=128
    )

if __name__ == "__main__":
    main()