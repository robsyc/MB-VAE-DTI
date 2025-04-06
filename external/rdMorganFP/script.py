"""
Script to generate Morgan fingerprints for drugs using RDKit.
See: https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os
import argparse
from tqdm import tqdm

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate Morgan fingerprints for drug SMILES strings")
    parser.add_argument("--input", required=True, help="TXT file with one SMILES string per line or a single SMILES string")
    parser.add_argument("--output", required=True, help="Output numpy file for fingerprints")
    return parser.parse_args()

def main():
    """Main function to handle file I/O and call the fingerprinting functions"""
    args = parse_args()
    
    # Check if input is a file or a single sequence
    if os.path.isfile(args.input):
        with open(args.input, 'r') as f:
            smiles_strings = [line.strip() for line in f if line.strip()]
    else:
        # Treat input as a single SMILES string
        smiles_strings = [args.input]
    
    print(f"Processing {len(smiles_strings)} SMILES strings...")
    
    # Generate fingerprints
    fingerprints = [get_drug_fingerprint(smiles) for smiles in tqdm(smiles_strings, desc="Generating Morgan fingerprints")]
    
    # Convert to numpy array
    fingerprints_array = np.array(fingerprints)
    
    # Save fingerprints
    np.save(args.output, fingerprints_array)
    print(f"Saved fingerprints with shape {fingerprints_array.shape} to {args.output}")
    print(f"Each fingerprint has {fingerprints_array.shape[1]} features")

if __name__ == "__main__":
    main()