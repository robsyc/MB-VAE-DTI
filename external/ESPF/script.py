"""
Script to generate ESP fingerprints for targets.
See: https://github.com/kexinhuang12345/ESPF
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import codecs
from subword_nmt.apply_bpe import BPE
from typing import List

# Add the parent directory to the Python path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import parse_args, add_embeddings_to_hdf5

# Define paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ESPF_DIR = SCRIPT_DIR
CODES_FILE = ESPF_DIR / "codes_protein.txt"
SUBWORD_MAP_FILE = ESPF_DIR / "subword_units_map_protein.csv"

# Define constants for this script
EMBEDDING_NAME = "FP-ESP" # Fingerprint
BATCH_SIZE = 512      # ESPF is fast, can use larger batch size for I/O

# Validate required files exist
if not CODES_FILE.exists():
    raise FileNotFoundError(f"Required file not found: {CODES_FILE}")
if not SUBWORD_MAP_FILE.exists():
    raise FileNotFoundError(f"Required file not found: {SUBWORD_MAP_FILE}")

# Load BPE codes and subword map
print("Loading ESPF BPE codes and subword map...")
bpe_codes_protein = codecs.open(str(CODES_FILE))
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv(str(SUBWORD_MAP_FILE))

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))
print("ESPF resources loaded.")

def get_target_fingerprint(s: str) -> np.ndarray:
    """
    Get the fingerprint of a target from its sequence string.
    
    Args:
        s (str): Sequence string of the target

    Returns:
        np.ndarray: Fingerprint of the target (ESP fingerprint)
    """
    t = pbpe.process_line(s).split()
    # Handle potential empty sequences or sequences that result in no tokens
    if not t:
        # Return a zero vector of the expected length
        return np.zeros(len(idx2word_p), dtype=np.float32) # Use float32 consistent with others
    
    # Use list comprehension with get for robustness against unseen subwords
    indices = [words2idx_p.get(subword) for subword in t]
    # Filter out None values if any subword wasn't found (though unlikely with BPE)
    valid_indices = [idx for idx in indices if idx is not None]

    v = np.zeros(len(idx2word_p), dtype=np.float32) # Use float32 consistent with others
    if valid_indices:
        v[valid_indices] = 1
    return v

def batch_get_target_fingerprint(sequences: List[str]) -> List[np.ndarray]:
    """
    Wrapper to apply get_target_fingerprint to a batch of sequences.
    
    Args:
        sequences: List of sequence strings.

    Returns:
        List of numpy arrays (fingerprints).
    """
    return [get_target_fingerprint(s) for s in sequences]

def main():
    """Main function to parse args and call the HDF5 processing utility."""
    args = parse_args() # Gets only --input
    
    add_embeddings_to_hdf5(
        h5_file_path=args.input,
        embedding_name=EMBEDDING_NAME,
        batch_processing_function=batch_get_target_fingerprint,
        batch_size=BATCH_SIZE
        # No specific model metadata for ESPF
    )

if __name__ == "__main__":
    main()