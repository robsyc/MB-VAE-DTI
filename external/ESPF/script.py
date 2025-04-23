"""
Script to generate ESP fingerprint for targets.
See: https://github.com/kexinhuang12345/ESPF
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import codecs
from subword_nmt.apply_bpe import BPE

# Add the parent directory to the Python path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import parse_args, process_embeddings_in_batches

# Define paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ESPF_DIR = SCRIPT_DIR
CODES_FILE = ESPF_DIR / "codes_protein.txt"
SUBWORD_MAP_FILE = ESPF_DIR / "subword_units_map_protein.csv"

# Validate required files exist
if not CODES_FILE.exists():
    raise FileNotFoundError(f"Required file not found: {CODES_FILE}")
if not SUBWORD_MAP_FILE.exists():
    raise FileNotFoundError(f"Required file not found: {SUBWORD_MAP_FILE}")

# Load BPE codes and subword map
bpe_codes_protein = codecs.open(str(CODES_FILE))
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv(str(SUBWORD_MAP_FILE))

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

def get_target_fingerprint(s: str) -> np.ndarray:
    """
    Get the fingerprint of a target from its sequence string.
    
    Args:
        s (str): Sequence string of the target

    Returns:
        np.ndarray: Fingerprint of the target (ESP fingerprint)
    """
    t = pbpe.process_line(s).split()
    i = [words2idx_p[i] for i in t]
    v = np.zeros(len(idx2word_p), )
    v[i] = 1
    return v

def main():
    """Main function to handle file I/O and call the fingerprinting functions"""
    args = parse_args()
    
    # Process the HDF5 file in batches
    process_embeddings_in_batches(
        h5_file_path=args.input,
        embedding_function=get_target_fingerprint,
        embedding_name="FP",
        batch_size=128
    )

if __name__ == "__main__":
    main()