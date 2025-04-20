"""
Script to generate ESP fingerprint for targets.
See: https://github.com/kexinhuang12345/ESPF
"""

import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

import codecs
from subword_nmt.apply_bpe import BPE

from pathlib import Path

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate ESP fingerprints for protein sequences")
    parser.add_argument("--input", required=True, help="TXT file with one sequence per line or a single sequence")
    parser.add_argument("--output", required=True, help="Output numpy file for fingerprints")
    return parser.parse_args()

def main():
    """Main function to handle file I/O and call the fingerprinting functions"""
    args = parse_args()
    
    # Check if input is a file or a single sequence
    if os.path.isfile(args.input):
        with open(args.input, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
    else:
        # Treat input as a single sequence string
        sequences = [args.input]
    
    print(f"Processing {len(sequences)} sequences...")
    
    # Generate fingerprints
    fingerprints = [get_target_fingerprint(seq) for seq in tqdm(sequences, desc="Generating ESP fingerprints")]
    
    # Convert to numpy array
    fingerprints_array = np.array(fingerprints)
    
    # Save fingerprints
    np.save(args.output, fingerprints_array)
    print(f"Saved fingerprints with shape {fingerprints_array.shape} to {args.output}")

if __name__ == "__main__":
    main()