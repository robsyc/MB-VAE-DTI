"""
This module contains helper functions for creating feature vectors 
for drug and target entity representations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from subword_nmt.apply_bpe import BPE
import codecs

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = Path("external")
ESPF_DIR = EXTERNAL_DIR / "ESPF"

# Assertions
assert ESPF_DIR.exists(), """ESPF directory does not exist. Cannot create protein fingerprints.
Please download the files `codes_protein.txt` and `subword_units_map_protein.csv` 
from https://github.com/kexinhuang12345/ESPF
and place them in the `external/ESPF` directory."""

## FINGERPRINTS - helper functions

def get_drug_fingerprint(s: str) -> np.ndarray:
    """
    Get the fingerprint of a drug from its SMILES string.
    
    Args:
        s (str): SMILES string of the drug
        
    Returns:
        np.ndarray: Fingerprint of the drug (Morgan fingerprint with radius 2 and size 2048)
    """
    # See: https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    molecule = Chem.MolFromSmiles(s)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array(mfpgen.GetFingerprint(molecule))

vocab_path = ESPF_DIR / 'codes_protein.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv(ESPF_DIR / 'subword_units_map_protein.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

def get_target_fingerprint(s: str) -> np.ndarray:
    """
    Get the fingerprint of a target from its sequence string.
    
    Args:
        s (str): Sequence string of the target

    Returns:
        np.ndarray: Fingerprint of the target (ESPF fingerprint)
    """
    # See: https://github.com/kexinhuang12345/ESPF
    t = pbpe.process_line(s).split()
    i = [words2idx_p[i] for i in t]
    v = np.zeros(len(idx2word_p), )
    v[i] = 1
    return v


## FOUNDATION MODELS - helper functions
# ...
