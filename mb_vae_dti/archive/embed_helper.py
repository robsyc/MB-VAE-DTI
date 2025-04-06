"""
This module contains helper functions for creating feature vectors 
for drug and target entity representations.
"""

import os
import dotenv

import pandas as pd
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from subword_nmt.apply_bpe import BPE
import codecs

from esm.sdk import client
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, List

dotenv.load_dotenv()
ESM_TOKEN = os.getenv("ESM_TOKEN")

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

########################################################################################################
## FINGERPRINTS - helper functions
########################################################################################################

# Drug Morgan fingerprint -> size 2048
# See: https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
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

# Target ESP fingerprint -> size 4170
# See: https://github.com/kexinhuang12345/ESPF
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
        np.ndarray: Fingerprint of the target (ESP fingerprint)
    """
    t = pbpe.process_line(s).split()
    i = [words2idx_p[i] for i in t]
    v = np.zeros(len(idx2word_p), )
    v[i] = 1
    return v


########################################################################################################
## FOUNDATION MODELS - helper functions
########################################################################################################

# Target ESM-C 6B (using API) -> size 2560 (80 layers)
# See: https://github.com/evolutionaryscale/esm
# Alternatively, we could use esmc-600m-2024-12 (smaller open source model with 36 layers)
model = client(
    model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=ESM_TOKEN
)

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_hidden_states=True, ith_hidden_layer=80 # alternative: layer 55 ?
)

def esm_embed_sequence(
    sequence: str, model: ESM3InferenceClient = model
) -> np.ndarray:
    """
    Embed a protein sequence using the ESM-C (6B with 80 layers) model.
    Returns the mean of the hidden states over the residues.
    """
    # TODO: add caching
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    emb = output.hidden_states.mean(dim=-2) # average over residues
    return emb.squeeze().numpy()

def esm_batch_embed(
    inputs: Sequence[ProteinType], model: ESM3InferenceClient = model
) -> List[np.ndarray]:
    """Forge supports auto-batching. So batch_embed() is as simple as running a collection
    of embed calls in parallel using asyncio.
    """
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(esm_embed_sequence, protein, model) for protein in inputs
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(ESMProteinError(500, str(e)))
    return results

# Target NT (v2-500m-multi-species) -> size 1024 (29 layers)
# See: https://github.com/instadeepai/nucleotide-transformer