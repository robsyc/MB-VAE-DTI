import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from subword_nmt.apply_bpe import BPE
import codecs

import h5torch
from typing import Literal


DATASET_PATH = "./data/dataset/"
assert os.path.exists(DATASET_PATH), "Dataset directory does not exist."
assert any([f.endswith(".h5t") for f in os.listdir(DATASET_PATH)]), "No .h5t files found in dataset directory."


## FINGERPRINTS

def get_drug_fingerprint(s: str) -> np.ndarray:
    # See: https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    molecule = Chem.MolFromSmiles(s)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array(mfpgen.GetFingerprint(molecule))

espf_folder = "./data/ESPF"
vocab_path = espf_folder + '/codes_protein.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv(espf_folder + '/subword_units_map_protein.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

def get_target_fingerprint(s: str) -> np.ndarray:
    # See: https://github.com/kexinhuang12345/ESPF
    t = pbpe.process_line(s).split()
    i = [words2idx_p[i] for i in t]
    v = np.zeros(len(idx2word_p), )
    v[i] = 1
    return v


## EMBEDDINGS


####################################################################################################################

def add_embeddings(name: Literal["BindingDB_Kd", "DAVIS", "KIBA"]) -> None:
    """
    Adds fingerprint and embedding columns to the specified dataset's existing h5torch file.
    """
    f = h5torch.File(DATASET_PATH + name + ".h5t", "a")
    drug_smiles = f["0/Drug_SMILES"][:]
    target_AA = f["1/Target_seq"][:]
    target_DNA = f["1/Target_seq_DNA"][:]

    drug_fingerprints = np.empty((len(drug_smiles), 2048), dtype=np.float32)
    for i, s in enumerate(drug_smiles):
        s = s.decode("utf-8")
        fp = get_drug_fingerprint(s)
        drug_fingerprints[i] = fp
    f.register(drug_fingerprints, mode="N-D", axis=0, name="Drug_fp", dtype_save="float32", dtype_load="float32")
    f.save()

    drug_emb_graph = np.empty((len(drug_smiles), 512), dtype=np.float32)
    drug_emb_image = np.empty((len(drug_smiles), 512), dtype=np.float32)
    drug_emb_text = np.empty((len(drug_smiles), 768), dtype=np.float32)

    target_fingerprints = np.empty((len(target_AA), 4170), dtype=np.float32)
    target_emb_T5 = np.empty((len(target_AA), 1024), dtype=np.float32)
    # target_emb_ESM = np.empty((len(target_AA), 1280), dtype=np.float32)
    # target_emb_DNA = np.empty((len(target_DNA), 1280), dtype=np.float32)

    # f.register(drug_embeddings_1, mode="N-D", axis=0, name="drug_embeddings_1", dtype_save="float32", dtype_load="float32")