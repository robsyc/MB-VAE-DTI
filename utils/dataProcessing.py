import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from tdc.multi_pred import DTI
from typing import Literal
import h5torch
from Bio import Entrez, SeqIO


np.random.seed(42)

# Filters (partially adapted from https://github.com/PatWalters/rd_filters)
MAX_N_HEAVY_ATOMS = 60
MAX_MOL_WEIGHT = 600
# BOUNDS_LogP = [-5, 5]
MAX_SEQ_LEN = 1200

Entrez.email = "robbe.claeys@ugent.be"
IMG_PATH = "./data/logs/plots/"
DATASET_PATH = "./data/dataset/"
os.makedirs(IMG_PATH, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)


def load_df(
        name: Literal["BindingDB_Kd", "DAVIS", "KIBA"],
        use_filters: bool = True,
        seed = 42,
    ) -> pd.DataFrame:
    """
    Loads the given DTI dataset from https://tdcommons.ai/multi_pred_tasks/dti and applies filters.
    
    Args:
        name: Name of the dataset to load (BindingDB_Kd, DAVIS, KIBA)
        use_filters: Whether to apply filters to the dataset
        to_log: Whether to convert the Y values to log scale
        seed: Seed for the random split
    """
    print(f"Loading {name} dataset...")
    data = DTI(name = name)
    if name == "DAVIS" or name == "BindingDB_Kd":
        data.convert_to_log(form = 'binding')
    
    # Get random (setting A at 7:2:1) and cold split (setting B at 7.5:1.5:1)
    split = data.get_split(method = 'random', seed = seed)
    train, valid, test = split['train'], split['valid'], split['test']
    df_rand = pd.concat([train, valid, test], axis=0)
    df_rand['split_rand'] = ["train"] * len(train) + ["valid"] * len(valid) + ["test"] * len(test)

    split = data.get_split('cold_split', column_name='Drug')
    train, valid, test = split['train'], split['valid'], split['test']
    df_cold = pd.concat([train, valid, test], axis=0)
    df_cold['split_cold'] = ["train"] * len(train) + ["valid"] * len(valid) + ["test"] * len(test)

    df = pd.merge(df_rand, df_cold, on=['Drug_ID', 'Drug', 'Target_ID', 'Target', 'Y'], how='inner')
    unique_drugs = df.Drug.unique()
    unique_targets = df.Target.unique()

    # Calculate metrics and filter the df
    unique_MW = {drug: Descriptors.MolWt(Chem.MolFromSmiles(drug)) for drug in unique_drugs}
    unique_LogP = {drug: Descriptors.MolLogP(Chem.MolFromSmiles(drug)) for drug in unique_drugs}
    unique_HeavyAtoms = {drug: Descriptors.HeavyAtomCount(Chem.MolFromSmiles(drug)) for drug in unique_drugs}
    unique_TagetLength = {target: len(target) for target in unique_targets}
    
    df['MW'] = df['Drug'].map(unique_MW)
    df['LogP'] = df['Drug'].map(unique_LogP)
    df['HeavyAtoms'] = df['Drug'].map(unique_HeavyAtoms)
    df['TargetLength'] = df['Target'].map(unique_TagetLength)
    df['standardized'] = (df['Y'] - df['Y'].min()) / (df['Y'].max() - df['Y'].min())
    if name == "DAVIS" or name == "BindingDB_Kd":
        df['Y_binary'] = (df['Y'] >= 7).astype(int)
    elif name == "KIBA":
        df['Y_binary'] = (df['Y'] >= 12.1).astype(int)

    if use_filters:
        df = df[df['HeavyAtoms'] <= MAX_N_HEAVY_ATOMS]
        df = df[df['MW'] <= MAX_MOL_WEIGHT]
        # df = df[(df['LogP'] >= BOUNDS_LogP[0]) & (df['LogP'] <= BOUNDS_LogP[1])]
        df = df[df['TargetLength'] <= MAX_SEQ_LEN]
    
    return df

def plot_Y_distribution(df: pd.DataFrame, threshold: float, title: str) -> None:
    """
    Plots the Y value distribution with a vertical line at the threshold.
    Caps extreme counts to improve visualization.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Y'], bins=12)
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.title(title)
    plt.xlabel('Y')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

def explore_df(
        df: pd.DataFrame, 
        name: Literal["BindingDB_Kd", "DAVIS", "KIBA"]
    ) -> None:
    """
    Explores the given DTI dataset by plotting drug, target and interaction properties.

    Args:
        df: Dataframe containing the DTI data
        name: Name of the dataset (BindingDB_Kd, DAVIS, KIBA)
    """
    if name == "DAVIS" or name == "BindingDB_Kd":
        y_lim = [0, 100, 8000, 10000]
        binding_threshold = 7
        xlabels = ['Binding Affinity (pKd)', 'Standardized Binding Affinity (pKd)']
    elif name == "KIBA":
        y_lim = [0, 5000, 11000, 11500]
        binding_threshold = 12.1
        xlabels = ['Binding Affinity (KIBA score)', 'Standardized Binding Affinity (KIBA score)']

    print(f"Exploring {name} dataset...")
    # Drug properties
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    sns.histplot(df['MW'], ax=axes[0])
    axes[0].set_title('Molecular Weight Distribution')
    axes[0].set_xlabel('MW')

    sns.histplot(df['LogP'], ax=axes[1])
    axes[1].set_title('LogP Distribution')
    axes[1].set_xlabel('LogP')

    sns.histplot(df['HeavyAtoms'], ax=axes[2])
    axes[2].set_title('Heavy Atom Count Distribution')
    axes[2].set_xlabel('Number of Heavy Atoms')

    plt.tight_layout()
    plt.show()

    # Target properties
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['TargetLength'], ax=ax)
    ax.set_title('Target Length Distribution')
    ax.set_xlabel('Target Length')

    plt.tight_layout()
    plt.show()

    # Interaction properties
    if name in ["DAVIS", "BindingDB_Kd"]:
        threshold = 7
    elif name == "KIBA":
        threshold = 12.1
    else:
        threshold = df['Y'].median()
    plot_Y_distribution(df, threshold, f'{name} Y Value Distribution')
    plt.figure(figsize=(6, 5))
    sns.countplot(x='Y_binary', data=df)
    plt.title('Binary Interaction Distribution')
    plt.xlabel('Y_binary')
    plt.tight_layout()
    plt.show()

hand_corrected_IDs = {
    # Davis
    "ABL1p": "ABL1",
    "AMPK-alpha1": "AMPK alpha 1",
    "AMPK-alpha2": "AMPK alpha 2",
    "FGFR3(G697C)": "FGFR3",
    "IKK-epsilon": "IKK-E",
    "JAK1(JH2domain-pseudokinase)": "JAK1",
    "KIT(V559D-V654A)": "KIT",
    "p38-alpha": "p38alpha",
    "p38-beta": "p38beta",
    "PFCDPK1(Pfalciparum)": "PF3D7_0217500",    # not human
    "PFPK5(Pfalciparum)": "An12g05120",         # not human
    "PKAC-alpha": "PKACA",
    "RET(V804M)": "RET",
    "RSK1(KinDom.2-C-terminal)": "RSK1",
    "RSK4(KinDom.2-C-terminal)": "RSK4",
    "TYK2(JH2domain-pseudokinase)": "TYK2",
    "CDK4-cyclinD3": "CCND3",
    "JAK2(JH1domain-catalytic)": "JAK2",
    "JAK3(JH1domain-catalytic)": "JAK3",
    "p38-gamma": "p38gamma",
    "PKNB(Mtuberculosis)": "PKNB",              # not human
    "RPS6KA5(KinDom.2-C-terminal)": "RPS6KA5",
    "RSK2(KinDom.1-N-terminal)": "RSK2",
    "RSK3(KinDom.2-C-terminal)": "RSK3",
    "FLT3(R834Q)": "FLT3",
    "PIK3CA(Q546K)": "PIK3CA",
    "RPS6KA4(KinDom.2-C-terminal)": "RPS6KA4",
    "BRAF(V600E)": "BRAF",
    "PKAC-beta": "PKACB",
    # KIBA
}

def fetch_dna_sequence(target_id):
    """
    Fetches coding DNA sequence of a protein target ID from NCBI.
    """
    if target_id in hand_corrected_IDs.keys():
        target_id = hand_corrected_IDs[target_id]

    query = f"{target_id}[Gene Symbol] AND Homo sapiens[Organism]"
    fallback_query = f"{target_id}[Gene Symbol]"
    for i, q in enumerate([query, fallback_query]):
        handle = Entrez.esearch(db="nucleotide", term=q, retmax=10)
        record = Entrez.read(handle)
        handle.close()
        if record["IdList"]:
            break
        print("Not found in Homo sapiens") if i == 0 else None
    if record["IdList"]:
        for seq_id in record["IdList"]:
            handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="gb", retmode="text")
            seq_record = SeqIO.read(handle, "genbank")
            handle.close()
            for feature in seq_record.features:
                if feature.type == 'CDS':
                    try:
                        return str(feature.location.extract(seq_record).seq)
                    except:
                        continue
        
        print(f"No CDS found for target {target_id}")
        return None
        
    print(f"No DNA sequences found for target {target_id}")
    return None

def generate_h5torch(
        df: pd.DataFrame, 
        name: Literal["BindingDB_Kd", "DAVIS", "KIBA"]
    ) -> None:
    """
    Generates and saves drugs x targets .h5t file for the given DTI dataframe.
    See: https://h5torch.readthedocs.io/en/latest/index.html
    Consists of:
        - Drugs x targets interaction matrix (y values) in COO format
        - Random & Cold split metadata for each y (train, valid, test)
        - Drug and target properties (ID, string representation)
    
    Args:
        df: Dataframe containing the DTI data
        name: Name of the dataset (BindingDB_Kd, DAVIS, KIBA)
    """
    # Get unique IDs and map them to integer indices
    unique_drugs = df["Drug_ID"].unique()
    unique_targets = df["Target_ID"].unique()

    drug_id2int = {drug: i for i, drug in enumerate(unique_drugs)}          # e.g. 0 - 63
    target_id2int = {target: i for i, target in enumerate(unique_targets)}

    df["Drug_index"] = df["Drug_ID"].map(drug_id2int)
    df["Target_index"] = df["Target_ID"].map(target_id2int)

    # Generate necessary COO matrix data
    coo_matrix_indices = df[["Drug_index", "Target_index"]].values.T
    coo_matrix_values = df["Y"].values.astype(np.float32)
    coo_matrix_shape = (df["Drug_index"].max() + 1, df["Target_index"].max() + 1)

    # Gather IDs and string representations in order of the integer indices
    drug_int2id = {v : k for k, v in drug_id2int.items()}
    drug_int2smiles = {k : v for k, v in zip(df["Drug_index"], df["Drug"])}
    drug_id = np.array([drug_int2id[i] for i in range(df["Drug_index"].max() + 1)])
    drug_smiles = np.array([drug_int2smiles[i] for i in range(df["Drug_index"].max() + 1)])

    target_int2id = {v : k for k, v in target_id2int.items()}
    target_int2seq = {k : v for k, v in zip(df["Target_index"], df["Target"])}
    target_id = np.array([target_int2id[i] for i in range(df["Target_index"].max() + 1)])
    target_seq = np.array([target_int2seq[i] for i in range(df["Target_index"].max() + 1)])

    # Gather DNA sequences for targets
    fasta_file = DATASET_PATH + name + "_target_seq_DNA.fasta"
    if not os.path.exists(fasta_file):
        with open(fasta_file, "w") as f:
            for i, s in enumerate(target_id):
                print(i, s)
                DNA_seq = fetch_dna_sequence(s)
                try:
                    print(len(DNA_seq))
                except:
                    print("No length for sequence")
                if DNA_seq:
                    f.write(f">{s}\n{DNA_seq}\n")
                else:
                    f.write(f">{s}\nNone\n")
    target_id2DNA = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
    target_int2DNA = {k : target_id2DNA[v] for k, v in target_int2id.items()}
    target_seq_DNA = np.array([target_int2DNA[i] for i in range(df["Target_index"].max() + 1)])

    # Assert that arrays have the same length
    assert len(drug_id) == len(drug_smiles) == coo_matrix_shape[0]
    assert len(target_id) == len(target_seq) == len(target_seq_DNA) == coo_matrix_shape[1]

    # Construct the h5torch file
    f = h5torch.File(DATASET_PATH + name + ".h5t", 'w')
    f.register((coo_matrix_indices, coo_matrix_values, coo_matrix_shape), axis="central", mode="coo", dtype_save="float32", dtype_load="float32")
    f.register(drug_id, axis=0, name="Drug_ID", dtype_save="bytes", dtype_load="str")
    f.register(drug_smiles, axis=0, name="Drug_SMILES", dtype_save="bytes", dtype_load="str")
    f.register(target_id, axis=1, name="Target_ID", dtype_save="bytes", dtype_load="str")
    f.register(target_seq, axis=1, name="Target_seq", dtype_save="bytes", dtype_load="str")
    f.register(target_seq_DNA, axis=1, name="Target_seq_DNA", dtype_save="bytes", dtype_load="str")
    f.register(df["split_rand"], axis="unstructured", name="split_rand", dtype_save="bytes", dtype_load="str")
    f.register(df["split_cold"], axis="unstructured", name="split_cold", dtype_save="bytes", dtype_load="str")
    f.close()

