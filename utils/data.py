import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import upsetplot
import h5py

from collections import defaultdict
from typing import Literal, Dict, List, Tuple, Set, Optional

from tdc.multi_pred import DTI

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
import pubchempy as pcp
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW

import h5torch

SEED = 42

THRESHOLD_pKd = 7.0
THRESHOLD_pKi = 7.6
THRESHOLD_KIBA = 12.1

MAX_N_HEAVY_ATOMS = 64
MAX_MOL_WEIGHT = 640
MAX_AA_SEQ_LEN = 1280

def load_Metz():
    try:
        return pd.read_csv('data/Metz.csv', usecols=['SMILES', 'ProteinSequence', 'Ki'])
    except FileNotFoundError:
        print('To run the Metz dataset, you need to download it from Kaggle and place it in the data folder.')
        print("https://www.kaggle.com/datasets/blk1804/metz-drug-binding-dataset")
        raise FileNotFoundError

def load_transform_data(
        datasets: list[str] = [
            "BindingDB_Kd", "DAVIS", # Kd
            "BindingDB_Ki", "Metz",  # Ki
            "KIBA",                  # KIBA score
        ], 
        verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Loads the data into pandas dataframes and transforms them into the correct format.
    
    Args:
        datasets: List of dataset names to load. If None, loads all datasets.
        verbose: Whether to print dataset statistics.
        
    Returns:
        Dictionary mapping dataset names to their dataframes.
        Each dataframe has columns:
            - Drug_SMILES (str): SMILES representation of the drug
            - Target_AA (str): Protein sequence of the target 
            - Y (bool): True if the interaction is bound, False otherwise
            - Y_{value} (float): interaction value of the drug-target interaction (if observed)
            - in_{name} (bool): True if the interaction is in the dataset, False otherwise
    """
    # ensure datasets is subset of default datasets
    if not all(dataset in ["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "KIBA", "Metz"] for dataset in datasets):
        raise ValueError("Invalid dataset name. Please use one of the following: DAVIS, BindingDB_Kd, BindingDB_Ki, KIBA, Metz")
        
    dataset_dfs = {}
    
    for name in datasets:
        print(f"Loading {name} dataset...")
        data = DTI(name=name) if name != "Metz" else load_Metz()

        if name in ["DAVIS", "BindingDB_Kd", "BindingDB_Ki"]:
            data.convert_to_log(form='binding')
            data.harmonize_affinities(mode='mean')

        df = data.get_data() if not isinstance(data, pd.DataFrame) else data

        # Standardize column names
        if name == "Metz":
            df.rename(columns={
                'SMILES': 'Drug_SMILES',
                'ProteinSequence': 'Target_AA',
                'Ki': 'Y_pKi'
            }, inplace=True)
            df["Y"] = df["Y_pKi"] >= THRESHOLD_pKi
        else:
            df.rename(columns={
                'Drug': 'Drug_SMILES',
                'Target': 'Target_AA'
            }, inplace=True)
            if name == "DAVIS":
                df.rename(columns={'Y': 'Y_pKd'}, inplace=True)
                df["Y"] = df["Y_pKd"] >= THRESHOLD_pKd
            elif name == "BindingDB_Kd":
                df.rename(columns={'Y': 'Y_pKd'}, inplace=True)
                df["Y"] = df["Y_pKd"] >= THRESHOLD_pKd
            elif name == "BindingDB_Ki":
                df.rename(columns={'Y': 'Y_pKi'}, inplace=True)
                df["Y"] = df["Y_pKi"] >= THRESHOLD_pKi
            elif name == "KIBA":
                df.rename(columns={'Y': 'Y_KIBA'}, inplace=True)
                df["Y"] = df["Y_KIBA"] >= THRESHOLD_KIBA

        df[f"in_{name}"] = True

        for col in df.columns:
            if col in ['Y_pKi', 'Y_pKd', 'Y_KIBA']:
                Y_col = col
                y_min = df[col].min()
                y_max = df[col].max()
                break

        # Convert to canonical SMILES efficiently using a dictionary
        print("Canonicalizing SMILES...")
        unique_smiles = df['Drug_SMILES'].unique()
        canon_smiles_dict = {smiles: Chem.CanonSmiles(smiles) for smiles in unique_smiles}
        df['Drug_SMILES'] = df['Drug_SMILES'].map(canon_smiles_dict)

        if verbose:
            n_drugs = df['Drug_SMILES'].nunique()
            n_targets = df['Target_AA'].nunique()
            n_pairs = df.shape[0]
            n_true = df['Y'].sum()
            percent_true = 100 * n_true / n_pairs
            percent_density = 100 * n_pairs / (n_drugs * n_targets)
            print(f"""--------- {name} Dataset Statistics ---------
Unique: {n_drugs:,} (drugs) x {n_targets:,} (targets) = {n_drugs * n_targets:,} possible pairs
Interactions: {n_pairs:,} pairs ({percent_true:.2f}% binding, {percent_density:.2f}% density)
Score ({Y_col}): {y_min:.2f} min, {y_max:.2f} max
--------------------------------------------\n""")

        dataset_dfs[name] = df[['Drug_SMILES', 'Target_AA', 'Y', Y_col, f"in_{name}"]]
        
    return dataset_dfs


def merge_dfs(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merges multiple DTI datasets into a single dataframe.
    
    Args:
        datasets: Dictionary of name: dataframe pairs
        
    Returns:
        Merged dataframe containing all interactions
    """          
    # Start with first dataframe & merge one-by-one
    dataset_names = list(datasets.keys())
    merged_df = datasets[dataset_names[0]].copy()
    for name in dataset_names[1:]:
        df = datasets[name]
        print("Merging with", name)
        # Merge on Drug_SMILES and Target_AA using outer join
        merged_df = pd.merge(
            merged_df, 
            df,
            on=['Drug_SMILES', 'Target_AA'],
            how='outer',
            suffixes=('', '_new')
        )
        # Update binary interaction (OR operation)
        merged_df['Y'] = merged_df['Y'].fillna(False) | merged_df['Y_new'].fillna(False)
        
        # Update measurement columns (max)
        value_cols = [col for col in merged_df.columns if col.startswith('Y_') and not col.endswith('_new')]
        for col in value_cols:
            new_col = f"{col}_new"
            if new_col in merged_df.columns:
                merged_df[col] = merged_df[[col, new_col]].max(axis=1)

        # Drop all temporary new columns
        new_cols = [col for col in merged_df.columns if col.endswith('_new')]
        merged_df.drop(columns=new_cols, inplace=True)
    
    # Fill missing dataset indicators with False
    for name in datasets.keys():
        if f"in_{name}" not in merged_df.columns:
            merged_df[f"in_{name}"] = False
        else:
            merged_df[f"in_{name}"] = merged_df[f"in_{name}"].fillna(False)

    # Ensure consistent column ordering
    cols = ['Drug_SMILES', 'Target_AA', 'Y']
    measure_cols = [col for col in merged_df.columns if col.startswith('Y_')]
    indicator_cols = [col for col in merged_df.columns if col.startswith('in_')]
    merged_df = merged_df[cols + measure_cols + indicator_cols]

    return merged_df


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the merged dataframe based on molecule properties and sequence length.
    Uses vectorized operations and efficient RDKit calculations.
    """
    # Create RDKit molecules once and store properties in dictionary
    properties = defaultdict(dict)
    for smiles in df['Drug_SMILES'].unique():
        if mol := Chem.MolFromSmiles(smiles):  # Using walrus operator for efficiency
            properties[smiles] = {
                'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'mol_weight': Descriptors.MolWt(mol)
            }
    
    # Create mask for sequence length (vectorized)
    seq_mask = df['Target_AA'].str.len() <= MAX_AA_SEQ_LEN
    
    # Create masks for molecular properties using map
    heavy_atoms_mask = df['Drug_SMILES'].map(
        lambda x: properties.get(x, {}).get('heavy_atoms', float('inf'))
    ) <= MAX_N_HEAVY_ATOMS
    
    mol_weight_mask = df['Drug_SMILES'].map(
        lambda x: properties.get(x, {}).get('mol_weight', float('inf'))
    ) <= MAX_MOL_WEIGHT
    
    # Combine all masks
    final_mask = seq_mask & heavy_atoms_mask & mol_weight_mask
    filtered_df = df[final_mask]

    # Report filtering results
    n_input = len(df)
    n_output = len(filtered_df)
    print(f"Filtering complete. Rows reduced from {n_input:,} to {n_output:,} "
          f"({100 * n_output/n_input:.1f}% retained)")
    
    return filtered_df


def add_splits(df: pd.DataFrame, seed: int = SEED, frac: list[float] = [0.7, 0.1, 0.2]) -> pd.DataFrame:
    """
    Adds split columns to the dataframe .
    - split_random: random split of interaction pairs into train/val/test
    - split_cold: cold split of drugs into train/val/test
    """
    np.random.seed(seed)
    train_frac, val_frac, test_frac = frac
    assert sum(frac) == 1, "Fraction must sum to 1"

    # Random split
    df['split_random'] = np.random.choice(['train', 'val', 'test'], size=len(df), p=[train_frac, val_frac, test_frac])
    
    # Cold split
    unique_drugs = df['Drug_SMILES'].unique()
    np.random.shuffle(unique_drugs)
    n_test = int(len(unique_drugs) * test_frac)
    test_drugs = set(unique_drugs[:n_test])
    df['split_cold'] = np.where(
        df['Drug_SMILES'].isin(test_drugs),
        'test',
        np.random.choice(['train', 'val'], size=len(df), p=[train_frac/(train_frac + val_frac), val_frac/(train_frac + val_frac)])
    )
    return df


def plot_dfDTI(df: pd.DataFrame):
    """
    Plots the distribution of interaction scores from a merged dataframe.
    Uses the 'in_{name}' columns to determine which dataset each interaction belongs to.
    
    Args:
        df: Merged dataframe containing interaction data and dataset indicators
    """
    sns.set_style("whitegrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define thresholds and y-limits for each type
    thresholds = {"Y_pKd": 7.0, "Y_pKi": 7.6, "Y_KIBA": 12.1}
    ylims = {"Y_pKd": (0, 20000), "Y_pKi": (0, 30000), "Y_KIBA": (0, 60000)}
    
    # Get dataset names from columns
    dataset_cols = [col for col in df.columns if col.startswith('in_')]
    names = [col.replace('in_', '') for col in dataset_cols]
    
    # Plot distributions for each interaction type
    for ax, score_type in zip([ax1, ax2, ax3], ["Y_pKd", "Y_pKi", "Y_KIBA"]):
        has_data = score_type in df.columns
        if has_data:
            # Plot each dataset's distribution separately
            for dataset_col, name in zip(dataset_cols, names):
                # only plot DAVIS and BindingDB_Kd for Y_pKd
                if score_type == "Y_pKd":
                    if name not in ["DAVIS", "BindingDB_Kd"]:
                        continue
                # only plot BindingDB_Ki for Y_pKi
                elif score_type == "Y_pKi":
                    if name not in ["BindingDB_Ki", "Metz"]:
                        continue
                # only plot KIBA for Y_KIBA
                elif score_type == "Y_KIBA":
                    if name != "KIBA":
                        continue
                # Filter data for this dataset, handling NaN values
                dataset_data = df[df[dataset_col].fillna(False)]
                # Only plot if we have data for this score type
                if len(dataset_data) > 0 and score_type in dataset_data.columns:
                    if not dataset_data[score_type].isna().all():  # Check if we have any non-NaN values
                        sns.histplot(
                            data=dataset_data,
                            x=score_type,
                            bins=25,
                            label=name,
                            ax=ax,
                            alpha=0.5
                        )
            
            # Add threshold line and set limits
            ax.axvline(x=thresholds[score_type], color='red', linestyle='--', 
                      label=f'Threshold ({thresholds[score_type]})')
            ax.set_ylim(ylims[score_type])
            
        ax.set_title(f'Histogram of {score_type.replace("Y_", "")} Distribution')
        if has_data:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)

    # Plot binary interaction distribution
    data = []
    # First add individual dataset distributions
    for dataset_col, name in zip(dataset_cols, names):
        # Filter data for this dataset, handling NaN values
        dataset_data = df[df[dataset_col].fillna(False)]
        if len(dataset_data) > 0:
            # Calculate percentage of binding interactions based on correct threshold for each dataset
            if name in ["DAVIS", "BindingDB_Kd"]:
                is_binding = dataset_data['Y_pKd'] >= thresholds['Y_pKd']
            elif name in ["BindingDB_Ki", "Metz"]:
                is_binding = dataset_data['Y_pKi'] >= thresholds['Y_pKi']
            elif name == "KIBA":
                is_binding = dataset_data['Y_KIBA'] >= thresholds['Y_KIBA']
            else:
                continue
                
            binding_count = is_binding.sum()
            total = len(dataset_data)
            data.append({
                'Dataset': name,
                'Percentage': (binding_count / total) * 100,
                'Type': 'Individual'
            })
    
    # Add merged dataset distribution using the Y boolean column
    total_interactions = len(df)
    binding_count = df['Y'].sum()
    data.append({
        'Dataset': 'Merged',
        'Percentage': (binding_count / total_interactions) * 100,
        'Type': 'Merged'
    })
    
    # Create bar plot
    plot_df = pd.DataFrame(data)
    sns.barplot(
        data=plot_df,
        x='Dataset',
        y='Percentage',
        hue='Type',
        ax=ax4,
        alpha=0.5,
        legend=False
    )
    
    # Customize the plot
    ax4.set_ylabel('Percentage of Binding Interactions (%)')
    ax4.set_xlabel('')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Binding Interaction Distribution')
    
    plt.tight_layout()
    plt.show()

def format_number(n: int) -> str:
    if n >= 10000:
        return f"{n//1000}k"
    elif n >= 1000:
        return f"{n/1000:.1f}k"
    else:
        return str(n)

def plot_dataset_overlap(df: pd.DataFrame):
    """
    Creates an UpSet plot showing the overlap of interaction pairs between datasets.
    Uses the 'in_{name}' columns to determine dataset membership.
    
    Args:
        df: Merged dataframe containing dataset indicator columns ('in_{name}')
    """
    from upsetplot import from_memberships, UpSet
    
    # Get dataset columns
    dataset_cols = [col for col in df.columns if col.startswith('in_')]
    dataset_names = [col.replace('in_', '') for col in dataset_cols]
    
    # Create list of sets for each data point's membership
    memberships = []
    for idx in df.index:
        members = set()
        for col, name in zip(dataset_cols, dataset_names):
            if df.loc[idx, col]:
                members.add(name)
        if members:  # Only add if the point belongs to at least one dataset
            memberships.append(members)
    
    # Create and plot the UpSet plot
    data = from_memberships(memberships)
    
    fig = plt.figure(figsize=(18, 6))
    upset = UpSet(data, show_counts=True, sort_by='cardinality', subset_size='count')
    
    # Customize the plot with formatted numbers
    upset.plot()
    
    # Format the count labels
    for text in plt.gca().texts:
        try:
            count = int(text.get_text())
            text.set_text(format_number(count))
        except ValueError:
            continue
            
    plt.title('Dataset Overlap Analysis')
    plt.show()





def getID_dict(smiles_list: List[str], aa_list: List[str]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Maps SMILES strings and amino acid sequences to their corresponding IDs from various datasets.
    
    Args:
        smiles_list: List of unique SMILES strings to look up
        aa_list: List of unique amino acid sequences to look up
        
    Returns:
        Tuple of (Drug_SMILE2IDs, Target_AA2IDs) dictionaries mapping strings to sets of ID strings
    """
    Drug_SMILE2IDs = {smile: set() for smile in smiles_list}
    Target_AA2IDs = {aa: set() for aa in aa_list}

    for name in ["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "Metz", "KIBA"]:
        # Load dataset
        if name == "Metz":
            data = pd.read_csv('data/Metz.csv')
            # Map protein sequence to unique GeneID and Symbol for Metz
            for seq, symbol in zip(data['ProteinSequence'], data['Symbol']):
                if seq in Target_AA2IDs:
                    Target_AA2IDs[seq].add(f"{symbol}")
        
        else:
            data = DTI(name=name).get_data()
            # Map SMILES to unique Drug_ID for other datasets
            for smile, drug_id in zip(data['Drug'], data['Drug_ID']):
                if smile in Drug_SMILE2IDs:
                    Drug_SMILE2IDs[smile].add(f"{drug_id}")
            
            # Map protein sequence to unique Target_ID for other datasets  
            for seq, target_id in zip(data['Target'], data['Target_ID']):
                if seq in Target_AA2IDs:
                    Target_AA2IDs[seq].add(f"{target_id}")

    return Drug_SMILE2IDs, Target_AA2IDs

def validate_drug_id(smiles: str, drug_id: str) -> tuple[bool, str]:
    """
    Validates a drug ID by checking if it returns the expected SMILES.
    """
    try:
        drug_id = drug_id.replace("CID:", "")
        compound = pcp.Compound.from_cid(int(drug_id))
        return Chem.CanonSmiles(compound.canonical_smiles) == smiles
    # TODO: add ChEMBL
    except Exception as e:
        print(f"Error validating drug_id {drug_id}: {e}")
        return False

def get_drug_id(smiles: str) -> str:
    """
    Searches for a drug ID using the SMILES string directly.
    Returns validated drug_id or None
    """
    try:
        # Search PubChem using SMILES
        results = pcp.get_compounds(smiles, 'smiles')
        if results:
            return f"CID:{results[0].cid}"
        # TODO: add ChEMBL
    except Exception as e:
        print(f"Error resolving SMILES {smiles}: {e}")
        return None

def resolve_target_sequence(sequence: str, target_ids: Set[str] = None) -> Tuple[str, str]:
    """
    Resolves a protein sequence to its best matching ID and DNA sequence.
    Returns (target_id, dna_sequence) or (None, None) if no match found.
    
    Args:
        sequence: Amino acid sequence
        target_ids: Optional set of potential target IDs to try first
    """
    # Configure Entrez
    Entrez.email = "robbe.claeys@ugent.be"  # Required by NCBI
    # Entrez.api_key = "your_api_key"  # Optional but recommended
    
    # 1. Try existing IDs first
    if target_ids:
        for target_id in target_ids:
            dna_seq = fetch_cds_by_id(target_id, sequence)
            if dna_seq:
                return target_id, dna_seq
    
    # 2. Search by protein sequence using BLAST
    try:
        result_handle = NCBIWWW.qblast(
            "blastp", "nr", sequence,
            expect=1e-10,  # Strict E-value threshold
            hitlist_size=5
        )
        blast_records = NCBIXML.parse(result_handle)
        
        for record in blast_records:
            for alignment in record.alignments:
                for hsp in alignment.hsps:
                    # Check for very high identity matches
                    if hsp.identities / float(hsp.align_length) > 0.95:
                        # Extract GI from alignment title
                        gi = alignment.title.split('|')[1]
                        dna_seq = fetch_cds_by_id(gi, sequence)
                        if dna_seq:
                            return gi, dna_seq
                            
    except Exception as e:
        print(f"BLAST search failed: {e}")
    
    return None, None

def fetch_cds_by_id(target_id: str, protein_seq: str) -> Optional[str]:
    """
    Fetches CDS sequence for a given target ID and verifies it translates 
    to our protein sequence.
    """
    try:
        # Search nucleotide database
        handle = Entrez.efetch(
            db="nucleotide", 
            id=target_id,
            rettype="gb", 
            retmode="text"
        )
        record = SeqIO.read(handle, "genbank")
        handle.close()
        
        # Look for CDS features
        for feature in record.features:
            if feature.type == 'CDS':
                try:
                    # Extract DNA sequence
                    dna_seq = str(feature.location.extract(record).seq)
                    
                    # Verify translation
                    if len(dna_seq) == 3*len(protein_seq) + 3:  # +3 for stop codon
                        translated = str(Seq(dna_seq).translate())[:-1]  # Remove stop codon
                        if translated == protein_seq:
                            return dna_seq
                except Exception:
                    continue
                    
    except Exception as e:
        print(f"Error fetching sequence for {target_id}: {e}")
    
    return None

def build_sequence_database(df: pd.DataFrame, 
                          target_aa2ids: Dict[str, Set[str]],
                          cache_file: str = "sequence_cache.h5") -> pd.DataFrame:
    """
    Builds a database mapping protein sequences to their IDs and DNA sequences.
    Uses caching to avoid repeated API calls.
    
    Args:
        df: DataFrame containing Target_AA sequences
        target_aa2ids: Dictionary mapping sequences to potential IDs
        cache_file: Path to HDF5 cache file
    """
    # Load cache if it exists
    cache = {}
    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            for key in f.keys():
                cache[key] = {
                    'target_id': f[key].attrs['target_id'],
                    'dna_sequence': f[key].attrs['dna_sequence']
                }
    
    # Process sequences
    results = []
    for seq in df['Target_AA'].unique():
        if seq in cache:
            target_id = cache[seq]['target_id']
            dna_seq = cache[seq]['dna_sequence']
        else:
            target_id, dna_seq = resolve_target_sequence(
                seq, 
                target_aa2ids.get(seq)
            )
            # Update cache
            if target_id and dna_seq:
                with h5py.File(cache_file, 'a') as f:
                    grp = f.create_group(seq)
                    grp.attrs['target_id'] = target_id
                    grp.attrs['dna_sequence'] = dna_seq
        
        results.append({
            'Target_AA': seq,
            'target_id': target_id,
            'dna_sequence': dna_seq
        })
    
    return pd.DataFrame(results)

    # try:
    #     # Use Entrez to fetch the sequence from the target ID
    #     handle = Entrez.efetch(db="protein", id=target_id, rettype="fasta")
    #     record = SeqIO.read(handle, "fasta")
    #     return record.seq == sequence
    # # TODO: add UniProt
    # except Exception as e:
    #     print(f"Error validating target_id {target_id}: {e}")
    #     return False

# TODO
# - use get_IDs to add DNA sequences to the AA sequences (and save fasta file)
# - add random- and cold-split columns to the dataframe
# - create h5torch dataset