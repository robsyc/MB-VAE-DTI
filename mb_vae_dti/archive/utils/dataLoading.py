import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from tdc.multi_pred import DTI
from typing import Literal, Dict, List, Tuple, Set
import h5torch
from collections import defaultdict
from Bio import Entrez, SeqIO

MAX_N_HEAVY_ATOMS = 64
MAX_AA_SEQ_LEN = 1280

def load_Metz():
    try:
        return pd.read_csv('data/Metz.csv', usecols=['SMILES', 'ProteinSequence', 'Ki'])
    except FileNotFoundError:
        print('To run the Metz dataset, you need to download it from Kaggle and place it in the data folder.')
        print("https://www.kaggle.com/datasets/blk1804/metz-drug-binding-dataset")
        raise FileNotFoundError

def load_transform_data(name: Literal["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "KIBA", "Metz"], verbose: bool = False):
    """
    Loads the data into a pandas dataframe and transforms it into the correct format.
    Columns:
        - Drug_SMILES (str): SMILES representation of the drug
        - Target_AA (str): Protein sequence of the target
        - Y (bool): True if the interaction is bound, False otherwise
        - Y_{value} (float): interaction value of the drug-target interaction (if observed)
        - in_{name} (bool): True if the interaction is in the dataset, False otherwise
    """
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
        df["Y"] = df["Y_pKi"] >= 7.6
    else:
        df.rename(columns={
            'Drug': 'Drug_SMILES',
            'Target': 'Target_AA'
        }, inplace=True)
        if name == "DAVIS":
            df.rename(columns={'Y': 'Y_pKd'}, inplace=True)
            df["Y"] = df["Y_pKd"] >= 7.0
        elif name == "BindingDB_Kd":
            df.rename(columns={'Y': 'Y_pKd'}, inplace=True)
            df["Y"] = df["Y_pKd"] >= 7.0
        elif name == "BindingDB_Ki":
            df.rename(columns={'Y': 'Y_pKi'}, inplace=True)
            df["Y"] = df["Y_pKi"] >= 7.6
        elif name == "KIBA":
            df.rename(columns={'Y': 'Y_KIBA'}, inplace=True)
            df["Y"] = df["Y_KIBA"] >= 12.1

    df[f"in_{name}"] = True

    for col in df.columns:
        if col in ['Y_pKi', 'Y_pKd', 'Y_KIBA']:
            Y_col = col
            y_min = df[col].min()
            y_max = df[col].max()
            break

    print("""--- Dataset Statistics ---
{} unique drugs.
{} unique targets.
{} drug-target pairs.
{} min interaction score.
{} max interaction score.
--------------------------\n""".format(
    df['Drug_SMILES'].nunique(), df['Target_AA'].nunique(), 
    df.shape[0], y_min, y_max)) if verbose else None

    return df[['Drug_SMILES', 'Target_AA', 'Y', Y_col, f"in_{name}"]]

def plot_interaction_distribution(
        dfs: list[pd.DataFrame],
        names: list[str]
    ):
    """
    Plots the distribution of interaction scores for given datasets.
    
    Args:
        dfs: List of dataframes containing interaction data
        names: List of dataset names corresponding to the dataframes
    """
    sns.set_style("whitegrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define thresholds and y-limits for each type
    thresholds = {"Y_pKd": 7.0, "Y_pKi": 7.6, "Y_KIBA": 12.1}
    ylims = {"Y_pKd": (0, 20000), "Y_pKi": (0, 40000), "Y_KIBA": (0, 80000)}
    
    # Plot distributions for each interaction type
    for ax, score_type in zip([ax1, ax2, ax3], ["Y_pKd", "Y_pKi", "Y_KIBA"]):
        has_data = False
        for df, name in zip(dfs, names):
            if score_type in df.columns:
                has_data = True
                sns.histplot(
                    data=df,
                    x=score_type,
                    bins=25,
                    label=name,
                    ax=ax,
                    alpha=0.5
                )
                ax.set_xlabel(score_type.replace("Y_", ""))
        
        # Add threshold line and set limits if we have data for this type
        if has_data:
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

    # Plot binary interaction distribution with colored bars for bound/unbound
    data = []
    for df, name in zip(dfs, names):
        # Calculate percentages
        counts = df['Y'].value_counts()
        total = len(df)
        data.append({
            'Dataset': name,
            'Status': 'Bound',
            'Percentage': (counts.get(True, 0) / total) * 100
        })
        data.append({
            'Dataset': name,
            'Status': 'Unbound',
            'Percentage': (counts.get(False, 0) / total) * 100
        })
    
    # Convert to DataFrame for easier plotting
    plot_df = pd.DataFrame(data)
    
    # Create grouped bar plot with seaborn styling
    sns.set_style("whitegrid")
    sns.barplot(
        data=plot_df,
        x='Dataset',
        y='Percentage',
        hue='Status',
        ax=ax4,
        alpha=0.5
    )
    
    # Customize the plot to match histogram style
    ax4.set_ylabel('Percentage of Interactions (%)')
    ax4.set_xlabel('')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Binary Interaction Distribution')
    ax4.set_xlabel('Interaction')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def merge_df(names: list[str]) -> pd.DataFrame:
    """
    Create a unified DTI dataset through efficient pandas operations.
    """
    # Phase 1: Load and transform individual datasets
    datasets = []
    for name in names:
        try:
            df = load_transform_data(name)
            df[f"in_{name}"] = True
            datasets.append(df)
        except FileNotFoundError:
            print(f"Dataset {name} not found. Skipping...")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets provided")
    
    # Phase 2: Merge datasets efficiently
    merged_df = datasets[0]
    
    for df in datasets[1:]:
        # Merge on Drug_SMILES and Target_AA using outer join
        merged_df = pd.merge(
            merged_df, 
            df,
            on=['Drug_SMILES', 'Target_AA'],
            how='outer',
            suffixes=('', '_right')
        )

        # Update binary interaction (OR operation)
        merged_df['Y'] = merged_df['Y'].fillna(False) | merged_df['Y_right'].fillna(False)
        
        # Update measurement columns
        value_cols = [col for col in merged_df.columns if (col.startswith('Y_p') or col.startswith('Y_KIBA')) and not col.endswith('_right')]
        for col in value_cols:
            right_col = f"{col}_right"
            if right_col in merged_df.columns:
                # Take max of the two columns, handling NaN values
                merged_df[col] = merged_df[[col, right_col]].max(axis=1)

        # Drop all temporary right columns
        right_cols = [col for col in merged_df.columns if col.endswith('_right')]
        merged_df.drop(columns=right_cols, inplace=True)
    
    # Fill missing dataset indicators with False
    for name in names:
        if f"in_{name}" not in merged_df.columns:
            merged_df[f"in_{name}"] = False
        else:
            merged_df[f"in_{name}"] = merged_df[f"in_{name}"].fillna(False)

    # Ensure consistent column ordering
    cols = ['Drug_SMILES', 'Target_AA', 'Y']
    measure_cols = [col for col in merged_df.columns if (col.startswith('Y_p') or col.startswith('Y_KIBA'))]
    indicator_cols = [col for col in merged_df.columns if col.startswith('in_')]
    merged_df = merged_df[cols + measure_cols + indicator_cols]

    return merged_df

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the merged dataframe.
    """
    unique_smiles = df['Drug_SMILES'].unique()
    heavy_atoms_dict = {
        smiles: Descriptors.HeavyAtomCount(Chem.MolFromSmiles(smiles)) 
        for smiles in unique_smiles
    }
    df['n_heavy_atoms'] = df['Drug_SMILES'].map(heavy_atoms_dict)
    filtered_df = df[
        # Filter by sequence length (vectorized operation)
        (df['Target_AA'].str.len() <= MAX_AA_SEQ_LEN) &
        # Filter by heavy atom count (using pre-computed values)
        (df['n_heavy_atoms'] <= MAX_N_HEAVY_ATOMS)
    ]

    print(f"Filtering complete. Rows reduced from {len(df)} to {len(filtered_df)}")
    
    return filtered_df.drop(columns=['n_heavy_atoms'])

def get_IDs():
    Drug_SMILE2IDs = {}
    Target_AA2IDs = {}

    for name in ["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "Metz", "KIBA"]:
        # Load dataset
        if name == "Metz":
            data = pd.read_csv('data/Metz.csv')
            # Map protein sequence to unique GeneID and Symbol for Metz
            for seq, symbol in zip(data['ProteinSequence'], data['Symbol']):
                if seq not in Target_AA2IDs:
                    Target_AA2IDs[seq] = set()
                Target_AA2IDs[seq].add(f"{symbol}")
        
        else:
            data = DTI(name=name).get_data()
            # Map SMILES to unique Drug_ID for other datasets
            for smile, drug_id in zip(data['Drug'], data['Drug_ID']):
                if smile not in Drug_SMILE2IDs:
                    Drug_SMILE2IDs[smile] = set()
                Drug_SMILE2IDs[smile].add(f"{drug_id}")
            
            # Map protein sequence to unique Target_ID for other datasets  
            for seq, target_id in zip(data['Target'], data['Target_ID']):
                if seq not in Target_AA2IDs:
                    Target_AA2IDs[seq] = set()
                Target_AA2IDs[seq].add(f"{target_id}")

    return Drug_SMILE2IDs, Target_AA2IDs


# TODO
# - use get_IDs to add DNA sequences to the AA sequences (and save fasta file)
# - add random- and cold-split columns to the dataframe
# - create h5torch dataset