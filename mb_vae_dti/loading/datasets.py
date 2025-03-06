"""
Dataset loading and processing functionality for MB-VAE-DTI.

This module provides functions for loading standard DTI datasets (DAVIS, KIBA, BindingDB),
as well as the Metz dataset. It also includes functionality for merging datasets,
filtering based on molecular properties, and saving processed data.
"""

import pandas as pd
from typing import Literal, List, Dict, Union
from rdkit import Chem
from rdkit.Chem import Descriptors
from tdc.multi_pred import DTI
from pathlib import Path

# Constants for filtering
MAX_N_HEAVY_ATOMS = 64
MAX_AA_SEQ_LEN = 1280

# Define paths
DATA_DIR = Path("data")
SOURCE_DIR = DATA_DIR / "source"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_heavy_atom_counts(smiles_list: List[str], verbose: bool = False) -> Dict[str, int]:
    """
    Compute heavy atom counts for a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        verbose: Whether to print progress information
        
    Returns:
        Dict[str, int]: Dictionary mapping SMILES strings to heavy atom counts
    """
    if verbose:
        print(f"Computing heavy atom counts for {len(smiles_list)} molecules...")
    
    heavy_atoms_dict = {}
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                heavy_atoms_dict[smiles] = Descriptors.HeavyAtomCount(mol)
            else:
                heavy_atoms_dict[smiles] = float('inf')  # Will be filtered out
        except:
            heavy_atoms_dict[smiles] = float('inf')  # Will be filtered out
    
    if verbose:
        print(f"Computed heavy atom counts for {len(heavy_atoms_dict)} molecules")
    
    return heavy_atoms_dict


def load_metz() -> pd.DataFrame:
    """
    Load the Metz dataset from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the Metz dataset with columns:
            - SMILES: SMILES representation of the drug
            - ProteinSequence: Protein sequence of the target
            - Ki: Binding affinity value
            
    Raises:
        FileNotFoundError: If the Metz.csv file is not found in the data directory
    """
    metz_path = SOURCE_DIR / "Metz.csv"
    try:
        return pd.read_csv(metz_path, usecols=['SMILES', 'ProteinSequence', 'Ki'])
    except FileNotFoundError:
        print(f'To run the Metz dataset, you need to download it from Kaggle and place it in {metz_path}')
        print("https://www.kaggle.com/datasets/blk1804/metz-drug-binding-dataset")
        raise FileNotFoundError(f"Metz dataset not found at {metz_path}")


def load_dataset(name: Literal["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "KIBA", "Metz"], 
                verbose: bool = False) -> pd.DataFrame:
    """
    Load a DTI dataset and transform it into a standardized format.
    
    Args:
        name: Name of the dataset to load
        verbose: Whether to print additional information
        
    Returns:
        pd.DataFrame: DataFrame with standardized columns:
            - Drug_SMILES: SMILES representation of the drug
            - Target_AA: Protein sequence of the target
            - Y: Binary interaction indicator (True if bound)
            - Y_{value}: Interaction value (e.g., Y_pKd, Y_pKi, Y_KIBA)
            
    Raises:
        ValueError: If the dataset name is not recognized
    """
    if verbose:
        print(f"Loading {name} dataset...")
    
    # Load dataset
    if name == "Metz":
        data = load_metz()
    else:
        # Use SOURCE_DIR as the path for TDC to store/load datasets
        data = DTI(name=name, path=str(SOURCE_DIR))
        
        # Apply transformations for binding affinity datasets
        if name in ["DAVIS", "BindingDB_Kd", "BindingDB_Ki"]:
            data.convert_to_log(form='binding')
            data.harmonize_affinities(mode='mean')
    
    # Convert to DataFrame if needed
    df = data.get_data() if not isinstance(data, pd.DataFrame) else data
    
    # Standardize column names
    if name == "Metz":
        df.rename(columns={
            'SMILES': 'Drug_SMILES',
            'ProteinSequence': 'Target_AA',
            'Ki': 'Y_pKi'
        }, inplace=True)
        # Create binary interaction column based on threshold
        df['Y'] = df['Y_pKi'] >= 7.6
    else:
        # Standardize TDC dataset columns
        df.rename(columns={
            'Drug': 'Drug_SMILES',
            'Target': 'Target_AA'
        }, inplace=True)
        if name == "DAVIS" or name == "BindingDB_Kd":
            df.rename(columns={'Y': 'Y_pKd'}, inplace=True)
            df['Y'] = df['Y_pKd'] >= 7.0
        elif name == "BindingDB_Ki":
            df.rename(columns={'Y': 'Y_pKi'}, inplace=True)
            df['Y'] = df['Y_pKi'] >= 7.6
        elif name == "KIBA":
            df.rename(columns={'Y': 'Y_KIBA'}, inplace=True)
            df['Y'] = df['Y_KIBA'] >= 12.1
    
    # Add binary interaction column based on threshold
    thresholds = {'Y_pKd': 7.0, 'Y_pKi': 7.6, 'Y_KIBA': 12.1}
    for col, threshold in thresholds.items():
        if col in df.columns:
            df['Y'] = df[col] >= threshold
            break
    
    # Drop rows with missing values in key columns
    rows_before = len(df)
    df = df.dropna(subset=['Drug_SMILES', 'Target_AA'])
    if verbose and len(df) < rows_before:
        print(f"Dropped {rows_before - len(df)} rows with missing Drug_SMILES or Target_AA values")
    
    # Remove duplicate drug-target pairs (keeping the one with the highest affinity)
    if len(df) > df[['Drug_SMILES', 'Target_AA']].drop_duplicates().shape[0]:
        if verbose:
            print(f"Found duplicate drug-target pairs in {name}. Keeping pairs with highest affinity.")
        
        # Get the first value column (more efficient than list comprehension)
        value_cols = [col for col in df.columns if col.startswith('Y_')]
        value_col = value_cols[0] if value_cols else None
        
        if value_col:
            # Use groupby with idxmax for better performance than sort+drop_duplicates
            idx = df.groupby(['Drug_SMILES', 'Target_AA'])[value_col].idxmax()
            df = df.loc[idx]
        else:
            # Fallback if no value column found
            df = df.drop_duplicates(subset=['Drug_SMILES', 'Target_AA'], keep='first')
    
    if verbose:
        print(f"Loaded {name} dataset with {len(df)} interactions")
    
    return df


def merge_datasets(names: List[str], verbose: bool = False) -> pd.DataFrame:
    """
    Merge multiple DTI datasets into a unified dataset.
    
    Args:
        names: List of dataset names to merge
        verbose: Whether to print additional information
        
    Returns:
        pd.DataFrame: Merged DataFrame with all interactions from the input datasets
        
    Raises:
        ValueError: If no valid datasets are provided
    """
    # Phase 1: Load and transform individual datasets
    datasets = []
    for name in names:
        try:
            df = load_dataset(name, verbose=verbose)
            df[f"in_{name}"] = True
            datasets.append(df)
        except FileNotFoundError:
            print(f"Dataset {name} not found. Skipping...")
            continue
        print("\n") if verbose else None
    
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
    
    if verbose:
        print(f"Merged dataset contains {len(merged_df)} unique drug-target pairs")
    
    return merged_df


def filter_dataset(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Filter a dataset based on molecular properties.
    
    Args:
        df: DataFrame to filter
        verbose: Whether to print additional information
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if verbose:
        print(f"Filtering dataset with {len(df)} rows...")
        print(f"  - Max heavy atoms: {MAX_N_HEAVY_ATOMS}")
        print(f"  - Max protein sequence length: {MAX_AA_SEQ_LEN}")
    
    # Compute heavy atom counts for unique SMILES (more efficient)
    unique_smiles = df['Drug_SMILES'].unique()
    heavy_atoms_dict = compute_heavy_atom_counts(unique_smiles, verbose=verbose)
    
    # Apply the filter using the precomputed dictionary
    df['n_heavy_atoms'] = df['Drug_SMILES'].map(heavy_atoms_dict)
    
    # Filter by sequence length and heavy atom count
    filtered_df = df[
        (df['Target_AA'].str.len() <= MAX_AA_SEQ_LEN) &
        (df['n_heavy_atoms'] <= MAX_N_HEAVY_ATOMS)
    ]
    
    # Remove the temporary column
    filtered_df = filtered_df.drop(columns=['n_heavy_atoms'])
    
    if verbose:
        print(f"Filtering complete. Rows reduced from {len(df)} to {len(filtered_df)}")
    
    return filtered_df


def save_dataset(df: pd.DataFrame, name: str, verbose: bool = False) -> str:
    """
    Save a dataset to a CSV file.
    
    Args:
        df: DataFrame to save
        name: Name of the dataset
        verbose: Whether to print additional information
        
    Returns:
        str: Path to the saved file
    """
    # Create the output path
    output_path = PROCESSED_DIR / f"{name}.csv"
    
    # Save the DataFrame
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"Saved dataset to {output_path}")
    
    return str(output_path)


def load_or_create_merged_dataset(
    dataset_names: List[str], 
    force_reload: bool = False,
    apply_filters: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load a merged dataset from disk or create it if it doesn't exist.
    
    Args:
        dataset_names: List of dataset names to include
        force_reload: Whether to force reloading the datasets even if a merged file exists
        apply_filters: Whether to apply filters to the merged dataset
        verbose: Whether to print additional information
        
    Returns:
        pd.DataFrame: Merged and optionally filtered DataFrame
    """
    # Generate a filename based on the dataset names and filtering
    dataset_key = "_".join(sorted(dataset_names))
    filter_suffix = "_filtered" if apply_filters else ""
    merged_filename = f"merged_{dataset_key}{filter_suffix}.csv"
    merged_path = PROCESSED_DIR / merged_filename
    
    # Check if the merged file already exists
    if not force_reload and merged_path.exists():
        if verbose:
            print(f"Loading existing merged dataset from {merged_path}")
        return pd.read_csv(merged_path)
    
    # Create the merged dataset
    if verbose:
        print(f"Creating merged dataset from: {', '.join(dataset_names)}")
    
    merged_df = merge_datasets(dataset_names, verbose=verbose)
    
    # Apply filters if requested
    if apply_filters:
        merged_df = filter_dataset(merged_df, verbose=verbose)
    
    # Save the merged dataset
    save_dataset(merged_df, merged_filename.replace(".csv", ""), verbose=verbose)
    
    return merged_df


def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Calculate statistics for a dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict: Dictionary containing dataset statistics
    """
    stats = {
        "n_interactions": len(df),
        "n_unique_drugs": df['Drug_SMILES'].nunique(),
        "n_unique_targets": df['Target_AA'].nunique(),
        "avg_drug_smiles_length": df['Drug_SMILES'].str.len().mean(),
        "avg_target_aa_length": df['Target_AA'].str.len().mean(),
    }
    
    # Add statistics for each measurement column
    for col in df.columns:
        if col.startswith('Y_'):
            stats[f"{col}_mean"] = df[col].mean()
            stats[f"{col}_std"] = df[col].std()
            stats[f"{col}_min"] = df[col].min()
            stats[f"{col}_max"] = df[col].max()
    
    # Add statistics for dataset indicators
    for col in df.columns:
        if col.startswith('in_'):
            stats[f"{col}_count"] = df[col].sum()
            stats[f"{col}_percentage"] = (df[col].sum() / len(df)) * 100
    
    return stats 