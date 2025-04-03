"""
This module provides functionality for storing the DTI data in an `h5torch` file.

The `h5torch` file is structured as a matrix of drug-target interactions, where the rows
correspond to drugs and the columns correspond to targets. The value at the intersection
of a row and a column is the interaction strength between the drug and the target.
"""

import pandas as pd
import h5torch
import numpy as np
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any, Tuple, Union

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

def _get_stratification_columns(df: pd.DataFrame) -> List[str]:
    """Helper function to get stratification columns from the dataframe."""
    return [col for col in df.columns if col.startswith('in_')]

def _create_composite_stratification_key(df: pd.DataFrame, strat_cols: List[str]) -> pd.Series:
    """Creates a composite stratification key from multiple columns."""
    strat_key = ''
    for col in strat_cols:
        strat_key += df[col].astype(str) + '_'
    return strat_key

def _assign_stratified_split(
    items: np.ndarray, 
    strat_dict: Dict[Any, List[Any]], 
    frac: Tuple[float, float, float]
) -> Dict[Any, str]:
    """
    Assign stratified train/valid/test splits to items based on stratification groups.
    
    Args:
        items: Array of item identifiers (drugs or indices)
        strat_dict: Dictionary mapping stratification keys to lists of items
        frac: Split fractions (train, valid, test)
        
    Returns:
        Dictionary mapping each item to its split assignment
    """
    item_to_split = {}
    
    # For each stratification group, split the items
    for strat_key, strat_items in strat_dict.items():
        # Shuffle the items
        np.random.shuffle(strat_items)
        
        # Calculate split sizes
        train_size = int(len(strat_items) * frac[0])
        valid_size = int(len(strat_items) * frac[1])
        
        # Assign splits
        for item in strat_items[:train_size]:
            item_to_split[item] = 'train'
        for item in strat_items[train_size:train_size+valid_size]:
            item_to_split[item] = 'valid'
        for item in strat_items[train_size+valid_size:]:
            item_to_split[item] = 'test'
    
    return item_to_split

def _split_items(
    items: np.ndarray, 
    frac: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split items into train, valid, test sets based on fractions.
    
    Args:
        items: Array of item identifiers
        frac: Split fractions (train, valid, test)
        
    Returns:
        Tuple of (train_items, valid_items, test_items)
    """
    np.random.shuffle(items)
    
    # Calculate split sizes
    train_size = int(len(items) * frac[0])
    valid_size = int(len(items) * frac[1])
    
    # Split items
    train_items = items[:train_size]
    valid_items = items[train_size:train_size+valid_size]
    test_items = items[train_size+valid_size:]
    
    return train_items, valid_items, test_items

def _items_to_split_dict(
    train_items: np.ndarray, 
    valid_items: np.ndarray, 
    test_items: np.ndarray
) -> Dict[Any, str]:
    """
    Create a mapping from items to their split assignments.
    
    Args:
        train_items: Items in the train set
        valid_items: Items in the valid set
        test_items: Items in the test set
        
    Returns:
        Dictionary mapping each item to its split assignment
    """
    item_to_split = {}
    
    for item in train_items:
        item_to_split[item] = 'train'
    for item in valid_items:
        item_to_split[item] = 'valid'
    for item in test_items:
        item_to_split[item] = 'test'
    
    return item_to_split

def add_split_cols(
        df: pd.DataFrame,
        frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        rand_split: bool = True,
        cold_split: bool = True,
        stratify: bool = True,
        seed: int = 42
) -> pd.DataFrame:
    """
    Adds split columns to the dataframe based on the given fractions.
    
    Args:
        df: dataframe to add split columns to
        frac: train, valid, test split fractions (default: (0.8, 0.1, 0.1))
        rand_split: whether to add a random split (default: True)
        cold_split: whether to add a cold split (default: True)
        stratify: whether to stratify the split on in_... columns (default: True)
        seed: random seed (default: 42)
        
    Returns:
        DataFrame with added split columns
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Validate fractions
    if sum(frac) != 1.0:
        raise ValueError(f"Split fractions must sum to 1.0, got {sum(frac)}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get stratification columns if needed
    strat_cols = _get_stratification_columns(df) if stratify else []
    
    # Random split
    if rand_split:
        if stratify and strat_cols:
            # Create a composite stratification key
            df['_strat_key'] = _create_composite_stratification_key(df, strat_cols)
            
            # Get unique stratification keys and their counts
            strat_counts = df['_strat_key'].value_counts()
            
            # Initialize split column
            df['split_rand'] = ''
            
            # For each stratification group
            for strat_key, count in strat_counts.items():
                # Get indices for this stratification group
                indices = df[df['_strat_key'] == strat_key].index
                
                # Shuffle indices
                shuffled_indices = np.random.permutation(indices)
                
                # Calculate split sizes
                train_size = int(count * frac[0])
                valid_size = int(count * frac[1])
                
                # Assign splits
                df.loc[shuffled_indices[:train_size], 'split_rand'] = 'train'
                df.loc[shuffled_indices[train_size:train_size+valid_size], 'split_rand'] = 'valid'
                df.loc[shuffled_indices[train_size+valid_size:], 'split_rand'] = 'test'
            
            # Remove temporary stratification key
            df.drop('_strat_key', axis=1, inplace=True)
        else:
            # Simple random split without stratification
            df['split_rand'] = np.random.choice(
                ['train', 'valid', 'test'], 
                size=len(df), 
                p=frac
            )
    
    # Cold drug split (drugs in test set don't appear in train set)
    if cold_split:
        unique_drugs = df['Drug_ID'].unique()
        
        if stratify and strat_cols:
            # For each drug, determine its stratification profile
            drug_strat_profiles = {}
            
            # Group by Drug_ID and aggregate the stratification columns
            drug_groups = df.groupby('Drug_ID')
            
            for drug, group in drug_groups:
                # Create a stratification profile for this drug
                # We'll use the most common value for each stratification column
                profile = {}
                for col in strat_cols:
                    # Get the most common value (True/False) for this column for this drug
                    profile[col] = group[col].mode()[0]
                
                # Create a composite key from the profile
                strat_key = '_'.join([f"{col}_{profile[col]}" for col in strat_cols])
                drug_strat_profiles[drug] = strat_key
            
            # Group drugs by their stratification profiles
            strat_groups = {}
            for drug, strat_key in drug_strat_profiles.items():
                if strat_key not in strat_groups:
                    strat_groups[strat_key] = []
                strat_groups[strat_key].append(drug)
            
            # Assign splits based on stratification
            drug_to_split = _assign_stratified_split(unique_drugs, strat_groups, frac)
        else:
            # Simple random split of drugs
            train_drugs, valid_drugs, test_drugs = _split_items(unique_drugs, frac)
            drug_to_split = _items_to_split_dict(train_drugs, valid_drugs, test_drugs)
        
        # Apply the mapping to create the split column
        df['split_cold'] = df['Drug_ID'].map(drug_to_split)
    
    return df

def create_h5torch(
    df: pd.DataFrame,
    output_filename: str = "data.h5torch",
    drug_id_col: str = "Drug_ID",
    target_id_col: str = "Target_ID",
    interaction_col: str = "Y",
    drug_feature_cols: List[str] = ["Drug_InChIKey", "Drug_SMILES"],
    target_feature_cols: List[str] = [
        "Target_UniProt_ID", "Target_Gene_name",
        "Target_RefSeq_ID", "Target_AA", "Target_DNA"],
    additional_y_cols: List[str] = ["Y_pKd", "Y_pKi", "Y_KIBA"],
    provenance_cols: List[str] = ["in_DAVIS", "in_BindingDB_Kd", "in_BindingDB_Ki", "in_Metz", "in_KIBA"],
    split_cols: List[str] = ["split_rand", "split_cold"]
) -> None:
    """
    Creates and saves an h5torch file for drug-target interaction data.

    The file is structured with drugs (axis 0) and targets (axis 1) as the two axes of an interaction matrix.
    The central object is a sparse COO matrix containing the binary interaction values (Y).
    Drug features are aligned to axis 0, and target features are aligned to axis 1.

    Interaction-specific data like continuous interaction values (Y_pKd, etc.),
    split assignments, and provenance flags (in_DAVIS, etc.) are stored as
    'unstructured' datasets. Each unstructured dataset has the same length as the
    number of observed interactions.

    This structure optimizes for "coo" sampling, allowing efficient retrieval of
    drug features, target features, and the interaction values for each observed interaction. 
    Accessing the unstructured interaction-specific data requires a custom Dataset class (see usage example). 
    Subsetting based on split or metadata columns is supported via the 'subset' argument.

    Args:
        df: DataFrame containing the DTI data (must include all specified columns)
        output_filename: Name of the output file (without path)
        drug_id_col: Column name for the primary drug identifier
        target_id_col: Column name for the primary target identifier
        interaction_col: Column name for the binary interaction values (Y)
        drug_feature_cols: List of column names for drug features (aligned to axis 0)
        target_feature_cols: List of column names for target features (aligned to axis 1)
        additional_y_cols: List of column names for continuous interaction values (unstructured)
        provenance_cols: List of column names for provenance flags (unstructured)
        split_cols: List of column names for split assignments (unstructured)

    Returns:
        None: The function saves the h5torch file to disk in the PROCESSED_DIR
    """
    df = df.copy()
    print(f"Creating h5torch file from dataframe with {len(df)} rows...")
    
    # Get unique drugs and targets and map them to integer indices
    unique_drugs = pd.unique(df[drug_id_col])
    unique_targets = pd.unique(df[target_id_col])

    print(f"Found {len(unique_drugs)} unique drugs and {len(unique_targets)} unique targets")

    drug_id2int = {drug: i for i, drug in enumerate(unique_drugs)}
    target_id2int = {target: i for i, target in enumerate(unique_targets)}

    # Add integer indices to the dataframe
    df["Drug_index"] = df[drug_id_col].map(drug_id2int)
    df["Target_index"] = df[target_id_col].map(target_id2int)

    # Ensure interaction column is boolean
    if not pd.api.types.is_bool_dtype(df[interaction_col]):
         print(f"Warning: Interaction column '{interaction_col}' is not boolean. Converting...")
         # Convert potential 1/0 or True/False strings/numbers to boolean
         df[interaction_col] = df[interaction_col].apply(lambda x: str(x).lower() in ['true', '1', '1.0'])


    # Generate COO matrix data for the central object (binary interaction matrix)
    # Interactions are ordered according to the input DataFrame rows
    coo_matrix_indices = df[["Drug_index", "Target_index"]].values.T  # Shape: (2, n_interactions)
    coo_matrix_values = df[interaction_col].values                    # Shape: (n_interactions,)
    coo_matrix_shape = (len(unique_drugs), len(unique_targets))       # Shape of the full matrix

    print(f"Creating central interaction matrix of shape {coo_matrix_shape} with {len(coo_matrix_values)} observed interactions")

    # Create the central object in the format expected by h5torch
    central_object = (coo_matrix_indices, coo_matrix_values, coo_matrix_shape)

    # --- Process Axis-Aligned Features ---
    # Process drug features aligned to axis 0
    drug_features = {} # dict of feature arrays indexed by Drug_index
    print("Processing drug features...")
    for col in [drug_id_col] + drug_feature_cols:
        if col in df.columns:
            # Use drop_duplicates based on Drug_ID to handle potential multiple rows per drug
            # Ensure we get one feature value per unique drug, ordered by Drug_index
            drug_df_unique = df[["Drug_index", col]].drop_duplicates(subset=["Drug_index"]).sort_values("Drug_index")

            # Check if we have features for all unique drugs
            if len(drug_df_unique) != len(unique_drugs):
                print(f"Warning: Found features for {len(drug_df_unique)} drugs in column '{col}', but expected {len(unique_drugs)}. Missing drugs will have default values.")
                # Create a Series indexed by Drug_index for easy alignment
                feature_series = pd.Series(index=np.arange(len(unique_drugs)), dtype=drug_df_unique[col].dtype)
                feature_series.loc[drug_df_unique["Drug_index"]] = drug_df_unique[col].values
                # Fill missing values (NaN for numeric/object, False for bool, "" for string?)
                if pd.api.types.is_numeric_dtype(feature_series.dtype):
                    feature_series = feature_series.fillna(np.nan)
                elif pd.api.types.is_bool_dtype(feature_series.dtype):
                    feature_series = feature_series.fillna(False)
                else: # Assume string/object
                    feature_series = feature_series.fillna("")
                feature_array = feature_series.values
            else:
                feature_array = drug_df_unique[col].values

            drug_features[col] = feature_array
        else:
            print(f"Warning: Drug feature column '{col}' not found in DataFrame.")

    # Process target features aligned to axis 1
    target_features = {} # dict of feature arrays indexed by Target_index
    print("Processing target features...")
    for col in [target_id_col] + target_feature_cols:
        if col in df.columns:
            # Use drop_duplicates based on Target_ID
            target_df_unique = df[["Target_index", col]].drop_duplicates(subset=["Target_index"]).sort_values("Target_index")

            if len(target_df_unique) != len(unique_targets):
                print(f"Warning: Found features for {len(target_df_unique)} targets in column '{col}', but expected {len(unique_targets)}. Missing targets will have default values.")
                feature_series = pd.Series(index=np.arange(len(unique_targets)), dtype=target_df_unique[col].dtype)
                feature_series.loc[target_df_unique["Target_index"]] = target_df_unique[col].values
                if pd.api.types.is_numeric_dtype(feature_series.dtype):
                    feature_series = feature_series.fillna(np.nan)
                elif pd.api.types.is_bool_dtype(feature_series.dtype):
                    feature_series = feature_series.fillna(False)
                else: # Assume string/object
                    feature_series = feature_series.fillna("")
                feature_array = feature_series.values
            else:
                feature_array = target_df_unique[col].values

            target_features[col] = feature_array
        else:
            print(f"Warning: Target feature column '{col}' not found in DataFrame.")

    # --- Create HDF5 File ---
    output_path = PROCESSED_DIR / output_filename
    with h5torch.File(str(output_path), 'w') as f:
        # Register the binary interaction matrix as the central object in COO format
        # The values are boolean. h5torch handles boolean type.
        f.register(
            central_object,
            "central",
            mode="coo"
            # dtype_save="bool" is implicit for boolean numpy array
        )

        # Register drug features aligned to axis 0
        print("Registering drug features...")
        for name, values in drug_features.items():
            if pd.api.types.is_string_dtype(values.dtype) or pd.api.types.is_object_dtype(values.dtype):
                # Convert string/object arrays to byte arrays for HDF5 compatibility
                f.register(values, 0, name=name, dtype_save="bytes", dtype_load="str")
            elif pd.api.types.is_bool_dtype(values.dtype):
                 f.register(values, 0, name=name, dtype_save="bool")
            else: # Assume numeric
                f.register(values, 0, name=name, dtype_save="float32")

        # Register target features aligned to axis 1
        print("Registering target features...")
        for name, values in target_features.items():
            if pd.api.types.is_string_dtype(values.dtype) or pd.api.types.is_object_dtype(values.dtype):
                f.register(values, 1, name=name, dtype_save="bytes", dtype_load="str")
            elif pd.api.types.is_bool_dtype(values.dtype):
                 f.register(values, 1, name=name, dtype_save="bool")
            else: # Assume numeric
                f.register(values, 1, name=name, dtype_save="float32")

        # --- Register Unstructured Data (Interaction-Level) ---
        # These datasets have length = number of interactions = len(df)

        # Register split information
        print("Registering split information as unstructured data...")
        for col in split_cols:
            if col in df.columns:
                # Splits are strings ('train', 'valid', 'test')
                f.register(df[col].values, "unstructured", name=col, dtype_save="bytes", dtype_load="str")
            else:
                print(f"Warning: Split column '{col}' not found in DataFrame.")

        # Register provenance flags as unstructured data
        print("Registering provenance flags...")
        for col in provenance_cols:
            if col in df.columns:
                # Provenance flags are typically boolean flags (e.g., in_DAVIS)
                data_to_register = df[col].values
                if pd.api.types.is_bool_dtype(data_to_register.dtype):
                    f.register(data_to_register, "unstructured", name=col, dtype_save="bool")
                else: # Assume numeric? Or force bool? Let's stick to bool or bytes.
                     print(f"Warning: Provenance flag column '{col}' has unexpected type {data_to_register.dtype}. Ignoring.")
            else:
                print(f"Warning: Provenance flag column '{col}' not found in DataFrame.")

        # Register additional Y values as unstructured data
        print("Registering additional Y values as unstructured data...")
        for col in additional_y_cols:
            if col in df.columns:
                # These are float values (e.g., pKd). Save as float32 for PyTorch.
                # Convert column to numeric, coercing errors to NaN, then fill NaN if needed before saving.
                values = pd.to_numeric(df[col], errors='coerce').fillna(np.nan).values
                f.register(values, "unstructured", name=col, dtype_save="float32")
            else:
                print(f"Warning: Additional Y column '{col}' not found in DataFrame.")

        # Add dataset attributes with useful information
        f.attrs["n_drugs"] = len(unique_drugs)
        f.attrs["n_targets"] = len(unique_targets)
        f.attrs["n_interactions"] = len(coo_matrix_values) # Number of rows in original df
        f.attrs["sparsity"] = len(coo_matrix_values) / (len(unique_drugs) * len(unique_targets))
        f.attrs["created_at"] = pd.Timestamp.now().isoformat()

    print(f"Created h5torch file at {output_path}")
    return None

