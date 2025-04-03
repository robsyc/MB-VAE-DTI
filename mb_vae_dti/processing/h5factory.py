"""
This module provides functionality for loading the h5torch file as a Dataset object.
"""

from typing import List, Dict, Union, Optional, Any, Tuple, Literal
from pathlib import Path
import h5torch
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"


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

    # Generate COO matrix data for the central object (binary interaction matrix)
    # Interactions are ordered according to the input DataFrame rows
    coo_matrix_indices = df[["Drug_index", "Target_index"]].values.T  # Shape: (2, n_interactions)
    coo_matrix_values = df[interaction_col].values.astype(bool)       # Shape: (n_interactions,)
    coo_matrix_shape = (len(unique_drugs), len(unique_targets))       # Shape of the full matrix

    central_object = (coo_matrix_indices, coo_matrix_values, coo_matrix_shape)

    print(f"Creating central interaction matrix of shape {coo_matrix_shape} with {len(coo_matrix_values)} observed interactions")

    # --- Process Axis-Aligned Features ---
    # Process drug features aligned to axis 0
    drug_features = {} # dict of feature arrays indexed by Drug_index
    print("Processing drug features...")
    for col in [drug_id_col] + drug_feature_cols:
        if col in df.columns:
            # Ensure we get one feature value per unique drug, ordered by Drug_index
            drug_df_unique = df[["Drug_index", col]].drop_duplicates(subset=["Drug_index"]).sort_values("Drug_index")

            # Check if we have features for all unique drugs
            assert len(drug_df_unique) == len(unique_drugs), f"Found features for {len(drug_df_unique)} drugs in column '{col}', but expected {len(unique_drugs)}."

            drug_features[col] = drug_df_unique[col].values
        else:
            print(f"Warning: Drug feature column '{col}' not found in DataFrame.")

    # Process target features aligned to axis 1
    target_features = {} # dict of feature arrays indexed by Target_index
    print("Processing target features...")
    for col in [target_id_col] + target_feature_cols:
        if col in df.columns:
            # Use drop_duplicates based on Target_ID
            target_df_unique = df[["Target_index", col]].drop_duplicates(subset=["Target_index"]).sort_values("Target_index")

            # Check if we have features for all unique targets
            assert len(target_df_unique) == len(unique_targets), f"Found features for {len(target_df_unique)} targets in column '{col}', but expected {len(unique_targets)}."

            target_features[col] = target_df_unique[col].values
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
            mode="coo",
            dtype_save="bool"
        )

        # --- Register Axis-Aligned Features ---
        # Register drug features aligned to axis 0
        for name, values in drug_features.items():
            f.register(values, 0, name=name, dtype_save="bytes", dtype_load="str")

        # Register target features aligned to axis 1
        for name, values in target_features.items():
            f.register(values, 1, name=name, dtype_save="bytes", dtype_load="str")

        # --- Register Unstructured Data (Interaction-Level) ---
        # Register split information
        for col in split_cols:
            if col in df.columns:
                # Splits are strings ('train', 'valid', 'test')
                f.register(df[col].values, "unstructured", name=col, dtype_save="bytes", dtype_load="str")
            else:
                print(f"Warning: Split column '{col}' not found in DataFrame.")

        # Register provenance flags
        for col in provenance_cols:
            if col in df.columns:
                # Provenance flags are boolean flags
                f.register(df[col].values, "unstructured", name=col, dtype_save="bool")
            else:
                print(f"Warning: Provenance flag column '{col}' not found in DataFrame.")

        # Register additional Y values
        for col in additional_y_cols:
            if col in df.columns:
                # These are float values (e.g., pKd), but may be NaN
                f.register(df[col].values, "unstructured", name=col, dtype_save="float32", dtype_load="float32")
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


class DTIDataset(h5torch.Dataset):
    """
    A dataset class for drug-target interaction data stored in h5torch format.
    
    This dataset extends h5torch.Dataset to additionally fetch interaction-specific 
    values (Y_pKd, Y_pKi, Y_KIBA) from the 'unstructured' group of the h5torch file.
    
    Attributes:
        additional_y_cols: List of column names in the 'unstructured' group to fetch
            during __getitem__. These are typically continuous interaction values.
    """
    
    def __init__(
        self, 
        filename: str, 
        additional_y_cols: List[str] = ["Y_pKd", "Y_pKi", "Y_KIBA"],
        subset: Optional[Union[np.ndarray, str, Tuple[str, str]]] = None
    ):
        """
        Initialize the DTIDataset.
        
        Args:
            filename: Path to the h5torch file
            additional_y_cols: List of column names in the 'unstructured' group to fetch
                during __getitem__. These are typically continuous interaction values.
            subset: Boolean mask of length n_interactions.
        """
        super().__init__(filename, subset=subset, sampling="coo", in_memory=True)
        self.additional_y_cols = additional_y_cols
        
        # Verify that all additional_y_cols exist in the unstructured group
        existing_cols = []
        for col in additional_y_cols:
            try:
                test = self.f[f"unstructured/{col}"][:]
                existing_cols.append(col)
            except KeyError:
                print(f"Warning: Column '{col}' not found in unstructured group. It will be ignored.")
        
        self.additional_y_cols = existing_cols
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.
        
        This overrides the parent method to additionally fetch the interaction-specific
        values from the 'unstructured' group.
        
        Args:
            idx: Index of the item to fetch
            
        Returns:
            Dict containing the central data (typically a drug-target pair), 
            aligned data (drug and target features), 
            and the additional interaction values.
        """
        # Get the base item from the parent class
        item = super().__getitem__(idx)
        
        # Add the additional interaction values
        for col in self.additional_y_cols:
            item[f'unstructured/{col}'] = self.f[f'unstructured/{col}'][idx]
            
        return item


def load_h5torch_DTI(
    filename: str = "data/processed/data.h5torch",
    setting: Literal["split_rand", "split_cold"] = "split_rand",
    split: Literal["train", "valid", "test"] = "train",
    datasets: Optional[List[str]] = None,
    additional_y_cols: List[str] = ["Y_pKd", "Y_pKi", "Y_KIBA"],
    transform: Optional[Any] = None,
) -> DTIDataset:
    """
    Load a DTIDataset from an h5torch file with specific filtering options.
    
    This helper function creates a subset of the original dataset based on:
    1. The split setting (random or cold-split)
    2. The specific split (train, valid, test)
    3. Optionally, specific source datasets (DAVIS, Metz, etc.)
    
    Args:
        filename: Path to the h5torch file
        setting: Which split setting to use ('split_rand' or 'split_cold')
        split: Which split to load ('train', 'valid', or 'test')
        datasets: Optional list of dataset sources to include (e.g. ['in_DAVIS', 'in_Metz']).
                 If None, all interactions are included regardless of source.
        additional_y_cols: List of additional Y columns to include in the dataset
        transform: Optional transform to apply to the data
        
    Returns:
        A DTIDataset object containing the specified subset of data
    """
    with h5torch.File(filename, 'r') as f:

        subset_mask = np.zeros(f["central/indices"][:].shape[-1], dtype=bool)

        # 1. Account for split setting
        split_key = f"unstructured/{setting}"
        if split_key not in f:
            raise ValueError(f"Split setting '{setting}' not found in file")
        values = np.array(
            [s.decode('utf-8') for s in f[split_key][:]])
        subset_mask[values == split] = True
        
        # 2. If specific datasets are requested, further filter
        if datasets is not None and len(datasets) > 0:
            # Create a dataset mask of the same size as the full dataset
            dataset_mask = np.zeros(len(subset_mask), dtype=bool)
            
            # For each dataset, update the mask (OR operation)
            for ds in datasets:
                ds_key = f"unstructured/{ds}"
                if ds_key not in f:
                    print(f"Warning: Dataset '{ds}' not found in file, ignoring")
                    continue
                    
                # Get the dataset flag values for the entire dataset
                ds_values = f[ds_key][:]
                # Update the dataset mask for the entire dataset
                dataset_mask = dataset_mask | ds_values
                
            # Apply the dataset mask to our split indices (AND operation)
            subset_mask = subset_mask & dataset_mask
                
    # Create and return the dataset with the filtered indices
    return DTIDataset(
        filename=filename,
        additional_y_cols=additional_y_cols,
        subset=subset_mask
    )