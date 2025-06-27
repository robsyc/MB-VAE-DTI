"""
Factory functions for creating h5torch-compatible HDF5 files from processed data.
"""

import h5py
import h5torch
import numpy as np
import logging
from pathlib import Path
from typing import Literal, List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import math
import pandas as pd

# Setup logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default batch size for reading/writing large datasets
DEFAULT_BATCH_SIZE = 1024

# --- Helper Functions ---

def _load_hdf5_file(h5_path: Path) -> Dict[str, Any]:
    """
    Opens an HDF5 file, extracts key metadata, and returns it with the file handle.

    Expected structure:
    - Root attributes: 'entity_type', 'representation_type'
    - Datasets: 'ids', 'data' (optional, for string representation)
    - Group: 'embeddings' containing feature datasets (e.g., 'FP-Morgan', 'EMB-ESM')

    Args:
        h5_path: Path to the input HDF5 file.

    Returns:
        A dictionary containing:
        - 'file_handle': The open h5py.File object.
        - 'num_items': Number of entities.
        - 'entity_type': Type of entity (e.g., 'drug', 'target').
        - 'ids': List of entity IDs.
        - 'repr_info': Dictionary with 'name' (e.g. 'SMILES'), 'path' ('data'), 'dtype'. None if no representation.
        - 'features': Dictionary where keys are feature names and values are dicts with 'path', 'shape', 'dtype'.

    Raises:
        FileNotFoundError: If the h5_path does not exist.
        KeyError: If essential datasets like 'ids' are missing.
        ValueError: If inconsistencies are found (e.g., feature length mismatch).
        IOError: If the file cannot be opened or read.
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    file_handle = None
    try:
        file_handle = h5py.File(h5_path, 'r')
        logger.debug(f"Opened HDF5 file: {h5_path}")

        # --- Extract Root Attributes ---
        entity_type = file_handle.attrs.get('entity_type')
        repr_type = file_handle.attrs.get('representation_type') # e.g., SMILES, AA, DNA
        if not entity_type:
            logger.warning(f"Missing 'entity_type' attribute in {h5_path.name}")
        if not repr_type:
             logger.warning(f"Missing 'representation_type' attribute in {h5_path.name}")

        # --- Extract IDs and Determine num_items ---
        if 'ids' not in file_handle:
            # Attempt to infer from 'data' if 'ids' is missing
            if 'data' in file_handle:
                 num_items = len(file_handle['data'])
                 logger.warning(f"Missing 'ids' dataset in {h5_path.name}. Inferred num_items={num_items} from 'data'. Generating sequential IDs.")
                 # Create sequential IDs as strings if 'ids' is missing
                 ids_list = [str(i) for i in range(num_items)]
            else:
                # Cannot determine num_items if both 'ids' and 'data' are missing
                raise KeyError(f"Missing 'ids' dataset and cannot infer size from 'data' in {h5_path}")
        else:
            ids_raw = file_handle['ids'][:]
            ids_list = [id_val.decode('utf-8') if isinstance(id_val, bytes) else str(id_val) for id_val in ids_raw]
            num_items = len(ids_list)
        
        logger.debug(f"Found {num_items} items with entity_type '{entity_type}' in {h5_path.name}")


        # --- Get Representation Info ---
        repr_info = None
        if 'data' in file_handle:
            if repr_type:
                 repr_dataset = file_handle['data']
                 if len(repr_dataset) != num_items:
                     raise ValueError(f"Representation 'data' in {h5_path.name} has {len(repr_dataset)} items, expected {num_items}.")
                 repr_info = {
                    'name': repr_type,
                    'path': 'data',
                    'dtype': repr_dataset.dtype
                 }
                 logger.debug(f"Found representation '{repr_type}' at path 'data'")
            else:
                logger.warning(f"Found 'data' dataset in {h5_path.name} but no 'representation_type' attribute. Ignoring 'data'.")
        elif repr_type:
             logger.warning(f"Found 'representation_type' attribute ('{repr_type}') but no 'data' dataset in {h5_path.name}.")


        # --- Get Feature Info (Embeddings/Fingerprints) ---
        features = {}
        if 'embeddings' in file_handle:
            embeddings_group = file_handle['embeddings']
            for name, dset in embeddings_group.items():
                if not isinstance(dset, h5py.Dataset):
                     logger.warning(f"Found non-dataset object '{name}' in embeddings group of {h5_path.name}. Skipping.")
                     continue

                if len(dset) != num_items:
                    raise ValueError(
                        f"Feature '{name}' in {h5_path.name} has {len(dset)} items, expected {num_items}."
                    )

                feature_path = f'embeddings/{name}'
                features[name] = {
                    'path': feature_path,
                    'shape': dset.shape,
                    'dtype': dset.dtype
                }
                logger.debug(f"Found feature '{name}' (Shape: {dset.shape}, Dtype: {dset.dtype}) at path '{feature_path}'")
        else:
            logger.warning(f"No 'embeddings' group found in {h5_path.name}")


        # --- Structure the return dictionary ---
        result = {
            'file_handle': file_handle,
            'num_items': num_items,
            'entity_type': entity_type,
            'ids': ids_list,
            'repr_info': repr_info,
            'features': features
        }
        return result

    except Exception as e:
        # If any error occurs, try to close the handle if it was opened
        if file_handle:
            try:
                file_handle.close()
                logger.debug(f"Closed HDF5 file {h5_path.name} due to error.")
            except Exception as close_err:
                 logger.error(f"Error closing HDF5 file {h5_path.name} after initial error: {close_err}")
        # Re-raise the original exception
        logger.error(f"Failed to load HDF5 metadata from {h5_path}: {e}")
        if isinstance(e, (FileNotFoundError, KeyError, ValueError, IOError)):
             raise # Re-raise specific expected errors
        else:
             raise IOError(f"An unexpected error occurred while reading {h5_path}: {e}") from e # Wrap unexpected errors

def _load_entity_h5torch(h5_paths: List[Path]) -> Dict[str, Any]:
    """
    Loads metadata from multiple HDF5 files for the same entity type, validates consistency,
    and merges the information, keeping track of data sources.

    Args:
        h5_paths: List of paths to the input HDF5 files.

    Returns:
        A dictionary containing merged metadata:
        - 'num_items': Number of entities (validated to be consistent).
        - 'entity_type': Entity type from the first file.
        - 'ids': List of entity IDs (validated to be consistent).
        - 'repr_sources': Dict mapping repr name to (file_handle, path, dtype).
        - 'feature_sources': Dict mapping feature name to (file_handle, path, shape, dtype).
        - 'file_handles': List of all opened h5py.File handles (caller must close).

    Raises:
        ValueError: If input list is empty, or if inconsistencies are found (num_items mismatch, ids mismatch).
        FileNotFoundError, KeyError, IOError: Propagated from _load_hdf5_file.
    """
    if not h5_paths:
        raise ValueError("Input HDF5 path list cannot be empty.")

    all_handles = []
    merged_info = {
        "num_items": None,
        "entity_type": None,
        "ids": None,
        "repr_sources": {},
        "feature_sources": {},
        "file_handles": all_handles
    }

    try:
        for i, h5_path in enumerate(h5_paths):
            logger.debug(f"Loading entity metadata from file {i+1}/{len(h5_paths)}: {h5_path.name}")
            file_info = _load_hdf5_file(h5_path)
            handle = file_info['file_handle']
            all_handles.append(handle)

            # --- Validation against the first file's info ---
            if i == 0:
                merged_info["num_items"] = file_info['num_items']
                merged_info["entity_type"] = file_info['entity_type']
                merged_info["ids"] = file_info['ids']
            else:
                # Check num_items consistency
                if file_info['num_items'] != merged_info["num_items"]:
                    raise ValueError(
                        f"Number of items mismatch: {h5_path.name} ({file_info['num_items']}) "
                        f"!= {h5_paths[0].name} ({merged_info['num_items']})."
                    )
                # Check ID consistency
                if file_info['ids'] != merged_info["ids"]:
                    # Provide more detail on mismatch if helpful (optional)
                    # diff = [(idx, id1, id2) for idx, (id1, id2) in enumerate(zip(merged_info["ids"], file_info['ids'])) if id1 != id2]
                    # logger.error(f"First ID differences: {diff[:5]}")
                    raise ValueError(f"ID list mismatch between {h5_path.name} and {h5_paths[0].name}.")

                # Warn if entity type differs
                if file_info['entity_type'] != merged_info["entity_type"]:
                    logger.warning(
                        f"Entity type mismatch: {h5_path.name} ('{file_info['entity_type']}') "
                        f"vs {h5_paths[0].name} ('{merged_info['entity_type']}'). Using first."
                    )

            # --- Merge Representation Info ---
            if file_info['repr_info']:
                repr_name = file_info['repr_info']['name']
                if repr_name in merged_info['repr_sources']:
                    logger.warning(f"Duplicate representation '{repr_name}' found in {h5_path.name}. Using source from first encountered file.")
                else:
                    merged_info['repr_sources'][repr_name] = (
                        handle,
                        file_info['repr_info']['path'],
                        file_info['repr_info']['dtype']
                    )
                    logger.debug(f"Added representation source: '{repr_name}' from {h5_path.name}")

            # --- Merge Feature Info ---
            for feat_name, feat_details in file_info['features'].items():
                if feat_name in merged_info['feature_sources']:
                    logger.warning(f"Duplicate feature '{feat_name}' found in {h5_path.name}. Using source from first encountered file.")
                else:
                    merged_info['feature_sources'][feat_name] = (
                        handle,
                        feat_details['path'],
                        feat_details['shape'],
                        feat_details['dtype']
                    )
                    logger.debug(f"Added feature source: '{feat_name}' from {h5_path.name}")

        if merged_info["num_items"] is None:
             raise ValueError("Could not determine number of items after processing input files.")
        
        logger.info(f"Successfully loaded and merged metadata for {merged_info['num_items']} entities ('{merged_info['entity_type']}') "
                    f"from {len(h5_paths)} file(s).")
        logger.info(f"Found representations: {list(merged_info['repr_sources'].keys())}")
        logger.info(f"Found features: {list(merged_info['feature_sources'].keys())}")

        return merged_info

    except Exception as e:
        # Ensure all opened handles are closed on error
        logger.error(f"Error during entity HDF5 loading/merging: {e}")
        for handle in all_handles:
            try:
                handle.close()
            except Exception as close_err:
                 logger.error(f"Error closing handle for {handle.filename} during cleanup: {close_err}")
        raise # Re-throw the original error

def _print_dataset_info(name: str, dset: h5py.Dataset):
    """Helper function to print standardized dataset info."""
    print(f"    - Name: {name}")
    print(f"      - Path: {dset.name}")
    # Handle shape display for different types
    shape_str = str(dset.shape) if hasattr(dset, 'shape') else 'N/A'
    if dset.dtype.kind in ('O', 'S', 'U') or h5py.check_vlen_dtype(dset.dtype):
         shape_str = f"Length: {len(dset)}" # More informative for strings/vlen
    print(f"      - Shape/Length: {shape_str}")
    print(f"      - Saved Dtype: {dset.dtype}")
    dtype_load = dset.attrs.get('dtype_load')
    if dtype_load:
        print(f"      - Load Dtype: {dtype_load}")

def _register_aligned_entity_data(
    h5_out: h5torch.File,
    axis: int,
    entity_ids: List[str],
    entity_info: Optional[Dict[str, Any]],
    h5_id_to_source_idx: Dict[str, int],
    num_entities: int,
    batch_size: int
) -> set:
    """
    Helper function to register aligned features and representations for a given axis (drug or target).

    Reads data from external HDF5 sources provided in entity_info and aligns it to the order specified by entity_ids. 
    Uses batch processing on the numerical features, bulk processing on the string representations.

    Args:
        h5_out: The open h5torch.File object to write to.
        axis: The axis index (0 for drugs, 1 for targets in case of COO DTI dataset, otherwise just 0).
        entity_ids: Ordered list of entity IDs for the current dataset.
        entity_info: Merged metadata dictionary from _load_entity_h5torch (contains sources).
        h5_id_to_source_idx: Mapping from source HDF5 entity ID to its index.
        num_entities: Total number of unique entities for this axis in the DTI dataset.
        batch_size: Batch size for reading/writing.

    Returns:
        A set containing the names of representations registered from external HDF5 files.
    """
    registered_repr_names = set()
    if not entity_info:
        logger.warning(f"No external entity info provided for axis {axis}. Skipping external data registration.")
        return registered_repr_names

    # 1. Process features (Embeddings, Fingerprints)
    for feat_name, (source_handle, path, shape, dtype) in entity_info['feature_sources'].items():
        logger.info(f"Registering external feature for axis {axis}: {feat_name} (Shape: {shape}, Dtype: {dtype})")
        feature_dataset = source_handle[path]

        feat_dim = shape[1:] if len(shape) > 1 else ()
        is_fingerprint = feat_name.startswith("FP-")

        # Determine save/load dtypes
        if is_fingerprint:
            dtype_save = 'uint8' # bool also takes up 8 bits (1 byte) so we cannot go smaller than this
            dtype_load = 'float32'
            output_dtype = np.dtype(np.uint8)
        else:
            dtype_save = None # Use original dtype (typically float32)
            dtype_load = 'float32'
            output_dtype = np.dtype(dtype)

        # --- Process First Batch ---
        first_batch_size = min(batch_size, num_entities)
        first_batch_aligned = np.empty((first_batch_size,) + feat_dim, dtype=output_dtype)
        logger.debug(f"Preparing first batch for axis {axis} feature '{feat_name}' (size: {first_batch_size})...")

        missing_count = 0
        for dti_idx in range(first_batch_size):
            h5_id = entity_ids[dti_idx]
            source_idx = h5_id_to_source_idx.get(h5_id)

            if source_idx is not None:
                source_data = feature_dataset[source_idx]
                if is_fingerprint:
                    first_batch_aligned[dti_idx] = (source_data > 0).astype(output_dtype)
                else:
                    first_batch_aligned[dti_idx] = source_data
            else:
                missing_count += 1
                # Handle missing: Fill with zeros
                fill_value = 0 if output_dtype.kind in 'iufc' else False
                first_batch_aligned[dti_idx] = fill_value

        if missing_count > 0:
             logger.warning(f"Axis {axis} Feature '{feat_name}': {missing_count} DTI entity IDs not found in source HDF5 file(s). Filled with default value.")

        # Register with the first batch
        h5_out.register(
            first_batch_aligned,
            axis=axis,
            name=feat_name,
            mode='N-D',
            dtype_save=dtype_save,
            dtype_load=dtype_load,
            length=num_entities
        )
        logger.info(f"Registered aligned[{axis}] '{feat_name}' (Target Shape: {(num_entities,) + feat_dim}, Saved: {dtype_save or dtype}, Loaded: {dtype_load or dtype})")

        # --- Append Remaining Batches ---
        num_batches = math.ceil(num_entities / batch_size)
        if num_batches > 1:
            logger.info(f"Appending remaining {num_entities - first_batch_size} items for axis {axis} feature '{feat_name}'...")
            for i in tqdm(range(1, num_batches), desc=f"Appending {feat_name} (Axis {axis})", unit="batch"):
                start_dti_idx = i * batch_size
                end_dti_idx = min(start_dti_idx + batch_size, num_entities)
                if start_dti_idx >= end_dti_idx: continue

                batch_len = end_dti_idx - start_dti_idx
                current_batch_aligned = np.empty((batch_len,) + feat_dim, dtype=output_dtype)

                for batch_idx, dti_idx in enumerate(range(start_dti_idx, end_dti_idx)):
                    h5_id = entity_ids[dti_idx]
                    source_idx = h5_id_to_source_idx.get(h5_id)

                    if source_idx is not None:
                        source_data = feature_dataset[source_idx]
                        if is_fingerprint:
                            current_batch_aligned[batch_idx] = (source_data > 0).astype(output_dtype)
                        else:
                            current_batch_aligned[batch_idx] = source_data
                    else:
                        # Handle missing (already logged warning for first batch)
                        fill_value = 0 if output_dtype.kind in 'iufc' else False
                        current_batch_aligned[batch_idx] = fill_value

                h5_out.append(current_batch_aligned, f"{axis}/{feat_name}")

    # 2. Process representations (SMILES, AA, DNA, etc.)
    for repr_name, (source_handle, path, dtype) in entity_info['repr_sources'].items():
        logger.info(f"Registering external representation for axis {axis}: {repr_name}")
        repr_dataset = source_handle[path]

        # For string data, collect all at once to avoid h5py dtype issues with length pre-specification
        logger.debug(f"Collecting all string data for axis {axis} representation '{repr_name}'...")
        missing_count = 0
        
        # Pre-allocate numpy array of strings
        all_repr_array = np.empty(num_entities, dtype=object)
        
        for dti_idx in range(num_entities):
            h5_id = entity_ids[dti_idx]
            source_idx = h5_id_to_source_idx.get(h5_id)

            if source_idx is not None:
                repr_val = repr_dataset[source_idx]
                # Ensure value is a string (decoding bytes if necessary)
                if isinstance(repr_val, bytes):
                    all_repr_array[dti_idx] = repr_val.decode('utf-8', errors='replace')
                else:
                    all_repr_array[dti_idx] = str(repr_val)
            else:
                missing_count += 1
                all_repr_array[dti_idx] = ""  # Fill missing strings with empty string

        if missing_count > 0:
             logger.warning(f"Axis {axis} Representation '{repr_name}': {missing_count} DTI entity IDs not found in source HDF5 file(s). Filled with empty string.")
        
        h5_out.register(
            all_repr_array,
            axis=axis,
            name=repr_name,
            mode='N-D',
            dtype_save='bytes',
            dtype_load='str',
        )
        logger.info(f"Registered aligned[{axis}] '{repr_name}' (Length: {len(all_repr_array)})")
        registered_repr_names.add(repr_name)

    return registered_repr_names

# --- Factory Functions ---

def _load_dti_df(
    dti_df: pd.DataFrame,
    drug_id_col: str = "Drug_ID",
    target_id_col: str = "Target_ID",
    interaction_col: str = "Y",
    drug_feature_cols: List[str] = ["Drug_InChIKey", "Drug_SMILES"],
    target_feature_cols: List[str] = [
        "Target_UniProt_ID", "Target_Gene_name",
        "Target_RefSeq_ID", "Target_AA", "Target_DNA"],
    additional_y_cols: Optional[List[str]] = ["Y_pKd", "Y_pKi", "Y_KIBA"],
    provenance_cols: Optional[List[str]] = ["in_DAVIS", "in_BindingDB_Kd", "in_BindingDB_Ki", "in_Metz", "in_KIBA"],
) -> Dict[str, Any]:
    """
    Processes a DTI DataFrame: extracts unique entities and their DataFrame features,
    creates ID-to-index mappings, prepares COO interaction data, and extracts 
    interaction-level unstructured data.

    Args:
        dti_df: DataFrame with DTI interactions.
        drug_id_col, target_id_col, interaction_col: Key column names in dti_df.
        drug_feature_cols, target_feature_cols: DataFrame columns to save as aligned features/representations.
        additional_y_cols, provenance_cols: DataFrame columns for unstructured data.

    Returns:
        Dict containing:
        - 'dti_drug_ids', 'dti_target_ids': Ordered lists of unique IDs found in the DataFrame.
        - 'drug_id_to_idx', 'target_id_to_idx': Mappings from ID string to DTI-specific integer index.
        - 'drug_features_df': Dict {feature_name: np.array} for drug features from df, ordered by dti_drug_ids.
        - 'target_features_df': Dict {feature_name: np.array} for target features from df, ordered by dti_target_ids.
        - 'coo_data': Tuple (indices, values, shape) for the central COO object.
        - 'unstructured_data': Dict {col_name: np.array} for interaction-level data (splits, Y_cont, provenance).
    """
    logger.info(f"Processing DTI DataFrame with {len(dti_df)} interactions...")
    df = dti_df.copy()

    # --- Input Validation ---
    required_cols = [drug_id_col, target_id_col, interaction_col]
    missing_req = [col for col in required_cols if col not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns in DataFrame: {missing_req}")

    all_feature_cols = drug_feature_cols + target_feature_cols
    all_unstructured_cols = (additional_y_cols or []) + (provenance_cols or [])

    optional_cols = all_feature_cols + all_unstructured_cols
    missing_optional = [col for col in optional_cols if col not in df.columns]
    if missing_optional:
        logger.warning(f"Optional columns not found in DataFrame and will be skipped: {missing_optional}")
        # Filter out missing columns from lists to avoid errors later
        drug_feature_cols = [c for c in drug_feature_cols if c in df.columns]
        target_feature_cols = [c for c in target_feature_cols if c in df.columns]
        additional_y_cols = [c for c in (additional_y_cols or []) if c in df.columns]
        provenance_cols = [c for c in (provenance_cols or []) if c in df.columns]

    # --- Identify Unique Entities and Create Mappings ---
    unique_drugs = pd.unique(df[drug_id_col])
    unique_targets = pd.unique(df[target_id_col])
    num_drugs = len(unique_drugs)
    num_targets = len(unique_targets)
    logger.info(f"Found {num_drugs} unique drugs and {num_targets} unique targets in the DataFrame.")

    # Create mappings and ordered lists
    drug_id_to_idx = {drug: i for i, drug in enumerate(unique_drugs)}
    target_id_to_idx = {target: i for i, target in enumerate(unique_targets)}
    dti_drug_ids = list(unique_drugs) # Already unique, preserve order for mapping
    dti_target_ids = list(unique_targets)

    # --- Extract Unique DataFrame Features per Entity ---
    drug_features_df = {}
    logger.info("Extracting unique drug features from DataFrame...")
    if drug_feature_cols:
        # Use drop_duplicates and sort to get features aligned with unique_drugs order
        drug_df_unique = df[[drug_id_col] + drug_feature_cols].drop_duplicates(subset=[drug_id_col])
        # Reindex based on the unique_drugs order to ensure alignment
        drug_df_unique = drug_df_unique.set_index(drug_id_col).reindex(unique_drugs).reset_index()
        for col in drug_feature_cols:
            feature_values = drug_df_unique[col].values
            # Fill NaN with empty string specifically for columns saved as strings
            feature_values_filled = pd.Series(feature_values).fillna("").values
            drug_features_df[col] = feature_values_filled
            logger.debug(f"  Extracted drug feature '{col}' (Length: {len(drug_features_df[col])})")
            if pd.isna(feature_values).any(): # Check original values for logging
                 logger.warning(f"  NaNs found in original drug feature '{col}', replaced with empty strings.")

    target_features_df = {}
    logger.info("Extracting unique target features from DataFrame...")
    if target_feature_cols:
        target_df_unique = df[[target_id_col] + target_feature_cols].drop_duplicates(subset=[target_id_col])
        target_df_unique = target_df_unique.set_index(target_id_col).reindex(unique_targets).reset_index()
        for col in target_feature_cols:
            feature_values = target_df_unique[col].values
            # Fill NaN with empty string specifically for columns saved as strings
            feature_values_filled = pd.Series(feature_values).fillna("").values
            target_features_df[col] = feature_values_filled
            logger.debug(f"  Extracted target feature '{col}' (Length: {len(target_features_df[col])})")
            if pd.isna(feature_values).any(): # Check original values for logging
                logger.warning(f"  NaNs found in original target feature '{col}', replaced with empty strings.")

    # --- Map DataFrame Rows and Sort for COO Consistency ---
    logger.info("Mapping interaction indices and sorting DataFrame...")
    df["Drug_dti_idx"] = df[drug_id_col].map(drug_id_to_idx)
    df["Target_dti_idx"] = df[target_id_col].map(target_id_to_idx)

    # Check for mapping errors (shouldn't happen if unique_drugs/targets derived from df)
    if df["Drug_dti_idx"].isnull().any() or df["Target_dti_idx"].isnull().any():
        logger.error("Internal error: Failed to map all drug/target IDs to indices.")
        # Handle appropriately, maybe raise error or filter NaNs
        df.dropna(subset=["Drug_dti_idx", "Target_dti_idx"], inplace=True)
        logger.warning("Removed rows with unmappable drug/target indices.")

    df["Drug_dti_idx"] = df["Drug_dti_idx"].astype(int)
    df["Target_dti_idx"] = df["Target_dti_idx"].astype(int)

    # Sort by drug index, then target index (CRITICAL for COO <-> unstructured alignment)
    df.sort_values(["Drug_dti_idx", "Target_dti_idx"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    num_interactions = len(df)
    logger.info(f"DataFrame sorted, {num_interactions} interactions remain.")

    # --- Prepare COO Data ---
    logger.info("Preparing COO interaction data...")
    coo_indices = df[["Drug_dti_idx", "Target_dti_idx"]].values.T # Shape (2, n_interactions)
    coo_values = df[interaction_col].values # Let h5torch handle dtype later
    coo_shape = (num_drugs, num_targets)
    logger.info(f"COO Matrix: Shape={coo_shape}, NNZ={len(coo_values)}")

    coo_data = (coo_indices, coo_values, coo_shape)

    # --- Extract Unstructured Data ---
    logger.info("Extracting unstructured interaction-level data...")
    unstructured_data = {}
    
    # Check for existing split columns
    split_cols_existing = [col for col in df.columns if col.startswith('split_')]
    all_unstructured_cols_to_extract = split_cols_existing + (additional_y_cols or []) + (provenance_cols or [])

    for col in all_unstructured_cols_to_extract:
        if col in df.columns:
            col_data = df[col].values
            if len(col_data) != num_interactions:
                 # This check should ideally not fail if logic is correct
                 logger.error(f"Length mismatch for unstructured column '{col}'. Expected {num_interactions}, got {len(col_data)}. Skipping.")
                 continue
            unstructured_data[col] = col_data
            logger.debug(f"  Extracted unstructured data '{col}' (Length: {len(col_data)})")
        else:
            # Should only happen if column wasn't present initially
            logger.warning(f"Unstructured column '{col}' not found in final DataFrame. Skipping.")

    # --- Assemble Result ---
    result = {
        'dti_drug_ids': dti_drug_ids,
        'dti_target_ids': dti_target_ids,
        'drug_id_to_idx': drug_id_to_idx,
        'target_id_to_idx': target_id_to_idx,
        'drug_features_df': drug_features_df,
        'target_features_df': target_features_df,
        'coo_data': coo_data,
        'unstructured_data': unstructured_data
    }

    logger.info("Finished processing DTI DataFrame.")
    return result

def create_dti_h5torch(
    drug_input_h5_paths: List[Path],
    target_input_h5_paths: List[Path],
    output_h5_path: Path,
    df: pd.DataFrame,
    drug_id_col: str = "Drug_ID",
    target_id_col: str = "Target_ID",
    interaction_col: str = "Y",
    drug_feature_cols: List[str] = ["Drug_InChIKey", "Drug_SMILES"],
    target_feature_cols: List[str] = [
        "Target_UniProt_ID", "Target_Gene_name",
        "Target_RefSeq_ID", "Target_AA", "Target_DNA"],
    additional_y_cols: Optional[List[str]] = ["Y_pKd", "Y_pKi", "Y_KIBA"],
    provenance_cols: Optional[List[str]] = ["in_DAVIS", "in_BindingDB_Kd", "in_BindingDB_Ki", "in_Metz", "in_KIBA"],
    batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """
    Creates a consolidated h5torch file for Drug-Target Interaction (DTI) data.

    Reads DTI interactions and DataFrame-specific features from `df`. Reads
    additional entity features (embeddings, fingerprints) and representations
    (SMILES, AA, DNA) from external HDF5 files specified in `drug_input_h5_paths`
    and `target_input_h5_paths`. It merges and structures the data for efficient
    COO sampling in h5torch.

    Structure:
    - Central Object: Sparse COO matrix of interactions (Y).
    - Aligned Axis 0 (Drugs): Features/Representations for unique drugs in DTI.
    - Aligned Axis 1 (Targets): Features/Representations for unique targets in DTI.
    - Unstructured Data: Interaction-level data (splits, provenance, continuous Y).

    Args:
        drug_input_h5_paths: List of paths to HDF5 files with drug features/representations.
        target_input_h5_paths: List of paths to HDF5 files with target features/representations.
        output_h5_path: Path for the output dti.h5torch file.
        df: DataFrame with DTI interactions.
        drug_id_col, target_id_col, interaction_col: Key column names in df.
        drug_feature_cols, target_feature_cols: DataFrame columns to save as aligned features/representations.
        additional_y_cols, provenance_cols: DataFrame columns for unstructured data.
        batch_size: Batch size for reading/writing HDF5 features.
    """
    logger.info(f"Starting creation of DTI h5torch file: {output_h5_path}")
    drug_info = None
    target_info = None
    
    try:
        # Step 1: Process the DTI DataFrame to prepare data structures
        logger.info("Processing DTI DataFrame...")
        dti_data = _load_dti_df(
            dti_df=df,
            drug_id_col=drug_id_col,
            target_id_col=target_id_col,
            interaction_col=interaction_col,
            drug_feature_cols=drug_feature_cols,
            target_feature_cols=target_feature_cols,
            additional_y_cols=additional_y_cols,
            provenance_cols=provenance_cols,
        )

        # Extract components from dti_data
        dti_drug_ids = dti_data['dti_drug_ids']
        dti_target_ids = dti_data['dti_target_ids']
        drug_id_to_idx = dti_data['drug_id_to_idx']
        target_id_to_idx = dti_data['target_id_to_idx']
        drug_features_df = dti_data['drug_features_df']
        target_features_df = dti_data['target_features_df']
        coo_indices, coo_values, coo_shape = dti_data['coo_data']
        unstructured_data = dti_data['unstructured_data']

        num_drugs = len(dti_drug_ids)
        num_targets = len(dti_target_ids)
        num_interactions = len(coo_values)

        logger.info(f"DTI dataset composition: {num_drugs} drugs × {num_targets} targets = {num_interactions} / {num_drugs * num_targets} interactions (sparsity: {num_interactions / (num_drugs * num_targets):.2%})")

        # Step 2: Load drug features from external HDF5 files
        if drug_input_h5_paths:
            logger.info(f"Loading drug features from {len(drug_input_h5_paths)} HDF5 file(s)...")
            drug_info = _load_entity_h5torch(drug_input_h5_paths)

            # Validate that the drug IDs match what we have in the DTI DataFrame
            # Note: We proceed even with mismatch, but _register_aligned_entity_data will log warnings
            if set(dti_drug_ids) != set(drug_info['ids']):
                logger.warning(
                    f"Drug ID set mismatch between DTI DataFrame and source HDF5 files. "
                    f"DTI contains {len(dti_drug_ids)} unique drugs, sources contain {len(drug_info['ids'])}."
                )
            logger.info(f"Loaded drug features: {list(drug_info.get('feature_sources', {}).keys())}")
            logger.info(f"Loaded drug representations: {list(drug_info.get('repr_sources', {}).keys())}")

        # Step 3: Load target features from external HDF5 files
        if target_input_h5_paths:
            logger.info(f"Loading target features from {len(target_input_h5_paths)} HDF5 file(s)...")
            target_info = _load_entity_h5torch(target_input_h5_paths)

            # Validate that the target IDs match what we have in the DTI DataFrame
            if set(dti_target_ids) != set(target_info['ids']):
                logger.warning(
                    f"Target ID set mismatch between DTI DataFrame and source HDF5 files. "
                    f"DTI contains {len(dti_target_ids)} unique targets, sources contain {len(target_info['ids'])}."
                )
            logger.info(f"Loaded target features: {list(target_info.get('feature_sources', {}).keys())}")
            logger.info(f"Loaded target representations: {list(target_info.get('repr_sources', {}).keys())}")

        # Precompute source ID to source index mappings for faster lookups
        drug_h5_id_to_source_idx = {h5_id: idx for idx, h5_id in enumerate(drug_info['ids'])} if drug_info else {}
        target_h5_id_to_source_idx = {h5_id: idx for idx, h5_id in enumerate(target_info['ids'])} if target_info else {}

        # Step 4: Create the output h5torch file
        logger.info(f"Creating output h5torch file: {output_h5_path}")
        with h5torch.File(str(output_h5_path), 'w') as h5_out:
            # Add file attributes
            h5_out.attrs['n_drugs'] = num_drugs
            h5_out.attrs['n_targets'] = num_targets
            h5_out.attrs['n_interactions'] = num_interactions
            h5_out.attrs['sparsity'] = num_interactions / (num_drugs * num_targets) if (num_drugs * num_targets) > 0 else 0
            h5_out.attrs['created_at'] = pd.Timestamp.now().isoformat()

            # Step 5: Register the central COO interaction matrix
            logger.info(f"Registering central COO interaction matrix (Shape: {coo_shape}, NNZ: {num_interactions})...")
            central_object = (coo_indices, coo_values, coo_shape)
            h5_out.register(
                central_object,
                "central",
                mode="coo",
                dtype_save="bool",  # Save interaction values as boolean
                dtype_load="float32" # Load interaction values as float32
            )

            # Step 6: Register drug features (Axis 0)
            logger.info("Registering drug features (Axis 0)...")

            # 6.1: Register drug IDs (essential reference)
            h5_out.register(
                np.array(dti_drug_ids),
                axis=0,
                name=drug_id_col, # Use the original column name
                mode='N-D',
                dtype_save="bytes", 
                dtype_load="str"
            )

            # 6.2: Register external drug features and representations using the helper
            drug_registered_reprs = _register_aligned_entity_data(
                h5_out=h5_out,
                axis=0,
                entity_ids=dti_drug_ids,
                entity_info=drug_info,
                h5_id_to_source_idx=drug_h5_id_to_source_idx,
                num_entities=num_drugs,
                batch_size=batch_size
            )

            # 6.3: Register DataFrame-derived drug features/representations
            # Only register if the name wasn't already registered from external source
            for feat_name, feat_values in drug_features_df.items():
                # Determine if this column name corresponds to a representation name from HDF5
                # e.g., df column 'Drug_SMILES' might correspond to hdf5 repr 'SMILES'
                potential_repr_name = feat_name.split('_', 1)[-1] # Simple heuristic: "Drug_SMILES" -> "SMILES"
                is_potentially_redundant = potential_repr_name in drug_registered_reprs

                if is_potentially_redundant:
                    logger.info(f"Skipping registration of DataFrame column '{feat_name}' for axis 0 as representation '{potential_repr_name}' was already registered from HDF5 source.")
                    continue

                logger.info(f"Registering DataFrame-derived data for axis 0: {feat_name}")
                # Assume these are string-like identifiers or metadata
                # Don't use length parameter for strings to avoid h5py size issues
                h5_out.register(
                    feat_values,
                    axis=0,
                    name=feat_name,
                    mode='N-D',
                    dtype_save="bytes",
                    dtype_load="str"
                )

            # Step 7: Register target features (Axis 1)
            logger.info("Registering target features (Axis 1)...")

            # 7.1: Register target IDs
            h5_out.register(
                np.array(dti_target_ids),
                axis=1,
                name=target_id_col, # Use the original column name
                mode='N-D',
                dtype_save="bytes",
                dtype_load='str'
            )

            # 7.2: Register external target features and representations using the helper
            target_registered_reprs = _register_aligned_entity_data(
                h5_out=h5_out,
                axis=1,
                entity_ids=dti_target_ids,
                entity_info=target_info,
                h5_id_to_source_idx=target_h5_id_to_source_idx,
                num_entities=num_targets,
                batch_size=batch_size
            )

            # 7.3: Register DataFrame-derived target features/representations
            # Only register if the name wasn't already registered from external source
            for feat_name, feat_values in target_features_df.items():
                 # Determine if this column name corresponds to a representation name from HDF5
                 potential_repr_name = feat_name.split('_', 1)[-1] # Simple heuristic: "Target_AA" -> "AA"
                 is_potentially_redundant = potential_repr_name in target_registered_reprs

                 if is_potentially_redundant:
                     logger.info(f"Skipping registration of DataFrame column '{feat_name}' for axis 1 as representation '{potential_repr_name}' was already registered from HDF5 source.")
                     continue

                 logger.info(f"Registering DataFrame-derived data for axis 1: {feat_name}")
                 # Assume these are string-like identifiers or metadata
                 h5_out.register(
                     feat_values,
                     axis=1,
                     name=feat_name,
                     mode='N-D',
                     dtype_save="bytes",
                     dtype_load='str'
                 )

            # Step 8: Register unstructured data (interaction-level data)
            logger.info("Registering unstructured interaction data...")
            if not unstructured_data:
                 logger.warning("No unstructured data found in DTI DataFrame processing result.")
            for col_name, values in unstructured_data.items():
                # Ensure length matches number of interactions in COO object
                if len(values) != num_interactions:
                    logger.error(f"Length mismatch for unstructured column '{col_name}'. Expected {num_interactions}, got {len(values)}. Skipping.")
                    continue

                # Determine appropriate data type handling based on column type/name
                dtype_save = None
                dtype_load = None
                mode = 'N-D'

                if col_name.startswith('Y_'):
                    # Continuous Y values (e.g., Y_pKd, Y_pKi, Y_KIBA)
                    logger.info(f"Registering continuous Y column: {col_name}")
                    dtype_save = 'float32'
                    dtype_load = 'float32'
                elif col_name.startswith('in_'):
                    # Provenance boolean columns
                    logger.info(f"Registering provenance column: {col_name}")
                    dtype_save = 'bool'
                    dtype_load = 'bool'
                elif col_name.startswith('split_'):
                    # Split string columns
                    logger.info(f"Registering split column: {col_name}")
                    dtype_save = "bytes"
                    dtype_load = 'str'
                else:
                    # Default for any other unstructured data - try to infer
                    logger.info(f"Registering other unstructured column: {col_name}")
                    if pd.api.types.is_numeric_dtype(values):
                        # Check if integer or float
                        if pd.api.types.is_integer_dtype(values):
                            dtype_save = 'int64'
                            dtype_load = 'int64'
                        else: # Float
                            dtype_save = 'float32'
                            dtype_load = 'float32'
                    elif pd.api.types.is_string_dtype(values) or pd.api.types.is_object_dtype(values):
                        dtype_save = "bytes"
                        dtype_load = 'str'
                    elif pd.api.types.is_bool_dtype(values):
                         dtype_save = 'bool'
                         dtype_load = 'bool'

                # Register the dataset
                h5_out.register(
                    values,
                    axis='unstructured',
                    name=col_name,
                    mode=mode,
                    dtype_save=dtype_save,
                    dtype_load=dtype_load,
                    length=num_interactions if dtype_save != "bytes" else None
                )

        logger.info(f"Successfully created DTI h5torch file: {output_h5_path}")

    except Exception as e:
        # Log the error before re-raising
        logger.error(f"Failed to create DTI h5torch file: {e}", exc_info=True)
        raise

    finally:
        # Ensure all input file handles are closed
        if drug_info and drug_info.get('file_handles'):
            logger.debug(f"Closing {len(drug_info['file_handles'])} drug input HDF5 file handle(s)...")
            for handle in drug_info['file_handles']:
                try: 
                    handle.close()
                except Exception as e: 
                    logger.error(f"Error closing drug input file {handle.filename}: {e}")
        if target_info and target_info.get('file_handles'):
             logger.debug(f"Closing {len(target_info['file_handles'])} target input HDF5 file handle(s)...")
             for handle in target_info['file_handles']:
                 try: 
                     handle.close()
                 except Exception as e: 
                     logger.error(f"Error closing target input file {handle.filename}: {e}")
        logger.debug("Finished closing input files (if any were opened).")

def create_pretrain_h5torch(
    input_h5_paths: List[Path],
    output_h5_path: Path,
    add_split: bool = True,
    train_frac: float = 0.9,
    split_seed: int = 42,
    split_name: str = 'is_train',
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Creates a h5torch file for a single entity type (e.g., drugs or targets)
    suitable for pre-training.

    Merges representations (e.g., SMILES, AA) and features (e.g., FP-Morgan, EMB-ESM) 
    from multiple input HDF5 files into a single h5torch file.

    Structure:
    - Central: 'index' (simple numerical index)
    - Aligned Axis 0: Features (FP-*, EMB-*) and Representations (SMILES, AA, DNA, etc.)
    - Unstructured: Split information (boolean is_train)

    Args:
        input_h5_paths: List of paths to input HDF5 files.
        output_h5_path: Path where the output h5torch file will be created.
        add_split: Whether to add split column.
        train_frac: Fraction of data for training (remainder is validation).
        split_seed: Seed for splitting.
        split_name: Name for split column (default: 'is_train').
        batch_size: Number of items to process per batch during writing.
    """
    logger.info(f"Creating pretrain h5torch file '{output_h5_path}' from {len(input_h5_paths)} input file(s)...")

    merged_info = None
    try:
        # Load and merge metadata from all input files
        merged_info = _load_entity_h5torch(input_h5_paths)
        num_items = merged_info['num_items']
        entity_type = merged_info['entity_type']
        entity_ids = merged_info['ids']

        # Handle case with zero items gracefully
        if num_items == 0:
            logger.warning("Input files contain 0 entities. Output file will be created but mostly empty.")
            # Create an empty file with just the attribute
            with h5torch.File(str(output_h5_path), 'w') as h5_out:
                if entity_type:
                     h5_out.attrs['entity_type'] = entity_type
            return # Exit early

        # Create h5_id_to_source_idx mapping for pretrain datasets
        # For pretrain, the entity_ids are sequential, so we can map directly
        h5_id_to_source_idx = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
        logger.debug(f"Created h5_id_to_source_idx mapping for {len(h5_id_to_source_idx)} entities")

        # Create the output file
        with h5torch.File(str(output_h5_path), 'w') as h5_out:
            logger.info(f"Writing pretrain data for {num_items} entities ('{entity_type}')...")
            # Add entity type attribute
            if entity_type:
                h5_out.attrs['entity_type'] = entity_type
            h5_out.attrs['n_items'] = num_items

            # 1. Register Central Index
            index_data = np.arange(num_items, dtype=np.uint32) # 16 bit isn't enough for 65535+ items...
            h5_out.register(
                data=index_data,
                axis='central',
                name='index',
                mode='N-D',
                length=num_items
            )
            logger.info(f"Registered central 'index' (Shape: {index_data.shape})")

            # 2. Register Aligned Axis 0 Data (Features AND Representations)
            logger.info("Registering aligned axis 0 data (Features and Representations)...")
            
            if not merged_info['feature_sources'] and not merged_info['repr_sources']:
                logger.warning("No features or representations found in input files.")
            
            registered_reprs = _register_aligned_entity_data(
                h5_out=h5_out,
                axis=0,
                entity_ids=entity_ids,
                entity_info=merged_info,
                h5_id_to_source_idx=h5_id_to_source_idx,
                num_entities=num_items,
                batch_size=batch_size
            )

            # 3. Add Split Data (Optional) - Use boolean for efficiency
            if add_split and num_items > 0:
                logger.info(f"Generating train/validation split (train_frac={train_frac}, split_seed={split_seed})...")
                np.random.seed(split_seed)
                indices = np.random.permutation(num_items)
                num_train = int(num_items * train_frac)

                # Create boolean array: True for train, False for validation
                split_data = np.zeros(num_items, dtype=bool)
                split_data[indices[:num_train]] = True

                h5_out.register(
                    split_data,
                    axis='unstructured',
                    name=split_name,
                    mode='N-D',
                    dtype_save='bool',
                    dtype_load='bool',
                    length=num_items
                )
                logger.info(f"Registered unstructured '{split_name}' (Train: {num_train}, Val: {num_items - num_train})")
            elif add_split:
                logger.warning("add_split is True, but num_items is 0. Skipping split generation.")

            logger.info(f"Successfully finished writing pretrain h5torch file: {output_h5_path}")

    except Exception as e:
         logger.error(f"Failed to create pretrain h5torch file: {e}")
         logger.exception("Traceback:")
         raise

    finally:
        # Ensure all input file handles are closed
        if merged_info and merged_info['file_handles']:
            logger.debug(f"Closing {len(merged_info['file_handles'])} input HDF5 file handle(s)...")
            for handle in merged_info['file_handles']:
                try:
                    handle.close()
                except Exception as e:
                    logger.error(f"Error closing input file {handle.filename}: {e}")

# --- Inspection Function ---

def inspect_h5torch_file(h5_path: Path) -> None:
    """
    Inspects an h5torch-formatted HDF5 file and prints a summary of its structure
    (attributes, central object, aligned axes, unstructured data).

    Works for both pretrain (single-axis) and DTI (multi-axis) h5torch files.

    Args:
        h5_path: Path to the h5torch HDF5 file.
    """
    if not h5_path.exists():
        logger.error(f"Inspection failed: File not found at {h5_path}")
        return

    logger.info(f"--- Inspecting H5torch File: {h5_path.name} ---")
    h5_file = None
    try:
        h5_file = h5py.File(h5_path, 'r')

        # 1. Root attributes
        print("\n[Root Attributes]")
        if not h5_file.attrs:
            print("  - No root attributes found.")
        else:
            for key, value in h5_file.attrs.items():
                print(f"  - {key}: {value}")

        # 2. Central dataset
        print("\n[Central Dataset]")
        if 'central' in h5_file:
            central_obj = h5_file['central']

            # Check if /central is a Dataset or a Group
            if isinstance(central_obj, h5py.Dataset):
                print("  Mode: N/A (Implicitly N-D or similar)")
                _print_dataset_info('central', central_obj)

            elif isinstance(central_obj, h5py.Group):
                central_group = central_obj # Rename for clarity in this block
                central_mode = central_group.attrs.get('mode', 'N/A')
                print(f"  Mode: {central_mode}")

                if central_mode == 'coo':
                    indices = central_group.get('indices')
                    values = central_group.get('values')
                    shape = central_group.attrs.get('shape')
                    print(f"  Shape (Attr): {shape}")
                    if indices is not None and isinstance(indices, h5py.Dataset):
                         _print_dataset_info('indices', indices)
                    else:
                         print("    - Dataset 'indices' not found or not a dataset.")
                    if values is not None and isinstance(values, h5py.Dataset):
                         _print_dataset_info('values', values) # dtype_load is typically on values
                    else:
                        print("    - Dataset 'values' not found or not a dataset.")
                else: # N-D, vlen, csr, separate etc. within a group
                    found_central_data = False
                    for name, dset in central_group.items():
                        if isinstance(dset, h5py.Dataset):
                            _print_dataset_info(name, dset)
                            found_central_data = True
                    if not found_central_data:
                        print("  - No central dataset found within the 'central' group.")
            else:
                print(f"  - Object at '/central' is neither a Group nor a Dataset (Type: {type(central_obj)}). Skipping.")

        else:
            print("  - Group or Dataset 'central' not found.")

        # 3. Aligned Axes
        print("\n[Aligned Axes]")
        found_axes = False
        axis_keys = sorted([k for k in h5_file.keys() if k.isdigit()]) # Find '0', '1', ...
        if not axis_keys:
             print("  - No aligned axes found (groups '0', '1', ...).")
        else:
            for axis_key in axis_keys:
                found_axes = True
                axis_group = h5_file[axis_key]
                print(f"\n  --- Axis {axis_key} ---")
                if not list(axis_group.keys()):
                    print("    - No datasets found.")
                    continue
                for name, dset in axis_group.items():
                    if isinstance(dset, h5py.Dataset):
                        _print_dataset_info(name, dset)
                    else:
                        print(f"    - Found non-dataset object: {name} (Type: {type(dset)}) within axis {axis_key}")

        # 4. Unstructured datasets
        print("\n[Unstructured Datasets]")
        if 'unstructured' in h5_file:
            unstructured_group = h5_file['unstructured']
            if not list(unstructured_group.keys()):
                print("  - No datasets found.")
            else:
                for name, dset in unstructured_group.items():
                    if isinstance(dset, h5py.Dataset):
                        _print_dataset_info(name, dset)
                    else:
                        print(f"    - Found non-dataset object: {name} (Type: {type(dset)}) within unstructured group")
        else:
            print("  - Group 'unstructured' not found.")

    except Exception as e:
        logger.error(f"Failed to inspect HDF5 file {h5_path}: {e}", exc_info=True) # Log with traceback
    finally:
        if h5_file:
            try:
                h5_file.close()
            except Exception as close_err:
                 logger.error(f"Error closing HDF5 file {h5_path.name} after inspection: {close_err}")
        logger.info(f"--- Finished Inspecting: {h5_path.name} ---")

