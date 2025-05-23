"""
Custom PyTorch (h5torch) Dataset classes for loading pretrain and DTI datasets.
"""

import h5py
import h5torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import torch


# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _calculate_subset_mask(
    h5_path: Path,
    filters: Optional[Dict[str, Union[str, List[str]]]] = None,
    is_coo: bool = False
) -> Union[np.ndarray, None]:
    """
    Calculates a boolean subset mask based on filters applied to unstructured data.

    Args:
        h5_path: Path to the h5torch HDF5 file.
        filters: Dictionary of filters to apply. Expected keys:
            - 'split_col' (str): Name of the split column in unstructured (e.g., 'split', 'split_rand').
            - 'split_value' (str): Value to keep in the split column (e.g., 'train', 'val').
            - 'provenance_cols' (List[str]): List of boolean provenance columns in unstructured
              (e.g., ['in_DAVIS', 'in_KIBA']). Items where *any* of these are True are kept,
              if the filter is active.
        is_coo: Whether the central object is COO. If True, mask length matches NNZ,
                otherwise matches the length of the first unstructured dataset found.

    Returns:
        A boolean numpy array mask, or None if no filters are provided or applicable.
        The mask is True for items to keep.
    """
    if not filters:
        return None

    logger.debug(f"Calculating subset mask for {h5_path} with filters: {filters}")

    try:
        with h5py.File(h5_path, 'r') as f:
            if "unstructured" not in f and filters:
                 logger.warning(f"Cannot apply filters: 'unstructured' group not found in {h5_path.name}, but filters were provided: {filters}")
                 return None # Cannot apply filters if group doesn't exist

            # Determine the size of the mask
            mask_size = None
            if is_coo:
                # For COO, mask aligns with interactions (NNZ)
                if "central/indices" in f:
                    mask_size = f["central/indices"].shape[1]
                    logger.debug(f"COO mask size based on central/indices: {mask_size}")
                elif "central/values" in f:
                     mask_size = len(f["central/values"])
                     logger.debug(f"COO mask size based on central/values: {mask_size}")
                else:
                     # If central indices/values are missing, try inferring from unstructured
                     logger.warning("COO central data missing. Attempting to infer mask size from unstructured.")
            
            # If not COO or COO inference failed, infer from unstructured/aligned data
            if mask_size is None:
                # Infer size from the first available filter column or any unstructured dataset
                unstructured_keys = list(f.get("unstructured", {}).keys())
                aligned_0_keys = list(f.get("0", {}).keys())
                # Prioritize checking filter keys first
                check_keys = list(filters.get('provenance_cols', []))
                if 'split_col' in filters and filters['split_col']:
                     check_keys.insert(0, filters['split_col'])
                
                found_size = False
                # Check unstructured first
                for key in check_keys + unstructured_keys:
                     dset_path = f"unstructured/{key}"
                     if dset_path in f:
                         mask_size = len(f[dset_path])
                         found_size = True
                         logger.debug(f"Inferred mask size {mask_size} from {dset_path}")
                         break
                # If still not found, check aligned axis 0 (should have same length as unstructured in pretrain)
                if not found_size and not is_coo:
                    for key in aligned_0_keys:
                        dset_path = f"0/{key}"
                        if dset_path in f:
                            mask_size = len(f[dset_path])
                            found_size = True
                            logger.debug(f"Inferred mask size {mask_size} from {dset_path}")
                            break

                if not found_size:
                    logger.warning(f"Cannot determine mask size for {h5_path.name}. No suitable datasets found.")
                    return None
            
            if mask_size == 0:
                logger.info("Dataset size is 0, returning empty mask.")
                return np.array([], dtype=bool)

            # Start with a mask allowing all items
            final_mask = np.ones(mask_size, dtype=bool)
            applied_filter = False

            # Apply split filter
            split_col = filters.get('split_col')
            split_value = filters.get('split_value')
            if split_col and split_value:
                split_key = f"unstructured/{split_col}"
                if split_key not in f:
                    logger.warning(f"Split filter column '{split_col}' not found. Skipping split filter.")
                else:
                    logger.debug(f"Applying split filter: '{split_col}' == '{split_value}'")
                    # Decode bytes if necessary, handle potential errors
                    try:
                        values = f[split_key][:]
                        if values.dtype.kind in ('O', 'S'): # Object or String/Bytes
                            values = np.array([s.decode('utf-8', errors='replace') if isinstance(s, bytes) else str(s) for s in values])
                        
                        if len(values) != mask_size:
                             raise ValueError(f"Length mismatch for '{split_col}' ({len(values)}) vs expected mask size ({mask_size})")

                        split_mask = (values == split_value)
                        final_mask &= split_mask
                        applied_filter = True
                        logger.debug(f"Split filter kept {np.sum(final_mask)} / {mask_size} items.")
                    except Exception as e:
                        logger.error(f"Error applying split filter on '{split_col}': {e}", exc_info=True)


            # Apply provenance filter (keep if ANY specified provenance is True)
            provenance_cols = filters.get('provenance_cols')
            if provenance_cols and isinstance(provenance_cols, list) and len(provenance_cols) > 0:
                provenance_combined_mask = np.zeros(mask_size, dtype=bool)
                found_prov_col = False
                logger.debug(f"Applying provenance filter (OR logic): keep if in {provenance_cols}")
                for prov_col in provenance_cols:
                    prov_key = f"unstructured/{prov_col}"
                    if prov_key not in f:
                        logger.warning(f"Provenance filter column '{prov_col}' not found. Skipping.")
                        continue
                    
                    try:
                        values = f[prov_key][:].astype(bool) # Ensure boolean
                        if len(values) != mask_size:
                             raise ValueError(f"Length mismatch for '{prov_col}' ({len(values)}) vs expected mask size ({mask_size})")
                        provenance_combined_mask |= values
                        found_prov_col = True
                    except Exception as e:
                        logger.error(f"Error applying provenance filter on '{prov_col}': {e}", exc_info=True)

                if found_prov_col:
                    final_mask &= provenance_combined_mask
                    applied_filter = True
                    logger.debug(f"Provenance filter kept {np.sum(provenance_combined_mask)} items (before combining with other filters).")
                    logger.debug(f"Combined filters kept {np.sum(final_mask)} / {mask_size} items.")
                else:
                    logger.warning("No valid provenance columns found for filtering.")

            if not applied_filter and filters: # Check filters were provided
                 # If filters were given but none were applicable, warn or maybe return None?
                 # Let's return None if no filter could be applied but filters were requested.
                 logger.warning("Filters specified, but none were applicable or found in the file. Returning None mask.")
                 return None

            # Only return the mask if filters were applied or no filters were specified initially
            if applied_filter or not filters:
                logger.info(f"Calculated subset mask for {h5_path.name}. Kept {np.sum(final_mask)} / {mask_size} items.")
                return final_mask
            else:
                 # This case should be caught above, but as a safeguard
                 return None


    except FileNotFoundError:
        logger.error(f"File not found: {h5_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to calculate subset mask for {h5_path}: {e}", exc_info=True)
        raise


class PretrainDataset(h5torch.Dataset):
    """
    A PyTorch Dataset for loading single-entity data from an h5torch file
    created by `create_pretrain_h5torch`.

    Assumes structure:
    - Central: 'index'
    - Aligned Axis 0: Features (e.g., 'FP-Morgan', 'EMB-ESM')
    - Unstructured: Representations (e.g., 'SMILES', 'AA') and optional splits.

    Returns structured dictionaries for each item:
    {
        'id': int, # The original index from the HDF5 file
        'representations': {
            'SMILES': str,
            'AA': str,
            ...
        },
        'features': {
            'FP-Morgan': np.ndarray (float32),
            'EMB-ESM': np.ndarray (float32),
             ...
        }
        # Note: Split column is used for filtering but not returned by default.
    }
    """
    def __init__(
        self,
        h5_path: Path,
        subset_filters: Optional[Dict[str, Union[str, List[str]]]] = None,
        load_in_memory: bool = False
    ):
        """
        Args:
            h5_path: Path to the h5torch HDF5 file.
            subset_filters: Dictionary of filters to apply (see _calculate_subset_mask).
                            Example: {'split_col': 'split', 'split_value': 'train'}
            load_in_memory: Whether to load the entire dataset into memory.
        """
        self.h5_path = Path(h5_path) # Ensure it's a Path object
        self.subset_filters = subset_filters

        # Calculate subset mask before initializing parent
        # For PretrainDataset, the central object is N-D (index), not COO.
        # The mask should align with the length of axis 0 / unstructured data.
        subset_mask = _calculate_subset_mask(self.h5_path, subset_filters, is_coo=False)

        # Initialize the h5torch Dataset. Sampling along axis 0 is driven by the central index.
        # h5torch handles loading based on dtype_load attributes.
        super().__init__(
            file=str(self.h5_path),
            subset=subset_mask,
            sampling=0, # Sample along axis 0 (features aligned to central index)
            in_memory=load_in_memory
        )

        # Store paths for easier access in __getitem__
        self._identify_paths()
        logger.info(f"Initialized PretrainDataset from {self.h5_path.name}. Size: {len(self)} items.")
        logger.info(f"  Feature paths (Axis 0): {list(self.feature_paths.keys())}")
        logger.info(f"  Representation paths (Unstructured): {list(self.repr_paths.keys())}")

    def _identify_paths(self):
        """Identify and store the HDF5 paths for features and representations."""
        self.feature_paths = {} # Aligned axis 0
        self.repr_paths = {}    # Unstructured

        # Use self.f which is the h5py.File object opened by h5torch.Dataset
        if '0' in self.f:
            for name in self.f['0'].keys():
                # Assume all datasets in axis 0 are features
                self.feature_paths[name] = f'0/{name}'
        
        if 'unstructured' in self.f:
            for name in self.f['unstructured'].keys():
                # Assume all datasets in unstructured are representations, *except* filter columns
                is_filter_col = False
                if self.subset_filters:
                    if name == self.subset_filters.get('split_col'):
                         is_filter_col = True
                    if name in self.subset_filters.get('provenance_cols', []):
                         is_filter_col = True
                
                if not is_filter_col:
                     self.repr_paths[name] = f'unstructured/{name}'

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches an item and structures it into 'id', 'representations', and 'features'.
        """
        # Get the flat dictionary from h5torch.Dataset's __getitem__
        # Keys will be like 'central/index', '0/FP-Morgan', 'unstructured/SMILES'
        # Values are already converted based on dtype_load by h5torch
        item_flat = super().__getitem__(idx)

        # Structure the output
        structured_item = {
            'id': -1, # Default value
            'representations': {},
            'features': {}
        }

        # Extract ID (assuming it's stored in central/index or central)
        id_val = item_flat.get('central/index', item_flat.get('central'))
        if id_val is not None:
            structured_item['id'] = int(id_val) # Ensure it's an int
        else:
            # Fallback: Use the dataset index if central/index isn't present
            # This might happen if the file wasn't created exactly as expected
            logger.warning(f"Could not find 'central/index' or 'central' in item keys: {list(item_flat.keys())}. Using dataset index {idx} as ID.")
            structured_item['id'] = idx

        # Populate representations from unstructured paths
        for name, path in self.repr_paths.items():
            if path in item_flat:
                 # Decode bytes to string if needed (though h5torch might handle it via dtype_load='str')
                 value = item_flat[path]
                 if isinstance(value, bytes):
                     structured_item['representations'][name] = value.decode('utf-8', errors='replace')
                 elif isinstance(value, np.ndarray) and value.dtype.kind in ('O', 'S', 'U'):
                      # Handle case where h5torch returns array of bytes/strings
                      structured_item['representations'][name] = value.item().decode('utf-8', errors='replace') if isinstance(value.item(), bytes) else str(value.item())
                 else:
                     structured_item['representations'][name] = value # Assume h5torch handled type conversion

        # Populate features from axis 0 paths
        for name, path in self.feature_paths.items():
            if path in item_flat:
                # Features are expected to be numerical arrays (or handled by h5torch load type)
                structured_item['features'][name] = item_flat[path]

        return structured_item


class DTIDataset(h5torch.Dataset):
    """
    A PyTorch Dataset for loading consolidated DTI data from an h5torch file
    created by `create_dti_h5torch`, using COO sampling.

    It retrieves the interaction value (Y), the aligned features for the
    interacting drug and target, and the corresponding unstructured
    interaction-level metadata (splits, provenance, continuous Y values).

    Returns structured dictionaries for each interaction:
    {
        'drug': { # Features/Representations from aligned axis 0
            'Drug_ID': str,
            'SMILES': str,
            'FP-Morgan': np.ndarray(float32),
            'EMB-BiomedGraph': np.ndarray(float32),
            ...
        },
        'target': { # Features/Representations from aligned axis 1
            'Target_ID': str,
            'AA': str, # Or 'DNA'
            'FP-ESPF': np.ndarray(float32),
            'EMB-ESM': np.ndarray(float32),
            ...
        },
        'interaction': { # From central (Y) and unstructured ('Y_*')
            'Y': int or float, # Binary interaction value (loaded as specified by dtype_load)
            'Y_pKd': float,
            'Y_pKi': float,
            ...
        },
        'metadata': { # From unstructured (excluding 'Y_*')
            'split_rand': str,
            'split_cold': str,
            'in_DAVIS': bool,
            ...
        }
    }
    """
    def __init__(
        self,
        h5_path: Path,
        subset_filters: Optional[Dict[str, Union[str, List[str]]]] = None,
        load_in_memory: bool = False
    ):
        """
        Args:
            h5_path: Path to the h5torch DTI HDF5 file.
            subset_filters: Dictionary of filters to apply (see _calculate_subset_mask).
                            Example: {'split_col': 'split_rand', 'split_value': 'train',
                                      'provenance_cols': ['in_DAVIS', 'in_KIBA']}
            load_in_memory: Whether to load the entire dataset into memory (potentially very large).
        """
        self.h5_path = Path(h5_path) # Ensure Path object
        self.subset_filters = subset_filters

        # Calculate subset mask for COO data (aligns with interactions/NNZ)
        subset_mask = _calculate_subset_mask(self.h5_path, subset_filters, is_coo=True)

        # Initialize the h5torch Dataset with COO sampling
        super().__init__(
            file=str(self.h5_path),
            subset=subset_mask,
            sampling='coo', # Sample interactions (drug, target, Y)
            in_memory=load_in_memory
        )

        # Store paths for easier access in __getitem__
        self._identify_paths()

        # Log summary
        logger.info(f"Initialized DTIDataset from {self.h5_path.name}. Size: {len(self)} interactions.")
        logger.info(f"  Drug paths (Axis 0): {list(self.drug_paths.keys())}")
        logger.info(f"  Target paths (Axis 1): {list(self.target_paths.keys())}")
        logger.info(f"  Interaction paths (Central/Unstructured): {list(self.interaction_paths.keys())}")
        logger.info(f"  Metadata paths (Unstructured): {list(self.metadata_paths.keys())}")

    def _identify_paths(self):
        """Identify and store the HDF5 paths for different data categories."""
        self.drug_paths = {}        # Aligned axis 0
        self.target_paths = {}      # Aligned axis 1
        self.interaction_paths = {} # Central 'values' and Unstructured 'Y_*'
        self.metadata_paths = {}    # Unstructured (others)

        # Central Data (Interaction Y value)
        # For COO, h5torch typically returns the value under 'central/values' or just 'central'
        # We'll handle this dynamically in __getitem__

        # Aligned Axis 0 (Drug data)
        if '0' in self.f:
            for name in self.f['0'].keys():
                self.drug_paths[name] = f'0/{name}'

        # Aligned Axis 1 (Target data)
        if '1' in self.f:
            for name in self.f['1'].keys():
                self.target_paths[name] = f'1/{name}'

        # Unstructured Data (Interaction Y values and Metadata)
        if 'unstructured' in self.f:
            for name in self.f['unstructured'].keys():
                path = f'unstructured/{name}'
                # Categorize based on name prefix
                if name.startswith('Y_'):
                    self.interaction_paths[name] = path
                else:
                    # Exclude filter columns from being returned as metadata
                    is_filter_col = False
                    if self.subset_filters:
                        if name == self.subset_filters.get('split_col'):
                             is_filter_col = True
                        if name in self.subset_filters.get('provenance_cols', []):
                             is_filter_col = True
                    if not is_filter_col:
                        self.metadata_paths[name] = path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches an interaction and structures it into 'drug', 'target',
        'interaction', and 'metadata'.
        """
        # Get the flat dictionary from h5torch.Dataset's __getitem__ for COO sampling
        # Keys will be like:
        # 'central/values': Y value
        # '0/Drug_ID': drug ID
        # '0/FP-Morgan': drug fingerprint
        # '1/Target_ID': target ID
        # '1/EMB-ESM': target embedding
        # 'unstructured/Y_pKd': pKd value
        # 'unstructured/split_rand': split value
        item_flat = super().__getitem__(idx)

        # Structure the output
        structured_item = {
            'drug': {},
            'target': {},
            'interaction': {},
            'metadata': {}
        }

        # Populate Drug data (Axis 0)
        for name, path in self.drug_paths.items():
            if path in item_flat:
                 value = item_flat[path]
                 # Handle potential bytes returned for string types
                 if isinstance(value, bytes):
                      structured_item['drug'][name] = value.decode('utf-8', errors='replace')
                 elif isinstance(value, np.ndarray) and value.dtype.kind in ('O', 'S', 'U'):
                      # Handle case where h5torch returns array of bytes/strings
                      structured_item['drug'][name] = value.item().decode('utf-8', errors='replace') if isinstance(value.item(), bytes) else str(value.item())
                 else:
                     structured_item['drug'][name] = value

        # Populate Target data (Axis 1)
        for name, path in self.target_paths.items():
            if path in item_flat:
                 value = item_flat[path]
                 if isinstance(value, bytes):
                     structured_item['target'][name] = value.decode('utf-8', errors='replace')
                 elif isinstance(value, np.ndarray) and value.dtype.kind in ('O', 'S', 'U'):
                      structured_item['target'][name] = value.item().decode('utf-8', errors='replace') if isinstance(value.item(), bytes) else str(value.item())
                 else:
                     structured_item['target'][name] = value

        # Populate Interaction data (Central Y and Unstructured Y_*)
        # Get central 'Y' value (h5torch might name it 'central/values' or 'central')
        y_value = item_flat.get('central/values', item_flat.get('central'))
        if y_value is not None:
             structured_item['interaction']['Y'] = y_value
        else:
             logger.warning(f"Could not find central interaction value ('central/values' or 'central') in item keys: {list(item_flat.keys())}")

        # Get additional Y_* values from unstructured
        for name, path in self.interaction_paths.items():
             if path in item_flat:
                 structured_item['interaction'][name] = item_flat[path]

        # Populate Metadata (Other Unstructured)
        for name, path in self.metadata_paths.items():
             if path in item_flat:
                 value = item_flat[path]
                 if isinstance(value, bytes):
                      structured_item['metadata'][name] = value.decode('utf-8', errors='replace')
                 elif isinstance(value, np.ndarray) and value.dtype.kind in ('O', 'S', 'U'):
                      structured_item['metadata'][name] = value.item().decode('utf-8', errors='replace') if isinstance(value.item(), bytes) else str(value.item())
                 else:
                      structured_item['metadata'][name] = value
                      
        return structured_item