"""
Custom PyTorch (h5torch) Dataset classes for loading pretrain and DTI datasets.
"""

import h5py
import h5torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any


# Setup logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
              (e.g., ['in_DAVIS', 'in_KIBA']). Items where *any* of these are True are kept.
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
            if "unstructured" not in f:
                logger.warning(f"Cannot apply filters: 'unstructured' group not found in {h5_path.name}")
                return None

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
                     logger.warning("COO central data missing. Attempting to infer mask size from unstructured.")
            
            # If not COO or COO inference failed, infer from unstructured data
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
            if split_col and split_value is not None:
                split_key = f"unstructured/{split_col}"
                if split_key in f:
                    logger.debug(f"Applying split filter: '{split_col}' == '{split_value}'")
                    values = f[split_key][:]
                    
                    # Handle boolean split columns
                    if isinstance(split_value, bool):
                        if values.dtype != bool:
                            values = values.astype(bool)
                        split_mask = (values == split_value)
                    else:
                        # Handle string split columns (object/string)
                        if values.dtype.kind in ('O', 'S'):
                            values = np.array([s.decode('utf-8', errors='replace') if isinstance(s, bytes) else str(s) for s in values])
                        split_mask = (values == split_value)
                    
                    final_mask &= split_mask
                    applied_filter = True
                else:
                    logger.warning(f"Split filter column '{split_col}' not found")

            # Apply provenance filter (keep if ANY specified provenance is True)
            provenance_cols = filters.get('provenance_cols')
            if provenance_cols and isinstance(provenance_cols, list) and len(provenance_cols) > 0:
                provenance_combined_mask = np.zeros(mask_size, dtype=bool)
                found_prov_col = False
                
                for prov_col in provenance_cols:
                    prov_key = f"unstructured/{prov_col}"
                    if prov_key in f:
                        values = f[prov_key][:].astype(bool)
                        provenance_combined_mask |= values
                        found_prov_col = True

                if found_prov_col:
                    final_mask &= provenance_combined_mask
                    applied_filter = True

            if applied_filter:
                logger.info(f"Subset mask for {h5_path.name}: kept {np.sum(final_mask)} / {mask_size} items")
                return final_mask
            else:
                logger.warning("Filters specified but none were applicable")
                return None

    except Exception as e:
        logger.error(f"Failed to calculate subset mask for {h5_path}: {e}")
        raise


class PretrainDataset(h5torch.Dataset):
    """
    A PyTorch Dataset for loading single-entity data from an h5torch file
    created by `create_pretrain_h5torch`.

    Assumes structure:
    - Central: 'index'
    - Aligned Axis 0: Features (e.g., 'FP-Morgan', 'EMB-ESM') and Representations (e.g., 'SMILES', 'AA')
    - Unstructured: Split information (boolean is_train)

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
    }
    """
    def __init__(
        self,
        h5_path: Path,
        subset_filters: Optional[Dict[str, bool]] = {'split_col': 'is_train', 'split_value': True},
        load_in_memory: bool = False
    ):
        """
        Args:
            h5_path: Path to the h5torch HDF5 file.
            subset_filters: Dictionary of filters to apply (see _calculate_subset_mask).
                            Example: {'split_col': 'is_train', 'split_value': True}
                            For boolean columns, split_value should be True/False
            load_in_memory: Whether to load the entire dataset into memory.
        """
        self.h5_path = Path(h5_path)
        self.subset_filters = subset_filters

        # Calculate subset mask before initializing parent
        subset_mask = _calculate_subset_mask(self.h5_path, subset_filters, is_coo=False)

        # Initialize the h5torch Dataset
        super().__init__(
            file=str(self.h5_path),
            subset=subset_mask,
            sampling=0, # Sample along axis 0
            in_memory=load_in_memory
        )

        # Store paths for efficient access in __getitem__
        self._identify_paths()
        logger.info(f"Initialized PretrainDataset from {self.h5_path.name}. Size: {len(self)} items.")
        logger.info(f"  Features (Axis 0): {list(self.feature_names)}")
        logger.info(f"  Representations (Axis 0): {list(self.repr_names)}")

    def _identify_paths(self):
        """Identify and store the HDF5 paths for features and representations."""
        self.feature_names = []  # Numerical features
        self.repr_names = []     # String representations

        # Use self.f which is the h5py.File object opened by h5torch.Dataset
        for name in self.f['0'].keys():
            dataset = self.f[f'0/{name}']
            # Distinguish between features (numerical) and representations (strings)
            if dataset.dtype.kind in ('O', 'S', 'U'):
                # String/object data -> representation
                self.repr_names.append(name)
            else:
                # Numerical data -> feature
                self.feature_names.append(name)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches an item and structures it into 'id', 'representations', and 'features'.
        """
        # Get the flat dictionary from h5torch.Dataset's __getitem__
        # Keys will be like 'central', '0/FP-Morgan', '0/SMILES', etc.
        # Values are already converted based on dtype_load by h5torch
        item_flat = super().__getitem__(idx)

        # Structure the output
        structured_item = {
            'id': int(item_flat.get('central', idx)),  # Use central or fallback to idx
            'representations': {},
            'features': {}
        }

        # Populate representations (h5torch handles string conversion via dtype_load='str')
        for name in self.repr_names:
            key = f'0/{name}'
            if key in item_flat:
                value = item_flat[key]
                # h5torch should return strings directly, but handle edge cases
                # if isinstance(value, bytes):
                #     structured_item['representations'][name] = value.decode('utf-8', errors='replace')
                # elif isinstance(value, np.ndarray) and value.ndim == 0:
                #     # Scalar array case
                #     val = value.item()
                #     structured_item['representations'][name] = val.decode('utf-8', errors='replace') if isinstance(val, bytes) else str(val)
                # else:
                structured_item['representations'][name] = value

        # Populate features (should be numpy arrays already)
        for name in self.feature_names:
            key = f'0/{name}'
            if key in item_flat:
                structured_item['features'][name] = item_flat[key]

        return structured_item
