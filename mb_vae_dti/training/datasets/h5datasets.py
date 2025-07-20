"""
Custom PyTorch (h5torch) Dataset classes for loading pretrain and DTI datasets.
"""

import h5py
import h5torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


# Setup logger
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
            
            # Convert OmegaConf ListConfig to regular list if needed
            if provenance_cols is not None and hasattr(provenance_cols, '__iter__'):
                provenance_cols = list(provenance_cols)
            
            if provenance_cols and isinstance(provenance_cols, list) and len(provenance_cols) > 0:
                logger.debug(f"Applying provenance filter for columns: {provenance_cols}")
                provenance_combined_mask = np.zeros(mask_size, dtype=bool)
                found_prov_col = False
                
                for prov_col in provenance_cols:
                    prov_key = f"unstructured/{prov_col}"
                    if prov_key in f:
                        values = f[prov_key][:].astype(bool)
                        provenance_combined_mask |= values
                        found_prov_col = True
                    else:
                        logger.warning(f"  Provenance column '{prov_col}' not found in unstructured data")

                if found_prov_col:
                    n_before = np.sum(final_mask)
                    final_mask &= provenance_combined_mask
                    n_after = np.sum(final_mask)
                    logger.debug(f"  Provenance filter: {n_before} -> {n_after} samples")
                    applied_filter = True
                else:
                    logger.warning("No provenance columns found, provenance filter not applied")

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


class DTIDataset(h5torch.Dataset):
    """
    A PyTorch Dataset for loading consolidated DTI data from an h5torch file
    created by `create_dti_h5torch`, using COO sampling.

    It retrieves the interaction value (Y), the aligned features for the
    interacting drug and target, and the corresponding unstructured
    interaction-level metadata (splits, provenance, continuous Y values).

    Returns structured dictionaries for each interaction:
    {
        'id': int, # COO index for the unique drug/target pair
        'y': {
            'Y': bool,
            'Y_pKd': float/None,
            'Y_pKi': float/None,
            'Y_KIBA': float/None,
        }
        'drug': {
            'id':{
                'Drug_ID': str,
                'Drug_InChIK': str
            },
                'representations': {
                    'SMILES': str
            },
            'features': {
                'FP-Morgan': np.ndarray (float32),
                'EMB-BiomedGraph': np.ndarray (float32),
                'EMB-BiomedImg': np.ndarray (float32),
                'EMB-BiomedText': np.ndarray (float32),
            }
        },
        'target': {
            'id':{
                'Target_ID': str,
                'Target_UniProt_ID': str,
                'Target_Gene_name': str,
                'Target_RefSeq_ID': str,
            },
            'representations': {
                'AA': str,
                'DNA': str
            },
            'features': {
                'FP-ESP': np.ndarray (float32),
                'EMB-ESM': np.ndarray (float32),
                'EMB-NT': np.ndarray (float32)
            }
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
        self.h5_path = Path(h5_path)
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
        
        # Pre-load unstructured Y data for efficient access
        # Since h5torch COO sampling doesn't include unstructured data,
        # we need to load it separately and map it to COO indices
        self._preload_unstructured_y_data()

        # Log summary
        logger.info(f"Initialized DTIDataset from {self.h5_path.name}. Size: {len(self)} interactions.")
        # logger.info(f"  Drug paths (Axis 0): {list(self.drug_paths.keys())}")
        # logger.info(f"  Target paths (Axis 1): {list(self.target_paths.keys())}")
        # logger.info(f"  Y paths (Central/Unstructured): {list(self.y_paths.keys())}")
        # logger.info(f"  Metadata paths (Unstructured): {list(self.metadata_paths.keys())}")
        # logger.info(f"  Pre-loaded unstructured Y data: {list(self.unstructured_y_data.keys()) if hasattr(self, 'unstructured_y_data') else 'None'}")

    def _identify_paths(self):
        """Identify and store the HDF5 paths for different data categories."""
        self.drug_paths = {}        # Aligned axis 0
        self.target_paths = {}      # Aligned axis 1
        self.y_paths = {}           # Central 'values' and Unstructured 'Y_*'
        self.metadata_paths = {}    # Unstructured (others)

        # Separate drug data into id, representations, and features
        self.drug_id_paths = {}
        self.drug_repr_paths = {}
        self.drug_feature_paths = {}

        # Separate target data into id, representations, and features
        self.target_id_paths = {}
        self.target_repr_paths = {}
        self.target_feature_paths = {}

        # Central Data (Interaction Y value) is handled automatically by h5torch for COO sampling

        # Aligned Axis 0 (Drug data)
        if '0' in self.f:
            for name in self.f['0'].keys():
                path = f'0/{name}'
                self.drug_paths[name] = path
                
                # Categorize drug data
                dataset = self.f[path]
                if name.endswith('_ID') or name in ['Drug_ID', 'Drug_InChIKey']:
                    self.drug_id_paths[name] = path
                elif dataset.dtype.kind in ('O', 'S', 'U') or name in ['SMILES', 'Drug_SMILES']:
                    self.drug_repr_paths[name] = path
                else:
                    self.drug_feature_paths[name] = path

        # Aligned Axis 1 (Target data)
        if '1' in self.f:
            for name in self.f['1'].keys():
                path = f'1/{name}'
                self.target_paths[name] = path
                
                # Categorize target data
                dataset = self.f[path]
                if name.endswith('_ID') or name in ['Target_ID', 'Target_UniProt_ID', 'Target_Gene_name', 'Target_RefSeq_ID']:
                    self.target_id_paths[name] = path
                elif dataset.dtype.kind in ('O', 'S', 'U') or name in ['AA', 'DNA', 'Target_AA', 'Target_DNA']:
                    self.target_repr_paths[name] = path
                else:
                    self.target_feature_paths[name] = path

        # Unstructured Data (Interaction Y values and Metadata)
        if 'unstructured' in self.f:
            for name in self.f['unstructured'].keys():
                path = f'unstructured/{name}'
                # Categorize based on name prefix
                if name == 'Y' or name.startswith('Y_'):
                    self.y_paths[name] = path
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

    def _preload_unstructured_y_data(self):
        """
        Pre-load unstructured Y data (Y_pKd, Y_pKi, Y_KIBA) for efficient access.
        
        Since h5torch COO sampling doesn't include unstructured data, we need to
        load it separately and map it to the subset indices if filtering is applied.
        """
        self.unstructured_y_data = {}
        
        # Get the Y_* columns from unstructured data
        y_columns = [name for name in self.y_paths.keys() if name.startswith('Y_')]
        
        if not y_columns:
            logger.debug("No unstructured Y_* columns found.")
            return
            
        # Load the data for each Y column
        for col_name in y_columns:
            path = self.y_paths[col_name]
            full_data = self.f[path][:]  # Load all data
            
            # If we have a subset mask, we need to apply it to get the corresponding Y values
            if hasattr(self, 'indices') and self.indices is not None:
                # self.indices contains the COO indices that were selected by the subset
                subset_data = full_data[self.indices]
                self.unstructured_y_data[col_name] = subset_data
                logger.debug(f"Pre-loaded {len(subset_data)} values for unstructured Y column '{col_name}' (subset applied)")
            else:
                # No subset, use all data
                self.unstructured_y_data[col_name] = full_data
                logger.debug(f"Pre-loaded {len(full_data)} values for unstructured Y column '{col_name}' (full dataset)")
        
        logger.debug(f"Pre-loaded unstructured Y data for {len(self.unstructured_y_data)} columns: {list(self.unstructured_y_data.keys())}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches an interaction and structures it into the specified format.
        """
        try:
            # Get the flat dictionary from h5torch.Dataset's __getitem__ for COO sampling
            # Keys will be like:
            # 'central': Y value (from COO values)
            # '0/Drug_ID': drug ID
            # '0/FP-Morgan': drug fingerprint
            # '1/Target_ID': target ID
            # '1/EMB-ESM': target embedding
            # 'unstructured/Y_pKd': pKd value
            # 'unstructured/split_rand': split value
            item_flat = super().__getitem__(idx)
            
            if item_flat is None:
                logger.warning(f"h5torch returned None for index {idx}")
                return self._create_empty_sample(idx)

            # Structure the output according to the specified format
            structured_item = {
                'id': idx,  # COO index for the unique drug/target pair
                'y': {},
                'drug': {
                    'id': {},
                    'representations': {},
                    'features': {}
                },
                'target': {
                    'id': {},
                    'representations': {},
                    'features': {}
                }
            }

            # Populate Y data (Central Y and Unstructured Y_*)
            # Get central 'Y' value (h5torch returns it as 'central' for COO sampling)
            central_y = item_flat.get('central')
            if central_y is not None:
                 structured_item['y']['Y'] = central_y
            else:
                 logger.warning(f"Could not find central interaction value ('central') in item keys: {list(item_flat.keys())}")
                 structured_item['y']['Y'] = 0  # Default value

            # Get additional Y_* values from pre-loaded unstructured data
            # Note: h5torch COO sampling doesn't include unstructured data, so we use pre-loaded data
            if hasattr(self, 'unstructured_y_data'):
                for col_name, data_array in self.unstructured_y_data.items():
                    if idx < len(data_array):
                        value = data_array[idx]
                        # Handle NaN values for continuous Y columns
                        if pd.isna(value):
                            value = None
                        structured_item['y'][col_name] = value
                    else:
                        logger.warning(f"Index {idx} out of bounds for unstructured Y column '{col_name}' (length: {len(data_array)})")
                        structured_item['y'][col_name] = None

            # Populate Drug data
            # Drug IDs
            for name, path in self.drug_id_paths.items():
                if path in item_flat:
                     value = item_flat[path]
                     if isinstance(value, bytes):
                          structured_item['drug']['id'][name] = value.decode('utf-8', errors='replace')
                     elif isinstance(value, np.ndarray) and value.dtype.kind in ('O', 'S', 'U'):
                          structured_item['drug']['id'][name] = value.item().decode('utf-8', errors='replace') if isinstance(value.item(), bytes) else str(value.item())
                     else:
                         structured_item['drug']['id'][name] = str(value)

            # Drug Representations
            for name, path in self.drug_repr_paths.items():
                if path in item_flat:
                     value = item_flat[path]
                     if isinstance(value, bytes):
                         structured_item['drug']['representations'][name] = value.decode('utf-8', errors='replace')
                     elif isinstance(value, np.ndarray) and value.dtype.kind in ('O', 'S', 'U'):
                          structured_item['drug']['representations'][name] = value.item().decode('utf-8', errors='replace') if isinstance(value.item(), bytes) else str(value.item())
                     else:
                         structured_item['drug']['representations'][name] = str(value)

            # Drug Features
            for name, path in self.drug_feature_paths.items():
                if path in item_flat:
                    feature_value = item_flat[path]
                    if feature_value is not None:
                        structured_item['drug']['features'][name] = feature_value
                    else:
                        logger.warning(f"None feature value for drug feature {name} at index {idx}")
                        # Create a dummy feature with appropriate shape
                        structured_item['drug']['features'][name] = np.zeros(1, dtype=np.float32)

            # Populate Target data
            # Target IDs
            for name, path in self.target_id_paths.items():
                if path in item_flat:
                     value = item_flat[path]
                     if isinstance(value, bytes):
                         structured_item['target']['id'][name] = value.decode('utf-8', errors='replace')
                     elif isinstance(value, np.ndarray) and value.dtype.kind in ('O', 'S', 'U'):
                          structured_item['target']['id'][name] = value.item().decode('utf-8', errors='replace') if isinstance(value.item(), bytes) else str(value.item())
                     else:
                         structured_item['target']['id'][name] = str(value)

            # Target Representations
            for name, path in self.target_repr_paths.items():
                if path in item_flat:
                     value = item_flat[path]
                     if isinstance(value, bytes):
                         structured_item['target']['representations'][name] = value.decode('utf-8', errors='replace')
                     elif isinstance(value, np.ndarray) and value.dtype.kind in ('O', 'S', 'U'):
                          structured_item['target']['representations'][name] = value.item().decode('utf-8', errors='replace') if isinstance(value.item(), bytes) else str(value.item())
                     else:
                         structured_item['target']['representations'][name] = str(value)

            # Target Features
            for name, path in self.target_feature_paths.items():
                if path in item_flat:
                    feature_value = item_flat[path]
                    if feature_value is not None:
                        structured_item['target']['features'][name] = feature_value
                    else:
                        logger.warning(f"None feature value for target feature {name} at index {idx}")
                        # Create a dummy feature with appropriate shape
                        structured_item['target']['features'][name] = np.zeros(1, dtype=np.float32)
                      
            return structured_item
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a valid but empty sample instead of None
            return self._create_empty_sample(idx)
    
    def _create_empty_sample(self, idx: int) -> Dict[str, Any]:
        """Create an empty but valid sample structure to avoid None returns."""
        return {
            'id': idx,
            'y': {
                'Y': 0,
                'Y_pKd': None,
                'Y_pKi': None, 
                'Y_KIBA': None
            },
            'drug': {
                'id': {},
                'representations': {},
                'features': {
                    'FP-Morgan': np.zeros(2048, dtype=np.float32),  # Common drug feature sizes
                    'EMB-BiomedGraph': np.zeros(512, dtype=np.float32),
                    'EMB-BiomedImg': np.zeros(512, dtype=np.float32),
                    'EMB-BiomedText': np.zeros(768, dtype=np.float32),
                }
            },
            'target': {
                'id': {},
                'representations': {},
                'features': {
                    'FP-ESP': np.zeros(4170, dtype=np.float32),  # Common target feature sizes
                    'EMB-ESM': np.zeros(1152, dtype=np.float32),
                    'EMB-NT': np.zeros(1024, dtype=np.float32),
                }
            }
        }
