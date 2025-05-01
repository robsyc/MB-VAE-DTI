"""
Custom PyTorch (h5torch) Dataset classes for loading pretrain and DTI datasets.
"""

import h5py
import h5torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict
import torch

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PretrainDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading entity data from an h5torch file
    created by `create_pretrain_h5torch`.

    Returns structured dictionaries for each item:
    {
        'id': int, # The central index
        'representations': {
            'smiles': str, 
            'aa': str, 
            ...
        },
        'features': {
            'fp': { 
                'FP-Morgan': np.ndarray (float32),
                 ...
             },
            'emb': {
                'EMB-BiomedGraph': np.ndarray (float32),
                ...
             }
        }
    }
    """
    def __init__(self, h5_path: Path, load_in_memory: bool = False):
        """
        Args:
            h5_path: Path to the h5torch HDF5 file.
            load_in_memory: Whether to load the entire HDF5 content into memory.
                            Use with caution for large files. Defaults to False.
        """
        if not h5_path.exists():
            raise FileNotFoundError(f"H5torch file not found: {h5_path}")
            
        self.h5_path = h5_path
        logger.info(f"Initializing PretrainDataset from: {self.h5_path}")
        
        # Use h5torch.Dataset internally, sampling along axis 0 (the entity index)
        # Pass in_memory flag
        # Note: h5torch.Dataset handles the dtype_load conversion internally
        self._internal_dataset = h5torch.Dataset(str(h5_path), sampling=0, in_memory=load_in_memory)
        self._length = len(self._internal_dataset)
        logger.info(f"Dataset contains {self._length} entities.")

        # Store names of available unstructured datasets
        self.unstructured_keys = []
        if 'unstructured' in self._internal_dataset.f:
            self.unstructured_keys = list(self._internal_dataset.f['unstructured'].keys())
            logger.info(f"Found unstructured representation keys: {self.unstructured_keys}")
        else:
            logger.warning(f"No 'unstructured' group found in {self.h5_path}")

    def __len__(self) -> int:
        """Returns the number of entities in the dataset."""
        return self._length

    def __getitem__(self, idx: int) -> Dict:
        """Retrieves and structures the data for a given entity index."""
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {self._length}")
            
        # Get the raw sample dictionary from h5torch.Dataset
        raw_sample = self._internal_dataset[idx]
        
        # DEBUG: Print raw keys returned by h5torch.Dataset
        logger.debug(f"Index {idx}: Raw keys from h5torch.Dataset: {list(raw_sample.keys())}")

        # DEBUG: Print raw sample
        # logger.debug(f"Index {idx}: Raw sample: {raw_sample}")
        # print(f"Index {idx}: Raw sample: {raw_sample}")

        # Structure the output dictionary
        structured_sample = {
            'id': None,
            'representations': {},
            'features': {
                'fp': {},
                'emb': {}
            }
        }

        for key, value in raw_sample.items():
            if key == 'central':
                # Ensure ID is a standard Python int if it's numpy int
                structured_sample['id'] = int(value) 
            elif key.startswith('0/'):
                # Aligned features (FP or EMB)
                feature_name = key.split('/', 1)[1] # Get name after '0/'
                if feature_name.startswith('FP-'):
                    # Verify fingerprint loaded as float32 as specified by dtype_load
                    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                        logger.warning(f"Fingerprint '{feature_name}' loaded with unexpected type: {type(value)}, {getattr(value, 'dtype', 'N/A')}. Expected float32 numpy array.")
                    structured_sample['features']['fp'][feature_name] = value
                elif feature_name.startswith('EMB-'):
                    structured_sample['features']['emb'][feature_name] = value
                else:
                     logger.warning(f"Unexpected key in aligned group '0/': {key}")
            else:
                # This case might now catch the unstructured keys if h5torch.Dataset returns them directly
                # We will handle unstructured keys explicitly below, so warn if it's not 'central' or '0/' prefixed
                if key != 'central' and not key.startswith('0/'):
                    logger.warning(f"Unexpected key in raw sample: {key}")
        
        # Manually fetch unstructured representations using the stored keys
        for repr_name in self.unstructured_keys:
            try:
                # Access the underlying h5py dataset directly
                value_raw = self._internal_dataset.f[f'unstructured/{repr_name}'][idx]
                # Decode if bytes (assuming utf-8 based on how we saved)
                value_str = value_raw.decode('utf-8') if isinstance(value_raw, bytes) else str(value_raw)
                structured_sample['representations'][repr_name] = value_str
            except KeyError:
                logger.error(f"KeyError accessing unstructured data: 'unstructured/{repr_name}' not found in {self.h5_path}")
            except IndexError:
                 logger.error(f"IndexError accessing unstructured data: Index {idx} out of bounds for 'unstructured/{repr_name}'")
            except Exception as e:
                 logger.error(f"Error fetching unstructured data '{repr_name}' for index {idx}: {e}")

        # Verify ID was found
        if structured_sample['id'] is None:
             logger.error(f"Could not find 'central' key for index {idx} in raw sample: {raw_sample}")
             # Decide how to handle: raise error or return incomplete dict?
             # Let's raise an error for clarity
             raise KeyError(f"Central index not found for sample index {idx} in {self.h5_path}")

        return structured_sample

    def close(self) -> None:
        """Closes the underlying h5torch file handle."""
        if hasattr(self, '_internal_dataset') and self._internal_dataset is not None:
            logger.info(f"Closing dataset: {self.h5_path}")
            self._internal_dataset.close()
            self._internal_dataset = None # Prevent closing again
        else:
             logger.debug(f"Dataset already closed or not initialized: {self.h5_path}")
             
    def __del__(self):
        """Attempt to close the dataset when the object is deleted."""
        # __del__ is not guaranteed to be called, explicit close() is better.
        self.close()


class DTIDataset(h5torch.Dataset):
    """
    A PyTorch Dataset for loading consolidated DTI data from an h5torch file
    created by `create_dti_h5torch`, using COO sampling.

    It retrieves the interaction value (Y), the aligned features for the
    interacting drug and target, and the corresponding unstructured
    interaction-level metadata (splits, provenance, continuous Y values).

    Returns structured dictionaries for each interaction:
    {
        'drug': { # Features from aligned axis 0
            'Drug_ID': str,
            'SMILES': str, # Or other representations like 'Graph', 'Image', 'Text'
            'FP-Morgan': np.ndarray(float32),
            'EMB-BiomedGraph': np.ndarray(float32),
            ...
        },
        'target': { # Features from aligned axis 1
            'Target_ID': str,
            'AA': str, # Or 'DNA'
            'FP-ESPF': np.ndarray(float32),
            'EMB-ESM': np.ndarray(float32),
            ...
        },
        'interaction': { # From central (Y) and unstructured
            'Y': int or float, # Binary interaction value (loaded as specified by dtype_load)
            'Y_pKd': float,
            'Y_pKi': float,
            ...
        },
        'metadata': { # From unstructured
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
        subset: Optional[Union[Tuple[str, str], np.ndarray]] = None,
        load_in_memory: bool = False
    ):
        """
        Args:
            h5_path: Path to the consolidated DTI h5torch HDF5 file.
            subset: Optional subset definition passed to h5torch.Dataset.
                    Examples:
                    - ('unstructured/split_rand', 'train')
                    - ('unstructured/in_DAVIS', True) # Requires bool stored correctly
                    - A boolean numpy array mask of length n_interactions.
            load_in_memory: Whether to load the entire HDF5 content into memory.
                            Use with caution for large files. Defaults to False.
        """
        if not h5_path.exists():
            raise FileNotFoundError(f"DTI H5torch file not found: {h5_path}")

        self.h5_path = h5_path
        logger.info(f"Initializing DTIDataset from: {self.h5_path}")

        # Initialize the underlying h5torch.Dataset with COO sampling
        # It will handle the subsetting internally
        super().__init__(
            file=str(h5_path),
            sampling='coo',
            subset=subset,
            in_memory=load_in_memory
            # sample_processor could be used for restructuring, but doing it here is clearer
        )

        # Store interaction value key (depends on dtype_load used during creation)
        self._y_key = 'central' # h5torch convention for COO sample value

        logger.info(f"DTIDataset initialized with {len(self)} interactions.")


    def __getitem__(self, idx: int) -> Dict:
        """Retrieves and structures the data for a given interaction index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")

        # Get the raw flat sample dictionary from h5torch.Dataset (using COO sampling)
        raw_sample = super().__getitem__(idx)

        # DEBUG: Print raw keys/sample
        # logger.debug(f"Index {idx}: Raw keys from h5torch.Dataset: {list(raw_sample.keys())}")
        # logger.debug(f"Index {idx}: Raw sample: {raw_sample}")

        # Structure the output dictionary
        structured_sample = {
            'drug': {},
            'target': {},
            'interaction': {},
            'metadata': {}
        }

        # Populate the structured dictionary from the flat raw sample
        for key, value in raw_sample.items():
            if key == self._y_key:
                # Determine the interaction column name used during creation
                # Assume it's 'Y' if not explicitly stored, or read from root attr if available
                interaction_col_name = self.f.attrs.get("interaction_col", "Y")
                structured_sample['interaction'][interaction_col_name] = value
            elif key.startswith('0/'): # Drug features (aligned axis 0)
                feature_name = key.split('/', 1)[1]
                structured_sample['drug'][feature_name] = value
            elif key.startswith('1/'): # Target features (aligned axis 1)
                feature_name = key.split('/', 1)[1]
                structured_sample['target'][feature_name] = value
            elif key.startswith('unstructured/'): # Interaction metadata
                metadata_name = key.split('/', 1)[1]
                # Check if it's an additional Y value or general metadata
                # This relies on naming conventions used in create_dti_h5torch
                if metadata_name.startswith('Y_'):
                     structured_sample['interaction'][metadata_name] = value
                else:
                     structured_sample['metadata'][metadata_name] = value
            else:
                # Should not happen with COO sampling returning central+aligned+unstructured
                logger.warning(f"Unexpected key in raw COO sample: {key}")

        # Ensure essential IDs are present
        drug_id_col = self.f.attrs.get("drug_id_col", "Drug_ID")
        target_id_col = self.f.attrs.get("target_id_col", "Target_ID")
        if drug_id_col not in structured_sample['drug']:
             logger.error(f"Drug ID column '{drug_id_col}' not found in sample drug features for index {idx}.")
             # Handle error appropriately, e.g., raise KeyError or return partial data
        if target_id_col not in structured_sample['target']:
             logger.error(f"Target ID column '{target_id_col}' not found in sample target features for index {idx}.")

        return structured_sample

    # close() and __del__() are inherited from h5torch.Dataset if needed (especially if not in_memory)