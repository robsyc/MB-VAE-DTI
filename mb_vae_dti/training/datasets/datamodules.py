"""
PyTorch Lightning DataModules for DTI datasets.

Provides efficient data loading and preprocessing for drug-target interaction datasets.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Union, List, Any, Literal
import logging
import torch
import numpy as np

from mb_vae_dti.training.datasets.h5datasets import DTIDataset, PretrainDataset


logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """
    Custom collate function to handle DTI dataset samples with None Y values.
    
    Uses masking approach for Y values where some scores may be None.
    """
    collated = {
        'id': [],
        'y': {},
        'drug': {},
        'target': {}
    }
    
    # Simple ID collation
    collated['id'] = torch.tensor([item['id'] for item in batch])
    
    # Binary values - always present
    collated['y']['Y'] = torch.tensor([item['y']['Y'] for item in batch], dtype=torch.float32)

    # Real-valued Y values with masking - this is key!
    for key in ['Y_KIBA', 'Y_pKd', 'Y_pKi']:
        values = [item['y'].get(key, None) for item in batch]

        tensor_values = []
        mask_values = []
        
        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                tensor_values.append(0.0)  # Fill with 0, mask will indicate invalid
                mask_values.append(False)  # Invalid
            else:
                tensor_values.append(float(v))
                mask_values.append(True)   # Valid
        
        collated['y'][key] = torch.tensor(tensor_values, dtype=torch.float32)
        collated['y'][f'{key}_mask'] = torch.tensor(mask_values, dtype=torch.bool)
    
    # Drug features - always present, no dummy data needed
    collated['drug']['features'] = {}
    for key in ['FP-Morgan', 'EMB-BiomedGraph', 'EMB-BiomedImg', 'EMB-BiomedText']:
        features = [item['drug']['features'][key] for item in batch]
        collated['drug']['features'][key] = torch.stack([
            torch.from_numpy(np.array(f, dtype=np.float32)) for f in features
        ])
    
    # Target features - always present, no dummy data needed  
    collated['target']['features'] = {}
    for key in ['FP-ESP', 'EMB-ESM', 'EMB-NT']:
        features = [item['target']['features'][key] for item in batch]
        collated['target']['features'][key] = torch.stack([
            torch.from_numpy(np.array(f, dtype=np.float32)) for f in features
        ])
    
    # Optional: Include other fields if needed (representations, IDs)
    collated['drug']['representations'] = {
        'smiles': [item['drug']['representations']['SMILES'] for item in batch]
    }
    
    return collated


class DTIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for DTI datasets.
    
    Handles train/val/test splits and efficient batch loading.
    Supports both random and cold-drug splits.
    
    Note: This module loads ALL available features. 
    Feature selection is handled by the model during forward pass.
    """
    
    def __init__(
        self,
        h5_path: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        drop_last: bool = False,
        # Dataset specific parameters
        split_type: Literal["split_rand", "split_cold"] = "split_rand",
        provenance_cols: Optional[List[Literal["in_DAVIS", "in_KIBA", "in_Metz", "in_BindingDB_Kd", "in_BindingDB_Ki"]]] = None,
        load_in_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.h5_path = Path(h5_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        
        self.split_type = split_type
        self.provenance_cols = provenance_cols
        self.load_in_memory = load_in_memory
        
        # Datasets will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """Verify data file exists."""
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.h5_path}")
            
    def setup(self, stage: Optional[str] = None):
        """Initialize datasets for each split."""
        
        # Common filter setup
        base_filters = {
            'split_col': self.split_type,
            'provenance_cols': self.provenance_cols
        } if self.provenance_cols else {'split_col': self.split_type}
        
        if stage == "fit" or stage is None:
            # Training dataset
            train_filters = {**base_filters, 'split_value': 'train'}
            self.train_dataset = DTIDataset(
                h5_path=self.h5_path,
                subset_filters=train_filters,
                load_in_memory=self.load_in_memory
            )
            
            # Validation dataset
            val_filters = {**base_filters, 'split_value': 'val'}
            self.val_dataset = DTIDataset(
                h5_path=self.h5_path,
                subset_filters=val_filters,
                load_in_memory=self.load_in_memory
            )
            
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Val dataset size: {len(self.val_dataset)}")
            
        if stage == "test" or stage is None:
            # Test dataset
            test_filters = {**base_filters, 'split_value': 'test'}
            self.test_dataset = DTIDataset(
                h5_path=self.h5_path,
                subset_filters=test_filters,
                load_in_memory=self.load_in_memory
            )
            logger.info(f"Test dataset size: {len(self.test_dataset)}")
            
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
            collate_fn=custom_collate_fn
        )
        
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=custom_collate_fn
        )
        
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=custom_collate_fn
        )
        
    def get_feature_dims(self) -> Dict[str, Dict[str, int]]:
        """
        Get dimensions of all features in the dataset.
        
        Returns:
            Dictionary with drug and target feature dimensions
        """
        if self.train_dataset is None:
            self.setup("fit")
            
        # Get a sample to determine feature dimensions
        sample = self.train_dataset[0]
        
        feature_dims = {
            'drug': {},
            'target': {}
        }
        
        # Extract all drug feature dimensions
        for feat_name, feat_value in sample['drug']['features'].items():
            if hasattr(feat_value, 'shape'):
                feature_dims['drug'][feat_name] = feat_value.shape[-1]
            else:
                feature_dims['drug'][feat_name] = 1
                    
        # Extract all target feature dimensions
        for feat_name, feat_value in sample['target']['features'].items():
            if hasattr(feat_value, 'shape'):
                feature_dims['target'][feat_name] = feat_value.shape[-1]
            else:
                feature_dims['target'][feat_name] = 1
                    
        return feature_dims
        
    def get_available_scores(self) -> List[str]:
        """
        Get list of available DTI scores in the dataset.
        
        Returns:
            List of score names (e.g., ['Y_pKd', 'Y_KIBA'])
        """
        if self.train_dataset is None:
            self.setup("fit")
            
        sample = self.train_dataset[0]
        return [k for k in sample['y'].keys() if k != 'Y']  # Exclude binary label 


class PretrainDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for pretrain datasets (drug or target branch pretraining).
    
    Handles train/val splits for contrastive learning pretraining.
    No test split for pretraining phase.
    
    Note: This module loads ALL available features for the specific entity type.
    Feature selection is handled by the model during forward pass.

    Note: These datasets are very large, so loading in memory may not be feasible.
    """
    
    def __init__(
        self,
        h5_path: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        drop_last: bool = False,
        load_in_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.h5_path = Path(h5_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        self.load_in_memory = load_in_memory
        
        # Datasets will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        # No test dataset for pretraining
        
    def prepare_data(self):
        """Verify data file exists."""
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.h5_path}")
            
    def setup(self, stage: Optional[str] = None):
        """Initialize datasets for each split."""
        
        if stage == "fit" or stage is None:
            # Training dataset (is_train=True)
            train_filters = {'split_col': 'is_train', 'split_value': True}
            self.train_dataset = PretrainDataset(
                h5_path=self.h5_path,
                subset_filters=train_filters,
                load_in_memory=self.load_in_memory
            )
            
            # Validation dataset (is_train=False)
            val_filters = {'split_col': 'is_train', 'split_value': False}
            self.val_dataset = PretrainDataset(
                h5_path=self.h5_path,
                subset_filters=val_filters,
                load_in_memory=self.load_in_memory
            )
            
            logger.info(f"Pretrain train dataset size: {len(self.train_dataset)}")
            logger.info(f"Pretrain val dataset size: {len(self.val_dataset)}")
            
        # No test setup for pretraining
            
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0
        )
        
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
        
    def test_dataloader(self):
        """Not used for pretraining."""
        return None
        
    def get_feature_dims(self) -> Dict[str, int]:
        """
        Get dimensions of all features in the dataset.
        
        Returns:
            Dictionary with feature dimensions
        """
        if self.train_dataset is None:
            self.setup("fit")
            
        # Get a sample to determine feature dimensions
        sample = self.train_dataset[0]
        
        feature_dims = {}
        
        # Extract all feature dimensions
        for feat_name, feat_value in sample['features'].items():
            if hasattr(feat_value, 'shape'):
                feature_dims[feat_name] = feat_value.shape[-1]
            else:
                feature_dims[feat_name] = 1
                    
        return feature_dims
        
    def get_available_features(self) -> List[str]:
        """
        Get list of available features in the dataset.
        
        Returns:
            List of feature names
        """
        if self.train_dataset is None:
            self.setup("fit")
            
        sample = self.train_dataset[0]
        return list(sample['features'].keys())
        
    def get_available_representations(self) -> List[str]:
        """
        Get list of available representations in the dataset.
        
        Returns:
            List of representation names
        """
        if self.train_dataset is None:
            self.setup("fit")
            
        sample = self.train_dataset[0]
        return list(sample['representations'].keys()) 


