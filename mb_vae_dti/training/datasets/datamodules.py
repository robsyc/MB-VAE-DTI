"""
PyTorch Lightning DataModules for DTI datasets.

Provides efficient data loading and preprocessing for drug-target interaction datasets.
"""

import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Union, List, Any, Literal
import logging
import torch
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import BondType as BT
from functools import partial

from mb_vae_dti.training.datasets.h5datasets import DTIDataset, PretrainDataset


logger = logging.getLogger(__name__)


class SmilesToPyG:
    """
    Efficient helper for converting a SMILES string to a PyTorch Geometric Data object,
    using only minimal node and edge features:
      - Node feature: atom type as integer index (from atom_types list)
      - Edge feature: bond type as integer index (from bond_types list)
    """
    def __init__(
            self, 
            atom_encoder = {"C": 0, "O": 1, "P": 2, "N": 3, "S": 4, "Cl": 5, "F": 6, "H": 7}, 
            bond_encoder = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
            ):
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder

    def smiles_to_mol(self, smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles('')
        return mol

    def smiles_to_pyg(self, smiles: str) -> Data:
        mol = self.smiles_to_mol(smiles)

        # Node features: atom type index
        try:
            x_indices = [self.atom_encoder.get(atom.GetSymbol(), 0) for atom in mol.GetAtoms()]  # Default to 0 for unknown atoms
        except:
            x_indices = [0]  # Fallback
        
        x_indices = torch.tensor(x_indices, dtype=torch.long)
        
        num_atom_types = max(self.atom_encoder.values()) + 1
        x = torch.nn.functional.one_hot(x_indices, num_classes=num_atom_types).float()

        # Edge features: bond type index, undirected edges
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type_idx = self.bond_encoder.get(bond.GetBondType(), 0)  # Default to 0 for unknown bond types
            edge_indices += [[i, j], [j, i]]
            edge_attrs += [[bond_type_idx], [bond_type_idx]]

        if edge_indices:  # Only if we have edges
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr_indices = torch.tensor(edge_attrs, dtype=torch.long).squeeze(-1)

            num_bond_types = max(self.bond_encoder.values()) + 1
            edge_attr = torch.nn.functional.one_hot(edge_attr_indices, num_classes=num_bond_types).float()

            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
        else:
            # No edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            num_bond_types = max(self.bond_encoder.values()) + 1
            edge_attr = torch.zeros((0, num_bond_types), dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def smiles_to_pyg_batch(self, smiles: List[str]) -> Batch:
        return Batch.from_data_list([self.smiles_to_pyg(smiles) for smiles in smiles])


def dti_collate_fn(batch, smiles_to_pyg_converter=None):
    """
    Custom collate function to handle DTI dataset samples with None Y values.
    
    Uses masking approach for Y values where some scores may be None.
    
    Args:
        batch: List of dataset items
        smiles_to_pyg_converter: Optional SmilesToPyG instance for converting drug SMILES to PyG objects
    
    Returns:
        Collated batch with optional 'G' key containing PyG batch for drug molecules
    """
    collated = {
        'id': [],
        'y': {},
        'drug': {},
        'target': {}
    }
    
    # Simple ID collation
    collated['id'] = torch.tensor([item['id'] for item in batch])
    
    # Binary interaction values - always present
    collated['y']['Y'] = torch.tensor([item['y']['Y'] for item in batch], dtype=torch.float32)

    # Real-valued Y values with masking
    for key in ['Y_KIBA', 'Y_pKd', 'Y_pKi']:
        values = [item['y'].get(key, None) for item in batch]

        tensor_values = [
            0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            for v in values
        ]
        mask_values = [
            False if (v is None or (isinstance(v, float) and np.isnan(v))) else True
            for v in values
        ]
        
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
    
    # Convert drug SMILES to PyG objects if converter is provided
    if smiles_to_pyg_converter is not None:
        # Check if SMILES representations are available in the batch
        if batch[0]['drug'].get('representations') and 'SMILES' in batch[0]['drug']['representations']:
            collated['drug']['representations'] = {
                'smiles': [item['drug']['representations']['SMILES'] for item in batch]
            }
            smiles = collated['drug']['representations']['smiles']
            try:
                collated['drug']['G'] = smiles_to_pyg_converter.smiles_to_pyg_batch(smiles)
            except Exception as e:
                logger.warning(f"Failed to convert drug SMILES batch to PyG: {e}")
                # Create empty batch as fallback
                empty_graph = Data(
                    x=torch.zeros((1, len(smiles_to_pyg_converter.atom_encoder)), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, len(smiles_to_pyg_converter.bond_encoder)), dtype=torch.float32)
                )
                collated['drug']['G'] = Batch.from_data_list([empty_graph] * len(smiles))
    
    # Optional: Include other fields if needed (representations, IDs)
    # collated['target']['representations'] = {
    #     'dna': [item['target']['representations']['DNA'] for item in batch]
    # }
    
    return collated


def pretrain_collate_fn(batch, smiles_to_pyg_converter=None):
    """
    Custom collate function for pretrain datasets with optional PyG graph support.
    
    Args:
        batch: List of dataset items
        smiles_to_pyg_converter: Optional SmilesToPyG instance for converting SMILES to PyG objects
    
    Returns:
        Collated batch with optional 'G' key containing PyG batch
    """
    collated = {
        'id': [],
        'representations': {},
        'features': {}
    }
    
    # Simple ID collation
    collated['id'] = torch.tensor([item['id'] for item in batch])

    # Collate features (convert to tensors)
    for key in batch[0]['features'].keys():
        features = [item['features'][key] for item in batch]
        collated['features'][key] = torch.stack([
            torch.from_numpy(np.array(f, dtype=np.float32)) for f in features
        ])
        
    
    # Convert SMILES to PyG objects if converter is provided
    if smiles_to_pyg_converter is not None:
        if 'smiles' in batch[0]['representations'].keys():
            collated['representations']['smiles'] = [item['representations']['smiles'] for item in batch]
            smiles = collated['representations']['smiles']
            try:
                collated['G'] = smiles_to_pyg_converter.smiles_to_pyg_batch(smiles)
            except Exception as e:
                logger.warning(f"Failed to convert SMILES batch to PyG: {e}")
                # Create empty batch as fallback
                empty_graph = Data(
                    x=torch.zeros((1, len(smiles_to_pyg_converter.atom_encoder)), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, len(smiles_to_pyg_converter.bond_encoder)), dtype=torch.float32)
                )
                collated['G'] = Batch.from_data_list([empty_graph] * len(smiles))
    
    # Optionally, add representations e.g. smiles, DNA, etc.
    # collated['representations'] = {
    #     'smiles': [item['representations']['aa'] for item in batch]
    # }

    return collated


class DTIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for DTI datasets.
    
    Handles train/val/test splits and efficient batch loading.
    Supports both random and cold-drug splits.
    
    Note: This module loads ALL available features. 
    Feature selection is handled by the model during forward pass.
    
    When return_pyg=True, drug SMILES are converted to PyTorch Geometric 
    graph objects and included in the batch as collated['drug']['G'].
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
        return_pyg: bool = False,
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
        
        # SMILES to PyG converter when return_pyg is enabled
        self.return_pyg = return_pyg
        self.smiles_to_pyg = SmilesToPyG() if return_pyg else None
        
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
        collate_fn = partial(dti_collate_fn, smiles_to_pyg_converter=self.smiles_to_pyg)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn
        )
        
    def val_dataloader(self):
        """Return validation dataloader."""
        collate_fn = partial(dti_collate_fn, smiles_to_pyg_converter=self.smiles_to_pyg)
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn
        )
        
    def test_dataloader(self):
        """Return test dataloader."""
        collate_fn = partial(dti_collate_fn, smiles_to_pyg_converter=self.smiles_to_pyg)
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn
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
            feature_dims['drug'][feat_name] = feat_value.shape[-1]
                    
        # Extract all target feature dimensions
        for feat_name, feat_value in sample['target']['features'].items():
            feature_dims['target'][feat_name] = feat_value.shape[-1]
                    
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
        return_pyg: bool = False,
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
        
        # Smiles to PyG converter when running the discrete diffusion decoder
        self.return_pyg = return_pyg
        self.smiles_to_pyg = SmilesToPyG() if return_pyg else None
        
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
        # Use custom collate function with PyG converter if needed
        collate_fn = None
        if self.return_pyg:
            from functools import partial
            collate_fn = partial(pretrain_collate_fn, smiles_to_pyg_converter=self.smiles_to_pyg)
        else:
            collate_fn = partial(pretrain_collate_fn, smiles_to_pyg_converter=None)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn
        )
        
    def val_dataloader(self):
        """Return validation dataloader."""
        # Use custom collate function with PyG converter if needed
        collate_fn = None
        if self.return_pyg:
            from functools import partial
            collate_fn = partial(pretrain_collate_fn, smiles_to_pyg_converter=self.smiles_to_pyg)
        else:
            collate_fn = partial(pretrain_collate_fn, smiles_to_pyg_converter=None)
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn
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
            feature_dims[feat_name] = feat_value.shape[-1]
                    
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


