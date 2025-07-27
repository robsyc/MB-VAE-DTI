"""
Utility classes and functions for DTI PyTorch Lightning modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Any, Tuple, Literal, List, Union
import logging
from pathlib import Path


# Data containers - available for structured data flow
from mb_vae_dti.training.data_containers import (
    GraphData, EmbeddingData, PredictionData, LossData, BatchData
)

# Diffusion utilities - only imported when needed for complex models
from mb_vae_dti.training.diffusion.utils import PlaceHolder, to_dense


logger = logging.getLogger(__name__)


# PARENT PYTORCH LIGHTNING MODULE

class AbstractDTIModel(pl.LightningModule):
    """
    Abstract base class for DTI PyTorch Lightning modules.
    
    Provides common functionality for:
    - Activation parsing from string names
    - Optimizer and scheduler configuration
    - Metrics setup and epoch handling
    - Feature extraction from batches
    - Checkpoint loading
    """
    
    def __init__(self):
        super().__init__()
        # Initialize metrics to None - will be set up by subclass
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        
        # Additional metrics for diffusion models (set by subclass if needed)
        self.train_diffusion_metrics = None  # BCEs & validity
        self.val_diffusion_metrics = None    # nll & validity
        self.test_diffusion_metrics = None   # nll & validity

    @staticmethod
    def parse_activation(activation: Union[str, nn.Module]) -> nn.Module:
        """
        Parse activation from string name or return the module if already instantiated.
        
        Args:
            activation: Either a string name (e.g., "relu", "gelu") or PyTorch module
            
        Returns:
            PyTorch activation module
            
        Raises:
            ValueError: If activation string is not recognized
        """
        if isinstance(activation, nn.Module):
            return activation
        
        if isinstance(activation, str):
            activation_map = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "elu": nn.ELU(),
                "leaky_relu": nn.LeakyReLU(),
                "selu": nn.SELU(),
                "swish": nn.SiLU(),  # SiLU is PyTorch's implementation of Swish
                "silu": nn.SiLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "mish": nn.Mish(),
                "glu": nn.GLU(),
            }
            
            activation_lower = activation.lower()
            if activation_lower in activation_map:
                return activation_map[activation_lower]
            else:
                available_activations = list(activation_map.keys())
                raise ValueError(
                    f"Unknown activation '{activation}'. "
                    f"Available activations: {available_activations}"
                )
        
        raise TypeError(
            f"Activation must be either a string or nn.Module, got {type(activation)}"
        )


    def configure_optimizers(self):
        """Configure optimizer and scheduler using utility function."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
            amsgrad=True  # More stable convergence
        )
        
        # Return just optimizer for constant scheduler
        if self.hparams.scheduler == "const" or self.hparams.scheduler is None:
            return optimizer
        
        # Configure scheduler with solid defaults
        if self.hparams.scheduler == "step":
            # Step every 30 epochs, reduce by factor of 10
            # Good for longer training runs (100+ epochs)
            scheduler_obj = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            
        elif self.hparams.scheduler == "one_cycle":
            # OneCycleLR with 30% warmup period
            # Excellent for faster convergence
            if self.trainer is None:
                raise ValueError("Trainer must be provided for OneCycleLR scheduler")
            
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler_obj = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(self.hparams.learning_rate),
                total_steps=stepping_batches,
                pct_start=0.3,  # 30% of cycle for warmup
                anneal_strategy='cos'  # Cosine annealing
            )
            
            # OneCycleLR needs step-wise scheduling
            return [optimizer], [{
                'scheduler': scheduler_obj,
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1,
            }]
            
        elif self.hparams.scheduler == "cosine":
            # Cosine annealing with small minimum LR
            # Good for fine-tuning and avoiding complete stagnation
            T_max = self.trainer.max_epochs if self.trainer else self.hparams.max_epochs
            scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=float(self.hparams.learning_rate) * 0.01  # 1% of initial LR as minimum
            )
            
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
        
        # Return optimizer and scheduler for epoch-based schedulers
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_obj,
                "monitor": "val/loss",  # Monitor validation loss
                "interval": "epoch",
                "frequency": 1,
            }
        } 


    def setup_metrics(self, phase: str, finetune_score: str = None):
        """
        Set up metrics based on training phase.
        
        Args:
            phase: Training phase ("pretrain_drug", "pretrain_target", "train", "finetune")
            finetune_score: Score to fine-tune on (required for finetune phase)
        """
        from mb_vae_dti.training.metrics import DTIMetricsCollection, RealDTIMetrics
        
        if phase in ["pretrain_drug", "pretrain_target"]:
            # No specific metrics for pretraining - just log contrastive loss
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
        elif phase == "train":
            # Multi-score metrics for general training
            self.train_metrics = DTIMetricsCollection(
                include_binary=True,
                include_real=True,
                real_score_names=["pKd", "pKi", "KIBA"],
                prefix="train/"
            )
            self.val_metrics = DTIMetricsCollection(
                include_binary=True,
                include_real=True,
                real_score_names=["pKd", "pKi", "KIBA"],
                prefix="val/"
            )
            self.test_metrics = DTIMetricsCollection(
                include_binary=True,
                include_real=True,
                real_score_names=["pKd", "pKi", "KIBA"],
                prefix="test/"
            )
        else:  # finetune
            # Single-score metrics for fine-tuning
            if finetune_score is None:
                raise ValueError("finetune_score must be specified for finetune phase")

            self.train_metrics = RealDTIMetrics(prefix=f"train/{finetune_score}_")
            self.val_metrics = RealDTIMetrics(prefix=f"val/{finetune_score}_")
            self.test_metrics = RealDTIMetrics(prefix=f"test/{finetune_score}_")


    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def freeze_encoders(self):
        if hasattr(self, "drug_encoder"):
            self._freeze_module(self.drug_encoder)
        if hasattr(self, "target_encoder"):
            self._freeze_module(self.target_encoder)
        if hasattr(self, "drug_encoders"):
            self._freeze_module(self.drug_encoders)
        if hasattr(self, "target_encoders"):
            self._freeze_module(self.target_encoders)

    def unfreeze_encoders(self):
        if hasattr(self, "drug_encoder"):
            self._unfreeze_module(self.drug_encoder)
        if hasattr(self, "target_encoder"):
            self._unfreeze_module(self.target_encoder)
        if hasattr(self, "drug_encoders"):
            self._unfreeze_module(self.drug_encoders)
        if hasattr(self, "target_encoders"):
            self._unfreeze_module(self.target_encoders)

    def load_pretrained_weights(self, checkpoint_path: str, prefix: str = None) -> None:
        """
        Load pretrained weights from checkpoint with smart matching.
        Use prefix to load only a subset of the weights (e.g. "drug_" or "target_")
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model_dict = self.state_dict()
        pretrained_dict = {}

        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                if prefix is None or k.startswith(prefix):
                    pretrained_dict[k] = v
                    logger.debug(f"Loaded weight: {k}")
            else:
                logger.warning(f"Skipping weight: {k} (shape mismatch or not found)")
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        logger.info(f"Loaded {len(pretrained_dict)} pretrained weights")

        # Optionally freeze encoder weights for fine-tuning
        if self.hparams.phase == "finetune":
            self.freeze_encoders()
    

    def _get_features_from_batch(
        self, 
        batch: Dict[str, Any]
    ) -> Union[
        Tuple[List[torch.Tensor], List[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        """
        Extract and prepare drug and target features from batch.
        Returns Tuple of (drug_feats, target_feats) (child classes should handle list)
        
        One of the items may be None when pretraining drug or target
        """        
        # Determine source based on phase
        if self.hparams.phase == "pretrain_drug":
            drug_feats, target_feats = [
                batch["features"][feat_name]
                for feat_name in self.hparams.drug_features.keys()
            ], []
        elif self.hparams.phase == "pretrain_target":
            drug_feats, target_feats = [], [
                batch["features"][feat_name]
                for feat_name in self.hparams.target_features.keys()
            ]
        else:
            # DTI training/fine-tuning
            drug_feats, target_feats = [
                batch["drug"]["features"][feat_name]
                for feat_name in self.hparams.drug_features.keys()
            ], [
                batch["target"]["features"][feat_name]
                for feat_name in self.hparams.target_features.keys()
            ]
        
        # Convert to single tensor if only one feature
        drug_feats = drug_feats[0] if len(drug_feats) == 1 else drug_feats
        target_feats = target_feats[0] if len(target_feats) == 1 else target_feats
        
        return drug_feats, target_feats
    
    def _get_smiles_from_batch(self, batch: Dict[str, Any]) -> List[str]:
        """Extract SMILES from batch."""
        # we don't actually use this anymore, SMILES to graph conversion is done in the dataloader
        if self.hparams.phase == "pretrain_drug":
            return batch["representations"]["smiles"] # pretrain dataloader
        else:
            return batch["drug"]["representations"]["smiles"]
    
    def _get_fingerprints_from_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and prepare drug and target fingerprints from batch for contrastive loss.
        Returns Tuple of (drug_fp, target_fp)
        One of the items may be None when pretraining drug or target
        """
        if self.hparams.phase == "pretrain_drug":
            return batch["features"]["FP-Morgan"], None
        elif self.hparams.phase == "pretrain_target":
            return None, batch["features"]["FP-ESP"]
        else:
            return batch["drug"]["features"]["FP-Morgan"], batch["target"]["features"]["FP-ESP"]
    

    def _get_targets_from_batch(
            self, 
            batch: Dict[str, Any]
    ) -> Union[
        torch.Tensor,
        Dict[Literal["Y", "Y_pKd", "Y_pKi", "Y_KIBA"], torch.Tensor]
    ]:
        """Get target scores from batch (single or multi-score)."""
        if self.hparams.phase == "finetune":
            return batch["y"][self.hparams.finetune_score].squeeze(-1)
        return {
            "Y": batch["y"]["Y"].squeeze(-1),
            "Y_pKd": batch["y"]["Y_pKd"].squeeze(-1),
            "Y_KIBA": batch["y"]["Y_KIBA"].squeeze(-1),
            "Y_pKi": batch["y"]["Y_pKi"].squeeze(-1)
        }
    
    def _get_masks_from_batch(
            self, 
            batch: Dict[str, Any]
    ) -> Dict[Literal["Y_pKd", "Y_pKi", "Y_KIBA"], torch.Tensor]:
        """Get target masks from batch (single or multi-score)."""
        if self.hparams.phase == "finetune":
            return batch["y"][f"{self.hparams.finetune_score}_mask"].squeeze(-1)
        return {
            "Y_pKd": batch["y"]["Y_pKd_mask"].squeeze(-1),
            "Y_KIBA": batch["y"]["Y_KIBA_mask"].squeeze(-1),
            "Y_pKi": batch["y"]["Y_pKi_mask"].squeeze(-1)
        }

    def _get_targets_masks_from_batch(
            self, 
            batch: Dict[str, Any]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Dict[Literal["Y", "Y_pKd", "Y_pKi", "Y_KIBA"], torch.Tensor], Dict[Literal["Y_pKd", "Y_pKi", "Y_KIBA"], torch.Tensor]]
    ]:
        """Get target scores and their masks from batch (single or multi-score)."""
        targets = self._get_targets_from_batch(batch)
        masks = self._get_masks_from_batch(batch)
        return targets, masks
    

    def _extract_graph_data(self, batch: Dict[str, Any]) -> 'GraphData':
        """
        Extract graph data from batch and convert to dense representation.
        
        Handles both DTI batches (batch["drug"]["G"]) and pretrain batches (batch["G"]).
        Only available when diffusion modules are imported.

        NOTE: graph data X & E are NOT masked, but the node_mask is returned
        """
        if to_dense is None:
            raise ImportError("Graph data extraction requires diffusion modules")
            
        # Get PyG batch based on batch type
        if "drug" in batch and "G" in batch["drug"]:
            pyg_batch = batch["drug"]["G"]  # DTI batch
        elif "G" in batch:
            pyg_batch = batch["G"]  # Pretrain batch
        else:
            raise ValueError("No graph data found in batch")
        
        graph_data = GraphData(pyg_batch=pyg_batch)
        
        # Convert to dense representation
        if pyg_batch is not None:
            dense_data, node_mask = to_dense( # TODO: do we want to omit it returning y ?
                pyg_batch.x, 
                pyg_batch.edge_index, 
                pyg_batch.edge_attr, 
                pyg_batch.batch
            )
            graph_data.X = dense_data.X
            graph_data.E = dense_data.E
            # graph_data.y = dense_data.y  # Will be replaced with drug_embedding later
            graph_data.node_mask = node_mask
        
        return graph_data
    

    def on_train_epoch_end(self):
        """Compute and log training metrics."""
        if self.train_metrics is not None:
            train_metrics = self.train_metrics.compute()
            for name, value in train_metrics.items():
                self.log(name, value)
            self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        if self.val_metrics is not None:
            val_metrics = self.val_metrics.compute()
            for name, value in val_metrics.items():
                self.log(name, value)
            self.val_metrics.reset()
    
    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        if self.test_metrics is not None:
            test_metrics = self.test_metrics.compute()
            for name, value in test_metrics.items():
                self.log(name, value)
            self.test_metrics.reset()