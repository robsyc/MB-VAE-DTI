"""
Baseline DTI PyTorch Lightning module with
- Unimodal encoders (ResidualEncoder or TransformerEncoder)
- Dot-product prediction of single DTI interaction score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Any, Tuple, Literal, List
import logging

from .utils import AbstractDTIModel

from mb_vae_dti.training.models import ResidualEncoder, TransformerEncoder
from mb_vae_dti.training.metrics import RealDTIMetrics


logger = logging.getLogger(__name__)


class BaselineDTIModel(AbstractDTIModel):
    """
    Baseline two-tower DTI model with dot-product prediction.
    
    Architecture:
    - Drug encoder: Single feature → Encoder → Embedding
    - Target encoder: Single feature → Encoder → Embedding  
    - Prediction: Dot product of embeddings → Score
    
    This is the simplest configuration for benchmarking.
    """
    def __init__(
        self,

        phase: Literal["finetune"], # baseline model only supports finetune phase
        finetune_score: Literal["Y_pKd", "Y_KIBA"], # required for finetune phase

        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]],

        drug_features: Dict[
            Literal["FP-Morgan", "EMB-BioMedGraph", "EMB-BioMedImg", "EMB-BioMedText"], 
            int],  # dict mapping feature name to input dimension
        target_features: Dict[
            Literal["FP-ESM", "FP-NT", "FP-ESP"], 
            int
        ],

        embedding_dim: int,
        hidden_dim: int,
        factor: int,
        n_layers: int,
        activation: nn.Module,
        dropout: float,
        bias: bool,
        
        encoder_type: Literal["resnet", "transformer"],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_dtype = torch.float32
        self.phase = phase
        self.finetune_score = finetune_score

        assert len(drug_features) == 1 and len(target_features) == 1, \
            "Baseline model only supports single feature for drug and target"

        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder
        
        self.drug_encoder = encoder_type(
            input_dim=list(drug_features.values())[0],
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            n_layers=n_layers,
            factor=factor,
            dropout=dropout,
            bias=bias,
            activation=activation
        )
        
        self.target_encoder = encoder_type(
            input_dim=list(target_features.values())[0],
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            n_layers=n_layers,
            factor=factor,
            dropout=dropout,
            bias=bias,
            activation=activation
        )

        # Metrics - using RealDTIMetrics for single score prediction
        self.train_metrics_dti = RealDTIMetrics(prefix="train/")
        self.val_metrics_dti = RealDTIMetrics(prefix="val/")
        self.test_metrics_dti = RealDTIMetrics(prefix="test/")

    def forward(
        self, 
        drug_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the two-tower model.
        
        Args:
            drug_features: Drug feature tensor [batch_size, drug_input_dim]
            target_features: Target feature tensor [batch_size, target_input_dim]
            
        Returns:
            predictions: dot-product predictions of DTI score (batch_size)
            outputs: intermediate embeddings & attention weights
        """
        # Encode drug and target
        drug_embedding = self.drug_encoder(drug_features)       # (batch_size, embedding_dim)
        target_embedding = self.target_encoder(target_features) # (batch_size, embedding_dim)
        
        # Predict DTI score w/ dot product
        score_pred = torch.sum(drug_embedding * target_embedding, dim=-1) # (batch_size)

        return score_pred.squeeze(-1), {
            "drug_embedding": drug_embedding,
            "target_embedding": target_embedding
        }
        
    def _common_step(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Common step logic shared across train/val/test.

        Returns three tensors:
            - predictions: dot-product dti_preds of finetune score (batch_size)
            - targets: true dti_targets after masking
            - losses: standard MSE loss (accuracy)
        """

        # Extract features and targets
        drug_features, target_features = self._get_features_from_batch(batch)
        dti_targets = self._get_target_from_batch(batch, self.hparams.finetune_score)
        dti_masks = self._get_target_mask_from_batch(batch, self.hparams.finetune_score)
        
        # Forward pass
        dti_preds, outputs = self.forward(drug_features, target_features)
        
        # Mask predictions and targets
        dti_preds = dti_preds[dti_masks]
        dti_targets = dti_targets[dti_masks]
        
        return dti_preds, dti_targets, F.mse_loss(dti_preds, dti_targets)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step."""
        dti_preds, dti_targets, loss = self._common_step(batch)

        # Update metrics
        self.train_metrics_dti.update(
            dti_preds, 
            dti_targets)
        
        # Logging
        self.log(f"train/loss", loss)
        
        return loss
        
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        dti_preds, dti_targets, loss = self._common_step(batch)
            
        self.val_metrics_dti.update(
            dti_preds, 
            dti_targets)
        
        self.log("val/loss", loss)
        return loss
            
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step - similar to validation."""
        dti_preds, dti_targets, loss = self._common_step(batch)
        
        self.test_metrics_dti.update(
            dti_preds, 
            dti_targets)
        
        self.log("test/loss", loss)
        return loss
        
    def on_train_epoch_end(self):
        """Compute and log training metrics."""
        train_metrics_dti = self.train_metrics_dti.compute()
        for name, value in train_metrics_dti.items():
            self.log(name, value)
        self.train_metrics_dti.reset()
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        val_metrics_dti = self.val_metrics_dti.compute()
        for name, value in val_metrics_dti.items():
            self.log(name, value)
        self.val_metrics_dti.reset()
    
    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        test_metrics_dti = self.test_metrics_dti.compute()
        for name, value in test_metrics_dti.items():
            self.log(name, value)
        self.test_metrics_dti.reset() 