"""
Utility functions for DTI PyTorch Lightning modules
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Any, Tuple, Literal
import logging

from mb_vae_dti.training.models import ResidualEncoder, TransformerEncoder
from mb_vae_dti.training.metrics import RealDTIMetrics
from .optimizer_utils import configure_optimizer_and_scheduler


logger = logging.getLogger(__name__)


class AbstractDTIModel(pl.LightningModule):
    """
    Abstract DTI model for downsteam configurations.
    
    Architecture:
    - Drug encoder: Single feature → Encoder → Embedding
    - Target encoder: Single feature → Encoder → Embedding  
    - Prediction: Dot product of embeddings → Score
    
    This is the simplest configuration for benchmarking.
    """
    def __init__(
        self,
        # Architecture parameters
        embedding_dim: int,
        drug_input_dim: int,   # dims of Morgan fingerprints
        target_input_dim: int, # dims of ESP fingerprints
        encoder_type: Literal["resnet", "transformer"],
        encoder_kwargs: Optional[Dict],
        
        # Training parameters
        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]],
        
        # Data parameters
        finetune_score: Literal["Y_pKd", "Y_KIBA"],
        drug_feature: Literal["FP-Morgan", "EMB-BiomedGraph", "EMB-BiomedImg", "EMB-BiomedText"],
        target_feature: Literal["FP-ESP", "EMB-ESM", "EMB-NT"],
    ):
        super().__init__()
        self.save_hyperparameters()
        
        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder
        
        self.drug_encoder = encoder_type(
            input_dim=drug_input_dim,
            output_dim=embedding_dim,
            **encoder_kwargs
        )
        
        self.target_encoder = encoder_type(
            input_dim=target_input_dim,
            output_dim=embedding_dim,
            **encoder_kwargs
        )
            
        # Loss functions (only MSE on DTI interaction score)
        self.mse_loss = nn.MSELoss()
        
        # Metrics - using RealDTIMetrics for single score prediction
        self.train_metrics = RealDTIMetrics(prefix="train/")
        self.val_metrics = RealDTIMetrics(prefix="val/")
        self.test_metrics = RealDTIMetrics(prefix="test/")

    def configure_optimizers(self):
        """Configure optimizer and scheduler using utility function."""
        return configure_optimizer_and_scheduler(
            model_parameters=self.parameters(),
            learning_rate=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            scheduler=self.hparams.scheduler,
            trainer=self.trainer
        )

    def forward(
        self, 
        drug_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the two-tower model.
        
        Args:
            drug_features: Drug feature tensor [batch_size, drug_input_dim]
            target_features: Target feature tensor [batch_size, target_input_dim]
            
        Returns:
            Dictionary containing predictions and embeddings
        """
        # Encode drug and target
        drug_embedding = self.drug_encoder(drug_features)       # (batch_size, embedding_dim)
        target_embedding = self.target_encoder(target_features) # (batch_size, embedding_dim)
        
        # Predict DTI score w/ dot product
        score_pred = torch.sum(drug_embedding * target_embedding, dim=-1) # (batch_size)
        
        outputs = {
            "score_pred": score_pred,
            "drug_embedding": drug_embedding,
            "target_embedding": target_embedding,
        }
        
        return outputs
        
    def _get_features_from_batch(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and prepare features from batch."""
        drug_feat = batch["drug"]["features"].get(self.hparams.drug_feature)
        target_feat = batch["target"]["features"].get(self.hparams.target_feature)

        if drug_feat is None or target_feat is None:
            raise ValueError("Could not find appropriate features in batch")
        
        return drug_feat, target_feat
        
    def _common_step(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Common step logic shared across train/val/test.
        
        Args:
            batch: Batch data
            
        Returns:
            Tuple of (predictions, targets, loss) for valid samples only
        """
        # Get features
        drug_features, target_features = self._get_features_from_batch(batch)
        
        # Forward pass
        outputs = self(drug_features, target_features)
        
        # Get target score
        target_score_value = batch["y"][self.hparams.finetune_score]
        
        # Filter out samples without target score using mask from collate function
        valid_mask = batch["y"][f"{self.hparams.finetune_score}_mask"]
        if not valid_mask.any():
            return None, None, None
            
        # Get valid predictions and targets
        valid_predictions = outputs["score_pred"][valid_mask]
        valid_targets = target_score_value[valid_mask]
        
        # Compute regression loss
        score_loss = self.mse_loss(valid_predictions, valid_targets)
        
        return valid_predictions, valid_targets, score_loss
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step."""
        valid_predictions, valid_targets, score_loss = self._common_step(batch)
        if valid_predictions is None:  # No valid samples
            return None
            
        # Update metrics
        self.train_metrics.update(valid_predictions, valid_targets)
        
        # Logging
        self.log(f"train/loss", score_loss)
        
        return score_loss
        
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        valid_predictions, valid_targets, score_loss = self._common_step(batch)
        if valid_predictions is None:  # No valid samples
            return None
            
        # Update metrics
        self.val_metrics.update(valid_predictions, valid_targets)
            
        # Logging
        self.log("val/loss", score_loss)
        
        return score_loss
            
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step - similar to validation."""
        valid_predictions, valid_targets, score_loss = self._common_step(batch)
        if valid_predictions is None:  # No valid samples
            return None
        
        # Update metrics
        self.test_metrics.update(valid_predictions, valid_targets)
        
        # Logging
        self.log("test/loss", score_loss)
        
    def on_train_epoch_end(self):
        """Compute and log training metrics."""
        train_metrics = self.train_metrics.compute()
        for name, value in train_metrics.items():
            self.log(name, value)
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        val_metrics = self.val_metrics.compute()
        for name, value in val_metrics.items():
            self.log(name, value)
        self.val_metrics.reset()
    
    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        test_metrics = self.test_metrics.compute()
        for name, value in test_metrics.items():
            self.log(name, value)
        self.test_metrics.reset() 