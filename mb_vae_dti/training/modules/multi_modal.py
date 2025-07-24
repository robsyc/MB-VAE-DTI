"""
Multi-modal DTI PyTorch Lightning module with
- Two branches with multiple encoders (ResidualEncoder or TransformerEncoder)
- Intra-branch aggregation module (concatenation, attentive or cross-attention)
- Dot-product prediction of single DTI interaction score
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional, Any, Tuple, Literal
import logging

from mb_vae_dti.training.models import (
    ResidualEncoder, TransformerEncoder,
    ConcatAggregator, AttentiveAggregator
)
from mb_vae_dti.training.metrics import RealDTIMetrics
from .utils import AbstractDTIModel


logger = logging.getLogger(__name__)


class MultiModalDTIModel(AbstractDTIModel):
    """
    Multi-modal two-tower DTI model with dot-product prediction.
    
    Architecture:
    - Drug branch: Multiple features → Multiple encoders → Aggregation → Embedding
    - Target branch: Multiple features → Multiple encoders → Aggregation → Embedding
    - Prediction: Dot product of aggregated embeddings → Score
    
    This model can handle multiple input modalities per branch and aggregates them
    using configurable aggregation strategies.
    """
    def __init__(
        self,
        # Architecture parameters
        embedding_dim: int,
        drug_features: Dict[str, int],
        target_features: Dict[str, int],
        encoder_type: Literal["resnet", "transformer"],
        encoder_kwargs: Optional[Dict],

        aggregator_type: Literal["concat", "attentive"],
        aggregator_kwargs: Optional[Dict],
        
        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]],
        
        finetune_score: Literal["Y_pKd", "Y_KIBA"],
    ):
        super().__init__()
        self.save_hyperparameters()
        
        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder

        self.drug_encoders = nn.ModuleDict()
        for feat_name in drug_features:
            self.drug_encoders[feat_name] = encoder_type(
                input_dim=drug_features[feat_name],
                output_dim=encoder_kwargs["hidden_dim"],
                **encoder_kwargs
            )
        
        self.target_encoders = nn.ModuleDict()
        for feat_name in target_features:
            self.target_encoders[feat_name] = encoder_type(
                input_dim=target_features[feat_name],
                output_dim=encoder_kwargs["hidden_dim"],
                **encoder_kwargs
            )
        
        aggregator_class_map = {
            "concat": ConcatAggregator,
            "attentive": AttentiveAggregator
        }
        aggregator_class = aggregator_class_map[aggregator_type]
        
        if len(drug_features) > 1:
            self.drug_aggregator = aggregator_class(
                input_dim=encoder_kwargs["hidden_dim"],
                n_features=len(drug_features),
                output_dim=embedding_dim,
                **aggregator_kwargs
            )
        else:
            self.drug_aggregator = None  # No aggregation needed for single feature
        
        if len(target_features) > 1:
            self.target_aggregator = aggregator_class(
                input_dim=encoder_kwargs["hidden_dim"],
                n_features=len(target_features),
                output_dim=embedding_dim,
                **aggregator_kwargs
            )
        else:
            self.target_aggregator = None  # No aggregation needed for single feature
            
        # Loss functions (only MSE on DTI interaction score)
        self.mse_loss = nn.MSELoss()
        
        # Metrics - using RealDTIMetrics for single score prediction
        self.train_metrics = RealDTIMetrics(prefix="train/")
        self.val_metrics = RealDTIMetrics(prefix="val/")
        self.test_metrics = RealDTIMetrics(prefix="test/")
        
        # Store aggregator type for handling different return formats
        self.aggregator_type = aggregator_type
    
    def forward(
        self, 
        drug_features: List[torch.Tensor], 
        target_features: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-modal two-tower model.
        
        Args:
            drug_features: List of drug feature tensors
            target_features: List of target feature tensors
            
        Returns:
            Dictionary containing predictions and embeddings
        """
        # Encode drug features
        drug_embeddings = []
        for feat_name, feat in zip(self.hparams.drug_features, drug_features):
            encoded = self.drug_encoders[feat_name](feat)
            drug_embeddings.append(encoded)
        
        # Aggregate drug embeddings
        if self.drug_aggregator is not None:
            if self.aggregator_type == "attentive":
                drug_embedding, drug_attention = self.drug_aggregator(drug_embeddings)
            else:
                drug_embedding = self.drug_aggregator(drug_embeddings)
        else:
            drug_embedding = drug_embeddings[0]  # Single feature, no aggregation needed
        
        # Encode target features
        target_embeddings = []
        for feat_name, feat in zip(self.hparams.target_features, target_features):
            encoded = self.target_encoders[feat_name](feat)
            target_embeddings.append(encoded)
        
        # Aggregate target embeddings
        if self.target_aggregator is not None:
            if self.aggregator_type == "attentive":
                target_embedding, target_attention = self.target_aggregator(target_embeddings)
            else:
                target_embedding = self.target_aggregator(target_embeddings)
        else:
            target_embedding = target_embeddings[0]  # Single feature, no aggregation needed
        
        # Predict DTI score w/ dot product
        score_pred = torch.sum(drug_embedding * target_embedding, dim=-1) # (batch_size)
        
        outputs = {
            "score_pred": score_pred,
            "drug_embedding": drug_embedding,
            "target_embedding": target_embedding,
        }
        
        # Add attention weights if using attentive aggregator
        if self.aggregator_type == "attentive":
            if self.drug_aggregator is not None:
                outputs["drug_attention"] = drug_attention
            if self.target_aggregator is not None:
                outputs["target_attention"] = target_attention
        
        return outputs
        
    def _get_features_from_batch(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract and prepare features from batch."""
        drug_feats = []
        target_feats = []
        
        # Extract drug features
        for feat_name in self.hparams.drug_features:
            if feat_name in batch["drug"]["features"]:
                drug_feats.append(batch["drug"]["features"][feat_name])
            else:
                raise ValueError(f"Missing drug feature '{feat_name}' in batch")
        
        # Extract target features
        for feat_name in self.hparams.target_features:
            if feat_name in batch["target"]["features"]:
                target_feats.append(batch["target"]["features"][feat_name])
            else:
                raise ValueError(f"Missing target feature '{feat_name}' in batch")
        
        return drug_feats, target_feats
        
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