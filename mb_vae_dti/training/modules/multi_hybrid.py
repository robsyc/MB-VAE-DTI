"""
Multi-hybrid DTI PyTorch Lightning module with
- Multi-modal input (multiple drug/target features with aggregation)
- Multi-output prediction (DTI head for multiple simultaneous DTI score prediction)
- Contrastive pretraining support (InfoNCE loss for individual branch pretraining)
- Support for all training phases: pretrain_drug, pretrain_target, train, finetune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Literal
import logging

from mb_vae_dti.training.models import (
    ResidualEncoder, TransformerEncoder,
    ConcatAggregator, AttentiveAggregator,
    CrossAttentionFusion
)
from mb_vae_dti.training.models.heads import DTIHead, InfoNCEHead
from mb_vae_dti.training.metrics import DTIMetricsCollection, RealDTIMetrics
from .utils import AbstractDTIModel


logger = logging.getLogger(__name__)


class MultiHybridDTIModel(AbstractDTIModel):
    """
    Multi-hybrid DTI model combining multi-modal inputs with multi-output predictions.
    
    Architecture:
    - Drug branch: Multiple features → Multiple encoders → Aggregation → Embedding
    - Target branch: Multiple features → Multiple encoders → Aggregation → Embedding
    - DTI prediction: Drug/Target embeddings → Fusion → DTI head → Multiple DTI scores
    - Contrastive heads: Branch embeddings → InfoNCE loss (for pretraining)
    
    Training phases:
    - pretrain_drug: Drug branch pretraining with InfoNCE loss
    - pretrain_target: Target branch pretraining with InfoNCE loss
    - train: General DTI training (multi-score prediction on combined dataset)
    - finetune: Fine-tuning on benchmark datasets (single-score prediction)
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

        fusion_kwargs: Optional[Dict],
        
        dti_head_kwargs: Optional[Dict],
        infonce_head_kwargs: Optional[Dict],
        
        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]],
        
        # Phase-specific parameters
        phase: Literal["pretrain_drug", "pretrain_target", "train", "finetune"],
        finetune_score: Optional[Literal["Y_pKd", "Y_KIBA"]],
        
        # Loss weights
        contrastive_weight: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create encoders based on type
        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder
        
        # Drug encoders
        self.drug_encoders = nn.ModuleDict()
        for feat_name in drug_features:
            self.drug_encoders[feat_name] = encoder_type(
                input_dim=drug_features[feat_name],
                output_dim=encoder_kwargs["hidden_dim"] if len(drug_features) > 1 else embedding_dim,
                **encoder_kwargs
            )
        
        # Target encoders
        self.target_encoders = nn.ModuleDict()
        for feat_name in target_features:
            self.target_encoders[feat_name] = encoder_type(
                input_dim=target_features[feat_name],
                output_dim=encoder_kwargs["hidden_dim"] if len(target_features) > 1 else embedding_dim,
                **encoder_kwargs
            )
        
        # Intra-branch aggregators
        aggregator_type = ConcatAggregator if aggregator_type == "concat" else AttentiveAggregator
        
        # Drug aggregator
        if len(drug_features) > 1:
            self.drug_aggregator = aggregator_type(
                input_dim=encoder_kwargs["hidden_dim"],
                output_dim=embedding_dim,
                n_features=len(drug_features),
                **aggregator_kwargs
            )
        else:
            self.drug_aggregator = None  # No aggregation needed for single feature
            
        # Target aggregator
        if len(target_features) > 1:
            self.target_aggregator = aggregator_type(
                input_dim=encoder_kwargs["hidden_dim"],
                output_dim=embedding_dim,
                n_features=len(target_features),
                **aggregator_kwargs
            )
        else:
            self.target_aggregator = None  # No aggregation needed for single feature
        
        # Inter-branch aggregator (for DTI prediction)
        if phase in ["train", "finetune"]:
            self.fusion = CrossAttentionFusion(
                input_dim=embedding_dim,
                **fusion_kwargs
            )
            
            # DTI prediction head
            self.dti_head = DTIHead(
                input_dim=fusion_kwargs["output_dim"],
                **dti_head_kwargs
            )
        
        # Contrastive learning heads (for pretraining)
        if phase in ["pretrain_drug", "pretrain_target"]:
            if phase == "pretrain_drug":
                self.drug_infonce_head = InfoNCEHead(
                    input_dim=embedding_dim,
                    **infonce_head_kwargs
                )
            elif phase == "pretrain_target":
                self.target_infonce_head = InfoNCEHead(
                    input_dim=embedding_dim,
                    **infonce_head_kwargs
                )
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
        # Loss weights for DTI prediction (based on inverse frequency in combined DTI dataset)
        self.register_buffer('loss_weights', torch.tensor([1.0, 0.903964, 0.282992, 0.755172]))
        
        # Setup metrics based on phase
        self.phase = phase
        self.finetune_score = finetune_score
        
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

            self.train_metrics = RealDTIMetrics(prefix="train/")
            self.val_metrics = RealDTIMetrics(prefix="val/")
            self.test_metrics = RealDTIMetrics(prefix="test/")
        
        # Store aggregator type for handling different return formats
        self.aggregator_type = aggregator_type
        
        # Note: Checkpoint loading is now handled in run.py setup function
        # to support both single and dual checkpoint loading
    
    def _encode_drug_features(self, drug_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode and aggregate drug features.
        
        Args:
            drug_features: List of drug feature tensors
            
        Returns:
            Aggregated drug embedding
        """
        # Encode individual drug features
        drug_embeddings = []
        for feat_name, feat in zip(self.hparams.drug_features, drug_features):
            encoded = self.drug_encoders[feat_name](feat)
            drug_embeddings.append(encoded)
        
        # Aggregate drug embeddings
        if self.drug_aggregator is not None:
            if self.aggregator_type == "attentive":
                drug_embedding, drug_attention = self.drug_aggregator(drug_embeddings)
                return drug_embedding, drug_attention
            else:
                drug_embedding = self.drug_aggregator(drug_embeddings)
                return drug_embedding
        else:
            return drug_embeddings[0]  # Single feature, no aggregation needed
    
    def _encode_target_features(self, target_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode and aggregate target features.
        
        Args:
            target_features: List of target feature tensors
            
        Returns:
            Aggregated target embedding
        """
        # Encode individual target features
        target_embeddings = []
        for feat_name, feat in zip(self.hparams.target_features, target_features):
            encoded = self.target_encoders[feat_name](feat)
            target_embeddings.append(encoded)
        
        # Aggregate target embeddings
        if self.target_aggregator is not None:
            if self.aggregator_type == "attentive":
                target_embedding, target_attention = self.target_aggregator(target_embeddings)
                return target_embedding, target_attention
            else:
                target_embedding = self.target_aggregator(target_embeddings)
                return target_embedding
        else:
            return target_embeddings[0]  # Single feature, no aggregation needed
    
    def forward(
        self, 
        drug_features: List[torch.Tensor] = None, 
        target_features: List[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-hybrid model.
        
        Args:
            drug_features: List of drug feature tensors
            target_features: List of target feature tensors
            
        Returns:
            Dictionary containing predictions and embeddings
        """
        if (drug_features is None or len(drug_features) == 0) and (target_features is None or len(target_features) == 0):
            raise ValueError("Either drug_features or target_features must be provided")
        
        # Encode drug features
        if drug_features is not None and len(drug_features) > 0:
            drug_result = self._encode_drug_features(drug_features)
            if self.aggregator_type == AttentiveAggregator and self.drug_aggregator is not None:
                drug_embedding, drug_attention = drug_result
            else:
                drug_embedding = drug_result
                drug_attention = None
        else:
            drug_embedding, drug_attention = None, None
        
        # Encode target features
        if target_features is not None and len(target_features) > 0:
            target_result = self._encode_target_features(target_features)
            if self.aggregator_type == AttentiveAggregator and self.target_aggregator is not None:
                target_embedding, target_attention = target_result
            else:
                target_embedding = target_result
                target_attention = None
        else:
            target_embedding, target_attention = None, None
        
        outputs = {
            "drug_embedding": drug_embedding,
            "target_embedding": target_embedding,
        }
        
        # Add attention weights if available
        if drug_attention is not None:
            outputs["drug_attention"] = drug_attention
        if target_attention is not None:
            outputs["target_attention"] = target_attention
        
        # DTI prediction (for train/finetune phases)
        if self.phase in ["train", "finetune"]:
            # Ensure both embeddings are available for DTI prediction
            if drug_embedding is None or target_embedding is None:
                raise ValueError("Both drug and target embeddings are required for DTI prediction")
            
            # Inter-branch aggregation with cross-attention module
            combined_features = self.fusion(drug_embedding, target_embedding)
            
            # DTI predictions
            predictions = self.dti_head(combined_features)
            outputs.update({
                "combined_features": combined_features,
                "binary_logits": predictions["binary_logits"],
                "binary_pred": torch.sigmoid(predictions["binary_logits"]),
                "pKd_pred": predictions["pKd_pred"],
                "pKi_pred": predictions["pKi_pred"],
                "KIBA_pred": predictions["KIBA_pred"],
            })
        
        return outputs
    
    def _get_features_from_batch(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract and prepare features from batch."""
        drug_feats = []
        target_feats = []
        
        if self.phase == "pretrain_drug":
            # For drug pretraining, extract drug features from batch["features"]
            for feat_name in self.hparams.drug_features:
                if feat_name in batch["features"]:
                    drug_feats.append(batch["features"][feat_name])
                else:
                    raise ValueError(f"Missing drug feature '{feat_name}' in batch")
            target_feats = []
    
        elif self.phase == "pretrain_target":
            # For target pretraining, extract target features from batch["features"]
            for feat_name in self.hparams.target_features:
                if feat_name in batch["features"]:
                    target_feats.append(batch["features"][feat_name])
                else:
                    raise ValueError(f"Missing target feature '{feat_name}' in batch")
            drug_feats = []  
            
        else: # DTI batch
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
    
    def _compute_contrastive_loss(
        self, 
        embeddings: torch.Tensor, 
        fingerprints: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss using InfoNCE with Tanimoto similarity.
        
        Args:
            embeddings: Learned embeddings (batch_size, embedding_dim)
            fingerprints: Molecular fingerprints for similarity computation
            
        Returns:
            Contrastive loss (scalar)
        """
        # Use the cleaner forward interface
        if self.phase == "pretrain_drug":
            loss = self.drug_infonce_head(embeddings, fingerprints)
        else:  # pretrain_target
            loss = self.target_infonce_head(embeddings, fingerprints)
        
        return loss
    
    def _compute_dti_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any],
        step_name: str = "train"
    ) -> torch.Tensor:
        """
        Compute DTI loss with proper masking and weighting.
        
        Args:
            outputs: Model outputs containing predictions
            batch: Batch data containing targets and masks
            step_name: Name of the training step (train/val/test)
            
        Returns:
            Combined loss tensor
        """
        total_loss = 0.0
        loss_count = 0
        
        # Binary loss (always present)
        binary_loss = self.bce_loss(
            outputs["binary_logits"].squeeze(), 
            batch["y"]["Y"].float()
        )
        total_loss += self.loss_weights[0] * binary_loss
        loss_count += 1
        
        # Real-valued losses (with masking)
        real_losses = []
        
        # pKd loss
        if "Y_pKd" in batch["y"] and batch["y"]["Y_pKd_mask"].any():
            valid_mask = batch["y"]["Y_pKd_mask"]
            pred_kd = outputs["pKd_pred"][valid_mask].squeeze()
            target_kd = batch["y"]["Y_pKd"][valid_mask].squeeze()
            kd_loss = self.mse_loss(pred_kd, target_kd)
            total_loss += self.loss_weights[1] * kd_loss
            loss_count += 1
            real_losses.append(("pKd", kd_loss.item()))
        
        # pKi loss
        if "Y_pKi" in batch["y"] and batch["y"]["Y_pKi_mask"].any():
            valid_mask = batch["y"]["Y_pKi_mask"]
            pred_ki = outputs["pKi_pred"][valid_mask].squeeze()
            target_ki = batch["y"]["Y_pKi"][valid_mask].squeeze()
            ki_loss = self.mse_loss(pred_ki, target_ki)
            total_loss += self.loss_weights[2] * ki_loss
            loss_count += 1
            real_losses.append(("pKi", ki_loss.item()))
        
        # KIBA loss
        if "Y_KIBA" in batch["y"] and batch["y"]["Y_KIBA_mask"].any():
            valid_mask = batch["y"]["Y_KIBA_mask"]
            pred_kiba = outputs["KIBA_pred"][valid_mask].squeeze()
            target_kiba = batch["y"]["Y_KIBA"][valid_mask].squeeze()
            kiba_loss = self.mse_loss(pred_kiba, target_kiba)
            total_loss += self.loss_weights[3] * kiba_loss
            loss_count += 1
            real_losses.append(("KIBA", kiba_loss.item()))
        
        # Normalize by number of active losses
        if loss_count > 0:
            total_loss = total_loss / loss_count
        
        # Log individual loss components
        self.log(f"{step_name}/binary_loss", binary_loss.item(), prog_bar=False)
        for loss_name, loss_value in real_losses:
            self.log(f"{step_name}/{loss_name}_loss", loss_value, prog_bar=False)
        
        return total_loss
    
    def _common_step(
        self, 
        batch: Dict[str, Any],
        step_name: str = "train"
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Common step logic shared across train/val/test.
        
        Args:
            batch: Batch data
            step_name: Name of the training step (train/val/test)
            
        Returns:
            Tuple of (outputs, loss)
        """
        # Get features
        drug_features, target_features = self._get_features_from_batch(batch)
        
        # Forward pass
        outputs = self(drug_features, target_features)
        
        # Compute loss based on phase
        if self.phase == "pretrain_drug":
            # Drug branch contrastive loss
            contrastive_loss = self._compute_contrastive_loss(
                outputs["drug_embedding"], 
                batch["features"]["FP-Morgan"]
            )
            loss = self.hparams.contrastive_weight * contrastive_loss
            self.log(f"{step_name}/contrastive_loss", contrastive_loss.item())
            
        elif self.phase == "pretrain_target":
            # Target branch contrastive loss
            contrastive_loss = self._compute_contrastive_loss(
                outputs["target_embedding"], 
                batch["features"]["FP-ESP"]
            )
            loss = self.hparams.contrastive_weight * contrastive_loss
            self.log(f"{step_name}/contrastive_loss", contrastive_loss.item())
            
        else:  # train or finetune
            # DTI prediction loss
            loss = self._compute_dti_loss(outputs, batch, step_name)
        
        return outputs, loss
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step."""
        outputs, loss = self._common_step(batch, "train")
        
        # Update metrics based on phase
        if self.phase == "train" and self.train_metrics is not None:
            self._update_multi_score_metrics(outputs, batch, self.train_metrics)
        elif self.phase == "finetune" and self.train_metrics is not None:
            self._update_single_score_metrics(outputs, batch, self.train_metrics)
        
        # Logging
        self.log("train/loss", loss)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        outputs, loss = self._common_step(batch, "val")
        
        # Update metrics based on phase
        if self.phase == "train" and self.val_metrics is not None:
            self._update_multi_score_metrics(outputs, batch, self.val_metrics)
        elif self.phase == "finetune" and self.val_metrics is not None:
            self._update_single_score_metrics(outputs, batch, self.val_metrics)
        
        # Logging
        self.log("val/loss", loss)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step."""
        outputs, loss = self._common_step(batch, "test")
        
        # Update metrics based on phase
        if self.phase == "train" and self.test_metrics is not None:
            self._update_multi_score_metrics(outputs, batch, self.test_metrics)
        elif self.phase == "finetune" and self.test_metrics is not None:
            self._update_single_score_metrics(outputs, batch, self.test_metrics)
        
        # Logging
        self.log("test/loss", loss)
    
    def _update_multi_score_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any], 
        metrics: DTIMetricsCollection
    ):
        """Update multi-score metrics for general training."""
        # Binary metrics
        binary_preds = outputs["binary_pred"].squeeze()
        binary_targets = batch["y"]["Y"]
        
        # Real-valued metrics
        real_preds = {}
        real_targets = {}
        real_masks = {}
        
        # Check each score
        for score_name, pred_key in [("pKd", "pKd_pred"), ("pKi", "pKi_pred"), ("KIBA", "KIBA_pred")]:
            batch_key = f"Y_{score_name}"
            mask_key = f"{batch_key}_mask"
            
            if batch_key in batch["y"] and mask_key in batch["y"]:
                real_preds[score_name] = outputs[pred_key].squeeze()
                real_targets[score_name] = batch["y"][batch_key]
                real_masks[score_name] = batch["y"][mask_key]
        
        # Update metrics
        metrics.update(
            binary_preds=binary_preds,
            binary_targets=binary_targets,
            real_preds=real_preds,
            real_targets=real_targets,
            real_masks=real_masks
        )
    
    def _update_single_score_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any], 
        metrics
    ):
        """Update single-score metrics for fine-tuning."""
        if self.finetune_score is None:
            return
        
        # Get the specific score being fine-tuned
        pred_key = f"{self.finetune_score.split('_')[1]}_pred"
        
        score_pred = outputs[pred_key].squeeze()
        score_target = batch["y"][self.finetune_score]
        score_mask = batch["y"][f"{self.finetune_score}_mask"]
        
        # Update metrics only for valid samples
        if score_mask.any():
            metrics.update(score_pred, score_target, score_mask)
    
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