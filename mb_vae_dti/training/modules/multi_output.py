"""
Multi-output DTI PyTorch Lightning module with
- Single drug/target encoders (ResidualEncoder or TransformerEncoder)
- DTI prediction head for multiple simultaneous DTI score prediction
- Support for both general DTI training and fine-tuning phases
- Checkpoint loading for fine-tuning
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Literal
import logging

from training.models import (
    ResidualEncoder, TransformerEncoder,
    CrossAttentionFusion
)
from training.models.heads import DTIHead
from training.metrics import DTIMetricsCollection, RealDTIMetrics
from .optimizer_utils import configure_optimizer_and_scheduler


logger = logging.getLogger(__name__)


class MultiOutputDTIModel(pl.LightningModule):
    """
    Multi-output DTI model with DTI prediction head for simultaneous multi-score prediction.
    
    Architecture:
    - Drug encoder: Single feature → Encoder → Embedding
    - Target encoder: Single feature → Encoder → Embedding
    - Feature aggregation: Drug/Target embeddings → Fusion → Combined features
    - DTI prediction head: Combined features → Multiple DTI scores
    
    This model supports:
    - General DTI training (multi-score prediction on combined dataset)
    - Fine-tuning on benchmark datasets (single-score prediction)
    - Checkpoint loading for transfer learning
    """
    def __init__(
        self,
        # Architecture parameters
        embedding_dim: int,
        drug_input_dim: int,
        target_input_dim: int,

        encoder_type: Literal["resnet", "transformer"],
        encoder_kwargs: Optional[Dict],

        fusion_kwargs: Optional[Dict], # input_dim of the aggregator == output_dim of the encoders
        
        dti_head_kwargs: Optional[Dict], # input_dim of the dti_head == output_dim of the aggregator
        
        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]],
        
        drug_feature: Literal["FP-Morgan", "EMB-BiomedGraph", "EMB-BiomedImg", "EMB-BiomedText"],
        target_feature: Literal["FP-ESP", "EMB-ESM", "EMB-NT"],
        
        # Phase-specific parameters
        phase: Literal["train", "finetune"],
        finetune_score: Optional[Literal["Y_pKd", "Y_KIBA"]]
    ):
        super().__init__()
        self.save_hyperparameters()

        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder

        # Create encoders based on type
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
        
        # Fusion module for aggregating drug and target embeddings
        self.fusion = CrossAttentionFusion(
            input_dim=embedding_dim,
            **fusion_kwargs
        )
        
        # DTI prediction head
        self.dti_head = DTIHead(
            input_dim=embedding_dim,
            **dti_head_kwargs
        )
        
        # Loss weights for balancing different loss components
        # [binary, Kd, Ki, KIBA] - based on inverse frequency in dataset
        self.register_buffer('loss_weights', torch.tensor([1.0, 0.903964, 0.282992, 0.755172]))
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
        # Phase-specific setup
        self.phase = phase
        self.finetune_score = finetune_score
        
        # Set up metrics based on phase
        if phase == "train":
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
        
    def load_pretrained_weights(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract state dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Load compatible weights
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
                logger.debug(f"Loaded weight: {k}")
            else:
                logger.warning(f"Skipping weight: {k} (shape mismatch or not found)")
        
        # Load weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
        logger.info(f"Loaded {len(pretrained_dict)} pretrained weights")
        
        # Freeze encoder weights for fine-tuning if specified
        # TODO: should we freeze?
        if self.phase == "finetune":
            self.freeze_encoders()
    
    def freeze_encoders(self) -> None:
        """Freeze encoder weights during fine-tuning."""
        logger.info("Freezing encoder weights for fine-tuning")
        
        for param in self.drug_encoder.parameters():
            param.requires_grad = False
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self) -> None:
        """Unfreeze encoder weights."""
        logger.info("Unfreezing encoder weights")
        
        for param in self.drug_encoder.parameters():
            param.requires_grad = True
        
        for param in self.target_encoder.parameters():
            param.requires_grad = True
    
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
        Forward pass through the multi-output model.
        
        Args:
            drug_features: Drug feature tensor [batch_size, drug_input_dim]
            target_features: Target feature tensor [batch_size, target_input_dim]
            
        Returns:
            Dictionary containing predictions and embeddings
        """
        # Encode drug and target
        drug_embedding = self.drug_encoder(drug_features)       # (batch_size, embedding_dim)
        target_embedding = self.target_encoder(target_features) # (batch_size, embedding_dim)
        
        # Aggregate features
        combined_features = self.fusion(drug_embedding, target_embedding)
        
        # Get predictions
        predictions = self.dti_head(combined_features)
        
        return {
            "binary_logits": predictions["binary_logits"],
            "binary_pred": torch.sigmoid(predictions["binary_logits"]),
            "pKd_pred": predictions["pKd_pred"],
            "pKi_pred": predictions["pKi_pred"],
            "KIBA_pred": predictions["KIBA_pred"],
            "drug_embedding": drug_embedding,
            "target_embedding": target_embedding,
            "combined_features": combined_features,
        }
    
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
    
    def _prepare_targets_for_dti_loss(
        self, 
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare targets for DTI loss module.
        
        Args:
            batch: Batch data
            
        Returns:
            Dict with targets for DTI loss
        """
        targets = {}
        
        # Binary target (always present)
        targets["binary"] = batch["y"]["Y"]
        
        # Real-valued targets (only if valid)
        if "Y_pKd" in batch["y"] and batch["y"]["Y_pKd_mask"].any():
            targets["Kd"] = batch["y"]["Y_pKd"]
        
        if "Y_pKi" in batch["y"] and batch["y"]["Y_pKi_mask"].any():
            targets["Ki"] = batch["y"]["Y_pKi"]
        
        if "Y_KIBA" in batch["y"] and batch["y"]["Y_KIBA_mask"].any():
            targets["KIBA"] = batch["y"]["Y_KIBA"]
        
        return targets
    
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
        
        # Normalize by number of active losses to maintain consistent scale
        if loss_count > 0:
            total_loss = total_loss / loss_count
        
        # Log individual loss components (for debugging)
        self.log(f"{step_name}/binary_loss", binary_loss.item(), prog_bar=False)
        for loss_name, loss_value in real_losses:
            self.log(f"{step_name}/{loss_name}_loss", loss_value, prog_bar=False)
        
        return total_loss
    
    def _common_step(
        self, 
        batch: Dict[str, Any],
        step_name: str = "train"
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Common step logic shared across train/val/test.
        
        Args:
            batch: Batch data
            step_name: Name of the training step (train/val/test)
            
        Returns:
            Tuple of (predictions, targets, loss)
        """
        # Get features
        drug_features, target_features = self._get_features_from_batch(batch)
        
        # Forward pass
        outputs = self(drug_features, target_features)
        
        # Prepare targets for metrics
        targets = self._prepare_targets_for_dti_loss(batch)
        
        # Compute loss
        loss = self._compute_dti_loss(outputs, batch, step_name)
        
        return outputs, targets, loss
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step."""
        outputs, targets, loss = self._common_step(batch, "train")
        
        # Update metrics based on phase
        if self.phase == "train":
            # Multi-score metrics
            self._update_multi_score_metrics(outputs, batch, self.train_metrics)
        else:
            # Single-score metrics for fine-tuning
            self._update_single_score_metrics(outputs, batch, self.train_metrics)
        
        # Logging
        self.log("train/loss", loss)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        outputs, targets, loss = self._common_step(batch, "val")
        
        # Update metrics based on phase
        if self.phase == "train":
            # Multi-score metrics
            self._update_multi_score_metrics(outputs, batch, self.val_metrics)
        else:
            # Single-score metrics for fine-tuning
            self._update_single_score_metrics(outputs, batch, self.val_metrics)
        
        # Logging
        self.log("val/loss", loss)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step."""
        outputs, targets, loss = self._common_step(batch, "test")
        
        # Update metrics based on phase
        if self.phase == "train":
            # Multi-score metrics
            self._update_multi_score_metrics(outputs, batch, self.test_metrics)
        else:
            # Single-score metrics for fine-tuning
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
