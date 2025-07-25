import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Any, Tuple, Literal, List, Union
import logging

from .utils import *
from mb_vae_dti.training.models import (
    ResidualEncoder, TransformerEncoder,
    CrossAttentionFusion, DTIHead
)
from mb_vae_dti.training.data_containers import (
    BatchData, EmbeddingData, PredictionData, LossData
)


logger = logging.getLogger(__name__)


class MultiOutputDTIModel(AbstractDTIModel):
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

        phase: Literal["finetune", "train"],
        finetune_score: Optional[Literal["Y_pKd", "Y_KIBA"]],

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
        activation: Union[str, nn.Module],
        dropout: float,
        bias: bool,
        
        encoder_type: Literal["resnet", "transformer"],
        dti_weights: Optional[List[float]] = None # Y, pKd, pKi, KIBA weights
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_dtype = torch.float32
        self.phase = phase
        self.finetune_score = finetune_score
        activation = self.parse_activation(activation)
        self.dti_weights = dti_weights

        # Check that the model is configured correctly
        assert phase in ["finetune", "train"], "Multi-output model only supports finetune and train phases"
        if phase == "finetune":
            assert finetune_score is not None, "finetune_score must be specified for finetune phase"
        elif phase == "train":
            assert dti_weights is not None, "dti_weights must be specified for general DTI training"
        else:
            raise ValueError(f"Invalid phase: {phase}")

        assert len(drug_features) == 1 and len(target_features) == 1, \
            "Multi-output model only supports single feature for drug and target"

        logger.info(f"""Multi-output model with:
        - Drug branch: {drug_features.keys()} (dims: {drug_features.values()})
        - Target branch: {target_features.keys()} (dims: {target_features.values()})
        - Phase: {phase} (finetune_score: {finetune_score})""")

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
        
        # Fusion module for aggregating drug and target embeddings
        self.fusion = CrossAttentionFusion(
            input_dim=embedding_dim,
            output_dim=hidden_dim,
            n_layers=n_layers,
            factor=factor,
            dropout=dropout,
            bias=bias,
            activation=activation
        )
        
        # DTI prediction head
        self.dti_head = DTIHead(
            input_dim=hidden_dim,
            proj_dim=hidden_dim // 2,
            dropout=dropout,
            bias=bias,
            activation=activation
        )
        
        # Setup metrics using AbstractDTIModel's method
        self.setup_metrics(phase=phase, finetune_score=finetune_score)
        
    def _create_batch_data( 
            self, 
            batch: Dict[str, Any]
        ) -> BatchData:
        """Create structured batch data using basic utilities & custom logic."""
        batch_data = BatchData(raw_batch=batch)
        batch_data.drug_features, batch_data.target_features = self._get_features_from_batch(batch)
        batch_data.dti_targets, batch_data.dti_masks = self._get_targets_masks_from_batch(batch)
        return batch_data
    
    
    def forward(
            self, 
            drug_features: torch.Tensor,
            target_features: torch.Tensor
        ) -> Tuple[EmbeddingData, PredictionData]:
        """
        Forward pass through the multi-output model.
        
        Args:
            drug_features: Drug feature tensor [batch_size, drug_input_dim]
            target_features: Target feature tensor [batch_size, target_input_dim]
            
        Returns:
            embedding_data: Structured embeddings for drugs, targets and fused
            prediction_data: Structured predictions with DTIHead or single score for finetune phase
        """
        # Encode drug and target
        embedding_data = EmbeddingData(
            drug_embedding=self.drug_encoder(drug_features),
            target_embedding=self.target_encoder(target_features)
        )
        # Aggregate features (ignore attention weights returned)
        embedding_data.fused_embedding, _ = self.fusion(
            embedding_data.drug_embedding, 
            embedding_data.target_embedding
        )
        # Get predictions (extract only single score for finetune phase)
        dti_scores = self.dti_head(embedding_data.fused_embedding)
        dti_scores = dti_scores[self.finetune_score] if self.phase == "finetune" else dti_scores
        prediction_data = PredictionData(
            dti_scores=dti_scores
        )
        return embedding_data, prediction_data
    

    def _common_step(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[BatchData, EmbeddingData, PredictionData, LossData]:
        """
        Common step logic shared across train/val/test.
        
        Returns:
            batch_data: Sturctured batch data incl. targets and masks
            prediction_data: Structured predictions (already masked for valid samples)
            embedding_data: Structured embeddings (incl. fused embedding)
            loss_data: Single and multi-score loss
        """
        # Create structured batch data
        batch_data = self._create_batch_data(batch)
        
        # Forward pass
        embedding_data, prediction_data = self.forward(
            batch_data.drug_features, 
            batch_data.target_features
        )

        # Apply mask(s) and compute loss(es)
        if self.phase == "finetune":
            if not batch_data.dti_masks.any(): # no valid samples
                return batch_data, embedding_data, prediction_data, LossData(
                    accuracy=torch.tensor(0.0, device=self.device)
                )
            
            prediction_data.dti_scores = prediction_data.dti_scores[batch_data.dti_masks]
            batch_data.dti_targets = batch_data.dti_targets[batch_data.dti_masks]

            loss_data = LossData(
                accuracy=F.mse_loss(
                    prediction_data.dti_scores,
                    batch_data.dti_targets
                )
            )
        else:
            components = {
                "Y": F.binary_cross_entropy_with_logits( # binary always present
                    prediction_data.dti_scores["Y"],
                    batch_data.dti_targets["Y"]
                )
            }
            
            # Continuous target losses (when data available)
            for score_name in ["Y_pKd", "Y_pKi", "Y_KIBA"]:
                valid_mask = batch_data.dti_masks[score_name]
                prediction_data.dti_scores[score_name] = prediction_data.dti_scores[score_name][valid_mask]
                batch_data.dti_targets[score_name] = batch_data.dti_targets[score_name][valid_mask]
                components[score_name] = F.mse_loss(
                    prediction_data.dti_scores[score_name],
                    batch_data.dti_targets[score_name]
                ) if valid_mask.any() else torch.tensor(0.0, device=self.device)
            
            # Compute weighted total accuracy loss
            total_accuracy = sum(
                self.dti_weights[i] * loss 
                for i, (score_name, loss) in enumerate(components.items())
            )
            
            loss_data = LossData(
                accuracy=total_accuracy,
                components=components
            )
        
        return batch_data, embedding_data, prediction_data, loss_data
        
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step using structured data containers."""
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update metrics & log loss(es) based on phase
        if self.phase == "finetune":
            if prediction_data.dti_scores is not None and len(prediction_data.dti_scores) > 0:
                self.train_metrics.update(
                    prediction_data.dti_scores, # were masked in _common_step
                    batch_data.dti_targets
                )
        else: # general DTI training with DTIMetricsCollection
            if prediction_data.dti_scores is not None:
                self.train_metrics.update(
                    prediction_data.dti_scores,
                    batch_data.dti_targets
                )
            for name, value in loss_data.components.items():
                self.log(f"train/loss_{name}", value)
        
        # Log finetune/DTI training loss and return to trainer
        self.log("train/loss", loss_data.accuracy)
        return loss_data.accuracy
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update metrics based on phase
        if self.phase == "finetune":
            if prediction_data.dti_scores is not None and len(prediction_data.dti_scores) > 0:
                self.val_metrics.update(
                    prediction_data.dti_scores, # were masked in _common_step
                    batch_data.dti_targets
                )
        else: # general DTI training with DTIMetricsCollection
            if prediction_data.dti_scores is not None:
                self.val_metrics.update(
                    prediction_data.dti_scores,
                    batch_data.dti_targets
                )
            for name, value in loss_data.components.items():
                self.log(f"val/loss_{name}", value)
        
        # Log finetune/DTI validation loss and return to trainer
        self.log("val/loss", loss_data.accuracy)
        return loss_data.accuracy
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step."""
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update metrics based on phase
        if self.phase == "finetune":
            if prediction_data.dti_scores is not None and len(prediction_data.dti_scores) > 0:
                self.test_metrics.update(
                    prediction_data.dti_scores, # were masked in _common_step
                    batch_data.dti_targets
                )
        else: # general DTI training with DTIMetricsCollection
            if prediction_data.dti_scores is not None:
                self.test_metrics.update(
                    prediction_data.dti_scores,
                    batch_data.dti_targets
                )
            for name, value in loss_data.components.items():
                self.log(f"test/loss_{name}", value)
        
        # Log finetune/DTI test loss and return to trainer
        self.log("test/loss", loss_data.accuracy)
        return loss_data.accuracy