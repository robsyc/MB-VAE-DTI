import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Any, Tuple, Literal
import logging

from .utils import *
from mb_vae_dti.training.models import (
    ResidualEncoder, TransformerEncoder,
    ConcatAggregator, AttentiveAggregator
)
from mb_vae_dti.training.data_containers import (
    BatchData, EmbeddingData, PredictionData, LossData
)


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
        activation: Union[str, nn.Module],
        dropout: float,
        bias: bool,
        
        encoder_type: Literal["resnet", "transformer"],
        aggregator_type: Literal["concat", "attentive"],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_dtype = torch.float32
        self.phase = phase
        self.finetune_score = finetune_score
        activation = self.parse_activation(activation)
        self.aggregator_type = aggregator_type

        # Check that the model is configured correctly
        assert phase == "finetune" and finetune_score in ["Y_pKd", "Y_KIBA"], \
            "Multi-modal model only supports finetune phase with Y_pKd or Y_KIBA score"
        if len(drug_features) < 2 and len(target_features) < 2:
            logger.warning("Multi-modal model supports multiple drug and/or target features, but only one feature was provided")
        
        logger.info(f"""Multi-modal model with:
        - Drug branch: {drug_features.keys()} (dims: {drug_features.values()})
        - Target branch: {target_features.keys()} (dims: {target_features.values()})
        - Phase: {phase} (finetune_score: {finetune_score})""")

        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder

        self.drug_encoders = nn.ModuleDict()
        for feat_name in drug_features:
            self.drug_encoders[feat_name] = encoder_type(
                input_dim=drug_features[feat_name],
                hidden_dim=hidden_dim,
                output_dim=hidden_dim if len(drug_features) > 1 else embedding_dim, # individual view encoders project to hidden_dim
                n_layers=n_layers,
                factor=factor,
                activation=activation,
                dropout=dropout,
                bias=bias
            )
        
        self.target_encoders = nn.ModuleDict()
        for feat_name in target_features:
            self.target_encoders[feat_name] = encoder_type(
                input_dim=target_features[feat_name],
                hidden_dim=hidden_dim,
                output_dim=hidden_dim if len(target_features) > 1 else embedding_dim,
                n_layers=n_layers,
                factor=factor,
                activation=activation,
                dropout=dropout,
                bias=bias
            )
        
        aggregator_type = ConcatAggregator if aggregator_type == "concat" else AttentiveAggregator
        
        if len(drug_features) > 1:
            self.drug_aggregator = aggregator_type(
                input_dim=hidden_dim,
                n_features=len(drug_features),
                output_dim=embedding_dim, # but together they are aggregated to embedding_dim
            )
        else:
            self.drug_aggregator = torch.nn.Identity()  # No aggregation needed for single feature
        
        if len(target_features) > 1:
            self.target_aggregator = aggregator_type(
                input_dim=hidden_dim,
                n_features=len(target_features),
                output_dim=embedding_dim,
            )
        else:
            self.target_aggregator = torch.nn.Identity()  # No aggregation needed for single feature
            
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
        drug_features: List[torch.Tensor], 
        target_features: List[torch.Tensor]
    ) -> Tuple[EmbeddingData, PredictionData]:
        """
        Forward pass through the multi-modal two-tower model.
        
        Args:
            drug_features: Drug feature tensor [batch_size, drug_input_dim]
            target_features: Target feature tensor [batch_size, target_input_dim]
            
        Returns:
            embedding_data: Structured drug & target embeddings (and attention weights if using attentive aggregator)
            prediction_data: Structured predictions with single-score dot-product prediction
        """
        embedding_data = EmbeddingData()

        # Encode drug & target features
        drug_embeddings = [
            self.drug_encoders[feat_name](feat)
            for feat_name, feat in zip(self.hparams.drug_features, drug_features)
        ]
        target_embeddings = [
            self.target_encoders[feat_name](feat)
            for feat_name, feat in zip(self.hparams.target_features, target_features)
        ]
        
        # Aggregate into embedding_dim
        if self.drug_aggregator is not None and self.aggregator_type == "attentive":
            embedding_data.drug_embedding, embedding_data.drug_attention = self.drug_aggregator(drug_embeddings)
        else:
            embedding_data.drug_embedding = self.drug_aggregator(drug_embeddings) # identity if no aggregation needed
        if self.target_aggregator is not None and self.aggregator_type == "attentive":
            embedding_data.target_embedding, embedding_data.target_attention = self.target_aggregator(target_embeddings)
        else:
            embedding_data.target_embedding = self.target_aggregator(target_embeddings) # identity if no aggregation needed
        
        # Predict DTI score w/ dot product
        prediction_data = PredictionData(
            dti_scores=torch.sum(
                embedding_data.drug_embedding * embedding_data.target_embedding, 
                dim=-1).squeeze(-1) # (batch_size)
        )
        
        return embedding_data, prediction_data
        

    def _common_step(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[BatchData, EmbeddingData, PredictionData, LossData]:
        """
        Common step logic shared across train/val/test.
        
        Args:
            batch: Batch data
            
        Returns:
            batch_data: Sturctured batch data incl. targets and masks
            prediction_data: Structured predictions (already masked for valid samples)
            embedding_data: Structured embeddings
            loss_data: MSE accuracy loss computed on valid samples
        """
        # Create structured batch data
        batch_data = self._create_batch_data(batch)
        
        # Forward pass
        embedding_data, prediction_data = self.forward(batch_data.drug_features, batch_data.target_features)
        
        # Apply mask and compute loss
        if not batch_data.dti_masks.any(): # no valid samples
            return batch_data, embedding_data, prediction_data, LossData(accuracy=torch.tensor(0.0, device=self.device))
            
        prediction_data.dti_scores = prediction_data.dti_scores[batch_data.dti_masks]
        batch_data.dti_targets = batch_data.dti_targets[batch_data.dti_masks]
        loss_data = LossData(
            accuracy=F.mse_loss(
                prediction_data.dti_scores,
                batch_data.dti_targets
            )
        )

        return batch_data, embedding_data, prediction_data, loss_data
        

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step using structured data containers."""
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update RealDTIMetrics if we have valid predictions
        if prediction_data.dti_scores is not None and len(prediction_data.dti_scores) > 0:
            self.train_metrics.update(
                prediction_data.dti_scores,
                batch_data.dti_targets
            )
        
        # Log MSE accuracy loss & return to trainer
        self.log("train/loss", loss_data.accuracy)
        return loss_data.accuracy
        
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step using structured data containers."""
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update RealDTIMetrics if we have valid predictions
        if prediction_data.dti_scores is not None and len(prediction_data.dti_scores) > 0:
            self.val_metrics.update(
                prediction_data.dti_scores,
                batch_data.dti_targets
            )
        
        # Log MSE accuracy loss & return to trainer
        self.log("val/loss", loss_data.accuracy)
        return loss_data.accuracy
            
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step using structured data containers."""
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update RealDTIMetrics if we have valid predictions
        if prediction_data.dti_scores is not None and len(prediction_data.dti_scores) > 0:
            self.test_metrics.update(
                prediction_data.dti_scores,
                batch_data.dti_targets
            )
        
        # Log MSE accuracy loss & return to trainer
        self.log("test/loss", loss_data.accuracy)
        return loss_data.accuracy