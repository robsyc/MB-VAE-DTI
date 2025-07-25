import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Any, Tuple, Literal, List, Union
import logging

from .utils import *
from mb_vae_dti.training.models import (
    ResidualEncoder, TransformerEncoder,
    ConcatAggregator, AttentiveAggregator,
    CrossAttentionFusion, DTIHead, InfoNCEHead
)
from mb_vae_dti.training.data_containers import (
    BatchData, EmbeddingData, PredictionData, LossData
)


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

        phase: Literal["pretrain_drug", "pretrain_target", "train", "finetune"],
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
        aggregator_type: Literal["concat", "attentive"],
        weights: Optional[List[float]] = None, # accuracy, complexity, contrastive, reconstruction
        dti_weights: Optional[List[float]] = None, # Y, pKd, pKi, KIBA weights
        contrastive_temp: Optional[float] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_dtype = torch.float32
        self.phase = phase
        self.finetune_score = finetune_score
        activation = self.parse_activation(activation)
        self.aggregator_type = aggregator_type

        self.weights = weights
        self.dti_weights = dti_weights
        self.contrastive_temp = contrastive_temp

        # Check that the model is configured correctly
        if phase == "finetune":
            assert finetune_score is not None, "finetune_score must be specified for finetune phase"
        elif phase == "train":
            assert dti_weights is not None, "dti_weights must be specified for general DTI training"
        else:
            raise ValueError(f"Invalid phase: {phase}")

        logger.info(f"""Multi-hybrid model with:
        - Drug branch: {drug_features.keys()} (dims: {drug_features.values()})
        - Target branch: {target_features.keys()} (dims: {target_features.values()})
        - Phase: {phase} (finetune_score: {finetune_score})
        - Aggregator type: {aggregator_type}""")

        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder
        aggregator_type = ConcatAggregator if aggregator_type == "concat" else AttentiveAggregator

        # Drug branch
        if self.phase != "pretrain_target":
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
            if len(drug_features) > 1:
                self.drug_aggregator = aggregator_type(
                    input_dim=hidden_dim,
                    n_features=len(drug_features),
                    output_dim=embedding_dim, # but together they are aggregated to embedding_dim
                )
            else:
                self.drug_aggregator = torch.nn.Identity()  # No aggregation needed for single feature
        
            self.drug_contrastive_head = InfoNCEHead(
                input_dim=embedding_dim,
                proj_dim=hidden_dim // 2,
                dropout=dropout,
                bias=bias,
                activation=activation
            )

        # Target branch
        if self.phase != "pretrain_drug":
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
            if len(target_features) > 1:
                self.target_aggregator = aggregator_type(
                    input_dim=hidden_dim,
                    n_features=len(target_features),
                    output_dim=embedding_dim,
                )
            else:
                self.target_aggregator = torch.nn.Identity()  # No aggregation needed for single feature
            
            self.target_contrastive_head = InfoNCEHead(
                input_dim=embedding_dim,
                proj_dim=hidden_dim // 2,
                dropout=dropout,
                bias=bias,
                activation=activation
            )

        # DTI prediction
        if phase in ["train", "finetune"]:
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

        # One will be None in pretrain_target/pretrain_drug case
        batch_data.drug_features, batch_data.target_features = self._get_features_from_batch(batch)

        # We want fingerprints for contrastive loss (one will be None in pretrain_target/pretrain_drug case)
        batch_data.drug_fp, batch_data.target_fp = self._get_fingerprints_from_batch(batch)

        # Only interested in targets during DTI train/finetune phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            batch_data.dti_targets, batch_data.dti_masks = self._get_targets_masks_from_batch(batch)
        
        return batch_data
    
    
    def forward(
        self, 
        drug_features: Optional[List[torch.Tensor]], 
        target_features: Optional[List[torch.Tensor]]
    ) -> Tuple[EmbeddingData, PredictionData]:
        """
        Forward pass through the multi-modal two-tower model.
        
        Args:
            drug_features: Drug feature tensors [batch_size, drug_input_dim] (optional in case of pretrain_target)
            target_features: Target feature tensors [batch_size, target_input_dim] (optional in case of pretrain_drug)
            
        Returns:
            embedding_data: Structured drug & target embeddings (and attention weights if using attentive aggregator)
            prediction_data: Structured predictions with single-score dot-product prediction
        """
        embedding_data = EmbeddingData()

        # Encode & aggregate drug and/or target features
        if self.phase != "pretrain_target":
            drug_embeddings = [
                self.drug_encoders[feat_name](feat)
                for feat_name, feat in zip(self.hparams.drug_features, drug_features)
            ]
            if self.aggregator_type == "attentive":
                embedding_data.drug_embedding, embedding_data.drug_attention = self.drug_aggregator(drug_embeddings)
            else:
                embedding_data.drug_embedding = self.drug_aggregator(drug_embeddings)

        if self.phase != "pretrain_drug":
            target_embeddings = [
                self.target_encoders[feat_name](feat)
                for feat_name, feat in zip(self.hparams.target_features, target_features)
            ]
            if self.aggregator_type == "attentive":
                embedding_data.target_embedding, embedding_data.target_attention = self.target_aggregator(target_embeddings)
            else:
                embedding_data.target_embedding = self.target_aggregator(target_embeddings)

        # Fuse drug and target embeddings & get DTI predictions
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
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
        else:
            prediction_data = PredictionData()

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
        batch_data = self._create_batch_data(batch) # also incl. fingerprints
        
        # Forward pass
        embedding_data, prediction_data = self.forward(
            batch_data.drug_features, 
            batch_data.target_features
        )

        loss_data = LossData()

        # Compute contrastive loss
        drug_contrastive_loss = self.drug_contrastive_head(
            x = embedding_data.drug_embedding,
            fingerprints = batch_data.drug_fp,
            temperature = self.contrastive_temp
        ) if self.phase != "pretrain_target" else torch.tensor(0.0, device=self.device)

        target_contrastive_loss = self.target_contrastive_head(
            x = embedding_data.target_embedding,
            fingerprints = batch_data.target_fp,
            temperature = self.contrastive_temp
        ) if self.phase != "pretrain_drug" else torch.tensor(0.0, device=self.device)
        
        loss_data.contrastive = drug_contrastive_loss + target_contrastive_loss
        loss_data.components = {
            "drug_contrastive": drug_contrastive_loss,
            "target_contrastive": target_contrastive_loss
        }

        # Apply mask(s) and compute accuracy loss(es)
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if self.phase == "finetune":
                # mask
                prediction_data.dti_scores = prediction_data.dti_scores[batch_data.dti_masks]
                batch_data.dti_targets = batch_data.dti_targets[batch_data.dti_masks]
                # handle no valid samples
                if not batch_data.dti_masks.any(): # no valid samples
                    loss_data.accuracy = torch.tensor(0.0, device=self.device)
                else:
                    # single score accuracy loss
                    loss_data.accuracy = F.mse_loss(
                        prediction_data.dti_scores,
                        batch_data.dti_targets
                    )
            else: # phase == "train"
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
                
                loss_data.accuracy = total_accuracy
                loss_data.components.update(components)
        
        return batch_data, embedding_data, prediction_data, loss_data
        
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update metrics & log loss(es) based on phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if prediction_data.dti_scores is not None:
                self.train_metrics.update(
                    prediction_data.dti_scores, # were masked in _common_step
                    batch_data.dti_targets
                )
            for name, value in loss_data.components.items():
                self.log(f"train/loss_{name}", value)
        
        # No contrastive loss metrics...

        loss = loss_data.compute_loss(self.weights)
        
        # Log finetune/DTI training loss and return to trainer
        self.log("train/loss_contrastive", loss_data.contrastive)
        self.log("train/loss_accuracy", loss_data.accuracy) # may be None in case of pretrain_target/pretrain_drug
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update metrics based on phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if prediction_data.dti_scores is not None:
                self.val_metrics.update(
                    prediction_data.dti_scores, # were masked in _common_step
                    batch_data.dti_targets
                )
            for name, value in loss_data.components.items():
                self.log(f"val/loss_{name}", value)
        
        loss = loss_data.compute_loss(self.weights)
        
        # Log finetune/DTI validation loss and return to trainer
        self.log("val/loss_contrastive", loss_data.contrastive)
        self.log("val/loss_accuracy", loss_data.accuracy) # may be None in case of pretrain_target/pretrain_drug
        self.log("val/loss", loss)
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update metrics based on phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if prediction_data.dti_scores is not None:
                self.test_metrics.update(
                    prediction_data.dti_scores, # were masked in _common_step
                    batch_data.dti_targets
                )
            for name, value in loss_data.components.items():
                self.log(f"test/loss_{name}", value)
        
        loss = loss_data.compute_loss(self.weights)

        # Log finetune/DTI test loss and return to trainer
        self.log("test/loss_contrastive", loss_data.contrastive)
        self.log("test/loss_accuracy", loss_data.accuracy) # may be None in case of pretrain_target/pretrain_drug
        self.log("test/loss", loss)
        return loss