"""
Utility functions for DTI PyTorch Lightning modules
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Any, Tuple, Literal, List, Union
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class AbstractDTIModel(pl.LightningModule):
    """
    Abstract base class for DTI PyTorch Lightning modules.
    
    Provides common functionality for:
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
        self.train_diffusion_metrics = None
        self.val_diffusion_metrics = None
        self.test_diffusion_metrics = None

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
                eta_min=1e-6  # Small minimum to avoid complete stagnation
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

            self.train_metrics = RealDTIMetrics(prefix="train/")
            self.val_metrics = RealDTIMetrics(prefix="val/")
            self.test_metrics = RealDTIMetrics(prefix="test/")

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
        if hasattr(self, 'phase') and getattr(self, 'phase', None) == "finetune":
            self.freeze_encoders()
    
    def _encode_drug_features(self, drug_features: List[torch.Tensor]) -> torch.Tensor:
        """Encode and aggregate drug features. To be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _encode_drug_features")
        
    def _encode_target_features(self, target_features: List[torch.Tensor]) -> torch.Tensor:
        """Encode and aggregate target features. To be implemented by subclass.""" 
        raise NotImplementedError("Subclass must implement _encode_target_features")
    
    def _get_features_from_batch(
        self, 
        batch: Dict[str, Any]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor]]]:
        """
        Extract and prepare drug and target features from batch.
        Returns either
            - Tuple of (drug_feat, target_feat) if single feature is used (e.g. Baseline & MultiOutput models)
            - Tuple of (drug_feats, target_feats) if multiple features are used (e.g. MultiModal, MultiHybrid and Full models)
        
        One of the items may be None when pretraining drug or target
        """
        drug_feats = []
        target_feats = []
        
        # Handle different phases
        if hasattr(self.hparams, 'phase') and self.hparams.phase == "pretrain_drug":
            # For drug pretraining, extract drug features from batch["features"]
            for feat_name in self.hparams.drug_features.keys():
                if feat_name in batch["features"]:
                    drug_feats.append(batch["features"][feat_name])
        elif hasattr(self.hparams, 'phase') and self.hparams.phase == "pretrain_target":
            # For target pretraining, extract target features from batch["features"]
            for feat_name in self.hparams.target_features.keys():
                if feat_name in batch["features"]:
                    target_feats.append(batch["features"][feat_name])
        else:
            # For DTI training/fine-tuning
            for feat_name in self.hparams.drug_features.keys():
                if feat_name in batch["drug"]["features"]:
                    drug_feats.append(batch["drug"]["features"][feat_name])
            for feat_name in self.hparams.target_features.keys():
                if feat_name in batch["target"]["features"]:
                    target_feats.append(batch["target"]["features"][feat_name])

        # Convert to single tensor if only one feature
        drug_feats = drug_feats[0] if len(drug_feats) == 1 else drug_feats
        target_feats = target_feats[0] if len(target_feats) == 1 else target_feats

        if not drug_feats and not target_feats:
            raise ValueError("Could not find appropriate features in batch")
        
        return drug_feats, target_feats
    
    def _get_smiles_from_batch(self, batch: Dict[str, Any]) -> List[str]:
        """Extract SMILES from batch."""
        if hasattr(self.hparams, 'phase') and self.hparams.phase == "pretrain_drug":
            return batch["representations"]["smiles"]
        else:
            return batch["drug"]["representations"]["smiles"]
    
    def _get_fingerprints_from_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and prepare drug and target fingerprints from batch for contrastive loss.
        Returns Tuple of (drug_fp, target_fp)
        One of the items may be None when pretraining drug or target
        """
        if hasattr(self.hparams, 'phase') and self.hparams.phase == "pretrain_drug":
            return batch["features"]["FP-Morgan"], None
        elif hasattr(self.hparams, 'phase') and self.hparams.phase == "pretrain_target":
            return None, batch["features"]["FP-ESP"]
        else:
            return batch["drug"]["features"]["FP-Morgan"], batch["target"]["features"]["FP-ESP"]
    
    def _get_target_from_batch(self, batch: Dict[str, Any], key: Literal["Y_pKd", "Y_KIBA"]) -> torch.Tensor:
        """Get single target score from batch."""
        return batch["y"][key]
    
    def _get_target_mask_from_batch(self, batch: Dict[str, Any], key: Literal["Y_pKd", "Y_KIBA"]) -> torch.Tensor:
        """Get mask for single target score from batch."""
        return batch["y"][f"{key}_mask"]
    
    def _get_targets_from_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Get all target scores from batch."""
        return {
            "Y": batch["y"]["Y"],
            "Y_pKd": batch["y"]["Y_pKd"],
            "Y_KIBA": batch["y"]["Y_KIBA"],
            "Y_pKi": batch["y"]["Y_pKi"]
        }

    def _get_targets_masks_from_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Get all target score masks from batch."""
        return {
            "Y": batch["y"]["Y_mask"],
            "Y_pKd": batch["y"]["Y_pKd_mask"],
            "Y_KIBA": batch["y"]["Y_KIBA_mask"],
            "Y_pKi": batch["y"]["Y_pKi_mask"]
        }
    
    def _update_multi_score_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any], 
        metrics
    ):
        """Update multi-score metrics for general training."""
        if metrics is None:
            return
            
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
        metrics,
        finetune_score: str = None
    ):
        """Update single-score metrics for fine-tuning."""
        if metrics is None or finetune_score is None:
            return
        
        # Get the specific score being fine-tuned
        if finetune_score == "Y_pKd":
            pred_key = "pKd_pred"
        elif finetune_score == "Y_KIBA":
            pred_key = "KIBA_pred"
        elif finetune_score == "Y_pKi":
            pred_key = "pKi_pred"
        else:
            # For baseline model with simple prediction
            pred_key = "score_pred"
        
        if pred_key not in outputs:
            return
            
        score_pred = outputs[pred_key].squeeze()
        score_target = batch["y"][finetune_score]
        score_mask = batch["y"][f"{finetune_score}_mask"]
        
        # Update metrics only for valid samples
        if score_mask.any():
            metrics.update(score_pred[score_mask], score_target[score_mask])

    def on_train_epoch_end(self):
        """Compute and log training metrics."""
        if self.train_metrics is not None:
            train_metrics = self.train_metrics.compute()
            for name, value in train_metrics.items():
                self.log(name, value)
            self.train_metrics.reset()
            
        # Handle diffusion metrics if present
        if self.train_diffusion_metrics is not None:
            train_diffusion_metrics = self.train_diffusion_metrics.compute()
            for name, value in train_diffusion_metrics.items():
                self.log(name, value)
            self.train_diffusion_metrics.reset()
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        if self.val_metrics is not None:
            val_metrics = self.val_metrics.compute()
            for name, value in val_metrics.items():
                self.log(name, value)
            self.val_metrics.reset()
            
        # Handle diffusion metrics if present
        if self.val_diffusion_metrics is not None:
            val_diffusion_metrics = self.val_diffusion_metrics.compute()
            for name, value in val_diffusion_metrics.items():
                self.log(name, value)
            self.val_diffusion_metrics.reset()
    
    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        if self.test_metrics is not None:
            test_metrics = self.test_metrics.compute()
            for name, value in test_metrics.items():
                self.log(name, value)
            self.test_metrics.reset()
            
        # Handle diffusion metrics if present
        if self.test_diffusion_metrics is not None:
            test_diffusion_metrics = self.test_diffusion_metrics.compute()
            for name, value in test_diffusion_metrics.items():
                self.log(name, value)
            self.test_diffusion_metrics.reset()