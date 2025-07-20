"""
Drug-Target Interaction (DTI) evaluation metrics for DTITree models.

This module contains TorchMetrics-compatible DTI evaluation metrics:
- DTIMetricsCollection: Convenient collection of all DTI metrics
    - RealDTIMetrics: Collection of metrics for real-valued DTI score prediction
        - MSE & RMSE Mean (https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html#mean-squared-error-mse)
        - PearsonCorrCoef (https://lightning.ai/docs/torchmetrics/stable/regression/pearson_corr_coef.html)
        - R2Score (https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html)
        - ConcordanceIndex (https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index & https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/refs/heads/master/lifelines/utils/concordance.py)
    - BinaryDTIMetrics: Collection of metrics for binary DTI score prediction
        - BinaryAccuracy (https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#binaryaccuracy)
        - BinaryConfusionMatrix (https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html#torchmetrics.classification.BinaryConfusionMatrix)
        - BinaryF1Score (https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html)
        - BinaryAUPRC (https://docs.pytorch.org/torcheval/main/generated/torcheval.metrics.BinaryAUPRC.html)
        - BinaryAUROC (https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html#binaryauroc)
"""

import torch
from torch import Tensor
import torch.nn.functional as F

from torchmetrics import Metric, MetricCollection
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, R2Score
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, BinaryF1Score, BinaryAUROC

from torcheval.metrics import BinaryAUPRC as _BinaryAUPRC
from lifelines.utils import concordance_index

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConcordanceIndex(Metric):
    """
    TorchMetrics-compatible Concordance Index (C-index) metric.
    
    The C-index is the probability that the predicted affinity scores of
    two randomly chosen drug-target pairs are in the correct order.
    
    Based on the lifelines implementation but adapted for TorchMetrics.
    """
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Store all predictions and targets to compute C-index at the end
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update state with predictions and targets.
        
        Args:
            preds: Predicted scores [batch_size]
            target: Ground truth scores [batch_size]
        """
        # Ensure tensors are on the same device
        preds = preds.detach()
        target = target.detach()
        
        self.predictions.append(preds)
        self.targets.append(target)
    
    def compute(self) -> Tensor:
        """
        Compute the concordance index.
        
        Returns:
            C-index value between 0 and 1
        """
        if len(self.predictions) == 0:
            return torch.tensor(0.5)  # Random performance
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Convert to numpy for lifelines compatibility
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        # Use lifelines concordance_index function
        # Note: lifelines expects event_times, predicted_scores
        # For DTI, we treat target scores as "event times" and predictions as "scores"
        try:
            c_index = concordance_index(targets_np, preds_np)
        except ZeroDivisionError:
            # Handle case when there are no admissible pairs (common in debug mode)
            # Return 0.5 as default (random performance)
            return torch.tensor(0.5, dtype=torch.float32)
        
        return torch.tensor(c_index, dtype=torch.float32)


class RMSEMetric(Metric):
    """Root Mean Squared Error metric."""
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.mse = MeanSquaredError()
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        self.mse.update(preds, target)
    
    def compute(self) -> Tensor:
        return torch.sqrt(self.mse.compute())


class BinaryAUPRC(Metric):
    """
    TorchMetrics-compatible wrapper for torcheval.metrics.BinaryAUPRC.
    
    This wrapper makes torcheval's BinaryAUPRC compatible with torchmetrics.MetricCollection.
    """
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Store predictions and targets to compute AUPRC at the end
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update state with predictions and targets.
        
        Args:
            preds: Predicted probabilities [batch_size]
            target: Ground truth binary labels [batch_size]
        """
        # Ensure tensors are on the same device
        preds = preds.detach()
        target = target.detach()
        
        self.predictions.append(preds)
        self.targets.append(target)
    
    def compute(self) -> Tensor:
        """
        Compute the AUPRC using torcheval.
        
        Returns:
            AUPRC value
        """
        if len(self.predictions) == 0:
            return torch.tensor(0.0)
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Use torcheval's BinaryAUPRC
        metric = _BinaryAUPRC()
        metric.update(all_preds, all_targets)
        return metric.compute()


class RealDTIMetrics(MetricCollection):
    """
    Collection of metrics for real-valued DTI score prediction.
    
    This collection handles multiple DTI scores (pKd, pKi, KIBA) and can
    compute metrics only for available ground truth values using masks.
    """
    
    def __init__(self, prefix: str = ""):
        """
        Initialize real-valued DTI metrics.
        
        Args:
            prefix: Prefix for metric names (e.g., "val_" or "test_")
        """
        metrics = {
            "mse": MeanSquaredError(),
            "rmse": RMSEMetric(),
            "pearson": PearsonCorrCoef(),
            "r2": R2Score(),
            "ci": ConcordanceIndex(),
        }
        
        super().__init__(metrics, prefix=prefix)
        
    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> None:
        """
        Update metrics with predictions and targets.
        
        Args:
            preds: Predicted scores [batch_size]
            target: Ground truth scores [batch_size]
            mask: Boolean mask for valid samples [batch_size]
        """
        if mask is not None:
            # Only use valid samples
            valid_mask = mask.bool()
            if not valid_mask.any():
                return  # No valid samples to update
            
            preds = preds[valid_mask]
            target = target[valid_mask]
        
        # Update all metrics
        super().update(preds, target)
    
    def compute(self) -> Dict[str, Tensor]:
        """
        Compute all metrics, handling cases where there are insufficient samples.
        
        Returns:
            Dict of computed metrics, with NaN for metrics that can't be computed
        """
        results = {}
        
        for name, metric in self.items():
            try:
                # Try to compute the metric
                value = metric.compute()
                results[name] = value
                
            except (ValueError, RuntimeError) as e:
                # If computation fails due to insufficient samples, return NaN
                error_msg = str(e).lower()
                if ("at least two samples" in error_msg or 
                    "sample size" in error_msg or 
                    "not enough" in error_msg or
                    "insufficient" in error_msg):
                    results[name] = torch.tensor(float('nan'))
                else:
                    # Re-raise other errors
                    raise e
        
        return results
    
    # def _check_metric_updated(self, metric) -> bool:
    #     """
    #     Check if a metric has been updated with any data.
    #     This avoids the "UserWarning: The ``compute`` method of metric ... was called before update()"
        
    #     Args:
    #         metric: The metric to check
            
    #     Returns:
    #         True if the metric has been updated, False otherwise
    #     """
    #     # Check common metric state attributes to determine if metric has been updated
        
    #     # For MSE, RMSE - check if sum_squared_error has been set
    #     if hasattr(metric, 'sum_squared_error'):
    #         return metric.sum_squared_error.numel() > 0 and metric.sum_squared_error.sum() != 0
        
    #     # For Pearson correlation - check if n_total > 0
    #     if hasattr(metric, 'n_total'):
    #         return metric.n_total > 0
        
    #     # For R2Score - check if sum_squared_error exists and has data
    #     if hasattr(metric, 'sum_squared_error') and hasattr(metric, 'sum_squared_observed'):
    #         return (metric.sum_squared_error.numel() > 0 and 
    #                 metric.sum_squared_observed.numel() > 0)
        
    #     # For ConcordanceIndex - check if estimates tensor has data
    #     if hasattr(metric, 'estimates'):
    #         return metric.estimates.numel() > 0
        
    #     # For metrics with a 'total' attribute (like accuracy)
    #     if hasattr(metric, 'total'):
    #         return metric.total > 0
        
    #     # For metrics with 'n_obs' attribute  
    #     if hasattr(metric, 'n_obs'):
    #         return metric.n_obs > 0
        
    #     # Default: assume it hasn't been updated if we can't determine otherwise
    #     return False


class MultiScoreRealDTIMetrics(torch.nn.Module):
    """
    Multi-score real-valued DTI metrics that can handle multiple DTI scores
    (pKd, pKi, KIBA) simultaneously with individual masking.
    """
    
    def __init__(self, score_names: List[str] = ["pKd", "pKi", "KIBA"], prefix: str = ""):
        """
        Initialize multi-score DTI metrics.
        
        Args:
            score_names: List of score names to track
            prefix: Prefix for metric names
        """
        super().__init__()
        self.score_names = score_names
        self.prefix = prefix
        
        # Create separate metric collections for each score
        self.metrics = torch.nn.ModuleDict({
            score: RealDTIMetrics(prefix=f"{prefix}{score}_")
            for score in score_names
        })
    
    def update(self, preds: Dict[str, Tensor], targets: Dict[str, Tensor], masks: Dict[str, Tensor]) -> None:
        """
        Update metrics for multiple scores.
        
        Args:
            preds: Dict mapping score names to predicted tensors
            targets: Dict mapping score names to target tensors  
            masks: Dict mapping score names to validity masks
        """
        for score_name in self.score_names:
            if score_name in preds and score_name in targets and score_name in masks:
                self.metrics[score_name].update(
                    preds[score_name], 
                    targets[score_name], 
                    masks[score_name]
                )
    
    def compute(self) -> Dict[str, Tensor]:
        """Compute all metrics."""
        results = {}
        for score_name in self.score_names:
            score_results = self.metrics[score_name].compute()
            results.update(score_results)
        return results
    
    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()


class BinaryDTIMetrics(MetricCollection):
    """
    Collection of metrics for binary DTI prediction.
    """
    
    def __init__(self, prefix: str = ""):
        """
        Initialize binary DTI metrics.
        
        Args:
            prefix: Prefix for metric names (e.g., "val_" or "test_")
        """
        metrics = {
            "accuracy": BinaryAccuracy(),
            "f1": BinaryF1Score(),
            "auroc": BinaryAUROC(),
            "auprc": BinaryAUPRC(),
            "confusion_matrix": BinaryConfusionMatrix(),
        }
        
        super().__init__(metrics, prefix=prefix)
    
    def compute(self) -> Dict[str, Tensor]:
        """
        Compute all metrics, handling non-scalar metrics appropriately.
        
        Returns:
            Dict of computed metrics (excluding non-scalar ones like confusion matrix)
        """
        results = {}
        
        for name, metric in self.items():
            value = metric.compute()
            
            # Skip non-scalar metrics (like confusion matrix) from logging
            if name.endswith("confusion_matrix"):
                continue
                
            results[name] = value
        
        return results


class DTIMetricsCollection(torch.nn.Module):
    """
    Complete collection of DTI metrics for both binary and real-valued predictions.
    
    This is the main metrics class that should be used in DTI models.
    """
    
    def __init__(
        self, 
        include_binary: bool = True,
        include_real: bool = True,
        real_score_names: List[str] = ["pKd", "pKi", "KIBA"],
        prefix: str = ""
    ):
        """
        Initialize DTI metrics collection.
        
        Args:
            include_binary: Whether to include binary classification metrics
            include_real: Whether to include real-valued regression metrics
            real_score_names: List of real-valued score names to track
            prefix: Prefix for metric names
        """
        super().__init__()
        self.include_binary = include_binary
        self.include_real = include_real
        self.prefix = prefix
        
        if include_binary:
            self.binary_metrics = BinaryDTIMetrics(prefix=f"{prefix}binary_")
        
        if include_real:
            self.real_metrics = MultiScoreRealDTIMetrics(
                score_names=real_score_names, 
                prefix=f"{prefix}real_"
            )
    
    def update(
        self, 
        binary_preds: Optional[Tensor] = None,
        binary_targets: Optional[Tensor] = None,
        real_preds: Optional[Dict[str, Tensor]] = None,
        real_targets: Optional[Dict[str, Tensor]] = None,
        real_masks: Optional[Dict[str, Tensor]] = None
    ) -> None:
        """
        Update all metrics.
        
        Args:
            binary_preds: Binary predictions [batch_size]
            binary_targets: Binary targets [batch_size]
            real_preds: Dict of real-valued predictions
            real_targets: Dict of real-valued targets
            real_masks: Dict of validity masks for real values
        """
        if self.include_binary and binary_preds is not None and binary_targets is not None:
            self.binary_metrics.update(binary_preds, binary_targets)
        
        if self.include_real and real_preds is not None and real_targets is not None and real_masks is not None:
            self.real_metrics.update(real_preds, real_targets, real_masks)
    
    def compute(self) -> Dict[str, Tensor]:
        """Compute all metrics."""
        results = {}
        
        if self.include_binary:
            binary_results = self.binary_metrics.compute()
            results.update(binary_results)
        
        if self.include_real:
            real_results = self.real_metrics.compute()
            results.update(real_results)
        
        return results
    
    def reset(self) -> None:
        """Reset all metrics."""
        if self.include_binary:
            self.binary_metrics.reset()
        
        if self.include_real:
            self.real_metrics.reset()

