"""
Training and evaluation metrics for DTITree models.

This module provides modular metrics for:
- Validation metrics (NLL, KL divergence, cross-entropy)
- Molecular generation metrics (validity, similarity, accuracy)
- DTI evaluation metrics (RMSE, Binary Accuracy, etc.)

Organization:
- validation_metrics.py: Standard validation metrics
- molecular_metrics.py: Molecular generation evaluation
- dti_metrics.py: Drug-target interaction evaluation
"""

from .validation_metrics import (
    NLL,
    SumExceptBatchKL,
    SumExceptBatchMetric
)
from .molecular_metrics import (
    TrainMolecularMetricsDiscrete
)
from .dti_metrics import (
    RealDTIMetrics,
    MultiScoreRealDTIMetrics,
    BinaryDTIMetrics,
    DTIMetricsCollection,
)

# Simplified diffusion metrics collection
class DiffusionMetrics:
    """Simplified collection of diffusion-related metrics."""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.nll = NLL()
        self.X_kl = SumExceptBatchKL()
        self.E_kl = SumExceptBatchKL()
        self.X_logp = SumExceptBatchMetric()
        self.E_logp = SumExceptBatchMetric()
        
        # Molecular metrics (to be added later if needed)
        # self.validity = ValidityMetric()
        # self.uniqueness = UniquenessMetric()
        # self.novelty = NoveltyMetric()
    
    def reset(self):
        """Reset all metrics."""
        self.nll.reset()
        self.X_kl.reset()
        self.E_kl.reset()
        self.X_logp.reset()
        self.E_logp.reset()
    
    def compute(self):
        """Compute all metrics."""
        return {
            f"{self.prefix}NLL": self.nll.compute(),
            f"{self.prefix}X_KL": self.X_kl.compute(),
            f"{self.prefix}E_KL": self.E_kl.compute(),
            f"{self.prefix}X_logp": self.X_logp.compute(),
            f"{self.prefix}E_logp": self.E_logp.compute(),
        }

__all__ = [
    # Validation Metrics
    "NLL",
    "SumExceptBatchKL",
    "SumExceptBatchMetric",
    
    # Molecular Generation Metrics
    "TrainMolecularMetricsDiscrete",
    
    # DTI Metrics
    "RealDTIMetrics",           # MSE, RMSE, PearsonCorrCoef, R2Score, ConcordanceIndex
    "MultiScoreRealDTIMetrics", # for pKd, pKi, and KIBA
    "BinaryDTIMetrics",         # BinaryAccuracy, BinaryConfusionMatrix, BinaryF1Score, BinaryAUPRC, BinaryAUROC
    "DTIMetricsCollection",
] 