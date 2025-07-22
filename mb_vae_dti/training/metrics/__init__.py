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