"""
Validation metrics for DTITree decoder (validation / test metrics for diffusion model).
"""

import torch
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
import torch.nn as nn
from typing import Dict, Any, Optional
import wandb


class NLL(Metric):
    """Basic NLL metric that computes average negative log likelihood over batches."""
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll: torch.Tensor) -> None:
        """Update with batch NLL values.
        
        Args:
            batch_nll: NLL values for each sample in batch [batch_size]
        """
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchMetric(Metric):
    """Generic metric that sums over all dimensions except batch."""
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor) -> None:
        """Update with batch values.
        
        Args:
            values: Values to accumulate [batch_size, ...]
        """
        batch_sums = values.sum(dim=tuple(range(1, len(values.shape))))  # Sum all except batch
        self.total_value += batch_sums.sum()
        self.total_samples += batch_sums.numel()

    def compute(self):
        return self.total_value / self.total_samples