"""
Validation metrics for DTITree decoder (validation / test metrics for diffusion model).
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics import Metric


class NLL(Metric):
    """
    Negative Log-Likelihood metric for diffusion models.
    """
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples


class SumExceptBatchKL(Metric):
    """
    KL divergence metric that sums over all dimensions except batch.
    """
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
    """
    Generic metric that sums over all dimensions except batch.
    """
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples