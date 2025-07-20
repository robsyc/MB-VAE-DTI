"""
Validation metrics for DTITree models.

This module contains TorchMetrics-compatible validation metrics for diffusion models,
leveraging TorchMetrics implementations where possible for reliability and maintainability.

Key metrics include:
- NLL: Negative Log-Likelihood for diffusion models
- SumExceptBatchKL: KL divergence summed over non-batch dimensions
- CrossEntropyMetric: Cross-entropy loss tracking
- ProbabilityMetric: Probability distribution tracking
- ELBOMetric: Evidence Lower Bound computation
- DiffusionMetricsCollection: Comprehensive diffusion validation metrics
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torchmetrics.aggregation import MeanMetric
from typing import Optional, Union, Dict


class NLL(MeanMetric):
    """
    Negative Log-Likelihood metric for diffusion models.
    
    Inherits from TorchMetrics MeanMetric for robust distributed training support.
    This metric tracks the negative log-likelihood of the diffusion model.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def update(self, batch_nll: Tensor) -> None:
        """Update with batch NLL values."""
        super().update(batch_nll)


class SumExceptBatchKL(Metric):
    """
    KL divergence metric that sums over all dimensions except batch.
    
    This is commonly used in diffusion models to compute KL divergence
    between predicted and true distributions. Custom implementation needed
    since TorchMetrics doesn't have this specific pattern.
    """
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('total_kl', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p: Tensor, q: Tensor) -> None:
        """Update with predicted and true distributions."""
        # Compute KL divergence: KL(p || q) = sum(p * log(p/q))
        eps = 1e-8
        p_safe = torch.clamp(p, min=eps)
        q_safe = torch.clamp(q, min=eps)
        
        # KL divergence computation
        kl_div = p_safe * torch.log(p_safe / q_safe)
        
        # Sum over all dimensions except batch
        batch_size = p.size(0)
        kl_sum = torch.sum(kl_div.view(batch_size, -1), dim=1)
        
        self.total_kl += torch.sum(kl_sum)
        self.total_samples += batch_size

    def compute(self) -> Tensor:
        """Compute average KL divergence."""
        if self.total_samples == 0:
            return torch.tensor(0.0)
        return self.total_kl / self.total_samples


class SumExceptBatchMetric(Metric):
    """
    Generic metric that sums over all dimensions except batch.
    
    Useful for computing various metrics in diffusion models where
    we want to sum over spatial/feature dimensions but average over batch.
    """
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('total_value', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, values: Tensor) -> None:
        """Update with tensor values."""
        batch_size = values.size(0)
        # Sum over all dimensions except batch
        batch_sums = torch.sum(values.view(batch_size, -1), dim=1)
        
        self.total_value += torch.sum(batch_sums)
        self.total_samples += batch_size

    def compute(self) -> Tensor:
        """Compute average value."""
        if self.total_samples == 0:
            return torch.tensor(0.0)
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    """
    Cross-entropy metric for discrete predictions.
    
    Computes cross-entropy loss between predicted and true discrete distributions.
    Follows TorchMetrics patterns for consistency.
    """
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('total_ce', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update with predictions and targets.
        
        Args:
            preds: Predicted logits (batch_size, ..., num_classes)
            target: True labels (batch_size, ..., num_classes) or indices
        """
        # Handle both one-hot and index targets
        if target.dtype == torch.long:
            # Target is indices
            target_indices = target
        else:
            # Target is one-hot, convert to indices
            target_indices = torch.argmax(target, dim=-1)
        
        # Reshape for cross-entropy computation
        preds_flat = preds.view(-1, preds.size(-1))
        target_flat = target_indices.view(-1)
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(preds_flat, target_flat, reduction='sum')
        
        self.total_ce += ce_loss
        self.total_samples += target_flat.numel()

    def compute(self) -> Tensor:
        """Compute average cross-entropy."""
        if self.total_samples == 0:
            return torch.tensor(0.0)
        return self.total_ce / self.total_samples


class ProbabilityMetric(MeanMetric):
    """
    Probability metric for tracking probability distributions.
    
    Computes statistics of predicted probabilities, useful for
    monitoring the quality of predicted distributions.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def update(self, preds: Tensor) -> None:
        """Update with predicted probabilities."""
        # Apply softmax to get probabilities
        probs = F.softmax(preds, dim=-1)
        
        # Compute mean probability (confidence measure)
        mean_prob = torch.mean(torch.max(probs, dim=-1)[0])
        
        super().update(mean_prob)


class ELBOMetric(Metric):
    """
    Evidence Lower Bound (ELBO) metric for diffusion models.
    
    Computes the ELBO components: log p(N), KL prior, diffusion loss, and reconstruction loss.
    This is essential for monitoring diffusion model training.
    """
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('log_pN_sum', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('kl_prior_sum', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('loss_t_sum', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('loss_0_sum', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, log_pN: Tensor, kl_prior: Tensor, loss_t: Tensor, loss_0: Tensor) -> None:
        """Update with ELBO components."""
        batch_size = log_pN.size(0)
        
        self.log_pN_sum += torch.sum(log_pN)
        self.kl_prior_sum += torch.sum(kl_prior)
        self.loss_t_sum += torch.sum(loss_t)
        self.loss_0_sum += torch.sum(loss_0)
        self.total_samples += batch_size

    def compute(self) -> dict:
        """Compute ELBO components."""
        if self.total_samples == 0:
            return {
                'log_pN': torch.tensor(0.0),
                'kl_prior': torch.tensor(0.0),
                'loss_t': torch.tensor(0.0),
                'loss_0': torch.tensor(0.0),
                'elbo': torch.tensor(0.0)
            }
        
        log_pN_avg = self.log_pN_sum / self.total_samples
        kl_prior_avg = self.kl_prior_sum / self.total_samples
        loss_t_avg = self.loss_t_sum / self.total_samples
        loss_0_avg = self.loss_0_sum / self.total_samples
        
        # ELBO = log p(N) - KL_prior + loss_t - loss_0
        elbo = log_pN_avg - kl_prior_avg + loss_t_avg - loss_0_avg
        
        return {
            'log_pN': log_pN_avg,
            'kl_prior': kl_prior_avg,
            'loss_t': loss_t_avg,
            'loss_0': loss_0_avg,
            'elbo': elbo
        }


class DiffusionMetricsCollection:
    """
    Collection of validation metrics for diffusion models.
    
    This class provides a convenient way to initialize and manage
    all the validation metrics needed for diffusion model evaluation.
    Uses TorchMetrics implementations where possible for reliability.
    """
    
    def __init__(self):
        # Core diffusion metrics
        self.nll = NLL()
        self.node_kl = SumExceptBatchKL()
        self.edge_kl = SumExceptBatchKL()
        self.node_ce = CrossEntropyMetric()
        self.edge_ce = CrossEntropyMetric()
        self.node_prob = ProbabilityMetric()
        self.edge_prob = ProbabilityMetric()
        self.elbo = ELBOMetric()
        
        # Additional metrics for monitoring
        self.node_logp = SumExceptBatchMetric()
        self.edge_logp = SumExceptBatchMetric()
        
        # Collect all metrics
        self.all_metrics = {
            'nll': self.nll,
            'node_kl': self.node_kl,
            'edge_kl': self.edge_kl,
            'node_ce': self.node_ce,
            'edge_ce': self.edge_ce,
            'node_prob': self.node_prob,
            'edge_prob': self.edge_prob,
            'node_logp': self.node_logp,
            'edge_logp': self.edge_logp,
            'elbo': self.elbo,
        }
    
    def update_reconstruction_metrics(self, pred_X: Tensor, pred_E: Tensor, 
                                    true_X: Tensor, true_E: Tensor):
        """Update reconstruction-related metrics."""
        # Update cross-entropy metrics
        if true_X.numel() > 0:
            self.node_ce.update(pred_X, true_X)
            self.node_prob.update(pred_X)
        
        if true_E.numel() > 0:
            self.edge_ce.update(pred_E, true_E)
            self.edge_prob.update(pred_E)
    
    def update_kl_metrics(self, pred_X: Tensor, pred_E: Tensor, 
                         true_X: Tensor, true_E: Tensor):
        """Update KL divergence metrics."""
        if true_X.numel() > 0:
            self.node_kl.update(pred_X, true_X)
        
        if true_E.numel() > 0:
            self.edge_kl.update(pred_E, true_E)
    
    def update_logp_metrics(self, node_logp: Tensor, edge_logp: Tensor):
        """Update log-probability metrics."""
        if node_logp.numel() > 0:
            self.node_logp.update(node_logp)
        
        if edge_logp.numel() > 0:
            self.edge_logp.update(edge_logp)
    
    def update_elbo_metrics(self, log_pN: Tensor, kl_prior: Tensor, 
                           loss_t: Tensor, loss_0: Tensor):
        """Update ELBO metrics."""
        self.elbo.update(log_pN, kl_prior, loss_t, loss_0)
    
    def update_nll(self, nll: Tensor):
        """Update NLL metric."""
        self.nll.update(nll)
    
    def compute_all(self) -> Dict[str, Tensor]:
        """Compute all metrics and return as dictionary."""
        results = {}
        
        for name, metric in self.all_metrics.items():
            if name == 'elbo':
                # ELBO returns a dictionary
                elbo_results = metric.compute()
                results.update(elbo_results)
            else:
                results[name] = metric.compute()
        
        return results
    
    def reset_all(self) -> None:
        """Reset all metrics."""
        for metric in self.all_metrics.values():
            metric.reset()
    
    def log_summary(self, results: Optional[Dict[str, Tensor]] = None) -> str:
        """Generate a summary string of all metrics."""
        if results is None:
            results = self.compute_all()
        
        summary_lines = ["Diffusion Validation Metrics:"]
        
        # Core metrics
        summary_lines.append(f"  NLL: {results.get('nll', 0.0):.4f}")
        summary_lines.append(f"  Node KL: {results.get('node_kl', 0.0):.4f}")
        summary_lines.append(f"  Edge KL: {results.get('edge_kl', 0.0):.4f}")
        summary_lines.append(f"  Node CE: {results.get('node_ce', 0.0):.4f}")
        summary_lines.append(f"  Edge CE: {results.get('edge_ce', 0.0):.4f}")
        
        # ELBO components
        summary_lines.append("  ELBO Components:")
        summary_lines.append(f"    log p(N): {results.get('log_pN', 0.0):.4f}")
        summary_lines.append(f"    KL prior: {results.get('kl_prior', 0.0):.4f}")
        summary_lines.append(f"    Loss t: {results.get('loss_t', 0.0):.4f}")
        summary_lines.append(f"    Loss 0: {results.get('loss_0', 0.0):.4f}")
        summary_lines.append(f"    ELBO: {results.get('elbo', 0.0):.4f}")
        
        return "\n".join(summary_lines) 