import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple, List, Union

from mb_vae_dti.training.diffusion.utils import PlaceHolder
import logging

logger = logging.getLogger(__name__)

class DTIHead(nn.Module):
    """
    Drug-Target Interaction prediction head with multi-target support.

    Handles binary interaction prediction (always present) and optional 
    continuous targets (Kd, Ki, KIBA).
    """
    def __init__(
            self, 
            input_dim: int, 
            proj_dim: int,
            dropout: float,
            bias: bool,
            activation: nn.Module
    ):
        super().__init__()
        
        self.activation = activation

        self.shared_layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, proj_dim, bias=bias),
            self.activation,
            nn.Dropout(dropout),
        )
        
        self.binary_head = nn.Linear(proj_dim, 1)
        self.pKd_head = nn.Linear(proj_dim, 1)
        self.pKi_head = nn.Linear(proj_dim, 1)
        self.KIBA_head = nn.Linear(proj_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Input features (batch_size, input_dim)
        
        Returns:
            predictions: Dict with dti_scores preds (Y, pKd, pKi, KIBA)
        """
        shared_features = self.shared_layers(features)
        return {
            'Y': self.binary_head(shared_features).squeeze(-1),  # (B,)
            'Y_pKd': self.pKd_head(shared_features).squeeze(-1),   # (B,)
            'Y_pKi': self.pKi_head(shared_features).squeeze(-1),   # (B,)
            'Y_KIBA': self.KIBA_head(shared_features).squeeze(-1)  # (B,)
        }


class InfoNCEHead(nn.Module):
    """
    Contrastive learning head using InfoNCE with molecular similarity weighting.
    
    This implementation uses Tanimoto similarity on molecular fingerprints
    to identify positive pairs and apply InfoNCE loss, encouraging the model to learn
    representations that respect molecular similarity.
    """
    def __init__(
            self, 
            input_dim: int, 
            proj_dim: int,
            dropout: float,
            bias: bool,
            activation: nn.Module
    ):
        super().__init__()
        self.activation = activation
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, proj_dim, bias=bias),
            self.activation,
            nn.Dropout(dropout),
        )

    @staticmethod
    def tanimoto_similarity(fp: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Tanimoto similarity matrix for binary fingerprints.
        
        Args:
            fp: Binary fingerprints (batch_size, n_bits)
        
        Returns:
            Similarity matrix (batch_size, batch_size)
        """        
        # Compute intersection and union
        intersection = torch.mm(fp, fp.T)
        union = (fp.sum(dim=1).unsqueeze(1) + 
                fp.sum(dim=1).unsqueeze(0) - intersection)
        
        # Avoid division by zero
        union = torch.clamp(union, min=1e-8)
        
        return intersection / union

    def info_nce_loss(
            self, 
            features: torch.Tensor, 
            similarity_matrix: torch.Tensor,
            temperature: float
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with discrete positive pairs based on Tanimoto similarity.
        
        For each anchor, finds the top-1 positive (highest Tanimoto similarity excluding self)
        and uses cross-entropy loss with these positive indices as targets.
        
        Args:
            features: Projected features (batch_size, output_dim)
            similarity_matrix: Pairwise Tanimoto similarity matrix (batch_size, batch_size)
            temperature: Temperature for logits scaling
        
        Returns:
            InfoNCE loss tensorscalar)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # 1. Project and normalize features
        features = F.normalize(features, dim=1)
        
        # 2. Compute cosine similarity logits
        logits = torch.mm(features, features.T) / temperature
        
        # 3. Mask out self-similarities (diagonal)
        eye_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        logits = logits.masked_fill(eye_mask, -float('inf'))
        
        # 4. Find positive pairs: top-1 similar molecule for each anchor
        # Mask out diagonal in similarity matrix
        similarity_matrix = similarity_matrix.masked_fill(eye_mask, 0.0)
        
        # Get the index of the most similar molecule for each anchor
        positive_indices = similarity_matrix.argmax(dim=1)
        
        # 5. Optional: Weight negatives by (1 - Tanimoto) to soften hard negatives
        # This reduces the penalty for negative pairs that are actually similar
        negative_weights = 1.0 - similarity_matrix
        negative_weights = negative_weights.masked_fill(eye_mask, 0.0)
        
        # Apply negative weighting to logits (vectorized approach)
        # Create mask for positive pairs
        positive_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
        positive_mask[torch.arange(batch_size), positive_indices] = True
        
        # Create combined mask: exclude self and positive pairs from negative weighting
        exclude_mask = eye_mask | positive_mask
        
        # Apply negative weighting: keep original logits for positives, weight negatives
        weighted_logits = logits.clone()
        logger.info(f"weighted_logits: {type(weighted_logits)}, dtype: {weighted_logits.dtype}")
        logger.info(f"logits: {type(logits)}, dtype: {logits.dtype}")
        logger.info(f"negative_weights: {type(negative_weights)}, dtype: {negative_weights.dtype}")
        logger.info(f"exclude_mask: {type(exclude_mask)}, dtype: {exclude_mask.dtype}")
        weighted_logits[~exclude_mask] = logits[~exclude_mask] * negative_weights[~exclude_mask]
        
        # 6. Compute cross-entropy loss
        # Use positive_indices as targets for cross-entropy
        loss = F.cross_entropy(weighted_logits, positive_indices)
        
        return loss

    def forward(
            self, 
            x: torch.Tensor, 
            fingerprints: torch.Tensor,
            temperature: float
            ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, input_dim)
            fingerprints: Binary fingerprints (batch_size, n_bits)
            temperature: Temperature for logits scaling

        Returns:
            loss: InfoNCE loss (scalar)
        """
        z = self.projection(x)
        similarity_matrix = self.tanimoto_similarity(fingerprints)
        return self.info_nce_loss(z, similarity_matrix, temperature)


class KLVariationalHead(nn.Module):
    """
    VAE-style complexity loss with reparameterization trick.
    
    This module projects input features to mean and log-variance,
    applies reparameterization sampling, and computes KL divergence
    from standard normal prior.
    """
    def __init__(
            self, 
            input_dim: int, 
            proj_dim: int
        ):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, proj_dim, bias=True)
        self.fc_logvar = nn.Linear(input_dim, proj_dim, bias=True)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for differentiable sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)   # sample epsilon from N(0, 1)
        return mu + eps * std

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from standard normal: KL(q(z|x) || p(z)).
        
        Proper normalization: divide by both batch_size AND latent_dimension
        to get average KL per sample per dimension, making it scale-invariant.
        """
        batch_size, latent_dim = mu.shape
        
        kl_total = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        kl_normalized = kl_total / (batch_size * latent_dim)
        
        return kl_normalized
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            z: sampled latent vector (batch_size, proj_dim)
            mu: mean of the latent distribution (batch_size, proj_dim)
            logvar: log-variance of the latent distribution (batch_size, proj_dim)
        """
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return self.reparameterize(mu, logvar), mu, logvar


class ReconstructionHead(nn.Module):
    """
    Reconstruction head for the diffusion model.
    """
    def __init__(self, diff_weights: List[int]):
        super().__init__()
        assert len(diff_weights) == 2, "Reconstruction head expects 2 weights for X and E"
        self.diff_weights = diff_weights

    def forward(
            self, 
            masked_pred_X: torch.Tensor, 
            masked_pred_E: torch.Tensor, 
            true_X: torch.Tensor, 
            true_E: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Args:
            masked_pred_X: Predicted masked graph X
            masked_pred_E: Predicted masked graph E
            true_X: True graph X
            true_E: True graph E
        
        Returns:
            loss: Reconstruction loss (scalar) weighted by diff_weights
        """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)
        
        flat_true_X = torch.argmax(true_X[mask_X, :], dim=-1)
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = torch.argmax(true_E[mask_E, :], dim=-1)
        flat_pred_E = masked_pred_E[mask_E, :]

        # Use sum reduction but normalize by number of samples (following DiffMS pattern)
        if flat_pred_X.numel() > 0:
            loss_X_raw = F.cross_entropy(flat_pred_X, flat_true_X, reduction='sum')
            loss_X = loss_X_raw / flat_pred_X.size(0)  # Normalize by number of samples
        else:
            loss_X = 0.0
            
        if flat_pred_E.numel() > 0:
            loss_E_raw = F.cross_entropy(flat_pred_E, flat_true_E, reduction='sum') 
            loss_E = loss_E_raw / flat_pred_E.size(0)  # Normalize by number of samples
        else:
            loss_E = 0.0

        return loss_X * self.diff_weights[0] + \
               loss_E * self.diff_weights[1]