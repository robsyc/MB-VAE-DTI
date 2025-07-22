import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple, List


class DTIHead(nn.Module):
    """
    Drug-Target Interaction prediction head with multi-target support.
    
    Handles binary interaction prediction (always present) and optional 
    continuous targets (Kd, Ki, KIBA).
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Prediction heads
        self.binary_head = nn.Linear(hidden_dim, 1)
        self.pKd_head = nn.Linear(hidden_dim, 1)
        self.pKi_head = nn.Linear(hidden_dim, 1)
        self.KIBA_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Input features (batch_size, input_dim)
        
        Returns:
            predictions: Dict with prediction for each target (binary, Kd, Ki, KIBA)
        """
        shared_features = self.shared_layers(features)
        
        # Get predictions
        pred_binary_logits = self.binary_head(shared_features)
        pred_pKd = self.pKd_head(shared_features)
        pred_pKi = self.pKi_head(shared_features)
        pred_KIBA = self.KIBA_head(shared_features)
        
        predictions = {
            'binary_logits': pred_binary_logits,
            'pKd_pred': pred_pKd,
            'pKi_pred': pred_pKi,
            'KIBA_pred': pred_KIBA
        }
        
        return predictions


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
            output_dim: int, 
            temperature: float = 0.07, 
            use_negative_weighting: bool = True):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.temperature = temperature
        self.use_negative_weighting = use_negative_weighting

    @staticmethod
    def tanimoto_similarity(fp: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Tanimoto similarity matrix for binary fingerprints.
        
        Args:
            fp: Binary fingerprints (batch_size, n_bits)
        
        Returns:
            Similarity matrix (batch_size, batch_size)
        """
        fp = fp.float()
        
        # Compute intersection and union
        intersection = torch.mm(fp, fp.T)
        union = (fp.sum(dim=1).unsqueeze(1) + 
                fp.sum(dim=1).unsqueeze(0) - intersection)
        
        # Avoid division by zero
        union = torch.clamp(union, min=1e-8)
        
        return intersection / union

    def simple_info_nce_loss(self, features: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
        """
        Compute standard InfoNCE loss with discrete positive pairs.
        
        Simple implementation following standard InfoNCE: project, normalize, 
        find top-1 positive, apply cross-entropy.
        
        Args:
            features: Projected features (batch_size, output_dim)
            similarity: Pairwise Tanimoto similarity matrix (batch_size, batch_size)
        
        Returns:
            InfoNCE loss (scalar)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # 1. Project and normalize features
        features = F.normalize(features, dim=1)
        
        # 2. Compute cosine similarity logits
        logits = torch.mm(features, features.T) / self.temperature
        
        # 3. Mask out self-similarities (diagonal)
        eye_mask = torch.eye(batch_size, device=device).bool()
        logits = logits.masked_fill(eye_mask, -float('inf'))
        
        # 4. Find positive pairs: top-1 similar molecule for each anchor
        similarity_no_diag = similarity.masked_fill(eye_mask, 0.0)
        positive_indices = similarity_no_diag.argmax(dim=1)
        
        # 5. Compute cross-entropy loss
        loss = F.cross_entropy(logits, positive_indices)
        
        return loss

    def info_nce_loss(self, features: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss with discrete positive pairs based on Tanimoto similarity.
        
        For each anchor, finds the top-1 positive (highest Tanimoto similarity excluding self)
        and uses cross-entropy loss with these positive indices as targets.
        
        Args:
            features: Projected features (batch_size, output_dim)
            similarity: Pairwise Tanimoto similarity matrix (batch_size, batch_size)
        
        Returns:
            InfoNCE loss (scalar)
        """
        # Use simple version if negative weighting is disabled
        if not self.use_negative_weighting:
            return self.simple_info_nce_loss(features, similarity)
        
        batch_size = features.shape[0]
        device = features.device
        
        # 1. Project and normalize features
        features = F.normalize(features, dim=1)
        
        # 2. Compute cosine similarity logits
        logits = torch.mm(features, features.T) / self.temperature
        
        # 3. Mask out self-similarities (diagonal)
        eye_mask = torch.eye(batch_size, device=device).bool()
        logits = logits.masked_fill(eye_mask, -float('inf'))
        
        # 4. Find positive pairs: top-1 similar molecule for each anchor
        # Mask out diagonal in similarity matrix
        similarity_no_diag = similarity.masked_fill(eye_mask, 0.0)
        
        # Get the index of the most similar molecule for each anchor
        positive_indices = similarity_no_diag.argmax(dim=1)
        
        # 5. Optional: Weight negatives by (1 - Tanimoto) to soften hard negatives
        # This reduces the penalty for negative pairs that are actually similar
        negative_weights = 1.0 - similarity_no_diag
        negative_weights = negative_weights.masked_fill(eye_mask, 0.0)
        
        # Apply negative weighting to logits (vectorized approach)
        # Create mask for positive pairs
        positive_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
        positive_mask[torch.arange(batch_size), positive_indices] = True
        
        # Create combined mask: exclude self and positive pairs from negative weighting
        exclude_mask = eye_mask | positive_mask
        
        # Apply negative weighting: keep original logits for positives, weight negatives
        weighted_logits = logits.clone()
        weighted_logits[~exclude_mask] = logits[~exclude_mask] * negative_weights[~exclude_mask]
        
        # 6. Compute cross-entropy loss
        # Use positive_indices as targets for cross-entropy
        loss = F.cross_entropy(weighted_logits, positive_indices)
        
        return loss

    def forward(self, embeddings: torch.Tensor, fingerprints: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through InfoNCE head.
        
        Args:
            embeddings: Raw embeddings from encoder (batch_size, input_dim)
            fingerprints: Binary fingerprints for similarity computation (batch_size, n_bits)
        
        Returns:
            InfoNCE loss (scalar)
        """
        # Project embeddings
        projected_features = self.projection(embeddings)
        
        # Compute Tanimoto similarity matrix
        similarity_matrix = self.tanimoto_similarity(fingerprints)
        
        # Compute InfoNCE loss
        loss = self.info_nce_loss(projected_features, similarity_matrix)
        
        return loss


class KLVariationalHead(nn.Module):
    """
    VAE-style complexity loss with reparameterization trick.
    
    This module projects input features to mean and log-variance,
    applies reparameterization sampling, and computes KL divergence
    from standard normal prior.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, output_dim)
        self.fc_logvar = nn.Linear(input_dim, output_dim)
        
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            mu and logvar: to be used for KL divergence
            sampled: sampled latent vector (batch_size, output_dim)
        """
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        sampled = self.reparameterize(mu, logvar)
        return mu, logvar, sampled