"""
Encoder, aggregator and fusion neural network modules for the DTITree model.

Used to process and transform fixed-size arbitrary-length feature vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Literal
import math


################################################################################
# Encoders
################################################################################

class ResidualEncoder(nn.Module):
    """ MLP encoder with residual connections and pre-norm architecture. """
    def __init__(
            self, 
            input_dim: int, 
            hidden_dim: int,
            output_dim: int, 
            n_layers: int, 
            factor: int = 4,
            dropout: float = 0.1, 
            bias: bool = True,
            activation: nn.Module = nn.ReLU()
):
        super().__init__()

        self.n_layers = n_layers
        self.activation = activation
        
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim, bias=bias),
                nn.Linear(hidden_dim, hidden_dim * factor, bias=bias),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * factor, hidden_dim, bias=bias)
            ))
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim, bias=bias),
            nn.Linear(hidden_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Encoded tensor of shape (batch_size, output_dim)
        """
        x = self.input_proj(x)
        
        for i in range(self.n_layers):
            x = x + self.layers[i](x)
        
        return self.output_proj(x)


class TransformerEncoder(nn.Module):
    """ Transformer encoder with residual connections and pre-norm architecture. """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
        output_dim: int, 
        n_layers: int,
        factor: int = 4, 
        dropout: float = 0.1, 
        bias: bool = True,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()

        self.n_layers = n_layers
        self.activation = activation

        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.att_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.att_layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim, bias=bias),
                nn.Linear(hidden_dim, hidden_dim // 4, bias=bias),
                self.activation,
                nn.Linear(hidden_dim // 4, hidden_dim, bias=bias),
                nn.Sigmoid(),  # Gate values between 0 and 1
                nn.Dropout(dropout)
            ))
            self.mlp_layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim, bias=bias),
                nn.Linear(hidden_dim, hidden_dim * factor, bias=bias),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * factor, hidden_dim, bias=bias)
            ))
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim, bias=bias),
            nn.Linear(hidden_dim, output_dim, bias=bias)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with alternating self-attention and MLP layers.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Encoded tensor of shape (batch_size, output_dim)
        """
        x = self.input_proj(x)
        
        for i in range(self.n_layers):
            x = x + x * self.att_layers[i](x) 
            x = x + self.mlp_layers[i](x)
        
        return self.output_proj(x)


################################################################################
# Aggregators
################################################################################

class ConcatAggregator(nn.Module):
    """ Concatenates multiple feature vectors and projects to output dimension. """
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            n_features: int,
            dropout: float = 0.1, 
            activation: nn.Module = nn.ReLU()
        ):
        super().__init__()

        total_dim = input_dim * n_features
        self.activation = activation

        self.projection = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, output_dim),
            self.activation,
            nn.Dropout(dropout)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of tensors, each of shape (batch_size, input_dim)
        
        Returns:
            Aggregated tensor of shape (batch_size, output_dim)
        """
        concatenated = torch.cat(features, dim=-1)
        aggregated = self.projection(concatenated)
        return aggregated


class AttentiveAggregator(nn.Module):
    """ Aggregates features using simple attention-weighted sum. """
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            n_features: int,
            dropout: float = 0.1, 
            activation: nn.Module = nn.ReLU()):
        super().__init__()
        
        self.activation = activation

        self.to_attn_logits = nn.Conv1d(input_dim, 1, 1)

        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            self.activation,
            nn.Dropout(dropout)
        )
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List of tensors, each of shape (batch_size, input_dim)
        
        Returns:
            aggregated: Aggregated tensor of shape (batch_size, output_dim)
            attention_weights: Attention weights of shape (batch_size, n_features)
        """
        stacked = torch.stack(features, dim=-1)                 # (B, input_dim, n_features)
        attention_logits = self.to_attn_logits(stacked)         # (B, 1, n_features)
        attention_weights = F.softmax(attention_logits, dim=-1) # (B, 1, n_features)
        aggregated = (stacked * attention_weights).sum(dim=-1)  # (B, input_dim)

        return self.projection(aggregated), attention_weights


################################################################################
# Fusion
################################################################################

class CrossAttentionFusion(nn.Module):
    """ Cross-attention fusion for drug-target interaction. """
    def __init__(
            self, 
            input_dim: int,
            output_dim: int,
            n_layers: int,
            factor: int = 4,
            dropout: float = 0.1, 
            bias: bool = True,
            activation: nn.Module = nn.ReLU()):
        super().__init__()

        self.n_layers = n_layers
        self.activation = activation
        
        # Cross-attention gates: simple gating mechanism for bidirectional information flow
        self.cross_gates_a_to_b = nn.ModuleList() # Calculates a weight for a parameter update
        self.cross_gates_b_to_a = nn.ModuleList()
        self.align_proj_b_to_a = nn.ModuleList()  # Projects b to align with a
        self.align_proj_a_to_b = nn.ModuleList()  # Projects a to align with b
        
        # Self-attention gates: refinement of each embedding
        self.self_gates_a = nn.ModuleList()
        self.self_gates_b = nn.ModuleList()
        
        # MLP layers for both drug and target
        self.mlp_a = nn.ModuleList()
        self.mlp_b = nn.ModuleList()
        
        for _ in range(n_layers):
            # Cross-attention gates (drug attends to target, target attends to drug)
            self.align_proj_b_to_a.append(nn.Linear(input_dim, input_dim, bias=False))
            self.align_proj_a_to_b.append(nn.Linear(input_dim, input_dim, bias=False))
            self.cross_gates_a_to_b.append(nn.Sequential(
                nn.LayerNorm(input_dim * 2, bias=bias),
                nn.Linear(input_dim * 2, input_dim // factor, bias=bias),
                self.activation,
                nn.Linear(input_dim // factor, input_dim, bias=bias),
                nn.Sigmoid(),
                nn.Dropout(dropout)
            ))
            self.cross_gates_b_to_a.append(nn.Sequential(
                nn.LayerNorm(input_dim * 2, bias=bias),
                nn.Linear(input_dim * 2, input_dim // factor, bias=bias),
                self.activation,
                nn.Linear(input_dim // factor, input_dim, bias=bias),
                nn.Sigmoid(),
                nn.Dropout(dropout)
            ))
            
            # Self-attention gates (same pattern as TransformerEncoder)
            self.self_gates_a.append(nn.Sequential(
                nn.LayerNorm(input_dim, bias=bias),
                nn.Linear(input_dim, input_dim // factor, bias=bias),
                self.activation,
                nn.Linear(input_dim // factor, input_dim, bias=bias),
                nn.Sigmoid(),
                nn.Dropout(dropout)
            ))
            self.self_gates_b.append(nn.Sequential(
                nn.LayerNorm(input_dim, bias=bias),
                nn.Linear(input_dim, input_dim // factor, bias=bias),
                self.activation,
                nn.Linear(input_dim // factor, input_dim, bias=bias),
                nn.Sigmoid(),
                nn.Dropout(dropout)
            ))

            # MLP blocks
            self.mlp_a.append(nn.Sequential(
                nn.LayerNorm(input_dim, bias=bias),
                nn.Linear(input_dim, input_dim, bias=bias),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(input_dim, input_dim, bias=bias)
            ))
            self.mlp_b.append(nn.Sequential(
                nn.LayerNorm(input_dim, bias=bias),
                nn.Linear(input_dim, input_dim, bias=bias),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(input_dim, input_dim, bias=bias)
            ))

        # Final aggregation
        self.att_agg = AttentiveAggregator(
            input_dim=input_dim,
            output_dim=output_dim,
            n_features=2,
            dropout=dropout,
            activation=activation
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: Drug embedding of shape (batch_size, input_dim)
            b: Target embedding of shape (batch_size, input_dim)

        Returns:
            Fused tensor of shape (batch_size, output_dim)
        """
        for i in range(self.n_layers):
            # Cross-attention: bidirectional information flow with gating
            gate_a = self.cross_gates_b_to_a[i](torch.cat([a, b], dim=-1))
            gate_b = self.cross_gates_a_to_b[i](torch.cat([b, a], dim=-1))
            a = a + gate_a * self.align_proj_b_to_a[i](b)
            b = b + gate_b * self.align_proj_a_to_b[i](a)

            # Self-attention: refine each embedding individually (like TransformerEncoder)
            a = a + a * self.self_gates_a[i](a)
            b = b + b * self.self_gates_b[i](b)

            # MLP blocks
            a = a + self.mlp_a[i](a)
            b = b + self.mlp_b[i](b)
        
        # Final aggregation
        return self.att_agg([a, b])