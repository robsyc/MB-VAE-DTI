"""
Encoder, aggregator and fusion neural network modules for the DTITree model.

Used to process and transform fixed-size arbitrary-length feature vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
import math


################################################################################
# Helper layers
################################################################################

class MLP(nn.Module):
    def __init__(
        self, 
        dim: int, 
        expansion_factor: int = 4,
        dropout: float = 0.1, 
        bias: bool = True,
        activation: nn.Module = nn.GELU()
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * expansion_factor, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim * expansion_factor, dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        

class SelfAttention(nn.Module):
    """ Multi-head self-attention layer that computes attention between feature dimensions. """
    def __init__(
        self, 
        dim: int,
        n_heads: int = 4, 
        dropout: float = 0.1, 
        bias: bool = True
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        B, L, D = x.shape

        # Project to QKV and split into heads
        qkv = self.qkv_proj(x)  # (B, L, 3*D)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim) 
        q, k, v = qkv.unbind(2)

        # Attention scores (between feature tokens)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, n_heads, L, L)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Combine with values
        attn_out = torch.matmul(attn_weights, v)  # (B, n_heads, L, head_dim)

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2)  # (B, L, n_heads, head_dim)
        attn_out = attn_out.reshape(B, L, D)  # (B, L, dim)
        return self.out_proj(attn_out)


class CrossAttention(nn.Module):
    """ Multi-head cross-attention layer. """
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (query) of shape (batch_size, seq_len_q, dim)
            context: Context tensor (key/value) of shape (batch_size, seq_len_kv, dim)

        Returns:
            Output tensor of shape (batch_size, seq_len_q, dim)
        """
        B_x, L_q, D = x.shape
        B_c, L_kv, _ = context.shape
        assert B_x == B_c, "batch size of x and context must be the same"

        q = self.q_proj(x)
        kv = self.kv_proj(context)

        q = q.view(B_x, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        
        kv = kv.view(B_c, L_kv, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B_x, L_q, D)
        
        return self.out_proj(attn_out)


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
        expansion_factor: int = 4,
        dropout: float = 0.1, 
        bias: bool = True,
        activation: Union[str, nn.Module] = "gelu"
    ):
        super().__init__()

        if isinstance(activation, str):
            activation_map = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "swish": nn.SiLU(),
                "mish": nn.Mish()
            }
            self.activation = activation_map.get(activation.lower(), nn.GELU())
        else:
            self.activation = activation
        
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.mlp_layers = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        
        for _ in range(n_layers):
            self.mlp_norms.append(nn.LayerNorm(hidden_dim, bias=bias))
            mlp_layer = MLP(hidden_dim, expansion_factor=expansion_factor, dropout=dropout, bias=bias, activation=self.activation)
            self.mlp_layers.append(mlp_layer)
        
        self.output_norm = nn.LayerNorm(hidden_dim, bias=bias)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Encoded tensor of shape (batch_size, output_dim)
        """
        h = self.input_proj(x)
        
        for mlp_layer, mlp_norm in zip(self.mlp_layers, self.mlp_norms):
            h = h + mlp_layer(mlp_norm(h))
        
        h = self.output_norm(h)
        return self.output_proj(h)


class TransformerEncoder(nn.Module):
    """ Transformer encoder with residual connections and pre-norm architecture. """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
        output_dim: int, 
        n_layers: int,
        expansion_factor: int = 4, 
        n_heads: int = 4,
        dropout: float = 0.1, 
        bias: bool = True,
        activation: Union[str, nn.Module] = "gelu"
    ):
        super().__init__()

        if isinstance(activation, str):
            activation_map = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "swish": nn.SiLU(),
                "mish": nn.Mish()
            }
            self.activation = activation_map.get(activation.lower(), nn.GELU())
        else:
            self.activation = activation
        
        self.input_proj = nn.Linear(1, hidden_dim, bias=bias)
        self.attention_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        
        for _ in range(n_layers):
            self.attention_layers.append(SelfAttention(
                dim=hidden_dim,
                n_heads=n_heads, 
                dropout=dropout, 
                bias=bias
            ))
            self.attn_norms.append(nn.LayerNorm(hidden_dim, bias=bias))
            
            self.mlp_layers.append(MLP(
                dim=hidden_dim,
                expansion_factor=expansion_factor,
                dropout=dropout,
                bias=bias,
                activation=self.activation
            ))
            self.mlp_norms.append(nn.LayerNorm(hidden_dim, bias=bias))
        
        self.output_norm = nn.LayerNorm(hidden_dim, bias=bias)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with alternating self-attention and MLP layers.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Encoded tensor of shape (batch_size, output_dim)
        """
        h = x.unsqueeze(-1)     # (B, input_dim) -> (B, input_dim, 1)
        h = self.input_proj(h)  # (B, input_dim, 1) -> (B, input_dim, hidden_dim)
        
        for i in range(len(self.attention_layers)):
            # Attention block
            residual = h
            h = self.attn_norms[i](h)
            h = self.attention_layers[i](h)
            h = residual + self.dropouts[i](h)
            
            # MLP block
            residual = h
            h = self.mlp_norms[i](h)
            h = self.mlp_layers[i](h)
            h = residual + self.dropouts[i](h)
        
        h = self.output_norm(h)
        h = h.mean(dim=1)
        return self.output_proj(h)


################################################################################
# Aggregators
################################################################################

class ConcatAggregator(nn.Module):
    """ Concatenates multiple feature vectors and projects to common dimension. """
    def __init__(
            self, 
            input_dim: int, 
            n_features: int,
            output_dim: int, 
            dropout: float = 0.1, 
            activation: Union[str, nn.Module] = "gelu"
        ):
        super().__init__()
        total_dim = input_dim * n_features

        if isinstance(activation, str):
            activation_map = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "swish": nn.SiLU(),
                "mish": nn.Mish()
            }
            self.activation = activation_map.get(activation.lower(), nn.GELU())
        else:
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
            features: List of tensors, each of shape (batch_size, dim_i)
        
        Returns:
            Aggregated tensor of shape (batch_size, output_dim)
        """
        concatenated = torch.cat(features, dim=-1)
        aggregated = self.projection(concatenated)
        return aggregated


class AttentiveAggregator(nn.Module):
    """ Aggregates features using attention mechanism. """
    def __init__(
            self, 
            input_dim: int, 
            n_features: int,
            output_dim: int, 
            dropout: float = 0.1, 
            activation: Union[str, nn.Module] = "gelu"):
        super().__init__()
        
        if isinstance(activation, str):
            activation_map = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "swish": nn.SiLU(),
                "mish": nn.Mish()
            }
            self.activation = activation_map.get(activation.lower(), nn.GELU())
        else:
            self.activation = activation

        self.norm = nn.LayerNorm(input_dim)
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
    """ Cross-attention fusion of feature vector pairs. """
    def __init__(
            self, 
            input_dim: int,
            hidden_dim: int,
            n_layers: int,
            n_heads: int = 4,
            expansion_factor: int = 4,
            dropout: float = 0.1, 
            bias: bool = True,
            activation: Union[str, nn.Module] = "gelu"):
        super().__init__()

        if isinstance(activation, str):
            activation_map = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "swish": nn.SiLU(),
                "mish": nn.Mish()
            }
            self.activation = activation_map.get(activation.lower(), nn.GELU())
        else:
            self.activation = activation

        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projections to map from 1 to hidden_dim (following TransformerEncoder pattern)
        self.input_proj_a = nn.Linear(1, hidden_dim, bias=bias)
        self.input_proj_b = nn.Linear(1, hidden_dim, bias=bias)

        # Cross-attention layers for a->b and b->a
        self.cross_attn_a_to_b = nn.ModuleList([
            CrossAttention(hidden_dim, n_heads, dropout, bias) 
            for _ in range(n_layers)
        ])
        self.cross_attn_b_to_a = nn.ModuleList([
            CrossAttention(hidden_dim, n_heads, dropout, bias) 
            for _ in range(n_layers)
        ])

        # Self-attention layers for a and b
        self.self_attn_a = nn.ModuleList([
            SelfAttention(hidden_dim, n_heads, dropout, bias) 
            for _ in range(n_layers)
        ])
        self.self_attn_b = nn.ModuleList([
            SelfAttention(hidden_dim, n_heads, dropout, bias) 
            for _ in range(n_layers)
        ])

        # MLP layers for a and b
        self.mlp_a = nn.ModuleList([
            MLP(hidden_dim, expansion_factor, dropout, bias, self.activation) 
            for _ in range(n_layers)
        ])
        self.mlp_b = nn.ModuleList([
            MLP(hidden_dim, expansion_factor, dropout, bias, self.activation) 
            for _ in range(n_layers)
        ])

        # Layer normalization for residual connections
        self.norm_cross_a = nn.ModuleList([
            nn.LayerNorm(hidden_dim, bias=bias) for _ in range(n_layers)
        ])
        self.norm_cross_b = nn.ModuleList([
            nn.LayerNorm(hidden_dim, bias=bias) for _ in range(n_layers)
        ])
        self.norm_self_a = nn.ModuleList([
            nn.LayerNorm(hidden_dim, bias=bias) for _ in range(n_layers)
        ])
        self.norm_self_b = nn.ModuleList([
            nn.LayerNorm(hidden_dim, bias=bias) for _ in range(n_layers)
        ])
        self.norm_mlp_a = nn.ModuleList([
            nn.LayerNorm(hidden_dim, bias=bias) for _ in range(n_layers)
        ])
        self.norm_mlp_b = nn.ModuleList([
            nn.LayerNorm(hidden_dim, bias=bias) for _ in range(n_layers)
        ])

        # Final aggregation using ConcatAggregator
        self.aggregator = ConcatAggregator(hidden_dim, 2, input_dim, dropout, activation)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: First input tensor of shape (batch_size, input_dim)
            b: Second input tensor of shape (batch_size, input_dim)

        Returns:
            Fused tensor of shape (batch_size, output_dim)
        """
        # Add sequence dimension, then project to hidden dimension
        a = a.unsqueeze(-1)       # (B, input_dim, 1)
        b = b.unsqueeze(-1)       # (B, input_dim, 1)
        a = self.input_proj_a(a)  # (B, input_dim, 1) -> (B, input_dim, hidden_dim)
        b = self.input_proj_b(b)  # (B, input_dim, 1) -> (B, input_dim, hidden_dim)

        for i in range(self.n_layers):
            # Cross-attention: update a with info from b, and b with info from a
            a_cross = a + self.cross_attn_a_to_b[i](self.norm_cross_a[i](a), b)
            b_cross = b + self.cross_attn_b_to_a[i](self.norm_cross_b[i](b), a)

            # Self-attention: refine each tensor individually
            a_self = a_cross + self.self_attn_a[i](self.norm_self_a[i](a_cross))
            b_self = b_cross + self.self_attn_b[i](self.norm_self_b[i](b_cross))

            # MLP blocks
            a = a_self + self.mlp_a[i](self.norm_mlp_a[i](a_self))
            b = b_self + self.mlp_b[i](self.norm_mlp_b[i](b_self))

        # Aggregate across sequence dimension and project back to input_dim
        a = a.mean(dim=1)  # (B, hidden_dim)
        b = b.mean(dim=1)  # (B, hidden_dim)

        # Aggregate and project to input dimension
        return self.aggregator([a, b])