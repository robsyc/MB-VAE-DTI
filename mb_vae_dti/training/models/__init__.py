"""
Neural network architectures for the DTITree model.

This module contains the core components for building multi-branch drug-target interaction models:
- Encoders: Transform feature vectors into embeddings
- Aggregators: Combine multiple embeddings 
- Decoders: Generate molecular structures from embeddings
- Graph models: Process molecular graphs
"""

from .blocks import (
    ResidualEncoder, TransformerEncoder, 
    ConcatAggregator, AttentiveAggregator,
    CrossAttentionFusion
)
from .graph_transformer import GraphTransformer

__all__ = [
    "ResidualEncoder",
    "TransformerEncoder", 
    "ConcatAggregator",
    "AttentiveAggregator",
    "CrossAttentionFusion",
    "GraphTransformer",
] 