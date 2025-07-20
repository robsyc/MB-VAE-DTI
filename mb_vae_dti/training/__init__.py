"""
Loading module for MB-DiffVAE-DTI - DTITree model.

Provides PyTorch Lightning components for training DTI models with:
- Multi-modal drug and target encoders
- Discrete graph diffusion-based conditional drug generation
- Various loss functions (KL, contrastive, reconstruction, accuracy)
- Flexible training procedures (pre-training, general DTI training, fine-tuning)
- Support for multiple model configurations (baseline to full complex model)

Components:
- models: Core neural network architectures
- metrics: Training and evaluation metrics
- diffusion: Diffusion model utilities
- datasets: Lightning data modules for h5torch datasets
- modules: Lightning modules for training DTI models
"""

# TODO: this file is very incomplete

# Core model components
from .models import (
    ResidualEncoder, TransformerEncoder,
    ConcatAggregator, AttentiveAggregator,
    CrossAttentionFusion
)
# from .models.decoders import (
#     DiscreteDiffusionDecoder,
# )
# from .models.graph_transformer import GraphTransformer

# Diffusion utilities
# from .diffusion.discrete_noise import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
# from .diffusion.utils import ...
# from .diffusion.augmentation import Augmentation

# Lightning modules
from .modules.baseline import BaselineDTIModel
from .datasets.datamodules import DTIDataModule

# TODO: Loss modules
# TODO: Metrics
# from .metrics


# Export key components for easy access
__all__ = [
    # Model architectures
    "ResidualEncoder",
    "TransformerEncoder", 
    "ConcatAggregator",
    "AttentiveAggregator",
    "CrossAttentionFusion",
    # "DiscreteDiffusionDecoder",
    # "GraphTransformer",
    
    # Diffusion components
    # "PredefinedNoiseScheduleDiscrete",
    # "MarginalUniformTransition",
    # "Augmentation",
    
    # Lightning components
    "BaselineDTIModel",
    "DTIDataModule",
]