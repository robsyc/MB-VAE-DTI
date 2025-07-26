"""
Data containers for DTI model architectures.

These containers provide structured data management for different types of information
flowing through DTI models, from simple baseline to complex full diffusion models.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal, Union, Any
import torch
from torch_geometric.data import Data, Batch


class PlaceHolder:
    def __init__(self, X, E, y = None):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        try:
            self.y = self.y.type_as(x)
        except:
            pass
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

@dataclass
class GraphData:
    """
    Container for graph data at different processing stages.
    
    Handles the complete pipeline: PyG (from dataloader) → Dense → Noisy → Augmented
    Integrates PlaceHolder functionality for compatibility with existing diffusion code.
    """
    # PyG data from dataloader (batch["G"])
    pyg_batch: Optional[Batch] = None
    
    # Dense representations
    X: Optional[torch.Tensor] = None  # Node features [batch, max_nodes, node_features] 
    E: Optional[torch.Tensor] = None  # Edge features [batch, max_nodes, max_nodes, edge_features]
    # y: Optional[torch.Tensor] = None  # Global features [batch, global_features] (will be drug_embedding)
    node_mask: Optional[torch.Tensor] = None  # [batch, max_nodes]
    
    # Noisy/discretized versions (after forward diffusion)
    X_t: Optional[torch.Tensor] = None  # Noisy node features 
    E_t: Optional[torch.Tensor] = None  # Noisy edge features
    # y_t: Optional[torch.Tensor] = None  # Noisy global features (not used)
    
    # Augmented versions of G_t with extra graph-structural, timestep and molecular features
    X_extra: Optional[torch.Tensor] = None  # Extra node features for transformer input (graph structural & molecular)
    # E_augmented: Optional[torch.Tensor] = None  # Extra edge features (not used)
    y_extra: Optional[torch.Tensor] = None  # Extra global features (timestep info, etc.)
    
    # Noise parameters and metadata
    noise_params: Optional[Dict[
        Literal["t_int", "t", "beta_t", "alpha_s_bar", "alpha_t_bar"], 
        torch.Tensor]] = None
    

@dataclass  
class EmbeddingData:
    """Container for embeddings and intermediate representations."""
    # Core embeddings
    drug_embedding: Optional[torch.Tensor] = None
    target_embedding: Optional[torch.Tensor] = None
    
    # Variational components (for VAE)
    drug_mu: Optional[torch.Tensor] = None
    drug_logvar: Optional[torch.Tensor] = None
    
    # Attention weights (for attentive aggregators)
    drug_attention: Optional[torch.Tensor] = None
    target_attention: Optional[torch.Tensor] = None
    
    # Fused representations (for DTI prediction)
    fused_embedding: Optional[torch.Tensor] = None
    
    def type_as(self, tensor: torch.Tensor) -> 'EmbeddingData':
        """Move all tensors to same device/dtype as reference tensor."""
        for attr_name in [
            'drug_embedding', 'target_embedding', 
            'drug_mu', 'drug_logvar',
            'drug_attention', 'target_attention', 
            'fused_embedding'
        ]:
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                setattr(self, attr_name, attr_value.type_as(tensor))
        return self


@dataclass
class PredictionData:
    """Container for model predictions."""
    # DTI score predictions (single or multi-score)
    dti_scores: Optional[Union[
        Dict[Literal["Y", "Y_pKd", "Y_pKi", "Y_KIBA"], torch.Tensor],
        torch.Tensor
    ]] = None
    
    # Graph predictions (for diffusion models)
    graph_reconstruction: Optional['PlaceHolder'] = None
    
    def type_as(self, tensor: torch.Tensor) -> 'PredictionData':
        """Move all tensors to same device/dtype as reference tensor."""
        if self.dti_scores is not None:
            self.dti_scores = {k: v.type_as(tensor) for k, v in self.dti_scores.items()}
        if self.graph_reconstruction is not None:
            self.graph_reconstruction = self.graph_reconstruction.type_as(tensor)
        return self


@dataclass
class LossData:
    """Container for different loss components with phase-aware computation."""
    # Core loss components
    accuracy: Optional[torch.Tensor] = None       # DTI prediction loss (or NLL for val/test)
    complexity: Optional[torch.Tensor] = None     # KL divergence (VAE)
    contrastive: Optional[torch.Tensor] = None    # InfoNCE loss
    reconstruction: Optional[torch.Tensor] = None # Graph reconstruction loss / NLL
    
    # Individual components for detailed logging
    components: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    def _get_device(self) -> torch.device:
        """Get device from first available tensor."""
        for loss_component in [self.accuracy, self.complexity, self.contrastive, self.reconstruction]:
            if loss_component is not None:
                return loss_component.device
        return torch.device('cpu')
    
    def compute_loss(self, weights: List[float]) -> torch.Tensor:
        assert len(weights) == 4, "Weights must be a list of 4 elements (accuracy, complexity, contrastive, reconstruction)"
        
        device = self._get_device()
        total_loss = torch.tensor(0.0, device=device)
        
        loss_components = [self.accuracy, self.complexity, self.contrastive, self.reconstruction]
        
        for weight, loss_component in zip(weights, loss_components):
            if weight != 0 and loss_component is not None:
                total_loss = total_loss + weight * loss_component
        
        return total_loss


@dataclass
class BatchData:
    """
    Container for processed batch data with structured access.
    
    Provides unified interface for extracting features, targets, graphs, etc.
    regardless of the specific batch format (DTI vs pretrain).
    """
    # Raw batch data
    raw_batch: Optional[Dict[str, Any]] = None
    
    # Extracted input features (single or multi-modal)
    drug_features: Optional[Union[
        torch.Tensor, 
        List[torch.Tensor]
    ]] = None
    target_features: Optional[Union[
        torch.Tensor, 
        List[torch.Tensor]
    ]] = None

    graph_data: Optional[GraphData] = None # graph data for full diffusion module
    
    # Targets and masks (single or multi-score)
    dti_targets: Optional[Union[
        torch.Tensor, 
        Dict[Literal["Y", "Y_pKd", "Y_pKi", "Y_KIBA"], torch.Tensor]
    ]] = None
    dti_masks: Optional[Union[
        torch.Tensor, 
        Dict[Literal["Y_pKd", "Y_pKi", "Y_KIBA"], torch.Tensor]
    ]] = None
    
    # Additional data
    drug_fp: Optional[torch.Tensor] = None
    target_fp: Optional[torch.Tensor] = None
    smiles: Optional[List[str]] = None
    
    def type_as(self, tensor: torch.Tensor) -> 'BatchData':
        """Move all tensors to same device/dtype as reference tensor."""
        if self.graph_data is not None:
            self.graph_data = self.graph_data.type_as(tensor)
        # Handle other tensor fields as needed
        return self 