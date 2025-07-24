"""
Data containers for DTI model architectures.

These containers provide structured data management for different types of information
flowing through DTI models, from simple baseline to complex full diffusion models.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal, Union, Any
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops

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
    y: Optional[torch.Tensor] = None  # Global features [batch, global_features] (will be drug_embedding)
    node_mask: Optional[torch.Tensor] = None  # [batch, max_nodes]
    
    # Noisy/discretized versions (after forward diffusion)
    X_t: Optional[torch.Tensor] = None  # Noisy node features 
    E_t: Optional[torch.Tensor] = None  # Noisy edge features
    y_t: Optional[torch.Tensor] = None  # Noisy global features (not used)
    
    # Augmented versions of G_t with extra graph-structural, timestep and molecular features
    X_augmented: Optional[torch.Tensor] = None  # Extra node features for transformer input
    E_augmented: Optional[torch.Tensor] = None  # Technically equal to E_t, since no extra features are added on edges
    y_augmented: Optional[torch.Tensor] = None  # Extra global features (timestep info, etc.)
    
    # Predicted node/edge/global feature probabilities (from GraphTransformer diffusion decoder)
    X_hat: Optional[torch.Tensor] = None  # Predicted clean node features (DiffMS doesn't use this in loss, we do)
    E_hat: Optional[torch.Tensor] = None  # Predicted clean edge features
    y_hat: Optional[torch.Tensor] = None  # Predicted clean global features (not used in loss)
    
    # Noise parameters and metadata
    noise_params: Optional[Dict[str, torch.Tensor]] = None
    timestep: Optional[torch.Tensor] = None
    
    def to_placeholder(self, kind: Literal["clean", "noisy", "augmented", "predicted"] = "clean") -> PlaceHolder:
        """
        Convert to PlaceHolder for compatibility with existing diffusion code.
        
        Args:
            tag: "clean", "noisy", "augmented", "predicted"
        """
        if kind == "clean":
            return PlaceHolder(X=self.X, E=self.E, y=self.y)
        elif kind == "noisy":
            return PlaceHolder(X=self.X_t, E=self.E_t, y=self.y_t)
        elif kind == "augmented":
            return PlaceHolder(X=self.X_augmented, E=self.E_augmented, y=self.y_augmented)
        elif kind == "predicted":
            return PlaceHolder(X=self.X_hat, E=self.E_hat, y=self.y_hat)
        else:
            raise ValueError(f"Invalid kind: {kind}")
    
    def from_placeholder(self, placeholder: PlaceHolder, kind: Literal["clean", "noisy", "augmented", "predicted"] = "clean") -> 'GraphData':
        """Update this GraphData from a PlaceHolder object."""
        if kind == "clean":
            self.X = placeholder.X
            self.E = placeholder.E
            self.y = placeholder.y
        elif kind == "noisy":
            self.X_t = placeholder.X
            self.E_t = placeholder.E
            self.y_t = placeholder.y
        elif kind == "augmented":
            self.X_augmented = placeholder.X
            self.E_augmented = placeholder.E
            self.y_augmented = placeholder.y
        elif kind == "predicted":
            self.X_hat = placeholder.X
            self.E_hat = placeholder.E
            self.y_hat = placeholder.y
        else:
            raise ValueError(f"Invalid kind: {kind}")
        return self
    
    def mask(self, node_mask: Optional[torch.Tensor] = None, collapse: bool = False) -> 'GraphData':
        """Apply masking similar to PlaceHolder.mask()."""
        if node_mask is None:
            node_mask = self.node_mask
        if node_mask is None:
            return self
            
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        
        if collapse:
            # Convert to discrete indices
            if self.X is not None:
                self.X = torch.argmax(self.X, dim=-1)
                self.X[node_mask == 0] = -1
            if self.E is not None:
                self.E = torch.argmax(self.E, dim=-1)
                self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            # Apply continuous masking
            if self.X is not None:
                self.X = self.X * x_mask
            if self.E is not None:
                self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
                
        return self
    
    def type_as(self, tensor: torch.Tensor) -> 'GraphData':
        """Move all tensors to same device/dtype as reference tensor."""
        for attr_name in [
            'X', 'E', 'y', 
            'X_t', 'E_t', 'y_t',
            'X_augmented', 'E_augmented', 'y_augmented', 
            'X_hat', 'E_hat', 'y_hat', 
            'node_mask', 'timestep', 'noise_params'
        ]:
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                setattr(self, attr_name, attr_value.type_as(tensor))
        
        # Handle noise_params dict
        if self.noise_params is not None:
            self.noise_params = {
                k: v.type_as(tensor) if isinstance(v, torch.Tensor) else v 
                for k, v in self.noise_params.items()
            }
        return self


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
    # DTI predictions - flexible to handle different phases
    dti_scores: Optional[Dict[
        Literal["Y", "Y_pKd", "Y_pKi", "Y_KIBA"], torch.Tensor
    ]] = None
    
    # Single score prediction (for baseline/single-output models)
    score_pred: Optional[torch.Tensor] = None
    
    # Graph predictions (for diffusion models)
    graph_reconstruction: Optional['PlaceHolder'] = None
    
    def get_dti_score(self, kind: Literal["Y", "Y_pKd", "Y_pKi", "Y_KIBA"]) -> Optional[torch.Tensor]:
        """Get specific DTI score prediction."""
        if self.dti_scores is not None:
            return self.dti_scores.get(kind)
        elif kind in ["Y_pKd", "Y_pKi", "Y_KIBA"] and self.score_pred is not None:
            return self.score_pred  # For baseline model
        return None
    
    def type_as(self, tensor: torch.Tensor) -> 'PredictionData':
        """Move all tensors to same device/dtype as reference tensor."""
        if self.dti_scores is not None:
            self.dti_scores = {k: v.type_as(tensor) for k, v in self.dti_scores.items()}
        if self.score_pred is not None:
            self.score_pred = self.score_pred.type_as(tensor)
        if self.graph_reconstruction is not None:
            self.graph_reconstruction = self.graph_reconstruction.type_as(tensor)
        return self


@dataclass
class LossData:
    """Container for different loss components with phase-aware computation."""
    # Core loss components
    accuracy: Optional[torch.Tensor] = None       # DTI prediction loss
    complexity: Optional[torch.Tensor] = None     # KL divergence (VAE)
    contrastive: Optional[torch.Tensor] = None    # InfoNCE loss
    reconstruction: Optional[torch.Tensor] = None # Graph reconstruction loss
    
    # Validation-specific loss (NLL)
    # TODO: this one is very tricky! We may still need to revise this
    # nll: Optional[torch.Tensor] = None
    
    # Individual components for detailed logging
    components: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # TODO: I feel iffy about this code below... perhaps we shouldn't be storing
    # the NLL loss here, since it serves more as a metric than a loss.
    # Maybe even a distinct NLLData class? We will have to dig deeper into the compute_val_loss
    def total_loss(self, weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Compute weighted total loss with phase-aware logic.
        
        Args:
            weights: [accuracy, complexity, contrastive, reconstruction] weights
            phase: "train" uses weighted sum, "val"/"test" may use NLL instead
        """
        # if phase in ["val", "test"] and self.nll is not None:
        #     # For validation/test, prioritize NLL if available
        #     return self.nll
            
        # For training or when NLL not available, use weighted component sum
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]  # Default equal weights
            
        total = torch.tensor(0.0, device=self._get_device())
        loss_components = [self.accuracy, self.complexity, self.contrastive, self.reconstruction]
        
        for weight, loss_component in zip(weights, loss_components):
            if loss_component is not None:
                total = total + weight * loss_component
                
        return total
    
    def _get_device(self) -> torch.device:
        """Get device from first available tensor."""
        for loss_component in [self.accuracy, self.complexity, self.contrastive, 
                              self.reconstruction, self.nll]:
            if loss_component is not None:
                return loss_component.device
        return torch.device('cpu')
    
    # NOTE: how is this to be used with logger_fn?
    def log_components(self, logger_fn, prefix: str = ""):
        """Log all loss components with given prefix."""
        if self.accuracy is not None:
            logger_fn(f"{prefix}accuracy_loss", self.accuracy)
        if self.complexity is not None:
            logger_fn(f"{prefix}complexity_loss", self.complexity)
        if self.contrastive is not None:
            logger_fn(f"{prefix}contrastive_loss", self.contrastive)
        if self.reconstruction is not None:
            logger_fn(f"{prefix}reconstruction_loss", self.reconstruction)
        if self.nll is not None:
            logger_fn(f"{prefix}nll", self.nll)
        
        # Log additional components
        for name, value in self.components.items():
            logger_fn(f"{prefix}{name}", value)


@dataclass
class BatchData:
    """
    Container for processed batch data with structured access.
    
    Provides unified interface for extracting features, targets, graphs, etc.
    regardless of the specific batch format (DTI vs pretrain).
    """
    # Raw batch data
    raw_batch: Dict[str, Any]
    
    # Processed components
    drug_features: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None   # single or multiple features
    target_features: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    graph_data: Optional[GraphData] = None
    
    # Targets and masks
    dti_targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None # single or multiple targets
    dti_masks: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None # single or multiple masks
    
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