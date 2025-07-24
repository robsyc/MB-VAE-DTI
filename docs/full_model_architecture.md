# Full Model Architecture Design

## Problem: Managing Complex Data Flows

The Full DTI Model with diffusion presents unique challenges:
- Multiple data types (features, graphs, embeddings, predictions)
- Complex preprocessing pipeline (SMILES -(dataloader)-> PyG → Dense → Noisy → Augmented)
- Multiple forward passes (encoders + diffusion decoder)
- Multiple loss components with different computation methods
- Different train vs val/test loss computation

## Solution: Structured Data Containers + Processing Pipeline

### 1. Data Container Abstractions

Extend the `PlaceHolder` concept with structured containers for different data types:

```python
from dataclasses import dataclass
from typing import Optional, Dict, List
import torch

@dataclass
class GraphData:
    """Container for graph data at different processing stages."""
    # Raw data
    smiles: Optional[List[str]] = None
    pyg_batch: Optional[Any] = None
    
    # Dense representations
    X: Optional[torch.Tensor] = None  # Node features
    E: Optional[torch.Tensor] = None  # Edge features  
    y: Optional[torch.Tensor] = None  # Global features
    node_mask: Optional[torch.Tensor] = None
    
    # Noisy/discretized/augmented versions
    X_noisy: Optional[torch.Tensor] = None
    E_noisy: Optional[torch.Tensor] = None
    # NOTE: I added these bcs we may need them for other methods in diffusion decoder (compute_val_loss, etc.)
    X_t: Optional[torch.Tensor] = None
    E_t: Optional[torch.Tensor] = None
    X_augmented: Optional[torch.Tensor] = None
    E_augmented: Optional[torch.Tensor] = None
    y_augmented: Optional[torch.Tensor] = None

    # NOTE: perhaps we also want X_hat, E_hat (y_hat) for the predicted node/edge states
    # and/or the sampled version of that prediction...
    
    # Noise parameters
    noise_params: Optional[Dict] = None
    
    def to_placeholder(self) -> PlaceHolder:
        """Convert to PlaceHolder for compatibility."""
        return PlaceHolder(X=self.X, E=self.E, y=self.y)
    
    def to_device(self, device):
        """Move all tensors to device."""
        # Implementation details...
        return self

@dataclass  
class EmbeddingData:
    """Container for embeddings and intermediate representations."""
    drug_embedding: Optional[torch.Tensor] = None
    target_embedding: Optional[torch.Tensor] = None
    
    # Variational components
    drug_mu: Optional[torch.Tensor] = None
    drug_logvar: Optional[torch.Tensor] = None
    
    # Attention weights
    drug_attention: Optional[torch.Tensor] = None
    target_attention: Optional[torch.Tensor] = None
    
    # Fused representations
    fused_embedding: Optional[torch.Tensor] = None

@dataclass
class PredictionData:
    """Container for model predictions."""
    # DTI predictions
    dti_scores: Optional[Dict[str, torch.Tensor]] = None  # {"pKd": tensor, "KIBA": tensor, ...}
    
    # Graph predictions (as PlaceHolder for compatibility)
    # NOTE: in light of this we may opt to store the G_hat here instead of in the GraphData class
    graph_reconstruction: Optional[PlaceHolder] = None
    
    def get_dti_score(self, score_type: Literal[
        "Y", "Y_pKd", "Y_pKi", "Y_KIBA"]) -> torch.Tensor:
        """Get specific DTI score prediction."""
        return self.dti_scores.get(score_type)

@dataclass
class LossData:
    """Container for different loss components."""
    accuracy: Optional[torch.Tensor] = None
    complexity: Optional[torch.Tensor] = None
    contrastive: Optional[torch.Tensor] = None
    reconstruction: Optional[torch.Tensor] = None

    # NOTE: I like this idea, but I'm wondering how we're going to make it click for all settings e.g. in finetune phase we only have one score to predict, ...
    
    def total_loss(self, weights: List[float]) -> torch.Tensor:
        """Compute weighted total loss."""
        losses = [self.accuracy, self.complexity, self.contrastive, self.reconstruction]
        return sum(w * loss for w, loss in zip(weights, losses) if loss is not None)
        # NOTE: very nice to have this None-awareness for the loss computation
```

### 2. Processing Pipeline Design

Organize the full model using a clear processing pipeline:

```python
class FullDTIModel(AbstractDTIModel):
    
    def _process_graphs(self, smiles: List[str]) -> GraphData:
        """Convert SMILES to processed graph data."""
        graph_data = GraphData(smiles=smiles)
        
        # SMILES -> PyG
        graph_data.pyg_batch = self.graph_converter.smiles_to_pyg_batch(smiles)

        # NOTE: see datamodules.py => the collate_fn is responsible for converting SMILES to PyG objects, a Batch object is returned in the batch["G"] key
        
        # PyG -> Dense
        dense_data, node_mask = to_dense(
            graph_data.pyg_batch.x, 
            graph_data.pyg_batch.edge_index, 
            graph_data.pyg_batch.edge_attr, 
            graph_data.pyg_batch.batch
        )
        graph_data.X = dense_data.X
        graph_data.E = dense_data.E  
        graph_data.y = dense_data.y
        graph_data.node_mask = node_mask
        
        # Apply noise
        G_t, noise_params = self.apply_noise(graph_data.to_placeholder(), node_mask)
        graph_data.X_noisy = G_t.X
        graph_data.E_noisy = G_t.E
        graph_data.noise_params = noise_params
        
        # Augment
        G_aug = self.augment_graph(G_t, noise_params, node_mask)
        graph_data.X_augmented = G_aug.X
        graph_data.E_augmented = G_aug.E  
        graph_data.y_augmented = G_aug.y
        
        return graph_data
    
    def _encode_features(self, drug_features: List[torch.Tensor], 
                        target_features: List[torch.Tensor]) -> EmbeddingData:
        """Encode drug and target features."""
        embeddings = EmbeddingData()
        
        # Drug encoding + variational
        drug_emb = self._encode_drug_features(drug_features)
        if self.attentive:
            embeddings.drug_embedding, embeddings.drug_attention = drug_emb
        else:
            embeddings.drug_embedding = drug_emb
            
        embeddings.drug_embedding, embeddings.drug_mu, embeddings.drug_logvar = \
            self.drug_kl_head(embeddings.drug_embedding)
        
        # Target encoding
        target_emb = self._encode_target_features(target_features)
        if self.attentive:
            embeddings.target_embedding, embeddings.target_attention = target_emb
        else:
            embeddings.target_embedding = target_emb
            
        # Fusion
        embeddings.fused_embedding = self.fusion(
            embeddings.drug_embedding, 
            embeddings.target_embedding
        )
        
        return embeddings
    
    def forward(self, drug_features: List[torch.Tensor], 
                target_features: List[torch.Tensor],
                graph_data: GraphData) -> Tuple[PredictionData, EmbeddingData]:
        """Pure forward pass with structured inputs/outputs."""
        
        # Encode features
        embeddings = self._encode_features(drug_features, target_features)
        
        # DTI predictions
        dti_preds = self.dti_head(embeddings.fused_embedding)
        
        # Graph reconstruction
        # Combine drug embedding with augmented graph features
        y_input = torch.cat([embeddings.drug_embedding, graph_data.y_augmented], dim=1)
        X_input = torch.cat([graph_data.X_noisy, graph_data.X_augmented], dim=2)
        E_input = torch.cat([graph_data.E_noisy, graph_data.E_augmented], dim=3)
        
        graph_pred = self.drug_decoder(X_input, E_input, y_input, graph_data.node_mask)
        
        # Structure outputs
        predictions = PredictionData(
            dti_scores=dti_preds,
            graph_reconstruction=graph_pred
        )
        
        return predictions, embeddings
    
    def _compute_losses(self, predictions: PredictionData, 
                       embeddings: EmbeddingData,
                       graph_data: GraphData,
                       batch: Dict[str, Any]) -> LossData:
        """Compute all loss components."""
        losses = LossData()
        
        # DTI accuracy loss
        dti_targets, dti_masks = self._get_targets_from_batch(batch)
        losses.accuracy = self.dti_head.loss(
            predictions=predictions.dti_scores,
            targets=dti_targets, 
            masks=dti_masks
        )
        
        # Complexity loss
        losses.complexity = self.drug_kl_head.kl_divergence(
            embeddings.drug_mu, embeddings.drug_logvar
        )
        
        # Contrastive loss
        drug_fp, target_fp = self._get_fingerprints_from_batch(batch)
        losses.contrastive = (
            self.drug_infonce_head(embeddings.drug_embedding, drug_fp) +
            self.target_infonce_head(embeddings.target_embedding, target_fp)
        )
        
        # Reconstruction loss
        losses.reconstruction = self.drug_reconstruction_head(
            pred=predictions.graph_reconstruction,
            true=graph_data.to_placeholder()
        )
        
        return losses
    
    def _common_step(self, batch: Dict[str, Any]) -> Tuple[PredictionData, EmbeddingData, GraphData, LossData]:
        """Structured common step using data containers."""
        
        # Extract features
        drug_features, target_features = self._get_features_from_batch(batch)
        smiles = self._get_smiles_from_batch(batch)
        
        # Process graphs (if needed for this phase)
        if self.phase != "pretrain_target":
            graph_data = self._process_graphs(smiles)
        else:
            graph_data = GraphData()  # Empty container
        
        # Forward pass
        predictions, embeddings = self.forward(drug_features, target_features, graph_data)
        
        # Compute losses
        losses = self._compute_losses(predictions, embeddings, graph_data, batch)
        
        return predictions, embeddings, graph_data, losses
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Clean training step using structured data."""
        predictions, embeddings, graph_data, losses = self._common_step(batch)
        
        # Compute total loss
        total_loss = losses.total_loss(self.weights)
        
        # Update metrics (using base class methods)
        if self.phase == "train":
            # Convert predictions to expected format for metrics
            outputs = {
                "binary_pred": predictions.dti_scores["binary"],
                "pKd_pred": predictions.dti_scores["pKd"], 
                "pKi_pred": predictions.dti_scores["pKi"],
                "KIBA_pred": predictions.dti_scores["KIBA"]
            }
            self._update_multi_score_metrics(outputs, batch, self.train_metrics)
        elif self.phase == "finetune":
            outputs = {"score_pred": predictions.get_dti_score(self.finetune_score)}
            self._update_single_score_metrics(outputs, batch, self.train_metrics, self.finetune_score)
        
        # Log components
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/accuracy_loss", losses.accuracy)
        self.log("train/complexity_loss", losses.complexity) 
        self.log("train/contrastive_loss", losses.contrastive)
        self.log("train/reconstruction_loss", losses.reconstruction)
        
        return total_loss
```

### 3. Benefits of This Approach

1. **Clear Separation**: Each processing stage has its own container and method
2. **Maintainable**: Easy to modify individual components without affecting others
3. **Extensible**: New data types or processing steps can be added easily
4. **Type Safety**: Clear interfaces and expected data structures
5. **Debuggable**: Easy to inspect intermediate states
6. **Compatible**: Still works with existing PlaceHolder and base class methods

### 4. Migration Strategy

1. **Phase 1**: Implement data containers
2. **Phase 2**: Refactor `_process_graphs()` method  
3. **Phase 3**: Refactor `forward()` to use structured inputs/outputs
4. **Phase 4**: Update `_common_step()` and step methods
5. **Phase 5**: Add validation and testing

## Questions for Further Design

1. Should we use dataclasses or custom classes with methods?
=> dataclasses are a good choice for us, that way we can still configure loss computation more flexibly

2. How much compatibility should we maintain with existing PlaceHolder?
=> we could completely integrate it into the GraphData class, but we need to make sure the methods are still compatible (type_as, mask). The to_placeholder method may not be enough, since the PlaceHolder class is used in various places e.g. limit_dist, etc. For our use-case it is also a bit trange to have `y` in the PlaceHolder class, since we treat the global feature (drug embedding) as much more than in the DiGress/DiffMS implementations.

3. Should validation/test steps use different loss computation (NLL vs training losses)?
=> I think it is important to maintain the nll computation implemented in the original val/test steps, though we may want to integrate it more wisely into our pipeline to be in accordance with our wider design goals. However, there is room to omit some of the DiGress/DiffMS metrics since we probably won't be using them. It remains strange to me how one loss is computed in the train step and another in the val/test step. Is overfitting such a big deal on these discrete diffusion models or what is the reason for this?

4. How should we handle the iterative sampling during evaluation? 
   - Part of the forward pass: definitly NOT, these computations are the iterative denoising from the (marginal) limit distribution node/edge types to the clean graph over all timesteps - unlike the forward pass which is a single shot denoising from the noisy graph to the clean graph. Going over all samples is a very expensive operation, so we should not do it too frequently (perhaps only at val/test epoch end). The original implementations had a counter to keep track of when/how much samples to sample
   - A seperate evaluation method: perhaps this is a good idea, that way we avoid having to implement it in both val/test, integrated into epoch end methods.

5. Migration Strategy: I advice the following steps:
   - Review baseline.py in light of this new found long-term design goal, and see if we can integrate the dataclasses further into the AbstractDTIModel and baseline module, in light of the design goals: phases (pretrain, train, finetune), metrics (single score dti, multi score dti, chemical validity), loss computation (accuracy, complexity, contrastive, reconstruction, NLL, ...)
   - Look at the new full.py (now mostly a skeleton) and see which parts need to be integrated where and how.
   - Review the old full_old.py and step-by-step migrate the code to the new design. Potentially we can also improve the `training/diffusion/` code, particularly the utils.py which is a bit messy and could be refactored, and the molecular validity and NLL-related metrics.