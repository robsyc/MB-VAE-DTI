# DTI Model Architecture Guide

This guide documents the optimal method responsibilities and design patterns for DTI PyTorch Lightning modules, following TorchMetrics and PyTorch Lightning best practices.

## Method Responsibilities

### 1. `__init__(self, ...)`
**Purpose**: Model initialization, architecture setup, and metrics configuration

**Responsibilities**:
- Parse and validate hyperparameters  
- Initialize model components (encoders, heads, etc.)
- Set up metrics using `self.setup_metrics(phase, finetune_score)`
- Register buffers and other model state

**Key Points**:
- Use `self.setup_metrics()` instead of manual metric creation
- Let the base class handle metric instantiation based on phase

```python
def __init__(self, phase, finetune_score, ...):
    super().__init__()
    self.save_hyperparameters()
    
    # Initialize model components
    self.drug_encoder = ...
    self.target_encoder = ...
    
    # Setup metrics automatically based on phase
    self.setup_metrics(phase=phase, finetune_score=finetune_score)
```

### 2. `forward(self, inputs)`
**Purpose**: Pure model computation - the mathematical forward pass

**Responsibilities**:
- Take model inputs (features, graphs, etc.)
- Perform forward computation through all model components
- Return predictions and intermediate outputs
- NO loss computation, NO masking, NO metrics updates

**Key Points**:
- Keep this method pure and stateless
- Return structured outputs for downstream use
- Handle all model architectures consistently

```python
def forward(self, drug_features, target_features):
    """Pure forward pass returning predictions and intermediate outputs."""
    drug_embedding = self.drug_encoder(drug_features)
    target_embedding = self.target_encoder(target_features)
    score_pred = torch.sum(drug_embedding * target_embedding, dim=-1)
    
    outputs = {
        "drug_embedding": drug_embedding,
        "target_embedding": target_embedding,
        "score_pred": score_pred
    }
    return score_pred, outputs
```

### 3. `_common_step(self, batch)`
**Purpose**: Shared logic for train/val/test steps

**Responsibilities**:
- Extract features from batch using base class methods
- Extract targets and masks from batch
- Call `forward()` to get predictions
- Compute losses on valid samples
- Return structured data for metrics and logging

**Key Points**:
- Use base class methods: `_get_features_from_batch()`, `_get_target_from_batch()`
- Handle masking consistently
- Return `None` values when no valid samples exist
- Keep loss computation simple and clear

```python
def _common_step(self, batch):
    """Shared logic for all step types."""
    # Extract data using base class methods
    drug_features, target_features = self._get_features_from_batch(batch)
    dti_target, dti_mask = self._get_target_from_batch(batch, self.finetune_score)
    
    # Forward pass
    dti_pred, outputs = self.forward(drug_features, target_features)
    
    # Handle masking
    if not dti_mask.any():
        return None, None, None
        
    dti_pred_masked = dti_pred[dti_mask]
    dti_target_masked = dti_target[dti_mask]
    loss = F.mse_loss(dti_pred_masked, dti_target_masked)
    
    return dti_pred_masked, dti_target_masked, loss
```

### 4. `training_step/validation_step/test_step(self, batch, batch_idx)`
**Purpose**: Phase-specific logic and metric updates

**Responsibilities**:
- Call `_common_step()` to get predictions, targets, and losses
- Update metrics with valid predictions/targets
- Log immediate losses and other step-level values
- Return loss for optimization (training) or logging (val/test)

**Key Points**:
- Handle `None` returns from `_common_step()` (no valid samples)
- Update metrics directly if using pre-masked data
- Use `self.log()` for immediate logging with `prog_bar=True` for key metrics
- Keep step methods lightweight

```python
def training_step(self, batch, batch_idx):
    """Training step with metric updates."""
    dti_preds, dti_targets, loss = self._common_step(batch)
    
    if dti_preds is None:  # No valid samples
        return None

    # Update metrics directly (already masked in _common_step)
    self.train_metrics.update(dti_preds, dti_targets)
    
    # Log immediate loss
    self.log("train/loss", loss, prog_bar=True)
    
    return loss
```

### 5. `on_*_epoch_end(self)` Methods
**Purpose**: End-of-epoch metric computation and logging

**Key Points**:
- **INHERIT FROM BASE CLASS** - Don't override unless absolutely necessary
- The base class automatically handles:
  - `metrics.compute()` - Get final metric values
  - `self.log()` - Log all computed metrics  
  - `metrics.reset()` - Reset for next epoch
- Follow TorchMetrics best practices automatically

```python
# DON'T OVERRIDE - inherit from AbstractDTIModel
# The base class handles this optimally:

def on_train_epoch_end(self):
    if self.train_metrics is not None:
        train_metrics = self.train_metrics.compute()
        for name, value in train_metrics.items():
            self.log(name, value)
        self.train_metrics.reset()
```

## Model Complexity Patterns

### Baseline Model (Simple)
- Single encoders
- Dot-product prediction
- Direct metric updates
- Minimal complexity

### Multi-Modal Model (Medium)
- Multiple encoders + aggregation
- Attention weights and multiple features
- Still only one output (e.g. Y_pKd) w/ dot-product prediction

### Multi-Output Model (Hard)
- Multiple outputs (e.g. Y (binary) and Y_pKd, Y_pKi, Y_KIBA)
- DTI Head for predicting each score (lends itself to pretraining)
- More complex metric collection

### Hybrid Model (Hard)
- Multi-modal + Multi-output
- Multiple loss components (contrastive and accuracy)

### Full Model (Complex)
- Multi-modal + Multi-output + Diffusion
- Multiple loss components (contrastive, complexity, reconstruction, accuracy)
- Additional diffusion metrics (MLL, and molecular validity/diversity)

## TorchMetrics Best Practices

### 1. Metric Setup
```python
# ✅ GOOD: Use setup_metrics()
self.setup_metrics(phase=phase, finetune_score=finetune_score)

# ❌ BAD: Manual metric creation
self.train_metrics = RealDTIMetrics(prefix="train/")
```

### 2. Metric Updates
```python
# ✅ GOOD: Direct update with masked data
self.train_metrics.update(masked_preds, masked_targets)

# ✅ GOOD: Use base class helper for complex outputs
self._update_single_score_metrics(outputs, batch, self.train_metrics, self.finetune_score)
```

### 3. Metric Logging
```python
# ✅ GOOD: Let base class handle epoch-end logging
# (inherit on_*_epoch_end methods)

# ❌ BAD: Manual epoch-end handling
def on_train_epoch_end(self):
    metrics = self.train_metrics.compute()
    for name, value in metrics.items():
        self.log(name, value)
    self.train_metrics.reset()
```

## Scaling to Complex Models

### Multi-Modal Extensions
- Use `_get_features_from_batch()` for consistent feature extraction
- Leverage `_update_single_score_metrics()` for complex output handling
- Maintain clear separation between feature processing and metric updates

### Multi-Output Extensions  
- Use `DTIMetricsCollection` for multiple scores
- Use `_update_multi_score_metrics()` for complex metric updates
- Handle different loss combinations in `_common_step()`

### Full Model Extensions
- Combine all patterns above
- Add diffusion-specific metrics as additional attributes
- Use weighted loss combinations
- Maintain method responsibility separation

## Common Pitfalls to Avoid

1. **Mixing masked and unmasked data** in metric updates
2. **Overriding epoch-end methods** unnecessarily  
3. **Manual metric reset** when using TorchMetrics
4. **Complex logic in forward()** method
5. **Inconsistent feature extraction** across models
6. **Logging metric objects and computed values** simultaneously

## Migration Guide

To migrate existing models to this pattern:

1. **Remove manual metric creation** → Use `self.setup_metrics()`
2. **Remove custom epoch-end methods** → Inherit from base class
3. **Simplify forward method** → Only model computation
4. **Extract common logic** → Move to `_common_step()`
5. **Use base class utilities** → `_get_features_from_batch()`, etc.
6. **Direct metric updates** → Use appropriate update method

### Example: MultiModalDTIModel Migration

**Before (current pattern):**
```python
class MultiModalDTIModel(AbstractDTIModel):
    def __init__(self, ...):
        # ❌ Manual metric creation
        self.train_metrics = RealDTIMetrics(prefix="train/")
        self.val_metrics = RealDTIMetrics(prefix="val/")
        self.test_metrics = RealDTIMetrics(prefix="test/")

    def training_step(self, batch, batch_idx):
        # ❌ Inline feature extraction and complex logic
        drug_feats = [batch["drug"]["features"][feat_name] for feat_name in self.hparams.drug_features]
        # ... complex inline logic
        self.train_metrics.update(valid_predictions, valid_targets)
        
    # ❌ Custom epoch end methods
    def on_train_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        for name, value in train_metrics.items():
            self.log(name, value)
        self.train_metrics.reset()
```

**After (new pattern):**
```python
class MultiModalDTIModel(AbstractDTIModel):
    def __init__(self, finetune_score, ...):
        super().__init__()
        # ... initialize encoders and aggregators ...
        
        # ✅ Use base class metrics setup
        self.setup_metrics(phase="finetune", finetune_score=finetune_score)

    def forward(self, drug_features: List[torch.Tensor], target_features: List[torch.Tensor]):
        """✅ Pure forward computation only."""
        # Encode and aggregate features
        drug_embedding = self.drug_aggregator([
            encoder(feat) for encoder, feat in zip(self.drug_encoders.values(), drug_features)
        ])
        target_embedding = self.target_aggregator([
            encoder(feat) for encoder, feat in zip(self.target_encoders.values(), target_features)  
        ])
        
        score_pred = torch.sum(drug_embedding * target_embedding, dim=-1)
        
        outputs = {
            "drug_embedding": drug_embedding,
            "target_embedding": target_embedding,
            "score_pred": score_pred
        }
        return score_pred, outputs

    def _common_step(self, batch):
        """✅ Shared logic with base class utilities."""
        # Use base class feature extraction
        drug_features, target_features = self._get_features_from_batch(batch)
        dti_target, dti_mask = self._get_target_from_batch(batch, self.hparams.finetune_score)
        
        # Forward pass
        score_pred, outputs = self.forward(drug_features, target_features)
        
        # Handle masking
        if not dti_mask.any():
            return None, None, None
            
        score_pred_masked = score_pred[dti_mask]
        dti_target_masked = dti_target[dti_mask]
        loss = F.mse_loss(score_pred_masked, dti_target_masked)
        
        return score_pred_masked, dti_target_masked, loss

    def training_step(self, batch, batch_idx):
        """✅ Simple step logic."""
        preds, targets, loss = self._common_step(batch)
        
        if preds is None:
            return None

        # Direct metric update with masked data
        self.train_metrics.update(preds, targets)
        self.log("train/loss", loss, prog_bar=True)
        
        return loss

    # ✅ Inherit epoch_end methods from AbstractDTIModel
    # No need to override on_train_epoch_end, on_validation_epoch_end, on_test_epoch_end
```

This architecture provides a clean, scalable foundation that works from simple baseline models to complex multi-modal, multi-output, diffusion-based architectures while following PyTorch Lightning and TorchMetrics best practices. 