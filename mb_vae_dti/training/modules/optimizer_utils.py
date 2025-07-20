"""
Utility functions for optimizer and scheduler configuration.
"""

import torch
from typing import Dict, Optional, Literal, Any, List, Union


def configure_optimizer_and_scheduler(
    model_parameters,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-12,
    scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]] = "const",
    trainer = None,
    max_epochs: int = 100
) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
    """
    Configure optimizer and scheduler with solid default parameters.
    
    Args:
        model_parameters: Model parameters to optimize
        learning_rate: Base learning rate
        weight_decay: Weight decay for regularization
        scheduler: Type of scheduler to use
        trainer: PyTorch Lightning trainer (needed for OneCycleLR)
        max_epochs: Maximum number of epochs (fallback for cosine scheduler)
        
    Returns:
        Optimizer or dictionary with optimizer and scheduler configuration
    """
    # Create optimizer with proven defaults
    optimizer = torch.optim.AdamW(
        model_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        amsgrad=True  # More stable convergence
    )
    
    # Return just optimizer for constant scheduler
    if scheduler == "const" or scheduler is None:
        return optimizer
    
    # Configure scheduler with solid defaults
    if scheduler == "step":
        # Step every 30 epochs, reduce by factor of 10
        # Good for longer training runs (100+ epochs)
        scheduler_obj = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
        
    elif scheduler == "one_cycle":
        # OneCycleLR with 30% warmup period
        # Excellent for faster convergence
        if trainer is None:
            raise ValueError("Trainer must be provided for OneCycleLR scheduler")
        
        stepping_batches = trainer.estimated_stepping_batches
        scheduler_obj = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(learning_rate),
            total_steps=stepping_batches,
            pct_start=0.3,  # 30% of cycle for warmup
            anneal_strategy='cos'  # Cosine annealing
        )
        
        # OneCycleLR needs step-wise scheduling
        return [optimizer], [{
            'scheduler': scheduler_obj,
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }]
        
    elif scheduler == "cosine":
        # Cosine annealing with small minimum LR
        # Good for fine-tuning and avoiding complete stagnation
        T_max = trainer.max_epochs if trainer else max_epochs
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-6  # Small minimum to avoid complete stagnation
        )
        
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")
    
    # Return optimizer and scheduler for epoch-based schedulers
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler_obj,
            "monitor": "val/loss",  # Monitor validation loss
            "interval": "epoch",
            "frequency": 1,
        }
    } 