import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

from mb_vae_dti.training.utils.config_manager import get_config_summary
from omegaconf import DictConfig
import logging
from pytorch_lightning import Trainer
from mb_vae_dti.training.utils.callbacks import BestMetricsCallback, TimingCallback

logger = logging.getLogger(__name__)


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_results(
    config: DictConfig,
    trainer: Trainer,
) -> Dict[str, Any]:
    """
    Collect best results for downstream analysis.
    
    Args:
        config: Configuration object containing experiment details
        trainer: PyTorch Lightning trainer with completed training
        
    Returns: Dict of flat dicts of all results
    """
    experiment_name = config.logging.experiment_name
    
    best_metrics_callback = None
    timing_callback = None
    
    for callback in trainer.callbacks:
        if isinstance(callback, BestMetricsCallback):
            best_metrics_callback = callback
        elif isinstance(callback, TimingCallback):
            timing_callback = callback
    
    # Get best results
    best_results = best_metrics_callback.get_best_results()
    best_val_loss = best_results["best_val_loss"]
    best_epoch = best_results["best_epoch"]
    val_metrics = best_results["val_metrics"]
    test_metrics = best_results["test_metrics"]
    
    # Get timing statistics
    timing_stats = timing_callback.get_timing_stats()

    # Get model information
    trainable_params = count_trainable_parameters(trainer.model)
    
    # Get metadata (flat config dict)
    config_summary = get_config_summary(config)
    
    # Create results dictionary
    results = {
        "experiment_name": experiment_name,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "timing": timing_stats,
        "trainable_params": trainable_params,
        "config": config_summary,
    }
    
    logger.info(f"Collected results for {experiment_name}: "
                f"best_val_loss={best_val_loss:.6f} at epoch {best_epoch}")
    
    logger.info(
        f"Training took {timing_stats.get('total_training_time', 0):.2f}s "
        f"({timing_stats.get('avg_time_per_epoch', 0):.2f}s per epoch over {timing_stats.get('total_epochs', 0)} epochs)")
    
    return results


def save_results(
    results: Dict[str, Any],
    save_dir: Path,
) -> None:
    """
    Save results to a JSON file for later analysis.
    
    Args:
        results: Results dictionary from collect_results
        save_dir: Directory to save results
    """
    # Create results directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename based on experiment name
    experiment_name = results["experiment_name"]
    filename = f"{experiment_name}_results.json"
    
    # Save results
    results_file = save_dir / filename
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {results_file}")