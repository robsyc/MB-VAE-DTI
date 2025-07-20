import json
import time
from pathlib import Path
from typing import Dict, Any

from mb_vae_dti.training.utils.config_manager import get_config_summary
from omegaconf import DictConfig
import logging
from pytorch_lightning import Trainer
from mb_vae_dti.training.utils.callbacks import BestValidationMetricsCallback, TimingCallback

logger = logging.getLogger(__name__)


def count_trainable_parameters(model) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model or PyTorch Lightning module
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_validation_results(
    config: DictConfig,
    trainer: Trainer,
) -> Dict[str, Any]:
    """
    Collect best validation results for downstream analysis of gridsearch.
    
    Args:
        config: Configuration object containing experiment details
        trainer: PyTorch Lightning trainer with completed training
        
    Returns:
        Dictionary containing:
            - experiment_name: Name of the experiment
            - best_val_loss: Best validation loss achieved
            - best_epoch: Epoch where best validation loss occurred
            - metadata: Flat config summary for analysis
            - metrics: All validation metrics from best epoch
            - timing: Training timing statistics
            - model_info: Model information (trainable parameters)
            - timestamp: Current timestamp
    """
    experiment_name = config.logging.experiment_name
    
    # Find the BestValidationMetricsCallback in the trainer's callbacks
    best_metrics_callback = None
    timing_callback = None
    
    for callback in trainer.callbacks:
        if isinstance(callback, BestValidationMetricsCallback):
            best_metrics_callback = callback
        elif isinstance(callback, TimingCallback):
            timing_callback = callback
    
    if best_metrics_callback is None:
        logger.error("BestValidationMetricsCallback not found in trainer callbacks")
        raise ValueError("BestValidationMetricsCallback not found - ensure gridsearch mode is enabled")
    
    # Get best results from callback
    best_results = best_metrics_callback.get_best_results()
    
    best_val_loss = best_results["best_val_loss"]
    best_epoch = best_results["best_epoch"]
    best_metrics = best_results["best_metrics"]
    
    # Get timing statistics if available
    timing_stats = {}
    if timing_callback is not None:
        timing_stats = timing_callback.get_timing_stats()
        logger.debug(f"Collected timing stats: {timing_stats}")
    else:
        logger.warning("TimingCallback not found - timing statistics will be unavailable")
    
    # Get model information
    trainable_params = None
    if trainer.model is not None:
        trainable_params = count_trainable_parameters(trainer.model)
        logger.debug(f"Model has {trainable_params:,} trainable parameters")
    else:
        logger.warning("Model not available - model information will be unavailable")
    
    # Get metadata (flat config dict)
    metadata = get_config_summary(config)
    
    # Create results dictionary
    results = {
        "experiment_name": experiment_name,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "metadata": metadata,
        "metrics": best_metrics,
        "timing": timing_stats,
        "trainable_params": trainable_params,
        "timestamp": time.time()
    }
    
    logger.info(f"Collected results for {experiment_name}: "
                f"best_val_loss={best_val_loss:.6f} at epoch {best_epoch}")
    
    if timing_stats:
        logger.info(f"Training took {timing_stats.get('total_training_time', 0):.2f}s "
                   f"({timing_stats.get('avg_time_per_epoch', 0):.2f}s per epoch)")
    
    return results


def save_gridsearch_results(
    results: Dict[str, Any],
    save_dir: Path,
) -> None:
    """
    Save gridsearch results to a JSON file for later analysis.
    
    Args:
        results: Results dictionary from collect_validation_results
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
    
    logger.info(f"Saved gridsearch results to {results_file}")