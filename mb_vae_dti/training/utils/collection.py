import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

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


def collect_test_results(
    config: DictConfig,
    trainer: Trainer,
    member_index: int
) -> Dict[str, Any]:
    """
    Collect test results for ensemble analysis.
    
    Args:
        config: Configuration object containing experiment details
        trainer: PyTorch Lightning trainer with completed testing
        member_index: Index of this ensemble member (0-based)
        
    Returns:
        Dictionary containing:
            - experiment_name: Name of the experiment
            - member_index: Index of ensemble member
            - test_metrics: All test metrics from final testing
            - metadata: Flat config summary for analysis
            - timing: Training timing statistics
            - model_info: Model information (trainable parameters)
            - timestamp: Current timestamp
    """
    experiment_name = config.logging.experiment_name
    
    # Get test metrics from trainer
    test_metrics = {}
    for key, value in trainer.logged_metrics.items():
        if key.startswith('test/'):
            test_metrics[key] = value.item() if hasattr(value, 'item') else value
    
    if not test_metrics:
        logger.warning("No test metrics found in trainer.logged_metrics")
        # Fallback: try to get from callback_metrics
        for key, value in trainer.callback_metrics.items():
            if key.startswith('test/'):
                test_metrics[key] = value.item() if hasattr(value, 'item') else value
    
    # Get timing statistics if available
    timing_stats = {}
    timing_callback = None
    for callback in trainer.callbacks:
        if isinstance(callback, TimingCallback):
            timing_callback = callback
            break
    
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
        "member_index": member_index,
        "test_metrics": test_metrics,
        "metadata": metadata,
        "timing": timing_stats,
        "trainable_params": trainable_params,
        "timestamp": time.time()
    }
    
    logger.info(f"Collected test results for ensemble member {member_index + 1}: "
                f"{len(test_metrics)} test metrics")
    
    if test_metrics:
        # Log a few key metrics
        for key, value in list(test_metrics.items())[:3]:  # First 3 metrics
            logger.info(f"  {key}: {value:.6f}")
    
    return results


def save_ensemble_member_results(
    results: Dict[str, Any],
    save_dir: Path,
    member_index: int,
) -> None:
    """
    Save individual ensemble member results to a JSON file.
    
    Args:
        results: Results dictionary from collect_test_results
        save_dir: Base directory to save results
        member_index: Index of ensemble member (0-based)
    """
    # Create ensemble results directory
    ensemble_dir = save_dir / "ensemble_results"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual member results
    member_filename = f"member_{member_index + 1:02d}.json"
    member_file = ensemble_dir / member_filename
    
    with open(member_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved ensemble member {member_index + 1} results to {member_file}")


def aggregate_ensemble_results(
    save_dir: Path,
    expected_members: int = 5
) -> Dict[str, Any]:
    """
    Aggregate results from all ensemble members and calculate statistics.
    
    Args:
        save_dir: Base directory containing ensemble_results
        expected_members: Expected number of ensemble members
        
    Returns:
        Dictionary containing aggregated statistics
    """
    ensemble_dir = save_dir / "ensemble_results"
    
    if not ensemble_dir.exists():
        raise FileNotFoundError(f"Ensemble results directory not found: {ensemble_dir}")
    
    # Load all member results
    member_results = []
    for i in range(expected_members):
        member_file = ensemble_dir / f"member_{i + 1:02d}.json"
        if member_file.exists():
            with open(member_file, 'r') as f:
                member_data = json.load(f)
                member_results.append(member_data)
        else:
            logger.warning(f"Missing results for ensemble member {i + 1}: {member_file}")
    
    if not member_results:
        raise ValueError("No ensemble member results found")
    
    logger.info(f"Found results for {len(member_results)}/{expected_members} ensemble members")
    
    # Extract all test metrics
    all_test_metrics = {}
    for member_data in member_results:
        test_metrics = member_data.get("test_metrics", {})
        for metric_name, metric_value in test_metrics.items():
            if metric_name not in all_test_metrics:
                all_test_metrics[metric_name] = []
            all_test_metrics[metric_name].append(metric_value)
    
    # Calculate statistics for each metric
    ensemble_stats = {}
    for metric_name, values in all_test_metrics.items():
        if values:  # Only if we have values
            ensemble_stats[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
                "count": len(values)
            }
    
    # Aggregate timing information
    timing_stats = {}
    training_times = []
    for member_data in member_results:
        timing = member_data.get("timing", {})
        if "total_training_time" in timing:
            training_times.append(timing["total_training_time"])
    
    if training_times:
        timing_stats = {
            "total_training_time": {
                "mean": float(np.mean(training_times)),
                "std": float(np.std(training_times)),
                "min": float(np.min(training_times)),
                "max": float(np.max(training_times)),
                "values": training_times
            }
        }
    
    # Create aggregated results
    aggregated_results = {
        "ensemble_size": len(member_results),
        "expected_size": expected_members,
        "test_metrics": ensemble_stats,
        "timing": timing_stats,
        "member_experiments": [m.get("experiment_name", f"member_{i+1}") for i, m in enumerate(member_results)],
        "timestamp": time.time()
    }
    
    # Save aggregated results
    aggregated_file = ensemble_dir / "aggregated_results.json"
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    logger.info(f"Saved aggregated ensemble results to {aggregated_file}")
    
    # Log summary statistics
    logger.info(f"Ensemble aggregation completed for {len(member_results)} members:")
    for metric_name, stats in ensemble_stats.items():
        logger.info(f"  {metric_name}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    
    return aggregated_results