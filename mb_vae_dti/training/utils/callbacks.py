from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
import torch
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TimingCallback(Callback):
    """
    Callback to track training timing metrics.
    
    This callback tracks:
    - Total training time
    - Time per epoch
    - Logs timing metrics to wandb/loggers
    """
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_start_time = None
        self.epoch_times = []
        self.total_time = None
        
    def on_train_start(self, trainer, pl_module):
        """Called at the start of training."""
        self.start_time = time.time()
        logger.info("Training started - timing tracking enabled")
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each training epoch."""
        self.epoch_start_time = time.time()
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Log epoch timing
            pl_module.log("timing/epoch_time", epoch_time)
            
            # Log running average of epoch times
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            pl_module.log("timing/avg_epoch_time", avg_epoch_time)
            
            logger.debug(f"Epoch {trainer.current_epoch} completed in {epoch_time:.2f}s")
            
    def on_train_end(self, trainer, pl_module):
        """Called at the end of training."""
        if self.start_time is not None:
            self.total_time = time.time() - self.start_time
            
            # Store timing data
            if len(self.epoch_times) > 0:
                avg_time_per_epoch = self.total_time / len(self.epoch_times)
                
                logger.info(f"Training completed in {self.total_time:.2f}s")
                logger.info(f"Average time per epoch: {avg_time_per_epoch:.2f}s")
                logger.info(f"Total epochs: {len(self.epoch_times)}")
            else:
                logger.warning("No epochs completed - timing statistics unavailable")
    
    def get_timing_stats(self) -> Dict[str, Any]:
        """Get timing statistics for gridsearch collection."""
        if not self.epoch_times:
            return {}
            
        return {
            "total_training_time": self.total_time,
            "avg_time_per_epoch": self.total_time / len(self.epoch_times) if self.total_time and self.epoch_times else None,
            "total_epochs": len(self.epoch_times),
            "min_epoch_time": min(self.epoch_times),
            "max_epoch_time": max(self.epoch_times),
        }


class BestValidationMetricsCallback(Callback):
    """
    Callback to track best validation metrics during training for gridsearch.
    
    This callback monitors validation loss and stores all validation metrics
    from the epoch where validation loss was minimum. This is essential for
    gridsearch experiments where we need to collect the best performance
    without saving full model checkpoints.
    """
    
    def __init__(self, monitor: str = "val/loss", mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_epoch = 0
        self.best_metrics = {}
        self.best_loss = float('inf') if mode == "min" else float('-inf')
        
        # Comparison function
        self.compare = torch.less if mode == "min" else torch.greater
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch."""
        # Get all logged metrics
        current_metrics = trainer.callback_metrics.copy()
        
        # Get the monitored metric
        if self.monitor not in current_metrics:
            logger.warning(f"Monitor metric '{self.monitor}' not found in logged metrics")
            return
            
        current_loss = current_metrics[self.monitor].item()
        
        # Check if this is the best validation loss so far
        if self.compare(torch.tensor(current_loss), torch.tensor(self.best_loss)):
            self.best_loss = current_loss
            self.best_epoch = trainer.current_epoch
            
            # Store all validation metrics from this epoch
            self.best_metrics = {
                key: value.item() if hasattr(value, 'item') else value
                for key, value in current_metrics.items()
                if key.startswith('val/')
            }
            
            logger.debug(f"New best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch}")
    
    def get_best_results(self) -> Dict[str, Any]:
        """Get the best validation results."""
        return {
            "best_val_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics
        }


def setup_callbacks(config: DictConfig, save_dir: Path, is_gridsearch: bool = False) -> list:
    """
    Setup training callbacks.
    
    Args:
        config: Configuration object
        save_dir: Directory to save checkpoints
        is_gridsearch: Disables checkpointing & adds BestValidationMetricsCallback
    """
    callbacks = []
    
    # Timing callback - always enabled
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)
    
    # Model checkpoint - only save best validation loss checkpoint (skip for gridsearch)
    if is_gridsearch:
        # Add best metrics tracking callback for gridsearch
        best_metrics_callback = BestValidationMetricsCallback(
            monitor="val/loss",
            mode="min"
        )
        callbacks.append(best_metrics_callback)
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir / "checkpoints",
            filename="best_model",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=config.training.early_stopping_patience,
        mode="min",
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Rich progress bar for better visualization
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    return callbacks