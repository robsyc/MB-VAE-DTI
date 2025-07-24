from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, ModelSummary
import torch
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TimingCallback(Callback):
    """
    Callback to track training timing metrics.
    
    This callback tracks the total training time and the time per epoch.
    """
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_start_time = None
        self.epoch_times = []
        self.total_time = None
        
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        logger.info(f"Training started at {self.start_time}")
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        logger.info(f"Epoch {trainer.current_epoch} started at T+ {self.epoch_start_time - self.start_time:.2f}s")
        
    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            logger.debug(f"Epoch {trainer.current_epoch} completed in {epoch_time:.2f}s")
            
    def on_train_end(self, trainer, pl_module):
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
        if not self.epoch_times:
            return {}
            
        return {
            "total_training_time": self.total_time,
            "avg_time_per_epoch": self.total_time / len(self.epoch_times) if self.total_time and self.epoch_times else None,
            "total_epochs": len(self.epoch_times)
        }


class BestMetricsCallback(Callback):
    """
    Callback to track best validation metrics during training for gridsearch.
    
    This callback monitors validation loss and stores all validation metrics
    from the epoch where validation loss was minimum. This is essential for
    gridsearch experiments where we need to collect the best performance
    without saving full model checkpoints.
    
    Also optionally stores the best model's state dict in memory for testing
    without writing to disk.
    """
    
    def __init__(self, monitor: str = "val/loss", mode: str = "min", store_best_model: bool = True):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.store_best_model = store_best_model
        self.best_epoch = 0
        self.best_val_metrics = {}
        self.test_metrics = {}
        self.best_loss = float('inf') if mode == "min" else float('-inf')
        self.best_model_state_dict = None
        
        # Comparison function
        self.compare = torch.less if mode == "min" else torch.greater
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Get all metrics available to callbacks e.g.
        # def training_step(self, batch, batch_idx):
        #       self.log("a_val", 2.0) # this will be in current_metrics
        current_metrics = trainer.callback_metrics.copy()
        
        if self.monitor not in current_metrics:
            logger.warning(f"Monitor metric '{self.monitor}' not found in logged metrics")
            return
            
        current_loss = current_metrics[self.monitor].item()
        if self.compare(torch.tensor(current_loss), torch.tensor(self.best_loss)):
            self.best_loss = current_loss
            self.best_epoch = trainer.current_epoch
            
            # Store all validation metrics from this epoch
            self.best_val_metrics = {
                key: value.item() if hasattr(value, 'item') else value
                for key, value in current_metrics.items()
                if key.startswith('val/')
            }
            
            # Store the best model's state dict in memory if requested
            if self.store_best_model:
                import copy
                self.best_model_state_dict = copy.deepcopy(pl_module.state_dict())
                
            logger.debug(f"New best validation loss `{self.monitor}`: {self.best_loss:.6f} at epoch {self.best_epoch}")

    def on_test_epoch_end(self, trainer, pl_module):
        current_metrics = trainer.callback_metrics.copy()
        self.test_metrics = {
            key: value.item() if hasattr(value, 'item') else value
            for key, value in current_metrics.items()
            if key.startswith('test/')
        }
        logger.debug(f"Test metrics: {self.test_metrics}")

    def load_best_model(self, pl_module):
        """Load the best model state dict into the provided module."""
        if self.best_model_state_dict is None:
            logger.warning("No best model state dict stored. Make sure store_best_model=True and training has completed.")
            return
        
        pl_module.load_state_dict(self.best_model_state_dict)
        logger.info(f"Loaded best model from epoch {self.best_epoch} (loss: {self.best_loss:.6f})")

    def get_best_results(self) -> Dict[str, Any]:
        """Get the best validation results."""
        return {
            "best_val_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "val_metrics": self.best_val_metrics,
            "test_metrics": self.test_metrics
        }


def setup_callbacks(
        config: DictConfig,
        save_dir: Path,
        save_checkpoint: bool = False,
    ) -> list:
    """
    Setup training callbacks.
    
    Args:
        config: Configuration object
        save_dir: Directory to save checkpoints
        save_checkpoint: Whether to save the final model to disk
    """
    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval='step'),
        TimingCallback(),
        EarlyStopping(
            monitor="val/loss",
            patience=config.training.early_stopping_patience,
            mode="min",
            verbose=True
        ),
        BestMetricsCallback(
            monitor="val/loss",
            mode="min",
            store_best_model=not save_checkpoint  # Store in memory when not saving to disk
        ),
    ]

    # Only add ModelCheckpoint if we actually want to save checkpoints
    if save_checkpoint:
        callbacks.append(ModelCheckpoint(
            dirpath=save_dir / "checkpoints",
            filename="best_model",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            verbose=True
        ))
    
    return callbacks