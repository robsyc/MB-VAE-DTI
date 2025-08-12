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
        logger.info(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        elapsed_seconds = self.epoch_start_time - self.start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Epoch {trainer.current_epoch} started at T+ {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
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
    Callback to track best validation loss and keep copy of best model's val/test metrics.
    """
    def __init__(self, monitor: str = "val/loss", mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_epoch = 0
        self.best_val_metrics = {}
        self.test_metrics = {}
        self.best_loss = float('inf') if mode == "min" else float('-inf')
        
        # Comparison function
        self.compare = torch.less if mode == "min" else torch.greater
        
    def on_validation_end(self, trainer, pl_module):
        """Capture validation metrics after model's on_validation_epoch_end has computed them."""
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
            logger.debug(f"New best validation loss `{self.monitor}`: {self.best_loss:.6f} at epoch {self.best_epoch}")
        
        # Always update validation metrics from the current epoch if it matches the best epoch
        # This ensures we capture complete metrics from the best epoch, even if computed later
        if trainer.current_epoch == self.best_epoch:
            # Store all validation metrics from this epoch
            self.best_val_metrics = {
                key: value.item() if hasattr(value, 'item') else value
                for key, value in current_metrics.items()
                if key.startswith('val/')
            }

    def on_test_end(self, trainer, pl_module):
        """Capture test metrics after model's on_test_epoch_end has computed them."""
        current_metrics = trainer.callback_metrics.copy()
        # Capture test metrics from callback_metrics
        self.test_metrics = {
            key: value.item() if hasattr(value, 'item') else value
            for key, value in current_metrics.items()
            if key.startswith('test/')
        }
        logger.info(f"Test metrics found: {self.test_metrics}")

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
        monitor: str = "val/loss",
    ) -> list:
    """
    Setup training callbacks.
    
    Args:
        config: Configuration object
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor (default: "val/loss")
    """
    return [
        ModelSummary(max_depth=1),
        LearningRateMonitor(logging_interval='step'),
        TimingCallback(),
        EarlyStopping(
            monitor=monitor,
            patience=config.training.early_stopping_patience,
            mode="min",
            verbose=True
        ),
        BestMetricsCallback(
            monitor=monitor,
            mode="min"
        ),
        ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="best_model",
        monitor=monitor,
        mode="min",
        save_top_k=3,
        save_last=True,
        save_weights_only=True,  # Save only model weights, not optimizer states
        verbose=True
        )
    ]