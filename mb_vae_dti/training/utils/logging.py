from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import logging
from typing import Optional
import time

logger = logging.getLogger(__name__)


def generate_experiment_name(
    original_name: str,
    batch_index: Optional[int] = None,
    config_index: Optional[int] = None
) -> str:
    """
    Generate a unique experiment name for gridsearch runs.
    
    Args:
        original_name: Original experiment name from config
        batch_index: Index of the current batch of gridsearch runs (if applicable)
        config_index: Index of config within the batch of gridsearch runs (if applicable)
        
    Returns:
        Unique experiment name
    """
    if batch_index is not None and config_index is not None:
        # For gridsearch: include batch index and config index
        # Also add timestamp to ensure uniqueness across runs
        timestamp = int(time.time())
        return f"{original_name}_b{batch_index:02d}c{config_index:04d}_{timestamp}"
    elif config_index is not None:
        # For single-batch gridsearch
        return f"{original_name}_{config_index:04d}"
    else:
        # For single runs
        return original_name
    

def setup_logging(config: DictConfig, save_dir: Path) -> list:
    """
    Setup logging with WandB.
    """
    loggers = []
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # WandB logger
    if config.logging.get('use_wandb', True):
        try:
            # Add tags for better organization
            tags = [
                getattr(config.model, 'name', 'unknown_model'),
                getattr(config.data, 'split_type', 'unknown_split')
            ]
            
            # Add gridsearch tag if this is a gridsearch run
            if '_b' in config.logging.experiment_name and 'c' in config.logging.experiment_name:
                tags.append('gridsearch')
            
            wandb_logger = WandbLogger(
                project=config.logging.project_name,
                name=config.logging.experiment_name,
                save_dir=str(save_dir),
                config=OmegaConf.to_container(config, resolve=True),
                tags=tags
            )
            loggers.append(wandb_logger)
            logger.info(f"Initialized WandB logger with tags: {tags}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB logger: {e}")
            logger.warning("Continuing without WandB logging")
    
    return loggers