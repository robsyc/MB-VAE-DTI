from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import logging
from typing import Optional
import time

logger = logging.getLogger(__name__)


def generate_experiment_name(
    model: str,
    phase: str,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    pretrain_target: Optional[str] = None,
    batch_index: Optional[int] = None,
    config_index: Optional[int] = None,
    ensemble: Optional[bool] = False
) -> str:
    """
    Generates a unique experiment name.

    Args:
        model: Model type
        phase: Training phase
        dataset: Dataset (for finetune phase)
        split: Split type (for finetune and train phases)
        pretrain_target: Pretrain target (for pretrain phase)
        batch_index: Index of the current batch of runs (if applicable)
        config_index: Index of config within the batch (if applicable)
        ensemble: Whether this is an ensemble run
    Returns:
        Unique experiment name
    """
    name = f"{model}_{phase}"

    if dataset is not None:
        name += f"_{dataset}"
    if split is not None:
        name += f"_{split}"
    if pretrain_target is not None:
        name += f"_{pretrain_target}"

    if batch_index is not None and config_index is not None:
        name += f"_b{batch_index:02d}c{config_index:04d}"
    elif config_index is not None:
        name += f"_{config_index:04d}"
    if ensemble:
        name += "_ensemble"

    return name


def setup_logging(
        config: DictConfig, 
        save_dir: Path, 
        model: str, 
        phase: str, 
        dataset: str = None, 
        split: str = None, 
        pretrain_target: str = None
    ) -> list:
    """
    Setup logging with WandB.
    """
    loggers = []
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # WandB logger
    if config.logging.get('use_wandb', True):
        try:
            tags = [model, phase]
            if dataset is not None:
                tags.append(dataset)
            if split is not None:
                tags.append(split)
            if pretrain_target is not None:
                tags.append(pretrain_target)
            
            # Add gridsearch tag if this is a gridsearch run
            if '_b' in config.logging.experiment_name and 'c' in config.logging.experiment_name:
                tags.append('gridsearch')
            if '_ensemble' in config.logging.experiment_name:
                tags.append('ensemble')
            
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