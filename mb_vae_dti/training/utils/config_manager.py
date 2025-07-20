"""
Configuration management utilities for hierarchical config loading and gridsearch generation.

This module provides utilities for:
1. Loading hierarchical configurations (common → model-specific → task-specific)
2. Generating gridsearch configurations with all parameter combinations
3. Shuffling and batching configurations for HPC job submission
4. Handling config overrides and validation
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import itertools
from copy import deepcopy
import random

import yaml
from omegaconf import OmegaConf, DictConfig

TRAINING_DIR = Path(__file__).parent.parent

print(TRAINING_DIR)
# list all files in the training directory
for file in TRAINING_DIR.iterdir():
    print(file)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_ROOT = (Path(__file__).parent.parent / "configs").resolve()

class ConfigManager:
    """
    Manages hierarchical configuration loading and gridsearch generation.
    
    Configuration hierarchy:
    1. configs/common.yaml (general settings)
    2. configs/{model_type}/common.yaml (model-specific settings)
    3. configs/{model_type}/{task_config}.yaml (task-specific settings)
    """
    
    def __init__(self, config_root: Union[str, Path] = DEFAULT_CONFIG_ROOT):
        """
        Initialize the ConfigManager.
        
        Args:
            config_root: Root directory containing all configuration files.
                         Can be absolute, or relative to the current working directory.
        """
        config_root = Path(config_root)
        # If config_root is not absolute, try relative to current working directory
        if not config_root.is_absolute():
            # Try relative to current working directory
            candidate = (Path.cwd() / config_root).resolve()
            if candidate.exists():
                config_root = candidate
            else:
                # Try relative to this file's parent (project-root/mb_vae_dti/training/configs)
                candidate = (Path(__file__).parent.parent / config_root).resolve()
                if candidate.exists():
                    config_root = candidate
        if not config_root.exists():
            raise FileNotFoundError(f"Config root directory not found: {config_root}")
        self.config_root = config_root
    
    def load_config(
        self, 
        config_path: str, 
        overrides: Optional[List[str]] = None,
        gridsearch: bool = False
    ) -> Union[DictConfig, List[DictConfig]]:
        """
        Load configuration with hierarchical merging.
        
        Args:
            config_path: Path to the specific config file (e.g., "baseline/finetune_DAVIS_rand.yaml")
            overrides: List of config overrides (e.g., ["model.embedding_dim=512"])
            gridsearch: Whether to generate gridsearch configurations
            
        Returns:
            Single config or list of configs if gridsearch=True
        """
        config_path = Path(config_path)
        logger.info(f"Loading config from {config_path}")
        
        # Determine model type from path
        model_type = config_path.parent.name
        logger.debug(f"Model type: {model_type}")
        
        # Load hierarchical configs
        config = self._load_hierarchical_config(model_type, config_path.name)
        
        # Apply overrides
        if overrides:
            config = self._apply_overrides(config, overrides)
        
        # Generate gridsearch configs if requested
        if gridsearch:
            if "gridsearch" not in config:
                logger.warning("No gridsearch section found in config, returning single config")
                return [config]
            
            gridsearch_params = OmegaConf.to_container(config.gridsearch)
            base_config = deepcopy(config)
            
            # Remove gridsearch section from base config
            if "gridsearch" in base_config:
                del base_config["gridsearch"]
            
            return self._generate_gridsearch_configs(base_config, gridsearch_params)
        
        # Remove gridsearch section if present
        if "gridsearch" in config:
            del config["gridsearch"]
        
        return config
    
    def _load_hierarchical_config(self, model_type: str, config_name: str) -> DictConfig:
        """
        Load configuration files in hierarchical order and merge them.
        
        Args:
            model_type: Type of model (e.g., "baseline", "multi_input")
            config_name: Name of the specific config file
            
        Returns:
            Merged configuration
        """
        configs_to_load = []
        
        # 1. Load general common.yaml
        general_common = self.config_root / "common.yaml"
        if general_common.exists():
            configs_to_load.append(general_common)
        
        # 2. Load model-specific common.yaml
        model_common = self.config_root / model_type / "common.yaml"
        if model_common.exists():
            configs_to_load.append(model_common)
        
        # 3. Load specific config
        specific_config = self.config_root / model_type / config_name
        if not specific_config.exists():
            # Try loading from root if not found in model directory
            specific_config = self.config_root / config_name
            if not specific_config.exists():
                raise FileNotFoundError(f"Config file not found: {config_name}")
        
        configs_to_load.append(specific_config)
        
        # Load and merge all configs
        merged_config = OmegaConf.create({})
        
        for config_file in configs_to_load:
            logger.debug(f"Loading config: {config_file}")
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            config_obj = OmegaConf.create(config_data)
            merged_config = OmegaConf.merge(merged_config, config_obj)
        
        # Convert scientific notation strings to floats
        merged_config = self._convert_scientific_notation(merged_config)
        
        logger.info(f"Loaded hierarchical config from {len(configs_to_load)} files:")
        for config_file in configs_to_load:
            logger.info(f"  - {config_file}")
        
        return merged_config
    
    def _apply_overrides(self, config: DictConfig, overrides: List[str]) -> DictConfig:
        """
        Apply command-line overrides to configuration.
        
        Args:
            config: Base configuration
            overrides: List of override strings (e.g., ["training.max_epochs=10"])
            
        Returns:
            Configuration with overrides applied
        """
        logger.info(f"Applying {len(overrides)} config overrides:")
        
        for override in overrides:
            if '=' not in override:
                raise ValueError(f"Invalid override format: {override}. Expected format: key=value")
            
            key, value = override.split('=', 1)
            original_value = OmegaConf.select(config, key)
            
            # Convert string values to appropriate types
            value = self._convert_value_type(value)
            
            OmegaConf.update(config, key, value)
            logger.info(f"  {key}: {original_value} -> {value}")
        
        return config
    
    def _convert_value_type(self, value: str) -> Any:
        """Convert string value to appropriate Python type."""
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        elif value.lower() in ['null', 'none']:
            return None
        else:
            # Try to convert to int or float
            try:
                if '.' in value or 'e' in value.lower():
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                # Keep as string if conversion fails
                return value
    
    def _convert_scientific_notation(self, config: DictConfig) -> DictConfig:
        """
        Recursively convert scientific notation strings to floats.
        
        Args:
            config: Configuration object to process
            
        Returns:
            Configuration with converted values
        """
        def convert_recursive(obj):
            if isinstance(obj, dict):
                return {k: convert_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Try to convert scientific notation strings to floats
                try:
                    if 'e' in obj.lower() and (obj.replace('.', '').replace('e', '').replace('-', '').replace('+', '').isdigit()):
                        return float(obj)
                except (ValueError, AttributeError):
                    pass
                return obj
            else:
                return obj
        
        converted_dict = convert_recursive(OmegaConf.to_container(config))
        return OmegaConf.create(converted_dict)
    
    def _generate_gridsearch_configs(
        self, 
        base_config: DictConfig, 
        gridsearch_params: Dict[str, List[Any]]
    ) -> List[DictConfig]:
        """
        Generate all combinations of gridsearch parameters.
        
        Args:
            base_config: Base configuration to modify
            gridsearch_params: Dictionary of parameter names to lists of values
            
        Returns:
            List of configurations for all parameter combinations
        """
        if not gridsearch_params:
            return [base_config]
        
        # Generate all combinations
        param_names = list(gridsearch_params.keys())
        param_values = list(gridsearch_params.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Generating {len(all_combinations)} gridsearch configurations")
        logger.info(f"Parameters: {param_names}")
        
        configs = []
        for combination in all_combinations:
            config = deepcopy(base_config)
            
            # Apply parameter combination
            for param_name, param_value in zip(param_names, combination):
                OmegaConf.update(config, param_name, param_value)
            
            configs.append(config)
        
        return configs
    
    def shuffle_configs(
        self, 
        configs: List[DictConfig], 
        seed: Optional[int] = None
    ) -> List[DictConfig]:
        """
        Shuffle configurations deterministically.
        
        Args:
            configs: List of configurations to shuffle
            seed: Random seed for reproducible shuffling
            
        Returns:
            Shuffled list of configurations
        """
        if seed is None:
            # Use seed from first config if available
            seed = OmegaConf.select(configs[0], "hardware.seed", default=42)
        
        logger.info(f"Shuffling {len(configs)} configurations with seed {seed}")
        
        # Create a copy to avoid modifying original list
        shuffled_configs = configs.copy()
        random.seed(seed)
        random.shuffle(shuffled_configs)
        
        return shuffled_configs
    
    def batch_configs(
        self, 
        configs: List[DictConfig], 
        batch_size: Optional[int] = None,
        total_batches: Optional[int] = None
    ) -> List[List[DictConfig]]:
        """
        Split configurations into batches - used for HPC submission.
        
        Args:
            configs: List of configurations to batch
            batch_size: Size of each batch (mutually exclusive with total_batches)
            total_batches: Total number of batches (mutually exclusive with batch_size)
            
        Returns:
            List of configuration batches
        """
        if batch_size is None and total_batches is None:
            raise ValueError("Must specify either batch_size or total_batches")
        
        if batch_size is not None and total_batches is not None:
            raise ValueError("Cannot specify both batch_size and total_batches")
        
        if not configs:
            return []
        
        if total_batches is not None:
            # Ensure we don't create more batches than configs
            actual_batches = min(total_batches, len(configs))
            
            # Calculate batch sizes to distribute configs evenly
            base_size = len(configs) // actual_batches
            remainder = len(configs) % actual_batches
            
            batches = []
            start_idx = 0
            
            for i in range(actual_batches):
                # First 'remainder' batches get one extra config
                current_batch_size = base_size + (1 if i < remainder else 0)
                end_idx = start_idx + current_batch_size
                
                batch = configs[start_idx:end_idx]
                batches.append(batch)
                start_idx = end_idx
            
        else:
            # Use specified batch_size
            batches = []
            for i in range(0, len(configs), batch_size):
                batch = configs[i:i + batch_size]
                batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches with sizes: {[len(b) for b in batches]}")
        
        return batches


def get_config_summary(config: DictConfig) -> Dict[str, Any]:
    """
    Get a flat dictionary of all configuration key parameters.
    
    Args:
        config: Configuration to summarize
        
    Returns:
        Flat dictionary with all configuration key parameters
    """
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Recursively flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Common parameters to omit from summary
    to_omit = {
        "training.gradient_clip_val",
        "training.weight_decay",
        "data.num_workers",
        "data.pin_memory",
        "data.shuffle_train", 
        "data.drop_last",
        "logging.project_name",
        "logging.log_every_n_steps",
        "logging.use_wandb",
    }
    
    # Convert OmegaConf to regular dict and flatten
    config_dict = OmegaConf.to_container(config, resolve=True)
    flattened = flatten_dict(config_dict)
    
    # Remove common parameters
    return {k: v for k, v in flattened.items() if k not in to_omit}


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)
    
    logger.info(f"Saved configuration to {save_path}") 