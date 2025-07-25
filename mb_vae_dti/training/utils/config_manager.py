"""
Configuration management utilities for config loading and gridsearch generation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import itertools
from copy import deepcopy
import random

import yaml
from omegaconf import OmegaConf, DictConfig


logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration loading and generation of gridsearch and ensemble configurations.
    """
    
    def __init__(self):
        """
        Initialize the ConfigManager.
        """
        pass
    
    def load_config(
        self, 
        config_path: str, 
        overrides: Optional[List[str]] = None,
        gridsearch: bool = False,
        ensemble: bool = False
    ) -> Union[DictConfig, List[DictConfig]]:
        """
        Load configuration from a single YAML file.
        
        Args:
            config_path: Path to the specific config file (e.g., "baseline/finetune_DAVIS_rand.yaml")
            overrides: List of config overrides (e.g., ["model.embedding_dim=512"])
            gridsearch: Whether to generate gridsearch configurations
            ensemble: Whether to generate ensemble configurations
            
        Returns:
            Single config or list of configs if gridsearch=True or ensemble=True
        """
        config_path = Path(config_path)
        logger.info(f"Loading config from {config_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = OmegaConf.create(config_data)
        
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
            if "ensemble" in base_config:
                del base_config["ensemble"]
            
            return self._generate_gridsearch_configs(base_config, gridsearch_params)
        
        # Generate ensemble configs if requested
        if ensemble:
            if "ensemble" not in config or "configs" not in config.ensemble:
                logger.warning("No ensemble.configs section found in config, returning single config")
                return [config]
            
            ensemble_configs = OmegaConf.to_container(config.ensemble.configs)
            base_config = deepcopy(config)
            
            # Remove ensemble section from base config
            if "ensemble" in base_config:
                del base_config["ensemble"]
            if "gridsearch" in base_config:
                del base_config["gridsearch"]
            
            return self._generate_ensemble_configs(base_config, ensemble_configs)
        
        # Remove gridsearch and ensemble sections if present
        if "gridsearch" in config:
            del config["gridsearch"]
        if "ensemble" in config:
            del config["ensemble"]
        
        return config
    
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
    
    def _generate_ensemble_configs(
        self, 
        base_config: DictConfig, 
        ensemble_configs: List[Dict[str, Any]]
    ) -> List[DictConfig]:
        """
        Generate ensemble configurations from predefined parameter sets.
        
        Args:
            base_config: Base configuration to modify
            ensemble_configs: List of parameter dictionaries to apply
            
        Returns:
            List of configurations for each ensemble member
        """
        logger.info(f"Generating {len(ensemble_configs)} ensemble configurations")
        
        configs = []
        for i, ensemble_params in enumerate(ensemble_configs):
            config = deepcopy(base_config)
            
            # Apply ensemble parameters
            for param_name, param_value in ensemble_params.items():
                OmegaConf.update(config, param_name, param_value)
            
            configs.append(config)
            logger.debug(f"Ensemble config {i+1}: {ensemble_params}")
        
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
        total_batches: Optional[int]
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
                # Convert lists to underscore-separated strings
                if isinstance(v, list):
                    v = '_'.join(str(item) for item in v)
                items.append((new_key, v))
        return dict(items)
    
    to_keep = {
        "training.learning_rate",
        "training.scheduler",
        "loss.weights",
        "loss.dti_weights",
        "loss.diff_weights",
        "loss.contrastive_temp",
        "data.batch_size",
        "data.h5_path",
        "data.drug_features",
        "data.target_features",
        "model.embedding_dim",
        "model.hidden_dim",
        "model.num_layers",
        "model.encoder_type",
        "model.aggregator_type"
    }
    
    # Convert OmegaConf to regular dict and flatten
    config_dict = OmegaConf.to_container(config, resolve=True)
    flattened = flatten_dict(config_dict)
    
    # Remove common parameters
    return {k: v for k, v in flattened.items() if k in to_keep}