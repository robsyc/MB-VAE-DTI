"""
Analysis utilities for gridsearch experiment results.

This module provides functions to analyze the performance of different
hyperparameter combinations from gridsearch experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd

logger = logging.getLogger(__name__)


def load_gridsearch_results(
    results_dir: Union[str, Path],
    pattern: str = "*_results.json"
) -> pd.DataFrame:
    """
    Load all gridsearch results from a directory into a pandas DataFrame.
    
    Args:
        results_dir: Directory containing JSON result files
        pattern: Glob pattern to match result files (default: "*_results.json")
        
    Returns:
        DataFrame with all results, with metadata columns flattened
        
    Raises:
        FileNotFoundError: If results directory doesn't exist
        ValueError: If no result files found
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all result files
    result_files = list(results_dir.glob(pattern))
    
    if not result_files:
        raise ValueError(f"No result files found in {results_dir} matching pattern '{pattern}'")
    
    logger.info(f"Loading {len(result_files)} result files from {results_dir}")
    
    # Load all results
    all_results = []
    failed_files = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            all_results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")
            failed_files.append(result_file)
    
    if not all_results:
        raise ValueError("No valid result files could be loaded")
    
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Flatten metadata columns
    if 'metadata' in df.columns:
        metadata_df = pd.json_normalize(df['metadata'])
        
        # Add metadata columns to main DataFrame
        for col in metadata_df.columns:
            df[f"metadata.{col}"] = metadata_df[col]
        
        # Remove original metadata column
        df = df.drop('metadata', axis=1)
    
    # Flatten metrics columns if present
    if 'metrics' in df.columns:
        metrics_df = pd.json_normalize(df['metrics'])
        
        # Add metrics columns to main DataFrame
        for col in metrics_df.columns:
            df[f"metrics.{col}"] = metrics_df[col]
        
        # Remove original metrics column
        df = df.drop('metrics', axis=1)
    
    # Flatten timing columns if present
    if 'timing' in df.columns:
        timing_df = pd.json_normalize(df['timing'])
        
        # Add timing columns to main DataFrame
        for col in timing_df.columns:
            df[f"timing.{col}"] = timing_df[col]
        
        # Remove original timing column
        df = df.drop('timing', axis=1)
    
    logger.info(f"Successfully loaded {len(df)} results with {len(df.columns)} columns")
    
    return df


def get_varying_attributes(
    df: pd.DataFrame,
    prefix: str = "metadata."
) -> List[str]:
    """
    Identify metadata attributes that vary across experiments.
    
    Args:
        df: DataFrame with gridsearch results
        prefix: Prefix to filter columns by (default: "metadata.")
        
    Returns:
        List of column names that have varying values
    """
    if df.empty:
        return []
    
    # Get all metadata columns
    metadata_cols = [col for col in df.columns if col.startswith(prefix)]
    
    varying_attrs = []
    
    for col in metadata_cols:
        # Check if column has more than one unique value
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 1:
            varying_attrs.append(col)
    
    logger.info(f"Found {len(varying_attrs)} varying attributes: {varying_attrs}")
    
    return varying_attrs


def get_top_performers(
    df: pd.DataFrame,
    metric: str = "best_val_loss",
    n_top: int = 10,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Get the top N performing experiments based on a metric.
    
    Args:
        df: DataFrame with gridsearch results
        metric: Column name to sort by (default: "best_val_loss")
        n_top: Number of top results to return (default: 10)
        ascending: Whether to sort in ascending order (default: True for loss)
        
    Returns:
        DataFrame with top N performers
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns")
    
    # Sort by metric and get top N
    top_performers = df.sort_values(metric, ascending=ascending).head(n_top)
    
    logger.info(f"Top {len(top_performers)} performers by {metric}:")
    for i, (_, row) in enumerate(top_performers.iterrows(), 1):
        logger.info(f"  {i}. {row['experiment_name']}: {metric}={row[metric]:.6f}")
    
    return top_performers
