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
    """
    results_dir = Path(results_dir)
    result_files = list(results_dir.glob(pattern))
    results = []

    logger.info(f"Loading {len(result_files)} result files from {results_dir}")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")

    df = pd.DataFrame(results)

    if 'val_metrics' in df.columns:
        val_metrics_df = pd.json_normalize(df['val_metrics'])
        for col in val_metrics_df.columns:
            df[col] = val_metrics_df[col]
        df = df.drop('val_metrics', axis=1)
    
    if 'test_metrics' in df.columns:
        test_metrics_df = pd.json_normalize(df['test_metrics'])
        for col in test_metrics_df.columns:
            df[col] = test_metrics_df[col]
        df = df.drop('test_metrics', axis=1)
    
    if "timing" in df.columns:
        timing_df = pd.json_normalize(df['timing'])
        for col in timing_df.columns:
            df[col] = timing_df[col]
        df = df.drop('timing', axis=1)
    
    if "config" in df.columns:
        config_df = pd.json_normalize(df['config'])
        for col in config_df.columns:
            df[f"config.{col}"] = config_df[col]
        df = df.drop('config', axis=1)
    
    return df

def get_test_averages(df: pd.DataFrame) -> pd.DataFrame:
    test_cols = [col for col in df.columns if col.startswith("test/")]
    test_df = df[test_cols]
    test_df = test_df.mean(axis=0)
    return test_df