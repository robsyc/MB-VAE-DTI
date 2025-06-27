"""
This module provides functionality for adding the data-splits to the DTI data 
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

def _get_stratification_columns(df: pd.DataFrame) -> List[str]:
    """Helper function to get stratification columns from the dataframe."""
    return [col for col in df.columns if col.startswith('in_')]


def _create_stratification_key(df: pd.DataFrame, stratify_cols: List[str]) -> pd.Series:
    """
    Create a stratification key by combining boolean columns.
    
    Args:
        df: DataFrame with stratification columns
        stratify_cols: List of column names to use for stratification
    
    Returns:
        Series with stratification keys
    """
    if not stratify_cols:
        return pd.Series(['all'] * len(df), index=df.index)
    
    # Create a string key from boolean combinations
    strat_key = df[stratify_cols].astype(str).agg('_'.join, axis=1)
    return strat_key


def _split_with_stratification(
    items: pd.Series, 
    split_fractions: Tuple[float, float, float],
    stratify_key: Optional[pd.Series] = None,
    random_state: int = 42
) -> pd.Series:
    """
    Split items into train/val/test with optional stratification.
    
    Args:
        items: Series of items to split
        split_fractions: Tuple of (train_frac, val_frac, test_frac)
        stratify_key: Optional stratification key
        random_state: Random seed
    
    Returns:
        Series with split assignments
    """
    train_frac, val_frac, test_frac = split_fractions
    
    # Validate fractions
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")
    
    # Initialize result series
    result = pd.Series(index=items.index, dtype=str)
    
    if stratify_key is None:
        # Simple random split without stratification
        items_shuffled = items.sample(frac=1, random_state=random_state)
        n_total = len(items_shuffled)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        
        result.loc[items_shuffled.index[:n_train]] = 'train'
        result.loc[items_shuffled.index[n_train:n_train + n_val]] = 'val'
        result.loc[items_shuffled.index[n_train + n_val:]] = 'test'
        
    else:
        # Stratified split
        unique_strats = stratify_key.unique()
        
        for strat in unique_strats:
            strat_mask = stratify_key == strat
            strat_items = items[strat_mask]
            
            if len(strat_items) == 0:
                continue
            elif len(strat_items) == 1:
                # Single item - assign to train
                result.loc[strat_items.index] = 'train'
            elif len(strat_items) == 2:
                # Two items - one to train, one to val
                strat_items_shuffled = strat_items.sample(frac=1, random_state=random_state)
                result.loc[strat_items_shuffled.index[0]] = 'train'
                result.loc[strat_items_shuffled.index[1]] = 'val'
            else:
                # Multiple items - proper stratified split
                strat_items_shuffled = strat_items.sample(frac=1, random_state=random_state)
                n_strat = len(strat_items_shuffled)
                n_train_strat = max(1, int(n_strat * train_frac))
                n_val_strat = max(1, int(n_strat * val_frac)) if val_frac > 0 else 0
                
                # Adjust if we exceed total
                if n_train_strat + n_val_strat >= n_strat:
                    if n_strat >= 3:
                        n_train_strat = n_strat - 2
                        n_val_strat = 1
                    else:
                        n_train_strat = n_strat - 1
                        n_val_strat = 0
                
                result.loc[strat_items_shuffled.index[:n_train_strat]] = 'train'
                result.loc[strat_items_shuffled.index[n_train_strat:n_train_strat + n_val_strat]] = 'val'
                result.loc[strat_items_shuffled.index[n_train_strat + n_val_strat:]] = 'test'
    
    return result


def add_random_split(
    df: pd.DataFrame,
    split_fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    stratify: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Add random split column to the dataframe.
    
    Args:
        df: DTI dataframe
        split_fractions: Tuple of (train_frac, val_frac, test_frac)
        stratify: Whether to stratify by provenance columns
        random_state: Random seed
    
    Returns:
        DataFrame with added 'split_rand' column
    """
    df_copy = df.copy()
    
    # Get stratification key if needed
    stratify_key = None
    if stratify:
        stratify_cols = _get_stratification_columns(df_copy)
        if stratify_cols:
            stratify_key = _create_stratification_key(df_copy, stratify_cols)
    
    # Create row indices series
    row_indices = pd.Series(df_copy.index, index=df_copy.index)
    
    # Perform split
    split_assignments = _split_with_stratification(
        row_indices, split_fractions, stratify_key, random_state
    )
    
    df_copy['split_rand'] = split_assignments
    return df_copy


def add_cold_drug_split(
    df: pd.DataFrame,
    split_fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    stratify: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Add cold drug split column to the dataframe.
    In cold split, unique drugs are assigned to splits, ensuring no drug appears in multiple splits.
    
    Args:
        df: DTI dataframe
        split_fractions: Tuple of (train_frac, val_frac, test_frac)
        stratify: Whether to stratify by provenance columns
        random_state: Random seed
    
    Returns:
        DataFrame with added 'split_cold' column
    """
    df_copy = df.copy()
    
    # Get unique drugs
    unique_drugs = df_copy['Drug_ID'].unique()
    unique_drugs_series = pd.Series(unique_drugs, index=unique_drugs)
    
    # Get stratification key if needed (based on drug-level aggregation)
    stratify_key = None
    if stratify:
        stratify_cols = _get_stratification_columns(df_copy)
        if stratify_cols:
            # Aggregate stratification info at drug level
            # A drug is in a dataset if it has at least one interaction in that dataset
            drug_strat = df_copy.groupby('Drug_ID')[stratify_cols].any()
            stratify_key = _create_stratification_key(drug_strat, stratify_cols)
            # Reindex to match unique_drugs_series
            stratify_key = stratify_key.reindex(unique_drugs_series.index)
    
    # Split unique drugs
    drug_splits = _split_with_stratification(
        unique_drugs_series, split_fractions, stratify_key, random_state
    )
    
    # Map drug splits back to all rows
    df_copy['split_cold'] = df_copy['Drug_ID'].map(drug_splits)
    
    return df_copy


def add_splits(
    df: pd.DataFrame,
    split_fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    stratify: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Add both random and cold drug splits to the dataframe.
    
    Args:
        df: DTI dataframe
        split_fractions: Tuple of (train_frac, val_frac, test_frac)
        stratify: Whether to stratify by provenance columns
        random_state: Random seed
    
    Returns:
        DataFrame with added 'split_rand' and 'split_cold' columns
    """
    # Add random split
    df_with_rand = add_random_split(df, split_fractions, stratify, random_state)
    
    # Add cold split
    df_with_both = add_cold_drug_split(df_with_rand, split_fractions, stratify, random_state)
    
    return df_with_both
