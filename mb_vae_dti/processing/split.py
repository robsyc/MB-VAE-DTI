"""
This module provides functionality for adding the data-splits to the DTI data 
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

def _get_stratification_columns(df: pd.DataFrame) -> List[str]:
    """Helper function to get stratification columns from the dataframe."""
    return [col for col in df.columns if col.startswith('in_')]

def _create_composite_stratification_key(df: pd.DataFrame, strat_cols: List[str]) -> pd.Series:
    """Creates a composite stratification key from multiple columns."""
    strat_key = ''
    for col in strat_cols:
        strat_key += df[col].astype(str) + '_'
    return strat_key

def _assign_stratified_split(
    items: np.ndarray, 
    strat_dict: Dict[Any, List[Any]], 
    frac: Tuple[float, float, float]
) -> Dict[Any, str]:
    """
    Assign stratified train/valid/test splits to items based on stratification groups.
    
    Args:
        items: Array of item identifiers (drugs or indices)
        strat_dict: Dictionary mapping stratification keys to lists of items
        frac: Split fractions (train, valid, test)
        
    Returns:
        Dictionary mapping each item to its split assignment
    """
    item_to_split = {}
    
    # For each stratification group, split the items
    for strat_key, strat_items in strat_dict.items():
        # Shuffle the items
        np.random.shuffle(strat_items)
        
        # Calculate split sizes
        train_size = int(len(strat_items) * frac[0])
        valid_size = int(len(strat_items) * frac[1])
        
        # Assign splits
        for item in strat_items[:train_size]:
            item_to_split[item] = 'train'
        for item in strat_items[train_size:train_size+valid_size]:
            item_to_split[item] = 'valid'
        for item in strat_items[train_size+valid_size:]:
            item_to_split[item] = 'test'
    
    return item_to_split

def _split_items(
    items: np.ndarray, 
    frac: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split items into train, valid, test sets based on fractions.
    
    Args:
        items: Array of item identifiers
        frac: Split fractions (train, valid, test)
        
    Returns:
        Tuple of (train_items, valid_items, test_items)
    """
    np.random.shuffle(items)
    
    # Calculate split sizes
    train_size = int(len(items) * frac[0])
    valid_size = int(len(items) * frac[1])
    
    # Split items
    train_items = items[:train_size]
    valid_items = items[train_size:train_size+valid_size]
    test_items = items[train_size+valid_size:]
    
    return train_items, valid_items, test_items

def _items_to_split_dict(
    train_items: np.ndarray, 
    valid_items: np.ndarray, 
    test_items: np.ndarray
) -> Dict[Any, str]:
    """
    Create a mapping from items to their split assignments.
    
    Args:
        train_items: Items in the train set
        valid_items: Items in the valid set
        test_items: Items in the test set
        
    Returns:
        Dictionary mapping each item to its split assignment
    """
    item_to_split = {}
    
    for item in train_items:
        item_to_split[item] = 'train'
    for item in valid_items:
        item_to_split[item] = 'valid'
    for item in test_items:
        item_to_split[item] = 'test'
    
    return item_to_split

def add_split_cols(
        df: pd.DataFrame,
        frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        rand_split: bool = True,
        cold_split: bool = True,
        stratify: bool = True,
        seed: int = 42
) -> pd.DataFrame:
    """
    Adds split columns to the dataframe based on the given fractions.
    
    Args:
        df: dataframe to add split columns to
        frac: train, valid, test split fractions (default: (0.8, 0.1, 0.1))
        rand_split: whether to add a random split (default: True)
        cold_split: whether to add a cold split (default: True)
        stratify: whether to stratify the split on in_... columns (default: True)
        seed: random seed (default: 42)
        
    Returns:
        DataFrame with added split columns
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Validate fractions
    if sum(frac) != 1.0:
        raise ValueError(f"Split fractions must sum to 1.0, got {sum(frac)}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get stratification columns if needed
    strat_cols = _get_stratification_columns(df) if stratify else []
    
    # Random split
    if rand_split:
        if stratify and strat_cols:
            # Create a composite stratification key
            df['_strat_key'] = _create_composite_stratification_key(df, strat_cols)
            
            # Get unique stratification keys and their counts
            strat_counts = df['_strat_key'].value_counts()
            
            # Initialize split column
            df['split_rand'] = ''
            
            # For each stratification group
            for strat_key, count in strat_counts.items():
                # Get indices for this stratification group
                indices = df[df['_strat_key'] == strat_key].index
                
                # Shuffle indices
                shuffled_indices = np.random.permutation(indices)
                
                # Calculate split sizes
                train_size = int(count * frac[0])
                valid_size = int(count * frac[1])
                
                # Assign splits
                df.loc[shuffled_indices[:train_size], 'split_rand'] = 'train'
                df.loc[shuffled_indices[train_size:train_size+valid_size], 'split_rand'] = 'valid'
                df.loc[shuffled_indices[train_size+valid_size:], 'split_rand'] = 'test'
            
            # Remove temporary stratification key
            df.drop('_strat_key', axis=1, inplace=True)
        else:
            # Simple random split without stratification
            df['split_rand'] = np.random.choice(
                ['train', 'valid', 'test'], 
                size=len(df), 
                p=frac
            )
    
    # Cold drug split (drugs in test set don't appear in train set)
    if cold_split:
        unique_drugs = df['Drug_ID'].unique()
        
        if stratify and strat_cols:
            # For each drug, determine its stratification profile
            drug_strat_profiles = {}
            
            # Group by Drug_ID and aggregate the stratification columns
            drug_groups = df.groupby('Drug_ID')
            
            for drug, group in drug_groups:
                # Create a stratification profile for this drug
                # We'll use the most common value for each stratification column
                profile = {}
                for col in strat_cols:
                    # Get the most common value (True/False) for this column for this drug
                    profile[col] = group[col].mode()[0]
                
                # Create a composite key from the profile
                strat_key = '_'.join([f"{col}_{profile[col]}" for col in strat_cols])
                drug_strat_profiles[drug] = strat_key
            
            # Group drugs by their stratification profiles
            strat_groups = {}
            for drug, strat_key in drug_strat_profiles.items():
                if strat_key not in strat_groups:
                    strat_groups[strat_key] = []
                strat_groups[strat_key].append(drug)
            
            # Assign splits based on stratification
            drug_to_split = _assign_stratified_split(unique_drugs, strat_groups, frac)
        else:
            # Simple random split of drugs
            train_drugs, valid_drugs, test_drugs = _split_items(unique_drugs, frac)
            drug_to_split = _items_to_split_dict(train_drugs, valid_drugs, test_drugs)
        
        # Apply the mapping to create the split column
        df['split_cold'] = df['Drug_ID'].map(drug_to_split)
    
    return df


def add_split_cols_drug_generation(
    df: pd.DataFrame,
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> pd.DataFrame:
    """
    Adds split columns to the dataframe based on the given fractions.
    """
    df = df.copy()
    
    # Validate fractions
    if sum(frac) != 1.0:
        raise ValueError(f"Split fractions must sum to 1.0, got {sum(frac)}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Add random train/valid/test split
    df['split'] = np.random.choice(
        ['train', 'valid', 'test'], 
        size=len(df), 
        p=frac
    )
    
    return df
