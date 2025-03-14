"""
Annotation functionality for MB-VAE-DTI.

This module provides functions for assigning proper IDs to unique drugs and targets,
fetching names from external sources, and building dictionaries mapping IDs to sets of names.
The main goal is to provide consistent identifiers across different datasets.
"""

from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, Literal, List, Set, Optional
from tdc.multi_pred import DTI
import json
import numpy as np
import warnings
from tqdm import tqdm

from mb_vae_dti.loading.datasets import canonicalize_smiles
from mb_vae_dti.loading.drug_annotation import annotate_drug, save_cache as save_drug_cache
from mb_vae_dti.loading.target_annotation import annotate_target, save_cache as save_target_cache

# Define paths
DATA_DIR = Path("data")
SOURCE_DIR = DATA_DIR / "source"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = PROCESSED_DIR / "annotations"

# Ensure directories exist
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

# Cache for potential IDs to avoid redundant processing
POTENTIAL_IDS_CACHE_FILE = ANNOTATION_DIR / "potential_ids_cache.json"

# Initialize cache
potential_ids_cache = {}

# Load cache if it exists
if POTENTIAL_IDS_CACHE_FILE.exists():
    try:
        with open(POTENTIAL_IDS_CACHE_FILE, 'r') as f:
            potential_ids_cache = json.load(f)
    except Exception as e:
        print(f"Failed to load potential IDs cache: {e}")

def save_potential_ids_cache() -> None:
    """Save potential IDs cache to disk."""
    try:  
        with open(POTENTIAL_IDS_CACHE_FILE, 'w') as f:
            json.dump(potential_ids_cache, f)
    except Exception as e:
        print(f"Failed to save potential IDs cache: {e}")


def load_metz_ids() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load the Metz drug and target IDs from a CSV file.

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]: Dictionaries mapping drug and target strings to their associated IDs
            - Drug_ID   : PubChem ID (e.g. 103904191: https://www.ncbi.nlm.nih.gov/compound/103904191/)
            - Target_ID : Gene name / symbol searchable on Uniprot (e.g. PRKAA1: https://www.uniprot.org/uniprotkb/Q13131/entry)
    """
    metz_path = SOURCE_DIR / "Metz.csv"
    try:
        df = pd.read_csv(metz_path, usecols=['SMILES', 'ProteinSequence', 'GeneID', 'Symbol'])
        df = df.drop_duplicates(subset=['SMILES', 'ProteinSequence'])
        
        df['GeneID'] = df['GeneID'].astype(str)
        df['Symbol'] = df['Symbol'].astype(str)

        drug_ids = dict(zip(df['SMILES'], df['GeneID']))
        target_ids = dict(zip(df['ProteinSequence'], df['Symbol']))

        return drug_ids, target_ids
    
    except FileNotFoundError:
        print(f'To run the Metz dataset, you need to download it from Kaggle and place it in {metz_path}')
        print("https://www.kaggle.com/datasets/blk1804/metz-drug-binding-dataset")
        raise FileNotFoundError(f"Metz dataset not found at {metz_path}")


def load_dataset_ids(
        name: Literal["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "KIBA", "Metz"], 
        verbose: bool = False
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load a DTI dataset and return the drug and target IDs

    Args:
        name: Name of the dataset to load
        verbose: Whether to print additional information

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]: Dictionaries mapping drug and target strings to their associated IDs
    """
    if verbose:
        print(f"Loading {name} dataset...")
    
    # Load dataset
    if name == "Metz":
        drug_ids, target_ids = load_metz_ids()
        canonical_dict = canonicalize_smiles(list(drug_ids.keys()), verbose=verbose)
        drug_ids = {smiles: drug_ids[original_smiles] for original_smiles, smiles in canonical_dict.items()}
    else:
        data = DTI(name=name, path=str(SOURCE_DIR))

        df = data.get_data()
        df = df.drop_duplicates(subset=['Drug', 'Target'])

        # Clean and convert IDs properly
        def clean_id(id_val):
            try:
                float_val = float(id_val)
                if pd.isna(float_val):
                    return None
                if float_val.is_integer():
                    return str(int(float_val))
                return str(float_val)
            except (ValueError, TypeError):
                return str(id_val) if not pd.isna(id_val) else None
        
        df['Drug_ID'] = df['Drug_ID'].apply(clean_id)
        df['Target_ID'] = df['Target_ID'].apply(clean_id)
        
        # Filter out entries with None/NaN IDs
        drug_ids = {drug: drug_id for drug, drug_id in zip(df['Drug'], df['Drug_ID']) 
                   if drug_id is not None}
        target_ids = {target: target_id for target, target_id in zip(df['Target'], df['Target_ID']) 
                     if target_id is not None}

        # Canonicalize SMILES strings
        unique_smiles = df['Drug'].unique()
        canonical_dict = canonicalize_smiles(unique_smiles, verbose=verbose)
        df['Drug'] = df['Drug'].map(canonical_dict)

    if verbose:
        print(f"Gathered {len(drug_ids)} unique drugs and {len(target_ids)} unique targets")

    return drug_ids, target_ids


def generate_unique_ids(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate unique IDs for drugs and targets based on SMILES and amino acid sequences.
    
    Args:
        df: merged DataFrame containing Drug_SMILES and Target_AA columns
        verbose: Whether to print additional information
        
    Returns:
        df_drugs, df_targets : Tuple[pd.DataFrame, pd.DataFrame]: 
        Tuple containing the two DataFrames of unique drugs and targets with added Drug_ID and Target_ID columns.
    """
    if verbose:
        print("Generating unique IDs for drugs and targets...")
    
    # Extract unique drugs and targets directly into DataFrames
    unique_drugs = df[['Drug_SMILES']].drop_duplicates().reset_index(drop=True)
    unique_targets = df[['Target_AA']].drop_duplicates().reset_index(drop=True)
    
    # Generate IDs directly in the DataFrames
    unique_drugs['Drug_ID'] = [f"D{i:06d}" for i in range(1, len(unique_drugs) + 1)]
    unique_targets['Target_ID'] = [f"T{i:06d}" for i in range(1, len(unique_targets) + 1)]
    
    if verbose:
        print(f"Generated {len(unique_drugs)} unique drug IDs and {len(unique_targets)} unique target IDs")
    
    return unique_drugs, unique_targets


def add_potential_ids(
    unique_drugs: pd.DataFrame, 
    unique_targets: pd.DataFrame, 
    dataset_names: List[str] = ["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "KIBA", "Metz"],
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add potential IDs from all datasets to the unique drugs and targets dataframes.
    
    Args:
        unique_drugs: DataFrame containing unique drugs with Drug_SMILES and Drug_ID columns
        unique_targets: DataFrame containing unique targets with Target_AA and Target_ID columns
        dataset_names: List of dataset names to fetch IDs from (default all five datasets)
        verbose: Whether to print additional information
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated dataframes with added Potential_ID columns
    """
    if verbose:
        print("Adding potential IDs from all datasets...")
    
    # Create a cache key based on the input data and dataset names
    drug_smiles_list = sorted(unique_drugs['Drug_SMILES'].tolist())
    target_aa_list = sorted(unique_targets['Target_AA'].tolist())
    dataset_names_sorted = sorted(dataset_names)
    
    # Create a cache key
    cache_key = f"{','.join(drug_smiles_list)}|{','.join(target_aa_list)}|{','.join(dataset_names_sorted)}"
    
    # Check if we have cached results for this input
    if cache_key in potential_ids_cache:
        if verbose:
            print("Using cached potential IDs...")
        
        cached_data = potential_ids_cache[cache_key]
        
        # Convert the cached string sets back to actual sets
        drug_potential_ids = {smiles: set(ids) for smiles, ids in cached_data['drug_potential_ids'].items()}
        target_potential_ids = {aa: set(ids) for aa, ids in cached_data['target_potential_ids'].items()}
        
        # Add potential IDs to dataframes
        unique_drugs['Potential_ID'] = unique_drugs['Drug_SMILES'].map(drug_potential_ids)
        unique_targets['Potential_ID'] = unique_targets['Target_AA'].map(target_potential_ids)
        
        if verbose:
            # Count total potential IDs
            total_drug_ids = sum(len(ids) for ids in drug_potential_ids.values())
            total_target_ids = sum(len(ids) for ids in target_potential_ids.values())
            
            # Count drugs/targets with at least one potential ID
            drugs_with_ids = sum(1 for ids in drug_potential_ids.values() if len(ids) > 0)
            targets_with_ids = sum(1 for ids in target_potential_ids.values() if len(ids) > 0)
            
            print(f"Added {total_drug_ids} potential drug IDs to {drugs_with_ids}/{len(unique_drugs)} drugs")
            print(f"Added {total_target_ids} potential target IDs to {targets_with_ids}/{len(unique_targets)} targets")
        
        return unique_drugs, unique_targets
    
    # Initialize dictionaries to store potential IDs
    drug_potential_ids = {smiles: set() for smiles in unique_drugs['Drug_SMILES']}
    target_potential_ids = {aa: set() for aa in unique_targets['Target_AA']}
    
    # Fetch IDs from all datasets
    for dataset_name in dataset_names:
        try:
            if verbose:
                print(f"Fetching IDs from {dataset_name}...")
            
            drug_ids, target_ids = load_dataset_ids(dataset_name, verbose=False)
            
            # Add IDs to potential IDs dictionaries if the SMILES/AA is in our dataset
            for smiles, id_val in drug_ids.items():
                if smiles in drug_potential_ids:
                    drug_potential_ids[smiles].add(id_val)
            
            for aa, id_val in target_ids.items():
                if aa in target_potential_ids:
                    target_potential_ids[aa].add(id_val)
                    
        except Exception as e:
            if verbose:
                print(f"Error loading IDs from {dataset_name}: {str(e)}")
    
    # Add potential IDs to dataframes
    unique_drugs['Potential_ID'] = unique_drugs['Drug_SMILES'].map(drug_potential_ids)
    unique_targets['Potential_ID'] = unique_targets['Target_AA'].map(target_potential_ids)
    
    # Cache the results
    # Convert sets to lists for JSON serialization
    cache_data = {
        'drug_potential_ids': {smiles: list(ids) for smiles, ids in drug_potential_ids.items()},
        'target_potential_ids': {aa: list(ids) for aa, ids in target_potential_ids.items()}
    }
    potential_ids_cache[cache_key] = cache_data
    save_potential_ids_cache()
    
    if verbose:
        # Count total potential IDs
        total_drug_ids = sum(len(ids) for ids in drug_potential_ids.values())
        total_target_ids = sum(len(ids) for ids in target_potential_ids.values())
        
        # Count drugs/targets with at least one potential ID
        drugs_with_ids = sum(1 for ids in drug_potential_ids.values() if len(ids) > 0)
        targets_with_ids = sum(1 for ids in target_potential_ids.values() if len(ids) > 0)
        
        print(f"Added {total_drug_ids} potential drug IDs to {drugs_with_ids}/{len(unique_drugs)} drugs")
        print(f"Added {total_target_ids} potential target IDs to {targets_with_ids}/{len(unique_targets)} targets")
    
    return unique_drugs, unique_targets


def annotate_drugs(drugs: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Annotate all unique drugs in DTI dataset. Without cached results, this will take a while.

    Args:
        drugs: DataFrame containing unique drugs with Drug_SMILES and Drug_ID columns
        verbose: Whether to print additional information

    Returns:
        DataFrame containing unique drugs with columns:
            - Drug_SMILES
            - Drug_ID
            - Drug_InChIKey
            - Drug_Valid
    """
    if verbose:
        print("Annotating drugs with potential IDs...")
    
    # Create a copy of the dataframe to avoid modifying the original
    annotated_drugs = drugs.copy()
    
    # Initialize new columns
    annotated_drugs['Drug_InChIKey'] = None
    annotated_drugs['Drug_Valid'] = False
    
    # Track progress
    total_drugs = len(annotated_drugs)
    valid_count = 0
    
    if verbose:
        print(f"Processing {total_drugs} drugs...")
        # Check if any drugs have potential IDs
        drugs_with_ids = sum(1 for ids in annotated_drugs.get('Potential_ID', []) if isinstance(ids, set) and len(ids) > 0)
        print(f"Found {drugs_with_ids} drugs with potential IDs")
    
    # Process each drug
    for idx, row in annotated_drugs.iterrows():
        # Get the SMILES and potential IDs
        smiles = row['Drug_SMILES']
        potential_ids = row.get('Potential_ID', set())
        
        # If potential_ids is not a set (e.g., it's NaN), convert to empty set
        if not isinstance(potential_ids, set):
            potential_ids = set()
        
        # Annotate the drug
        annotation = annotate_drug(smiles, potential_ids, verbose=verbose)
        
        # Update the dataframe with annotation results
        annotated_drugs.at[idx, 'Drug_SMILES'] = annotation.smiles  # Update with canonical SMILES
        annotated_drugs.at[idx, 'Drug_InChIKey'] = annotation.inchikey
        annotated_drugs.at[idx, 'Drug_Valid'] = annotation.valid
        
        if annotation.valid:
            valid_count += 1
        
        # Print progress every 100 drugs if verbose
        if verbose and (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_drugs} drugs ({valid_count} valid so far)")
        
        # Save cache periodically to avoid losing data if the process is interrupted
        if (idx + 1) % 500 == 0:
            save_drug_cache()
    
    # Final save of the API response caches
    save_drug_cache()
    
    if verbose:
        # RDLogger.EnableLog('rdApp.*')
        print(f"Annotation complete: {valid_count}/{total_drugs} drugs are valid")
        print(f"Added InChIKeys to {sum(annotated_drugs['Drug_InChIKey'].notna())} drugs")
    
    return annotated_drugs


def annotate_targets(targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Annotate all unique targets in DTI dataset. Without cached results, this will take a while.

    Args:
        targets: DataFrame containing unique targets with Target_AA and Target_ID columns
        verbose: Whether to print additional information

    Returns:
        DataFrame containing unique targets with columns:
            - Target_AA
            - Target_ID
            - Target_Valid
            - Target_DNA
            - Target_UniProt_ID
            - Target_Gene_ID
            - Target_DNA_Similarity
            - Target_UniProt_Source
            - Target_Gene_Source
            - Target_DNA_Source
    """
    if verbose:
        print("Annotating targets with potential IDs...")
    
    # Create a copy of the dataframe to avoid modifying the original
    annotated_targets = targets.copy()
    
    # Initialize new columns
    annotated_targets['Target_DNA'] = None
    annotated_targets['Target_UniProt_ID'] = None
    annotated_targets['Target_Gene_ID'] = None
    annotated_targets['Target_Valid'] = False
    annotated_targets['Target_DNA_Similarity'] = 0.0
    annotated_targets['Target_UniProt_Source'] = None
    annotated_targets['Target_Gene_Source'] = None
    annotated_targets['Target_DNA_Source'] = None
    
    # Track progress
    total_targets = len(annotated_targets)
    valid_count = 0
    
    if verbose:
        print(f"Processing {total_targets} targets...")
        # Check if any targets have potential IDs
        targets_with_ids = sum(1 for ids in annotated_targets.get('Potential_ID', []) if isinstance(ids, set) and len(ids) > 0)
        print(f"Found {targets_with_ids} targets with potential IDs")
    
    # Process each target
    for idx, row in annotated_targets.iterrows():
        # Get the amino acid sequence and potential IDs
        aa_sequence = row['Target_AA']
        potential_ids = row.get('Potential_ID', set())
        
        # If potential_ids is not a set (e.g., it's NaN), convert to empty set
        if not isinstance(potential_ids, set):
            potential_ids = set()
        
        # Annotate the target
        annotation = annotate_target(aa_sequence, potential_ids, verbose=verbose)
        
        # Update the dataframe with annotation results
        annotated_targets.at[idx, 'Target_DNA'] = annotation.dna_sequence
        annotated_targets.at[idx, 'Target_UniProt_ID'] = annotation.uniprot_id
        annotated_targets.at[idx, 'Target_Gene_name'] = annotation.gene_name
        annotated_targets.at[idx, 'Target_RefSeq_ID'] = annotation.refseq_id
        annotated_targets.at[idx, 'Target_Valid'] = annotation.valid
        annotated_targets.at[idx, 'Target_DNA_Similarity'] = annotation.similarity
        
        if annotation.valid:
            valid_count += 1
        
        # Print progress every 10 targets if verbose
        if verbose and (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{total_targets} targets ({valid_count} valid so far)")
        
        # Save cache periodically to avoid losing data if the process is interrupted
        if (idx + 1) % 50 == 0:
            save_target_cache()
    
    # Final save of the API response caches
    save_target_cache()
    
    if verbose:
        print(f"Annotation complete: {valid_count}/{total_targets} targets are valid")
        print(f"Added DNA sequences to {sum(annotated_targets['Target_DNA'].notna())} targets")
        print(f"Added UniProt IDs to {sum(annotated_targets['Target_UniProt_ID'].notna())} targets")
        print(f"Added Gene IDs to {sum(annotated_targets['Target_Gene_name'].notna())} targets")
        print(f"Added RefSeq IDs to {sum(annotated_targets['Target_RefSeq_ID'].notna())} targets")

    return annotated_targets

