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
from tqdm import tqdm
from rdkit import Chem
import hashlib
import traceback

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
        # Ensure directory exists
        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Make a backup of the existing cache file if it exists
        if POTENTIAL_IDS_CACHE_FILE.exists():
            import shutil
            backup_file = POTENTIAL_IDS_CACHE_FILE.with_suffix('.json.bak')
            shutil.copy2(POTENTIAL_IDS_CACHE_FILE, backup_file)
        
        # Save the cache
        with open(POTENTIAL_IDS_CACHE_FILE, 'w') as f:
            json.dump(potential_ids_cache, f)
            
        print(f"Successfully saved potential IDs cache to {POTENTIAL_IDS_CACHE_FILE}")
    except Exception as e:
        import traceback
        print(f"Failed to save potential IDs cache: {e}")
        print(traceback.format_exc())


def canonicalize_smiles(smiles_list: List[str], verbose: bool = False) -> Dict[str, str]:
    """
    Canonicalize a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings to canonicalize
        verbose: Whether to print additional information

    Returns:
        Dictionary mapping original SMILES strings to their canonical forms
    """
    if verbose:
        print(f"Canonicalizing {len(smiles_list)} SMILES strings...")
    
    result = {}
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                result[smiles] = canonical_smiles
            else:
                if verbose:
                    print(f"Warning: Could not parse SMILES: {smiles}")
                # Use original SMILES as fallback
                result[smiles] = smiles
        except Exception as e:
            if verbose:
                print(f"Error canonicalizing SMILES {smiles}: {str(e)}")
            # Use original SMILES as fallback
            result[smiles] = smiles
    
    if verbose:
        print(f"Canonicalized {len(result)} SMILES strings")
    
    return result


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
    
    try:
        # Load dataset
        if name == "Metz":
            try:
                drug_ids, target_ids = load_metz_ids()
                canonical_dict = canonicalize_smiles(list(drug_ids.keys()), verbose=verbose)
                drug_ids = {canonical_dict[original_smiles]: drug_ids[original_smiles] 
                            for original_smiles in drug_ids.keys() 
                            if original_smiles in canonical_dict}
            except Exception as e:
                if verbose:
                    print(f"Error processing Metz dataset: {str(e)}")
                    print(traceback.format_exc())
                return {}, {}
        else:
            try:
                data = DTI(name=name, path=str(SOURCE_DIR))
                df = data.get_data()
                df = df.drop_duplicates(subset=['Drug', 'Target'])
            except Exception as e:
                if verbose:
                    print(f"Error loading dataset {name}: {str(e)}")
                    print(traceback.format_exc())
                return {}, {}

            # Clean and convert IDs properly
            def clean_id(id_val):
                try:
                    if pd.isna(id_val):
                        return None
                    try:
                        float_val = float(id_val)
                        if pd.isna(float_val):
                            return None
                        if float_val.is_integer():
                            return str(int(float_val))
                        return str(float_val)
                    except (ValueError, TypeError):
                        return str(id_val)
                except Exception:
                    return None
            
            try:
                df['Drug_ID'] = df['Drug_ID'].apply(clean_id)
                df['Target_ID'] = df['Target_ID'].apply(clean_id)
            except Exception as e:
                if verbose:
                    print(f"Error cleaning IDs in {name}: {str(e)}")
                    print(traceback.format_exc())
                return {}, {}
            
            # Filter out entries with None/NaN IDs
            drug_ids = {drug: drug_id for drug, drug_id in zip(df['Drug'], df['Drug_ID']) 
                      if drug_id is not None}
            target_ids = {target: target_id for target, target_id in zip(df['Target'], df['Target_ID']) 
                         if target_id is not None}

            # Canonicalize SMILES strings
            try:
                unique_smiles = list(drug_ids.keys())
                canonical_dict = canonicalize_smiles(unique_smiles, verbose=verbose)
                # Create new drug_ids dict with canonical SMILES as keys
                drug_ids = {canonical_dict[smiles]: drug_id 
                           for smiles, drug_id in drug_ids.items() 
                           if smiles in canonical_dict}
            except Exception as e:
                if verbose:
                    print(f"Error canonicalizing SMILES in {name}: {str(e)}")
                    print(traceback.format_exc())

        if verbose:
            print(f"Gathered {len(drug_ids)} unique drugs and {len(target_ids)} unique targets from {name}")

        return drug_ids, target_ids
    
    except Exception as e:
        if verbose:
            print(f"Unexpected error loading {name}: {str(e)}")
            print(traceback.format_exc())
        return {}, {}


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
    
    # Create a hash-based cache key using the counts and first few items
    
    # Get basic stats for drugs and targets
    n_drugs = len(unique_drugs)
    n_targets = len(unique_targets)
    
    # Get a sample of drugs and targets for the hash (first 5 and last 5)
    if n_drugs > 0:
        drug_sample = unique_drugs['Drug_SMILES'].iloc[:5].tolist()
        if n_drugs > 10:
            drug_sample.extend(unique_drugs['Drug_SMILES'].iloc[-5:].tolist())
    else:
        drug_sample = []
        
    if n_targets > 0:
        target_sample = unique_targets['Target_AA'].iloc[:5].tolist()
        if n_targets > 10:
            target_sample.extend(unique_targets['Target_AA'].iloc[-5:].tolist())
    else:
        target_sample = []
    
    # Create a string with counts, samples, and dataset names
    hash_input = f"{n_drugs},{n_targets},{','.join(sorted(dataset_names))},{','.join(drug_sample)},{','.join(target_sample)}"
    
    # Generate a hash for the cache key
    hash_obj = hashlib.md5(hash_input.encode())
    cache_key = hash_obj.hexdigest()
    
    if verbose:
        print(f"Generated cache key: {cache_key} for {n_drugs} drugs and {n_targets} targets")
    
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
    
    # Fetch IDs from all datasets - use tqdm for progress tracking
    dataset_iterator = tqdm(dataset_names, desc="Fetching IDs from datasets") if verbose else dataset_names
    for dataset_name in dataset_iterator:
        try:
            if verbose:
                print(f"Fetching IDs from {dataset_name}...")
            
            drug_ids, target_ids = load_dataset_ids(dataset_name, verbose=verbose)
            
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
    
    # Process each drug - use tqdm for progress tracking regardless of verbose setting
    iterator = tqdm(annotated_drugs.iterrows(), total=total_drugs, desc="Annotating drugs") if not verbose else annotated_drugs.iterrows()
    for idx, row in iterator:
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
    annotated_targets['Target_Gene_name'] = None
    annotated_targets['Target_RefSeq_ID'] = None
    annotated_targets['Target_Valid'] = False
    annotated_targets['Target_DNA_Similarity'] = 0.0
    
    # Track progress
    total_targets = len(annotated_targets)
    valid_count = 0
    
    if verbose:
        print(f"Processing {total_targets} targets...")
        # Check if any targets have potential IDs
        targets_with_ids = sum(1 for ids in annotated_targets.get('Potential_ID', []) if isinstance(ids, set) and len(ids) > 0)
        print(f"Found {targets_with_ids} targets with potential IDs")
    
    # Process each target - use tqdm for progress tracking regardless of verbose setting
    iterator = tqdm(annotated_targets.iterrows(), total=total_targets, desc="Annotating targets") if not verbose else annotated_targets.iterrows()
    for idx, row in iterator:
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


def annotate_dti(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Annotate a drug-target interaction dataset.

    Args:
        df: DataFrame containing drug-target interactions with Drug_SMILES and Target_AA columns
        verbose: Whether to print additional information

    Returns:
        Annotated DataFrame with the following columns:
        - Drug_ID: Unique ID for the drug
        - Drug_InChIKey: InChI key for the drug
        - Drug_SMILES: SMILES string for the drug
        - Target_ID: Unique ID for the target
        - Target_UniProt_ID: UniProt ID for the target
        - Target_Gene_name: Gene name for the target
        - Target_RefSeq_ID: RefSeq ID for the target
        - Target_AA: Amino acid sequence for the target
        - Target_DNA: DNA sequence for the target
        - Y: Binary interaction indicator
        - Y_*: Interaction values (e.g., Y_pKd, Y_KIBA, Y_pKi)
        - in_*: Dataset indicators (e.g., in_DAVIS, in_KIBA)
    """
    print("Starting annotation process...")
    print(f"Input dataset contains {len(df)} drug-target interactions")
    
    # Step 1: Extract unique drugs and targets
    print("\nStep 1: Extracting unique drugs and targets...")
    unique_drugs = df[['Drug_SMILES']].drop_duplicates().reset_index(drop=True)
    unique_targets = df[['Target_AA']].drop_duplicates().reset_index(drop=True)
    
    print(f"Found {len(unique_drugs)} unique drugs and {len(unique_targets)} unique targets")
    
    # Step 2: Add potential IDs from all datasets
    print("\nStep 2: Adding potential IDs...")
    unique_drugs, unique_targets = add_potential_ids(unique_drugs, unique_targets, verbose=verbose)
    
    # Step 3: Annotate drugs
    print("\nStep 3: Annotating drugs...")
    df_drugs = annotate_drugs(unique_drugs, verbose=verbose)
    
    # Step 4: Annotate targets
    print("\nStep 4: Annotating targets...")
    df_targets = annotate_targets(unique_targets, verbose=verbose)
    
    # Step 5: Filter for valid drugs and targets
    print("\nStep 5: Filtering for valid drugs and targets...")
    # Fill NA values in the validity columns with False to avoid filtering errors
    df_drugs['Drug_Valid'] = df_drugs['Drug_Valid'].fillna(False)
    df_targets['Target_Valid'] = df_targets['Target_Valid'].fillna(False)
    valid_drugs = df_drugs[df_drugs['Drug_Valid']]
    valid_targets = df_targets[df_targets['Target_Valid']]
    
    print(f"Found {len(valid_drugs)}/{len(df_drugs)} valid drugs")
    print(f"Found {len(valid_targets)}/{len(df_targets)} valid targets")

    # Print sample invalid entries with formatting
    print("\nSample invalid drugs (max 10):")
    print(df_drugs[~df_drugs['Drug_Valid']].head(10).to_markdown(index=False))
    
    print("\nSample invalid targets (max 10):") 
    print(df_targets[~df_targets['Target_Valid']].head(10).to_markdown(index=False))
    print("\n")
    
    # Step 6: Generate unique IDs for valid drugs and targets
    print("\nStep 6: Generating unique IDs for valid drugs and targets...")
    
    # Generate IDs directly in the DataFrames - only for valid drugs and targets
    valid_drugs['Drug_ID'] = [f"D{i:06d}" for i in range(1, len(valid_drugs) + 1)]
    valid_targets['Target_ID'] = [f"T{i:06d}" for i in range(1, len(valid_targets) + 1)]
    
    print(f"Generated {len(valid_drugs)} unique drug IDs and {len(valid_targets)} unique target IDs")
    
    # Step 7: Create a mapping between original data and annotated data
    # Check for duplicate SMILES strings in valid_drugs
    if valid_drugs['Drug_SMILES'].duplicated().any():
        print(f"Warning: Found {valid_drugs['Drug_SMILES'].duplicated().sum()} duplicate drug SMILES strings")
        # Keep only the first occurrence of each SMILES string
        valid_drugs = valid_drugs.drop_duplicates(subset=['Drug_SMILES'], keep='first')
        print(f"After removing duplicates: {len(valid_drugs)} unique drugs")
    
    # Check for duplicate AA sequences in valid_targets
    if valid_targets['Target_AA'].duplicated().any():
        print(f"Warning: Found {valid_targets['Target_AA'].duplicated().sum()} duplicate target sequences")
        # Keep only the first occurrence of each AA sequence
        valid_targets = valid_targets.drop_duplicates(subset=['Target_AA'], keep='first')
        print(f"After removing duplicates: {len(valid_targets)} unique targets")
    
    # Now create the mapping with unique indices
    drug_map = valid_drugs.set_index('Drug_SMILES')
    target_map = valid_targets.set_index('Target_AA')
    
    # Step 8: Filter the original dataset to only include valid drugs and targets
    df_filtered = df[
        df['Drug_SMILES'].isin(valid_drugs['Drug_SMILES']) &
        df['Target_AA'].isin(valid_targets['Target_AA'])
    ]
    
    filtered_interactions = len(df_filtered)
    print(f"Filtered dataset contains {filtered_interactions}/{len(df)} interactions")
    print(f"Retention rate: {filtered_interactions/len(df)*100:.2f}%")
    
    # Step 9: Create the final dataset by joining the filtered data with the annotations
    # Add drug information
    df_annotated = df_filtered.copy()
    df_annotated['Drug_ID'] = df_annotated['Drug_SMILES'].map(drug_map['Drug_ID'])
    df_annotated['Drug_InChIKey'] = df_annotated['Drug_SMILES'].map(drug_map['Drug_InChIKey'])
    
    # Add target information
    df_annotated['Target_ID'] = df_annotated['Target_AA'].map(target_map['Target_ID'])
    df_annotated['Target_UniProt_ID'] = df_annotated['Target_AA'].map(target_map['Target_UniProt_ID'])
    df_annotated['Target_Gene_name'] = df_annotated['Target_AA'].map(target_map['Target_Gene_name'])
    df_annotated['Target_RefSeq_ID'] = df_annotated['Target_AA'].map(target_map['Target_RefSeq_ID'])
    df_annotated['Target_DNA'] = df_annotated['Target_AA'].map(target_map['Target_DNA'])
    
    # Reorder columns to match the desired output format
    columns = [
        'Drug_ID', 'Drug_InChIKey', 'Drug_SMILES',
        'Target_ID', 'Target_UniProt_ID', 'Target_Gene_name', 'Target_RefSeq_ID', 'Target_AA', 'Target_DNA',
        'Y'
    ]
    
    # Add value columns
    value_columns = [col for col in df_annotated.columns if col.startswith('Y_')]
    columns.extend(value_columns)
    
    # Add dataset indicator columns
    indicator_columns = [col for col in df_annotated.columns if col.startswith('in_')]
    columns.extend(indicator_columns)
    
    # Reorder the columns
    df_annotated = df_annotated[columns]
    
    if verbose:
        print(f"\nFinal annotated dataset contains {len(df_annotated)} interactions")
        print(f"Involving {df_annotated['Drug_ID'].nunique()} unique drugs and {df_annotated['Target_ID'].nunique()} unique targets")
    
    return df_annotated

