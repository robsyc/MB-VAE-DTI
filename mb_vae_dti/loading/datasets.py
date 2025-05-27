"""
Dataset loading and processing functionality for MB-VAE-DTI.

This module provides functions for loading standard DTI datasets (DAVIS, KIBA, BindingDB),
as well as the Metz dataset. It also includes functionality for merging datasets,
filtering based on molecular properties, and saving processed data.
"""

import pandas as pd
from typing import Literal, List, Dict, Union
from rdkit import Chem
from rdkit.Chem import Descriptors
from tdc.multi_pred import DTI
from pathlib import Path

# Constants for filtering
MAX_N_HEAVY_ATOMS = 64
FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P'}
MAX_MOL_WEIGHT = 1500
MAX_AA_SEQ_LEN = 1280

# Thresholds for binarization
pKd_THRESHOLD = 7.0
pKi_THRESHOLD = 7.6
KIBA_THRESHOLD = 12.1

# Define paths
DATA_DIR = Path("data")
SOURCE_DIR = DATA_DIR / "source"
PROCESSED_DIR = DATA_DIR / "processed"

# Global cache for molecule filtering results to avoid redundant computation
MOLECULE_FILTER_CACHE = {}

# Ensure directories exist
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_metz() -> pd.DataFrame:
    """
    Load the Metz dataset from a CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the Metz dataset with columns:
            - SMILES: SMILES representation of the drug
            - ProteinSequence: Protein sequence of the target
            - Ki: Binding affinity value
            
    Raises:
        FileNotFoundError: If the Metz.csv file is not found in the data directory
    """
    metz_path = SOURCE_DIR / "Metz.csv"
    try:
        # Note: even though the column's name is Ki, it's actually pKi!
        return pd.read_csv(metz_path, usecols=['SMILES', 'ProteinSequence', 'Ki'])
    except FileNotFoundError:
        print(f'To run the Metz dataset, you need to download it from Kaggle and place it in {metz_path}')
        print("https://www.kaggle.com/datasets/blk1804/metz-drug-binding-dataset")
        raise FileNotFoundError(f"Metz dataset not found at {metz_path}")


def load_dataset(
        name: Literal["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "KIBA", "Metz"], 
        verbose: bool = False,
        apply_filters: bool = False
    ) -> pd.DataFrame:
    """
    Load a DTI dataset and transform it into a standardized format.
    
    Args:
        name: Name of the dataset to load
        verbose: Whether to print additional information
        apply_filters: Whether to apply molecular and protein filters
        
    Returns:
        pd.DataFrame: DataFrame with standardized columns:
            - Drug_SMILES: SMILES representation of the drug (canonicalized)
            - Target_AA: Protein sequence of the target
            - Y: Binary interaction indicator (True if bound)
            - Y_{value}: Interaction value (e.g., Y_pKd, Y_pKi, Y_KIBA)
    """
    if verbose:
        print(f"\n📂 Loading {name} dataset...")
    
    # Load dataset
    if name == "Metz":
        data = load_metz()
    else:
        # Use SOURCE_DIR as the path for TDC to store/load datasets
        data = DTI(name=name, path=str(SOURCE_DIR))
        
        # Apply transformations for binding affinity datasets
        if name in ["DAVIS", "BindingDB_Kd", "BindingDB_Ki"]:
            data.convert_to_log(form='binding')
            data.harmonize_affinities(mode='mean')
    
    # Convert to DataFrame if needed
    df = data.get_data() if not isinstance(data, pd.DataFrame) else data
    
    # Standardize column names
    if name == "Metz":
        df.rename(columns={
            'SMILES': 'Drug_SMILES',
            'ProteinSequence': 'Target_AA',
            'Ki': 'Y_pKi'
        }, inplace=True)
    else:
        # Standardize TDC dataset columns
        df.rename(columns={
            'Drug': 'Drug_SMILES',
            'Target': 'Target_AA'
        }, inplace=True)
        if name == "DAVIS" or name == "BindingDB_Kd":
            df.rename(columns={'Y': 'Y_pKd'}, inplace=True)
        elif name == "BindingDB_Ki":
            df.rename(columns={'Y': 'Y_pKi'}, inplace=True)
        elif name == "KIBA":
            df.rename(columns={'Y': 'Y_KIBA'}, inplace=True)
    
    # Drop rows with missing values in key columns
    rows_before = len(df)
    df = df.dropna(subset=['Drug_SMILES', 'Target_AA'])
    if verbose and len(df) < rows_before:
        print(f"  - Dropped {rows_before - len(df):,} rows with missing values")
    
    # Add binary interaction column based on threshold
    thresholds = {'Y_pKd': pKd_THRESHOLD, 'Y_pKi': pKi_THRESHOLD, 'Y_KIBA': KIBA_THRESHOLD}
    for col, threshold in thresholds.items():
        if col in df.columns:
            df['Y'] = df[col] >= threshold
            break
    
    # Remove duplicate drug-target pairs (keeping the one with the highest affinity)
    if len(df) > df[['Drug_SMILES', 'Target_AA']].drop_duplicates().shape[0]:
        if verbose:
            print(f"  - Found duplicate drug-target pairs in {name}")
        
        value_cols = [col for col in df.columns if col.startswith('Y_')]
        value_col = value_cols[0]
        idx = df.groupby(['Drug_SMILES', 'Target_AA'])[value_col].idxmax()
        df = df.loc[idx]
    
    initial_rows = len(df)
    initial_drugs = df['Drug_SMILES'].nunique()
    initial_targets = df['Target_AA'].nunique()
    
    if verbose:
        print(f"  - Loaded {initial_rows:,} interactions")
        print(f"  - Unique drugs: {initial_drugs:,}")
        print(f"  - Unique targets: {initial_targets:,}")
    
    # Apply filters if requested
    if apply_filters:
        if verbose:
            print(f"\n🔍 Applying filters to {name} dataset...")
        df = filter_dataset(df, verbose=verbose)
    
    return df


def filter_molecules(smiles_list: List[str], verbose: bool = False) -> Dict[str, Dict[str, Union[bool, float, str]]]:
    """
    Perform comprehensive filtering on a list of SMILES strings.
    
    Applies multiple filters:
    1. Canonicalization (with isomericSmiles=False)
    2. Heavy atom count (MAX_N_HEAVY_ATOMS)
    3. Molecular weight (MAX_MOL_WEIGHT)
    4. Allowed atom types (FILTER_ATOMS)
    5. No charged atoms
    6. No multi-component molecules (no "." in SMILES)
    
    Args:
        smiles_list: List of SMILES strings to filter
        verbose: Whether to print progress information
        
    Returns:
        Dict[str, Dict]: Dictionary mapping SMILES to their filter results and properties
    """
    global MOLECULE_FILTER_CACHE
    
    # Filter out SMILES already in cache
    uncached_smiles = [s for s in smiles_list if s not in MOLECULE_FILTER_CACHE]
    
    if verbose:
        cache_hits = len(smiles_list) - len(uncached_smiles)
        if cache_hits > 0:
            print(f"ℹ️ Using cached results for {cache_hits:,} molecules")
        print(f"⏳ Filtering {len(uncached_smiles):,} new molecules...")
    
    # Process uncached SMILES
    if not uncached_smiles:
        return {s: MOLECULE_FILTER_CACHE[s] for s in smiles_list}
    
    results = {}
    
    # Counters for verbose reporting
    fail_counts = {
        "parsing": 0,
        "multi_component": 0,
        "mol_weight": 0,
        "atom_charge": 0,
        "atom_type": 0,
        "heavy_atoms": 0
    }
    
    # Track canonical SMILES for deduplication analysis
    canonical_smiles_set = set()
    canonical_to_original = {}
    
    for smiles in uncached_smiles:
        try:
            # Initialize result dictionary
            result = {
                "canonical_smiles": smiles,  # Default to original in case of failure
                "passes_filter": False,
                "heavy_atom_count": float('inf'),
                "mol_weight": float('inf'),
                "fail_reason": None
            }
            
            # First try to parse the SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fail_counts["parsing"] += 1
                result["fail_reason"] = "invalid_smiles"
                MOLECULE_FILTER_CACHE[smiles] = result
                continue
            
            # Convert to canonical SMILES with no stereochemistry
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            result["canonical_smiles"] = canonical_smiles
            
            # Track which original SMILES map to this canonical form
            if canonical_smiles not in canonical_to_original:
                canonical_to_original[canonical_smiles] = []
            canonical_to_original[canonical_smiles].append(smiles)
            
            # Recreate molecule from canonical SMILES
            mol = Chem.MolFromSmiles(canonical_smiles)
            
            # Check for multi-component molecules
            if "." in canonical_smiles:
                fail_counts["multi_component"] += 1
                result["fail_reason"] = "multi_component"
                MOLECULE_FILTER_CACHE[smiles] = result
                continue
            
            # Check molecular weight
            mol_weight = Descriptors.MolWt(mol)
            result["mol_weight"] = mol_weight
            if mol_weight >= MAX_MOL_WEIGHT:
                fail_counts["mol_weight"] += 1
                result["fail_reason"] = "mol_weight_too_high"
                MOLECULE_FILTER_CACHE[smiles] = result
                continue
            
            # Count heavy atoms
            heavy_atom_count = Descriptors.HeavyAtomCount(mol)
            result["heavy_atom_count"] = heavy_atom_count
            if heavy_atom_count > MAX_N_HEAVY_ATOMS:
                fail_counts["heavy_atoms"] += 1
                result["fail_reason"] = "too_many_heavy_atoms"
                MOLECULE_FILTER_CACHE[smiles] = result
                continue
            
            # Check atom types and charges
            atom_type_ok = True
            charge_ok = True
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() != 0:
                    charge_ok = False
                    break
                if atom.GetSymbol() not in FILTER_ATOMS:
                    atom_type_ok = False
                    break
            
            if not charge_ok:
                fail_counts["atom_charge"] += 1
                result["fail_reason"] = "charged_atoms"
                MOLECULE_FILTER_CACHE[smiles] = result
                continue
            
            if not atom_type_ok:
                fail_counts["atom_type"] += 1
                result["fail_reason"] = "unsupported_atom_types"
                MOLECULE_FILTER_CACHE[smiles] = result
                continue
            
            # If we get here, molecule passes all filters
            result["passes_filter"] = True
            result["fail_reason"] = None
            MOLECULE_FILTER_CACHE[smiles] = result
            
            # Add to set of canonical SMILES that passed
            canonical_smiles_set.add(canonical_smiles)
            
        except Exception as e:
            fail_counts["parsing"] += 1
            result = {
                "canonical_smiles": smiles,
                "passes_filter": False,
                "heavy_atom_count": float('inf'),
                "mol_weight": float('inf'),
                "fail_reason": f"exception: {str(e)}"
            }
            MOLECULE_FILTER_CACHE[smiles] = result
    
    # Prepare results from cache for all requested SMILES
    results = {s: MOLECULE_FILTER_CACHE[s] for s in smiles_list}
    
    if verbose and uncached_smiles:
        # Calculate statistics only for newly processed molecules
        total_passed = sum(1 for s in uncached_smiles if MOLECULE_FILTER_CACHE[s]["passes_filter"])
        total_failed = len(uncached_smiles) - total_passed
        
        print(f"✅ Filtering results for new molecules: {total_passed:,} passed, {total_failed:,} failed")
        if total_failed > 0:
            print("📊 Failure breakdown:")
            for reason, count in fail_counts.items():
                if count > 0:
                    print(f"  - {reason}: {count:,} molecules")
                 
        # Analyze canonicalization effect
        duplicates_count = sum(len(orig_list) - 1 for orig_list in canonical_to_original.values() if len(orig_list) > 1)
        if duplicates_count > 0:
            print(f"ℹ️ Found {duplicates_count:,} molecules that canonicalize to the same SMILES as other molecules")
            print(f"  - Unique canonical SMILES that passed filters: {len(canonical_smiles_set):,}")
    
    return results


def filter_dataset(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Filter a dataset based on molecular properties.
    
    Args:
        df: DataFrame to filter
        verbose: Whether to print additional information
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    initial_rows = len(df)
    initial_drugs = df['Drug_SMILES'].nunique()
    initial_targets = df['Target_AA'].nunique()
    
    if verbose:
        print(f"  - Initial rows: {initial_rows:,}")
        print(f"  - Unique drugs: {initial_drugs:,}")
        print(f"  - Unique targets: {initial_targets:,}")
    
    # 1. First filter proteins by sequence length (cheap operation)
    protein_filtered_df = df[df['Target_AA'].str.len() <= MAX_AA_SEQ_LEN]
    
    if verbose and len(protein_filtered_df) < initial_rows:
        lost_rows = initial_rows - len(protein_filtered_df)
        lost_targets = initial_targets - protein_filtered_df['Target_AA'].nunique()
        print(f"  - Filtered out {lost_rows:,} rows ({lost_rows/initial_rows:.2%}) due to protein sequence length")
        print(f"  - Lost {lost_targets:,} unique targets")
    
    # 2. Apply comprehensive filtering to unique drug molecules
    unique_smiles = protein_filtered_df['Drug_SMILES'].unique()
    filter_results = filter_molecules(unique_smiles, verbose=verbose)
    
    # Create a mapping from original SMILES to filtering results
    is_valid = {smiles: result["passes_filter"] for smiles, result in filter_results.items()}
    canonical_map = {smiles: result["canonical_smiles"] for smiles, result in filter_results.items()}
    
    # Apply the filtering
    protein_filtered_df['passes_filter'] = protein_filtered_df['Drug_SMILES'].map(is_valid)
    filtered_df = protein_filtered_df[protein_filtered_df['passes_filter'] == True]
    
    # Count unique SMILES before canonicalization
    pre_canon_drug_count = filtered_df['Drug_SMILES'].nunique()
    
    # Update SMILES to canonical form
    filtered_df['Drug_SMILES'] = filtered_df['Drug_SMILES'].map(canonical_map)
    
    # After canonicalization, we might have duplicate rows, so we need to resolve them
    if len(filtered_df) > filtered_df[['Drug_SMILES', 'Target_AA']].drop_duplicates().shape[0]:
        if verbose:
            print("  - Resolving duplicate drug-target pairs after canonicalization")
        
        # Group by drug-target pairs and keep the row with the highest affinity value
        value_cols = [col for col in filtered_df.columns if col.startswith('Y_')]
        value_col = value_cols[0]
        idx = filtered_df.groupby(['Drug_SMILES', 'Target_AA'])[value_col].idxmax()
        filtered_df = filtered_df.loc[idx]
    
    # Count unique canonical SMILES
    post_canon_drug_count = filtered_df['Drug_SMILES'].nunique()
    
    # Remove temporary columns
    filtered_df = filtered_df.drop(columns=['passes_filter'])
    
    if verbose:
        final_rows = len(filtered_df)
        final_drugs = filtered_df['Drug_SMILES'].nunique()
        final_targets = filtered_df['Target_AA'].nunique()
        
        lost_rows = initial_rows - final_rows
        lost_drugs = initial_drugs - final_drugs
        
        print(f"\n✅ Filtering complete:")
        print(f"  - Rows reduced from {initial_rows:,} to {final_rows:,} ({lost_rows/initial_rows:.2%} reduction)")
        print(f"  - Unique drugs reduced from {initial_drugs:,} to {final_drugs:,} ({lost_drugs/initial_drugs:.2%} reduction)")
        print(f"  - Unique targets reduced from {initial_targets:,} to {final_targets:,}")
        
        # Explain canonical SMILES effect
        if pre_canon_drug_count != post_canon_drug_count:
            canon_reduction = pre_canon_drug_count - post_canon_drug_count
            print(f"  - Canonicalization effect: {canon_reduction:,} drugs consolidated")
    
    return filtered_df


def merge_datasets(names: List[str], verbose: bool = False, pre_filtered: bool = False) -> pd.DataFrame:
    """
    Merge multiple DTI datasets into a unified dataset.
    
    Args:
        names: List of dataset names to merge
        verbose: Whether to print additional information
        pre_filtered: Whether the input datasets are already filtered
        
    Returns:
        pd.DataFrame: Merged DataFrame with all interactions from the input datasets
        
    Raises:
        ValueError: If no valid datasets provided
    """
    # Check if KIBA is in the list and reorder to ensure it's processed last for priority
    if "KIBA" in names:
        has_kiba = True
        # Create a new list without KIBA, then append KIBA at the end
        reordered_names = [name for name in names if name != "KIBA"]
        reordered_names.append("KIBA")
        names = reordered_names
        
        if verbose:
            print("ℹ️ KIBA dataset detected. Reordering to prioritize KIBA values in conflicts.")
    else:
        has_kiba = False
        
    # Phase 1: Load and transform individual datasets
    datasets = []
    if verbose:
        print(f"\n🔄 Merging datasets: {', '.join(names)}")
        
    for name in names:
        try:
            # Apply filters before merging if pre_filtered is True
            df = load_dataset(name, verbose=verbose, apply_filters=pre_filtered)
            df[f"in_{name}"] = True
            datasets.append(df)
        except Exception as e:
            if verbose:
                print(f"❌ Failed to load {name}: {str(e)}")
    
    if not datasets:
        raise ValueError("No valid datasets provided")

    # Phase 2: Merge datasets efficiently
    if verbose:
        print(f"\n🔄 Merging {len(datasets)} datasets...")
        
    merged_df = datasets[0]
    
    # Initialize counters for tracking merges and conflicts
    total_merges = 0
    conflicting_merges = 0
    
    for i, df in enumerate(datasets[1:], 1):
        current_dataset_name = names[i]
        
        if verbose:
            print(f"  - Merging {current_dataset_name} ({len(df):,} pairs)...")
            
        # Check if this is the KIBA dataset (which should be the last one if present)
        is_kiba_dataset = current_dataset_name == "KIBA"
        
        # Merge on Drug_SMILES and Target_AA using outer join
        merged_df = pd.merge(
            merged_df, 
            df,
            on=['Drug_SMILES', 'Target_AA'],
            how='outer',
            suffixes=('', '_right')
        )

        # Count pairs that appear in both datasets
        pairs_in_both = merged_df.dropna(subset=['Y', 'Y_right']).shape[0]
        total_merges += pairs_in_both
        
        # Count pairs with conflicting Y values
        if pairs_in_both > 0:
            conflicts = ((merged_df['Y'] == True) & (merged_df['Y_right'] == False)) | \
                        ((merged_df['Y'] == False) & (merged_df['Y_right'] == True))
            current_conflicts = conflicts.sum()
            conflicting_merges += current_conflicts
            
            if verbose and current_conflicts > 0:
                print(f"  - Found {current_conflicts:,} conflicting Y values ({current_conflicts/pairs_in_both:.2%} of overlapping pairs; {current_conflicts/total_merges:.2%} of total merges)")

        # Update binary interaction indicator (logical OR)
        merged_df['Y'] = merged_df['Y'].fillna(False) | merged_df['Y_right'].fillna(False)
        
        # Update measurement columns (e.g., Y_pKd, Y_KIBA)
        value_cols = [col for col in merged_df.columns if (col.startswith('Y_p') or col.startswith('Y_KIBA')) and not col.endswith('_right')]
        for col in value_cols:
            right_col = f"{col}_right"
            if right_col in merged_df.columns:
                # If this is the KIBA dataset, prioritize its values over existing ones
                if is_kiba_dataset:
                    # Keep the KIBA values where they exist, otherwise keep existing values
                    merged_df[col] = merged_df[right_col].fillna(merged_df[col])
                    if verbose and col == "Y_KIBA":
                        print(f"  - Prioritizing KIBA dataset values for column {col}")
                else:
                    # For non-KIBA datasets, take max of the two columns, handling NaN values
                    merged_df[col] = merged_df[[col, right_col]].max(axis=1)

        # Drop all temporary right columns
        right_cols = [col for col in merged_df.columns if col.endswith('_right')]
        merged_df.drop(columns=right_cols, inplace=True)
    
    # Fill missing dataset indicators with False
    for name in names:
        if f"in_{name}" not in merged_df.columns:
            merged_df[f"in_{name}"] = False
        else:
            merged_df[f"in_{name}"] = merged_df[f"in_{name}"].fillna(False)

    # Ensure consistent column ordering
    cols = ['Drug_SMILES', 'Target_AA', 'Y']
    measure_cols = [col for col in merged_df.columns if (col.startswith('Y_p') or col.startswith('Y_KIBA'))]
    indicator_cols = [col for col in merged_df.columns if col.startswith('in_')]
    merged_df = merged_df[cols + measure_cols + indicator_cols]
    
    if verbose:
        total_pairs = len(merged_df)
        unique_drugs = merged_df['Drug_SMILES'].nunique()
        unique_targets = merged_df['Target_AA'].nunique()
        
        print(f"\n✅ Merged dataset contains {total_pairs:,} unique drug-target pairs")
        print(f"  - Unique drugs: {unique_drugs:,}")
        print(f"  - Unique targets: {unique_targets:,}")
        
        if total_merges > 0:
            print(f"  - Total overlapping pairs: {total_merges:,}")
            print(f"  - Conflicting Y-values: {conflicting_merges:,} ({(conflicting_merges/total_merges)*100:.2f}% of overlaps; {conflicting_merges/total_pairs*100:.2f}% of total pairs)")
        else:
            print("  - No overlapping drug-target pairs found")
        
        if has_kiba:
            print("  - Note: KIBA dataset values were prioritized in conflicts")
    
    return merged_df


def save_dataset(df: pd.DataFrame, name: str, verbose: bool = False) -> str:
    """
    Save a dataset to a CSV file.
    
    Args:
        df: DataFrame to save
        name: Name of the dataset
        verbose: Whether to print additional information
        
    Returns:
        str: Path to the saved file
    """
    # Create the output path
    output_path = PROCESSED_DIR / f"{name}.csv"
    
    # Save the DataFrame
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"📥 Saved dataset to {output_path}")
    
    return str(output_path)


def load_or_create_merged_dataset(
    dataset_names: List[str], 
    force_reload: bool = False,
    apply_filters: bool = True,
    filter_before_merge: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load a merged dataset from disk or create it if it doesn't exist.
    
    Args:
        dataset_names: List of dataset names to include
        force_reload: Whether to force reloading the datasets even if a merged file exists
        apply_filters: Whether to apply filters to the dataset
        filter_before_merge: Whether to filter individual datasets before merging (recommended)
        verbose: Whether to print additional information
        
    Returns:
        pd.DataFrame: Merged and optionally filtered DataFrame
    """
    # Generate a filename based on the dataset names and filtering
    # Always sort dataset names for consistent filenames, but note that in the merging process
    # KIBA will be processed last if present (prioritized by merge_datasets function)
    dataset_key = "_".join(sorted(dataset_names))
    kiba_suffix = "_kiba_prioritized" if "KIBA" in dataset_names else ""
    filter_suffix = "_filtered" if apply_filters else ""
    filter_method = "_pre_filtered" if filter_before_merge and apply_filters else ""
    merged_filename = f"merged_{dataset_key}{kiba_suffix}{filter_suffix}{filter_method}.csv"
    merged_path = PROCESSED_DIR / merged_filename
    
    # Check if the merged file already exists
    if not force_reload and merged_path.exists():
        if verbose:
            print(f"📂 Loading existing merged dataset from {merged_path}")
        return pd.read_csv(merged_path)
    
    # Create the merged dataset
    if verbose:
        print(f"🔄 Creating merged dataset from: {', '.join(dataset_names)}")
        if "KIBA" in dataset_names:
            print("ℹ️ KIBA dataset will be prioritized during merging")
        if apply_filters and filter_before_merge:
            print("ℹ️ Datasets will be filtered before merging")
    
    # Choose the appropriate merging approach
    if apply_filters and filter_before_merge:
        # Filter each dataset before merging
        merged_df = merge_datasets(dataset_names, verbose=verbose, pre_filtered=True)
    else:
        # Merge first, then filter if needed
        merged_df = merge_datasets(dataset_names, verbose=verbose, pre_filtered=False)
        if apply_filters:
            merged_df = filter_dataset(merged_df, verbose=verbose)
    
    # Save the merged dataset
    save_dataset(merged_df, merged_filename.replace(".csv", ""), verbose=verbose)
    
    return merged_df


def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Calculate statistics for a dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict: Dictionary containing dataset statistics
    """
    stats = {
        "n_interactions": len(df),
        "n_unique_drugs": df['Drug_SMILES'].nunique(),
        "n_unique_targets": df['Target_AA'].nunique(),
        "avg_drug_smiles_length": df['Drug_SMILES'].str.len().mean(),
        "avg_target_aa_length": df['Target_AA'].str.len().mean(),
    }
    
    # Add statistics for each measurement column
    for col in df.columns:
        if col.startswith('Y_'):
            stats[f"{col}_mean"] = df[col].mean()
            stats[f"{col}_std"] = df[col].std()
            stats[f"{col}_min"] = df[col].min()
            stats[f"{col}_max"] = df[col].max()
    
    # Add statistics for dataset indicators
    for col in df.columns:
        if col.startswith('in_'):
            stats[f"{col}_count"] = df[col].sum()
            stats[f"{col}_percentage"] = (df[col].sum() / len(df)) * 100
    
    return stats 