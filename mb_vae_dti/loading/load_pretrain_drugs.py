import os
from tdc.generation import MolGen
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Union, Set

# Define paths
DATA_DIR = Path("data")
SOURCE_DIR = DATA_DIR / "source"
PROCESSED_DIR = DATA_DIR / "processed"

# Constants for filtering (same as in datasets.py)
MAX_N_HEAVY_ATOMS = 64
FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P'}
MAX_MOL_WEIGHT = 1500

# Disable RDKit logging for better performance
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def filter_molecule(smiles: str) -> Dict[str, Union[bool, str, float]]:
    """
    Filter a single molecule based on chemical properties.
    
    Args:
        smiles: SMILES string to filter
        
    Returns:
        Dict with filtering results including canonical SMILES and pass/fail status
    """
    # Initialize result dictionary
    result = {
        "canonical_smiles": "",
        "passes_filter": False,
        "fail_reason": None
    }
    
    try:
        # Parse the SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result["fail_reason"] = "invalid_smiles"
            return result
        
        # Convert to canonical SMILES with no stereochemistry
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        result["canonical_smiles"] = canonical_smiles
        
        # Recreate molecule from canonical SMILES for consistent property calculation
        mol = Chem.MolFromSmiles(canonical_smiles)
        
        # Check for multi-component molecules
        if "." in canonical_smiles:
            result["fail_reason"] = "multi_component"
            return result
        
        # Check molecular weight
        mol_weight = Descriptors.MolWt(mol)
        if mol_weight >= MAX_MOL_WEIGHT:
            result["fail_reason"] = "mol_weight_too_high"
            return result
        
        # Count heavy atoms
        heavy_atom_count = Descriptors.HeavyAtomCount(mol)
        if heavy_atom_count > MAX_N_HEAVY_ATOMS:
            result["fail_reason"] = "too_many_heavy_atoms"
            return result
        
        # Check atom types and charges
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                result["fail_reason"] = "charged_atoms"
                return result
            if atom.GetSymbol() not in FILTER_ATOMS:
                result["fail_reason"] = "unsupported_atom_types"
                return result
        
        # If we get here, molecule passes all filters
        result["passes_filter"] = True
        
    except Exception as e:
        result["fail_reason"] = f"exception: {str(e)}"
    
    return result

def load_drug_generation_datasets(
    datasets: List[str] = ["MOSES", "ZINC", "ChEMBL_V29"], 
    path: Path = SOURCE_DIR,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetches and merges multiple molecular datasets, removing duplicate SMILES
    and filtering them based on molecular properties.
    
    Args:
        datasets: List of dataset names to fetch from TDC
        path: Path to store/load the datasets
        verbose: Whether to print detailed progress information
        
    Returns:
        pandas.DataFrame: DataFrame with unique filtered canonical SMILES
    """
    all_smiles = set()
    dataset_sizes = {}
    
    if verbose:
        print(f"📂 Loading datasets: {', '.join(datasets)}")
    
    # First, collect all SMILES from all datasets
    for dataset_name in tqdm(datasets, desc="Loading datasets"):
        if verbose:
            print(f"  - Loading {dataset_name}...")
        data = MolGen(name=dataset_name, path=path)
        df = data.get_data()
        current_smiles = set(df['smiles'].dropna().unique())
        dataset_sizes[dataset_name] = len(current_smiles)
        all_smiles.update(current_smiles)
        if verbose:
            print(f"    ✓ {len(current_smiles):,} unique molecules loaded")
    
    # Convert to list for processing
    unique_smiles = list(all_smiles)
    
    if verbose:
        print(f"\n📊 Statistics:")
        print(f"  - Total molecules across all datasets: {sum(dataset_sizes.values()):,}")
        print(f"  - Unique molecules before filtering: {len(unique_smiles):,}")
        print(f"  - Deduplication removed {sum(dataset_sizes.values()) - len(unique_smiles):,} duplicates")
    
    # Filter the molecules
    if verbose:
        print(f"\n🔍 Filtering {len(unique_smiles):,} unique molecules...")
    
    # Initialize storage for results
    canonical_dict = {}  # Maps canonical SMILES to original SMILES (for tracking)
    filtered_results = []
    fail_counts = {
        "parsing": 0,
        "multi_component": 0,
        "mol_weight": 0,
        "atom_charge": 0,
        "atom_type": 0,
        "heavy_atoms": 0,
        "other": 0
    }
    
    # Process all unique SMILES
    for smiles in tqdm(unique_smiles, desc="Filtering molecules"):
        result = filter_molecule(smiles)
        
        if not result["passes_filter"]:
            # Count failures by reason
            reason = result["fail_reason"]
            if reason and any(key in reason for key in fail_counts.keys()):
                for key in fail_counts.keys():
                    if key in reason:
                        fail_counts[key] += 1
                        break
            else:
                fail_counts["other"] += 1
            continue
        
        # Track which canonical forms we've seen
        canonical = result["canonical_smiles"]
        if canonical not in canonical_dict:
            canonical_dict[canonical] = []
            filtered_results.append({"smiles": canonical})
        
        # Keep track of original SMILES that map to this canonical form
        canonical_dict[canonical].append(smiles)
    
    # Create final dataframe from filtered results
    filtered_df = pd.DataFrame(filtered_results)
    
    if verbose:
        print(f"\n✅ Filtering complete:")
        print(f"  - Molecules that passed filtering: {len(filtered_df):,} ({len(filtered_df)/len(unique_smiles):.2%})")
        
        if sum(fail_counts.values()) > 0:
            print(f"  - Failed molecules breakdown:")
            for reason, count in fail_counts.items():
                if count > 0:
                    print(f"    - {reason}: {count:,} molecules")
        
        # Report on canonicalization effect
        duplicates_from_canon = sum(len(orig_list) - 1 for orig_list in canonical_dict.values() if len(orig_list) > 1)
        if duplicates_from_canon > 0:
            print(f"  - Canonicalization consolidated {duplicates_from_canon:,} additional molecules")
    
    return filtered_df

def save_filtered_dataset(df: pd.DataFrame, name: str = "drugs", verbose: bool = True) -> str:
    """
    Save a filtered dataset to a CSV file.
    
    Args:
        df: DataFrame to save
        name: Name to use for the file
        verbose: Whether to print save information
        
    Returns:
        str: Path to the saved file
    """
    # Create the output path
    output_path = PROCESSED_DIR / f"{name}.csv"
    
    # Save the DataFrame
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"💾 Saved {len(df):,} filtered molecules to {output_path}")
    
    return str(output_path)

# Example usage:
# df = load_drug_generation_datasets(["MOSES", "ZINC", "ChEMBL_V29"])
# save_filtered_dataset(df, "pretrain_drugs")