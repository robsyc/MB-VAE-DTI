import os
from tdc.generation import MolGen
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

# Define paths
DATA_DIR = Path("data")
SOURCE_DIR = DATA_DIR / "source"
PROCESSED_DIR = DATA_DIR / "processed"

MAX_N_HEAVY_ATOMS = 64
MAX_N_WORKERS = min(os.cpu_count() or 1, 16)

# Disable RDKit logging for better performance
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def canonicalize_smiles(smiles):
    """Convert a SMILES string to its canonical form"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol and Descriptors.HeavyAtomCount(mol) <= MAX_N_HEAVY_ATOMS:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return ""
    except:
        return ""

def process_chunk(smiles_chunk):
    """Process a chunk of SMILES strings in parallel"""
    return [canonicalize_smiles(smiles) for smiles in smiles_chunk]

def load_drug_generation_datasets(datasets=["MOSES", "ZINC", "ChEMBL_V29"], path=SOURCE_DIR, n_workers=MAX_N_WORKERS, chunk_size=10000):
    """
    Fetches and merges multiple molecular datasets, removing duplicate SMILES
    and ensuring all SMILES are in canonical form using RDKit with parallel processing.
    
    Args:
        datasets (list): List of dataset names to fetch
        path (str): Path to store/load the datasets
        n_workers (int): Number of parallel workers
        chunk_size (int): Size of chunks for parallel processing
        
    Returns:
        pandas.DataFrame: Merged dataframe with unique canonical SMILES
    """
    all_data = []
    
    for dataset_name in tqdm(datasets, desc="Loading datasets"):
        print(f"Loading {dataset_name}...")
        data = MolGen(name=dataset_name, path=path)
        df = data.get_data()
        all_data.append(df)
        print(f"  {len(df)} molecules loaded")
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"Total molecules before deduplication: {len(merged_df)}")
    
    # First deduplication on exact SMILES strings
    merged_df = merged_df.drop_duplicates(subset=['smiles'])
    print(f"Unique molecules before canonicalization: {len(merged_df)}")
    
    # Split SMILES into chunks for parallel processing
    smiles_list = merged_df['smiles'].tolist()
    smiles_chunks = [smiles_list[i:i+chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    # Use parallel processing for canonicalization
    print(f"Canonicalizing SMILES using {n_workers} workers...")
    canonical_smiles_chunks = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(smiles_chunks)}
        
        # Process results with tqdm progress bar
        with tqdm(total=len(smiles_chunks), desc="Canonicalizing SMILES") as pbar:
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    canonical_smiles_chunks.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
                    pbar.update(1)
    
    # Flatten the list of chunks
    canonical_smiles = []
    for chunk in tqdm(canonical_smiles_chunks, desc="Combining results"):
        canonical_smiles.extend(chunk)
    
    # Add canonical SMILES to dataframe
    merged_df['canonical_smiles'] = canonical_smiles

    # Remove duplicates based on canonical SMILES and filter out invalid molecules
    merged_df = merged_df[merged_df['canonical_smiles'] != ""]
    unique_df = merged_df.drop_duplicates(subset=['canonical_smiles'])
    
    # Keep original column structure if needed
    unique_df['smiles'] = unique_df['canonical_smiles']
    unique_df.drop(columns=['canonical_smiles'], inplace=True)
    
    print(f"Total valid molecules after processing: {len(unique_df)}")    
    return unique_df

# Example usage
# datasets = ["MOSES", "ZINC", "ChEMBL_V29"]
# df = load_drug_generation_datasets(datasets)