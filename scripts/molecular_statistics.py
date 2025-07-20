#!/usr/bin/env python3
"""
Generate molecular statistics report for all datasets.

This script processes SMILES strings from various datasets and generates
a comprehensive report including:
- Distribution of node counts
- Node type marginals (atom type probabilities)
- Edge type marginals (bond type probabilities)

The script uses parallelization and progress tracking for efficiency.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Optional, Tuple
import logging
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Mol

# Import from our modules
from mb_vae_dti.processing.split import add_splits
from mb_vae_dti.processing import PretrainDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

ATOMS_TYPES = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

def compute_marginals(
    molecules: List[Mol],
    atom_types: List[str] = ATOMS_TYPES,
    bond_types: dict = BONDS,
    include_no_bond: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute node and edge type marginals from SMILES strings.
    
    Args:
        molecules: List of RDKit molecule objects
        atom_types: List of atom symbols to consider
        bond_types: Dictionary mapping RDKit bond types to indices (defaults to BONDS)
        include_no_bond: Whether to include "no bond" (non-connected pairs) in edge marginals
    
    Returns:
        Tuple of (node_marginals, edge_marginals) as numpy arrays
        - node_marginals: shape (num_atom_types,) - probability of each atom type
        - edge_marginals: shape (num_bond_types + include_no_bond,) - probability of each bond type
    """
    # Create atom type encoder/decoder
    atom_to_idx = {atom: idx for idx, atom in enumerate(atom_types)}
    
    # Initialize counters
    node_counts = np.zeros(len(atom_types), dtype=np.int64)
    
    if include_no_bond:
        edge_counts = np.zeros(len(bond_types) + 1, dtype=np.int64)  # +1 for no_bond
        no_bond_idx = 0
        bond_offset = 1
    else:
        edge_counts = np.zeros(len(bond_types), dtype=np.int64)
        bond_offset = 0
    
    total_possible_edges = 0
    total_actual_bonds = 0
    
    for mol in molecules:
        n_atoms = mol.GetNumAtoms()
        
        # Count node types
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_to_idx:
                node_counts[atom_to_idx[symbol]] += 1
        
        # Count bond types
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type in bond_types:
                bond_idx = bond_types[bond_type] + bond_offset
                edge_counts[bond_idx] += 1
                total_actual_bonds += 1
        
        # Track total possible edges for no_bond calculation
        if include_no_bond:
            # For undirected graphs: n_atoms * (n_atoms - 1) / 2
            total_possible_edges += n_atoms * (n_atoms - 1) // 2
    
    # Add no_bond counts
    if include_no_bond:
        edge_counts[no_bond_idx] = total_possible_edges - total_actual_bonds
    
    # Normalize to get marginals (probabilities)
    node_marginals = node_counts.astype(np.float64)
    if node_marginals.sum() > 0:
        node_marginals /= node_marginals.sum()
    
    edge_marginals = edge_counts.astype(np.float64)
    if edge_marginals.sum() > 0:
        edge_marginals /= edge_marginals.sum()
    
    return node_marginals, edge_marginals


def process_single_smiles(smiles: str) -> Optional[Dict]:
    """
    Process a single SMILES string and extract molecular properties.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with molecular properties or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'smiles': smiles,
            'num_atoms': len(mol.GetAtoms()),
            'atom_types': [atom.GetSymbol() for atom in mol.GetAtoms()],
            'bond_types': [bond.GetBondType() for bond in mol.GetBonds()],
            'mol': mol  # We'll need this for marginal calculations
        }
    except Exception as e:
        logger.warning(f"Failed to process SMILES '{smiles}': {e}")
        return None

def process_smiles_batch(smiles_batch: List[str]) -> List[Dict]:
    """Process a batch of SMILES strings."""
    results = []
    for smiles in smiles_batch:
        result = process_single_smiles(smiles)
        if result is not None:
            results.append(result)
    return results


def get_dataset_smiles() -> Dict[str, Set[str]]:
    """
    Load and extract SMILES strings from all datasets.
    
    Returns:
        Dictionary mapping dataset names to sets of SMILES strings
    """
    logger.info("Loading datasets...")
    
    # Load main DTI dataset
    df = pd.read_csv(BASE_DIR / "data/processed/dti.csv")
    df = add_splits(df, split_fractions=(0.8, 0.1, 0.1), stratify=True, random_state=42)

    # Remove duplicates based on Drug_SMILES for each split
    drugs_rand = df[df["split_rand"] == "train"].drop_duplicates(subset="Drug_SMILES")
    drugs_cold = df[df["split_cold"] == "train"].drop_duplicates(subset="Drug_SMILES")

    # Further subsets for DAVIS and KIBA
    drugs_rand_davis = drugs_rand[drugs_rand["in_DAVIS"]].copy()
    drugs_rand_kiba = drugs_rand[drugs_rand["in_KIBA"]].copy()
    drugs_cold_davis = drugs_cold[drugs_cold["in_DAVIS"]].copy()
    drugs_cold_kiba = drugs_cold[drugs_cold["in_KIBA"]].copy()

    # Collect datasets
    datasets = {
        'drugs_rand': set(drugs_rand["Drug_SMILES"].tolist()),
        'drugs_cold': set(drugs_cold["Drug_SMILES"].tolist()),
        'drugs_rand_davis': set(drugs_rand_davis["Drug_SMILES"].tolist()),
        'drugs_rand_kiba': set(drugs_rand_kiba["Drug_SMILES"].tolist()),
        'drugs_cold_davis': set(drugs_cold_davis["Drug_SMILES"].tolist()),
        'drugs_cold_kiba': set(drugs_cold_kiba["Drug_SMILES"].tolist()),
    }
    
    # Load pretrain dataset
    try:
        drugs_pretrain = PretrainDataset(
            h5_path=BASE_DIR / "data/input/drugs.h5torch",
            subset_filters={'split_col': 'is_train', 'split_value': True}
        )
        
        # Sample SMILES from pretrain dataset
        pretrain_smiles = []
        logger.info("Sampling SMILES from pretrain dataset...")
        for i, data in enumerate(tqdm(drugs_pretrain, desc="Loading pretrain")):
            pretrain_smiles.append(data["representations"]["smiles"])

        datasets['drugs_pretrain'] = set(pretrain_smiles)
        
    except Exception as e:
        logger.warning(f"Failed to load pretrain dataset: {e}")
        datasets['drugs_pretrain'] = set()
    
    # Log dataset sizes
    for name, smiles_set in datasets.items():
        logger.info(f"{name}: {len(smiles_set)} unique SMILES")
    
    return datasets

def calculate_dataset_statistics(
    dataset_name: str, 
    dataset_smiles: Set[str], 
    processed_molecules: Dict[str, Dict]
) -> Dict:
    """
    Calculate statistics for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_smiles: Set of SMILES strings in this dataset  
        processed_molecules: Dictionary mapping SMILES to processed molecule data
        
    Returns:
        Dictionary with dataset statistics
    """
    logger.info(f"Calculating statistics for {dataset_name}...")
    
    # Filter molecules for this dataset
    dataset_molecules = []
    valid_smiles = []
    
    for smiles in dataset_smiles:
        if smiles in processed_molecules:
            dataset_molecules.append(processed_molecules[smiles])
            valid_smiles.append(smiles)
    
    if not dataset_molecules:
        return {
            'dataset_name': dataset_name,
            'total_molecules': 0,
            'valid_molecules': 0,
            'max_nodes': 0,
            'node_distribution': {},
            'node_marginals': [],
            'edge_marginals': [],
            'atom_types': []
        }
    
    # Calculate basic statistics
    num_nodes = [mol['num_atoms'] for mol in dataset_molecules]
    max_nodes = max(num_nodes)
    node_distribution = dict(Counter(num_nodes))
    
    # Calculate marginals
    molecules_for_marginals = [mol['mol'] for mol in dataset_molecules]
    node_marginals, edge_marginals = compute_marginals(molecules_for_marginals)
    
    return {
        'dataset_name': dataset_name,
        'total_molecules': len(dataset_smiles),
        'valid_molecules': len(dataset_molecules),
        'max_nodes': max_nodes,
        'node_distribution': node_distribution,
        'node_marginals': node_marginals.tolist(),
        'edge_marginals': edge_marginals.tolist(),
        'atom_types': ATOMS_TYPES,
        'bond_types': [str(bt) for bt in BONDS.keys()],
        'example_smiles': valid_smiles[:5]  # Include a few examples
    }

def main():
    """Main execution function."""
    logger.info("Starting molecular statistics generation...")
    
    # Get all dataset SMILES
    datasets = get_dataset_smiles()
    
    # Collect all unique SMILES
    all_smiles = set()
    for smiles_set in datasets.values():
        all_smiles.update(smiles_set)
    
    logger.info(f"Total unique SMILES across all datasets: {len(all_smiles)}")
    
    # Process all unique SMILES in parallel
    all_smiles_list = list(all_smiles)
    batch_size = max(1, len(all_smiles_list) // (cpu_count() * 4))
    
    # Create batches
    smiles_batches = []
    for i in range(0, len(all_smiles_list), batch_size):
        smiles_batches.append(all_smiles_list[i:i + batch_size])
    
    logger.info(f"Processing {len(all_smiles)} SMILES in {len(smiles_batches)} batches using {cpu_count()} cores...")
    
    # Process in parallel
    processed_molecules = {}
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_smiles_batch, smiles_batches),
            total=len(smiles_batches),
            desc="Processing SMILES"
        ))
    
    # Flatten results
    for batch_results in results:
        for mol_data in batch_results:
            processed_molecules[mol_data['smiles']] = mol_data
    
    logger.info(f"Successfully processed {len(processed_molecules)} molecules")
    
    # Calculate statistics for each dataset
    final_report = {
        'metadata': {
            'total_unique_smiles': len(all_smiles),
            'successfully_processed': len(processed_molecules),
            'processing_success_rate': len(processed_molecules) / len(all_smiles) if all_smiles else 0,
            'datasets_processed': list(datasets.keys())
        },
        'datasets': {}
    }
    
    for dataset_name, dataset_smiles in datasets.items():
        stats = calculate_dataset_statistics(dataset_name, dataset_smiles, processed_molecules)
        final_report['datasets'][dataset_name] = stats
    
    # Save report
    output_path = BASE_DIR / "data/processed/molecular_statistics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info(f"Report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("MOLECULAR STATISTICS SUMMARY")
    print("="*60)
    
    for dataset_name, stats in final_report['datasets'].items():
        print(f"\n{dataset_name}:")
        print(f"  Total molecules: {stats['total_molecules']}")
        print(f"  Valid molecules: {stats['valid_molecules']}")
        print(f"  Max nodes: {stats['max_nodes']}")
        if stats['node_distribution']:
            most_common = dict(sorted(Counter(stats['node_distribution']).most_common(5)))
            print(f"  Most common node counts: {most_common}")
        else:
            print(f"  Most common node counts: {{}}")
    
    print(f"\nFull report saved to: {output_path}")

if __name__ == "__main__":
    main() 