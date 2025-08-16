#!/usr/bin/env python3
"""
Generate molecular statistics report for all datasets.

This script processes SMILES strings from various datasets and generates
a comprehensive report including:
- Distribution of node counts (number of atoms per molecule)
- Node type marginals (atom type probabilities)
- Edge type marginals (bond type probabilities)
- Valency distribution (distribution of atom valencies)
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
MAX_N_NODES = 64

ATOM_TO_VALENCY = {
    'H': 1,
    'He': 0,
    'Li': 1,
    'Be': 2,
    'B': 3,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'Ne': 0,
    'Na': 1,
    'Mg': 2,
    'Al': 3,
    'Si': 4,
    'P': 3,
    'S': 2,
    'Cl': 1,
    'Ar': 0,
    'K': 1,
    'Ca': 2,
    'Sc': 3,
    'Ti': 4,
    'V': 5,
    'Cr': 2,
    'Mn': 7,
    'Fe': 2,
    'Co': 3,
    'Ni': 2,
    'Cu': 2,
    'Zn': 2,
    'Ga': 3,
    'Ge': 4,
    'As': 3,
    'Se': 2,
    'Br': 1,
    'Kr': 0,
    'Rb': 1,
    'Sr': 2,
    'Y': 3,
    'Zr': 2,
    'Nb': 2,
    'Mo': 2,
    'Tc': 6,
    'Ru': 2,
    'Rh': 3,
    'Pd': 2,
    'Ag': 1,
    'Cd': 1,
    'In': 1,
    'Sn': 2,
    'Sb': 3,
    'Te': 2,
    'I': 1,
    'Xe': 0,
    'Cs': 1,
    'Ba': 2,
    'La': 3,
    'Ce': 3,
    'Pr': 3,
    'Nd': 3,
    'Pm': 3,
    'Sm': 2,
    'Eu': 2,
    'Gd': 3,
    'Tb': 3,
    'Dy': 3,
    'Ho': 3,
    'Er': 3,
    'Tm': 2,
    'Yb': 2,
    'Lu': 3,
    'Hf': 4,
    'Ta': 3,
    'W': 2,
    'Re': 1,
    'Os': 2,
    'Ir': 1,
    'Pt': 1,
    'Au': 1,
    'Hg': 1,
    'Tl': 1,
    'Pb': 2,
    'Bi': 3,
    'Po': 2,
    'At': 1,
    'Rn': 0,
    'Fr': 1,
    'Ra': 2,
    'Ac': 3,
    'Th': 4,
    'Pa': 5,
    'U': 2,
}

ATOM_TO_WEIGHT = {
    'H': 1,
    'He': 4,
    'Li': 7,
    'Be': 9,
    'B': 11,
    'C': 12,
    'N': 14,
    'O': 16,
    'F': 19,
    'Ne': 20,
    'Na': 23,
    'Mg': 24,
    'Al': 27,
    'Si': 28,
    'P': 31,
    'S': 32,
    'Cl': 35,
    'Ar': 40,
    'K': 39,
    'Ca': 40,
    'Sc': 45,
    'Ti': 48,
    'V': 51,
    'Cr': 52,
    'Mn': 55,
    'Fe': 56,
    'Co': 59,
    'Ni': 59,
    'Cu': 64,
    'Zn': 65,
    'Ga': 70,
    'Ge': 73,
    'As': 75,
    'Se': 79,
    'Br': 80,
    'Kr': 84,
    'Rb': 85,
    'Sr': 88,
    'Y': 89,
    'Zr': 91,
    'Nb': 93,
    'Mo': 96,
    'Tc': 98,
    'Ru': 101,
    'Rh': 103,
    'Pd': 106,
    'Ag': 108,
    'Cd': 112,
    'In': 115,
    'Sn': 119,
    'Sb': 122,
    'Te': 128,
    'I': 127,
    'Xe': 131,
    'Cs': 133,
    'Ba': 137,
    'La': 139,
    'Ce': 140,
    'Pr': 141,
    'Nd': 144,
    'Pm': 145,
    'Sm': 150,
    'Eu': 152,
    'Gd': 157,
    'Tb': 159,
    'Dy': 163,
    'Ho': 165,
    'Er': 167,
    'Tm': 169,
    'Yb': 173,
    'Lu': 175,
    'Hf': 178,
    'Ta': 181,
    'W': 184,
    'Re': 186,
    'Os': 190,
    'Ir': 192,
    'Pt': 195,
    'Au': 197,
    'Hg': 201,
    'Tl': 204,
    'Pb': 207,
    'Bi': 209,
    'Po': 209,
    'At': 210,
    'Rn': 222,
    'Fr': 223,
    'Ra': 226,
    'Ac': 227,
    'Th': 232,
    'Pa': 231,
    'U': 238,
    'Np': 237,
    'Pu': 244,
    'Am': 243,
    'Cm': 247,
    'Bk': 247,
    'Cf': 251,
    'Es': 252,
    'Fm': 257,
    'Md': 258,
    'No': 259,
    'Lr': 262,
    'Rf': 267,
    'Db': 270,
    'Sg': 269,
    'Bh': 264,
    'Hs': 269,
    'Mt': 278,
    'Ds': 281,
    'Rg': 282,
    'Cn': 285,
    'Nh': 286,
    'Fl': 289,
    'Mc': 290,
    'Lv': 293,
    'Ts': 294,
    'Og': 294,
}

PROCESSED_MOLECULES = {}


def get_marginal_counts(
    molecule: Mol,
    atom_types: List[str] = ATOMS_TYPES,
    bond_types: dict = BONDS,
    include_no_bond: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute node and edge type counts from a molecule.
    
    Args:
        molecule: RDKit molecule object
        atom_types: List of atom symbols to consider
        bond_types: Dictionary mapping RDKit bond types to indices (defaults to BONDS)
        include_no_bond: Whether to include "no bond" (non-connected pairs) in edge counts
    
    Returns:
        Tuple of (node_counts, edge_counts) as numpy arrays
        - node_counts: shape (num_atom_types,) - count of each atom type
        - edge_counts: shape (num_bond_types + include_no_bond,) - count of each bond type
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
    
    n_atoms = molecule.GetNumAtoms()
        
    # Count node types
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_to_idx:
            node_counts[atom_to_idx[symbol]] += 1
        
    # Count bond types
    for bond in molecule.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type in bond_types:
            bond_idx = bond_types[bond_type] + bond_offset
            edge_counts[bond_idx] += 1
            total_actual_bonds += 1
    
    # Track total possible edges for no_bond calculation
    if include_no_bond:
        total_possible_edges = n_atoms * (n_atoms - 1) // 2
        edge_counts[no_bond_idx] = total_possible_edges - total_actual_bonds
    
    return node_counts, edge_counts


def process_single_smiles(smiles: str, atom_types: List[str], bond_types: dict, include_no_bond: bool) -> Optional[Dict]:
    """
    Process a single SMILES string and extract molecular properties.
    
    Args:
        smiles: SMILES string
        atom_types: List of atom symbols to consider
        bond_types: Dictionary mapping RDKit bond types to indices
        include_no_bond: Whether to include "no bond" in edge counts
        
    Returns:
        Dictionary with molecular properties or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
            
        node_counts, edge_counts = get_marginal_counts(mol, atom_types, bond_types, include_no_bond)
        
        # Calculate valency counts following DiGress methodology
        valency_counts = np.zeros(3 * MAX_N_NODES - 2, dtype=np.int64)
        
        # Bond multipliers: [no bond, single, double, triple, aromatic]  
        multiplier = np.array([0, 1, 2, 3, 1.5])
        
        n_atoms = mol.GetNumAtoms()
        
        # Validate molecule size
        if n_atoms > MAX_N_NODES:
            logger.warning(f"Molecule with {n_atoms} atoms exceeds MAX_N_NODES={MAX_N_NODES}, skipping")
            return None
        
        # For each atom, calculate its valency
        for atom_idx in range(n_atoms):
            atom_bonds = np.zeros(len(bond_types) + 1, dtype=np.int32)  # +1 for "no bond"
            
            # Count bonds for this atom
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() == atom_idx or bond.GetEndAtomIdx() == atom_idx:
                    bond_type = bond.GetBondType()
                    if bond_type in bond_types:
                        bond_idx = bond_types[bond_type] + 1  # +1 because index 0 is "no bond"
                        atom_bonds[bond_idx] += 1
            
            # Calculate valency as weighted sum of bonds
            valency = np.sum(atom_bonds * multiplier)
            valency_int = int(valency)
            
            # Increment count for this valency
            if valency_int < len(valency_counts):
                valency_counts[valency_int] += 1
        
        return {
            'num_atoms': n_atoms,
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'valency_counts': valency_counts,
        }
    except Exception as e:
        logger.warning(f"Failed to process SMILES '{smiles}': {e}")
        return None


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
    pretrain_drugs = pd.read_csv(BASE_DIR / "data/processed/pretrain_drugs.csv")
    datasets['drugs_pretrain'] = set(pretrain_drugs["smiles"].tolist())
    
    # Log dataset sizes
    for name, smiles_set in datasets.items():
        logger.info(f"{name}: {len(smiles_set)} unique SMILES")
    
    return datasets


def calculate_dataset_statistics(
    dataset_name: str, 
    dataset_smiles: Set[str], 
    use_cache: bool = True,
    atom_types: List[str] = ATOMS_TYPES,
    bond_types: dict = BONDS,
    include_no_bond: bool = True
) -> Dict:
    """
    Calculate statistics for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_smiles: Set of SMILES strings in this dataset  
        use_cache: Whether to check/update the PROCESSED_MOLECULES cache
        atom_types: List of atom symbols to consider
        bond_types: Dictionary mapping RDKit bond types to indices
        include_no_bond: Whether to include "no bond" (non-connected pairs) in edge counts
        
    Returns:
        Dictionary with dataset statistics
    """
    global PROCESSED_MOLECULES
    
    logger.info(f"Calculating statistics for {dataset_name}...")

    # Initialize counters
    num_atoms_counts = Counter()
    node_counts = np.zeros(len(atom_types), dtype=np.int64)
    edge_counts = np.zeros(len(bond_types) + 1, dtype=np.int64) if include_no_bond else np.zeros(len(bond_types), dtype=np.int64)
    valency_counts = np.zeros(3 * MAX_N_NODES - 2, dtype=np.int64)
    
    # Add progress bar for processing SMILES
    for smiles in tqdm(dataset_smiles, desc=f"Processing {dataset_name}", unit="molecules"):
        if use_cache and smiles in PROCESSED_MOLECULES:
            result = PROCESSED_MOLECULES[smiles]
        else:
            result = process_single_smiles(smiles, atom_types, bond_types, include_no_bond)
            if result is not None and use_cache:
                # Store the result in cache for future use
                PROCESSED_MOLECULES[smiles] = result
        
        if result is not None:
            num_atoms_counts[result['num_atoms']] += 1
            node_counts += result['node_counts']
            edge_counts += result['edge_counts']
            valency_counts += result['valency_counts']
    
    max_nodes = max(num_atoms_counts.keys()) if num_atoms_counts else 1
    valency_distribution = valency_counts / valency_counts.sum() if valency_counts.sum() > 0 else valency_counts
    node_marginals = node_counts / node_counts.sum() if node_counts.sum() > 0 else node_counts
    edge_marginals = edge_counts / edge_counts.sum() if edge_counts.sum() > 0 else edge_counts
    
    # Log statistics for debugging
    logger.info(f"{dataset_name} - Total molecules: {sum(num_atoms_counts.values())}")
    logger.info(f"{dataset_name} - Node marginals sum: {node_marginals.sum():.6f}")
    logger.info(f"{dataset_name} - Edge marginals sum: {edge_marginals.sum():.6f}")
    logger.info(f"{dataset_name} - Valency distribution sum: {valency_distribution.sum():.6f}")
    
    return {
        'dataset_name': dataset_name,
        'max_nodes': max_nodes,
        'node_count_distribution': dict(num_atoms_counts),
        'node_marginals': node_marginals.tolist(),
        'edge_marginals': edge_marginals.tolist(),
        'valency_distribution': valency_distribution.tolist(),
    }


def main():
    """Main execution function."""
    logger.info("Starting molecular statistics generation...")

    datasets = get_dataset_smiles()

    stats = {
        'general': {
            'atom_types': ATOMS_TYPES,
            'bond_types': ["NONE"] + [bond.name for bond in BONDS.keys()],
            'num_atom_types': len(ATOMS_TYPES),
            'max_n_nodes': MAX_N_NODES,
            'atom_valencies': [ATOM_TO_VALENCY[atom] for atom in ATOMS_TYPES],
            'atom_weights': [ATOM_TO_WEIGHT[atom] for atom in ATOMS_TYPES],
            'max_weight': max([ATOM_TO_WEIGHT[atom] for atom in ATOMS_TYPES]),
        },
        'datasets': {},
    }

    # Dataset statistics
    for dataset_name, dataset_smiles in datasets.items():
        if dataset_name != 'drugs_pretrain':
            stats['datasets'][dataset_name] = calculate_dataset_statistics(dataset_name, dataset_smiles)
        else: # pretrains dataset are all unique, so no point in tracking them in the PROCESSED_MOLECULES cache
            stats['datasets'][dataset_name] = calculate_dataset_statistics(dataset_name, dataset_smiles, use_cache=False)
        
    # Save report
    output_path = BASE_DIR / "data/processed/molecular_statistics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info(f"Report saved to {output_path}")
    logger.info(f"Total unique molecules cached: {len(PROCESSED_MOLECULES)}")

if __name__ == "__main__":
    main() 