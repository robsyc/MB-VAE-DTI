"""
Drug annotation functionality for MB-VAE-DTI.

This module provides functions for annotating drugs using their potential IDs
and SMILES strings. It uses RDKit for molecule handling and validation, and
external APIs for retrieving drug information.
"""

import pandas as pd
import requests
import time
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from rdkit import Chem
from pathlib import Path
import json
from pydantic import BaseModel
import warnings

# Silence RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = PROCESSED_DIR / "annotations"

# Ensure directories exist
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

# Cache for API responses to avoid redundant calls
PUBCHEM_CACHE_FILE = ANNOTATION_DIR / "pubchem_cache.json"
CHEMBL_CACHE_FILE = ANNOTATION_DIR / "chembl_cache.json"

# Initialize caches
pubchem_cache = {}
chembl_cache = {}

# Load caches if they exist
if PUBCHEM_CACHE_FILE.exists():
    try:
        with open(PUBCHEM_CACHE_FILE, 'r') as f:
            pubchem_cache = json.load(f)
    except Exception as e:
        print(f"Failed to load PubChem cache: {e}")

if CHEMBL_CACHE_FILE.exists():
    try:
        with open(CHEMBL_CACHE_FILE, 'r') as f:
            chembl_cache = json.load(f)
    except Exception as e:
        print(f"Failed to load ChEMBL cache: {e}")

def save_cache() -> None:
    """Save API response caches to disk."""
    try:  
        with open(PUBCHEM_CACHE_FILE, 'w') as f:
            json.dump(pubchem_cache, f)
        
        with open(CHEMBL_CACHE_FILE, 'w') as f:
            json.dump(chembl_cache, f)
    except Exception as e:
        print(f"Failed to save caches: {e}")


class DrugAnnotation(BaseModel):
    smiles: str
    inchikey: str
    valid: bool


def inchiKey_to_inchi(inchiKey: str) -> str:
    """
    Convert an InChIKey to an InChI string using the PubChem API.
    
    Args:
        inchiKey: The InChIKey to convert
    
    Returns:
        The InChI string or None if the conversion failed
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchiKey}/property/InChI/TXT"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    else:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string to its canonical form using RDKit.
    
    Args:
        smiles: SMILES string to canonicalize
        
    Returns:
        Canonical SMILES string or None if the input is invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return None
    except:
        return None


def get_molecule_from_pubchem_id(pubchem_id: str) -> Tuple[Optional[Chem.Mol], Optional[Dict[str, str]]]:
    """
    Retrieve a molecule from PubChem using its ID.
    
    Args:
        pubchem_id: PubChem ID (CID)
        
    Returns:
        Tuple containing:
        - RDKit molecule object or None if retrieval failed
        - Dictionary with SMILES and InChIKey or None if retrieval failed
    """
    # Check cache first
    if pubchem_id in pubchem_cache:
        smiles = pubchem_cache[pubchem_id].get('smiles')
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            return mol, pubchem_cache[pubchem_id]
        else:
            # Return cached negative result
            return None, None
    
    try:
        # Try to interpret the ID as a PubChem CID
        # e.g. https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/103905596/property/IsomericSMILES,CanonicalSMILES,InChIKey/JSON
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/property/IsomericSMILES,CanonicalSMILES,InChIKey/JSON"
        response = requests.get(url)
        
        # Respect PubChem's rate limit (max 5 requests per second)
        time.sleep(0.2)
        
        if response.status_code == 200:
            data = response.json()
            properties = data['PropertyTable']['Properties'][0]
            
            # Get SMILES and create molecule
            smiles = properties.get('CanonicalSMILES')
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                
                # Cache the result
                pubchem_cache[pubchem_id] = {
                    'smiles': smiles,
                    'inchikey': properties.get('InChIKey')
                }
                
                return mol, pubchem_cache[pubchem_id]
        
        # Cache negative result
        pubchem_cache[pubchem_id] = {'smiles': None, 'inchikey': None}
        save_cache()
        return None, None
    
    except Exception as e:
        print(f"Error retrieving molecule from PubChem ID {pubchem_id}: {e}")
        # Cache negative result
        pubchem_cache[pubchem_id] = {'smiles': None, 'inchikey': None}
        save_cache()
        return None, None


def get_molecule_from_chembl_id(chembl_id: str) -> Tuple[Optional[Chem.Mol], Optional[Dict[str, str]]]:
    """
    Retrieve a molecule from ChEMBL using its ID.
    
    Args:
        chembl_id: ChEMBL ID (e.g., CHEMBL1234)
        
    Returns:
        Tuple containing:
        - RDKit molecule object or None if retrieval failed
        - Dictionary with SMILES and InChIKey or None if retrieval failed
    """
    # Check if the ID looks like a ChEMBL ID
    if not isinstance(chembl_id, str) or not chembl_id.startswith('CHEMBL'):
        return None, None
    
    # Check cache first
    if chembl_id in chembl_cache:
        smiles = chembl_cache[chembl_id].get('smiles')
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            return mol, chembl_cache[chembl_id]
        else:
            # Return cached negative result
            return None, None
    
    try:
        # Use ChEMBL API to get molecule information
        # e.g. https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL1980995.json
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        response = requests.get(url)
        
        # Respect rate limits
        time.sleep(0.2)
        
        if response.status_code == 200:
            data = response.json()
            
            # Get SMILES and create molecule
            smiles = data.get('molecule_structures', {}).get('canonical_smiles')
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                
                # Cache the result
                chembl_cache[chembl_id] = {
                    'smiles': smiles,
                    'inchikey': data.get('molecule_structures', {}).get('standard_inchi_key')
                }
                
                return mol, chembl_cache[chembl_id]
        
        # Cache negative result
        chembl_cache[chembl_id] = {'smiles': None, 'inchikey': None}
        save_cache()
        return None, None
    
    except Exception as e:
        print(f"Error retrieving molecule from ChEMBL ID {chembl_id}: {e}")
        # Cache negative result
        chembl_cache[chembl_id] = {'smiles': None, 'inchikey': None}
        save_cache()
        return None, None


def get_molecule_from_id(potential_id: str, verbose: bool = False) -> Tuple[Optional[Chem.Mol], Optional[Dict[str, str]]]:
    """
    Try to retrieve a molecule using a potential ID, which could be from various sources.
    
    Args:
        potential_id: Potential ID (PubChem, ChEMBL, etc.)
        verbose: Whether to print debug information
        
    Returns:
        Tuple containing:
        - RDKit molecule object or None if retrieval failed
        - Dictionary with SMILES and InChIKey or None if retrieval failed
    """
    # Try as ChEMBL ID first if it looks like one
    if isinstance(potential_id, str) and potential_id.startswith('CHEMBL'):
        mol, info = get_molecule_from_chembl_id(potential_id)
        if mol:
            return mol, info
    
    # Try as PubChem ID
    try:
        # Convert to string if it's a number
        if isinstance(potential_id, (int, float)):
            potential_id = str(int(potential_id))
        
        mol, info = get_molecule_from_pubchem_id(potential_id)
        if mol:
            return mol, info
    except:
        pass
    
    # If we get here, we couldn't retrieve the molecule
    return None, None


def annotate_drug(smiles: str, potential_ids: Set[str], verbose: bool = False) -> DrugAnnotation:
    """
    Annotate a drug using its SMILES string and potential IDs.
    
    Args:
        smiles: SMILES string of the drug
        potential_ids: Set of potential IDs for the drug
        verbose: Whether to print progress information
        
    Returns:
        DrugAnnotation object (smiles, inchikey, valid)
    """
    # Canonicalize the input SMILES
    canonical_smiles = canonicalize_smiles(smiles)
    if not canonical_smiles:
        if verbose:
            print(f"Invalid SMILES: {smiles}")
        return DrugAnnotation(smiles=smiles, inchikey=None, valid=False)
    
    # Try to get molecule from potential IDs
    for potential_id in potential_ids:  
        mol, info = get_molecule_from_id(potential_id, verbose=verbose)
        
        if mol:
            # Validate by comparing canonical SMILES
            retrieved_canonical = Chem.MolToSmiles(mol)
            original_mol = Chem.MolFromSmiles(canonical_smiles)
            
            if original_mol:
                original_canonical = Chem.MolToSmiles(original_mol)
                
                # If the canonical SMILES match, we have a valid annotation
                if retrieved_canonical == original_canonical:
                    return DrugAnnotation(smiles=canonical_smiles, inchikey=info.get('inchikey'), valid=True)
    
    # If we get here, we couldn't find a matching molecule from the IDs
    # Use the SMILES directly
    if verbose:
        print(f"  No matching molecule found from IDs, using SMILES directly: \n  - SMILES: {canonical_smiles} \n  - Potential IDs: {potential_ids}")
    
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol:
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.InchiToInchiKey(inchi) if inchi else None
        
        return DrugAnnotation(smiles=canonical_smiles, inchikey=inchikey, valid=True)
    else:
        return DrugAnnotation(smiles=canonical_smiles, inchikey=None, valid=False)

