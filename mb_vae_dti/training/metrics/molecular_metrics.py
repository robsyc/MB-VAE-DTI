"""
Molecular generation evaluation metrics for DTITree models.

This module contains metrics for evaluating molecular generation quality:
- ValidityMetric: Checks if generated molecules are chemically valid
- TopKAccuracy: Measures exact match accuracy in top-k generated molecules
- TopKSimilarity: Measures similarity (Tanimoto/Cosine) in top-k molecules
- MolecularMetricsCollection: Convenient collection of all molecular metrics

These metrics are designed to work with RDKit Mol objects and follow
TorchMetrics patterns for consistency with the rest of the codebase.
"""

import torch
from torch import Tensor
from torchmetrics import Metric
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter
import logging

# RDKit imports with fallback
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Molecular metrics will not work.")

logger = logging.getLogger(__name__)


def is_valid_mol(mol) -> bool:
    """
    Check if a molecule is chemically valid.
    
    Args:
        mol: RDKit Mol object or None
        
    Returns:
        True if molecule is valid, False otherwise
    """
    if not RDKIT_AVAILABLE:
        return False
    
    if mol is None:
        return False
    
    try:
        # Check if molecule has atoms
        if mol.GetNumAtoms() == 0:
            return False
        
        # Try to sanitize the molecule
        Chem.SanitizeMol(mol)
        
        # Check if SMILES can be generated
        smiles = Chem.MolToSmiles(mol)
        if not smiles:
            return False
        
        # Try to recreate molecule from SMILES
        recreated = Chem.MolFromSmiles(smiles)
        if recreated is None:
            return False
        
        return True
    except Exception:
        return False


def mol_to_inchi(mol) -> Optional[str]:
    """
    Convert molecule to InChI string.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        InChI string or None if conversion fails
    """
    if not RDKIT_AVAILABLE or mol is None:
        return None
    
    try:
        return Chem.MolToInchi(mol)
    except Exception:
        return None


def canonical_mol_from_inchi(inchi: str):
    """
    Create canonical molecule from InChI string.
    
    Args:
        inchi: InChI string
        
    Returns:
        RDKit Mol object or None if conversion fails
    """
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


class ValidityMetric(Metric):
    """
    Metric for measuring the validity of generated molecules.
    
    This metric tracks the percentage of generated molecules that are
    chemically valid according to RDKit's validation rules.
    """
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('valid_count', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('total_count', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, generated_mols: List) -> None:
        """
        Update the metric with a batch of generated molecules.
        
        Args:
            generated_mols: List of RDKit Mol objects or None
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, validity metric will return 0")
            self.total_count += len(generated_mols)
            return
        
        for mol in generated_mols:
            if is_valid_mol(mol):
                self.valid_count += 1
            self.total_count += 1

    def compute(self) -> Tensor:
        """Compute the validity percentage."""
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.valid_count.device)
        return self.valid_count.float() / self.total_count.float()


class TopKAccuracy(Metric):
    """
    Top-K accuracy metric for molecular generation.
    
    This metric measures whether the true molecule appears in the top-k
    generated molecules, using InChI strings for exact matching.
    """
    
    def __init__(self, k: int = 1, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state('correct_count', default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state('total_count', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, generated_mols: List, true_mol) -> None:
        """
        Update the metric with generated molecules and true molecule.
        
        Args:
            generated_mols: List of RDKit Mol objects
            true_mol: RDKit Mol object (ground truth)
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, accuracy metric will return 0")
            self.total_count += 1
            return
        
        # Convert true molecule to InChI
        true_inchi = mol_to_inchi(true_mol)
        if true_inchi is None:
            self.total_count += 1
            return
        
        # Filter valid molecules and convert to InChI
        valid_inchis = []
        for mol in generated_mols:
            if is_valid_mol(mol):
                inchi = mol_to_inchi(mol)
                if inchi is not None:
                    valid_inchis.append(inchi)
        
        # Count unique InChIs by frequency
        inchi_counter = Counter(valid_inchis)
        unique_inchis = [inchi for inchi, _ in inchi_counter.most_common()]
        
        # Check if true InChI is in top-k
        if true_inchi in unique_inchis[:self.k]:
            self.correct_count += 1
        
        self.total_count += 1

    def compute(self) -> Tensor:
        """Compute the top-k accuracy."""
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.correct_count.device)
        return self.correct_count.float() / self.total_count.float()


class TopKSimilarity(Metric):
    """
    Top-K similarity metric for molecular generation.
    
    This metric measures the maximum similarity (Tanimoto or Cosine) between
    the true molecule and the top-k generated molecules.
    """
    
    def __init__(self, k: int = 1, similarity_type: str = "tanimoto", 
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.similarity_type = similarity_type
        self.add_state('similarity_sum', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total_count', default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_similarity(self, mol1, mol2) -> float:
        """Compute similarity between two molecules."""
        if not RDKIT_AVAILABLE or mol1 is None or mol2 is None:
            return 0.0
        
        try:
            # Generate fingerprints
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            
            # Compute similarity
            if self.similarity_type == "tanimoto":
                return DataStructs.TanimotoSimilarity(fp1, fp2)
            elif self.similarity_type == "cosine":
                return DataStructs.CosineSimilarity(fp1, fp2)
            else:
                raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        except Exception:
            return 0.0

    def update(self, generated_mols: List, true_mol) -> None:
        """
        Update the metric with generated molecules and true molecule.
        
        Args:
            generated_mols: List of RDKit Mol objects
            true_mol: RDKit Mol object (ground truth)
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, similarity metric will return 0")
            self.total_count += 1
            return
        
        # Filter valid molecules and get unique ones
        valid_mols = []
        seen_inchis = set()
        
        for mol in generated_mols:
            if is_valid_mol(mol):
                inchi = mol_to_inchi(mol)
                if inchi is not None and inchi not in seen_inchis:
                    valid_mols.append(mol)
                    seen_inchis.add(inchi)
        
        # Compute maximum similarity with top-k molecules
        max_similarity = 0.0
        for mol in valid_mols[:self.k]:
            similarity = self._compute_similarity(mol, true_mol)
            max_similarity = max(max_similarity, similarity)
        
        self.similarity_sum += max_similarity
        self.total_count += 1

    def compute(self) -> Tensor:
        """Compute the average maximum similarity."""
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.similarity_sum.device)
        return self.similarity_sum / self.total_count.float()


class MolecularMetricsCollection:
    """
    Collection of molecular generation metrics.
    
    This class provides a convenient way to initialize and manage
    all the molecular metrics needed for evaluating generation quality.
    """
    
    def __init__(self, k_values: List[int] = [1, 2, 5]):
        """
        Initialize the metrics collection.
        
        Args:
            k_values: List of k values for top-k metrics
        """
        self.k_values = k_values
        
        # Validity metric
        self.validity = ValidityMetric()
        
        # Top-k accuracy metrics
        self.accuracy_metrics = {
            f"acc_at_{k}": TopKAccuracy(k=k) for k in k_values
        }
        
        # Top-k similarity metrics
        self.similarity_metrics = {
            f"tanimoto_at_{k}": TopKSimilarity(k=k, similarity_type="tanimoto") 
            for k in k_values
        }
        self.similarity_metrics.update({
            f"cosine_at_{k}": TopKSimilarity(k=k, similarity_type="cosine") 
            for k in k_values
        })
        
        # Combine all metrics
        self.all_metrics = {
            'validity': self.validity,
            **self.accuracy_metrics,
            **self.similarity_metrics
        }

    def update(self, generated_mols: List, true_mol) -> None:
        """
        Update all metrics with generated molecules and true molecule.
        
        Args:
            generated_mols: List of RDKit Mol objects
            true_mol: RDKit Mol object (ground truth)
        """
        # Update validity metric
        self.validity.update(generated_mols)
        
        # Update top-k metrics
        for metric in self.accuracy_metrics.values():
            metric.update(generated_mols, true_mol)
        
        for metric in self.similarity_metrics.values():
            metric.update(generated_mols, true_mol)

    def compute_all(self) -> Dict[str, Tensor]:
        """Compute all metrics and return as dictionary."""
        return {name: metric.compute() for name, metric in self.all_metrics.items()}

    def reset_all(self) -> None:
        """Reset all metrics."""
        for metric in self.all_metrics.values():
            metric.reset()

    def log_summary(self) -> str:
        """Generate a summary string of all metrics."""
        results = self.compute_all()
        
        summary = "Molecular Generation Metrics:\n"
        summary += f"  Validity: {results['validity']:.3f}\n"
        
        for k in self.k_values:
            summary += f"  Top-{k} Accuracy: {results[f'acc_at_{k}']:.3f}\n"
            summary += f"  Top-{k} Tanimoto: {results[f'tanimoto_at_{k}']:.3f}\n"
            summary += f"  Top-{k} Cosine: {results[f'cosine_at_{k}']:.3f}\n"
        
        return summary


class MolecularPropertyMetrics:
    """
    Metrics for evaluating molecular properties.
    
    This class provides utilities for computing molecular properties
    and comparing them between generated and true molecules.
    """
    
    def __init__(self):
        self.property_functions = {
            'mol_weight': self._mol_weight,
            'logp': self._logp,
            'tpsa': self._tpsa,
            'num_atoms': self._num_atoms,
            'num_bonds': self._num_bonds,
            'num_rings': self._num_rings,
        }

    def _mol_weight(self, mol) -> float:
        """Calculate molecular weight."""
        if not RDKIT_AVAILABLE or mol is None:
            return 0.0
        try:
            return Chem.Descriptors.MolWt(mol)
        except Exception:
            return 0.0

    def _logp(self, mol) -> float:
        """Calculate LogP."""
        if not RDKIT_AVAILABLE or mol is None:
            return 0.0
        try:
            return Chem.Descriptors.MolLogP(mol)
        except Exception:
            return 0.0

    def _tpsa(self, mol) -> float:
        """Calculate topological polar surface area."""
        if not RDKIT_AVAILABLE or mol is None:
            return 0.0
        try:
            return Chem.Descriptors.TPSA(mol)
        except Exception:
            return 0.0

    def _num_atoms(self, mol) -> int:
        """Count number of atoms."""
        if not RDKIT_AVAILABLE or mol is None:
            return 0
        try:
            return mol.GetNumAtoms()
        except Exception:
            return 0

    def _num_bonds(self, mol) -> int:
        """Count number of bonds."""
        if not RDKIT_AVAILABLE or mol is None:
            return 0
        try:
            return mol.GetNumBonds()
        except Exception:
            return 0

    def _num_rings(self, mol) -> int:
        """Count number of rings."""
        if not RDKIT_AVAILABLE or mol is None:
            return 0
        try:
            return mol.GetRingInfo().NumRings()
        except Exception:
            return 0

    def compute_properties(self, mol) -> Dict[str, float]:
        """Compute all molecular properties."""
        return {name: func(mol) for name, func in self.property_functions.items()}

    def compare_properties(self, generated_mol, true_mol) -> Dict[str, float]:
        """Compare properties between generated and true molecules."""
        gen_props = self.compute_properties(generated_mol)
        true_props = self.compute_properties(true_mol)
        
        differences = {}
        for prop_name in gen_props:
            diff = abs(gen_props[prop_name] - true_props[prop_name])
            differences[f"{prop_name}_diff"] = diff
        
        return differences
