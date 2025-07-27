import torch
from torchmetrics import Metric, MetricCollection
from torch import Tensor
import wandb
import torch.nn as nn
from typing import List, Optional, Dict, Any
from collections import Counter
import numpy as np

# RDKit imports for molecular validation
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings


# ========================================================================================
# Utility functions for molecular validation (adapted from DiffMS)
# ========================================================================================

def mol2smiles(mol):
    """Convert RDKit mol to SMILES string."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def is_valid(mol):
    """Check if a molecule is valid according to RDKit."""
    if mol is None:
        return False
        
    smiles = mol2smiles(mol)
    if smiles is None:
        return False

    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except:
        return False
    if len(mol_frags) > 1:
        return False
    
    return True


def mol2fp(mol, radius: int = 2, nBits: int = 2048):
    """Safely compute Morgan fingerprint for a molecule with proper validation."""
    if not is_valid(mol):
        return None
    
    try:
        # Ensure molecule is properly sanitized
        Chem.SanitizeMol(mol)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    except:
        return None


# ========================================================================================
# Training Cross-Entropy Metrics (existing functionality)
# ========================================================================================

class CEPerClass(Metric):
    full_state_update = False
    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples

# Define specific atom type metrics
class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

# Define bond type metrics
class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self, atom_types: list[str]):
        class_dict = {'H': HydrogenCE, 'C': CarbonCE, 'N': NitroCE, 'O': OxyCE, 'F': FluorCE, 'B': BoronCE,
                      'Br': BrCE, 'Cl': ClCE, 'I': IodineCE, 'P': PhosphorusCE, 'S': SulfurCE, 'Se': SeCE,
                      'Si': SiCE}

        metrics_list = []
        for i, atom_type in enumerate(atom_types):
            try:
                metrics_list.append(class_dict[atom_type](i))
            except KeyError:
                pass
        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


class TrainMolecularMetricsDiscrete(nn.Module):
    """Training molecular metrics that track cross-entropy losses during training."""
    def __init__(self, atom_types: list[str], prefix: str = "train/"):
        super().__init__()
        self.prefix = prefix
        self.train_atom_metrics = AtomMetricsCE(atom_types=atom_types)
        self.train_bond_metrics = BondMetricsCE()

    def update(self, masked_pred_X, masked_pred_E, true_X, true_E):
        """Update training metrics."""
        self.train_atom_metrics(masked_pred_X, true_X)
        self.train_bond_metrics(masked_pred_E, true_E)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all training metrics."""
        results = {}
        
        # Add atom metrics with prefix
        for key, val in self.train_atom_metrics.compute().items():
            results[f"{self.prefix}atom_{key}"] = val
            
        # Add bond metrics with prefix  
        for key, val in self.train_bond_metrics.compute().items():
            results[f"{self.prefix}bond_{key}"] = val
            
        return results

    def reset(self):
        """Reset all metrics."""
        self.train_atom_metrics.reset()
        self.train_bond_metrics.reset()


# ========================================================================================
# Validation Molecular Quality Metrics
# ========================================================================================

class MolecularValidity(Metric):
    """Tracks validity of generated molecules using RDKit validation."""
    def __init__(self):
        super().__init__()
        self.add_state("valid", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_mols: List[Optional[Chem.Mol]]):
        """Update with a batch of generated molecules.
        
        Args:
            generated_mols: List of RDKit molecule objects (or None for invalid)
        """
        for mol in generated_mols:
            if is_valid(mol):
                self.valid += 1
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.valid.device)
        return self.valid.float() / self.total.float()


class MolecularAccuracy(Metric):
    """
    Tracks accuracy between generated and target molecules.
    
    Handles multiple samples per embedding: if any generated sample matches target, counts as correct.
    """
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total_groups", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(
        self, 
        generated_mols: List[List[Optional[Chem.Mol]]], 
        target_smiles: List[str],
    ):
        """Update with generated and target molecules.
        
        Args:
            generated_mols_groups: List of lists, where each inner list contains generated molecules for one target
            target_smiles: List of target SMILES strings (from batch_data.smiles)
        """
        for target_smiles, gen_mols in zip(target_smiles, generated_mols):
            for mol in gen_mols:
                if mol2smiles(mol) == target_smiles:
                    self.correct += 1
                    break
            self.total_groups += 1

    def compute(self) -> torch.Tensor:
        if self.total_groups == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total_groups.float()


class TanimotoSimilarity(Metric):
    """
    Tracks best Tanimoto similarity between generated and target molecules.
    
    Handles multiple samples per embedding: takes the best similarity among generated samples.
    """
    def __init__(self, radius: int = 2, nBits: int = 2048):
        super().__init__()
        self.radius = radius
        self.nBits = nBits
        self.add_state("total_similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_groups", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(
        self, 
        generated_mols: List[List[Optional[Chem.Mol]]], 
        target_fps: List[torch.Tensor],
    ):
        """Update with generated and target molecules.
        
        Args:
            generated_mols: List of lists, where each inner list contains generated molecules for one target
            target_fps: List of pre-computed target fingerprints (from batch_data.drug_fp)
        """
        for target_fp, gen_mols in zip(target_fps, generated_mols):
            best_similarity = 0.0
            for gen_mol in gen_mols:
                if gen_mol is not None:
                    # Use safe fingerprint computation to avoid RDKit errors
                    gen_fp = mol2fp(gen_mol, self.radius, self.nBits)
                    if gen_fp is not None:
                        sim = Chem.DataStructs.TanimotoSimilarity(gen_fp, target_fp)
                        if sim > best_similarity:
                            best_similarity = sim
            self.total_similarity += best_similarity
            self.total_groups += 1

    def compute(self) -> torch.Tensor:
        if self.total_groups == 0:
            return torch.tensor(0.0, device=self.total_similarity.device)
        return self.total_similarity / self.total_groups


class ValidationMolecularMetrics(nn.Module):
    """
    Collection of molecular quality metrics for validation/testing, following DTI pattern.
    
    Handles multiple samples per embedding appropriately.
    """
    def __init__(self, prefix: str = "", radius: int = 2, nBits: int = 2048):
        """
        Initialize validation molecular metrics.
        
        Args:
            prefix: Prefix for metric names (e.g., "val/" or "test/")
            radius: Morgan fingerprint radius
            nBits: Morgan fingerprint size
        """
        super().__init__()
        self.prefix = prefix
        
        self.validity = MolecularValidity()
        self.accuracy = MolecularAccuracy()
        self.tanimoto_similarity = TanimotoSimilarity(radius=radius, nBits=nBits)

    def update(
        self, 
        generated_mols: List[List[Optional[Chem.Mol]]], 
        target_smiles: List[str],
        target_fps: List[torch.Tensor],
    ):
        """Update all metrics with grouped data structure.
        
        Args:
            generated_mols_groups: List of lists, where each inner list contains generated molecules for one target
            target_smiles: List of target SMILES strings (from batch_data.smiles)
            target_fps: List of pre-computed target fingerprints (from batch_data.drug_fp)
        """
        assert len(generated_mols) == len(target_smiles) == len(target_fps), "Number of generated molecules must match number of targets"

        # Flatten generated molecules for validity metric
        self.validity.update([mol for group in generated_mols for mol in group])
        
        # Use grouped update for accuracy and similarity
        self.accuracy.update(generated_mols, target_smiles)
        self.tanimoto_similarity.update(generated_mols, target_fps)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics."""
        return {
            f"{self.prefix}mol_validity": self.validity.compute(),
            f"{self.prefix}mol_accuracy": self.accuracy.compute(),
            f"{self.prefix}mol_tanimoto_sim": self.tanimoto_similarity.compute(),
        }

    def reset(self):
        """Reset all metrics."""
        self.validity.reset()
        self.accuracy.reset()
        self.tanimoto_similarity.reset()
