from collections import Counter
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

ATOMS = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
ATOM_TO_WEIGHT = {
    'C': 12,
    'O': 16,
    'P': 31,
    'N': 14,
    'S': 32,
    'Cl': 35,
    'F': 19,
    'H': 1,
}
ATOM_TO_VALENCE = {
    'C': 4,
    'O': 2,
    'P': 3,
    'N': 3,
    'S': 2,
    'Cl': 1,
    'F': 1,
    'H': 1,
}

