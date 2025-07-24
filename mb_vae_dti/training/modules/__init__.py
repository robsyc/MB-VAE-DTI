"""
Training modules for DTI prediction.
"""

from .baseline import BaselineDTIModel
from .multi_modal import MultiModalDTIModel
from .multi_output import MultiOutputDTIModel
from .multi_hybrid import MultiHybridDTIModel
from .full import FullDTIModel

__all__ = [
    "BaselineDTIModel",
    "MultiModalDTIModel", 
    "MultiOutputDTIModel",
    "MultiHybridDTIModel",
    "FullDTIModel"
]