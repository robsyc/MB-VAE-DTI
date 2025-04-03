"""
This module provides functionality for loading the h5torch file as a Dataset object.
"""

from typing import List, Dict, Union, Optional, Any, Tuple, Literal
from pathlib import Path
import h5torch
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"


# class DTIDataset(Dataset):
#     """
#     A dataset class for drug-target interaction data stored in h5torch format.