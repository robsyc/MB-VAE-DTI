"""
This module provides functionality for adding embeddings to existing h5torch files.
It allows for batch processing of features to avoid memory issues.
"""

from typing import Callable, List, Optional, Union, Literal, Dict, Any, Tuple
from pathlib import Path
import h5torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
