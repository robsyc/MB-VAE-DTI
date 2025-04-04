import pandas as pd
import numpy as np
import os
from pathlib import Path
import tqdm
from typing import Callable, List, Dict, Optional, Union, Tuple, Any

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from subword_nmt.apply_bpe import BPE
import codecs
import h5torch

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = Path("external")
ESPF_DIR = EXTERNAL_DIR / "ESPF"

# Assertions
assert ESPF_DIR.exists(), """ESPF directory does not exist. Cannot create protein fingerprints.
Please download the files `codes_protein.txt` and `subword_units_map_protein.csv` 
from https://github.com/kexinhuang12345/ESPF
and place them in the `external/ESPF` directory."""

## FINGERPRINTS - helper functions

def get_drug_fingerprint(s: str) -> np.ndarray:
    # See: https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    molecule = Chem.MolFromSmiles(s)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array(mfpgen.GetFingerprint(molecule))

vocab_path = ESPF_DIR / 'codes_protein.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv(ESPF_DIR / 'subword_units_map_protein.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

def get_target_fingerprint(s: str) -> np.ndarray:
    # See: https://github.com/kexinhuang12345/ESPF
    t = pbpe.process_line(s).split()
    i = [words2idx_p[i] for i in t]
    v = np.zeros(len(idx2word_p), )
    v[i] = 1
    return v


## FOUNDATION MODELS - helper functions
# ...


## ADDING FEATURES TO H5TORCH FILES

class FeatureGenerator:
    """
    A class for generating and adding feature vectors to h5torch files.
    
    This class provides functionality to:
    1. Open and work with h5torch files
    2. Iterate over drugs/targets (in batches)
    3. Apply feature generation functions
    4. Save the generated feature vectors back to the h5torch file
    
    It uses batch processing when possible and saves intermediate results to 
    avoid memory constraints.
    """
    
    def __init__(
        self, 
        h5torch_path: Union[str, Path] = PROCESSED_DIR / "data.h5torch",
        mode: str = 'a'
    ):
        """
        Initialize the FeatureGenerator with a path to an h5torch file.
        
        Args:
            h5torch_path: Path to the h5torch file
            mode: File access mode ('r' for read-only, 'a' for read/write)
        """
        self.h5torch_path = Path(h5torch_path)
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """Context manager entry point."""
        self.file = h5torch.File(str(self.h5torch_path), self.mode)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        if self.file is not None:
            self.file.close()
            self.file = None
    
    def get_axis_items(self, axis: int, column: str) -> np.ndarray:
        """
        Get all items from a specific axis and column.
        
        Args:
            axis: The axis (0 for drugs, 1 for targets)
            column: The column name
            
        Returns:
            Array of items
        """
        key = f"{axis}/{column}"
        if key not in self.file:
            raise KeyError(f"Column '{column}' not found in axis {axis}")
        
        return self.file[key][:]

    def process_items_in_batches(
        self,
        axis: int,
        source_column: str,
        feature_function: Callable,
        output_column: str,
        batch_size: int = 32,
        dtype: np.dtype = np.float32,
        decode_strings: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> None:
        """
        Process items from an axis in batches, applying a feature function.
        
        Args:
            axis: The axis (0 for drugs, 1 for targets)
            source_column: The column containing the source data
            feature_function: Function to apply to each item (or batch of items)
            output_column: Name for the output feature column
            batch_size: Size of batches for processing
            dtype: Data type for the feature vectors
            decode_strings: Whether to decode byte strings to UTF-8
            verbose: Whether to show progress bar
            **kwargs: Additional arguments for the feature function
        """
        # Get items from the h5torch file
        items = self.get_axis_items(axis, source_column)
        n_items = len(items)
        
        # Check if the feature function supports batch processing
        supports_batch = hasattr(feature_function, 'batch_process')
        
        # Get the shape of the output by processing a single item
        if decode_strings:
            sample_item = items[0].decode('utf-8')
        else:
            sample_item = items[0]
            
        sample_output = feature_function(sample_item, **kwargs)
        output_shape = sample_output.shape
        
        # Prepare a temporary output file for storing the results
        temp_filename = self.h5torch_path.with_suffix('.temp.npy')
        total_shape = (n_items,) + output_shape
        
        # Create a memory-mapped array for the results
        result_array = np.memmap(
            temp_filename, 
            dtype=dtype, 
            mode='w+', 
            shape=total_shape
        )
        
        # Process items in batches
        iterator = range(0, n_items, batch_size)
        if verbose:
            iterator = tqdm.tqdm(iterator, desc=f"Processing {source_column} -> {output_column}")
        
        for i in iterator:
            batch_end = min(i + batch_size, n_items)
            batch_items = items[i:batch_end]
            
            # Decode strings if necessary
            if decode_strings:
                batch_items = [item.decode('utf-8') for item in batch_items]
            
            if supports_batch:
                # Process the entire batch at once
                batch_results = feature_function.batch_process(batch_items, **kwargs)
                result_array[i:batch_end] = batch_results
            else:
                # Process each item individually
                for j, item in enumerate(batch_items):
                    result_array[i + j] = feature_function(item, **kwargs)
        
        # Flush changes to disk
        result_array.flush()
        
        # Register the results in the h5torch file
        if f"{axis}/{output_column}" in self.file:
            del self.file[f"{axis}/{output_column}"]
            
        self.file.register(
            result_array, 
            mode="N-D", 
            axis=axis, 
            name=output_column, 
            dtype_save=dtype, 
            dtype_load=dtype
        )
        
        # Clean up the temporary file
        del result_array
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    def add_drug_fingerprints(
        self, 
        source_column: str = "Drug_SMILES", 
        output_column: str = "Drug_fp", 
        batch_size: int = 64
    ) -> None:
        """
        Add Morgan fingerprints for drugs.
        
        Args:
            source_column: Column containing SMILES strings
            output_column: Name for the output fingerprint column
            batch_size: Size of batches for processing
        """
        self.process_items_in_batches(
            axis=0, 
            source_column=source_column, 
            feature_function=get_drug_fingerprint,
            output_column=output_column,
            batch_size=batch_size,
            dtype=np.float32,
            decode_strings=True,
            verbose=True
        )
    
    def add_target_fingerprints(
        self, 
        source_column: str = "Target_AA", 
        output_column: str = "Target_fp", 
        batch_size: int = 64
    ) -> None:
        """
        Add ESPF fingerprints for targets.
        
        Args:
            source_column: Column containing amino acid sequences
            output_column: Name for the output fingerprint column
            batch_size: Size of batches for processing
        """
        self.process_items_in_batches(
            axis=1, 
            source_column=source_column, 
            feature_function=get_target_fingerprint,
            output_column=output_column,
            batch_size=batch_size,
            dtype=np.float32,
            decode_strings=True,
            verbose=True
        )
    
    def add_all_fingerprints(
        self,
        drug_source: str = "Drug_SMILES",
        drug_output: str = "Drug_fp",
        target_source: str = "Target_AA",
        target_output: str = "Target_fp",
        batch_size: int = 64
    ) -> None:
        """
        Add both drug and target fingerprints.
        
        Args:
            drug_source: Column containing drug SMILES strings
            drug_output: Name for the drug fingerprint column
            target_source: Column containing target amino acid sequences
            target_output: Name for the target fingerprint column
            batch_size: Size of batches for processing
        """
        print(f"Adding drug fingerprints from {drug_source} to {drug_output}...")
        self.add_drug_fingerprints(
            source_column=drug_source,
            output_column=drug_output,
            batch_size=batch_size
        )
        
        print(f"Adding target fingerprints from {target_source} to {target_output}...")
        self.add_target_fingerprints(
            source_column=target_source,
            output_column=target_output,
            batch_size=batch_size
        )
        
        print("Finished adding fingerprints.")

# Usage example
if __name__ == "__main__":
    # Example of how to use the FeatureGenerator class
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Generate and add feature vectors to h5torch files")
    parser.add_argument(
        "--input", 
        type=str, 
        default=str(PROCESSED_DIR / "data.h5torch"),
        help="Path to the h5torch file"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--drug-source", 
        type=str, 
        default="Drug_SMILES",
        help="Column containing drug SMILES strings"
    )
    parser.add_argument(
        "--drug-output", 
        type=str, 
        default="Drug_fp",
        help="Name for the drug fingerprint column"
    )
    parser.add_argument(
        "--target-source", 
        type=str, 
        default="Target_AA",
        help="Column containing target amino acid sequences"
    )
    parser.add_argument(
        "--target-output", 
        type=str, 
        default="Target_fp",
        help="Name for the target fingerprint column"
    )
    
    args = parser.parse_args()
    
    print(f"Processing h5torch file: {args.input}")
    
    # Use context manager to ensure proper file closure
    with FeatureGenerator(args.input) as generator:
        generator.add_all_fingerprints(
            drug_source=args.drug_source,
            drug_output=args.drug_output,
            target_source=args.target_source,
            target_output=args.target_output,
            batch_size=args.batch_size
        )
