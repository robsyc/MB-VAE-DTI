#!/usr/bin/env python
"""
Example script to demonstrate using external repository embedding scripts
with h5torch files.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import h5torch
import logging
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('example_embeddings')

# Add parent directory to path to import our module
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent.parent))

from mb_vae_dti.processing.embedding import (
    add_embeddings_to_h5torch,
    generate_embeddings,
    PROCESSED_DIR
)

# Example protein sequences (short ones for demonstration)
EXAMPLE_PROTEINS = [
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS",  # KRAS
    "MAESSDKLYRVEYAKSGRASCKKCSESIPKDSLRMAIMVQSPMFDGKVPHWYHFSCFWKVGHSIRHPDVEVDGFSELRWDDQQKVKKTAEAGGVTGKGQDGIGSKAAEKAGAAKAEDRQNHSSSRGSGGPGGLSHSTSPGPGLNGTSMSPPMLVSSPPSSVSYEYVVTRYGGKKKGKSLPALLPPLGSAKSQ",  # PARP1 (partial)
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"  # Alpha-synuclein
]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate embeddings and add to h5torch files")
    parser.add_argument(
        "--repos", 
        nargs="+", 
        default=["ESM", "ESPF"],
        help="Repositories to use for embeddings"
    )
    parser.add_argument(
        "--h5torch-file", 
        type=str,
        default="protein_embeddings_demo.h5torch",
        help="H5torch file to create/update"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite existing features"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode (keep temporary files)"
    )
    return parser.parse_args()

def create_demo_h5torch_file(file_name: str, sequences: List[str], overwrite: bool = False) -> Path:
    """
    Create a demonstration h5torch file with proteins.
    
    Args:
        file_name: Name of the h5torch file
        sequences: List of protein sequences to include
        overwrite: Whether to overwrite existing file
        
    Returns:
        Path to the created h5torch file
    """
    file_path = PROCESSED_DIR / file_name
    
    if file_path.exists() and not overwrite:
        logger.info(f"File {file_path} already exists. Not overwriting.")
        return file_path
    
    idx = np.arange(len(sequences))
    sequences_array = np.array(sequences)
    
    # Create the h5torch file
    f = h5torch.File(file_path, "w")
    
    # Register the ids as the central object
    f.register(idx, "central")
    
    # Register protein sequences as metadata aligned with axis 0
    f.register(sequences_array, 0, name="protein_sequences", dtype_save="bytes", dtype_load="str")
    
    # Create some dummy labels for the proteins
    protein_labels = np.random.rand(len(sequences))
    f.register(protein_labels, 0, name="y")
    
    f.close()
    logger.info(f"Created demo h5torch file at {file_path}")
    
    return file_path

def main():
    """Main function to run the example."""
    args = parse_args()
    
    logger.info(f"Running with repositories: {args.repos}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Create the demo file
    h5torch_file_path = create_demo_h5torch_file(args.h5torch_file, EXAMPLE_PROTEINS, args.overwrite)
    
    # Generate embeddings for each repository
    for repo_name in args.repos:
        logger.info(f"\n=== Generating {repo_name} embeddings ===")
        
        # Different settings for different repos
        use_batch = False  # Default to False for safety
        feature_name = f"{repo_name.lower()}_embeddings"
        
        try:
            add_embeddings_to_h5torch(
                h5torch_file_name=args.h5torch_file,
                repo_name=repo_name,
                entity_axis=0,  # Proteins are on axis 0
                feature_name=feature_name,
                sequences=EXAMPLE_PROTEINS,
                use_batch=use_batch,
                overwrite=args.overwrite,
                debug=args.debug
            )
            logger.info(f"Successfully added {feature_name}")
        except Exception as e:
            logger.error(f"Error generating {repo_name} embeddings: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 