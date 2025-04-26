"""
Script to embed protein sequences using ESM-C 600M (open model with 36 layers i.e. `esmc-600m-2024-12`)
See: https://github.com/evolutionaryscale/esm
Resources:
    - ESM-C blog post: https://www.evolutionaryscale.ai/blog/esm-cambrian
    - Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11601519/ on how medium-sized models with mean embeddings are good (+ implementation details for locally run ESM-C)
    - Paper: https://www.biorxiv.org/content/10.1101/2024.02.05.578959v2 on how last-hidden-state embeddings are generally quite good
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from typing import List

# Add the parent directory to the Python path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import parse_args, add_embeddings_to_hdf5

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# Define constants for this script
BATCH_SIZE = 16 # Smaller batch size suitable for local GPU processing
MODEL_NAME = "esmc_600m"
EMBEDDING_NAME = "EMB-ESM" # Simplified name
# EMBEDDING_LAYER = 36 # Not needed when using output.embeddings
LIMIT_SEQ = True # limit amino acid sequences for testing purposes

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load Model
print(f"Loading ESM model ({MODEL_NAME})...")
client = ESMC.from_pretrained(MODEL_NAME).to(DEVICE)
client.eval()  # Set model to evaluation mode
print(f"ESM model ({MODEL_NAME}) loaded.")
# print(f"Assuming {MODEL_NAME} has 36 layers and extracting embeddings from index: {EMBEDDING_LAYER}")

# Define Logits Configuration - Removed as not used with direct model call
# EMBEDDING_CONFIG = LogitsConfig(sequence=True, return_hidden_states=True)

def embed_batch(sequences: List[str]) -> List[np.ndarray]:
    """
    Embed a batch of protein sequences using the loaded local ESM-C model.
    Extracts the final output embeddings and performs mean pooling over non-padding tokens.

    Args:
        sequences: List of protein sequences as strings

    Returns:
        List[numpy.ndarray]: List of mean embedding vectors for the batch.
    """
    if not sequences:
        return []
    
    if LIMIT_SEQ:
        sequences = [s[:10] for s in sequences]
        
    tokenizer = client.tokenizer
    
    with torch.no_grad():
        try:
            protein_tensors = tokenizer(
                sequences,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
        except Exception as e:
            print(f"Error during batch tokenization: {e}")
            raise RuntimeError("Failed to tokenize batch.") from e

        protein_tensors = {k: v.to(DEVICE) for k, v in protein_tensors.items()}
        
        # Direct model call (forward pass) using only input_ids
        output = client(protein_tensors['input_ids'])
        # Output object likely contains embeddings, logits, hidden_states etc.
        
        # Extract the final output embeddings
        # Shape: (batch_size, seq_len, hidden_dim)
        embeddings = output.embeddings 
        
        # Get attention mask from the tokenized batch output
        attention_mask = protein_tensors['attention_mask']
        
        # Calculate sequence lengths (sum of non-padding tokens in the mask)
        lengths = attention_mask.sum(dim=1)
        
        # Mask padding tokens in the embeddings before pooling
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        # Calculate mean embedding over non-padding tokens
        mean_embeddings = masked_embeddings.sum(dim=1) / (lengths.unsqueeze(-1).to(masked_embeddings.device) + 1e-8)
        
        # Convert to list of numpy arrays (move to CPU first)
        embeddings_np = [emb.cpu().numpy() for emb in mean_embeddings]
        
    return embeddings_np

def main():
    """Main function to parse args and call the HDF5 processing utility."""
    args = parse_args() # Gets only --input
    
    # Prepare model metadata
    model_metadata = {
        'model_name': MODEL_NAME,
    }

    add_embeddings_to_hdf5(
        h5_file_path=args.input,
        embedding_name=EMBEDDING_NAME,
        batch_processing_function=embed_batch, 
        batch_size=BATCH_SIZE,
        model_metadata=model_metadata 
    )

if __name__ == "__main__":
    main()