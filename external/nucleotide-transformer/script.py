"""
Script to generate embeddings for DNA sequences using Nucleotide Transformer.
See: https://github.com/instadeepai/nucleotide-transformer
"""

import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple, Any, Dict

# Add the parent directory to the Python path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import parse_args, add_embeddings_to_hdf5

# Constants
MODEL_NAME = "500M_multi_species_v2"
EMBEDDING_NAME = "EMB-NT"
BATCH_SIZE = 16       # Adjust based on GPU memory
EMBEDDING_LAYER = 29  # Using layer 29 (last layer for this model config? check docs)
PADDING_LENGTH = 650  # Max length = 640 6-mers for 1280 residue amino acid sequences + margin

LIMIT_SEQ = False # limit DNA sequences for testing purposes
if LIMIT_SEQ:
    PADDING_LENGTH = 16

def load_model() -> Tuple[hk.Params, hk.Transformed, Any, Dict]:
    """
    Load the pretrained Nucleotide Transformer model.
    
    Returns:
        Tuple of (parameters, forward_fn, tokenizer, config)
    """
    print(f"Loading Nucleotide Transformer model ({MODEL_NAME}), padding to {PADDING_LENGTH}...")
    try:
        parameters, forward_fn, tokenizer, config = get_pretrained_model(
            model_name=MODEL_NAME,
            embeddings_layers_to_save=(EMBEDDING_LAYER,),
            max_positions=PADDING_LENGTH,
        )
        forward_fn = hk.transform(forward_fn)
        print("Model loaded successfully")
        # Explicitly check embedding layer against config if possible
        # config might contain layer count, but structure varies.
        # Assuming layer 29 is valid based on prior knowledge/docs for 500M.
        print(f"Configured to extract embeddings from layer: {EMBEDDING_LAYER}")
        return parameters, forward_fn, tokenizer, config
    except Exception as e:
        print(f"Error loading Nucleotide Transformer model: {e}")
        print("Ensure the model name is correct and dependencies are installed.")
        raise

def embed_batch(sequences: List[str], parameters, forward_fn, tokenizer, random_key) -> List[np.ndarray]:
    """
    Embed a batch of sequences using the Nucleotide Transformer model.
    Performs mean pooling over non-padding tokens.
    
    Args:
        sequences: List of DNA sequences to embed
        parameters: Model parameters
        forward_fn: Forward function
        tokenizer: Tokenizer
        random_key: Random key for JAX
        
    Returns:
        List[numpy.ndarray]: List of embedding vectors for the batch.
    """
    if not sequences:
        return []
    
    if LIMIT_SEQ:
        sequences = [s[:10] for s in sequences]
        
    # Tokenize sequences
    # Assuming tokenizer.batch_tokenize returns list of tuples (sequence, tokens_ids)
    tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
    
    # Run model inference
    # Add exception handling around model apply
    try:
        outs = forward_fn.apply(parameters, random_key, tokens)
    except Exception as e:
        print(f"Error during model inference (forward_fn.apply): {e}")
        # Depending on error, might relate to sequence length, OOM, etc.
        print(f"Batch size: {len(sequences)}, Max sequence length in batch: {max(len(s) for s in sequences)}")
        raise
    
    # Get embeddings and process according to instructions
    # Ensure the key matches how embeddings are stored in `outs`
    embedding_key = f"embeddings_{EMBEDDING_LAYER}"
    if embedding_key not in outs:
        raise KeyError(f"Embedding layer {EMBEDDING_LAYER} not found in model output keys: {outs.keys()}")
        
    embeddings = outs[embedding_key]
    # Shape: (batch_size, seq_len, embedding_dim)
    
    # Remove CLS token representation if present (often at index 0)
    # Check model specifics - NT usually has CLS
    # Assuming CLS is at index 0 based on common practice
    embeddings = embeddings[:, 1:, :] # Remove CLS token at pos 0
    tokens_for_mask = tokens[:, 1:]   # Adjust mask accordingly
    
    # Create padding mask (make sure pad_token_id is correct)
    padding_mask = jnp.expand_dims(tokens_for_mask != tokenizer.pad_token_id, axis=-1)
    
    # Mask embeddings & calculate mean over non-padding tokens
    masked_embeddings = embeddings * padding_mask
    sequences_lengths = jnp.sum(padding_mask, axis=1)
    # Add epsilon for stability
    mean_embeddings = jnp.sum(masked_embeddings, axis=1) / (sequences_lengths + 1e-8)
    
    # Convert to list of numpy arrays
    embeddings_np = [np.array(emb) for emb in mean_embeddings]
    return embeddings_np

def main():
    """Main function to handle file I/O and call the embedding functions"""
    args = parse_args() # Gets only --input
    
    # Load model
    parameters, forward_fn, tokenizer, config = load_model()
    
    # Initialize random key
    random_key = jax.random.PRNGKey(0)
    
    print(f"Processing DNA sequences with Nucleotide Transformer ({MODEL_NAME})")
    
    # Define the processing function closure to include model components
    def nt_batch_processor(sequences: List[str]) -> List[np.ndarray]:
        # We need a new key for each batch if we want true randomness,
        # but for deterministic embeddings, reusing or splitting is fine.
        # Let's split for potential minor variation if desired, though PRNGKey(0) is deterministic.
        # key, subkey = jax.random.split(random_key) # Example if splitting needed
        return embed_batch(sequences, parameters, forward_fn, tokenizer, random_key)

    # Prepare model metadata
    model_metadata = {
        'model_name': MODEL_NAME,
        'embedding_layer': EMBEDDING_LAYER,
        'padding_length': PADDING_LENGTH
    }

    # Process the HDF5 file using the utility function
    add_embeddings_to_hdf5(
        h5_file_path=args.input,
        embedding_name=EMBEDDING_NAME,
        batch_processing_function=nt_batch_processor, # Pass the closure
        batch_size=BATCH_SIZE,
        model_metadata=model_metadata
    )

if __name__ == "__main__":
    main()